"""Step 4 - Machine-learning triage models.

A. Routing model    : narrative text -> product team (9 classes). Answers "can we auto-route
                      incoming complaints, and how many can we route without a human?"
B. Relief model     : P(complaint ends in monetary or non-monetary relief) using only what is
                      known at intake (text, product, issue, company, state, consumer tags).
                      Compares structured-only vs. text-only vs. combined to measure what the
                      AI text features add.

Evaluation uses a hold-out set of 20% of complaints, split by *narrative text group* so a
copy-pasted complaint can never appear in both training and test data.

Outputs (outputs/tables/): routing_report.csv, routing_confusion.csv, routing_automation.csv,
relief_model_comparison.csv, relief_deciles.csv
and data/processed/model_scores.csv.gz (scores for every complaint, used by the dashboards).
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

import config as C
from tfidf import fit_tfidf

CATEGORICAL = ["product", "sub_product", "issue", "sub_issue", "company_top100", "state"]
FLAGS = ["older_american", "servicemember", "mentions_attorney", "mentions_legal_action",
         "mentions_regulator", "cites_law", "mentions_identity_theft", "mentions_fees"]


def load() -> pd.DataFrame:
    df = pd.read_csv(C.CLEAN)
    df["narrative_clean"] = df["narrative_clean"].fillna("")
    top100 = df["company"].value_counts().head(100).index
    df["company_top100"] = df["company"].where(df["company"].isin(top100), "Other")
    df["log_words"] = np.log1p(df["narrative_words"]) / np.log1p(df["narrative_words"].max())
    return df


def text_features(texts: pd.Series, max_features: int):
    """TF-IDF on 1-2 grams fitted on `texts`; returns (fitted vectorizer, matrix for `texts`)."""
    return fit_tfidf(texts, max_features, ngram_range=(1, 2), min_df=5, max_df=0.5)


# --------------------------------------------------------------------------------------
# A. Routing model
# --------------------------------------------------------------------------------------
def routing_model(df, tr, te):
    vec, X_tr = text_features(df["narrative_clean"].iloc[tr], 100_000)
    X_te = vec.transform(df["narrative_clean"].iloc[te])
    y_tr, y_te = df["product"].iloc[tr], df["product"].iloc[te]

    clf = LogisticRegression(C=10, solver="saga", max_iter=300, random_state=C.RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    pred = clf.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)

    acc = accuracy_score(y_te, pred)
    macro = f1_score(y_te, pred, average="macro")
    print(f"[Routing] hold-out accuracy {acc:.1%} | macro-F1 {macro:.3f} | n={len(y_te):,}")

    report = pd.DataFrame(classification_report(y_te, pred, output_dict=True)).T
    report.round(4).to_csv(C.TABLES / "routing_report.csv")
    labels = clf.classes_
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels, normalize="true"),
                      index=labels, columns=labels)
    cm.round(4).to_csv(C.TABLES / "routing_confusion.csv")

    # Automation curve: route automatically only when the model is confident enough.
    correct = (pred == y_te.values)
    rows = []
    for t in np.round(np.arange(0.30, 0.995, 0.01), 2):
        m = conf >= t
        rows.append({"threshold": t, "coverage": m.mean(),
                     "accuracy_on_auto_routed": correct[m].mean() if m.any() else np.nan})
    auto = pd.DataFrame(rows)
    auto.round(4).to_csv(C.TABLES / "routing_automation.csv", index=False)
    ok = auto[auto["accuracy_on_auto_routed"] >= 0.95]
    if len(ok):
        best = ok.iloc[0]
        print(f"[Routing] at confidence >= {best.threshold:.2f}: auto-route {best.coverage:.1%} "
              f"of complaints with {best.accuracy_on_auto_routed:.1%} accuracy")
    return pd.DataFrame({"route_pred": pred, "route_confidence": conf.round(4)},
                        index=df.index[te])


# --------------------------------------------------------------------------------------
# B. Relief model
# --------------------------------------------------------------------------------------
def build_features(df, tr_idx, te_idx, vec=None, ohe=None):
    """Return (X_struct_tr, X_struct_te, X_text_tr, X_text_te, vec, ohe)."""
    if ohe is None:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
        ohe.fit(df[CATEGORICAL].iloc[tr_idx])
    num = df[FLAGS + ["log_words"]].to_numpy(dtype=float)
    S = hstack([ohe.transform(df[CATEGORICAL]), csr_matrix(num)]).tocsr()
    if vec is None:
        vec, _ = text_features(df["narrative_clean"].iloc[tr_idx], 50_000)
    T = vec.transform(df["narrative_clean"])
    return S[tr_idx], S[te_idx], T[tr_idx], T[te_idx], vec, ohe


def lift_stats(y, score):
    order = np.argsort(-score, kind="stable")
    y_sorted = np.asarray(y)[order]
    n = len(y_sorted)
    top20 = y_sorted[: int(0.2 * n)]
    top10 = y_sorted[: int(0.1 * n)]
    return {
        "capture_top20_pct": top20.sum() / y_sorted.sum(),
        "precision_top10_pct": top10.mean(),
        "lift_top10": top10.mean() / y_sorted.mean(),
    }


def relief_model(df, tr, te):
    S_tr, S_te, T_tr, T_te, _, _ = build_features(df, tr, te)
    y_tr, y_te = df["relief"].iloc[tr].values, df["relief"].iloc[te].values

    variants = {
        "Structured fields only (product, issue, company, state, tags)": (S_tr, S_te),
        "Narrative text only (TF-IDF)": (T_tr, T_te),
        "Combined: structured + narrative text": (hstack([S_tr, T_tr]).tocsr(), hstack([S_te, T_te]).tocsr()),
    }
    rows, models = [], {}
    for name, (A, B) in variants.items():
        clf = LogisticRegression(C=0.5, solver="liblinear", max_iter=2000, random_state=C.RANDOM_STATE)
        clf.fit(A, y_tr)
        # Rounded to 6 decimals: smaller differences are floating-point noise, not signal
        s = clf.predict_proba(B)[:, 1].round(6)
        models[name] = (clf, s)
        rows.append({"model": name, "roc_auc": roc_auc_score(y_te, s),
                     "pr_auc": average_precision_score(y_te, s), "base_rate": y_te.mean(),
                     **lift_stats(y_te, s)})
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(C.TABLES / "relief_model_comparison.csv", index=False)
    print("[Relief]\n" + comp.round(3).to_string(index=False))

    # Decile table for the combined model
    clf, s = models["Combined: structured + narrative text"]
    dec = pd.qcut(pd.Series(s).rank(method="first", ascending=False), 10, labels=range(1, 11))
    deciles = (pd.DataFrame({"decile": dec, "relief": y_te, "score": s})
                 .groupby("decile", observed=True)
                 .agg(complaints=("relief", "size"), relief_rate=("relief", "mean"),
                      avg_score=("score", "mean")))
    deciles["cumulative_capture"] = (deciles["relief_rate"] * deciles["complaints"]).cumsum() / y_te.sum()
    deciles["lift"] = deciles["relief_rate"] / y_te.mean()
    deciles.reset_index().round(4).to_csv(C.TABLES / "relief_deciles.csv", index=False)
    # Phrase-level explanations are produced descriptively (and mix-adjusted) in step 3,
    # outputs/tables/phrase_relief_gaps.csv, which reads more clearly than raw coefficients.


def out_of_fold_relief_scores(df, n_splits=5):
    """Relief score for every complaint, each one predicted by a model that never saw it
    (or any copy of its text). Used for dashboards / Power BI prioritisation views."""
    scores = np.zeros(len(df))
    # shuffle=True assigns text groups to folds with a seeded shuffle. The default assignment
    # orders groups with an unstable sort, so the folds could differ between machines.
    folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    for k, (tr, te) in enumerate(folds.split(df, groups=df["text_group"])):
        S_tr, S_te, T_tr, T_te, _, _ = build_features(df, tr, te)
        clf = LogisticRegression(C=0.5, solver="liblinear", max_iter=2000, random_state=C.RANDOM_STATE)
        clf.fit(hstack([S_tr, T_tr]).tocsr(), df["relief"].iloc[tr].values)
        scores[te] = clf.predict_proba(hstack([S_te, T_te]).tocsr())[:, 1]
        print(f"  fold {k + 1}/{n_splits} done")
    return scores


def main() -> None:
    df = load()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.RANDOM_STATE)
                  .split(df, groups=df["text_group"]))
    print(f"Train {len(tr):,} | hold-out {len(te):,} (grouped by narrative text)")

    routes = routing_model(df, tr, te)
    relief_model(df, tr, te)

    print("Scoring all complaints out-of-fold for dashboards ...")
    oof = out_of_fold_relief_scores(df)
    print(f"[Relief] out-of-fold ROC AUC on all complaints: {roc_auc_score(df['relief'], oof):.3f}")

    out = pd.DataFrame({"complaint_id": df["complaint_id"], "in_holdout": 0,
                        "relief_score": oof.round(4)})
    out.loc[df.index[te], "in_holdout"] = 1
    out = out.join(routes)
    out["relief_priority_decile"] = pd.qcut(out["relief_score"].rank(method="first", ascending=False),
                                            10, labels=range(1, 11)).astype(int)
    out.to_csv(C.PROCESSED / "model_scores.csv.gz", index=False, compression="gzip")
    print("Saved -> data/processed/model_scores.csv.gz")


if __name__ == "__main__":
    main()
