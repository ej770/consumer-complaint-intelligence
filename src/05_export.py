"""Step 5 - Export model-ready outputs for BI tools and the web dashboard.

* powerbi/   star schema (fact + 4 dimensions) ready to load into Power BI, plus starter DAX
* dashboard/dashboard_data.json   compact, filterable data for dashboard/index.html
"""
import base64
import json
import re

import numpy as np
import pandas as pd

import config as C

SIGNAL_COLS = ["mentions_attorney", "mentions_legal_action", "mentions_regulator",
               "cites_law", "mentions_identity_theft", "mentions_fees"]
SIGNAL_LABELS = ["Mentions an attorney or lawyer", "Mentions suing or court",
                 "Mentions a regulator (CFPB, AG, FTC, BBB)", "Cites a consumer law (FCRA, FDCPA)",
                 "Mentions identity theft", "Mentions fees or overdraft"]
GEO_LABELS = ["Rest of U.S.", "Texas, outside Greater Houston", "Greater Houston"]
TONE_LABELS = ["Most negative", "Negative", "Neutral", "Mild", "Least negative"]


def load_all() -> pd.DataFrame:
    df = pd.read_csv(C.CLEAN)
    text = pd.read_csv(C.PROCESSED / "text_features.csv.gz")
    scores = pd.read_csv(C.PROCESSED / "model_scores.csv.gz")
    df = df.merge(text, on="complaint_id", how="left").merge(scores, on="complaint_id", how="left")
    df["tone_quintile"] = pd.qcut(df["sentiment"].rank(method="first"), 5, labels=False).astype(int)
    return df


# ---------------------------------------------------------------------------------------
# Power BI star schema
# ---------------------------------------------------------------------------------------
def export_powerbi(df: pd.DataFrame) -> None:
    C.POWERBI.mkdir(parents=True, exist_ok=True)
    products = (df[["product", "product_full"]].drop_duplicates().sort_values("product")
                  .reset_index(drop=True))
    products.insert(0, "product_id", range(1, len(products) + 1))
    terms = pd.read_csv(C.TABLES / "topic_terms.csv").groupby("theme")["top_terms"].first()
    themes = pd.DataFrame({"theme": sorted(df["theme"].unique())})
    themes.insert(0, "theme_id", range(1, len(themes) + 1))
    themes["top_terms"] = themes["theme"].map(terms).fillna("")
    companies = (df[["company", "company_group"]].drop_duplicates().sort_values("company")
                   .reset_index(drop=True))
    companies.insert(0, "company_id", range(1, len(companies) + 1))
    dates = pd.DataFrame({"date": pd.date_range(df["date_received"].min(), df["date_received"].max())})
    dates["year"] = dates["date"].dt.year
    dates["quarter"] = "Q" + dates["date"].dt.quarter.astype(str)
    dates["month_number"] = dates["date"].dt.month
    dates["month"] = dates["date"].dt.strftime("%b %Y")
    dates["month_start"] = dates["date"].dt.to_period("M").dt.start_time.dt.date
    dates["weekday"] = dates["date"].dt.day_name()

    fact = df.merge(products[["product_id", "product"]], on="product") \
             .merge(themes[["theme_id", "theme"]], on="theme") \
             .merge(companies[["company_id", "company"]], on="company")
    fact["route_correct"] = np.where(fact["route_pred"].isna(), np.nan,
                                     (fact["route_pred"] == fact["product"]).astype(float))
    cols = ["complaint_id", "date_received", "product_id", "theme_id", "company_id", "state", "zip3",
            "is_texas", "is_houston", "issue", "sub_issue", "company_response", "relief",
            "monetary_relief", "timely", "days_to_company", "older_american", "servicemember",
            *SIGNAL_COLS, "sentiment", "tone_quintile", "is_duplicate_text", "narrative_words",
            "relief_score", "relief_priority_decile", "in_holdout", "route_pred", "route_confidence",
            "route_correct"]
    fact[cols].to_csv(C.POWERBI / "fact_complaints.csv", index=False)
    products.to_csv(C.POWERBI / "dim_product.csv", index=False)
    themes.to_csv(C.POWERBI / "dim_theme.csv", index=False)
    companies.to_csv(C.POWERBI / "dim_company.csv", index=False)
    dates.to_csv(C.POWERBI / "dim_date.csv", index=False)
    print(f"Power BI tables -> {C.POWERBI.relative_to(C.ROOT)}/ (fact rows: {len(fact):,})")


# ---------------------------------------------------------------------------------------
# Dashboard data
# ---------------------------------------------------------------------------------------
def pick_excerpts(df: pd.DataFrame, per_theme: int = 2) -> dict:
    """Short, strongly on-theme narratives (unique text, 40-90 words) for the 'Voice of the
    consumer' panel. Redaction masks (XXXX) are kept so the page can show them as bars."""
    out = {}
    pool = df[(df["dup_count"] == 1) & df["narrative_words"].between(40, 90)]
    for theme, g in pool.groupby("theme"):
        if theme == "Too short to classify":
            continue
        g = g.sort_values("theme_strength", ascending=False).head(300)
        picks, seen = [], []
        for _, r in g.iterrows():
            text = re.sub(r"\s+", " ", r["narrative"]).strip()
            text = re.sub(r"\{\$([\d,\.]+)\}", r"$\1", text)
            if len(text) > 520:
                continue
            words = set(re.findall(r"[a-z]+", text.lower()))
            if any(len(words & s) / len(words | s) > 0.5 for s in seen):
                continue                     # skip near-copies of an excerpt already chosen
            seen.append(words)
            picks.append({"id": int(r["complaint_id"]), "product": r["product"],
                          "outcome": r["company_response"], "text": text})
            if len(picks) == per_theme:
                break
        out[theme] = picks
    return out


def export_dashboard(df: pd.DataFrame) -> None:
    products = df["product"].value_counts().index.tolist()
    themes = [t for t in df["theme"].value_counts().index if t != "Too short to classify"] + \
             ["Too short to classify"]
    companies = df.loc[df["company_group"] != "All other companies", "company_group"] \
                  .value_counts().index.tolist() + ["All other companies"]
    months = sorted(df["month"].unique())

    geo = np.where(df["is_houston"] == 1, 2, np.where(df["is_texas"] == 1, 1, 0))
    signals = sum(df[c].astype(int).values << i for i, c in enumerate(SIGNAL_COLS))
    b0 = df["product"].map({p: i for i, p in enumerate(products)}).values | (geo << 4) | \
         (df["relief"].values << 6) | (df["monetary_relief"].values << 7)
    b1 = df["theme"].map({t: i for i, t in enumerate(themes)}).values | (df["tone_quintile"].values << 5)
    b2 = df["company_group"].map({c: i for i, c in enumerate(companies)}).values | \
         (df["month"].map({m: i for i, m in enumerate(months)}).values << 4)
    b3 = signals | (df["timely"].values << 6) | (df["in_holdout"].values << 7)
    b4 = (df["relief_priority_decile"].values - 1) | (df["is_duplicate_text"].values << 4) | \
         (df["older_american"].values << 5) | (df["servicemember"].values << 6)
    packed = np.stack([b0, b1, b2, b3, b4], axis=1).astype(np.uint8).ravel()

    T = lambda name: pd.read_csv(C.TABLES / f"{name}.csv")
    routing = T("routing_report").rename(columns={"Unnamed: 0": "label"})
    auto = T("routing_automation")
    ok = auto[auto["accuracy_on_auto_routed"] >= 0.95].iloc[0]
    relief_cmp = T("relief_model_comparison")
    deciles = T("relief_deciles")
    phrases = T("phrase_relief_gaps")
    confusion = pd.read_csv(C.TABLES / "routing_confusion.csv", index_col=0)
    pairs = [(a, b, confusion.loc[a, b]) for a in confusion.index for b in confusion.columns if a != b]
    pairs = sorted(pairs, key=lambda x: -x[2])[:4]
    topic_terms = T("topic_terms").groupby("theme")["top_terms"].first()

    acc_row = routing[routing["label"] == "accuracy"].iloc[0]
    macro_row = routing[routing["label"] == "macro avg"].iloc[0]
    data = {
        "meta": {
            "n": int(len(df)), "date_min": str(df["date_received"].min())[:10],
            "date_max": str(df["date_received"].max())[:10], "complete_months": list(C.COMPLETE_MONTHS),
            "source": "CFPB Consumer Complaint Database (extract distributed with Hvitfeldt & Silge, "
                      "Supervised Machine Learning for Text Analysis in R)",
        },
        "dims": {"product": products, "theme": themes, "company": companies, "month": months,
                 "company_display": [C.COMPANY_DISPLAY.get(c, c) for c in companies],
                 "other_company_count": int(df.loc[df["company_group"] == "All other companies", "company"].nunique()),
                 "geo": GEO_LABELS, "signal": SIGNAL_LABELS, "tone": TONE_LABELS},
        "packed": base64.b64encode(packed.tobytes()).decode("ascii"),
        "theme_terms": {t: ", ".join(topic_terms.get(t, "").split(", ")[:6]) for t in themes},
        "excerpts": pick_excerpts(df),
        "models": {
            "routing": {
                "accuracy": round(float(acc_row["precision"]), 4),
                "macro_f1": round(float(macro_row["f1-score"]), 4),
                "n_test": int(macro_row["support"]),
                "per_class": [{"product": r["label"], "precision": r["precision"], "recall": r["recall"],
                               "f1": r["f1-score"], "support": int(r["support"])}
                              for _, r in routing[routing["label"].isin(products)].iterrows()],
                "automation": auto.dropna().to_dict(orient="records"),
                "at95": {"threshold": float(ok["threshold"]), "coverage": float(ok["coverage"]),
                         "accuracy": float(ok["accuracy_on_auto_routed"])},
                "confusions": [{"actual": a, "predicted": b, "share": round(float(v), 4)} for a, b, v in pairs],
            },
            "relief": {
                "comparison": relief_cmp.to_dict(orient="records"),
                "deciles": deciles.to_dict(orient="records"),
                "phrases_up": phrases[phrases["direction"].str.startswith("more")][["phrase", "complaints", "relief_rate", "gap_pts"]].to_dict(orient="records"),
                "phrases_down": phrases[phrases["direction"].str.startswith("less")][["phrase", "complaints", "relief_rate", "gap_pts"]].to_dict(orient="records"),
            },
        },
    }
    C.DASHBOARD.mkdir(parents=True, exist_ok=True)
    out = C.DASHBOARD / "dashboard_data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Dashboard data -> {out.relative_to(C.ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    df = load_all()
    export_powerbi(df)
    export_dashboard(df)


if __name__ == "__main__":
    main()
