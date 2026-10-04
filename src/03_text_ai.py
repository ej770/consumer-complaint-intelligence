"""Step 3 - AI/NLP layer: discover complaint themes and measure tone.

* Topic modeling: TF-IDF (1-2 grams) + Non-negative Matrix Factorization (NMF) learns
  18 topics from the narratives. Topics are named from their top terms and merged into
  16 business-friendly themes (three near-identical credit-repair form-letter topics are
  combined).
* Tone: VADER sentiment is scored sentence by sentence and averaged per complaint.

Outputs
  data/processed/text_features.csv.gz        complaint_id -> theme, theme strength, sentiment
  outputs/tables/topic_terms.csv             top terms per learned topic (for transparency)
  outputs/tables/theme_summary.csv           volume and outcomes per theme
  outputs/tables/theme_product_mix.csv       how themes cut across CFPB product categories
  outputs/tables/sentiment_vs_relief.csv     relief rate by tone quintile (raw + mix-adjusted)
  outputs/tables/phrase_relief_gaps.csv      phrases linked to more / less relief than expected
"""
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config as C

N_TOPICS = 18

# Each theme is defined by anchor terms; every learned topic is assigned to the theme whose
# anchors carry the most weight in that topic. This keeps labels stable even if a different
# library version returns the topics in another order.
THEMES = {
    "Customer service failures": ["customer service", "representative", "email", "spoke"],
    "Requests to remove report items": ["removed credit", "remove", "items credit", "deleted"],
    "Credit-repair form letters": ["harming credit", "partial account", "subsection", "reseller",
                                   "verifiable proof", "consumer contract", "original signed"],
    "Identity theft": ["identity theft", "victim identity", "theft report", "police report"],
    "Credit card charges & limits": ["credit card", "card company", "card account", "limit"],
    "Balances, fees & payoffs": ["account paid", "charged", "owed", "fee"],
    "Debt not owed / validation": ["collection agency", "debt collection", "collect debt", "owe"],
    "Unauthorized hard inquiries": ["hard inquiry", "hard inquiries", "inquiries credit"],
    "Bank deposits, funds & checks": ["checking account", "deposit", "deposited", "funds"],
    "Late payment reporting": ["late payment", "late payments", "reporting late"],
    "Collection calls & contact": ["phone calls", "calling", "phone number", "stop"],
    "Fraudulent accounts (not mine)": ["fraudulent accounts", "accounts belong", "following accounts"],
    "Loan repayment & forgiveness": ["student loan", "forgiveness", "repayment", "school"],
    "Dispute & verification process": ["certified", "requested", "documentation", "request"],
    "Mortgage servicing & escrow": ["escrow", "modification", "foreclosure", "property"],
    "Inaccurate bureau reporting": ["credit bureaus", "inaccurate", "bankruptcy", "agencies"],
}


def fit_topics(df: pd.DataFrame):
    """Fit TF-IDF + NMF on unique narratives so copy-paste templates do not dominate."""
    stop = list(set(ENGLISH_STOP_WORDS) | C.STOP_EXTRA)
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=25, max_df=0.4,
                          sublinear_tf=True, max_features=30000,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b")
    train = df.drop_duplicates("text_group")
    train = train[train["narrative_words"] >= 5]
    X_train = vec.fit_transform(train["narrative_clean"])
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=C.RANDOM_STATE, max_iter=300)
    nmf.fit(X_train)
    print(f"Topic model fit on {X_train.shape[0]:,} unique narratives x {X_train.shape[1]:,} terms")
    return vec, nmf


def label_topics(vec, nmf) -> dict:
    terms = vec.get_feature_names_out()
    index = {t: i for i, t in enumerate(terms)}
    comps = nmf.components_ / nmf.components_.max(axis=1, keepdims=True)
    mapping = {}
    for k, comp in enumerate(comps):
        scores = {theme: sum(comp[index[a]] for a in anchors if a in index)
                  for theme, anchors in THEMES.items()}
        mapping[k] = max(scores, key=scores.get)
    missing = set(THEMES) - set(mapping.values())
    if missing:
        print(f"WARNING: no topic matched these themes: {missing}")
    return mapping


def tone(texts: pd.Series) -> pd.DataFrame:
    """Average VADER compound score over sentences (first 40 per complaint)."""
    sia = SentimentIntensityAnalyzer()
    splitter = re.compile(r"(?<=[.!?])\s+|\n+")
    mean_compound, neg_share = [], []
    for text in texts:
        sents = [s for s in splitter.split(re.sub(r"X{2,}", "", text)) if len(s.split()) >= 3][:40]
        if not sents:
            mean_compound.append(0.0)
            neg_share.append(0.0)
            continue
        scores = [sia.polarity_scores(s)["compound"] for s in sents]
        mean_compound.append(float(np.mean(scores)))
        neg_share.append(float(np.mean([v <= -0.05 for v in scores])))
    return pd.DataFrame({"sentiment": mean_compound, "negative_sentence_share": neg_share},
                        index=texts.index)


def phrase_gaps(df: pd.DataFrame, min_docs: int = 400, top_n: int = 12) -> pd.DataFrame:
    """For every word or two-word phrase used in at least `min_docs` complaints, compare the
    relief rate of complaints containing it with the rate expected for the same product mix
    (each product's relief rate among complaints WITHOUT the phrase). Vectorised with sparse
    matrix products, so all phrases are scored at once.

    Two filters keep the list about consumer problems rather than artefacts:
      * company names: phrases where one company accounts for >35% of uses are dropped;
      * form letters: phrases where >30% of uses come from copy-paste narratives are dropped.
    """
    stop = list(set(ENGLISH_STOP_WORDS) | C.STOP_EXTRA)
    vec = CountVectorizer(binary=True, stop_words=stop, ngram_range=(1, 2), min_df=min_docs,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b")
    X = vec.fit_transform(df["narrative_clean"]).tocsc()               # complaints x phrases

    def onehot(col):
        codes = pd.Categorical(df[col]).codes
        return csr_matrix((np.ones(len(df)), (np.arange(len(df)), codes)))

    P = onehot("product")                                              # complaints x products
    relief = df["relief"].to_numpy(dtype=float)
    n_tp = (X.T @ P).toarray()                                         # phrase x product counts
    r_tp = (X.T @ P.multiply(relief[:, None])).toarray()               # phrase x product relief
    n_p = np.asarray(P.sum(axis=0)).ravel()
    r_p = np.asarray(P.multiply(relief[:, None]).sum(axis=0)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.where(n_p - n_tp > 0, (r_p - r_tp) / (n_p - n_tp), r_p / n_p)
    n_t = n_tp.sum(axis=1)
    top_company_share = np.asarray((X.T @ onehot("company")).max(axis=1).todense()).ravel() / n_t
    template_share = np.asarray(X.T @ df["is_duplicate_text"].to_numpy(dtype=float)).ravel() / n_t
    out = pd.DataFrame({
        "phrase": vec.get_feature_names_out(), "complaints": n_t,
        "relief_rate": r_tp.sum(axis=1) / n_t,
        "expected_relief_given_products": (n_tp * base).sum(axis=1) / n_t,
        "top_company_share": top_company_share, "template_share": template_share,
    })
    out["gap_pts"] = 100 * (out["relief_rate"] - out["expected_relief_given_products"])
    out = out[(out["top_company_share"] <= 0.35) & (out["template_share"] <= 0.30)]
    # Words that are too vague to read on their own (or are parts of company names)
    vague = {"systems", "corporation", "seriously", "annual", "decreased", "lowered", "protected",
             "equipment", "scores"}
    out = out[~out["phrase"].isin(vague)]

    def stem(word):
        for suffix in ("ing", "ed", "es", "s", "e"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    def pick(frame):
        chosen, used = [], set()
        for _, r in frame.iterrows():
            words = {stem(w) for w in r["phrase"].split()}
            if words & used:            # skip near-duplicates such as "fee" / "overdraft fees"
                continue
            chosen.append(r)
            used |= words
            if len(chosen) == top_n:
                break
        return pd.DataFrame(chosen)

    up = pick(out.sort_values("gap_pts", ascending=False)).assign(direction="more relief than expected")
    down = pick(out.sort_values("gap_pts")).assign(direction="less relief than expected")
    return pd.concat([up, down]).round(4)


def mix_adjusted(df: pd.DataFrame, flag: pd.Series) -> float:
    """Relief rate expected for flagged rows if they had the product-level relief rates of
    non-flagged rows (direct standardization)."""
    base = df.loc[~flag].groupby("product")["relief"].mean()
    return df.loc[flag, "product"].map(base).mean()


def main() -> None:
    df = pd.read_csv(C.CLEAN)
    df[["narrative", "narrative_clean"]] = df[["narrative", "narrative_clean"]].fillna("")
    vec, nmf = fit_topics(df)
    mapping = label_topics(vec, nmf)

    # Transparency table: top terms of every learned topic and the theme it was mapped to
    terms = vec.get_feature_names_out()
    rows = []
    for k, comp in enumerate(nmf.components_):
        top = terms[comp.argsort()[::-1][:15]]
        rows.append({"topic_id": k, "theme": mapping[k], "top_terms": ", ".join(top)})
    topic_terms = pd.DataFrame(rows)
    topic_terms.to_csv(C.TABLES / "topic_terms.csv", index=False)
    print(topic_terms.to_string(index=False))

    # Score every complaint (duplicates included)
    W = nmf.transform(vec.transform(df["narrative_clean"]))
    strength = W.max(axis=1)
    topic = W.argmax(axis=1)
    feats = pd.DataFrame({
        "complaint_id": df["complaint_id"],
        "topic_id": topic,
        "theme": [mapping[t] for t in topic],
        "theme_strength": strength.round(4),
    })
    feats.loc[strength == 0, ["theme"]] = "Too short to classify"

    print("Scoring tone with VADER ...")
    feats = feats.join(tone(df["narrative"]).round(4))
    feats.to_csv(C.PROCESSED / "text_features.csv.gz", index=False, compression="gzip")

    # ---- Theme summary ------------------------------------------------------------
    d = df.join(feats[["theme", "sentiment"]])
    overall = d["relief"].mean()
    summary = (d.groupby("theme")
                 .agg(complaints=("relief", "size"), relief_rate=("relief", "mean"),
                      monetary_rate=("monetary_relief", "mean"), avg_sentiment=("sentiment", "mean"),
                      texas_complaints=("is_texas", "sum"),
                      top_product=("product", lambda s: s.value_counts().index[0]),
                      top_product_share=("product", lambda s: s.value_counts(normalize=True).iloc[0]),
                      products_spanned=("product", lambda s: int((s.value_counts(normalize=True) >= 0.05).sum())))
                 .sort_values("complaints", ascending=False))
    summary["expected_relief_given_products"] = [mix_adjusted(d, d["theme"] == t) for t in summary.index]
    summary["share"] = summary["complaints"] / len(d)
    summary["relief_index_vs_avg"] = summary["relief_rate"] / overall
    summary.reset_index().round(4).to_csv(C.TABLES / "theme_summary.csv", index=False)
    print("\n", summary.round(3).to_string())

    mix = pd.crosstab(d["theme"], d["product"], normalize="index").round(4)
    mix.to_csv(C.TABLES / "theme_product_mix.csv")

    # ---- Does tone matter? --------------------------------------------------------
    d["tone_quintile"] = pd.qcut(d["sentiment"].rank(method="first"), 5,
                                 labels=["1 Most negative", "2", "3", "4", "5 Least negative"])
    tone_tbl = (d.groupby("tone_quintile", observed=True)
                  .agg(complaints=("relief", "size"), avg_sentiment=("sentiment", "mean"),
                       relief_rate=("relief", "mean")))
    tone_tbl["expected_relief_given_products"] = [
        mix_adjusted(d, d["tone_quintile"] == q) for q in tone_tbl.index]
    tone_tbl.reset_index().round(4).to_csv(C.TABLES / "sentiment_vs_relief.csv", index=False)
    print("\n", tone_tbl.round(3).to_string())

    # ---- Which phrases go with more (or less) relief? -------------------------------
    gaps = phrase_gaps(df)
    gaps.to_csv(C.TABLES / "phrase_relief_gaps.csv", index=False)
    print("\n", gaps[["direction", "phrase", "complaints", "relief_rate", "gap_pts"]].to_string(index=False))


if __name__ == "__main__":
    main()
