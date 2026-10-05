"""TF-IDF with a vocabulary that is identical on every machine.

scikit-learn's `max_features` keeps the most frequent terms, but terms tied at the cutoff are
ordered by an unstable sort whose result depends on the CPU (AVX-512 and AVX2 machines keep
different tied terms). A handful of different terms is enough to change a topic model, so
this helper applies the same rule and breaks ties alphabetically instead.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline


def fit_tfidf(texts, max_features: int, sublinear_tf: bool = True, **params):
    """Fit TF-IDF on `texts`, keeping the `max_features` most frequent terms.

    `params` are CountVectorizer settings (ngram_range, min_df, max_df, stop_words, ...).
    Returns the fitted vectorizer (a pipeline with .transform and .get_feature_names_out)
    and the TF-IDF matrix for `texts`.
    """
    counter = CountVectorizer(**params)
    counts = counter.fit_transform(texts)                      # columns in alphabetical order
    totals = np.asarray(counts.sum(axis=0)).ravel()
    keep = np.sort(np.argsort(-totals, kind="stable")[:max_features])
    vocabulary = counter.get_feature_names_out()[keep].tolist()

    tfidf = TfidfTransformer(sublinear_tf=sublinear_tf)
    X = tfidf.fit_transform(counts[:, keep])
    tokenizer = {k: v for k, v in params.items() if k not in ("min_df", "max_df")}
    vectorizer = make_pipeline(CountVectorizer(vocabulary=vocabulary, **tokenizer), tfidf)
    return vectorizer, X
