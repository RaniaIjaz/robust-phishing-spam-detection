"""Model definitions.

A deliberate ladder of increasing capacity, so the experiments can show what
each step actually buys. Character n-grams are included because they are the
natural defence against the character-level evasion attacks in perturb.py -
word features break completely when a token is rewritten, character features
degrade more gracefully.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEED


def _word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           strip_accents=None, lowercase=True)


def _char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, lowercase=True)


def build_model(name: str) -> Pipeline:
    """Return an unfitted pipeline by name."""
    if name == "nb_word":
        return Pipeline([("tfidf", _word_tfidf()),
                         ("clf", MultinomialNB())])

    if name == "logreg_word":
        return Pipeline([("tfidf", _word_tfidf()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced",
                                                    random_state=SEED))])

    if name == "logreg_char":
        return Pipeline([("tfidf", _char_tfidf()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced",
                                                    random_state=SEED))])

    if name == "logreg_union":
        feats = FeatureUnion([("w", _word_tfidf()), ("c", _char_tfidf())])
        return Pipeline([("tfidf", feats),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced",
                                                    random_state=SEED))])

    if name == "svm_union":
        feats = FeatureUnion([("w", _word_tfidf()), ("c", _char_tfidf())])
        # LinearSVC has no predict_proba; wrap it so every model exposes the
        # same scoring interface and thresholds stay comparable.
        return Pipeline([("tfidf", feats),
                         ("clf", CalibratedClassifierCV(
                             LinearSVC(class_weight="balanced",
                                       random_state=SEED),
                             cv=3))])

    if name == "logreg_email":
        # Capped vocabulary and truncated input keep the 82k-message email
        # corpus tractable; char n-grams over full email bodies are an order
        # of magnitude more expensive to fit than over SMS.
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                      max_features=300_000,
                                      sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=SEED)),
        ])

    if name == "logreg_email_union":
        # The deployable email model. Character n-grams are included because
        # scripts/03_robustness.py shows word-only features lose a third of
        # their recall to zero-width-character evasion. Both vocabularies are
        # capped so fitting stays tractable on full email bodies.
        feats = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                  max_features=200_000, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=5, max_features=200_000,
                                  sublinear_tf=True)),
        ])
        return Pipeline([
            ("tfidf", feats),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=SEED)),
        ])

    raise ValueError(f"unknown model: {name}")


MODELS = ["nb_word", "logreg_word", "logreg_char", "logreg_union", "svm_union"]
