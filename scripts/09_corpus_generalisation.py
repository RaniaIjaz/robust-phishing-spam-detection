"""Experiment 4 - is this benchmark measuring phishing, or measuring Enron?

Four evaluations of the same data, in increasing order of honesty:

  A  corpus probe        can a model identify which corpus a message is from?
  B  naive mixed         random split over all six corpora pooled
  C  leave-one-corpus-out train on five, test on the sixth
  D  within-corpus       train and test inside one mixed-label corpus

If A is easy and B is near-perfect while C collapses, then B was measuring
corpus identity, not phishing.

Usage:  python scripts/09_corpus_generalisation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config import RESULTS, SEED
from src.corpora import leave_one_corpus_out, load_phishing
from src.evaluate import metrics_at, scores
from src.models import build_model

MODEL = "logreg_email"
TRUNCATE = 2000          # chars; keeps fitting time bounded
MIXED = ["ceas", "spamassassin", "enron", "ling"]   # both classes present


def prep(df):
    df = df.copy()
    df["text"] = df["text"].str.slice(0, TRUNCATE)
    return df


def probe_corpus_identity(df) -> float:
    """A - 5-fold accuracy at predicting `source` from text alone."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=5,
                                  max_features=100_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=600, random_state=SEED)),
    ])
    sample = df.sample(n=min(20_000, len(df)), random_state=SEED)
    return float(cross_val_score(pipe, sample["text"], sample["source"],
                                 cv=3, scoring="accuracy").mean())


def fit_eval(train, test, note: str) -> dict:
    model = build_model(MODEL)
    model.fit(train["text"], train["label"])
    y_score = scores(model, test["text"])

    single_class = test["label"].nunique() < 2
    if single_class:
        # Only one label present: precision/AUC are undefined, recall is not.
        thr = 0.5
        recall = float(((y_score >= thr) == test["label"].iloc[0]).mean())
        return {"eval": note, "n_train": len(train), "n_test": len(test),
                "test_phish_pct": round(100.0 * test["label"].mean(), 1),
                "recall": round(recall, 4), "f1": None, "precision": None,
                "pr_auc": None, "note": "single-class test set: recall only"}

    m = metrics_at(test["label"], y_score, 0.5)
    return {"eval": note, "n_train": len(train), "n_test": len(test),
            "test_phish_pct": round(100.0 * test["label"].mean(), 1),
            "recall": m["recall"], "f1": m["f1"],
            "precision": m["precision"], "pr_auc": m["pr_auc"], "note": ""}


def main() -> int:
    df = prep(load_phishing())
    print(f"{len(df)} messages, {df['label'].mean():.1%} phishing\n")

    acc = probe_corpus_identity(df)
    print(f"A  corpus-identity probe: {acc:.4f} accuracy "
          f"({len(df['source'].unique())} classes, "
          f"chance = {1 / df['source'].nunique():.3f})\n")

    rows = []

    tr, te = train_test_split(df, test_size=0.25, random_state=SEED,
                              stratify=df["label"])
    rows.append(fit_eval(tr, te, "B naive mixed (random split)"))
    print(f"B  naive mixed        F1={rows[-1]['f1']}")

    print("\nC  leave-one-corpus-out:")
    for corpus in sorted(df["source"].unique()):
        tr, te = leave_one_corpus_out(df, corpus)
        rows.append(fit_eval(tr, te, f"C holdout={corpus}"))
        r = rows[-1]
        print(f"   {corpus:<14} phish={r['test_phish_pct']:>5}%  "
              f"recall={r['recall']}  F1={r['f1']}")

    print("\nD  within-corpus (both classes present):")
    for corpus in MIXED:
        sub = df[df["source"] == corpus]
        tr, te = train_test_split(sub, test_size=0.25, random_state=SEED,
                                  stratify=sub["label"])
        rows.append(fit_eval(tr, te, f"D within={corpus}"))
        r = rows[-1]
        print(f"   {corpus:<14} F1={r['f1']}  recall={r['recall']}")

    out = pd.DataFrame(rows)
    out.insert(0, "probe_accuracy", round(acc, 4))
    out.to_csv(RESULTS / "corpus_generalisation.csv", index=False)
    print(f"\nwrote {RESULTS / 'corpus_generalisation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
