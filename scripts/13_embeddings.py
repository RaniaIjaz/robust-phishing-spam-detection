"""Experiment 5 - word embeddings vs TF-IDF.

  Part 1  SMS, leakage-aware split: GloVe / word2vec / FastText vs TF-IDF
  Part 2  SMS robustness: the five evasion attacks, with out-of-vocabulary
          rates, to test whether subword embeddings (FastText) survive the
          way character TF-IDF did while whole-word embeddings collapse
  Part 3  Email corpus leakage with GloVe: corpus probe, pooled split,
          leave-one-corpus-out, within-corpus - the same four evaluations as
          experiment 4, on a representation that never saw this data

Only GloVe runs in Part 3. word2vec and FastText would have to be retrained
for every held-out corpus (training them on all six would leak the held-out
corpus's vocabulary into the representation), which is too slow on CPU.

Usage:  python scripts/13_embeddings.py [sms|email|all]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RESULTS, SEED
from src.data import grouped_split, load_sms_spam
from src.embeddings import (SOURCES, MeanEmbedding, build_embedding_model,
                            cosine)
from src.evaluate import best_threshold, metrics_at, scores
from src.models import build_model
from src.perturb import ATTACKS, apply_attack, homoglyph

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RATES = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]     # same grid as experiment 2
TFIDF_REFERENCE = ["logreg_word", "logreg_char"]
EMAIL_TRUNCATE = 2000                             # same as experiment 4
MIXED = ["ceas", "spamassassin", "enron", "ling"]


PROBE_WORDS = ["free", "call", "prize", "love", "home", "password"]


def neighbours_report(fitted) -> None:
    """Qualitative check that the vectors learned meaning, plus the homoglyph
    case that separates whole-word from subword models."""
    print("\n--- nearest neighbours ---")
    report = {}
    for name in SOURCES:
        kv = fitted[name][0].named_steps["embed"].kv_
        report[name] = {}
        for word in PROBE_WORDS:
            near = kv.most_similar(word, topn=6)
            report[name][word] = [[w, round(s, 3)] for w, s in near]
            shown = ", ".join(w for w, _ in near) if near else "(not in vocabulary)"
            print(f"  {name:<9} {word:<9} {shown}")

    attacked = homoglyph("free", rate=1.0, seed=0)
    print(f"\n--- homoglyph attack: 'free' -> {ascii(attacked)} "
          f"(looks identical) ---")
    report["homoglyph"] = {"clean": "free", "attacked": ascii(attacked)}
    for name in SOURCES:
        kv = fitted[name][0].named_steps["embed"].kv_
        try:
            sim = cosine(kv[attacked], kv["free"])
            near = ", ".join(w for w, _ in kv.most_similar(attacked, topn=5))
            print(f"  {name:<9} cosine to 'free' = {sim:.3f}; nearest: {near}")
            report["homoglyph"][name] = round(sim, 3)
        except KeyError:
            print(f"  {name:<9} no vector at all - the word vanishes from "
                  f"the message")
            report["homoglyph"][name] = None

    (RESULTS / "embedding_neighbours.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def sms_parts() -> None:
    df = load_sms_spam()
    train_df, test_df = grouped_split(df)
    tr, val = train_test_split(train_df, test_size=0.2, random_state=SEED,
                               stratify=train_df["label"])

    print("=== Part 1: SMS, leakage-aware split ===")
    rows, fitted = [], {}
    for name in TFIDF_REFERENCE + SOURCES:
        model = (build_model(name) if name in TFIDF_REFERENCE
                 else build_embedding_model(name))
        model.fit(tr["text"], tr["label"])
        thr, _ = best_threshold(val["label"], scores(model, val["text"]))
        m = metrics_at(test_df["label"], scores(model, test_df["text"]), thr)
        m["model"] = name
        rows.append(m)
        fitted[name] = (model, thr)
        print(f"  {name:<12} F1={m['f1']:.4f}  recall={m['recall']:.4f}  "
              f"precision={m['precision']:.4f}  PR-AUC={m['pr_auc']:.4f}")
    pd.DataFrame(rows).to_csv(RESULTS / "embeddings_sms.csv", index=False)

    neighbours_report(fitted)

    print("\n=== Part 2: SMS robustness (spam recall, whole-word OOV rate) ===")
    rob = []
    for name in SOURCES:
        model, thr = fitted[name]
        embed: MeanEmbedding = model.named_steps["embed"]
        for attack in ATTACKS:
            for rate in RATES:
                texts = (test_df["text"].tolist() if rate == 0.0 else
                         apply_attack(test_df["text"].tolist(), attack, rate,
                                      seed=SEED))
                m = metrics_at(test_df["label"], scores(model, texts), thr)
                rob.append({"model": name, "attack": attack, "rate": rate,
                            "recall": m["recall"], "f1": m["f1"],
                            "oov_rate": round(embed.oov_rate(texts), 4)})
            first, last = rob[-len(RATES)], rob[-1]
            print(f"  {name:<9} {attack:<11} recall {first['recall']:.3f} -> "
                  f"{last['recall']:.3f}   OOV {first['oov_rate']:.1%} -> "
                  f"{last['oov_rate']:.1%}")
    pd.DataFrame(rob).to_csv(RESULTS / "embeddings_robustness.csv",
                             index=False)


def fit_eval(Xtr, ytr, Xte, yte, note: str) -> dict:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000,
                                           class_weight="balanced",
                                           random_state=SEED))
    clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    row = {"eval": note, "n_train": len(ytr), "n_test": len(yte),
           "test_phish_pct": round(100.0 * float(np.mean(yte)), 1)}
    if len(np.unique(yte)) < 2:
        # Single-class test set: only recall is defined.
        row.update(recall=round(float(np.mean(s >= 0.5)), 4), f1=None,
                   precision=None, pr_auc=None)
    else:
        m = metrics_at(yte, s, 0.5)
        row.update(recall=m["recall"], f1=m["f1"], precision=m["precision"],
                   pr_auc=m["pr_auc"])
    return row


def email_part() -> None:
    from src.corpora import load_phishing

    print("\n=== Part 3: email corpus leakage, GloVe mean vectors ===")
    df = load_phishing()
    df["text"] = df["text"].str.slice(0, EMAIL_TRUNCATE)

    # GloVe is fixed, so every message's vector is independent of any split
    # and can be computed once. No fold ever sees held-out text.
    embed = MeanEmbedding(source="glove").fit([])
    print(f"  embedding {len(df)} emails...")
    X = embed.transform(df["text"].tolist())
    y = df["label"].to_numpy()
    src = df["source"].to_numpy()
    print(f"  GloVe OOV rate over the corpus: "
          f"{embed.oov_rate(df['text'].sample(5000, random_state=SEED)):.1%}")

    sample = np.random.default_rng(SEED).choice(len(df), size=20_000,
                                                replace=False)
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000,
                                             random_state=SEED))
    acc = float(cross_val_score(probe, X[sample], src[sample], cv=3,
                                scoring="accuracy").mean())
    print(f"\n  A  corpus-identity probe: {acc:.4f}  "
          f"(TF-IDF probe was 0.9191; chance 0.167)")

    rows = []
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25,
                                      random_state=SEED, stratify=y)
    rows.append(fit_eval(X[idx_tr], y[idx_tr], X[idx_te], y[idx_te],
                         "B naive mixed (random split)"))
    print(f"  B  naive mixed        F1={rows[-1]['f1']}")

    print("\n  C  leave-one-corpus-out:")
    for corpus in sorted(np.unique(src)):
        tr, te = src != corpus, src == corpus
        rows.append(fit_eval(X[tr], y[tr], X[te], y[te],
                             f"C holdout={corpus}"))
        r = rows[-1]
        print(f"     {corpus:<14} phish={r['test_phish_pct']:>5}%  "
              f"recall={r['recall']}  F1={r['f1']}")

    print("\n  D  within-corpus:")
    for corpus in MIXED:
        idx = np.flatnonzero(src == corpus)
        a, b = train_test_split(idx, test_size=0.25, random_state=SEED,
                                stratify=y[idx])
        rows.append(fit_eval(X[a], y[a], X[b], y[b], f"D within={corpus}"))
        r = rows[-1]
        print(f"     {corpus:<14} F1={r['f1']}  recall={r['recall']}")

    out = pd.DataFrame(rows)
    out.insert(0, "probe_accuracy", round(acc, 4))
    out.to_csv(RESULTS / "embeddings_email.csv", index=False)


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in {"sms", "email", "all"}:
        print(__doc__)
        return 2
    if which in {"sms", "all"}:
        sms_parts()
    if which in {"email", "all"}:
        email_part()
    print("\nwrote results/embeddings_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
