"""Experiment 1 - does the split strategy change the answer?

Trains every model under both an i.i.d. random split and a leakage-aware
grouped split, and writes the comparison to results/baselines.csv.

Usage:  python scripts/02_run_baselines.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RESULTS, SEED
from src.data import grouped_split, leakage_report, load_sms_spam, random_split
from src.evaluate import best_threshold, metrics_at, scores
from src.models import MODELS, build_model


def run(train_df, test_df, model_name: str, split_name: str) -> dict:
    # Carve a validation slice out of train for threshold selection, so the
    # test set is never used to tune anything.
    tr, val = train_test_split(train_df, test_size=0.2, random_state=SEED,
                               stratify=train_df["label"])

    model = build_model(model_name)
    model.fit(tr["text"], tr["label"])

    thr, _ = best_threshold(val["label"], scores(model, val["text"]))
    m = metrics_at(test_df["label"], scores(model, test_df["text"]), thr)
    m.update(model=model_name, split=split_name,
             n_train=len(tr), n_test=len(test_df))
    return m


def main() -> int:
    df = load_sms_spam()
    print(f"loaded {len(df)} messages "
          f"({df['label'].mean():.1%} spam)\n")

    leak = leakage_report(df)
    print("leakage report:", json.dumps(leak, indent=2))
    (RESULTS / "leakage_report.json").write_text(json.dumps(leak, indent=2))

    splits = {
        "random": random_split(df),
        "grouped": grouped_split(df),
    }

    rows = []
    for split_name, (train_df, test_df) in splits.items():
        print(f"\n--- {split_name} split "
              f"(train={len(train_df)}, test={len(test_df)}) ---")
        for model_name in MODELS:
            row = run(train_df, test_df, model_name, split_name)
            rows.append(row)
            print(f"  {model_name:<14} "
                  f"F1={row['f1']:.4f}  recall={row['recall']:.4f}  "
                  f"PR-AUC={row['pr_auc']:.4f}")

    out = pd.DataFrame(rows)
    cols = ["split", "model", "f1", "recall", "precision", "pr_auc",
            "roc_auc", "accuracy", "brier", "threshold",
            "tp", "fp", "tn", "fn", "n_train", "n_test"]
    out = out[cols]
    out.to_csv(RESULTS / "baselines.csv", index=False)

    # The headline comparison: same models, two ways of splitting.
    print("\n=== F1 by split ===")
    pivot = out.pivot(index="model", columns="split", values="f1")
    pivot["delta"] = (pivot["grouped"] - pivot["random"]).round(4)
    print(pivot.to_string())
    pivot.to_csv(RESULTS / "split_comparison.csv")

    print(f"\nwrote {RESULTS / 'baselines.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
