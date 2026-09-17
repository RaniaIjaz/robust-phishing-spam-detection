"""Experiment 3 - what is the model actually getting wrong?

Dumps the most confidently misclassified messages and the highest-weighted
features to results/, for manual reading. The point is to look at the data,
not only at the metrics.

Usage:  python scripts/04_error_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from config import RESULTS, SEED
from src.data import grouped_split, load_sms_spam
from src.evaluate import best_threshold, metrics_at, scores, top_errors
from src.models import build_model

MODEL = "logreg_union"

# Printed messages contain characters cp1252 consoles cannot encode.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def top_features(model, n: int = 25) -> dict:
    """Most spam-indicative and most ham-indicative features."""
    names = np.asarray(model.named_steps["tfidf"].get_feature_names_out())
    coefs = model.named_steps["clf"].coef_.ravel()
    order = np.argsort(coefs)
    return {
        "spam_indicative": [
            {"feature": str(names[i]), "weight": round(float(coefs[i]), 3)}
            for i in order[::-1][:n]
        ],
        "ham_indicative": [
            {"feature": str(names[i]), "weight": round(float(coefs[i]), 3)}
            for i in order[:n]
        ],
    }


def main() -> int:
    df = load_sms_spam()
    train_df, test_df = grouped_split(df)
    tr, val = train_test_split(train_df, test_size=0.2, random_state=SEED,
                               stratify=train_df["label"])

    model = build_model(MODEL)
    model.fit(tr["text"], tr["label"])

    y_score = scores(model, test_df["text"])
    thr, _ = best_threshold(val["label"], scores(model, val["text"]))
    m = metrics_at(test_df["label"], y_score, thr)

    report = {
        "model": MODEL,
        "metrics": m,
        "missed_spam": top_errors(test_df["text"], test_df["label"],
                                  y_score, thr, "fn"),
        "false_alarms": top_errors(test_df["text"], test_df["label"],
                                   y_score, thr, "fp"),
        "features": top_features(model),
    }

    path = RESULTS / "error_analysis.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    print(f"threshold={thr:.3f}  recall={m['recall']:.4f}  "
          f"precision={m['precision']:.4f}")
    print(f"{m['fn']} missed spam, {m['fp']} false alarms\n")

    print("most confidently missed spam:")
    for e in report["missed_spam"][:5]:
        print(f"  [{e['score']:.3f}] {e['text'][:100]}")

    print("\ntop spam-indicative features:")
    print("  " + ", ".join(f["feature"]
                           for f in report["features"]["spam_indicative"][:12]))

    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
