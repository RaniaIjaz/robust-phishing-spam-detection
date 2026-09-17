"""Train and save the deployable classifiers to models/.

Each model is saved with a JSON card recording its decision threshold and the
performance it should honestly be expected to deliver. For the email model
that is the leave-one-corpus-out range, not the pooled validation score: the
experiments show the pooled number overstates real-world recall badly.

Usage:
    python scripts/11_train_final.py sms      # ~1 min
    python scripts/11_train_final.py email    # slow: char n-grams over 82k emails
    python scripts/11_train_final.py all
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FN_COST_RATIO, RESULTS, ROOT, SEED
from src.data import grouped_split, load_sms_spam
from src.evaluate import best_threshold, metrics_at, scores
from src.models import build_model

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

EMAIL_TRUNCATE = 2000


def save(model, card: dict, stem: str) -> None:
    joblib.dump(model, MODELS_DIR / f"{stem}.joblib", compress=3)
    (MODELS_DIR / f"{stem}.json").write_text(json.dumps(card, indent=2))
    size = (MODELS_DIR / f"{stem}.joblib").stat().st_size / 1e6
    print(f"  saved models/{stem}.joblib ({size:.1f} MB) + {stem}.json")


def train_sms() -> None:
    name = "logreg_union"
    print(f"\n=== SMS classifier ({name}) ===")
    df = load_sms_spam()

    # Leakage-aware split, so the validation score on the card is honest.
    train, val = grouped_split(df, test_size=0.2)
    model = build_model(name)
    model.fit(train["text"], train["label"])

    val_scores = scores(model, val["text"])
    thr, _ = best_threshold(val["label"], val_scores)
    m = metrics_at(val["label"], val_scores, thr)
    print(f"  threshold={thr:.3f}  F1={m['f1']}  recall={m['recall']}  "
          f"precision={m['precision']}")

    save(model, {
        "task": "sms_spam",
        "model": name,
        "features": "word 1-2gram + char_wb 2-5gram TF-IDF",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(train),
        "n_validation": len(val),
        "threshold": thr,
        "fn_cost_ratio": FN_COST_RATIO,
        "validation_split": "grouped (near-duplicate clusters kept together)",
        "validation_metrics": m,
        "expected_performance": (
            f"F1 ~{m['f1']:.2f} on SMS resembling the 2011 UCI corpus. "
            "Robust to character-level evasion (see results/robustness.csv)."),
        "known_limits": [
            "Trained on 2011-era UK/Singapore SMS; modern scam styles may differ.",
            "Label noise present in training data.",
            "Not evaluated under adaptive (model-querying) attacks.",
        ],
    }, "sms_classifier")


def train_email() -> None:
    from src.corpora import load_phishing

    name = "logreg_email_union"
    print(f"\n=== Email classifier ({name}) ===")
    df = load_phishing()
    df["text"] = df["text"].str.slice(0, EMAIL_TRUNCATE)
    print(f"  {len(df)} emails; fitting word + char features (slow)")

    train, val = train_test_split(df, test_size=0.2, random_state=SEED,
                                  stratify=df["label"])
    model = build_model(name)
    model.fit(train["text"], train["label"])

    val_scores = scores(model, val["text"])
    thr, _ = best_threshold(val["label"], val_scores)
    m = metrics_at(val["label"], val_scores, thr)
    print(f"  threshold={thr:.3f}  pooled-validation F1={m['f1']}")

    # The pooled validation score is known to be inflated by corpus leakage.
    # Record the leave-one-corpus-out range as the honest expectation.
    loco = {}
    gen_path = RESULTS / "corpus_generalisation.csv"
    if gen_path.exists():
        gen = pd.read_csv(gen_path)
        for _, r in gen[gen["eval"].str.startswith("C ")].iterrows():
            loco[r["eval"].split("=")[1]] = {
                "recall": r["recall"],
                "f1": None if pd.isna(r["f1"]) else r["f1"],
            }

    save(model, {
        "task": "phishing_email",
        "model": name,
        "features": "word 1-2gram + char_wb 3-5gram TF-IDF (capped)",
        "input_truncation_chars": EMAIL_TRUNCATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(train),
        "n_validation": len(val),
        "threshold": thr,
        "fn_cost_ratio": FN_COST_RATIO,
        "validation_split": "random over pooled corpora",
        "validation_metrics": m,
        "WARNING": (
            "validation_metrics are inflated. Train and validation share "
            "source corpora, and a probe identifies the corpus with 92% "
            "accuracy, so the model can score well by recognising the "
            "corpus. Use leave_one_corpus_out below as the realistic "
            "expectation on mail from an unseen source."),
        "leave_one_corpus_out_model": (
            "logreg_email (word-only). Measured by "
            "scripts/09_corpus_generalisation.py on the word-only variant, "
            "not on this word+char model; treat as indicative rather than "
            "exact for this model."),
        "leave_one_corpus_out": loco,
        "expected_performance": (
            "Recall on phishing from an unfamiliar source may be as low as "
            "~0.66 (Nazario holdout). Do not quote the validation F1 as "
            "real-world accuracy."),
        "known_limits": [
            "Training corpora date mostly from 2002 and 2008.",
            "52% phishing base rate; real mailflow is a few percent, so "
            "precision in deployment will be substantially lower.",
            "Headers, URLs and attachments are ignored; text only.",
            "Text alone cannot separate a genuine invoice from invoice "
            "fraud: the demo's legitimate invoice email scores as phishing. "
            "Sender and header signals are needed for that distinction.",
            "Unusual characters raise the phishing score, so non-ASCII or "
            "copy-pasted legitimate mail is prone to false positives. "
            "Consistent with corpus leakage: the phishing corpora carry "
            "encoding artifacts the ham corpora do not.",
        ],
    }, "email_classifier")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "sms"
    if which not in {"sms", "email", "all"}:
        print(__doc__)
        return 2
    if which in {"sms", "all"}:
        train_sms()
    if which in {"email", "all"}:
        train_email()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
