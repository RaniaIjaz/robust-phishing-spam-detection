"""Classify messages with a saved model.

Usage:
    python scripts/12_predict.py sms   "WINNER!! Claim your prize, call 09061701461"
    python scripts/12_predict.py email "Your account is locked. Verify at http://..."
    python scripts/12_predict.py sms   --file messages.txt     # one per line
    python scripts/12_predict.py sms   --demo                  # built-in examples
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from config import ROOT
from src.perturb import ZERO_WIDTH, apply_attack

# Windows consoles default to cp1252, which cannot encode '£' or the
# zero-width characters the demo inserts, and crash on print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS_DIR = ROOT / "models"

LABELS = {
    "sms": ("ham", "SPAM"),
    "email": ("legitimate", "PHISHING"),
}

DEMO = {
    "sms": [
        "Hey, are we still on for dinner at 7? Let me know",
        "WINNER!! You have been selected to receive a £900 prize reward! "
        "To claim call 09061701461. Valid 12 hours only.",
        "Can you pick up milk on the way home",
        "URGENT! Your mobile number has won £2000. Call 09058094455 from "
        "landline. Claim code K52. Valid 12hrs",
    ],
    "email": [
        "Meeting moved\nHi team, the Thursday sync is moved to 3pm. "
        "Agenda attached. Thanks, Sarah",
        "Your account has been suspended\nDear customer, we detected unusual "
        "activity. Verify your identity within 24 hours at "
        "http://secure-verify-account.com/login or your account will be closed.",
        "Invoice #4471\nPlease find attached the invoice for last month's "
        "consulting work. Payment terms are net 30.",
        "URGENT: Confirm your password\nIT department requires all staff to "
        "re-enter their credentials at the link below to avoid mailbox "
        "deactivation.",
    ],
}


def load(task: str):
    stem = f"{task}_classifier"
    path = MODELS_DIR / f"{stem}.joblib"
    if not path.exists():
        sys.exit(f"no saved model at {path}\n"
                 f"train it first: python scripts/11_train_final.py {task}")
    card = json.loads((MODELS_DIR / f"{stem}.json").read_text())
    return joblib.load(path), card


def classify(model, card, texts):
    trunc = card.get("input_truncation_chars")
    if trunc:
        texts = [t[:trunc] for t in texts]
    probs = model.predict_proba(texts)[:, 1]
    return [(p, p >= card["threshold"]) for p in probs]


def show(task, texts, results, card):
    neg, pos = LABELS[task]
    for text, (p, hit) in zip(texts, results):
        label = pos if hit else neg
        # Strip zero-width characters for display only: they are invisible
        # anyway, and the model still scored the text that contained them.
        visible = "".join(c for c in text if c not in ZERO_WIDTH)
        preview = " ".join(visible.split())
        preview = preview if len(preview) <= 72 else preview[:69] + "..."
        print(f"  {label:<10} p={p:.3f}   {preview}")
    print(f"\n  threshold {card['threshold']:.3f} "
          f"(false negatives weighted {card['fn_cost_ratio']:.0f}x)")


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in LABELS:
        print(__doc__)
        return 2

    task, arg = sys.argv[1], sys.argv[2]
    model, card = load(task)

    if arg == "--demo":
        texts = DEMO[task]
        print(f"\n{task} classifier - clean examples\n")
        show(task, texts, classify(model, card, texts), card)

        # Show the robustness property live: same messages, evasion applied.
        attacked = apply_attack(texts, "zero_width", rate=0.5, seed=0)
        print(f"\nsame messages with zero-width-character evasion "
              f"(invisible to a human reader)\n")
        show(task, attacked, classify(model, card, attacked), card)
    elif arg == "--file":
        if len(sys.argv) < 4:
            sys.exit("--file needs a path")
        texts = [ln for ln in Path(sys.argv[3]).read_text(
                 encoding="utf-8").splitlines() if ln.strip()]
        show(task, texts, classify(model, card, texts), card)
    else:
        texts = [" ".join(sys.argv[2:])]
        show(task, texts, classify(model, card, texts), card)

    if "WARNING" in card:
        print(f"\n  note: {card['expected_performance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
