"""Experiment 2 - how far does accuracy fall under evasion?

Trains on clean text, then evaluates on test sets rewritten with each attack
in src/perturb.py at increasing strength. Writes results/robustness.csv and a
plot at results/robustness.png.

Usage:  python scripts/03_robustness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RESULTS, SEED
from src.data import grouped_split, load_sms_spam
from src.evaluate import best_threshold, metrics_at, scores
from src.models import build_model
from src.perturb import ATTACKS, apply_attack

RATES = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
COMPARE = ["logreg_word", "logreg_char", "logreg_union"]


def main() -> int:
    df = load_sms_spam()
    train_df, test_df = grouped_split(df)
    tr, val = train_test_split(train_df, test_size=0.2, random_state=SEED,
                               stratify=train_df["label"])

    rows = []
    for model_name in COMPARE:
        print(f"\n--- {model_name} ---")
        model = build_model(model_name)
        model.fit(tr["text"], tr["label"])
        thr, _ = best_threshold(val["label"], scores(model, val["text"]))

        for attack in ATTACKS:
            for rate in RATES:
                texts = (test_df["text"].tolist() if rate == 0.0
                         else apply_attack(test_df["text"].tolist(), attack,
                                           rate, seed=SEED))
                m = metrics_at(test_df["label"], scores(model, texts), thr)
                m.update(model=model_name, attack=attack, rate=rate)
                rows.append(m)
            clean = rows[-len(RATES)]["recall"]
            worst = rows[-1]["recall"]
            print(f"  {attack:<12} recall {clean:.3f} -> {worst:.3f} "
                  f"at rate={RATES[-1]}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "robustness.csv", index=False)

    # One panel per attack; each line is a feature representation.
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(4 * len(ATTACKS), 3.6),
                             sharey=True)
    for ax, attack in zip(axes, ATTACKS):
        sub = out[out["attack"] == attack]
        for model_name in COMPARE:
            s = sub[sub["model"] == model_name].sort_values("rate")
            ax.plot(s["rate"], s["recall"], marker="o", label=model_name)
        ax.set_title(attack)
        ax.set_xlabel("perturbation rate")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("spam recall")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Spam recall under evasion attacks (trained on clean text)")
    fig.tight_layout()
    fig.savefig(RESULTS / "robustness.png", dpi=150)

    print(f"\nwrote {RESULTS / 'robustness.csv'} and robustness.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
