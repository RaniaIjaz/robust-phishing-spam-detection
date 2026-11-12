"""Experiment 6 - can the evasion attacks be defended against, honestly?

Experiment 2 attacked the SMS models; this one tries to fix them and then
attacks the fix. Four defence settings per model:

  none                 trained on clean text (the experiment 2 baseline)
  normalise            rule-based input normalisation (src/defences.py)
  adv_train            adversarial training on all five attacks
  normalise+adv_train  both

and two checks that a defence is not simply grading itself:

  leave-one-attack-out   adversarial training on four attacks, tested on the
                         fifth. A defence that only works on tricks it was
                         shown is a lookup table, not robustness.
  held-out homoglyphs    an attacker who has read the normaliser and uses
                         lookalike characters it does not list
                         (perturb.HELD_OUT_ATTACKS). No defence is trained
                         on it.

Thresholds are chosen on clean validation data, as in experiment 2, so
defended and undefended models are compared at their own honest operating
points. The clean row shows what each defence costs on ordinary mail.

Writes results/defences.csv, results/defences_summary.csv and
results/defences.png.

Usage:  python scripts/14_defences.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RESULTS, SEED
from src.data import grouped_split, load_sms_spam
from src.defences import augment, with_normaliser
from src.evaluate import best_threshold, metrics_at, scores
from src.models import build_model
from src.perturb import ATTACKS, HELD_OUT_ATTACKS, apply_attack

MODELS = ["logreg_word", "logreg_char", "logreg_union"]
DEFENCES = ["none", "normalise", "adv_train", "normalise+adv_train"]
EVAL_ATTACKS = list(ATTACKS) + list(HELD_OUT_ATTACKS)
RATES = [0.20, 0.50]


def fit(model_name, defence, tr, train_attacks):
    model = build_model(model_name)
    if "normalise" in defence:
        model = with_normaliser(model)
    texts, labels = tr["text"].tolist(), tr["label"].tolist()
    if "adv_train" in defence:
        texts, labels = augment(texts, labels, train_attacks)
    model.fit(texts, labels)
    return model


def evaluate(model, thr, test_df, attacks, attacked):
    """Score one fitted model on clean text and on each attack at each rate."""
    rows = []
    m = metrics_at(test_df["label"], scores(model, test_df["text"]), thr)
    m.update(attack="clean", rate=0.0)
    rows.append(m)
    for attack in attacks:
        for rate in RATES:
            m = metrics_at(test_df["label"],
                           scores(model, attacked[(attack, rate)]), thr)
            m.update(attack=attack, rate=rate)
            rows.append(m)
    return rows


def main() -> int:
    df = load_sms_spam()
    train_df, test_df = grouped_split(df)
    tr, val = train_test_split(train_df, test_size=0.2, random_state=SEED,
                               stratify=train_df["label"])

    # Attack the test set once, with the same seed as experiment 2, so the
    # "none" rows reproduce results/robustness.csv exactly.
    attacked = {(a, r): apply_attack(test_df["text"].tolist(), a, r, seed=SEED)
                for a in EVAL_ATTACKS for r in RATES}

    rows = []

    # Full defences: trained on every attack in ATTACKS.
    for model_name in MODELS:
        for defence in DEFENCES:
            t0 = time.time()
            model = fit(model_name, defence, tr, list(ATTACKS))
            thr, _ = best_threshold(val["label"], scores(model, val["text"]))
            for r in evaluate(model, thr, test_df, EVAL_ATTACKS, attacked):
                r.update(model=model_name, defence=defence,
                         setting="all_attacks")
                rows.append(r)
            print(f"{model_name:<13} {defence:<20} {time.time() - t0:5.1f}s")

    # Leave-one-attack-out: adversarial training never sees the attack it is
    # scored on. Only the attacked rows are kept; clean cost is reported by
    # the full-defence rows above.
    for model_name in MODELS:
        for defence in ["adv_train", "normalise+adv_train"]:
            for held in ATTACKS:
                others = [a for a in ATTACKS if a != held]
                model = fit(model_name, defence, tr, others)
                thr, _ = best_threshold(val["label"],
                                        scores(model, val["text"]))
                for r in evaluate(model, thr, test_df, [held], attacked):
                    if r["attack"] == "clean":
                        continue
                    r.update(model=model_name, defence=defence,
                             setting="leave_one_attack_out")
                    rows.append(r)
            print(f"{model_name:<13} {defence:<20} leave-one-attack-out done")

    out = pd.DataFrame(rows)
    lead = ["model", "defence", "setting", "attack", "rate"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.to_csv(RESULTS / "defences.csv", index=False)

    # Summary: spam recall at the strongest attack rate, plus clean cost.
    strong = out[(out["rate"] == max(RATES)) | (out["attack"] == "clean")]
    summary = strong.pivot_table(index=["model", "defence", "setting"],
                                 columns="attack", values="recall")
    summary = summary[["clean"] + EVAL_ATTACKS] \
        if "clean" in summary.columns else summary
    clean = out[out["attack"] == "clean"].set_index(
        ["model", "defence", "setting"])[["f1", "fp", "threshold"]]
    summary = summary.join(clean.add_prefix("clean_"))
    summary.to_csv(RESULTS / "defences_summary.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nSpam recall at rate", max(RATES), "(clean column = no attack)")
    print(summary.round(3).to_string())

    _plot(out)
    print(f"\nwrote {RESULTS / 'defences.csv'}, defences_summary.csv "
          f"and defences.png")
    return 0


def _plot(out: pd.DataFrame) -> None:
    """Grouped bars: recall per attack at the strongest rate, one panel per
    model, one bar per defence; hatched bars are leave-one-attack-out."""
    rate = max(RATES)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS), 4),
                             sharey=True)
    x = np.arange(len(EVAL_ATTACKS))
    series = [("none", "all_attacks"), ("normalise", "all_attacks"),
              ("adv_train", "all_attacks"),
              ("adv_train", "leave_one_attack_out")]
    labels = ["no defence", "normalise", "adv. training",
              "adv. training, attack held out"]
    width = 0.8 / len(series)
    for ax, model_name in zip(axes, MODELS):
        for k, ((defence, setting), label) in enumerate(zip(series, labels)):
            sub = out[(out["model"] == model_name)
                      & (out["defence"] == defence)
                      & (out["setting"] == setting) & (out["rate"] == rate)]
            vals = [sub.loc[sub["attack"] == a, "recall"].mean()
                    for a in EVAL_ATTACKS]
            ax.bar(x + (k - 1.5) * width, vals, width, label=label,
                   hatch="//" if setting == "leave_one_attack_out" else None,
                   edgecolor="white")
        ax.set_xticks(x, [a.replace("_", "\n") for a in EVAL_ATTACKS],
                      fontsize=8)
        ax.set_title(model_name)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"spam recall at perturbation rate {rate}")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Defences against evasion (SMS, leakage-aware split). "
                 "homoglyph_unseen is never trained on.")
    fig.tight_layout()
    fig.savefig(RESULTS / "defences.png", dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
