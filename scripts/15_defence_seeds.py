"""Experiment 6b - are the adversarial-training results a lucky seed?

Adversarial training draws a random perturbation for every training message,
so its results depend on that draw. This reruns the three headline cases from
experiment 6 with five augmentation seeds. The test set and its attacks are
fixed, so only the training-side randomness varies.

Writes results/defence_seeds.csv.

Usage:  python scripts/15_defence_seeds.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RESULTS, SEED
from src.data import grouped_split, load_sms_spam
from src.defences import augment, with_normaliser
from src.evaluate import best_threshold, metrics_at, scores
from src.models import build_model
from src.perturb import ATTACKS, apply_attack

SEEDS = [SEED + 1000 * k for k in range(1, 6)]   # 1042 is experiment 6's
RATE = 0.50
CASES = [
    ("logreg_word", "adv_train", "zero_width"),
    ("logreg_word", "adv_train", "typo"),
    ("logreg_union", "normalise+adv_train", "typo"),
]


def main() -> int:
    df = load_sms_spam()
    train_df, test_df = grouped_split(df)
    tr, val = train_test_split(train_df, test_size=0.2, random_state=SEED,
                               stratify=train_df["label"])

    rows = []
    for model_name, defence, attack in CASES:
        attacked = apply_attack(test_df["text"].tolist(), attack, RATE,
                                seed=SEED)
        for setting in ["all_attacks", "leave_one_attack_out"]:
            train_attacks = (list(ATTACKS) if setting == "all_attacks"
                             else [a for a in ATTACKS if a != attack])
            for seed in SEEDS:
                texts, labels = augment(tr["text"], tr["label"],
                                        train_attacks, seed=seed)
                model = build_model(model_name)
                if "normalise" in defence:
                    model = with_normaliser(model)
                model.fit(texts, labels)
                thr, _ = best_threshold(val["label"],
                                        scores(model, val["text"]))
                hit = metrics_at(test_df["label"], scores(model, attacked),
                                 thr)
                clean = metrics_at(test_df["label"],
                                   scores(model, test_df["text"]), thr)
                rows.append(dict(model=model_name, defence=defence,
                                 attack=attack, setting=setting, seed=seed,
                                 recall=hit["recall"], clean_f1=clean["f1"],
                                 clean_fp=clean["fp"]))
            sub = pd.DataFrame(rows[-len(SEEDS):])
            print(f"{model_name:<13} {defence:<20} {attack:<11} "
                  f"{setting:<21} recall {sub.recall.min():.3f}-"
                  f"{sub.recall.max():.3f}  clean FP "
                  f"{sub.clean_fp.min()}-{sub.clean_fp.max()}")

    pd.DataFrame(rows).to_csv(RESULTS / "defence_seeds.csv", index=False)
    print(f"\nwrote {RESULTS / 'defence_seeds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
