"""Metrics, threshold selection and calibration.

Accuracy is a poor headline number on this task: the corpus is ~87% ham, so a
model that never predicts spam already scores 0.87. Recall on the spam class
and the cost-weighted operating point are the numbers that matter.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FN_COST_RATIO


def scores(model, texts) -> np.ndarray:
    """Spam probability for each message."""
    return model.predict_proba(list(texts))[:, 1]


def metrics_at(y_true, y_score, threshold: float = 0.5) -> dict:
    """Full metric bundle at a fixed decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((tp + tn) / len(y_true)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                 zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred,
                                           zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def expected_cost(y_true, y_score, threshold: float,
                  fn_ratio: float = FN_COST_RATIO) -> float:
    """Cost of one operating point, with false negatives weighted heavier."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true), y_pred,
                                      labels=[0, 1]).ravel()
    return float(fp + fn_ratio * fn)


def best_threshold(y_true, y_score,
                   fn_ratio: float = FN_COST_RATIO) -> tuple[float, float]:
    """Sweep thresholds and return the cost-minimising one.

    Must be fitted on validation data, never on test - doing it on test is the
    second most common way this kind of project overstates its results.
    """
    grid = np.unique(np.round(np.asarray(y_score), 3))
    grid = np.clip(grid, 0.001, 0.999)
    costs = [(expected_cost(y_true, y_score, t, fn_ratio), t) for t in grid]
    cost, thr = min(costs)
    return float(thr), float(cost)


def top_errors(texts, y_true, y_score, threshold: float, kind: str, n: int = 25):
    """Return the most confidently wrong predictions for manual reading.

    kind: "fn" for missed spam, "fp" for wrongly flagged ham.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    texts = np.asarray(list(texts), dtype=object)

    if kind == "fn":
        mask = (y_true == 1) & (y_score < threshold)
        order = np.argsort(y_score[mask])           # most confidently missed
    else:
        mask = (y_true == 0) & (y_score >= threshold)
        order = np.argsort(-y_score[mask])          # most confidently flagged

    return [{"score": round(float(s), 4), "text": t}
            for s, t in zip(y_score[mask][order][:n], texts[mask][order][:n])]
