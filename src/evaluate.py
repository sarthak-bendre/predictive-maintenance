"""Metrics for an imbalanced classifier and cost-based threshold selection.

Accuracy is deliberately not the headline metric: always predicting "no failure"
already scores ~96.6% on this data.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import COST_FN, COST_FP


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "cost": float(fn * COST_FN + fp * COST_FP),
    }


def best_threshold(y_true, proba, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Threshold that minimises total cost = FN * cost_fn + FP * cost_fp.

    Should be called on out-of-fold training predictions, never on the test set.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in grid:
        pred = proba >= t
        fn = np.sum((~pred) & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        costs.append(fn * cost_fn + fp * cost_fp)
    return round(float(grid[int(np.argmin(costs))]), 2)
