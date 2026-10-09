import numpy as np

from src.evaluate import best_threshold, classification_metrics


def test_metrics_on_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["recall"] == m["precision"] == m["pr_auc"] == 1.0
    assert m["cost"] == 0


def test_expensive_misses_push_threshold_down():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(rng.normal(0.35 + 0.3 * y, 0.12), 0, 1)
    t_cheap = best_threshold(y, proba, cost_fn=1, cost_fp=1)
    t_costly = best_threshold(y, proba, cost_fn=20, cost_fp=1)
    assert t_costly < t_cheap
