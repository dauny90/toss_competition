from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def weighted_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted log loss with equal class contribution (0.5/0.5)."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    y_true = y_true.astype(np.float64)
    n = y_true.size
    n_pos = y_true.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        # Fallback to standard logloss if only one class exists.
        return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return float(-np.mean(w_pos * y_true * np.log(y_pred) + w_neg * (1.0 - y_true) * np.log(1.0 - y_pred)))


def average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_pred))


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ap = average_precision(y_true, y_pred)
    wll = weighted_log_loss(y_true, y_pred)
    return float(0.5 * ap + 0.5 * (1.0 / (1.0 + wll)))
