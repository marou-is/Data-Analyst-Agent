from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Point-wise precision/recall/F1 plus event recall over labelled windows."""
    t = y_true.to_numpy(dtype=bool)
    p = y_pred.to_numpy(dtype=bool)
    tp = int((t & p).sum())
    precision = tp / p.sum() if p.sum() else 0.0
    recall = tp / t.sum() if t.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "event_recall": event_recall(t, p),
    }


def event_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of contiguous labelled windows containing at least one detection."""
    events = _segments(np.asarray(y_true, dtype=bool))
    if not events:
        return 0.0
    hits = sum(bool(np.asarray(y_pred, dtype=bool)[s:e].any()) for s, e in events)
    return hits / len(events)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def forecast_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = t - p
    nonzero = t != 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err[nonzero] / t[nonzero])) * 100) if nonzero.any() else float("nan"),
    }
