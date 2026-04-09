"""
Threshold optimization utilities for multi-label classification.

This module provides functions for finding and applying optimal thresholds
per label to improve multi-label classification performance.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import precision_recall_curve
import json
from pathlib import Path


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: List[str],
    metric: str = "f1",
    return_metrics: bool = False,
) -> Dict[str, float]:
    """
    Find optimal thresholds for each label using precision-recall curves.

    Args:
        y_true: True labels array of shape [n_samples, n_labels]
        y_proba: Predicted probabilities array of shape [n_samples, n_labels]
        label_names: List of label names
        metric: Metric to optimize ('f1', 'precision', 'recall', or 'f1_balanced')
        return_metrics: If True, also return metrics at optimal thresholds

    Returns:
        Dictionary mapping label names to optimal thresholds
        If return_metrics=True, returns (thresholds_dict, metrics_dict)
    """
    optimal_thresholds = {}
    optimal_metrics = {}

    for i, label_name in enumerate(label_names):
        true = y_true[:, i]
        proba = y_proba[:, i]

        # Skip if no positive samples
        if np.sum(true) == 0:
            optimal_thresholds[label_name] = 0.5
            if return_metrics:
                optimal_metrics[label_name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
            continue

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(true, proba)

        if metric == "f1":
            # Maximize F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
        elif metric == "precision":
            # Maximize precision (at recall >= 0.5)
            valid_mask = recall >= 0.5
            if np.any(valid_mask):
                optimal_idx = np.argmax(precision[valid_mask])
                optimal_idx = np.where(valid_mask)[0][optimal_idx]
            else:
                optimal_idx = np.argmax(precision)
        elif metric == "recall":
            # Maximize recall (at precision >= 0.5)
            valid_mask = precision >= 0.5
            if np.any(valid_mask):
                optimal_idx = np.argmax(recall[valid_mask])
                optimal_idx = np.where(valid_mask)[0][optimal_idx]
            else:
                optimal_idx = np.argmax(recall)
        elif metric == "f1_balanced":
            # Balance precision and recall (minimize |precision - recall|)
            diff = np.abs(precision - recall)
            optimal_idx = np.argmin(diff)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Get optimal threshold
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5

        optimal_thresholds[label_name] = float(optimal_threshold)

        if return_metrics:
            # Calculate metrics at optimal threshold
            pred_optimal = (proba >= optimal_threshold).astype(int)
            tp = np.sum((true == 1) & (pred_optimal == 1))
            fp = np.sum((true == 0) & (pred_optimal == 1))
            fn = np.sum((true == 1) & (pred_optimal == 0))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec + 1e-10)

            optimal_metrics[label_name] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "threshold": float(optimal_threshold),
            }

    if return_metrics:
        return optimal_thresholds, optimal_metrics
    return optimal_thresholds


def apply_thresholds(
    y_proba: np.ndarray, thresholds: Dict[str, float], label_names: List[str]
) -> np.ndarray:
    """
    Apply label-specific thresholds to probability predictions.

    Args:
        y_proba: Predicted probabilities array of shape [n_samples, n_labels]
        thresholds: Dictionary mapping label names to thresholds
        label_names: List of label names (must match y_proba columns)

    Returns:
        Binary predictions array of shape [n_samples, n_labels]
    """
    y_pred = np.zeros_like(y_proba, dtype=int)

    for i, label_name in enumerate(label_names):
        threshold = thresholds[0].get(label_name, 0.5)
        y_pred[:, i] = (y_proba[:, i] >= threshold).astype(int)

    return y_pred


def apply_thresholds_torch(
    y_proba: torch.Tensor, thresholds: Dict[str, float], label_names: List[str]
) -> torch.Tensor:
    """
    Apply label-specific thresholds to probability predictions (PyTorch version).

    Args:
        y_proba: Predicted probabilities tensor of shape [n_samples, n_labels]
        thresholds: Dictionary mapping label names to thresholds
        label_names: List of label names (must match y_proba columns)

    Returns:
        Binary predictions tensor of shape [n_samples, n_labels]
    """
    device = y_proba.device
    y_pred = torch.zeros_like(y_proba, dtype=torch.int)

    for i, label_name in enumerate(label_names):
        threshold = thresholds.get(label_name, 0.5)
        y_pred[:, i] = (y_proba[:, i] >= threshold).int()

    return y_pred


def save_thresholds(
    thresholds: Dict[str, float],
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None,
):
    """
    Save thresholds to JSON file.

    Args:
        thresholds: Dictionary mapping label names to thresholds
        filepath: Path to save JSON file
        metadata: Optional metadata to include (e.g., metric used, dataset info)
    """
    filepath = Path(filepath)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {"thresholds": thresholds, "metadata": metadata or {}}

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Thresholds saved to: {filepath}")


def load_thresholds(filepath: Union[str, Path]) -> Tuple[Dict[str, float], Dict]:
    """
    Load thresholds from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Tuple of (thresholds_dict, metadata_dict)
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    thresholds = data.get("thresholds", {})
    metadata = data.get("metadata", {})

    print(f"✓ Thresholds loaded from: {filepath}")
    return thresholds, metadata


def compare_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: List[str],
    thresholds_dict: Dict[str, float],
    default_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance with different threshold strategies.

    Args:
        y_true: True labels array of shape [n_samples, n_labels]
        y_proba: Predicted probabilities array of shape [n_samples, n_labels]
        label_names: List of label names
        thresholds_dict: Dictionary of optimal thresholds per label
        default_threshold: Default threshold to compare against

    Returns:
        Dictionary with comparison metrics for each label
    """
    comparison = {}

    for i, label_name in enumerate(label_names):
        true = y_true[:, i]
        proba = y_proba[:, i]

        # Default threshold predictions
        pred_default = (proba >= default_threshold).astype(int)

        # Optimal threshold predictions
        optimal_threshold = thresholds_dict.get(label_name, default_threshold)
        pred_optimal = (proba >= optimal_threshold).astype(int)

        # Calculate metrics for both
        def calc_metrics(y_t, y_p):
            tp = np.sum((y_t == 1) & (y_p == 1))
            fp = np.sum((y_t == 0) & (y_p == 1))
            fn = np.sum((y_t == 1) & (y_p == 0))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec + 1e-10)

            return {"precision": prec, "recall": rec, "f1": f1}

        metrics_default = calc_metrics(true, pred_default)
        metrics_optimal = calc_metrics(true, pred_optimal)

        comparison[label_name] = {
            "default_threshold": default_threshold,
            "optimal_threshold": optimal_threshold,
            "default_metrics": metrics_default,
            "optimal_metrics": metrics_optimal,
            "improvement": {
                "precision": metrics_optimal["precision"]
                - metrics_default["precision"],
                "recall": metrics_optimal["recall"] - metrics_default["recall"],
                "f1": metrics_optimal["f1"] - metrics_default["f1"],
            },
        }

    return comparison
