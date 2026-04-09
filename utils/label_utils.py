"""
Label utilities for multi-label pathology classification.

This module provides helper functions for:
1. Parsing pathology labels from JSON format
2. Calculating class weights for imbalanced data
3. Formatting predictions for readability
"""

import json
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Union


# Label order (must match dataset.py)
LABEL_NAMES = ["disc_herniation", "disc_bulging", "spondylolisthesis", "disc_narrowing"]


def parse_pathology_labels(pathology_details: Union[str, Dict]) -> torch.Tensor:
    """
    Convert pathology_details JSON or dict to 4-element tensor.

    Args:
        pathology_details: JSON string or dictionary with pathology flags

    Returns:
        torch.Tensor of shape [4] with values [0, 1, 0, 1] etc.
        Order: [disc_herniation, disc_bulging, spondylolisthesis, disc_narrowing]

    Example:
        >>> details = '{"disc_herniation": 0, "disc_bulging": 1, "spondylolisthesis": 0, "disc_narrowing": 1}'
        >>> labels = parse_pathology_labels(details)
        >>> labels
        tensor([0., 1., 0., 1.])
    """
    # Parse JSON if string
    if isinstance(pathology_details, str):
        try:
            pathology_details = json.loads(pathology_details)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string: {pathology_details}")

    # Create tensor
    labels = torch.zeros(4, dtype=torch.float32)

    for i, label_name in enumerate(LABEL_NAMES):
        labels[i] = float(pathology_details.get(label_name, 0))

    return labels


def get_label_weights(manifest_path: Union[str, Path]) -> List[float]:
    """
    Calculate positive class weights from manifest for handling class imbalance.

    Weights are calculated as: total_samples / (num_classes * positive_samples_per_class)
    This is the inverse frequency weighting.

    Args:
        manifest_path: Path to pathology_training_manifest.csv

    Returns:
        List of 4 positive class weights [herniation, bulging, spondylolisthesis, narrowing]

    Example:
        >>> weights = get_label_weights('data/processed/pathology_training_manifest.csv')
        >>> weights
        [14.5, 3.7, 30.4, 21.2]
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load manifest
    df = pd.read_csv(manifest_path)

    # Parse pathology_details
    df["pathology_details_parsed"] = df["pathology_details"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Count positive samples for each label
    total_samples = len(df)
    positive_counts = []

    for label_name in LABEL_NAMES:
        positive_count = (
            df["pathology_details_parsed"]
            .apply(lambda x: x.get(label_name, 0) if isinstance(x, dict) else 0)
            .sum()
        )
        positive_counts.append(positive_count)

    # Calculate weights: total_samples / (num_classes * positive_count)
    # For binary classification with imbalance, we use: total_samples / (2 * positive_count)
    weights = []
    for positive_count in positive_counts:
        if positive_count > 0:
            # Standard positive class weight: total_neg / total_pos
            # Or: total_samples / (2 * positive_count)
            weight = total_samples / (2.0 * positive_count)
        else:
            # If no positive samples, use a default weight
            weight = 1.0
        weights.append(weight)

    return weights


def get_label_weights_torch(manifest_path: Union[str, Path]) -> torch.Tensor:
    """
    Get label weights as torch.Tensor for use in BCEWithLogitsLoss.

    Args:
        manifest_path: Path to pathology_training_manifest.csv

    Returns:
        torch.Tensor of shape [4] with positive class weights
    """
    weights = get_label_weights(manifest_path)
    return torch.tensor(weights, dtype=torch.float32)


def label_names() -> List[str]:
    """
    Return ordered list of pathology label names.

    Returns:
        List of label names in order: [disc_herniation, disc_bulging, spondylolisthesis, disc_narrowing]
    """
    return LABEL_NAMES.copy()


def format_predictions(
    logits: torch.Tensor, threshold: float = 0.5, apply_sigmoid: bool = True
) -> Dict[str, Union[float, bool]]:
    """
    Convert model logits to readable prediction format.

    Args:
        logits: Model output logits of shape [4] or [batch_size, 4]
        threshold: Threshold for binary classification (default: 0.5)
        apply_sigmoid: Whether to apply sigmoid to logits (default: True)

    Returns:
        Dictionary with predictions for each pathology type

    Example:
        >>> logits = torch.tensor([0.2, 2.5, -1.0, 1.8])
        >>> preds = format_predictions(logits)
        >>> preds
        {
            'disc_herniation': {'probability': 0.55, 'predicted': True},
            'disc_bulging': {'probability': 0.92, 'predicted': True},
            'spondylolisthesis': {'probability': 0.27, 'predicted': False},
            'disc_narrowing': {'probability': 0.86, 'predicted': True}
        }
    """
    # Handle batch dimension
    if logits.dim() == 2:
        # If batch, take first sample
        logits = logits[0]

    # Apply sigmoid if needed
    if apply_sigmoid:
        probs = torch.sigmoid(logits)
    else:
        probs = logits

    # Convert to numpy for easier handling
    probs_np = probs.detach().cpu().numpy()

    # Create predictions dictionary
    predictions = {}
    for i, label_name in enumerate(LABEL_NAMES):
        prob = float(probs_np[i])
        predicted = prob >= threshold
        predictions[label_name] = {"probability": prob, "predicted": bool(predicted)}

    return predictions


def format_predictions_batch(
    logits: torch.Tensor, threshold: float = 0.5, apply_sigmoid: bool = True
) -> List[Dict[str, Union[float, bool]]]:
    """
    Convert batch of model logits to readable prediction format.

    Args:
        logits: Model output logits of shape [batch_size, 4]
        threshold: Threshold for binary classification (default: 0.5)
        apply_sigmoid: Whether to apply sigmoid to logits (default: True)

    Returns:
        List of dictionaries, one per sample in batch
    """
    batch_size = logits.shape[0]
    predictions_list = []

    for i in range(batch_size):
        sample_logits = logits[i]
        predictions_list.append(
            format_predictions(sample_logits, threshold, apply_sigmoid)
        )

    return predictions_list


def get_label_statistics(manifest_path: Union[str, Path]) -> Dict:
    """
    Get statistics about label distribution in the manifest.

    Args:
        manifest_path: Path to pathology_training_manifest.csv

    Returns:
        Dictionary with label statistics
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load manifest
    df = pd.read_csv(manifest_path)

    # Parse pathology_details
    df["pathology_details_parsed"] = df["pathology_details"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    total_samples = len(df)
    stats = {"total_samples": total_samples, "labels": {}}

    for label_name in LABEL_NAMES:
        positive_count = (
            df["pathology_details_parsed"]
            .apply(lambda x: x.get(label_name, 0) if isinstance(x, dict) else 0)
            .sum()
        )
        negative_count = total_samples - positive_count
        positive_rate = positive_count / total_samples if total_samples > 0 else 0

        stats["labels"][label_name] = {
            "positive": int(positive_count),
            "negative": int(negative_count),
            "positive_rate": float(positive_rate),
            "weight": float(total_samples / (2.0 * positive_count))
            if positive_count > 0
            else 1.0,
        }

    return stats
