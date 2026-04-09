"""
Attention visualization utilities for multi-sequence fusion models.

This module provides functions to visualize and analyze attention weights
from pathology-aware attention mechanisms, helping understand which sequences
contribute most to each pathology prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from utils.figure_utils import save_figure, setup_plot_style
from utils.label_utils import label_names


def visualize_sequence_attention(
    attention_weights: Dict[str, Dict[str, torch.Tensor]],
    sequence_names: List[str],
    pathology_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualize sequence attention weights per pathology.

    Args:
        attention_weights: Dictionary from model.get_attention_weights()
            Format: {'scale_0': {'pathology_name': [B, num_sequences]}, ...}
        sequence_names: List of sequence names
        pathology_names: List of pathology names (default: from label_utils)
        output_path: Optional path to save figure
        figsize: Figure size
        dpi: Figure resolution

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()

    if pathology_names is None:
        pathology_names = label_names()

    # Average attention weights across batch and scales
    avg_attention = {}
    for path_name in pathology_names:
        path_weights = []
        for scale_key, scale_weights in attention_weights.items():
            if path_name in scale_weights:
                weights = scale_weights[path_name].detach().cpu().numpy()
                # Average over batch dimension
                weights = weights.mean(axis=0)  # [num_sequences]
                path_weights.append(weights)

        if path_weights:
            # Average over scales
            avg_attention[path_name] = np.mean(path_weights, axis=0)

    # Create figure with subplots
    n_pathologies = len(pathology_names)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Plot attention weights for each pathology
    for idx, path_name in enumerate(pathology_names):
        ax = axes[idx]

        if path_name in avg_attention:
            weights = avg_attention[path_name]

            # Create bar plot
            bars = ax.bar(
                range(len(sequence_names)),
                weights,
                color=plt.cm.viridis(np.linspace(0, 1, len(sequence_names))),
            )

            # Customize plot
            ax.set_xticks(range(len(sequence_names)))
            ax.set_xticklabels(sequence_names, rotation=45, ha="right")
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"{path_name.replace('_', ' ').title()}", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim([0, 1])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No attention weights available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{path_name.replace('_', ' ').title()}", fontweight="bold")

    plt.suptitle(
        "Sequence Attention Weights by Pathology",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        save_figure(fig, str(output_path), dpi=dpi, bbox_inches="tight")

    return fig


def visualize_attention_heatmap(
    attention_weights: Dict[str, Dict[str, torch.Tensor]],
    sequence_names: List[str],
    pathology_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """
    Create heatmap of attention weights (pathologies x sequences).

    Args:
        attention_weights: Dictionary from model.get_attention_weights()
        sequence_names: List of sequence names
        pathology_names: List of pathology names
        output_path: Optional path to save figure
        figsize: Figure size
        dpi: Figure resolution

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()

    if pathology_names is None:
        pathology_names = label_names()

    # Average attention weights across batch and scales
    attention_matrix = []
    for path_name in pathology_names:
        path_weights = []
        for scale_key, scale_weights in attention_weights.items():
            if path_name in scale_weights:
                weights = scale_weights[path_name].detach().cpu().numpy()
                weights = weights.mean(axis=0)  # Average over batch
                path_weights.append(weights)

        if path_weights:
            avg_weights = np.mean(path_weights, axis=0)  # Average over scales
            attention_matrix.append(avg_weights)
        else:
            attention_matrix.append(np.zeros(len(sequence_names)))

    attention_matrix = np.array(attention_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Format pathology names for display
    display_pathology_names = [p.replace("_", " ").title() for p in pathology_names]

    sns.heatmap(
        attention_matrix,
        xticklabels=sequence_names,
        yticklabels=display_pathology_names,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Attention Weight"},
        ax=ax,
    )

    ax.set_title(
        "Sequence Attention Weights by Pathology",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("MRI Sequence", fontsize=11)
    ax.set_ylabel("Pathology Type", fontsize=11)

    plt.tight_layout()

    if output_path:
        save_figure(fig, str(output_path), dpi=dpi, bbox_inches="tight")

    return fig


def save_attention_analysis(
    attention_weights: Dict[str, Dict[str, torch.Tensor]],
    sequence_names: List[str],
    output_dir: Path,
    model_name: str = "model",
    pathology_names: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Save comprehensive attention analysis (figures and data).

    Args:
        attention_weights: Dictionary from model.get_attention_weights()
        sequence_names: List of sequence names
        output_dir: Directory to save outputs
        model_name: Name of model (for file naming)
        pathology_names: List of pathology names

    Returns:
        Dictionary mapping output type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pathology_names is None:
        pathology_names = label_names()

    saved_files = {}

    # 1. Visualize attention weights (bar charts)
    fig1 = visualize_sequence_attention(
        attention_weights, sequence_names, pathology_names
    )
    bar_chart_path = output_dir / f"attention_weights_bars_{model_name}.png"
    save_figure(fig1, str(bar_chart_path), dpi=300, bbox_inches="tight")
    saved_files["bar_chart"] = bar_chart_path
    plt.close(fig1)

    # 2. Visualize attention heatmap
    fig2 = visualize_attention_heatmap(
        attention_weights, sequence_names, pathology_names
    )
    heatmap_path = output_dir / f"attention_weights_heatmap_{model_name}.png"
    save_figure(fig2, str(heatmap_path), dpi=300, bbox_inches="tight")
    saved_files["heatmap"] = heatmap_path
    plt.close(fig2)

    # 3. Save attention weights as CSV
    # Average across batch and scales
    attention_data = []
    for path_name in pathology_names:
        path_weights = []
        for scale_key, scale_weights in attention_weights.items():
            if path_name in scale_weights:
                weights = scale_weights[path_name].detach().cpu().numpy()
                weights = weights.mean(axis=0)  # Average over batch
                path_weights.append(weights)

        if path_weights:
            avg_weights = np.mean(path_weights, axis=0)  # Average over scales
            row = {"pathology": path_name}
            for seq_idx, seq_name in enumerate(sequence_names):
                row[seq_name] = float(avg_weights[seq_idx])
            attention_data.append(row)

    if attention_data:
        df = pd.DataFrame(attention_data)
        csv_path = output_dir / f"attention_weights_{model_name}.csv"
        df.to_csv(csv_path, index=False)
        saved_files["csv"] = csv_path

    return saved_files


def analyze_attention_patterns(
    attention_weights: Dict[str, Dict[str, torch.Tensor]],
    sequence_names: List[str],
    pathology_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze attention patterns and extract insights.

    Args:
        attention_weights: Dictionary from model.get_attention_weights()
        sequence_names: List of sequence names
        pathology_names: List of pathology names

    Returns:
        Dictionary with analysis results:
        {
            'pathology_name': {
                'most_important_sequence': 'seq_name',
                'attention_distribution': {...},
                'clinical_alignment': 'high/medium/low'
            }
        }
    """
    if pathology_names is None:
        pathology_names = label_names()

    # Clinical expectations (from ClinicalSequencePrior)
    clinical_expectations = {
        "disc_herniation": "ax_t2",
        "disc_bulging": "sag_t2",  # or multiple
        "spondylolisthesis": "sag_t2",
        "disc_narrowing": "sag_t1",
    }

    results = {}

    # Average attention weights
    for path_name in pathology_names:
        path_weights = []
        for scale_key, scale_weights in attention_weights.items():
            if path_name in scale_weights:
                weights = scale_weights[path_name].detach().cpu().numpy()
                weights = weights.mean(axis=0)  # Average over batch
                path_weights.append(weights)

        if path_weights:
            avg_weights = np.mean(path_weights, axis=0)  # Average over scales

            # Find most important sequence
            most_important_idx = np.argmax(avg_weights)
            most_important_seq = sequence_names[most_important_idx]

            # Create attention distribution
            attention_dist = {
                seq: float(avg_weights[idx]) for idx, seq in enumerate(sequence_names)
            }

            # Check clinical alignment
            expected_seq = clinical_expectations.get(path_name.lower(), None)
            if expected_seq:
                expected_idx = (
                    sequence_names.index(expected_seq)
                    if expected_seq in sequence_names
                    else -1
                )
                if expected_idx >= 0:
                    expected_weight = avg_weights[expected_idx]
                    max_weight = avg_weights.max()

                    if expected_weight >= max_weight * 0.9:
                        alignment = "high"
                    elif expected_weight >= max_weight * 0.7:
                        alignment = "medium"
                    else:
                        alignment = "low"
                else:
                    alignment = "unknown"
            else:
                alignment = "unknown"

            results[path_name] = {
                "most_important_sequence": most_important_seq,
                "attention_distribution": attention_dist,
                "clinical_alignment": alignment,
            }

    return results
