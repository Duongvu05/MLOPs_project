"""
Figure saving utilities for notebooks.

This module provides helper functions for saving figures generated in notebooks
with consistent naming conventions and proper organization.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.figure


def get_figures_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the path to the figures directory.

    Args:
        subdir: Optional subdirectory within figures/ (e.g., "exploration", "analysis", "results")

    Returns:
        Path to the figures directory or subdirectory
    """
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "notebooks" / "figures"

    if subdir:
        figures_dir = figures_dir / subdir
        figures_dir.mkdir(parents=True, exist_ok=True)

    return figures_dir


def save_figure(
    fig: Union[plt.Figure, matplotlib.figure.Figure],
    filename: str,
    subdir: Optional[str] = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    formats: list = ["png", "pdf"],
    overwrite: bool = False,
    add_timestamp: bool = False,
) -> dict:
    """
    Save a figure with consistent naming and in multiple formats.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        subdir: Optional subdirectory within figures/ (e.g., "exploration", "analysis", "results")
        dpi: Resolution for raster formats (default: 300)
        bbox_inches: Bounding box setting (default: "tight")
        formats: List of formats to save (default: ["png", "pdf"])
        overwrite: Whether to overwrite existing files (default: False)
        add_timestamp: Whether to add timestamp to filename (default: False)

    Returns:
        Dictionary mapping format to saved file path
    """
    figures_dir = get_figures_dir(subdir)

    # Add timestamp if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"

    saved_files = {}

    for fmt in formats:
        filepath = figures_dir / f"{filename}.{fmt}"

        # Check if file exists
        if filepath.exists() and not overwrite:
            raise FileExistsError(
                f"Figure already exists: {filepath}\n"
                f"Set overwrite=True to replace it, or use a different filename."
            )

        # Save figure
        fig.savefig(
            filepath,
            format=fmt,
            dpi=dpi,
            bbox_inches=bbox_inches,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        saved_files[fmt] = str(filepath)

    return saved_files


def save_figure_simple(
    fig: Union[plt.Figure, matplotlib.figure.Figure],
    filename: str,
    subdir: Optional[str] = None,
    format: str = "png",
    dpi: int = 300,
) -> str:
    """
    Simple wrapper to save a figure in a single format.

    Args:
        fig: Matplotlib figure object
        filename: Filename (with or without extension)
        subdir: Optional subdirectory within figures/
        format: File format (default: "png")
        dpi: Resolution (default: 300)

    Returns:
        Path to saved file
    """
    # Remove extension if present
    if filename.endswith((".png", ".pdf", ".svg", ".jpg", ".jpeg")):
        filename = os.path.splitext(filename)[0]

    saved_files = save_figure(
        fig=fig,
        filename=filename,
        subdir=subdir,
        formats=[format],
        dpi=dpi,
    )

    return saved_files[format]


def create_figure_subdir(subdir: str) -> Path:
    """
    Create a subdirectory in figures/ for organizing figures by category.

    Common subdirectories:
    - "exploration": Data exploration and EDA figures
    - "preprocessing": Data preprocessing and cleaning visualizations
    - "analysis": Statistical analysis and model analysis figures
    - "results": Model results, predictions, and evaluation figures
    - "comparisons": Model comparison and ablation study figures

    Args:
        subdir: Name of subdirectory to create

    Returns:
        Path to created subdirectory
    """
    figures_dir = get_figures_dir(subdir)
    return figures_dir


def get_standard_figure_name(
    base_name: str,
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None,
    metric: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Generate a standardized figure filename.

    Format: [experiment_][model_][base_name][_metric][_suffix]

    Args:
        base_name: Base name for the figure (e.g., "loss_curve", "confusion_matrix")
        experiment_name: Optional experiment name
        model_name: Optional model name
        metric: Optional metric name
        suffix: Optional suffix (e.g., "train", "val", "test")

    Returns:
        Standardized filename
    """
    parts = []

    if experiment_name:
        parts.append(experiment_name)
    if model_name:
        parts.append(model_name)

    parts.append(base_name)

    if metric:
        parts.append(metric)
    if suffix:
        parts.append(suffix)

    return "_".join(parts)


def close_figure(
    fig: Union[plt.Figure, matplotlib.figure.Figure], save: bool = False, **save_kwargs
):
    """
    Close a figure, optionally saving it first.

    Args:
        fig: Matplotlib figure object
        save: Whether to save the figure before closing
        **save_kwargs: Arguments to pass to save_figure if save=True
    """
    if save:
        if "filename" not in save_kwargs:
            raise ValueError("filename is required when save=True")
        save_figure(fig, **save_kwargs)

    plt.close(fig)


def setup_plot_style(style: str = "seaborn-v0_8", figsize: tuple = (10, 6), **kwargs):
    """
    Set up consistent plotting style for all figures.

    Args:
        style: Matplotlib style (default: "seaborn-v0_8")
        figsize: Default figure size (default: (10, 6))
        **kwargs: Additional style parameters
    """
    plt.style.use(style)
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = kwargs.get("dpi", 100)
    plt.rcParams["savefig.dpi"] = kwargs.get("save_dpi", 300)
    plt.rcParams["savefig.bbox"] = kwargs.get("bbox_inches", "tight")

    # Update with any additional parameters
    for key, value in kwargs.items():
        if key not in ["dpi", "save_dpi", "bbox_inches"]:
            plt.rcParams[key] = value
