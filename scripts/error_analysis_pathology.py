# -*- coding: utf-8 -*-
"""
Error Analysis for Pathology Classification Models

This script performs comprehensive error analysis on the best pathology classification models,
including visualization of DICOM images with predictions for correct and incorrect cases.
"""

import sys
from pathlib import Path
import warnings
import json
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pydicom

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import project modules
from utils.dataset import get_default_transforms
from utils.multi_sequence_dataset import MultiSequencePathologyDataset
from utils.label_utils import label_names
from utils.threshold_utils import load_thresholds, apply_thresholds
from utils.figure_utils import save_figure, setup_plot_style
from models.pathology_model import create_pathology_model
from training.pathology_training_utils import (
    is_multi_sequence_model,
    prepare_model_input,
)

# Set up plotting style for medical presentation
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
setup_plot_style(figsize=(10, 6))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Using device: {device}")
print(f"[OK] Project root: {project_root}")

# Configuration
BEST_MODEL_RUN_ID = "20251123_185414"  # DenseNet-121
MANIFEST_PATH = Path("../data/mri_phen/processed/pathology_training_manifest.csv")
OUTPUT_DIR = project_root / "notebooks" / "figures" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Label names
label_names_list = label_names()


def load_dicom_image(
    dicom_path: str,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
) -> np.ndarray:
    """Load and normalize DICOM image for visualization."""
    try:
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)

        # Apply windowing if available
        if window_center is None:
            window_center = getattr(ds, "WindowCenter", None)
            if isinstance(window_center, (list, tuple)):
                window_center = window_center[0]

        if window_width is None:
            window_width = getattr(ds, "WindowWidth", None)
            if isinstance(window_width, (list, tuple)):
                window_width = window_width[0]

        if window_center is not None and window_width is not None:
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            image = np.clip(image, window_min, window_max)

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        return image
    except Exception as e:
        print(f"Error loading DICOM {dicom_path}: {e}")
        return None


def visualize_sample_predictions(
    model_name: str,
    samples: List[Dict],
    output_path: Path,
    n_samples_per_category: int = 5,
):
    """
    Visualize DICOM images with predictions for correct and incorrect cases.

    Args:
        model_name: Name of the model
        samples: List of sample dictionaries with 'image_path', 'true_labels', 'pred_labels',
                 'pred_probs', 'patient_id', 'ivd_label', 'correct'
        output_path: Path to save the figure
        n_samples_per_category: Number of samples to show per category
    """
    # Organize samples by correctness and label
    correct_samples = [s for s in samples if s["correct"]]
    incorrect_samples = [s for s in samples if not s["correct"]]

    # Get samples for each label
    label_categories = {}
    for label_idx, label_name in enumerate(label_names_list):
        label_correct = [s for s in correct_samples if s["true_labels"][label_idx] == 1]
        label_incorrect = [
            s for s in incorrect_samples if s["true_labels"][label_idx] == 1
        ]

        label_categories[label_name] = {
            "correct": label_correct[:n_samples_per_category],
            "incorrect": label_incorrect[:n_samples_per_category],
        }

    # Create figure with more samples
    n_labels = len(label_names_list)
    fig, axes = plt.subplots(
        n_labels,
        n_samples_per_category * 2,
        figsize=(n_samples_per_category * 2.5, n_labels * 2.5),
    )

    if n_labels == 1:
        axes = axes.reshape(1, -1)

    for label_idx, label_name in enumerate(label_names_list):
        row_axes = axes[label_idx]

        # Correct predictions
        for col_idx, sample in enumerate(label_categories[label_name]["correct"]):
            ax = row_axes[col_idx]

            # Load and display DICOM image
            image = load_dicom_image(sample["image_path"])
            if image is not None:
                ax.imshow(image, cmap="gray", aspect="auto")

            # Add title with prediction info
            prob = sample["pred_probs"][label_idx]
            true_label = (
                "Positive" if sample["true_labels"][label_idx] == 1 else "Negative"
            )
            pred_label = (
                "Positive" if sample["pred_labels"][label_idx] == 1 else "Negative"
            )
            title = f"{true_label}→{pred_label}\nP={prob:.2f}"
            ax.set_title(title, fontsize=8, color="green", fontweight="bold")
            ax.axis("off")

        # Incorrect predictions
        for col_idx, sample in enumerate(label_categories[label_name]["incorrect"]):
            ax = row_axes[col_idx + n_samples_per_category]

            # Load and display DICOM image
            image = load_dicom_image(sample["image_path"])
            if image is not None:
                ax.imshow(image, cmap="gray", aspect="auto")

            # Add title with prediction info
            prob = sample["pred_probs"][label_idx]
            true_label = (
                "Positive" if sample["true_labels"][label_idx] == 1 else "Negative"
            )
            pred_label = (
                "Positive" if sample["pred_labels"][label_idx] == 1 else "Negative"
            )
            title = f"{true_label}→{pred_label}\nP={prob:.2f}"
            ax.set_title(title, fontsize=8, color="red", fontweight="bold")
            ax.axis("off")

        # Add label name on the left
        fig.text(
            0.02,
            0.75 - label_idx * 0.25,
            label_name.replace("_", " ").title(),
            fontsize=12,
            fontweight="bold",
            rotation=90,
            va="center",
        )

    # Add column headers
    fig.text(
        0.25, 0.98, "Correct Predictions", fontsize=12, fontweight="bold", ha="center"
    )
    fig.text(
        0.75, 0.98, "Incorrect Predictions", fontsize=12, fontweight="bold", ha="center"
    )

    plt.suptitle(
        f"Error Analysis: {model_name}\nSample Predictions by Pathology Type",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.05, 0.02, 0.98, 0.96])

    # Save figure
    save_figure(fig, str(output_path), dpi=300, bbox_inches="tight", overwrite=True)
    plt.close(fig)
    print(f"[OK] Saved visualization to {output_path}")


def analyze_errors(
    model,
    dataloader,
    dataset,
    device,
    model_name: str,
    thresholds: Dict[str, float],
    output_dir: Path,
):
    """Perform comprehensive error analysis."""
    print(f"\n{'=' * 80}")
    print(f"Error Analysis for {model_name}")
    print(f"{'=' * 80}")

    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    all_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Generating predictions")
        ):
            # Prepare input
            is_multi_seq = is_multi_sequence_model(model)
            model_input, labels = prepare_model_input(
                batch, device, is_multi_seq, model
            )

            # Get predictions
            if is_multi_seq:
                sequences, sequence_available = model_input
                logits = model(sequences, sequence_available)
            else:
                logits = model(model_input)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = apply_thresholds(probs, thresholds, label_names_list)

            # Store predictions
            all_true_labels.append(labels.cpu().numpy())
            all_pred_labels.append(preds)
            all_pred_probs.append(probs)

            # Store sample information for visualization
            batch_size = labels.size(0)
            for i in range(batch_size):
                sample_idx = batch_idx * dataloader.batch_size + i
                if sample_idx < len(dataset):
                    sample_info = dataset.df.iloc[sample_idx]

                    # Get DICOM path
                    dicom_paths = sample_info.get("dicom_paths_parsed", [])
                    sequence_types = sample_info.get("sequence_types_parsed", [])

                    # Get preferred sequence
                    preferred_seq = "SAG_T2"
                    dicom_path = None
                    if dicom_paths and sequence_types:
                        for path, seq_type in zip(dicom_paths, sequence_types):
                            if preferred_seq in str(seq_type).upper():
                                dicom_path = path
                                break
                        if dicom_path is None:
                            dicom_path = dicom_paths[0] if dicom_paths else None

                    # Check if prediction is correct
                    true_label = labels[i].cpu().numpy()
                    pred_label = preds[i]
                    correct = np.array_equal(true_label, pred_label)

                    all_samples.append(
                        {
                            "image_path": str(project_root / dicom_path)
                            if dicom_path
                            else None,
                            "true_labels": true_label,
                            "pred_labels": pred_label,
                            "pred_probs": probs[i],
                            "patient_id": sample_info.get("patient_id", "unknown"),
                            "ivd_label": sample_info.get("ivd_label", "unknown"),
                            "correct": correct,
                        }
                    )

    # Concatenate all predictions
    all_true_labels = np.vstack(all_true_labels)
    all_pred_labels = np.vstack(all_pred_labels)
    all_pred_probs = np.vstack(all_pred_probs)

    # Calculate per-label error statistics
    print("\n" + "=" * 80)
    print("Per-Label Error Statistics")
    print("=" * 80)

    error_stats = {}
    for label_idx, label_name in enumerate(label_names_list):
        true = all_true_labels[:, label_idx]
        pred = all_pred_labels[:, label_idx]
        prob = all_pred_probs[:, label_idx]

        # Confusion matrix components
        tp = np.sum((true == 1) & (pred == 1))
        fp = np.sum((true == 0) & (pred == 1))
        tn = np.sum((true == 0) & (pred == 0))
        fn = np.sum((true == 1) & (pred == 0))

        # Error rates
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        error_stats[label_name] = {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

        print(f"\n{label_name.replace('_', ' ').title()}:")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp} (Rate: {false_positive_rate:.3f})")
        print(f"  True Negatives: {tn}")
        print(f"  False Negatives: {fn} (Rate: {false_negative_rate:.3f})")
        print(f"  Precision: {error_stats[label_name]['precision']:.3f}")
        print(f"  Recall: {error_stats[label_name]['recall']:.3f}")

    # Visualize error analysis
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)

    # 1. Confusion matrices per label
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for label_idx, label_name in enumerate(label_names_list):
        ax = axes[label_idx]
        true = all_true_labels[:, label_idx]
        pred = all_pred_labels[:, label_idx]

        cm = confusion_matrix(true, pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Count"}
        )
        ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
        ax.set_ylabel("True", fontsize=11, fontweight="bold")
        ax.set_title(
            label_name.replace("_", " ").title(), fontsize=12, fontweight="bold"
        )
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])

    plt.suptitle(f"Confusion Matrices: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    confusion_path = (
        output_dir
        / f"error_analysis_confusion_matrices_{model_name.lower().replace(' ', '_')}.png"
    )
    save_figure(fig, str(confusion_path), dpi=300, bbox_inches="tight", overwrite=True)
    plt.close(fig)
    print(f"[OK] Saved confusion matrices to {confusion_path}")

    # 2. Sample predictions visualization (overview)
    visualize_sample_predictions(
        model_name=model_name,
        samples=all_samples,
        output_path=output_dir
        / f"error_analysis_samples_{model_name.lower().replace(' ', '_')}.png",
        n_samples_per_category=5,
    )

    # 3. Per-pathology detailed sample visualizations
    for label_idx, label_name in enumerate(label_names_list):
        # Get samples for this label
        label_samples = [s for s in all_samples if s["image_path"] is not None]

        # Separate by error type
        false_positives = [
            s
            for s in label_samples
            if s["true_labels"][label_idx] == 0 and s["pred_labels"][label_idx] == 1
        ]
        false_negatives = [
            s
            for s in label_samples
            if s["true_labels"][label_idx] == 1 and s["pred_labels"][label_idx] == 0
        ]
        true_positives = [
            s
            for s in label_samples
            if s["true_labels"][label_idx] == 1 and s["pred_labels"][label_idx] == 1
        ]
        true_negatives = [
            s
            for s in label_samples
            if s["true_labels"][label_idx] == 0 and s["pred_labels"][label_idx] == 0
        ]

        # Create visualization for this pathology
        n_samples = 4
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2.5, 5))

        # True Positives
        for i, sample in enumerate(true_positives[:n_samples]):
            ax = axes[0, i]
            image = load_dicom_image(sample["image_path"])
            if image is not None:
                ax.imshow(image, cmap="gray", aspect="auto")
            prob = sample["pred_probs"][label_idx]
            ax.set_title(
                f"TP\nP={prob:.2f}", fontsize=10, color="green", fontweight="bold"
            )
            ax.axis("off")

        # False Negatives
        for i, sample in enumerate(false_negatives[:n_samples]):
            if i < n_samples:
                ax = axes[1, i]
                image = load_dicom_image(sample["image_path"])
                if image is not None:
                    ax.imshow(image, cmap="gray", aspect="auto")
                prob = sample["pred_probs"][label_idx]
                ax.set_title(
                    f"FN\nP={prob:.2f}", fontsize=10, color="red", fontweight="bold"
                )
                ax.axis("off")

        # Fill remaining slots with False Positives if available
        if len(false_negatives) < n_samples and false_positives:
            for i in range(len(false_negatives), n_samples):
                if i - len(false_negatives) < len(false_positives):
                    sample = false_positives[i - len(false_negatives)]
                    ax = axes[1, i]
                    image = load_dicom_image(sample["image_path"])
                    if image is not None:
                        ax.imshow(image, cmap="gray", aspect="auto")
                    prob = sample["pred_probs"][label_idx]
                    ax.set_title(
                        f"FP\nP={prob:.2f}",
                        fontsize=10,
                        color="orange",
                        fontweight="bold",
                    )
                    ax.axis("off")

        plt.suptitle(
            f"{label_name.replace('_', ' ').title()}: True Positives (Top) vs False Negatives/Positives (Bottom)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        per_label_path = (
            output_dir
            / f"error_analysis_samples_{label_name}_{model_name.lower().replace(' ', '_')}.png"
        )
        save_figure(
            fig, str(per_label_path), dpi=300, bbox_inches="tight", overwrite=True
        )
        plt.close(fig)
        print(f"[OK] Saved per-label visualization to {per_label_path}")

    # 3. Error rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    labels_display = [ln.replace("_", " ").title() for ln in label_names_list]
    fp_rates = [error_stats[ln]["false_positive_rate"] for ln in label_names_list]
    fn_rates = [error_stats[ln]["false_negative_rate"] for ln in label_names_list]

    x = np.arange(len(labels_display))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        fp_rates,
        width,
        label="False Positive Rate",
        color="#e74c3c",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        fn_rates,
        width,
        label="False Negative Rate",
        color="#3498db",
        alpha=0.8,
    )

    ax.set_xlabel("Pathology Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Error Rates by Pathology Type: {model_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    error_rate_path = (
        output_dir
        / f"error_analysis_error_rates_{model_name.lower().replace(' ', '_')}.png"
    )
    save_figure(fig, str(error_rate_path), dpi=300, bbox_inches="tight", overwrite=True)
    plt.close(fig)
    print(f"[OK] Saved error rates to {error_rate_path}")

    return error_stats, all_samples


def main():
    """Main function to run error analysis."""
    print("=" * 80)
    print("Pathology Model Error Analysis")
    print("=" * 80)

    # Load best model
    runs_dir = project_root / "outputs" / "pathology_model" / "runs"
    model_dir = runs_dir / BEST_MODEL_RUN_ID

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = model_dir / "weights" / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]
    hyperparams = checkpoint.get("hyperparams", {})

    # Create model
    model = create_pathology_model(
        architecture=model_config["architecture"],
        num_labels=model_config["num_labels"],
        pretrained=model_config.get("pretrained", True),
        in_channels=model_config.get("in_channels", 1),
        dropout_rate=model_config.get("dropout_rate", 0.5),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded: {model_config['architecture']}")

    # Load thresholds
    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        thresholds, _ = load_thresholds(thresholds_path)
        print(f"[OK] Loaded optimal thresholds from {thresholds_path}")
    else:
        thresholds = {label: 0.5 for label in label_names_list}
        print("[WARNING] Using default threshold 0.5 for all labels")

    # Load test dataset (always use multi-sequence)
    print(f"\nLoading test dataset from: {MANIFEST_PATH}")
    test_transform = get_default_transforms(mode="test", input_size=(224, 224))
    test_dataset = MultiSequencePathologyDataset(
        manifest_path=MANIFEST_PATH,
        project_root=project_root,
        split="test",
        transform=test_transform,
        sequences=["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"],
        handle_missing="zero_pad",
        return_binary=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"[OK] Loaded {len(test_dataset)} test samples")

    # Perform error analysis
    model_name = "DenseNet-121"  # Best model
    error_stats, samples = analyze_errors(
        model=model,
        dataloader=test_loader,
        dataset=test_dataset,
        device=device,
        model_name=model_name,
        thresholds=thresholds,
        output_dir=OUTPUT_DIR,
    )

    # Save error statistics
    stats_path = (
        OUTPUT_DIR / f"error_analysis_stats_{model_name.lower().replace(' ', '_')}.json"
    )
    with open(stats_path, "w") as f:
        json.dump(error_stats, f, indent=2)
    print(f"[OK] Saved error statistics to {stats_path}")

    print("\n" + "=" * 80)
    print("Error Analysis Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Generated files:")
    print("  - Confusion matrices")
    print("  - Sample predictions visualization")
    print("  - Error rates comparison")
    print("  - Error statistics (JSON)")


if __name__ == "__main__":
    main()
