#!/usr/bin/env python3
"""
Evaluation script for Pfirrmann grade classification models.

This script can evaluate a single model or compare all trained models.

Usage:
    # Evaluate all trained models
    python scripts/evaluate_pfirrman_model.py --all

    # Evaluate a specific model
    python scripts/evaluate_pfirrman_model.py --run_dir outputs/pfirrman_model/runs/20251203_185414

    # Evaluate with specific checkpoint
    python scripts/evaluate_pfirrman_model.py --run_dir outputs/pfirrman_model/runs/20251203_185414 --checkpoint final_model.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict
import warnings
import traceback

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.dataset import get_default_transforms

try:
    from utils.pfirrman_dataset import MultiSequencePfirrmannDataset

    PFIRRMAN_DATASET_AVAILABLE = True
except ImportError:
    PFIRRMAN_DATASET_AVAILABLE = False
    MultiSequencePfirrmannDataset = None

from models.pfirrman_model import create_pfirrman_model
from training.pfirrman_training_utils import (
    calculate_pfirrman_metrics,
    generate_pfirrman_predictions,
)


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """
    Convert DataFrame to markdown table format.
    Falls back to simple text format if tabulate is not available.
    """
    try:
        return df.to_markdown(index=index)
    except ImportError:
        # Fallback: create a simple markdown table manually
        lines = []
        # Header
        headers = list(df.columns)
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        # Rows
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row) + " |")
        return "\n".join(lines)


def format_model_name(architecture: str, hyperparams: Optional[Dict] = None) -> str:
    """Convert architecture name to readable format."""
    if architecture.startswith("resnet"):
        num = architecture.replace("resnet", "")
        return f"ResNet-{num}"
    elif architecture.startswith("efficientnet"):
        variant = architecture.replace("efficientnet_", "").upper()
        return f"EfficientNet-{variant}"
    elif architecture.startswith("densenet"):
        num = architecture.replace("densenet", "")
        return f"DenseNet-{num}"
    elif architecture.startswith("vit"):
        if "base" in architecture:
            return "ViT-Base"
        elif "large" in architecture:
            return "ViT-Large"
        else:
            return "ViT"
    else:
        return architecture.replace("_", "-").title()


def evaluate_single_model(
    run_dir: Path, checkpoint_name: str = "best_model.pth", device: torch.device = None
) -> dict:
    """Evaluate a single Pfirrmann model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_dir = run_dir / "weights"
    checkpoint_path = weights_dir / checkpoint_name

    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return None

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint["model_config"]
        hyperparams = checkpoint.get("hyperparams", {})
        architecture = model_config["architecture"]
        display_name = format_model_name(architecture, hyperparams)

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {display_name} ({architecture})")
        print(f"Run: {run_dir.name}")
        print(f"{'=' * 80}")

        # Get model kwargs from checkpoint
        multi_seq_config = hyperparams.get("multi_sequence", {})
        model_kwargs = {
            "sequences": [
                s.lower()
                for s in multi_seq_config.get(
                    "sequences", ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]
                )
            ],
            # Note: num_classes and pretrained are passed explicitly below, not in kwargs
            "in_channels": model_config.get("in_channels", 1),
            "feature_dims": multi_seq_config.get("feature_dims", [256, 512, 1024]),
            "fusion_dim": multi_seq_config.get("fusion_dim", 512),
            "attention_heads": multi_seq_config.get("attention", {}).get(
                "num_heads", 8
            ),
            "attention_dropout": multi_seq_config.get("attention", {}).get(
                "dropout", 0.1
            ),
            "head_hidden_dim": multi_seq_config.get("heads", {}).get("hidden_dim", 256),
            "head_dropout": multi_seq_config.get("heads", {}).get("dropout", 0.3),
        }

        # Create model
        model = create_pfirrman_model(
            architecture=model_config["architecture"],
            num_classes=model_config.get("num_classes", 5),
            pretrained=model_config.get("pretrained", True),
            **model_kwargs,
        )

        # Load state dict
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as e:
            print(f"⚠ Skipping {run_dir.name}: Checkpoint architecture mismatch")
            print(f"  Error: {str(e)[:200]}...")
            return None

        model = model.to(device)
        model.eval()

        # Load data config
        data_config = hyperparams.get(
            "data", {"input_size": [224, 224], "normalization": "imagenet"}
        )

        # Create test dataset
        manifest_path = Path(
            "../data/mri_phen/processed/pfirrman_training_manifest.csv"
        )
        test_transforms = get_default_transforms(
            mode="val",
            input_size=tuple(data_config["input_size"]),
            normalization=data_config["normalization"],
            in_channels=model_config.get("in_channels", 1),
        )

        if not PFIRRMAN_DATASET_AVAILABLE:
            raise ImportError(
                "Pfirrmann multi-sequence dataset is required but not available."
            )

        sequences = multi_seq_config.get(
            "sequences", ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]
        )
        handle_missing = multi_seq_config.get("handle_missing", "zero_pad")

        test_dataset = MultiSequencePfirrmannDataset(
            manifest_path=manifest_path,
            project_root=project_root,
            split="test",
            transform=test_transforms,
            sequences=[s.upper() for s in sequences],
            handle_missing=handle_missing,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        # Generate predictions
        y_true, y_pred, y_proba = generate_pfirrman_predictions(
            model, test_loader, device, "test", return_dataframe=False
        )

        # Calculate metrics
        metrics = calculate_pfirrman_metrics(y_true, y_pred, y_proba)

        result = {
            "run_id": run_dir.name,
            "architecture": architecture,
            "display_name": display_name,
            "val_f1": checkpoint.get("val_f1", 0),
            "epochs": checkpoint.get("epoch", 0) + 1,
            "overall_metrics": {
                "accuracy": metrics["overall/accuracy"],
                "macro_precision": metrics["overall/macro_precision"],
                "macro_recall": metrics["overall/macro_recall"],
                "macro_f1": metrics["overall/macro_f1"],
                "weighted_precision": metrics["overall/weighted_precision"],
                "weighted_recall": metrics["overall/weighted_recall"],
                "weighted_f1": metrics["overall/weighted_f1"],
                "macro_auroc": metrics.get("overall/macro_auroc"),
                "macro_auprc": metrics.get("overall/macro_auprc"),
                "multiclass_auroc": metrics.get("overall/multiclass_auroc"),
            },
            "per_class_metrics": {},
        }

        # Extract per-class metrics
        for grade in range(1, 6):
            class_name = f"Grade_{grade}"
            result["per_class_metrics"][class_name] = {
                "precision": metrics[f"{class_name}/precision"],
                "recall": metrics[f"{class_name}/recall"],
                "f1": metrics[f"{class_name}/f1"],
                "auroc": metrics.get(f"{class_name}/auroc"),
                "auprc": metrics.get(f"{class_name}/auprc"),
            }

        # Add confusion matrix
        result["confusion_matrix"] = metrics["confusion_matrix"]

        # Format AUROC/AUPRC for display
        macro_auroc_str = (
            f"{metrics.get('overall/macro_auroc', 0):.4f}"
            if metrics.get("overall/macro_auroc") is not None
            else "N/A"
        )
        macro_auprc_str = (
            f"{metrics.get('overall/macro_auprc', 0):.4f}"
            if metrics.get("overall/macro_auprc") is not None
            else "N/A"
        )

        print(
            f"\n✓ {display_name}: Accuracy: {metrics['overall/accuracy']:.4f} | "
            f"Weighted F1: {metrics['overall/weighted_f1']:.4f} | "
            f"Macro F1: {metrics['overall/macro_f1']:.4f} | "
            f"Macro AUROC: {macro_auroc_str} | "
            f"Macro AUPRC: {macro_auprc_str} | "
            f"Val F1: {checkpoint.get('val_f1', 0):.4f}"
        )

        return result

    except Exception as e:
        print(f"⚠ Error evaluating {run_dir.name}: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pfirrmann grade classification model(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all trained models
  python scripts/evaluate_pfirrman_model.py --all
  
  # Evaluate specific model
  python scripts/evaluate_pfirrman_model.py --run_dir outputs/pfirrman_model/runs/20251203_185414
  
  # Evaluate with final checkpoint
  python scripts/evaluate_pfirrman_model.py --run_dir outputs/pfirrman_model/runs/20251203_185414 --checkpoint final_model.pth
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained models in outputs/pfirrman_model/runs/",
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Specific run directory to evaluate (e.g., outputs/pfirrman_model/runs/20251203_185414)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pth",
        choices=["best_model.pth", "final_model.pth"],
        help="Checkpoint to use (default: best_model.pth)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: outputs/pfirrman_model/evaluation_results/)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    print(f"✓ Project root: {project_root}")

    if args.all:
        # Evaluate all models
        runs_dir = project_root / "outputs" / "pfirrman_model" / "runs"

        if not runs_dir.exists():
            print("⚠ Runs directory does not exist")
            return

        run_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        all_results = []

        print(f"\n{'=' * 80}")
        print("EVALUATING ALL AVAILABLE MODELS")
        print(f"{'=' * 80}")

        for run_dir in run_dirs:
            # Try to find checkpoint
            best_model_path = run_dir / "weights" / "best_model.pth"
            final_model_path = run_dir / "weights" / "final_model.pth"

            checkpoint_path = None
            checkpoint_name = None
            if best_model_path.exists():
                checkpoint_path = best_model_path
                checkpoint_name = "best_model.pth"
            elif final_model_path.exists():
                checkpoint_path = final_model_path
                checkpoint_name = "final_model.pth"

            if checkpoint_path is None:
                print(f"  ⚠ Skipping {run_dir.name}: No model checkpoint found")
                continue

            # Quick preview
            try:
                checkpoint_preview = torch.load(checkpoint_path, map_location="cpu")
                preview_arch = checkpoint_preview.get("model_config", {}).get(
                    "architecture", "unknown"
                )
                print(
                    f"  Found checkpoint in {run_dir.name}: architecture = {preview_arch}"
                )
            except Exception as e:
                print(f"  ⚠ Skipping {run_dir.name}: Cannot load checkpoint - {e}")
                continue

            # Evaluate
            result = evaluate_single_model(run_dir, checkpoint_name, device)
            if result:
                all_results.append(result)

        print(f"\n{'=' * 80}")
        print(f"Evaluated {len(all_results)} models")
        print(f"{'=' * 80}")

        if len(all_results) > 0:
            # Create comparison DataFrame
            comparison_data = []
            for result in all_results:
                overall = result["overall_metrics"]
                comparison_data.append(
                    {
                        "Architecture": result["display_name"],
                        "Architecture (Raw)": result["architecture"],
                        "Run ID": result["run_id"],
                        "Val F1": result["val_f1"],
                        "Epochs": result["epochs"],
                        "Accuracy": overall["accuracy"],
                        "Macro Precision": overall["macro_precision"],
                        "Macro Recall": overall["macro_recall"],
                        "Macro F1": overall["macro_f1"],
                        "Weighted Precision": overall["weighted_precision"],
                        "Weighted Recall": overall["weighted_recall"],
                        "Weighted F1": overall["weighted_f1"],
                        "Macro AUROC": overall.get("macro_auroc"),
                        "Macro AUPRC": overall.get("macro_auprc"),
                        "Multiclass AUROC": overall.get("multiclass_auroc"),
                    }
                )

            comparison_df = pd.DataFrame(comparison_data)
            # Replace None with NaN for proper CSV handling
            comparison_df = comparison_df.fillna("N/A")
            comparison_df = comparison_df.sort_values("Weighted F1", ascending=False)

            # Save results
            output_dir = (
                project_root / "outputs" / "pfirrman_model" / "evaluation_results"
            )
            if args.output_dir:
                output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save full results
            results_path = output_dir / "pfirrman_evaluation_results.csv"
            comparison_df.to_csv(results_path, index=False)
            print(f"\n✓ Saved full results: {results_path}")

            # Save summary (best per architecture)
            summary_data = []
            for arch in comparison_df["Architecture (Raw)"].unique():
                arch_df = comparison_df[comparison_df["Architecture (Raw)"] == arch]
                best = arch_df.iloc[0]
                summary_data.append(best.to_dict())

            summary_df = pd.DataFrame(summary_data)
            summary_path = output_dir / "pfirrman_evaluation_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"✓ Saved summary: {summary_path}")

            # Generate markdown report
            report_lines = [
                "# Pfirrmann Model Evaluation Report",
                "",
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Total models evaluated: {len(all_results)}",
                "",
                "## Best Models by Architecture",
                "",
                dataframe_to_markdown(summary_df, index=False),
                "",
                "## Overall Performance Comparison",
                "",
                dataframe_to_markdown(comparison_df, index=False),
                "",
                "## Per-Class Performance (Best Model)",
                "",
            ]

            # Add per-class metrics for best overall model
            best_result = max(
                all_results, key=lambda x: x["overall_metrics"]["weighted_f1"]
            )
            report_lines.append(
                f"**Best Model: {best_result['display_name']} (Weighted F1: {best_result['overall_metrics']['weighted_f1']:.4f})**"
            )
            report_lines.append("")

            per_class_data = []
            for class_name, metrics in best_result["per_class_metrics"].items():
                per_class_data.append(
                    {
                        "Class": class_name,
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1": metrics["f1"],
                        "AUROC": metrics.get("auroc", "N/A"),
                        "AUPRC": metrics.get("auprc", "N/A"),
                    }
                )

            per_class_df = pd.DataFrame(per_class_data)
            report_lines.append(dataframe_to_markdown(per_class_df, index=False))

            report_path = output_dir / "pfirrman_evaluation_report.md"
            with open(report_path, "w") as f:
                f.write("\n".join(report_lines))
            print(f"✓ Saved report: {report_path}")

            # Print summary
            print(f"\n{'=' * 80}")
            print("SUMMARY")
            print(f"{'=' * 80}")
            print(summary_df.to_string(index=False))
            print(f"\n{'=' * 80}")

    else:
        # Evaluate single model
        if args.run_dir is None:
            print("Error: Must specify --run_dir or --all")
            return

        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            return

        result = evaluate_single_model(run_dir, args.checkpoint, device)

        if result:
            print(f"\n{'=' * 80}")
            print("EVALUATION RESULTS")
            print(f"{'=' * 80}")
            print(f"Architecture: {result['display_name']}")
            print(f"Accuracy: {result['overall_metrics']['accuracy']:.4f}")
            print(f"Weighted F1: {result['overall_metrics']['weighted_f1']:.4f}")
            print(f"Macro F1: {result['overall_metrics']['macro_f1']:.4f}")
            print("\nPer-Class Metrics:")
            for class_name, metrics in result["per_class_metrics"].items():
                print(
                    f"  {class_name}: Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
                )
            print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
