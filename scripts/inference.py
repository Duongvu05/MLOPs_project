#!/usr/bin/env python3
"""
Inference script for pathology classification models.

This script loads a trained model and performs inference on test samples.
It can process individual samples or batch inference on the entire test set.

Usage:
    # Inference on a single sample (by index)
    python scripts/inference.py --model_path outputs/pathology_model/runs/resnet18/weights/best_model.pth --sample_idx 0

    # Inference on multiple samples
    python scripts/inference.py --model_path outputs/pathology_model/runs/densenet121/weights/best_model.pth --sample_idx 5,10,15

    # Inference on all test samples
    python scripts/inference.py --model_path outputs/pathology_model/runs/efficientnet_b0/weights/best_model.pth --all_test

    # Inference with custom threshold
    python scripts/inference.py --model_path outputs/pathology_model/runs/resnet18/weights/best_model.pth --sample_idx 0 --threshold 0.5
"""

import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Union
import json

import torch
from torch.utils.data import DataLoader
from loguru import logger
import pandas as pd

warnings.filterwarnings("ignore")

# Configure loguru logger with colors
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.dataset import get_default_transforms

try:
    from utils.multi_sequence_dataset import MultiSequencePathologyDataset

    MULTI_SEQUENCE_DATASET_AVAILABLE = True
except ImportError:
    MULTI_SEQUENCE_DATASET_AVAILABLE = False
    MultiSequencePathologyDataset = None

from utils.label_utils import label_names
from utils.threshold_utils import load_thresholds, apply_thresholds
from training.pathology_training_utils import (
    create_model_with_verification,
    prepare_model_input,
    is_multi_sequence_model,
)


class PathologyInference:
    """
    Pathology classification inference class.

    Handles loading trained models and performing inference on test samples.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[torch.device] = None,
        threshold_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize inference class.

        Args:
            model_path: Path to trained model checkpoint
            device: PyTorch device (auto-detect if None)
            threshold_path: Optional path to thresholds.json file
        """
        self.model_path = Path(model_path)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.threshold_path = Path(threshold_path) if threshold_path else None

        # Initialize attributes
        self.model = None
        self.model_config = None
        self.hyperparams = None
        self.label_names_list = label_names()
        self.optimal_thresholds = None

        # Load model and thresholds
        self._load_model()
        self._load_thresholds()

        logger.success(f"Inference initialized with device: {self.device}")

    def _load_model(self):
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        logger.info(f"Loading model from: {self.model_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )

            # Extract configuration
            self.model_config = checkpoint["model_config"]
            self.hyperparams = checkpoint.get("hyperparams", {})

            # Log model info
            architecture = self.model_config.get("architecture", "unknown")
            backbone = checkpoint.get("backbone", "unknown")
            val_f1 = checkpoint.get("val_f1", "unknown")
            epoch = checkpoint.get("epoch", "unknown")

            logger.info(f"Model architecture: {architecture}")
            logger.info(f"Model backbone: {backbone}")
            logger.info(f"Validation F1: {val_f1}")
            logger.info(
                f"Trained epochs: {epoch + 1 if isinstance(epoch, int) else epoch}"
            )

            # Create model
            self.model = create_model_with_verification(
                self.model_config,
                self.device,
                hyperparams=self.hyperparams,
                enable_wandb_logging=False,
            )

            # Load state dict
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.success("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_thresholds(self):
        """Load optimal thresholds if available."""
        if self.threshold_path and self.threshold_path.exists():
            self.optimal_thresholds, _ = load_thresholds(self.threshold_path)
            logger.info(f"Loaded optimal thresholds from: {self.threshold_path}")
        elif self.model_path.parent.parent.exists():
            # Try to find thresholds.json in the same run directory
            run_dir = self.model_path.parent.parent
            threshold_file = run_dir / "thresholds.json"
            if threshold_file.exists():
                self.optimal_thresholds, _ = load_thresholds(threshold_file)
                logger.info(f"Found and loaded thresholds: {threshold_file}")
            else:
                logger.warning(
                    "No optimal thresholds found, using default threshold of 0.5"
                )
                self.optimal_thresholds = {
                    label: 0.5 for label in self.label_names_list
                }
        else:
            logger.warning(
                "No optimal thresholds found, using default threshold of 0.5"
            )
            self.optimal_thresholds = {label: 0.5 for label in self.label_names_list}

    def create_test_dataset(self) -> MultiSequencePathologyDataset:
        """Create test dataset."""
        if not MULTI_SEQUENCE_DATASET_AVAILABLE:
            raise ImportError("MultiSequencePathologyDataset not available")

        # Get data configuration
        data_config = self.hyperparams.get(
            "data", {"input_size": [224, 224], "normalization": "imagenet"}
        )

        # Create test transforms
        test_transforms = get_default_transforms(
            mode="val",  # Use val mode for inference (no augmentation)
            input_size=tuple(data_config["input_size"]),
            normalization=data_config["normalization"],
            in_channels=self.model_config.get("in_channels", 1),
        )

        # Create dataset
        manifest_path = Path(
            "../data/mri_phen/processed/pathology_training_manifest.csv"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

        multi_sequence_config = self.hyperparams.get("multi_sequence", {})
        sequences = multi_sequence_config.get(
            "sequences", ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]
        )
        handle_missing = multi_sequence_config.get("handle_missing", "zero_pad")

        test_dataset = MultiSequencePathologyDataset(
            manifest_path=manifest_path,
            project_root=project_root,
            split="test",
            transform=test_transforms,
            sequences=[s.upper() for s in sequences],
            handle_missing=handle_missing,
            return_binary=True,
        )

        return test_dataset

    def predict_sample(self, sample: Dict) -> Dict:
        """
        Predict on a single sample.

        Args:
            sample: Sample dictionary from dataset

        Returns:
            Dictionary with predictions and metadata
        """
        # Prepare model input
        is_multi_seq = is_multi_sequence_model(self.model)

        # Create a batch of size 1
        batch = {}
        for key, value in sample.items():
            if key == "sequences":
                # Handle sequences dictionary
                sequences_batch = {}
                for seq_name, seq_tensor in value.items():
                    if seq_tensor is not None:
                        sequences_batch[seq_name] = seq_tensor.unsqueeze(
                            0
                        )  # Add batch dimension
                    else:
                        sequences_batch[seq_name] = None
                batch[key] = sequences_batch
            elif key == "sequence_available":
                # Convert boolean availability to tensor format expected by prepare_model_input
                availability_batch = {}
                for seq_name, available in value.items():
                    availability_batch[seq_name] = torch.tensor(
                        [available], dtype=torch.bool
                    )
                batch[key] = availability_batch
            elif isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)  # Add batch dimension
            else:
                batch[key] = [value] if not isinstance(value, list) else value

        # Prepare input for model
        model_input, labels = prepare_model_input(
            batch, self.device, is_multi_seq, self.model
        )

        # Make prediction
        with torch.no_grad():
            if is_multi_seq:
                sequences_dict, sequence_available_dict, ivd_levels = model_input
                outputs = self.model(
                    sequences_dict, sequence_available_dict, ivd_levels
                )
            else:
                outputs = self.model(model_input)

        # Convert logits to probabilities
        probabilities = (
            torch.sigmoid(outputs).cpu().numpy()[0]
        )  # Remove batch dimension

        # Apply thresholds to get predictions (manually for single sample)
        predictions = []
        for i, label_name in enumerate(self.label_names_list):
            threshold = self.optimal_thresholds.get(label_name, 0.5)
            predictions.append(probabilities[i] >= threshold)

        # Get ground truth labels
        ground_truth = labels.cpu().numpy()[0]  # Remove batch dimension

        return {
            "patient_id": sample["patient_id"],
            "ivd_label": sample["ivd_label"],
            "ground_truth": {
                label: bool(gt)
                for label, gt in zip(self.label_names_list, ground_truth)
            },
            "probabilities": {
                label: float(prob)
                for label, prob in zip(self.label_names_list, probabilities)
            },
            "predictions": {
                label: bool(pred)
                for label, pred in zip(self.label_names_list, predictions)
            },
            "thresholds": self.optimal_thresholds,
            "sequence_available": {
                k: bool(v) for k, v in sample["sequence_available"].items()
            },  # Convert to bool
        }

    def predict_samples(self, sample_indices: List[int]) -> List[Dict]:
        """
        Predict on multiple samples.

        Args:
            sample_indices: List of sample indices

        Returns:
            List of prediction dictionaries
        """
        test_dataset = self.create_test_dataset()

        results = []
        for idx in sample_indices:
            if idx >= len(test_dataset):
                logger.warning(
                    f"Sample index {idx} out of range (max: {len(test_dataset) - 1})"
                )
                continue

            sample = test_dataset[idx]
            result = self.predict_sample(sample)
            results.append(result)

            logger.info(
                f"Sample {idx}: Patient {result['patient_id']}, IVD {result['ivd_label']}"
            )

        return results

    def predict_all_test(self) -> List[Dict]:
        """
        Predict on all test samples.

        Returns:
            List of prediction dictionaries
        """
        test_dataset = self.create_test_dataset()

        logger.info(f"Running inference on {len(test_dataset)} test samples")

        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        results = []
        is_multi_seq = is_multi_sequence_model(self.model)

        with torch.no_grad():
            for batch in test_loader:
                # Prepare input for model
                model_input, labels = prepare_model_input(
                    batch, self.device, is_multi_seq, self.model
                )

                # Make prediction
                if is_multi_seq:
                    sequences_dict, sequence_available_dict, ivd_levels = model_input
                    outputs = self.model(
                        sequences_dict, sequence_available_dict, ivd_levels
                    )
                else:
                    outputs = self.model(model_input)

                # Convert logits to probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                ground_truth = labels.cpu().numpy()

                # Apply thresholds using the utility function (expects 2D array)
                y_pred = apply_thresholds(
                    probabilities, [self.optimal_thresholds], self.label_names_list
                )

                # Process each sample in batch
                for i in range(len(probabilities)):
                    predictions = y_pred[i]

                    result = {
                        "patient_id": batch["patient_id"][i],
                        "ivd_label": batch["ivd_label"][i],
                        "ground_truth": {
                            label: bool(gt)
                            for label, gt in zip(self.label_names_list, ground_truth[i])
                        },
                        "probabilities": {
                            label: float(prob)
                            for label, prob in zip(
                                self.label_names_list, probabilities[i]
                            )
                        },
                        "predictions": {
                            label: bool(pred)
                            for label, pred in zip(self.label_names_list, predictions)
                        },
                        "thresholds": self.optimal_thresholds,
                        "sequence_available": {
                            k: bool(v[i].item())
                            if isinstance(v[i], torch.Tensor)
                            else bool(v[i] if isinstance(v, list) else v)
                            for k, v in batch["sequence_available"].items()
                        },
                    }
                    results.append(result)

        logger.success(f"Completed inference on {len(results)} test samples")
        return results


def print_prediction_result(result: Dict):
    """Print formatted prediction result."""
    print(f"\n{'=' * 80}")
    print("INFERENCE RESULT")
    print(f"{'=' * 80}")
    print(f"Patient ID: {result['patient_id']}")
    print(f"IVD Level: {result['ivd_label']}")
    print("\nSequence Availability:")
    for seq_name, available in result["sequence_available"].items():
        # Convert tensor to boolean if needed
        if isinstance(available, torch.Tensor):
            available = available.item()
        print(f"  {seq_name.upper()}: {'✓' if available else '✗'}")

    print("\nPathology Predictions:")
    print(f"{'Pathology':<25} {'GT':<5} {'Pred':<5} {'Prob':<8} {'Thresh':<8}")
    print(f"{'-' * 25} {'-' * 5} {'-' * 5} {'-' * 8} {'-' * 8}")

    for label in result["ground_truth"].keys():
        gt = "✓" if result["ground_truth"][label] else "✗"
        pred = "✓" if result["predictions"][label] else "✗"
        prob = f"{result['probabilities'][label]:.3f}"
        thresh = f"{result['thresholds'][label]:.3f}"
        print(f"{label:<25} {gt:<5} {pred:<5} {prob:<8} {thresh:<8}")

    print(f"{'=' * 80}")


def save_results(results: List[Dict], output_path: Union[str, Path]):
    """Save prediction results to JSON or CSV file based on extension."""
    output_path = Path(output_path)

    if output_path.suffix.lower() == ".json":
        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.success(f"Results saved to JSON: {output_path}")
    else:
        # Save as CSV (flatten results for DataFrame)
        rows = []
        for result in results:
            row = {"patient_id": result["patient_id"], "ivd_label": result["ivd_label"]}

            # Add sequence availability
            for seq_name, available in result["sequence_available"].items():
                # Convert tensor to boolean if needed
                if isinstance(available, torch.Tensor):
                    available = available.item()
                row[f"seq_{seq_name.lower()}_available"] = available

            # Add ground truth
            for label, gt in result["ground_truth"].items():
                row[f"gt_{label}"] = gt

            # Add predictions
            for label, pred in result["predictions"].items():
                row[f"pred_{label}"] = pred

            # Add probabilities
            for label, prob in result["probabilities"].items():
                row[f"prob_{label}"] = prob

            # Add thresholds
            for label, thresh in result["thresholds"].items():
                row[f"thresh_{label}"] = thresh

            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.success(f"Results saved to CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for pathology classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference on a single sample
  python scripts/inference.py --backbone resnet18 --sample_idx 0
  
  # Inference on multiple samples
  python scripts/inference.py --backbone densenet121 --sample_idx 5,10,15
  
  # Inference on all test samples
  python scripts/inference.py --backbone efficientnet_b0 --all_test
  
  # Save results to file
  python scripts/inference.py --backbone efficientnet_b0 --all_test --output test_results.json
  python scripts/inference.py --backbone resnet50 --all_test --output test_results.json
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pth file)",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone name (e.g., resnet18, densenet121, efficientnet_b0) - will auto-find best_model.pth",
    )

    parser.add_argument(
        "--sample_idx",
        type=str,
        default=None,
        help="Sample index(es) to predict on (single: 0, multiple: 0,5,10)",
    )

    parser.add_argument(
        "--all_test", action="store_true", help="Run inference on all test samples"
    )

    parser.add_argument(
        "--threshold_path",
        type=str,
        default=None,
        help="Path to thresholds.json file (auto-detect if not provided)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (.json or .csv format based on extension)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model_path and not args.backbone:
        parser.print_help()
        logger.error("Must specify either --model_path or --backbone")
        return

    if args.model_path and args.backbone:
        logger.error("Cannot specify both --model_path and --backbone. Use either one.")
        return

    if not args.sample_idx and not args.all_test:
        parser.print_help()
        logger.error("Must specify either --sample_idx or --all_test")
        return

    if args.sample_idx and args.all_test:
        logger.error("Cannot specify both --sample_idx and --all_test")
        return

    # Determine model path
    if args.backbone:
        # Auto-find model path from backbone name
        model_path = (
            project_root
            / "outputs"
            / "pathology_model"
            / "runs"
            / args.backbone
            / "weights"
            / "best_model.pth"
        )
        if not model_path.exists():
            # Try final_model.pth as fallback
            model_path = (
                project_root
                / "outputs"
                / "pathology_model"
                / "runs"
                / args.backbone
                / "weights"
                / "final_model.pth"
            )
            if not model_path.exists():
                logger.error(f"No trained model found for backbone '{args.backbone}'")
                logger.error(
                    f"Expected: {project_root / 'outputs' / 'pathology_model' / 'runs' / args.backbone / 'weights' / 'best_model.pth'}"
                )
                return
            else:
                logger.info(f"Using final_model.pth for backbone: {args.backbone}")
        else:
            logger.info(f"Using best_model.pth for backbone: {args.backbone}")
    else:
        # Use provided model path
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model checkpoint not found: {model_path}")
            return

    logger.success(f"Project root: {project_root}")

    try:
        # Initialize inference
        inference = PathologyInference(
            model_path=model_path, threshold_path=args.threshold_path
        )

        # Run inference
        if args.all_test:
            results = inference.predict_all_test()

            if results:
                logger.info(f"Processed {len(results)} samples")
                # Show first few results
                for i, result in enumerate(results[:3]):
                    print(f"\n--- Sample {i} ---")
                    print_prediction_result(result)

                if len(results) > 3:
                    logger.info(f"... and {len(results) - 3} more samples")

        else:
            # Parse sample indices
            try:
                if "," in args.sample_idx:
                    sample_indices = [
                        int(x.strip()) for x in args.sample_idx.split(",")
                    ]
                else:
                    sample_indices = [int(args.sample_idx)]
            except ValueError:
                logger.error(
                    "Invalid sample index format. Use single number or comma-separated list."
                )
                return

            results = inference.predict_samples(sample_indices)

            # Print results
            for result in results:
                print_prediction_result(result)

        # Save results if output path provided
        if args.output and results:
            save_results(results, args.output)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
