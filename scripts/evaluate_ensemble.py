#!/usr/bin/env python3
"""
Standalone script to evaluate ensemble models with TTA.

Usage:
    python scripts/evaluate_ensemble.py --checkpoints checkpoint1.pth checkpoint2.pth checkpoint3.pth
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score,
    average_precision_score,
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.ensemble_model import EnsembleModel
from utils.dataset import get_default_transforms
from utils.multi_sequence_dataset import MultiSequencePathologyDataset
from utils.label_utils import label_names
from utils.threshold_utils import find_optimal_thresholds, apply_thresholds
from utils.tta_utils import get_tta_transforms, tta_predict
from training.pathology_training_utils import (
    prepare_model_input,
)
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble model with TTA")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to model checkpoints to ensemble",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--tta", action="store_true", help="Enable test-time augmentation"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=8,
        help="Number of TTA augmentations (default: 8)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: outputs/pathology_model/ensemble/)",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoints and create model configs
    model_configs = []
    model_checkpoints = []

    for checkpoint_path in args.checkpoints:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_configs.append(
            {
                "model_config": checkpoint.get("model_config", {}),
                "hyperparams": checkpoint.get("hyperparams", {}),
            }
        )
        model_checkpoints.append(checkpoint_path)

    # Create ensemble
    print(f"\nCreating ensemble with {len(model_configs)} models...")
    ensemble_model = EnsembleModel(
        model_configs=model_configs, model_checkpoints=model_checkpoints, device=device
    )

    # Load dataset
    data_dir = Path("../data/mri_phen/processed")
    manifest_path = data_dir / "pathology_training_manifest.csv"

    # BREAKING CHANGE: All models now use multi-sequence
    # Always use MultiSequencePathologyDataset
    test_transforms = get_default_transforms(
        mode="val", input_size=(224, 224), normalization="imagenet", in_channels=1
    )

    sequences = ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]
    dataset = MultiSequencePathologyDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        split=args.split,
        transform=test_transforms,
        sequences=sequences,
        handle_missing="zero_pad",
        return_binary=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Loaded {args.split} dataset: {len(dataset)} samples")

    # Generate predictions
    label_names_list = label_names()

    if args.tta:
        print(
            f"\nGenerating predictions with TTA ({args.num_augmentations} augmentations)..."
        )
        tta_transforms = get_tta_transforms(
            input_size=(224, 224),
            num_augmentations=args.num_augmentations,
            normalization="imagenet",
            in_channels=1,
        )

        true_labels, pred_probs = tta_predict(
            model=ensemble_model,
            dataloader=dataloader,
            device=device,
            tta_transforms=tta_transforms,
            num_augmentations=args.num_augmentations,
            input_size=(224, 224),
            normalization="imagenet",
            in_channels=1,
        )
    else:
        print("\nGenerating predictions (no TTA)...")
        true_labels = []
        pred_probs = []

        ensemble_model.eval()
        with torch.no_grad():
            from tqdm import tqdm

            for batch in tqdm(dataloader, desc="Predicting"):
                model_input, labels = prepare_model_input(
                    batch, device, is_multi_seq, model=ensemble_model
                )

                if is_multi_seq:
                    sequences, sequence_available = model_input
                    logits = ensemble_model(sequences, sequence_available)
                else:
                    logits = ensemble_model(model_input)

                probs = torch.sigmoid(logits).cpu().numpy()
                true_labels.append(labels.cpu().numpy())
                pred_probs.append(probs)

        true_labels = np.vstack(true_labels)
        pred_probs = np.vstack(pred_probs)

    # Optimize thresholds
    print("\nOptimizing thresholds...")
    optimal_thresholds = find_optimal_thresholds(
        y_true=true_labels,
        y_proba=pred_probs,
        label_names=label_names_list,
        metric="f1",
    )

    pred_labels = apply_thresholds(pred_probs, optimal_thresholds, label_names_list)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL PERFORMANCE")
    print("=" * 80)

    metrics_dict = {}
    for i, label_name in enumerate(label_names_list):
        y_true = true_labels[:, i]
        y_pred = pred_labels[:, i]
        y_proba = pred_probs[:, i]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            roc_auc = (
                roc_auc_score(y_true, y_proba)
                if y_true.sum() > 0 and y_true.sum() < len(y_true)
                else 0.0
            )
        except:
            roc_auc = 0.0

        try:
            auprc = (
                average_precision_score(y_true, y_proba) if y_true.sum() > 0 else 0.0
            )
        except:
            auprc = 0.0

        metrics_dict[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "auprc": auprc,
        }

        print(f"\n{label_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  AUPRC: {auprc:.4f}")

    # Overall metrics
    macro_f1 = np.mean([m["f1"] for m in metrics_dict.values()])
    macro_precision = np.mean([m["precision"] for m in metrics_dict.values()])
    macro_recall = np.mean([m["recall"] for m in metrics_dict.values()])
    macro_roc_auc = np.mean([m["roc_auc"] for m in metrics_dict.values()])
    macro_auprc = np.mean([m["auprc"] for m in metrics_dict.values()])
    subset_acc = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_labels, pred_labels)

    print("\nOverall Metrics:")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Macro Recall: {macro_recall:.4f}")
    print(f"  Macro ROC-AUC: {macro_roc_auc:.4f}")
    print(f"  Macro AUPRC: {macro_auprc:.4f}")
    print(f"  Subset Accuracy: {subset_acc:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")

    print("\n" + "=" * 80)

    # Save results if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        import pandas as pd

        predictions_df = pd.DataFrame(
            {
                "patient_id": [
                    dataset[i].get("patient_id", f"patient_{i}")
                    for i in range(len(dataset))
                ],
                "ivd_label": [
                    dataset[i].get("ivd_label", "") for i in range(len(dataset))
                ],
            }
        )

        for i, label_name in enumerate(label_names_list):
            predictions_df[f"true_{label_name}"] = true_labels[:, i]
            predictions_df[f"pred_prob_{label_name}"] = pred_probs[:, i]
            predictions_df[f"pred_{label_name}"] = pred_labels[:, i]

        predictions_df.to_csv(output_dir / "ensemble_predictions.csv", index=False)
        print(f"\n✓ Predictions saved to: {output_dir / 'ensemble_predictions.csv'}")


if __name__ == "__main__":
    main()
