#!/usr/bin/env python3
"""
Training script for Pfirrmann grade classification models.

This script supports training all model architectures with multi-sequence fusion:
- ResNet: resnet18, resnet34, resnet50
- EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2
- DenseNet: densenet121, densenet169, densenet201
- Vision Transformer: vit_base_patch16_224, vit_large_patch16_224

Usage:
    python scripts/train_pfirrman_model.py --architecture resnet18
    python scripts/train_pfirrman_model.py --architecture densenet121 --config configs/wandb_config.yaml
    python scripts/train_pfirrman_model.py --architecture vit_base_patch16_224
"""

import argparse
import sys
from pathlib import Path
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.wandb_utils import log_metrics, log_model_checkpoint, finish_run
from utils.loss_utils import create_loss_function
from training.pfirrman_training_utils import (
    load_config_and_override_architecture,
    initialize_wandb_for_training,
    create_pfirrman_datasets,
    create_pfirrman_model_with_verification,
    train_epoch_pfirrman,
    validate_epoch_pfirrman,
    generate_pfirrman_predictions,
    compute_class_weights,
)


def get_architecture_family(architecture: str) -> str:
    """Get architecture family name from architecture string."""
    if architecture.startswith("resnet"):
        return "resnet"
    elif architecture.startswith("efficientnet"):
        return "efficientnet"
    elif architecture.startswith("densenet"):
        return "densenet"
    elif architecture.startswith("vit"):
        return "vit"
    else:
        return "unknown"


def get_architecture_options(architecture_family: str) -> list:
    """Get list of valid architecture options for a family."""
    options = {
        "resnet": ["resnet18", "resnet34", "resnet50"],
        "efficientnet": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
        "densenet": ["densenet121", "densenet169", "densenet201"],
        "vit": ["vit_base_patch16_224", "vit_large_patch16_224"],
    }
    return options.get(architecture_family, [architecture_family])


def main():
    parser = argparse.ArgumentParser(
        description="Train Pfirrmann grade classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ResNet-18
  python scripts/train_pfirrman_model.py --architecture resnet18
  
  # Train DenseNet-121
  python scripts/train_pfirrman_model.py --architecture densenet121
  
  # Train ViT-Base
  python scripts/train_pfirrman_model.py --architecture vit_base_patch16_224
  
  # Train with custom config
  python scripts/train_pfirrman_model.py --architecture efficientnet_b0 --config configs/wandb_config.yaml

Supported architectures:
  ResNet: resnet18, resnet34, resnet50
  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2
  DenseNet: densenet121, densenet169, densenet201
  Vision Transformer: vit_base_patch16_224, vit_large_patch16_224
        """,
    )

    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Model architecture (e.g., resnet18, densenet121, vit_base_patch16_224)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/wandb_config.yaml",
        help="Path to configuration file (default: configs/wandb_config.yaml)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints (default: outputs/pfirrman_model/runs/)",
    )

    args = parser.parse_args()

    # Validate architecture
    architecture_family = get_architecture_family(args.architecture)
    if architecture_family == "unknown":
        print(f"Error: Unknown architecture '{args.architecture}'")
        print("Supported architectures:")
        print("  ResNet: resnet18, resnet34, resnet50")
        print("  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2")
        print("  DenseNet: densenet121, densenet169, densenet201")
        print("  Vision Transformer: vit_base_patch16_224, vit_large_patch16_224")
        sys.exit(1)

    # Check for timm dependency if using ViT
    if architecture_family == "vit":
        try:
            import timm
        except ImportError:
            print("Error: ViT models require the 'timm' library.")
            print("Install it with: pip install timm")
            print("\nAlternatively, use a different architecture:")
            print("  - ResNet: resnet18, resnet34, resnet50")
            print("  - DenseNet: densenet121, densenet169, densenet201")
            print("  - EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2")
            sys.exit(1)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    print(f"✓ Project root: {project_root}")

    # Load configuration and override architecture
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    architecture_options = get_architecture_options(architecture_family)
    experiment_prefix = f"pfirrman_multiclass_{architecture_family}"

    config, hyperparams, model_config, experiment_name, timestamp = (
        load_config_and_override_architecture(
            config_path=config_path,
            target_architecture=architecture_family,
            architecture_options=architecture_options,
            experiment_prefix=experiment_prefix,
        )
    )

    # Override with user-specified architecture
    model_config["architecture"] = args.architecture

    # Extract configuration sections (use Pfirrmann-specific if available)
    if "pfirrman" in hyperparams:
        pfirrman_config = hyperparams["pfirrman"]
        training_config = pfirrman_config.get(
            "training", hyperparams.get("training", {})
        )
        data_config = pfirrman_config.get("data", hyperparams.get("data", {}))
        loss_config = pfirrman_config.get("loss", hyperparams.get("loss", {}))
    else:
        training_config = hyperparams.get("training", {})
        data_config = hyperparams.get("data", {})
        loss_config = hyperparams.get("loss", {})

    # Initialize wandb
    run = initialize_wandb_for_training(
        config=config,
        hyperparams=hyperparams,
        experiment_name=experiment_name,
        model_config=model_config,
        architecture_family=architecture_family,
        device=device,
        project_root=project_root,
    )

    # Set up paths
    data_dir = Path("../data/mri_phen/processed")
    manifest_path = data_dir / "pfirrman_training_manifest.csv"

    if not manifest_path.exists():
        print(f"Error: Training manifest not found: {manifest_path}")
        print("Please ensure data preprocessing has been completed.")
        sys.exit(1)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "outputs" / "pfirrman_model" / "runs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Output directory: {output_dir}")

    # Create datasets (multi-sequence only)
    train_dataset, val_dataset, test_dataset = create_pfirrman_datasets(
        manifest_path=manifest_path, project_root=project_root, data_config=data_config
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 16),
        shuffle=True,
        num_workers=hyperparams.get("num_workers", 4),
        pin_memory=hyperparams.get("pin_memory", True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("batch_size", 16),
        shuffle=False,
        num_workers=hyperparams.get("num_workers", 4),
        pin_memory=hyperparams.get("pin_memory", True),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.get("batch_size", 16),
        shuffle=False,
        num_workers=hyperparams.get("num_workers", 4),
        pin_memory=hyperparams.get("pin_memory", True),
    )

    log_metrics(
        {
            "dataset/train_samples": len(train_dataset),
            "dataset/val_samples": len(val_dataset),
            "dataset/test_samples": len(test_dataset),
        }
    )

    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(
        train_dataset, smoothing=loss_config.get("weight_smoothing", 0.1)
    )
    class_weights = class_weights.to(device)
    print(f"\n✓ Computed class weights: {class_weights.tolist()}")
    log_metrics({"loss/class_weights": class_weights.tolist()})

    # Set random seed
    torch.manual_seed(hyperparams.get("seed", 42))
    import numpy as np

    np.random.seed(hyperparams.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hyperparams.get("seed", 42))

    # Create model
    model = create_pfirrman_model_with_verification(
        model_config, device, hyperparams=hyperparams
    )

    # Setup loss function
    loss_type = loss_config.get("type", "weighted_ce")

    if loss_type == "cross_entropy":
        criterion = create_loss_function(
            loss_type="cross_entropy", reduction="mean", task_type="multi_class"
        )
    elif loss_type == "weighted_ce":
        criterion = create_loss_function(
            loss_type="weighted_ce",
            class_weights=class_weights,
            reduction="mean",
            task_type="multi_class",
        )
    elif loss_type == "focal_multiclass":
        focal_config = loss_config.get("focal", {})
        criterion = create_loss_function(
            loss_type="focal_multiclass",
            focal_alpha=focal_config.get("alpha"),
            focal_gamma=focal_config.get("gamma", 2.0),
            reduction="mean",
            task_type="multi_class",
        )
    else:
        # Default to weighted cross-entropy
        criterion = create_loss_function(
            loss_type="weighted_ce",
            class_weights=class_weights,
            reduction="mean",
            task_type="multi_class",
        )

    # Setup optimizer
    if training_config.get("optimizer", "adam").lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
    elif training_config.get("optimizer", "adam").lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config.get("learning_rate", 0.001),
            momentum=0.9,
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {training_config.get('optimizer')}")

    # Setup scheduler
    scheduler_type = training_config.get("scheduler", "cosine").lower()
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.get("num_epochs", 50), eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    else:
        scheduler = None

    # Training loop
    num_epochs = training_config.get("num_epochs", 50)
    best_val_f1 = 0.0
    best_epoch = 0
    training_history = []

    print(f"\n{'=' * 80}")
    print(f"Starting training for {num_epochs} epochs...")
    print(f"{'=' * 80}")

    # Early stopping setup
    early_stopping_config = training_config.get("early_stopping", {})
    early_stopping_enabled = early_stopping_config.get("enabled", False)
    if early_stopping_enabled:
        patience = early_stopping_config.get("patience", 15)
        monitor_metric = early_stopping_config.get("monitor", "val_overall/weighted_f1")
        mode = early_stopping_config.get("mode", "max")
        min_delta = early_stopping_config.get("min_delta", 0.001)
        best_metric = float("-inf") if mode == "max" else float("inf")
        patience_counter = 0
        print(
            f"✓ Early stopping enabled: patience={patience}, monitor={monitor_metric}, mode={mode}"
        )
    else:
        print("✓ Early stopping disabled")

    # Gradient clipping
    gradient_clip = training_config.get("gradient_clip", 0.0)
    if gradient_clip > 0.0:
        print(f"✓ Gradient clipping enabled: {gradient_clip}")
    else:
        print("✓ Gradient clipping disabled")

    for epoch in range(num_epochs):
        # Train
        train_loss, train_metrics = train_epoch_pfirrman(
            model, train_loader, criterion, optimizer, device, epoch, gradient_clip
        )

        # Validate
        val_loss, val_metrics = validate_epoch_pfirrman(
            model, val_loader, criterion, device, epoch
        )

        # Update learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["overall/weighted_f1"])
                current_lr = optimizer.param_groups[0]["lr"]
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = training_config.get("learning_rate", 0.001)

        # Prepare metrics for logging
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }

        # Add per-class metrics
        for grade in range(1, 6):
            class_name = f"Grade_{grade}"
            epoch_metrics[f"train_{class_name}/precision"] = train_metrics[
                f"{class_name}/precision"
            ]
            epoch_metrics[f"train_{class_name}/recall"] = train_metrics[
                f"{class_name}/recall"
            ]
            epoch_metrics[f"train_{class_name}/f1"] = train_metrics[f"{class_name}/f1"]
            epoch_metrics[f"val_{class_name}/precision"] = val_metrics[
                f"{class_name}/precision"
            ]
            epoch_metrics[f"val_{class_name}/recall"] = val_metrics[
                f"{class_name}/recall"
            ]
            epoch_metrics[f"val_{class_name}/f1"] = val_metrics[f"{class_name}/f1"]

        # Add overall metrics
        epoch_metrics.update(
            {
                "train_overall/accuracy": train_metrics["overall/accuracy"],
                "train_overall/macro_precision": train_metrics[
                    "overall/macro_precision"
                ],
                "train_overall/macro_recall": train_metrics["overall/macro_recall"],
                "train_overall/macro_f1": train_metrics["overall/macro_f1"],
                "train_overall/weighted_precision": train_metrics[
                    "overall/weighted_precision"
                ],
                "train_overall/weighted_recall": train_metrics[
                    "overall/weighted_recall"
                ],
                "train_overall/weighted_f1": train_metrics["overall/weighted_f1"],
                "val_overall/accuracy": val_metrics["overall/accuracy"],
                "val_overall/macro_precision": val_metrics["overall/macro_precision"],
                "val_overall/macro_recall": val_metrics["overall/macro_recall"],
                "val_overall/macro_f1": val_metrics["overall/macro_f1"],
                "val_overall/weighted_precision": val_metrics[
                    "overall/weighted_precision"
                ],
                "val_overall/weighted_recall": val_metrics["overall/weighted_recall"],
                "val_overall/weighted_f1": val_metrics["overall/weighted_f1"],
            }
        )

        training_history.append(epoch_metrics)
        log_metrics(epoch_metrics, step=epoch)

        # Check for best model (use weighted F1 as primary metric)
        current_val_f1 = val_metrics["overall/weighted_f1"]
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": val_loss,
                "model_config": model_config,
                "hyperparams": hyperparams,
            }

            best_model_path = weights_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)

            log_model_checkpoint(
                str(best_model_path),
                artifact_name="pfirrman_model_best",
                artifact_type="model",
                aliases=["best"],
                metadata={"epoch": epoch, "val_f1": best_val_f1},
            )

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(
            f"  Val Weighted F1: {current_val_f1:.4f}, Best: {best_val_f1:.4f} (Epoch {best_epoch + 1})"
        )
        print(f"  Val Accuracy: {val_metrics['overall/accuracy']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Early stopping check
        if early_stopping_enabled:
            current_metric = val_metrics.get(
                monitor_metric, val_loss if "loss" in monitor_metric else current_val_f1
            )

            if mode == "max":
                is_better = current_metric > (best_metric + min_delta)
            else:
                is_better = current_metric < (best_metric - min_delta)

            if is_better:
                best_metric = current_metric
                patience_counter = 0
                print(
                    f"  ✓ Improvement detected ({monitor_metric}: {current_metric:.4f})"
                )
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\n{'=' * 80}")
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    print(
                        f"Best model: Epoch {best_epoch + 1}, Val F1: {best_val_f1:.4f}"
                    )
                    print(f"{'=' * 80}")
                    break

    # Save final model
    final_checkpoint = {
        "epoch": num_epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": val_metrics["overall/weighted_f1"],
        "val_loss": val_loss,
        "model_config": model_config,
        "hyperparams": hyperparams,
    }
    final_model_path = weights_dir / "final_model.pth"
    torch.save(final_checkpoint, final_model_path)

    log_model_checkpoint(
        str(final_model_path),
        artifact_name="pfirrman_model_final",
        artifact_type="model",
        aliases=["latest"],
        metadata={
            "epoch": num_epochs - 1,
            "val_f1": val_metrics["overall/weighted_f1"],
        },
    )

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # Save config
    config_path_save = output_dir / "config.yaml"
    import yaml

    with open(config_path_save, "w") as f:
        yaml.dump({"hyperparams": hyperparams, "model_config": model_config}, f)

    # Generate predictions on all splits
    print("\n" + "=" * 80)
    print("Generating predictions...")
    print("=" * 80)

    # Load best model for predictions
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        predictions_df = generate_pfirrman_predictions(
            model, loader, device, dataset_name=split_name, return_dataframe=True
        )
        predictions_path = predictions_dir / f"{split_name}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✓ Saved {split_name} predictions: {predictions_path}")

    print(f"\n{'=' * 80}")
    print("Training completed!")
    print(f"Best model: Epoch {best_epoch + 1}, Val Weighted F1: {best_val_f1:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}")

    # Finish wandb run
    finish_run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
