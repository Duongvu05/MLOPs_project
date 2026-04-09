#!/usr/bin/env python3
"""
Training script for pathology classification models using Simple Multi-Sequence Fusion.

This script trains models using the simple_multi_sequence_fusion architecture
with different backbone options:
- ResNet: resnet18, resnet34, resnet50
- EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2
- DenseNet: densenet121, densenet169, densenet201
- Vision Transformer: vit_base_patch16_224, vit_large_patch16_224

Usage:
    python scripts/train_pathology_model.py --backbone resnet18
    python scripts/train_pathology_model.py --backbone densenet121 --config configs/wandb_config.yaml
    python scripts/train_pathology_model.py --backbone vit_base_patch16_224
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import List

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

from utils.label_utils import label_names, get_label_weights_torch, get_label_statistics
from utils.wandb_utils import (
    log_metrics,
    log_model_checkpoint,
    log_data_artifact,
    finish_run,
)
from utils.loss_utils import create_loss_function
from training.pathology_training_utils import (
    load_config_and_override_architecture,
    initialize_wandb_for_training,
    create_pathology_datasets,
    create_model_with_verification,
    train_epoch,
    validate_epoch,
    generate_predictions,
    optimize_and_save_thresholds,
)


def compare_model_structures(
    current_model: torch.nn.Module, checkpoint_state_dict: dict, logger
) -> list:
    """
    Compare current model structure with checkpoint state dict to identify differences.

    Args:
        current_model: Current model instance
        checkpoint_state_dict: State dict from checkpoint
        logger: Logger instance

    Returns:
        List of difference descriptions
    """
    differences = []

    current_keys = set(current_model.state_dict().keys())
    checkpoint_keys = set(checkpoint_state_dict.keys())

    # Keys only in current model
    current_only = current_keys - checkpoint_keys
    if current_only:
        differences.append(
            f"Keys only in current model ({len(current_only)}): {list(current_only)[:5]}{'...' if len(current_only) > 5 else ''}"
        )

    # Keys only in checkpoint
    checkpoint_only = checkpoint_keys - current_keys
    if checkpoint_only:
        differences.append(
            f"Keys only in checkpoint ({len(checkpoint_only)}): {list(checkpoint_only)[:5]}{'...' if len(checkpoint_only) > 5 else ''}"
        )

    # Common keys with different shapes
    common_keys = current_keys & checkpoint_keys
    shape_mismatches = []
    for key in common_keys:
        current_shape = current_model.state_dict()[key].shape
        checkpoint_shape = checkpoint_state_dict[key].shape
        if current_shape != checkpoint_shape:
            shape_mismatches.append(f"{key}: {current_shape} vs {checkpoint_shape}")

    if shape_mismatches:
        differences.append(
            f"Shape mismatches ({len(shape_mismatches)}): {shape_mismatches[:3]}{'...' if len(shape_mismatches) > 3 else ''}"
        )

    return differences


def get_backbone_family(backbone: str) -> str:
    """Get backbone family name from backbone string."""
    if backbone.startswith("resnet"):
        return "resnet"
    elif backbone.startswith("efficientnet"):
        return "efficientnet"
    elif backbone.startswith("densenet"):
        return "densenet"
    elif backbone.startswith("vit"):
        return "vit"
    else:
        return "unknown"


def validate_sequences(sequences: List[str]) -> List[str]:
    """Validate and normalize sequence names."""
    valid_sequences = ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]

    if not sequences:
        return valid_sequences  # Use all by default

    # Normalize sequence names to upper case
    normalized = [seq.upper() for seq in sequences]

    # Validate sequences
    invalid = [seq for seq in normalized if seq not in valid_sequences]
    if invalid:
        raise ValueError(
            f"Invalid sequences: {invalid}. Must be from: {valid_sequences}"
        )

    return normalized


def get_valid_backbones() -> list:
    """Get list of all valid backbone options."""
    return [
        "resnet18",
        "resnet34",
        "resnet50",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "densenet121",
        "densenet169",
        "densenet201",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
    ]


def validate_and_clean_checkpoint(
    checkpoint_path: Path, expected_backbone: str, logger
) -> bool:
    """
    Validate if checkpoint is compatible with current backbone.
    If not compatible, optionally remove it (with user confirmation).

    Returns:
        bool: True if checkpoint is compatible or doesn't exist, False if incompatible
    """
    if not checkpoint_path.exists():
        return True  # No checkpoint to validate

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Check backbone compatibility
        checkpoint_backbone = checkpoint.get("backbone")
        checkpoint_hyperparams = checkpoint.get("hyperparams", {})

        if (
            not checkpoint_backbone
            and "simple_multi_sequence_fusion" in checkpoint_hyperparams
        ):
            checkpoint_backbone = checkpoint_hyperparams[
                "simple_multi_sequence_fusion"
            ].get("backbone_type")

        if checkpoint_backbone and checkpoint_backbone != expected_backbone:
            logger.warning(f"Found incompatible checkpoint at {checkpoint_path}")
            logger.warning(
                f"Checkpoint backbone: {checkpoint_backbone}, Expected: {expected_backbone}"
            )

            # For now, just warn and return False - don't auto-delete
            return False

        return True  # Compatible or no backbone info (assume compatible)

    except Exception as e:
        logger.warning(
            f"Error validating checkpoint {checkpoint_path}: {str(e)[:200]}..."
        )
        return False  # Assume incompatible if can't load


def main():
    parser = argparse.ArgumentParser(
        description="Train pathology classification model using Simple Multi-Sequence Fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with ResNet-18 backbone
  python scripts/train_pathology_model.py --backbone resnet18
  
  # Train with DenseNet-121 backbone
  python scripts/train_pathology_model.py --backbone densenet121
  
  # Train with ViT-Base backbone
  python scripts/train_pathology_model.py --backbone vit_base_patch16_224
  
  # Train with custom config
  python scripts/train_pathology_model.py --backbone efficientnet_b0 --config configs/wandb_config.yaml

Supported backbones:
  ResNet: resnet18, resnet34, resnet50
  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2
  DenseNet: densenet121, densenet169, densenet201
  Vision Transformer: vit_base_patch16_224, vit_large_patch16_224
        """,
    )

    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Backbone architecture (e.g., resnet18, densenet121, vit_base_patch16_224)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/wandb_config.yaml",
        help="Path to configuration file (default: configs/wandb_config.yaml)",
    )

    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequences to use for training (e.g., --sequences AX_T2 SAG_STIR). Available: SAG_T2 SAG_T1 AX_T2 SAG_STIR",
    )

    parser.add_argument(
        "--no-wandb-upload",
        action="store_true",
        help="Disable uploading model checkpoints to wandb (save locally only)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints (default: outputs/pathology_model/runs/)",
    )

    args = parser.parse_args()

    # Validate backbone
    valid_backbones = get_valid_backbones()
    if args.backbone not in valid_backbones:
        logger.error(f"Unknown backbone '{args.backbone}'")
        logger.error("Supported backbones:")
        logger.error("  ResNet: resnet18, resnet34, resnet50")
        logger.error(
            "  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2"
        )
        logger.error("  DenseNet: densenet121, densenet169, densenet201")
        logger.error(
            "  Vision Transformer: vit_base_patch16_224, vit_large_patch16_224"
        )
        sys.exit(1)

    # Validate and normalize sequences
    selected_sequences = validate_sequences(args.sequences)
    logger.info(f"Using sequences: {selected_sequences}")

    backbone_family = get_backbone_family(args.backbone)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.success(f"Using device: {device}")
    logger.success(f"Project root: {project_root}")

    # Load configuration and override architecture
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Always use simple_multi_sequence_fusion
    architecture_options = ["simple_multi_sequence_fusion"]
    experiment_prefix = f"pathology_multilabel_{backbone_family}"

    config, hyperparams, model_config, experiment_name, timestamp = (
        load_config_and_override_architecture(
            config_path=config_path,
            target_architecture="simple_multi_sequence_fusion",
            architecture_options=architecture_options,
            experiment_prefix=experiment_prefix,
        )
    )

    # Always use simple_multi_sequence_fusion architecture
    model_config["architecture"] = "simple_multi_sequence_fusion"

    # Set backbone for simple_multi_sequence_fusion
    if "simple_multi_sequence_fusion" not in hyperparams:
        hyperparams["simple_multi_sequence_fusion"] = {}
    hyperparams["simple_multi_sequence_fusion"]["backbone_type"] = args.backbone
    logger.info(f"Using simple_multi_sequence_fusion with backbone: {args.backbone}")

    # Update sequences in configuration
    if "multi_sequence" not in hyperparams:
        hyperparams["multi_sequence"] = {}
    hyperparams["multi_sequence"]["sequences"] = selected_sequences

    if "simple_multi_sequence_fusion" in hyperparams:
        hyperparams["simple_multi_sequence_fusion"]["sequences"] = selected_sequences

    logger.success(f"Configured sequences: {selected_sequences}")

    # Extract configuration sections
    task_config = hyperparams.get("task", {})
    training_config = hyperparams["training"]
    data_config = hyperparams["data"]
    loss_config = hyperparams["loss"]

    # Initialize wandb
    run = initialize_wandb_for_training(
        config=config,
        hyperparams=hyperparams,
        experiment_name=experiment_name,
        model_config=model_config,
        architecture_family=backbone_family,
        device=device,
        project_root=project_root,
    )

    # Set up paths
    data_dir = Path("../data/mri_phen/processed")
    manifest_path = data_dir / "pathology_training_manifest.csv"

    if not manifest_path.exists():
        logger.error(f"Training manifest not found: {manifest_path}")
        logger.error("Please ensure data preprocessing has been completed.")
        sys.exit(1)

    # Create output directory organized by backbone
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create directory structure: outputs/pathology_model/runs/{backbone}/
        output_dir = (
            project_root / "outputs" / "pathology_model" / "runs" / f"{args.backbone}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Output directory: {output_dir}")

    # Validate existing checkpoints
    best_checkpoint_path = weights_dir / "best_model.pth"
    final_checkpoint_path = weights_dir / "final_model.pth"

    if not validate_and_clean_checkpoint(best_checkpoint_path, args.backbone, logger):
        logger.warning(f"Existing best checkpoint is incompatible with {args.backbone}")
        logger.info("Training will create new checkpoint with correct architecture")

    if not validate_and_clean_checkpoint(final_checkpoint_path, args.backbone, logger):
        logger.warning(
            f"Existing final checkpoint is incompatible with {args.backbone}"
        )
        logger.info("Training will create new checkpoint with correct architecture")

    # Create datasets
    # Always use multi-sequence datasets (breaking change)
    train_dataset, val_dataset, test_dataset = create_pathology_datasets(
        manifest_path=manifest_path,
        project_root=project_root,
        data_config=data_config,
        multi_sequence_config=hyperparams.get("multi_sequence", {}),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=hyperparams.get("num_workers", 4),
        pin_memory=hyperparams.get("pin_memory", True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=hyperparams.get("num_workers", 4),
        pin_memory=hyperparams.get("pin_memory", True),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
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

    # Get label statistics and weights
    label_stats = get_label_statistics(manifest_path)
    logger.info("Label Statistics:")
    for label_name, stats in label_stats["labels"].items():
        logger.info(
            f"  {label_name}: {stats['positive']} positive ({stats['positive_rate'] * 100:.1f}%), weight: {stats['weight']:.2f}"
        )

    if loss_config.get("pos_weights") is not None:
        pos_weights = torch.tensor(loss_config["pos_weights"], dtype=torch.float32).to(
            device
        )
        logger.success(f"Using configured pos_weights: {pos_weights.tolist()}")
    else:
        pos_weights = get_label_weights_torch(manifest_path).to(device)
        logger.success(f"Calculated pos_weights: {pos_weights.tolist()}")

    log_metrics({"loss/pos_weights": pos_weights.tolist()})

    # Set random seed
    torch.manual_seed(hyperparams.get("seed", 42))

    np.random.seed(hyperparams.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hyperparams.get("seed", 42))

    # Create model with IVD encoding from config
    # Read IVD encoding settings from config instead of hardcoding
    if "multi_sequence" not in model_config:
        model_config["multi_sequence"] = {}

    # Get IVD encoding settings from config with proper fallback logic
    use_ivd_encoding = False  # default
    ivd_encoding_mode = "positional"  # default
    ivd_embedding_dim = 16  # default

    # First check simple_multi_sequence_fusion config (more specific)
    if "simple_multi_sequence_fusion" in hyperparams:
        simple_config = hyperparams["simple_multi_sequence_fusion"]
        use_ivd_encoding = simple_config.get("use_ivd_encoding", use_ivd_encoding)
        ivd_encoding_mode = simple_config.get("ivd_encoding_mode", ivd_encoding_mode)
        ivd_embedding_dim = simple_config.get("ivd_embedding_dim", ivd_embedding_dim)

    # Then check general multi_sequence config (fallback)
    if "multi_sequence" in hyperparams:
        multi_config = hyperparams["multi_sequence"]
        use_ivd_encoding = multi_config.get("use_ivd_encoding", use_ivd_encoding)
        ivd_encoding_mode = multi_config.get("ivd_encoding_mode", ivd_encoding_mode)
        ivd_embedding_dim = multi_config.get("ivd_embedding_dim", ivd_embedding_dim)

    # Apply to model config
    model_config["multi_sequence"]["use_ivd_encoding"] = use_ivd_encoding
    model_config["multi_sequence"]["ivd_encoding_mode"] = ivd_encoding_mode
    model_config["multi_sequence"]["ivd_embedding_dim"] = ivd_embedding_dim

    logger.success(
        f"IVD Encoding: {'Enabled' if model_config['multi_sequence'].get('use_ivd_encoding', False) else 'Disabled'}"
    )
    if model_config["multi_sequence"].get("use_ivd_encoding", False):
        logger.success(
            f"IVD Encoding Mode: {model_config['multi_sequence'].get('ivd_encoding_mode', 'positional')}"
        )
        if model_config["multi_sequence"].get("ivd_encoding_mode") == "positional":
            logger.success(
                f"IVD Embedding Dimension: {model_config['multi_sequence'].get('ivd_embedding_dim', 16)}"
            )

    model = create_model_with_verification(
        model_config, device, hyperparams=hyperparams
    )

    # Setup loss function
    loss_type = loss_config.get("type", "bce_with_logits")

    if loss_type == "bce_with_logits":
        criterion = create_loss_function(
            loss_type="bce_with_logits", pos_weights=pos_weights, reduction="mean"
        )
    elif loss_type == "focal":
        focal_config = loss_config.get("focal", {})
        criterion = create_loss_function(
            loss_type="focal",
            focal_alpha=focal_config.get("alpha"),
            focal_gamma=focal_config.get("gamma", 2.0),
            reduction="mean",
        )
    elif loss_type == "focal_with_weights":
        focal_config = loss_config.get("focal", {})
        criterion = create_loss_function(
            loss_type="focal_with_weights",
            pos_weights=pos_weights,
            focal_gamma=focal_config.get("gamma", 2.0),
            reduction="mean",
        )
    elif loss_type == "asymmetric":
        asym_config = loss_config.get("asymmetric", {})
        criterion = create_loss_function(
            loss_type="asymmetric",
            asym_gamma_neg=asym_config.get("gamma_neg", 4.0),
            asym_gamma_pos=asym_config.get("gamma_pos", 1.0),
            reduction="mean",
        )
    else:
        criterion = create_loss_function(
            loss_type="bce_with_logits", pos_weights=pos_weights, reduction="mean"
        )

    # Setup optimizer
    if training_config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
    elif training_config["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config["learning_rate"],
            momentum=0.9,
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")

    # Setup scheduler
    scheduler_type = training_config.get("scheduler", "cosine").lower()
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config["num_epochs"], eta_min=1e-6
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
    num_epochs = training_config["num_epochs"]
    best_val_f1 = 0.0
    best_epoch = 0
    training_history = []
    label_names_list = label_names()
    best_model_path = weights_dir / "best_model.pth"  # Initialize best model path

    logger.info(f"{'=' * 80}")
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Label names: {label_names_list}")
    logger.info(f"{'=' * 80}")

    # Early stopping setup
    early_stopping_config = training_config.get("early_stopping", {})
    early_stopping_enabled = early_stopping_config.get("enabled", False)
    if early_stopping_enabled:
        patience = early_stopping_config.get("patience", 15)
        monitor_metric = early_stopping_config.get("monitor", "val_macro_f1")
        mode = early_stopping_config.get("mode", "max")
        min_delta = early_stopping_config.get("min_delta", 0.001)
        best_metric = float("-inf") if mode == "max" else float("inf")
        patience_counter = 0
        logger.info(
            f"Early stopping enabled: patience={patience}, monitor={monitor_metric}, mode={mode}"
        )
    else:
        logger.info("Early stopping disabled")

    # Gradient clipping
    gradient_clip = training_config.get("gradient_clip", 0.0)
    if gradient_clip > 0.0:
        logger.info(f"Gradient clipping enabled: {gradient_clip}")
    else:
        logger.info("Gradient clipping disabled")

    for epoch in range(num_epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            label_names_list,
            gradient_clip,
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, label_names_list
        )

        # Update learning rate
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["overall/macro_f1"])
                current_lr = optimizer.param_groups[0]["lr"]
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = training_config["learning_rate"]

        # Prepare metrics for logging
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }

        # Add per-label metrics
        for label_name in label_names_list:
            epoch_metrics[f"train_{label_name}/precision"] = train_metrics[
                f"{label_name}/precision"
            ]
            epoch_metrics[f"train_{label_name}/recall"] = train_metrics[
                f"{label_name}/recall"
            ]
            epoch_metrics[f"train_{label_name}/f1"] = train_metrics[f"{label_name}/f1"]
            epoch_metrics[f"train_{label_name}/roc_auc"] = train_metrics[
                f"{label_name}/roc_auc"
            ]
            epoch_metrics[f"val_{label_name}/precision"] = val_metrics[
                f"{label_name}/precision"
            ]
            epoch_metrics[f"val_{label_name}/recall"] = val_metrics[
                f"{label_name}/recall"
            ]
            epoch_metrics[f"val_{label_name}/f1"] = val_metrics[f"{label_name}/f1"]
            epoch_metrics[f"val_{label_name}/roc_auc"] = val_metrics[
                f"{label_name}/roc_auc"
            ]

        # Add overall metrics
        epoch_metrics.update(
            {
                "train_overall/hamming_loss": train_metrics["overall/hamming_loss"],
                "train_overall/subset_accuracy": train_metrics[
                    "overall/subset_accuracy"
                ],
                "train_overall/macro_precision": train_metrics[
                    "overall/macro_precision"
                ],
                "train_overall/macro_recall": train_metrics["overall/macro_recall"],
                "train_overall/macro_f1": train_metrics["overall/macro_f1"],
                "train_overall/micro_precision": train_metrics[
                    "overall/micro_precision"
                ],
                "train_overall/micro_recall": train_metrics["overall/micro_recall"],
                "train_overall/micro_f1": train_metrics["overall/micro_f1"],
                "val_overall/hamming_loss": val_metrics["overall/hamming_loss"],
                "val_overall/subset_accuracy": val_metrics["overall/subset_accuracy"],
                "val_overall/macro_precision": val_metrics["overall/macro_precision"],
                "val_overall/macro_recall": val_metrics["overall/macro_recall"],
                "val_overall/macro_f1": val_metrics["overall/macro_f1"],
                "val_overall/micro_precision": val_metrics["overall/micro_precision"],
                "val_overall/micro_recall": val_metrics["overall/micro_recall"],
                "val_overall/micro_f1": val_metrics["overall/micro_f1"],
            }
        )

        training_history.append(epoch_metrics)
        # Log epoch metrics (let wandb handle step management automatically)
        log_metrics(epoch_metrics, step=None)

        # Log epoch summary
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(
            f"  Val Macro F1: {val_metrics['overall/macro_f1']:.4f}, Best: {best_val_f1:.4f} (Epoch {best_epoch + 1})"
        )
        logger.info(f"  Learning Rate: {current_lr:.6f}")

        # Check for best model
        current_val_f1 = val_metrics["overall/macro_f1"]
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
                "backbone": args.backbone,  # Add backbone info for compatibility check
                "architecture": "simple_multi_sequence_fusion",  # Add architecture info
                # Enhanced model compatibility information
                "model_class": type(model).__name__,
                "model_state_keys": list(model.state_dict().keys()),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "model_structure_hash": hash(str(sorted(model.state_dict().keys()))),
                # Add IVD encoding information for better compatibility checking
                "ivd_encoding_enabled": model_config["multi_sequence"].get(
                    "use_ivd_encoding", False
                ),
                "ivd_encoding_mode": model_config["multi_sequence"].get(
                    "ivd_encoding_mode", "positional"
                ),
                "ivd_embedding_dim": model_config["multi_sequence"].get(
                    "ivd_embedding_dim", 16
                ),
                "sequences": selected_sequences,
                "training_timestamp": timestamp,
            }

            best_model_path = weights_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)

            # Only log model checkpoint to wandb if enabled in config and not disabled by flag
            should_upload = (
                config.get("artifacts", {}).get("log_model_checkpoints", False)
                and not args.no_wandb_upload
            )

            if should_upload:
                log_model_checkpoint(
                    str(best_model_path),
                    artifact_name=f"pathology_model_best_{args.backbone}",  # Include backbone in artifact name
                    artifact_type="model",
                    aliases=["best"],
                    metadata={
                        "epoch": epoch,
                        "val_f1": best_val_f1,
                        "backbone": args.backbone,
                    },
                )
                logger.info("✓ Model checkpoint uploaded to wandb")
            else:
                if args.no_wandb_upload:
                    logger.info(
                        "✓ Model checkpoint upload disabled by --no-wandb-upload flag"
                    )
                else:
                    logger.info(
                        "✓ Model checkpoint upload disabled in config - saved locally only"
                    )

        # Log epoch summary to console
        logger.success(f"Epoch {epoch + 1}/{num_epochs} completed")
        logger.success(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.success(
            f"  Val Macro F1: {current_val_f1:.4f}, Best: {best_val_f1:.4f} (Epoch {best_epoch + 1})"
        )
        logger.success(f"  Learning Rate: {current_lr:.6f}")

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
                logger.info(
                    f"  ✓ Improvement detected ({monitor_metric}: {current_metric:.4f})"
                )
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{patience})")

            if is_better:
                best_metric = current_metric
                patience_counter = 0
                logger.success(
                    f"  Improvement detected ({monitor_metric}: {current_metric:.4f})"
                )
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    logger.warning(f"{'=' * 80}")
                    logger.warning(f"Early stopping triggered at epoch {epoch + 1}")
                    logger.warning(
                        f"Best model: Epoch {best_epoch + 1}, Val F1: {best_val_f1:.4f}"
                    )
                    logger.warning(f"{'=' * 80}")
                    break

    # Save final model
    final_checkpoint = {
        "epoch": num_epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": val_metrics["overall/macro_f1"],
        "val_loss": val_loss,
        "model_config": model_config,
        "hyperparams": hyperparams,
        "backbone": args.backbone,  # Add backbone info for compatibility check
        "architecture": "simple_multi_sequence_fusion",  # Add architecture info
        # Enhanced model compatibility information
        "model_class": type(model).__name__,
        "model_state_keys": list(model.state_dict().keys()),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "model_structure_hash": hash(str(sorted(model.state_dict().keys()))),
        # Add IVD encoding information for better compatibility checking
        "ivd_encoding_enabled": model_config["multi_sequence"].get(
            "use_ivd_encoding", False
        ),
        "ivd_encoding_mode": model_config["multi_sequence"].get(
            "ivd_encoding_mode", "positional"
        ),
        "ivd_embedding_dim": model_config["multi_sequence"].get(
            "ivd_embedding_dim", 16
        ),
        "sequences": selected_sequences,
        "training_timestamp": timestamp,
    }
    final_model_path = weights_dir / "final_model.pth"
    torch.save(final_checkpoint, final_model_path)

    # Save training history
    history_df = pd.DataFrame(training_history)
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    log_data_artifact(str(history_path), "training_history", "dataset")

    # Optimize thresholds - with safe checkpoint loading
    best_model_checkpoint_path = weights_dir / "best_model.pth"
    if best_model_checkpoint_path.exists():
        try:
            logger.info(f"Loading best model checkpoint: {best_model_checkpoint_path}")
            checkpoint = torch.load(
                best_model_checkpoint_path, map_location=device, weights_only=False
            )

            # Comprehensive checkpoint compatibility verification
            checkpoint_config = checkpoint.get("model_config", {})
            checkpoint_hyperparams = checkpoint.get("hyperparams", {})
            checkpoint_backbone = checkpoint.get("backbone")  # Direct backbone field
            checkpoint_architecture = checkpoint.get(
                "architecture"
            )  # Direct architecture field

            # Get current model configuration
            current_backbone = args.backbone
            current_architecture = model_config["architecture"]

            # Multiple compatibility checks
            compatibility_issues = []

            # Check backbone compatibility
            checkpoint_backbone_from_hyperparams = None
            if "simple_multi_sequence_fusion" in checkpoint_hyperparams:
                checkpoint_backbone_from_hyperparams = checkpoint_hyperparams[
                    "simple_multi_sequence_fusion"
                ].get("backbone_type")

            # Use the most reliable backbone source
            effective_checkpoint_backbone = (
                checkpoint_backbone or checkpoint_backbone_from_hyperparams
            )

            if (
                effective_checkpoint_backbone
                and effective_checkpoint_backbone != current_backbone
            ):
                compatibility_issues.append(
                    f"backbone mismatch: {effective_checkpoint_backbone} != {current_backbone}"
                )

            # Check architecture compatibility
            if (
                checkpoint_architecture
                and checkpoint_architecture != current_architecture
            ):
                compatibility_issues.append(
                    f"architecture mismatch: {checkpoint_architecture} != {current_architecture}"
                )

            # Check model configuration compatibility
            if checkpoint_config.get("num_labels") != model_config.get("num_labels"):
                compatibility_issues.append(
                    f"num_labels mismatch: {checkpoint_config.get('num_labels')} != {model_config.get('num_labels')}"
                )

            # Check model class compatibility
            checkpoint_model_class = checkpoint.get("model_class")
            current_model_class = type(model).__name__
            if checkpoint_model_class and checkpoint_model_class != current_model_class:
                compatibility_issues.append(
                    f"model class mismatch: {checkpoint_model_class} != {current_model_class}"
                )

            # Check IVD encoding compatibility - this is critical for model structure
            checkpoint_ivd_encoding = False
            if checkpoint_config.get("multi_sequence", {}).get("use_ivd_encoding"):
                checkpoint_ivd_encoding = True
            elif "simple_multi_sequence_fusion" in checkpoint_hyperparams:
                checkpoint_ivd_encoding = checkpoint_hyperparams[
                    "simple_multi_sequence_fusion"
                ].get("use_ivd_encoding", False)
            elif "multi_sequence" in checkpoint_hyperparams:
                checkpoint_ivd_encoding = checkpoint_hyperparams["multi_sequence"].get(
                    "use_ivd_encoding", False
                )

            current_ivd_encoding = model_config["multi_sequence"].get(
                "use_ivd_encoding", False
            )

            # Only treat IVD encoding mismatch as a warning, not a blocking issue
            # This allows loading models trained with different IVD settings
            if checkpoint_ivd_encoding != current_ivd_encoding:
                logger.warning(
                    f"IVD encoding mismatch: checkpoint={checkpoint_ivd_encoding}, current={current_ivd_encoding}"
                )
                logger.warning(
                    "Will attempt to load with non-strict mode to handle structure differences"
                )

            # Skip structure hash check if IVD encoding is different, as this is expected
            if checkpoint_ivd_encoding == current_ivd_encoding:
                checkpoint_structure_hash = checkpoint.get("model_structure_hash")
                current_structure_hash = hash(str(sorted(model.state_dict().keys())))
                if (
                    checkpoint_structure_hash
                    and checkpoint_structure_hash != current_structure_hash
                ):
                    compatibility_issues.append(
                        "model structure hash mismatch (indicates structural differences)"
                    )

            # If there are compatibility issues, warn and skip loading
            if compatibility_issues:
                logger.warning("Checkpoint compatibility issues detected:")
                for issue in compatibility_issues:
                    logger.warning(f"  - {issue}")
                logger.warning(
                    "Skipping checkpoint loading for threshold optimization."
                )
                logger.warning("Using current model state for threshold optimization.")

                # Log some debug info about the models
                logger.info(
                    f"Checkpoint info: epoch={checkpoint.get('epoch')}, class={checkpoint.get('model_class')}, params={checkpoint.get('num_parameters')}"
                )
                logger.info(
                    f"Current model: class={current_model_class}, params={sum(p.numel() for p in model.parameters())}"
                )

            else:
                # Try to load state dict with comprehensive error handling
                try:
                    # First, try strict loading
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    logger.success(
                        f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
                    )
                except RuntimeError as e:
                    error_msg = str(e)
                    if any(
                        keyword in error_msg
                        for keyword in [
                            "size mismatch",
                            "Missing key",
                            "Unexpected key",
                        ]
                    ):
                        logger.warning(
                            f"Strict checkpoint loading failed: {error_msg[:200]}..."
                        )

                        # Try non-strict loading for IVD encoding mismatches
                        checkpoint_ivd_encoding = False
                        if checkpoint_config.get("multi_sequence", {}).get(
                            "use_ivd_encoding"
                        ):
                            checkpoint_ivd_encoding = True
                        elif "simple_multi_sequence_fusion" in checkpoint_hyperparams:
                            checkpoint_ivd_encoding = checkpoint_hyperparams[
                                "simple_multi_sequence_fusion"
                            ].get("use_ivd_encoding", False)
                        elif "multi_sequence" in checkpoint_hyperparams:
                            checkpoint_ivd_encoding = checkpoint_hyperparams[
                                "multi_sequence"
                            ].get("use_ivd_encoding", False)

                        current_ivd_encoding = model_config["multi_sequence"].get(
                            "use_ivd_encoding", False
                        )

                        if checkpoint_ivd_encoding != current_ivd_encoding:
                            logger.info(
                                "Attempting non-strict loading due to IVD encoding difference"
                            )
                            logger.info(
                                f"Checkpoint IVD: {checkpoint_ivd_encoding}, Current IVD: {current_ivd_encoding}"
                            )

                            try:
                                # Load with non-strict mode
                                missing_keys, unexpected_keys = model.load_state_dict(
                                    checkpoint["model_state_dict"], strict=False
                                )

                                # Filter IVD-related keys from the warnings
                                ivd_related_patterns = [
                                    "ivd_embedding",
                                    "ivd_encoder",
                                    "ivd_layer",
                                ]
                                missing_ivd_keys = [
                                    k
                                    for k in missing_keys
                                    if any(
                                        pattern in k.lower()
                                        for pattern in ivd_related_patterns
                                    )
                                ]
                                unexpected_ivd_keys = [
                                    k
                                    for k in unexpected_keys
                                    if any(
                                        pattern in k.lower()
                                        for pattern in ivd_related_patterns
                                    )
                                ]

                                other_missing = [
                                    k for k in missing_keys if k not in missing_ivd_keys
                                ]
                                other_unexpected = [
                                    k
                                    for k in unexpected_keys
                                    if k not in unexpected_ivd_keys
                                ]

                                if missing_ivd_keys or unexpected_ivd_keys:
                                    logger.info(
                                        "Expected IVD-related key differences:"
                                    )
                                    if missing_ivd_keys:
                                        logger.info(
                                            f"  Missing IVD keys: {missing_ivd_keys[:3]}{'...' if len(missing_ivd_keys) > 3 else ''}"
                                        )
                                    if unexpected_ivd_keys:
                                        logger.info(
                                            f"  Unexpected IVD keys: {unexpected_ivd_keys[:3]}{'...' if len(unexpected_ivd_keys) > 3 else ''}"
                                        )

                                if other_missing or other_unexpected:
                                    logger.warning("Non-IVD key differences detected:")
                                    if other_missing:
                                        logger.warning(
                                            f"  Missing keys: {other_missing[:3]}{'...' if len(other_missing) > 3 else ''}"
                                        )
                                    if other_unexpected:
                                        logger.warning(
                                            f"  Unexpected keys: {other_unexpected[:3]}{'...' if len(other_unexpected) > 3 else ''}"
                                        )
                                    logger.warning(
                                        "This may indicate incompatible model changes beyond IVD encoding."
                                    )

                                logger.success(
                                    f"Successfully loaded checkpoint with non-strict mode from epoch {checkpoint.get('epoch', 'unknown')}"
                                )

                            except RuntimeError as e2:
                                logger.error(
                                    f"Non-strict loading also failed: {str(e2)[:200]}..."
                                )
                                logger.warning(
                                    "Using current model state for threshold optimization."
                                )
                        else:
                            # Provide detailed diagnostics for other mismatches
                            differences = compare_model_structures(
                                model, checkpoint["model_state_dict"], logger
                            )
                            if differences:
                                logger.warning("Model structure differences detected:")
                                for diff in differences:
                                    logger.warning(f"  - {diff}")

                            logger.warning(
                                "This typically occurs when model architecture has changed."
                            )
                            logger.warning(
                                "Using current model state for threshold optimization."
                            )
                    else:
                        # Re-raise unexpected errors
                        logger.error(
                            f"Unexpected error during checkpoint loading: {error_msg}"
                        )
                        raise e

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)[:200]}...")
            logger.warning("Using current model state for threshold optimization.")
    else:
        logger.warning(f"Best model checkpoint not found: {best_model_checkpoint_path}")
        logger.warning("Using current model state for threshold optimization.")

    thresholds_config = hyperparams.get("thresholds", {})
    thresholds_save_path = output_dir / "thresholds.json"
    optimal_thresholds = optimize_and_save_thresholds(
        model=model,
        val_loader=val_loader,
        device=device,
        label_names=label_names_list,
        weights_dir=weights_dir,
        thresholds_config=thresholds_config,
        thresholds_save_path=thresholds_save_path,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
    )

    # Generate predictions
    logger.info("Generating predictions...")
    train_predictions = generate_predictions(
        model,
        train_loader,
        device,
        "train",
        return_dataframe=True,
        label_names=label_names_list,
    )
    val_predictions = generate_predictions(
        model,
        val_loader,
        device,
        "val",
        return_dataframe=True,
        label_names=label_names_list,
    )
    test_predictions = generate_predictions(
        model,
        test_loader,
        device,
        "test",
        return_dataframe=True,
        label_names=label_names_list,
    )

    # Save predictions
    train_pred_path = predictions_dir / "train_predictions.csv"
    val_pred_path = predictions_dir / "val_predictions.csv"
    test_pred_path = predictions_dir / "test_predictions.csv"

    train_predictions.to_csv(train_pred_path, index=False)
    val_predictions.to_csv(val_pred_path, index=False)
    test_predictions.to_csv(test_pred_path, index=False)

    log_data_artifact(str(train_pred_path), "train_predictions", "dataset")
    log_data_artifact(str(val_pred_path), "val_predictions", "dataset")
    log_data_artifact(str(test_pred_path), "test_predictions", "dataset")

    # Log final summary
    log_metrics(
        {
            "final/best_epoch": best_epoch,
            "final/best_val_f1": best_val_f1,
            "final/final_val_f1": val_metrics["overall/macro_f1"],
        }
    )

    finish_run()

    logger.success(f"\n{'=' * 80}")
    logger.success("TRAINING SUMMARY")
    logger.success(f"{'=' * 80}")
    logger.success("\nBest Model:")
    logger.success(f"  Epoch: {best_epoch + 1}")
    logger.success(f"  Val Macro F1: {best_val_f1:.4f}")
    logger.success(f"  Model path: {best_model_path}")
    logger.success("\nFinal Model:")
    logger.success(f"  Epoch: {num_epochs}")
    logger.success(f"  Val Macro F1: {val_metrics['overall/macro_f1']:.4f}")
    logger.success(f"  Model path: {final_model_path}")
    logger.success("\nPredictions:")
    logger.success(f"  Train: {train_pred_path}")
    logger.success(f"  Val: {val_pred_path}")
    logger.success(f"  Test: {test_pred_path}")
    logger.success("\nWandb Run:")
    logger.success(f"  URL: {run.url}")
    logger.success(f"  Project: {run.project}")
    logger.success(f"  Experiment: {experiment_name}")
    logger.success(f"\n{'=' * 80}")
    logger.success("Training completed successfully!")
    logger.success(f"{'=' * 80}")


if __name__ == "__main__":
    main()
