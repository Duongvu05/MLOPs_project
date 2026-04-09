#!/usr/bin/env python3
"""
Evaluation script for pathology classification models using Simple Multi-Sequence Fusion.

This script can evaluate a single backbone model or compare all trained backbone models.

Usage:
    # Evaluate all trained backbone models
    python scripts/evaluate_pathology_model.py --all

    # Evaluate a specific backbone model
    python scripts/evaluate_pathology_model.py --backbone resnet18

    # Evaluate with specific checkpoint
    python scripts/evaluate_pathology_model.py --backbone densenet121 --checkpoint final_model.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from loguru import logger
import traceback
import wandb

# Configure loguru logger with colors
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

import pandas as pd
import torch
from torch.utils.data import DataLoader

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
    calculate_metrics,
    generate_predictions,
    create_model_with_verification,
)


def validate_sequences(sequences: Optional[List[str]]) -> Optional[List[str]]:
    """Validate and normalize sequence names."""
    if not sequences:
        return None  # Use model's saved configuration

    valid_sequences = ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]

    # Normalize sequence names to upper case
    normalized = [seq.upper() for seq in sequences]

    # Validate sequences
    invalid = [seq for seq in normalized if seq not in valid_sequences]
    if invalid:
        raise ValueError(
            f"Invalid sequences: {invalid}. Must be from: {valid_sequences}"
        )

    return normalized


def analyze_checkpoint_architecture(checkpoint_path: Path) -> dict:
    """
    Analyze checkpoint to determine its architecture and compatibility.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with analysis results
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", {})

        # Architecture indicators
        has_sequence_encoders = any(
            "sequence_encoders." in key for key in state_dict.keys()
        )
        has_sequence_projections = any(
            "sequence_projections." in key for key in state_dict.keys()
        )
        has_pathology_heads = any(
            "pathology_heads." in key for key in state_dict.keys()
        )
        has_old_backbone = any("backbone." in key for key in state_dict.keys())
        has_multi_scale = any(
            "scale1_" in key or "scale2_" in key or "scale3_" in key
            for key in state_dict.keys()
        )
        has_transformer = any("transformer" in key.lower() for key in state_dict.keys())
        has_attention = any("attention" in key.lower() for key in state_dict.keys())

        # Determine architecture type
        if has_sequence_encoders and has_sequence_projections and has_pathology_heads:
            architecture_type = "simple_multi_sequence_fusion"
        elif has_multi_scale and has_attention:
            architecture_type = "multi_sequence_fusion"
        elif has_transformer:
            architecture_type = "transformer_fusion"
        elif has_old_backbone:
            architecture_type = "legacy_single_sequence"
        else:
            architecture_type = "unknown"

        return {
            "architecture_type": architecture_type,
            "checkpoint_architecture": checkpoint.get("architecture", "unknown"),
            "checkpoint_backbone": checkpoint.get("backbone", "unknown"),
            "has_sequence_encoders": has_sequence_encoders,
            "has_sequence_projections": has_sequence_projections,
            "has_pathology_heads": has_pathology_heads,
            "has_old_backbone": has_old_backbone,
            "has_multi_scale": has_multi_scale,
            "has_transformer": has_transformer,
            "has_attention": has_attention,
            "num_keys": len(state_dict),
            "key_sample": list(state_dict.keys())[:5],
        }

    except Exception as e:
        return {"architecture_type": "error", "error": str(e), "num_keys": 0}


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


def get_base_backbone_name(backbone: str) -> str:
    """Extract base backbone name from directory name, removing suffixes like _sat."""
    base_backbones = get_valid_backbones()

    # Check if backbone is already a base backbone
    if backbone in base_backbones:
        return backbone

    # Try to find matching base backbone by removing common suffixes
    for base_backbone in base_backbones:
        if backbone.startswith(base_backbone):
            return base_backbone

    # If no match found, return as-is
    return backbone


def is_valid_backbone_dir(backbone_dir: str) -> bool:
    """Check if a backbone directory name corresponds to a valid backbone."""
    base_backbone = get_base_backbone_name(backbone_dir)
    return base_backbone in get_valid_backbones()


def format_model_name(
    backbone: str,
    architecture: str = "simple_multi_sequence_fusion",
    use_ivd_encoding: bool = True,
    ivd_encoding_mode: str = "positional",
) -> str:
    """Convert backbone name to readable format for Simple Multi-Sequence Fusion."""
    ivd_suffix = ""
    if use_ivd_encoding:
        mode_short = "PE" if ivd_encoding_mode == "positional" else "LE"
        ivd_suffix = f" + IVD({mode_short})"

    if backbone.startswith("resnet"):
        num = backbone.replace("resnet", "")
        return f"Simple Multi-Seq Fusion (ResNet-{num}){ivd_suffix}"
    elif backbone.startswith("efficientnet"):
        variant = backbone.replace("efficientnet_", "").upper()
        return f"Simple Multi-Seq Fusion (EfficientNet-{variant}){ivd_suffix}"
    elif backbone.startswith("densenet"):
        num = backbone.replace("densenet", "")
        return f"Simple Multi-Seq Fusion (DenseNet-{num}){ivd_suffix}"
    elif backbone.startswith("vit"):
        if "base" in backbone:
            return f"Simple Multi-Seq Fusion (ViT-Base){ivd_suffix}"
        elif "large" in backbone:
            return f"Simple Multi-Seq Fusion (ViT-Large){ivd_suffix}"
        else:
            return f"Simple Multi-Seq Fusion (ViT){ivd_suffix}"
    else:
        return f"Simple Multi-Seq Fusion ({backbone}){ivd_suffix}"


def evaluate_single_model(
    backbone: str,
    checkpoint_name: str = "best_model.pth",
    device: torch.device = None,
    override_sequences: Optional[List[str]] = None,
) -> dict:
    """Evaluate a single backbone model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct path to backbone model directory
    run_dir = project_root / "outputs" / "pathology_model" / "runs" / backbone
    weights_dir = run_dir / "weights"
    checkpoint_path = weights_dir / checkpoint_name

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    try:
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model_config = checkpoint["model_config"]
        hyperparams = checkpoint.get("hyperparams", {})
        state_dict = checkpoint["model_state_dict"]

        # Extract base backbone name for model creation
        base_backbone = get_base_backbone_name(backbone)

        # Architecture will be determined later from checkpoint analysis
        # Temporarily set to simple_multi_sequence_fusion as default
        architecture = "simple_multi_sequence_fusion"

        # Detect IVD encoding settings from checkpoint
        if "multi_sequence" not in model_config:
            model_config["multi_sequence"] = {}

        # Get IVD encoding settings from multiple sources with proper fallback
        use_ivd_encoding = False  # default
        ivd_encoding_mode = "positional"  # default
        ivd_embedding_dim = 16  # default

        # First check checkpoint hyperparams for simple_multi_sequence_fusion config
        if "simple_multi_sequence_fusion" in hyperparams:
            simple_config = hyperparams["simple_multi_sequence_fusion"]
            use_ivd_encoding = simple_config.get("use_ivd_encoding", use_ivd_encoding)
            ivd_encoding_mode = simple_config.get(
                "ivd_encoding_mode", ivd_encoding_mode
            )
            ivd_embedding_dim = simple_config.get(
                "ivd_embedding_dim", ivd_embedding_dim
            )

        # Then check general multi_sequence config in hyperparams
        if "multi_sequence" in hyperparams:
            multi_config = hyperparams["multi_sequence"]
            use_ivd_encoding = multi_config.get("use_ivd_encoding", use_ivd_encoding)
            ivd_encoding_mode = multi_config.get("ivd_encoding_mode", ivd_encoding_mode)
            ivd_embedding_dim = multi_config.get("ivd_embedding_dim", ivd_embedding_dim)

        # Finally check model_config (lowest priority)
        if "use_ivd_encoding" not in model_config["multi_sequence"]:
            model_config["multi_sequence"]["use_ivd_encoding"] = use_ivd_encoding
        if "ivd_encoding_mode" not in model_config["multi_sequence"]:
            model_config["multi_sequence"]["ivd_encoding_mode"] = ivd_encoding_mode
        if "ivd_embedding_dim" not in model_config["multi_sequence"]:
            model_config["multi_sequence"]["ivd_embedding_dim"] = ivd_embedding_dim

        # Detect IVD usage from checkpoint state dict if not found in config
        has_ivd_embedding = any("ivd_embedding" in key for key in state_dict.keys())
        if has_ivd_embedding and not model_config["multi_sequence"].get(
            "use_ivd_encoding", False
        ):
            logger.info(
                "Detected IVD embedding in checkpoint state dict, enabling IVD encoding"
            )
            model_config["multi_sequence"]["use_ivd_encoding"] = True
            model_config["multi_sequence"]["ivd_embedding_dim"] = ivd_embedding_dim

        # Detect old architecture with projector layer
        has_projector = any("sequence_projections" in key for key in state_dict.keys())
        if has_projector:
            logger.warning("⚠️  ARCHITECTURE MISMATCH DETECTED ⚠️")
            logger.warning(
                "This checkpoint was trained with the OLD architecture (with projector layer)"
            )
            logger.warning(
                "Current code uses the NEW architecture (direct concatenation, no projector)"
            )
            logger.warning("")
            logger.warning(
                "The checkpoint is INCOMPATIBLE with the current model architecture."
            )
            logger.warning("You have two options:")
            logger.warning(
                "  1. Retrain the model with the new architecture (recommended)"
            )
            logger.warning(
                "  2. Checkout an older version of the code that had the projector layer"
            )
            logger.warning("")
            logger.warning(
                f"To retrain: python scripts/train_pathology_model.py --backbone {backbone}"
            )
            logger.warning("")
            return None

        # Set backbone in hyperparams - use base backbone name
        if "simple_multi_sequence_fusion" not in hyperparams:
            hyperparams["simple_multi_sequence_fusion"] = {}
        hyperparams["simple_multi_sequence_fusion"]["backbone_type"] = base_backbone

        # Architecture detection and model config update will happen below
        # after checkpoint analysis

        logger.info(f"{'=' * 80}")
        logger.info(f"Evaluating: {backbone} (base: {base_backbone})")
        logger.info(f"{'=' * 80}")

        # Check checkpoint compatibility with detailed analysis
        checkpoint_architecture = checkpoint.get("architecture", "unknown")
        checkpoint_backbone = checkpoint.get("backbone", "unknown")

        logger.info(
            f"Checkpoint info: architecture={checkpoint_architecture}, backbone={checkpoint_backbone}"
        )

        # Analyze checkpoint structure
        has_sequence_encoders = any(
            "sequence_encoders." in key for key in state_dict.keys()
        )
        has_sequence_projections = any(
            "sequence_projections." in key for key in state_dict.keys()
        )
        has_pathology_heads = any(
            "pathology_heads." in key for key in state_dict.keys()
        )
        has_old_backbone = any("backbone." in key for key in state_dict.keys())

        logger.info(
            f"Checkpoint structure: sequence_encoders={has_sequence_encoders}, sequence_projections={has_sequence_projections}, pathology_heads={has_pathology_heads}, old_backbone={has_old_backbone}"
        )

        # Detect actual architecture from checkpoint
        is_simple_multi_fusion = (
            has_sequence_encoders and has_sequence_projections and has_pathology_heads
        )
        is_old_single_sequence = has_old_backbone and not is_simple_multi_fusion

        # Determine which architecture to use
        actual_architecture = None
        if is_simple_multi_fusion:
            actual_architecture = "simple_multi_sequence_fusion"
            logger.info("Detected simple_multi_sequence_fusion architecture")
        elif is_old_single_sequence:
            # Try to use the old single sequence architecture
            logger.warning(
                "Detected old single-sequence architecture, attempting to evaluate with legacy mode"
            )
            actual_architecture = "legacy_single_sequence"
        else:
            # Try to determine from checkpoint metadata
            checkpoint_arch = checkpoint.get("architecture", "unknown")
            if checkpoint_arch in [
                "multi_sequence_fusion",
                "transformer_fusion",
                "adaptive_attention",
            ]:
                actual_architecture = checkpoint_arch
                logger.info(f"Using architecture from checkpoint: {checkpoint_arch}")
            else:
                logger.warning(
                    f"Unknown architecture for {backbone}, attempting simple_multi_sequence_fusion"
                )
                actual_architecture = "simple_multi_sequence_fusion"

        # Update model config based on detected architecture
        model_config["architecture"] = actual_architecture
        architecture = actual_architecture

        # Format display name based on detected architecture
        if actual_architecture == "simple_multi_sequence_fusion":
            display_name = format_model_name(
                base_backbone,
                architecture,
                model_config["multi_sequence"].get("use_ivd_encoding", False),
                model_config["multi_sequence"].get("ivd_encoding_mode", "positional"),
            )
        else:
            display_name = (
                f"{architecture.replace('_', ' ').title()} ({base_backbone.upper()})"
            )

        # Add suffix information to display name if present
        suffix = backbone.replace(base_backbone, "")
        if suffix:
            display_name += f" ({suffix.lstrip('_').upper()})"

        logger.info(f"Architecture: {architecture}")
        logger.info(f"Display Name: {display_name}")
        ivd_enabled = model_config.get("multi_sequence", {}).get(
            "use_ivd_encoding", False
        )
        logger.info(f"IVD Encoding: {'Enabled' if ivd_enabled else 'Disabled'}")
        if ivd_enabled:
            ivd_mode = model_config["multi_sequence"].get(
                "ivd_encoding_mode", "positional"
            )
            logger.info(f"IVD Encoding Mode: {ivd_mode}")
            if ivd_mode == "positional":
                logger.info(
                    f"IVD Embedding Dimension: {model_config['multi_sequence'].get('ivd_embedding_dim', 16)}"
                )

        # Check if this is a simple_multi_sequence_fusion architecture
        simple_multi_fusion = (
            model_config.get("architecture") == "simple_multi_sequence_fusion"
        )
        if simple_multi_fusion:
            # For simple multi-sequence fusion, check if backbone is specified in hyperparams
            simple_config = hyperparams.get("simple_multi_sequence_fusion", {})
            backbone_type = simple_config.get("backbone_type", base_backbone)
            logger.info(
                f"Using simple_multi_sequence_fusion with backbone: {backbone_type}"
            )

        # Create model using the same logic as training (disable wandb logging)
        model = create_model_with_verification(
            model_config, device, hyperparams=hyperparams, enable_wandb_logging=False
        )

        # Try to load state dict with comprehensive error handling
        load_success = False
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.success(
                f"Successfully loaded checkpoint for {backbone} (strict=True)"
            )
            load_success = True
        except RuntimeError as e:
            error_msg = str(e)

            # Check if it's an IVD encoding mismatch
            if "ivd_embedding" in error_msg.lower():
                current_ivd = model_config.get("multi_sequence", {}).get(
                    "use_ivd_encoding", False
                )
                checkpoint_ivd = has_ivd_embedding
                logger.warning("IVD encoding mismatch detected:")
                logger.warning(f"  Current model IVD encoding: {current_ivd}")
                logger.warning(f"  Checkpoint has IVD embedding: {checkpoint_ivd}")

                # Try to fix the mismatch by updating model config
                if checkpoint_ivd and not current_ivd:
                    logger.info(
                        "Attempting to enable IVD encoding to match checkpoint..."
                    )
                    model_config["multi_sequence"]["use_ivd_encoding"] = True
                    # Recreate model with correct IVD settings
                    model = create_model_with_verification(
                        model_config,
                        device,
                        hyperparams=hyperparams,
                        enable_wandb_logging=False,
                    )
                    try:
                        model.load_state_dict(state_dict, strict=True)
                        logger.success(
                            "Successfully loaded checkpoint after enabling IVD encoding"
                        )
                        load_success = True
                    except RuntimeError as retry_error:
                        logger.warning(
                            f"Still failed after enabling IVD encoding: {str(retry_error)[:200]}..."
                        )
                elif not checkpoint_ivd and current_ivd:
                    logger.info(
                        "Attempting to disable IVD encoding to match checkpoint..."
                    )
                    model_config["multi_sequence"]["use_ivd_encoding"] = False
                    # Recreate model with correct IVD settings
                    model = create_model_with_verification(
                        model_config,
                        device,
                        hyperparams=hyperparams,
                        enable_wandb_logging=False,
                    )
                    try:
                        model.load_state_dict(state_dict, strict=True)
                        logger.success(
                            "Successfully loaded checkpoint after disabling IVD encoding"
                        )
                        load_success = True
                    except RuntimeError as retry_error:
                        logger.warning(
                            f"Still failed after disabling IVD encoding: {str(retry_error)[:200]}..."
                        )

            # If not fixed yet, try non-strict loading
            if not load_success:
                # Check if the error is due to projector mismatch
                has_projector_in_checkpoint = any(
                    "sequence_projections" in key for key in state_dict.keys()
                )
                has_projector_in_model = any(
                    "sequence_projections" in key for key in model.state_dict().keys()
                )

                if has_projector_in_checkpoint and not has_projector_in_model:
                    logger.error(
                        "❌ CRITICAL: Checkpoint has projector layer, but current model doesn't"
                    )
                    logger.error(
                        "This checkpoint cannot be loaded with the new architecture."
                    )
                    logger.error(
                        f"Solution: Retrain using: python scripts/train_pathology_model.py --backbone {backbone}"
                    )
                    return None

                try:
                    missing_keys, unexpected_keys = model.load_state_dict(
                        state_dict, strict=False
                    )

                    # Filter out projector-related keys from unexpected_keys for clearer reporting
                    projector_keys = [
                        k for k in unexpected_keys if "sequence_projections" in k
                    ]
                    other_unexpected = [
                        k for k in unexpected_keys if "sequence_projections" not in k
                    ]

                    if projector_keys:
                        logger.warning(
                            f"⚠️  Found {len(projector_keys)} projector-related keys in checkpoint (old architecture)"
                        )
                        logger.warning(
                            "These keys will be ignored, but the model may not work correctly."
                        )

                    if missing_keys:
                        logger.warning(
                            f"Missing keys in checkpoint: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}"
                        )
                    if other_unexpected:
                        logger.warning(
                            f"Other unexpected keys: {other_unexpected[:3]}{'...' if len(other_unexpected) > 3 else ''}"
                        )

                    if projector_keys:
                        logger.error(
                            "❌ Partial loading with projector mismatch is not recommended."
                        )
                        logger.error(
                            "The model will NOT work correctly. Please retrain."
                        )
                        return None

                    logger.success(
                        f"Successfully loaded checkpoint for {backbone} (strict=False)"
                    )
                    load_success = True
                except Exception as strict_false_error:
                    logger.error(
                        f"Failed to load checkpoint for {backbone} even with strict=False"
                    )
                    logger.error(f"  Strict loading error: {str(e)[:200]}...")
                    logger.error(
                        f"  Non-strict loading error: {str(strict_false_error)[:200]}..."
                    )

                    # Provide detailed diagnostics
                    current_keys = set(model.state_dict().keys())
                    checkpoint_keys = set(state_dict.keys())

                    missing_in_current = checkpoint_keys - current_keys
                    missing_in_checkpoint = current_keys - checkpoint_keys

                    if missing_in_current:
                        logger.warning(
                            f"  Keys in checkpoint but not in current model ({len(missing_in_current)}): {list(missing_in_current)[:3]}{'...' if len(missing_in_current) > 3 else ''}"
                        )
                    if missing_in_checkpoint:
                        logger.warning(
                            f"  Keys in current model but not in checkpoint ({len(missing_in_checkpoint)}): {list(missing_in_checkpoint)[:3]}{'...' if len(missing_in_checkpoint) > 3 else ''}"
                        )

                    logger.warning(
                        "  Solution: Retrain the model with the current architecture using:"
                    )
                    logger.warning(
                        f"    python scripts/train_pathology_model.py --backbone {backbone}"
                    )

                    return None

        if not load_success:
            return None

        model = model.to(device)
        model.eval()

        # Load data config
        hyperparams = checkpoint.get("hyperparams", {})
        data_config = hyperparams.get(
            "data", {"input_size": [224, 224], "normalization": "imagenet"}
        )

        # Create test dataset
        manifest_path = Path(
            "../data/mri_phen/processed/pathology_training_manifest.csv"
        )
        test_transforms = get_default_transforms(
            mode="val",
            input_size=tuple(data_config["input_size"]),
            normalization=data_config["normalization"],
            in_channels=model_config.get("in_channels", 1),
        )

        # Determine dataset type based on architecture
        use_multi_sequence = actual_architecture in [
            "simple_multi_sequence_fusion",
            "multi_sequence_fusion",
            "transformer_fusion",
            "adaptive_attention",
        ]

        if use_multi_sequence:
            # Use multi-sequence dataset for modern architectures
            if not MULTI_SEQUENCE_DATASET_AVAILABLE:
                raise ImportError(
                    "Multi-sequence dataset is required but not available. Install required dependencies."
                )

            multi_sequence_config = hyperparams.get("multi_sequence", {})
            # Use override_sequences if provided, otherwise use config
            sequences = (
                override_sequences
                if override_sequences
                else multi_sequence_config.get(
                    "sequences", ["SAG_T2", "AX_T2", "SAG_STIR"]
                )
            )
            handle_missing = multi_sequence_config.get("handle_missing", "zero_pad")

            if override_sequences:
                logger.info(f"Overriding sequences with: {override_sequences}")
            else:
                logger.info(f"Using sequences from model config: {sequences}")

            # Use multi-sequence dataset
            test_dataset = MultiSequencePathologyDataset(
                manifest_path=manifest_path,
                project_root=project_root,
                split="test",
                transform=test_transforms,
                sequences=[s.upper() for s in sequences],  # Convert to uppercase
                handle_missing=handle_missing,
                return_binary=True,
            )
            logger.info(f"Using multi-sequence dataset with sequences: {sequences}")
        else:
            # For legacy architectures, try to use single-sequence dataset
            logger.warning(
                "Legacy architecture detected, attempting to use multi-sequence dataset anyway"
            )
            # Fall back to multi-sequence dataset but with warning
            if not MULTI_SEQUENCE_DATASET_AVAILABLE:
                raise ImportError(
                    "Multi-sequence dataset is required but not available. Install required dependencies."
                )

            multi_sequence_config = hyperparams.get("multi_sequence", {})
            # Use override_sequences if provided, otherwise use config
            sequences = (
                override_sequences
                if override_sequences
                else multi_sequence_config.get("sequences", ["SAG_T2"])
            )  # Use only one sequence for legacy
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
            logger.warning(
                f"Using multi-sequence dataset with limited sequences: {sequences}"
            )

        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        # Load optimal thresholds if available
        thresholds_path = run_dir / "thresholds.json"
        if thresholds_path.exists():
            optimal_thresholds = load_thresholds(thresholds_path)
            logger.success(f"Loaded optimal thresholds from: {thresholds_path}")
        else:
            optimal_thresholds = None
            logger.warning("No optimal thresholds found, using default 0.5")

        # Generate predictions
        label_names_list = label_names()
        y_true, y_pred, y_proba = generate_predictions(
            model,
            test_loader,
            device,
            "test",
            return_dataframe=False,
            label_names=label_names_list,
        )

        # Apply optimal thresholds if available
        if optimal_thresholds:
            y_pred = apply_thresholds(y_proba, optimal_thresholds, label_names_list)

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_proba, label_names_list)

        result = {
            "backbone": backbone,
            "base_backbone": base_backbone,
            "architecture": architecture,
            "display_name": display_name,
            "sequences_used": override_sequences if override_sequences else sequences,
            "val_f1": checkpoint.get("val_f1", 0),
            "epochs": checkpoint.get("epoch", 0) + 1,
            "overall_metrics": metrics,
            "per_label_metrics": {},
        }

        # Extract per-label metrics
        for label_name in label_names_list:
            result["per_label_metrics"][label_name] = {
                "precision": metrics[f"{label_name}/precision"],
                "recall": metrics[f"{label_name}/recall"],
                "f1": metrics[f"{label_name}/f1"],
                "roc_auc": metrics.get(f"{label_name}/roc_auc", 0),
                "auprc": 0,  # Would need to calculate separately
            }

        logger.success(
            f"{display_name}: Macro F1: {metrics['overall/macro_f1']:.4f} | "
            f"Subset Acc: {metrics['overall/subset_accuracy']:.4f} | "
            f"Val F1: {checkpoint.get('val_f1', 0):.4f}"
        )

        return result

    except Exception as e:
        logger.error(f"Error evaluating {run_dir.name}: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pathology classification model(s) using Simple Multi-Sequence Fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all trained backbone models
  python scripts/evaluate_pathology_model.py --all
  
  # Evaluate specific backbone model
  python scripts/evaluate_pathology_model.py --backbone resnet18
  
  # Evaluate with final checkpoint
  python scripts/evaluate_pathology_model.py --backbone densenet121 --checkpoint final_model.pth
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained backbone models in outputs/pathology_model/runs/",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Specific backbone to evaluate (e.g., resnet18, densenet121, efficientnet_b0)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pth",
        choices=["best_model.pth", "final_model.pth"],
        help="Checkpoint to use (default: best_model.pth)",
    )

    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Sequences to use for evaluation (e.g., --sequences AX_T2 SAG_STIR). Available: SAG_T2 SAG_T1 AX_T2 SAG_STIR. If not specified, uses saved model configuration.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: same as run_dir)",
    )

    parser.add_argument(
        "--analyze_checkpoints",
        action="store_true",
        help="Analyze checkpoint architecture compatibility without running evaluation",
    )

    # parser.add_argument(
    #     '--run-number',
    #     type=int,
    #     default=1,
    #     help='Run number to append to wandb project name (default: 1)'
    # )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.backbone:
        parser.print_help()
        logger.error("Must specify either --all or --backbone")
        return

    # Validate sequences if provided
    selected_sequences = validate_sequences(args.sequences)
    if selected_sequences:
        logger.info(f"Override sequences: {selected_sequences}")
    else:
        logger.info("Using sequences from saved model configuration")

    if args.backbone:
        valid_backbones = get_valid_backbones()
        base_backbone = get_base_backbone_name(args.backbone)
        if base_backbone not in valid_backbones:
            logger.error(
                f"Unknown backbone '{args.backbone}' (base: '{base_backbone}')"
            )
            logger.error("Supported backbones:")
            logger.error("  ResNet: resnet18, resnet34, resnet50")
            logger.error(
                "  EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2"
            )
            logger.error("  DenseNet: densenet121, densenet169, densenet201")
            logger.error(
                "  Vision Transformer: vit_base_patch16_224, vit_large_patch16_224"
            )
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.success(f"Using device: {device}")
    logger.success(f"Project root: {project_root}")

    if args.all:
        # Evaluate all backbone models
        runs_dir = project_root / "outputs" / "pathology_model" / "runs"

        if not runs_dir.exists():
            logger.warning("Runs directory does not exist")
            return

        # Get all backbone directories
        backbone_dirs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()])

        # Filter to only valid backbones (including those with suffixes like _sat)
        available_backbones = [b for b in backbone_dirs if is_valid_backbone_dir(b)]

        all_results = []
        label_names_list = label_names()

        logger.info(f"{'=' * 80}")
        logger.info("EVALUATING ALL AVAILABLE BACKBONE MODELS")
        logger.info(f"{'=' * 80}")

        for backbone in available_backbones:
            # Try to find checkpoint
            backbone_dir = runs_dir / backbone
            best_model_path = backbone_dir / "weights" / "best_model.pth"
            final_model_path = backbone_dir / "weights" / "final_model.pth"

            checkpoint_name = None
            if best_model_path.exists():
                checkpoint_name = "best_model.pth"
            elif final_model_path.exists():
                checkpoint_name = "final_model.pth"

            if checkpoint_name is None:
                logger.warning(f"  Skipping {backbone}: No model checkpoint found")
                continue

            logger.info(f"  Found checkpoint for {backbone}: {checkpoint_name}")

            # Evaluate
            result = evaluate_single_model(backbone, checkpoint_name, device)
            if result:
                all_results.append(result)

        logger.info(f"{'=' * 80}")
        logger.info(f"Evaluated {len(all_results)} models")
        logger.info(f"{'=' * 80}")

        if len(all_results) > 0:
            # Create comparison DataFrame
            comparison_data = []
            for result in all_results:
                overall = result["overall_metrics"]
                comparison_data.append(
                    {
                        "Backbone": result["backbone"],
                        "Display Name": result["display_name"],
                        "Val F1": result["val_f1"],
                        "Epochs": result["epochs"],
                        "Macro F1": overall["overall/macro_f1"],
                        "Weighted F1": overall.get(
                            "overall/weighted_f1", 0
                        ),  # Add Weighted F1
                        "Macro Precision": overall["overall/macro_precision"],
                        "Macro Recall": overall["overall/macro_recall"],
                        "Macro ROC-AUC": overall.get("overall/macro_roc_auc", 0),
                        "Macro AUPRC": overall.get("overall/macro_auprc", 0),
                        "Subset Accuracy": overall["overall/subset_accuracy"],
                        "Hamming Loss": overall["overall/hamming_loss"],
                    }
                )

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values("Macro F1", ascending=False)

            # Save comparison
            output_path = (
                project_root / "outputs" / "pathology_model" / "model_comparison.csv"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(output_path, index=False)

            logger.info(f"{'=' * 80}")
            logger.info("MODEL COMPARISON")
            logger.info(f"{'=' * 80}")
            logger.info(f"\n{comparison_df.to_string(index=False)}")
            logger.success(f"Comparison saved to: {output_path}")

            # Summary
            architectures_found = [result["display_name"] for result in all_results]
            logger.info("Architectures evaluated:")
            for arch in sorted(set(architectures_found)):
                count = architectures_found.count(arch)
                logger.info(f"  - {arch}: {count} model(s)")

            best_model = comparison_df.iloc[0]
            logger.success(
                f"Best Model: {best_model['Display Name']} (Backbone: {best_model['Backbone']})"
            )
            logger.success(f"  Macro F1: {best_model['Macro F1']:.4f}")
            logger.success(f"  Weighted F1: {best_model['Weighted F1']:.4f}")
            logger.success(f"  Subset Accuracy: {best_model['Subset Accuracy']:.4f}")

    elif args.backbone:
        # Evaluate single backbone model
        result = evaluate_single_model(
            args.backbone, args.checkpoint, device, selected_sequences
        )

        if result:
            logger.success(f"\n{'=' * 80}")
            logger.success("EVALUATION RESULTS")
            logger.success(f"{'=' * 80}")
            logger.info("\nOverall Metrics:")
            overall = result["overall_metrics"]
            logger.info(f"  Macro F1: {overall['overall/macro_f1']:.4f}")
            logger.info(
                f"  Weighted F1: {overall.get('overall/weighted_f1', 0):.4f}"
            )  # Add Weighted F1
            logger.info(
                f"  Macro ROC-AUC: {overall.get('overall/macro_roc_auc', 0):.4f}"
            )
            logger.info(f"  Macro AUPRC: {overall.get('overall/macro_auprc', 0):.4f}")
            logger.info(f"  Macro Precision: {overall['overall/macro_precision']:.4f}")
            logger.info(f"  Macro Recall: {overall['overall/macro_recall']:.4f}")
            logger.info(f"  Subset Accuracy: {overall['overall/subset_accuracy']:.4f}")
            logger.info(f"  Hamming Loss: {overall['overall/hamming_loss']:.4f}")

            logger.info("\nPer-Label Metrics:")

            # Create project name based on sequences used
            sequences_used = (
                selected_sequences
                if selected_sequences
                else result.get("sequences_used", ["SAG_T2", "AX_T2", "SAG_STIR"])
            )
            sequences_str = " and ".join(sequences_used)
            project_name = f"{sequences_str} NO IVD PAPER"
            run_name = f"Evaluation_{args.backbone}"

            with wandb.init(
                project=project_name,
                name=run_name,
            ) as run:
                my_table = wandb.Table(
                    columns=[
                        "Backbone",
                        "Macro F1",
                        "Weighted F1",
                        "Macro ROC-AUC",
                        "Macro AUPRC",
                        "Macro Precision",
                        "Macro Recall",
                        "Subset Accuracy",
                    ]
                )
                my_table.add_data(
                    args.backbone,
                    overall["overall/macro_f1"],
                    overall.get("overall/weighted_f1", 0),
                    overall.get("overall/macro_roc_auc", 0),
                    overall.get("overall/macro_auprc", 0),
                    overall["overall/macro_precision"],
                    overall["overall/macro_recall"],
                    overall["overall/subset_accuracy"],
                )
                wandb.log({"Evaluation Results": my_table})

            logger.success("Results saved to wandb")
            for label_name in label_names():
                label_metrics = result["per_label_metrics"][label_name]
                logger.info(f"  {label_name}:")
                logger.info(f"    Precision: {label_metrics['precision']:.4f}")
                logger.info(f"    Recall: {label_metrics['recall']:.4f}")
                logger.info(f"    F1: {label_metrics['f1']:.4f}")
                logger.info(f"    ROC-AUC: {label_metrics['roc_auc']:.4f}")
                logger.info(f"    AUPRC: {label_metrics.get('auprc', 0):.4f}")


if __name__ == "__main__":
    main()
