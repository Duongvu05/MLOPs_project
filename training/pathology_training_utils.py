"""
Training utilities for pathology classification models.

This module provides reusable functions for:
- Configuration loading and architecture override
- Wandb initialization with architecture-specific setup
- Model creation and verification
- Dataset creation
- Training and validation loops
- Metrics calculation
- Prediction generation
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score,
    average_precision_score,
)

from utils.dataset import get_default_transforms

try:
    from utils.multi_sequence_dataset import MultiSequencePathologyDataset

    MULTI_SEQUENCE_DATASET_AVAILABLE = True
except ImportError:
    MULTI_SEQUENCE_DATASET_AVAILABLE = False
    MultiSequencePathologyDataset = None
from utils.wandb_utils import init_wandb, log_metrics
from utils.threshold_utils import find_optimal_thresholds, save_thresholds
from models.pathology_model import create_pathology_model

# Check if multi-sequence model is available
try:
    from models.multi_sequence_fusion_model import MultiSequenceMultiScaleFusion

    MULTI_SEQUENCE_AVAILABLE = True
except ImportError:
    MULTI_SEQUENCE_AVAILABLE = False
    MultiSequenceMultiScaleFusion = None


def is_multi_sequence_model(model: nn.Module) -> bool:
    """Check if model is a multi-sequence fusion model."""
    from models.pathology_model import (
        MULTI_SEQUENCE_AVAILABLE,
        TRANSFORMER_FUSION_AVAILABLE,
    )

    try:
        from models.multi_sequence_fusion_model import MultiSequenceMultiScaleFusion
    except ImportError:
        MultiSequenceMultiScaleFusion = None
    try:
        from models.transformer_fusion_model import TransformerHierarchicalFusion
    except ImportError:
        TransformerHierarchicalFusion = None
    try:
        from models.pathology_model import SimpleMultiSequenceFusion
    except ImportError:
        SimpleMultiSequenceFusion = None

    # Check if it's a multi-sequence model
    is_multi_seq_fusion = MULTI_SEQUENCE_AVAILABLE and isinstance(
        model, MultiSequenceMultiScaleFusion
    )
    is_transformer_fusion = TRANSFORMER_FUSION_AVAILABLE and isinstance(
        model, TransformerHierarchicalFusion
    )
    is_simple_multi_seq_fusion = SimpleMultiSequenceFusion is not None and isinstance(
        model, SimpleMultiSequenceFusion
    )

    return is_multi_seq_fusion or is_transformer_fusion or is_simple_multi_seq_fusion


def prepare_model_input(
    batch: Dict,
    device: torch.device,
    is_multi_seq: bool,
    model: Optional[nn.Module] = None,
) -> Tuple:
    """
    Prepare model input from batch for multi-sequence models.

    BREAKING CHANGE: All models now use multi-sequence fusion. This function always
    expects multi-sequence input format.

    Args:
        batch: Data batch from dataloader (must have 'sequences' key)
        device: PyTorch device
        is_multi_seq: Whether model is multi-sequence (should always be True now)
        model: Optional model instance to get expected sequences from

    Returns:
        Tuple of (model_input, labels) where model_input is (sequences_dict, sequence_available_dict, ivd_levels)
    """
    labels = batch["labels"].to(device)

    # Extract IVD levels if available
    ivd_levels = None
    if "ivd_levels" in batch and batch["ivd_levels"] is not None:
        ivd_levels = batch["ivd_levels"].to(device)
    elif "ivd_label" in batch and batch["ivd_label"] is not None:
        # Convert from string to integer if needed
        if isinstance(batch["ivd_label"][0], str):
            ivd_levels = torch.tensor(
                [int(x) for x in batch["ivd_label"]], device=device
            )
        else:
            ivd_levels = batch["ivd_label"].to(device)

    # BREAKING CHANGE: All models now use multi-sequence
    # Always use multi-sequence format
    if is_multi_seq:
        # Multi-sequence model expects dict of sequences
        sequences = {}
        sequence_available = {}

        # Get expected sequences from model if available
        expected_sequences = None
        if model is not None and hasattr(model, "sequences"):
            expected_sequences = [seq.lower() for seq in model.sequences]

        if "sequences" in batch:
            # Multi-sequence dataset format
            batch_sequences = batch["sequences"]
            batch_available = batch.get("sequence_available", {})

            # Get all sequence names (from batch or model)
            if expected_sequences:
                all_seq_names = expected_sequences
            else:
                all_seq_names = list(
                    set(list(batch_sequences.keys()) + list(batch_available.keys()))
                )

            batch_size = labels.size(0)

            for seq_name in all_seq_names:
                if seq_name in batch_sequences:
                    seq_tensor = batch_sequences[seq_name]
                    if seq_tensor is not None:
                        sequences[seq_name] = seq_tensor.to(device)
                    else:
                        sequences[seq_name] = None
                else:
                    sequences[seq_name] = None

                # Set availability
                if seq_name in batch_available:
                    sequence_available[seq_name] = batch_available[seq_name].to(device)
                else:
                    # Default to available if sequence tensor exists, otherwise unavailable
                    if (
                        seq_name in batch_sequences
                        and batch_sequences[seq_name] is not None
                    ):
                        sequence_available[seq_name] = torch.ones(
                            batch_size, device=device, dtype=torch.bool
                        )
                    else:
                        sequence_available[seq_name] = torch.zeros(
                            batch_size, device=device, dtype=torch.bool
                        )
        else:
            # This should not happen - all datasets now use multi-sequence format
            raise ValueError(
                "Batch must contain 'sequences' key. "
                "All datasets now use MultiSequencePathologyDataset which provides 'sequences' in the batch."
            )

        model_input = (
            sequences,
            sequence_available if sequence_available else None,
            ivd_levels,
        )
    else:
        # This should not happen - all models now use multi-sequence
        # But keep as fallback for error messages
        raise ValueError(
            "All models now use multi-sequence fusion. "
            "Batch must contain 'sequences' key. "
            "If you're loading an old checkpoint, it's incompatible with the new architecture."
        )

    return model_input, labels


def load_config_and_override_architecture(
    config_path: Path,
    target_architecture: str,
    architecture_options: List[str],
    experiment_prefix: str = "pathology_multilabel",
) -> Tuple[Dict, Dict, str, str]:
    """
    Load configuration and override architecture to ensure correct model is used.

    Args:
        config_path: Path to wandb_config.yaml
        target_architecture: Target architecture name (e.g., 'vit_base_patch16_224')
        architecture_options: List of valid architecture options
        experiment_prefix: Prefix for experiment name

    Returns:
        Tuple of (config, hyperparams, model_config, experiment_name)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    hyperparams = config["hyperparameters"]
    model_config = hyperparams["model"]

    # Override architecture
    original_arch = model_config.get("architecture", "unknown")
    if not model_config.get("architecture", "").startswith(
        target_architecture.split("_")[0]
    ):
        # Use first option as default if target not in options
        default_arch = (
            architecture_options[0] if architecture_options else target_architecture
        )
        model_config["architecture"] = default_arch
        print(
            f"⚠ Overriding architecture from '{original_arch}' to {target_architecture.split('_')[0]}: {model_config['architecture']}"
        )
    else:
        print(
            f"✓ Using {target_architecture.split('_')[0]} architecture: {model_config['architecture']}"
        )

    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{experiment_prefix}_{model_config['architecture']}_{timestamp}"

    return config, hyperparams, model_config, experiment_name, timestamp


def initialize_wandb_for_training(
    config: Dict,
    hyperparams: Dict,
    experiment_name: str,
    model_config: Dict,
    architecture_family: str,
    device: torch.device,
    project_root: Path,
) -> Any:
    """
    Initialize wandb for training with architecture-specific tags.

    Args:
        config: Full config dictionary
        hyperparams: Hyperparameters dictionary
        experiment_name: Experiment name
        model_config: Model configuration dictionary
        architecture_family: Architecture family name (e.g., 'vit', 'densenet')
        device: PyTorch device
        project_root: Project root path

    Returns:
        wandb run object
    """
    wandb_settings = {
        "project_name": config["project"]["name"],
        "experiment_name": experiment_name,
        "entity": config["project"].get("entity"),
        "tags": config["run"].get("tags", [])
        + [
            "pathology",
            "multi-label",
            architecture_family,
            model_config["architecture"],
        ],
        "notes": config["run"].get(
            "notes", "Multi-label pathology classification training"
        ),
        "mode": config["run"].get("mode", "online"),
    }

    # Combine all hyperparameters for wandb
    wandb_config = {
        **hyperparams,
        "config_file": str(project_root / "configs" / "wandb_config.yaml"),
        "device": str(device),
    }

    # Initialize wandb run
    run = init_wandb(
        project_name=wandb_settings["project_name"],
        experiment_name=wandb_settings["experiment_name"],
        config=wandb_config,
        entity=wandb_settings["entity"],
        tags=wandb_settings["tags"],
        notes=wandb_settings["notes"],
        mode=wandb_settings["mode"],
    )

    # Log configuration file
    if config.get("artifacts", {}).get("log_configs", True):
        from utils.wandb_utils import log_config_file

        log_config_file(str(project_root / "configs" / "wandb_config.yaml"))

    print(f"✓ Wandb initialized: {run.url}")
    print(f"✓ Experiment: {experiment_name}")

    return run


def create_pathology_datasets(
    manifest_path: Path,
    project_root: Path,
    data_config: Dict,
    multi_sequence_config: Optional[Dict] = None,
) -> Tuple:
    """
    Create train, validation, and test datasets.

    BREAKING CHANGE: Always uses multi-sequence datasets. All models now require multi-sequence input.

    Args:
        manifest_path: Path to training manifest CSV
        project_root: Project root directory
        data_config: Data configuration dictionary
        multi_sequence_config: Configuration for multi-sequence dataset

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if not MULTI_SEQUENCE_DATASET_AVAILABLE:
        raise ImportError(
            "Multi-sequence dataset is required but not available. Install required dependencies."
        )

    train_transforms = get_default_transforms(
        mode="train",
        input_size=tuple(data_config["input_size"]),
        normalization=data_config["normalization"],
        in_channels=data_config.get("in_channels", 1),
        augmentation_strength=data_config.get("augmentation_strength", "medium"),
    )
    val_test_transforms = get_default_transforms(
        mode="val",
        input_size=tuple(data_config["input_size"]),
        normalization=data_config["normalization"],
        in_channels=data_config.get("in_channels", 1),
    )

    # Always use multi-sequence dataset
    DatasetClass = MultiSequencePathologyDataset
    multi_seq_config = multi_sequence_config or {}
    sequences = multi_seq_config.get(
        "sequences", ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]
    )
    handle_missing = multi_seq_config.get("handle_missing", "zero_pad")

    # Create multi-sequence datasets
    train_dataset = DatasetClass(
        manifest_path=manifest_path,
        project_root=project_root,
        split="train",
        transform=train_transforms,
        sequences=sequences,
        handle_missing=handle_missing,
    )

    val_dataset = DatasetClass(
        manifest_path=manifest_path,
        project_root=project_root,
        split="val",
        transform=val_test_transforms,
        sequences=sequences,
        handle_missing=handle_missing,
    )

    test_dataset = DatasetClass(
        manifest_path=manifest_path,
        project_root=project_root,
        split="test",
        transform=val_test_transforms,
        sequences=sequences,
        handle_missing=handle_missing,
    )

    # Print sequence statistics
    print("\nSequence Availability Statistics (Train Set):")
    train_stats = train_dataset.get_sequence_statistics()
    for seq_name, stats in train_stats.items():
        print(
            f"  {seq_name}: {stats['available']}/{len(train_dataset)} ({stats['available_pct']:.1f}%)"
        )

    print(f"\n✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def create_model_with_verification(
    model_config: Dict,
    device: torch.device,
    hyperparams: Optional[Dict] = None,
    enable_wandb_logging: bool = True,
) -> nn.Module:
    """
    Create model and verify architecture is correct.

    Args:
        model_config: Model configuration dictionary
        device: PyTorch device
        hyperparams: Optional hyperparameters dictionary (for extracting model-specific kwargs)
        enable_wandb_logging: Whether to log metrics to wandb (set False during evaluation)

    Returns:
        Model instance
    """
    print(f"\n{'=' * 80}")
    print(f"Creating model with architecture: {model_config['architecture']}")
    print(f"{'=' * 80}")

    # Build model kwargs based on architecture
    # BREAKING CHANGE: All architectures now use multi-sequence fusion
    model_kwargs = {}

    # All architectures (resnet, densenet, efficientnet, vit, multi_sequence_fusion) use multi-sequence
    if (
        model_config["architecture"].startswith("resnet")
        or model_config["architecture"].startswith("densenet")
        or model_config["architecture"].startswith("efficientnet")
        or model_config["architecture"].startswith("vit")
        or model_config["architecture"] == "multi_sequence_fusion"
    ):
        # Extract multi-sequence fusion parameters
        multi_seq_config = (hyperparams or {}).get("multi_sequence", {})
        attention_config = multi_seq_config.get("attention", {})
        model_kwargs = {
            "sequences": [
                s.lower()
                for s in multi_seq_config.get(
                    "sequences", ["SAG_T2", "AX_T2", "SAG_STIR"]
                )
            ],
            "feature_dims": multi_seq_config.get("multi_scale", {}).get(
                "feature_dims", [256, 512, 1024]
            ),
            "fusion_dim": multi_seq_config.get("fusion", {}).get("fusion_dim", 512),
            "attention_heads": attention_config.get("num_heads", 8),
            "attention_dropout": attention_config.get("dropout", 0.1),
            "head_hidden_dim": multi_seq_config.get("fusion", {}).get(
                "head_hidden_dim", 256
            ),
            "head_dropout": multi_seq_config.get("fusion", {}).get("head_dropout", 0.3),
            # Enhanced parameters
            "use_pathology_specific_attention": attention_config.get(
                "use_pathology_specific_attention", False
            ),
            "use_clinical_priors": attention_config.get("use_clinical_priors", True),
            "prior_strength": attention_config.get("prior_strength", 0.3),
        }
        # For explicit multi_sequence_fusion, allow backbone_type override
        if model_config["architecture"] == "multi_sequence_fusion":
            model_kwargs["backbone_type"] = multi_seq_config.get(
                "backbone_type", "densenet121"
            )
    elif model_config["architecture"] == "transformer_fusion":
        # Extract transformer fusion parameters
        transformer_config = (hyperparams or {}).get("transformer_fusion", {})
        model_kwargs = {
            "sequences": [
                s.lower()
                for s in transformer_config.get(
                    "sequences", ["SAG_T2", "AX_T2", "SAG_STIR"]
                )
            ],
            "feature_dim": transformer_config.get("feature_dim", 512),
            "num_transformer_layers": transformer_config.get(
                "num_transformer_layers", 2
            ),
            "num_attention_heads": transformer_config.get("attention", {}).get(
                "num_heads", 8
            ),
            "attention_dropout": transformer_config.get("attention", {}).get(
                "dropout", 0.1
            ),
            "head_hidden_dim": transformer_config.get("heads", {}).get(
                "hidden_dim", 256
            ),
            "head_dropout": transformer_config.get("heads", {}).get("dropout", 0.3),
        }
    elif model_config["architecture"] == "adaptive_attention":
        # Extract adaptive attention parameters
        adaptive_config = (hyperparams or {}).get("adaptive_attention", {})
        model_kwargs = {
            "backbone": adaptive_config.get("backbone", "efficientnet_b0"),
            "attention_dim": adaptive_config.get("attention_dim", 256),
            "num_attention_heads": adaptive_config.get("num_attention_heads", 8),
            "attention_dropout": adaptive_config.get("attention_dropout", 0.1),
            "fusion_method": adaptive_config.get("fusion_method", "adaptive"),
        }
    elif model_config["architecture"] == "simple_multi_sequence_fusion":
        # Extract simple multi-sequence fusion parameters
        simple_config = (hyperparams or {}).get("simple_multi_sequence_fusion", {})
        # Also check multi_sequence config as fallback
        multi_seq_config = (hyperparams or {}).get("multi_sequence", {})

        model_kwargs = {
            "sequences": [
                s.lower()
                for s in simple_config.get("sequences", ["sag_t2", "ax_t2", "sag_stir"])
            ],
            "backbone_type": simple_config.get("backbone_type", "densenet121"),
            "projection_dim": simple_config.get("projection_dim", 10),
            "hidden_dim": simple_config.get("hidden_dim", 128),
            "use_ivd_encoding": simple_config.get(
                "use_ivd_encoding", multi_seq_config.get("use_ivd_encoding", True)
            ),
            "ivd_encoding_mode": simple_config.get(
                "ivd_encoding_mode",
                multi_seq_config.get("ivd_encoding_mode", "positional"),
            ),
            # dropout_rate will be passed separately to avoid conflict
        }

        # Determine dropout_rate for simple_multi_sequence_fusion
        dropout_rate = model_config.get(
            "dropout_rate", simple_config.get("dropout_rate", 0.3)
        )

        model = create_pathology_model(
            architecture=model_config["architecture"],
            num_labels=model_config["num_labels"],
            pretrained=model_config.get("pretrained", True),
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=dropout_rate,
            **model_kwargs,
        )
    else:
        model = create_pathology_model(
            architecture=model_config["architecture"],
            num_labels=model_config["num_labels"],
            pretrained=model_config.get("pretrained", True),
            in_channels=model_config.get("in_channels", 1),
            dropout_rate=model_config.get("dropout_rate", 0.5),
            **model_kwargs,
        )

    model = model.to(device)

    # Verify model architecture
    print(f"\n✓ Model created: {model_config['architecture']}")
    print(f"✓ Model type: {type(model).__name__}")
    if hasattr(model, "architecture"):
        print(f"✓ Model architecture attribute: {model.architecture}")
        # Check if architecture matches expected family
        arch_family = model_config["architecture"].split("_")[0]
        if arch_family not in model.architecture.lower():
            print(
                f"⚠ WARNING: Model architecture '{model.architecture}' may not match expected family '{arch_family}'!"
            )
    else:
        print("⚠ Model does not have 'architecture' attribute")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"{'=' * 80}\n")

    # Log model info to wandb (only if enabled)
    if enable_wandb_logging:
        log_metrics(
            {
                "model/architecture": model_config["architecture"],
                "model/num_params": sum(p.numel() for p in model.parameters()),
                "model/trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )

    return model


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, label_names: List[str]
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-label classification.

    Args:
        y_true: True labels [N, num_labels]
        y_pred: Predicted labels [N, num_labels]
        y_proba: Predicted probabilities [N, num_labels]
        label_names: List of label names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Per-label metrics
    for i, label_name in enumerate(label_names):
        metrics[f"{label_name}/precision"] = precision_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )
        metrics[f"{label_name}/recall"] = recall_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )
        metrics[f"{label_name}/f1"] = f1_score(
            y_true[:, i], y_pred[:, i], zero_division=0
        )

        # ROC-AUC and AUPRC (if we have enough positive samples)
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                metrics[f"{label_name}/roc_auc"] = roc_auc_score(
                    y_true[:, i], y_proba[:, i]
                )
            except:
                metrics[f"{label_name}/roc_auc"] = 0.0

            # Add AUPRC (Area Under Precision-Recall Curve)
            try:
                metrics[f"{label_name}/auprc"] = average_precision_score(
                    y_true[:, i], y_proba[:, i]
                )
            except:
                metrics[f"{label_name}/auprc"] = 0.0

    # Overall metrics
    metrics["overall/hamming_loss"] = hamming_loss(y_true, y_pred)

    # Subset accuracy (exact match across all labels)
    subset_acc = accuracy_score(y_true, y_pred)
    metrics["overall/subset_accuracy"] = subset_acc

    # Debug info for subset accuracy
    exact_matches = np.all(y_true == y_pred, axis=1).sum()
    total_samples = len(y_true)
    # print(f"Debug: Exact matches: {exact_matches}/{total_samples} = {exact_matches/total_samples:.4f}")  # Uncomment for debugging

    # Macro-averaged metrics
    metrics["overall/macro_precision"] = np.mean(
        [metrics[f"{ln}/precision"] for ln in label_names]
    )
    metrics["overall/macro_recall"] = np.mean(
        [metrics[f"{ln}/recall"] for ln in label_names]
    )
    metrics["overall/macro_f1"] = np.mean([metrics[f"{ln}/f1"] for ln in label_names])

    # Macro ROC-AUC (average of per-label ROC-AUC)
    roc_aucs = [metrics.get(f"{ln}/roc_auc", 0.0) for ln in label_names]
    metrics["overall/macro_roc_auc"] = np.mean(roc_aucs)

    # Macro AUPRC (average of per-label AUPRC)
    auprcs = [metrics.get(f"{ln}/auprc", 0.0) for ln in label_names]
    metrics["overall/macro_auprc"] = np.mean(auprcs)

    # Micro-averaged metrics
    metrics["overall/micro_precision"] = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["overall/micro_recall"] = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["overall/micro_f1"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    # Weighted-averaged metrics
    metrics["overall/weighted_precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["overall/weighted_recall"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["overall/weighted_f1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    label_names: List[str],
    gradient_clip: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: PyTorch device
        epoch: Current epoch number
        label_names: List of label names
        gradient_clip: Gradient clipping value (0.0 to disable)

    Returns:
        Tuple of (epoch_loss, metrics_dict)
    """
    model.train()
    running_loss = 0.0

    all_preds = []
    all_probs = []
    all_targets = []

    # Check if model is multi-sequence
    is_multi_seq = is_multi_sequence_model(model)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Prepare model input (multi-sequence format with IVD levels)
        model_input, labels = prepare_model_input(
            batch, device, is_multi_seq, model=model
        )

        # Forward pass
        optimizer.zero_grad()
        if is_multi_seq:
            sequences, sequence_available, ivd_levels = model_input
            # Check if model supports IVD encoding
            from models.pathology_model import SimpleMultiSequenceFusion

            if isinstance(model, SimpleMultiSequenceFusion):
                # SimpleMultiSequenceFusion supports IVD levels
                output = model(sequences, sequence_available, ivd_levels)
            else:
                # Other multi-sequence models support return_attention_weights
                output = model(
                    sequences,
                    ivd_levels,
                    sequence_available,
                    return_attention_weights=False,
                )
            # Handle tuple return (logits, attention_weights) or just logits
            logits = output[0] if isinstance(output, tuple) else output
        else:
            output = model(model_input)
            logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Statistics
        running_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_preds.append(preds)
        all_probs.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Log batch metrics to wandb (every 10 batches)
        # Note: Using None for step to let wandb handle step management automatically
        if batch_idx % 10 == 0:
            log_metrics(
                {
                    "train/batch_loss": loss.item(),
                },
                step=None,
            )

    # Calculate epoch metrics
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    epoch_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(all_targets, all_preds, all_probs, label_names)

    return epoch_loss, metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    label_names: List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: PyTorch device
        epoch: Current epoch number
        label_names: List of label names

    Returns:
        Tuple of (epoch_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_probs = []
    all_targets = []

    # Check if model is multi-sequence
    is_multi_seq = is_multi_sequence_model(model)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")
        for batch in pbar:
            # Prepare model input (multi-sequence format with IVD levels)
            model_input, labels = prepare_model_input(
                batch, device, is_multi_seq, model=model
            )

            # Forward pass
            if is_multi_seq:
                sequences, sequence_available, ivd_levels = model_input
                # Check if model supports IVD encoding
                from models.pathology_model import SimpleMultiSequenceFusion

                if isinstance(model, SimpleMultiSequenceFusion):
                    # SimpleMultiSequenceFusion supports IVD levels
                    output = model(sequences, sequence_available, ivd_levels)
                else:
                    # Other multi-sequence models support return_attention_weights
                    output = model(
                        sequences,
                        ivd_levels,
                        sequence_available,
                        return_attention_weights=False,
                    )
                logits = output[0] if isinstance(output, tuple) else output
            else:
                output = model(model_input)
                logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, labels)

            running_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate epoch metrics
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    epoch_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(all_targets, all_preds, all_probs, label_names)

    return epoch_loss, metrics


def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str = "dataset",
    return_dataframe: bool = True,
    label_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: PyTorch device
        dataset_name: Name of the dataset (for logging)
        return_dataframe: If True, return DataFrame with metadata (default: True)
        label_names: List of label names (required if return_dataframe=True)

    Returns:
        If return_dataframe=True: pandas DataFrame with columns:
            - patient_id, ivd_label
            - true_{label_name} for each label
            - pred_{label_name} for each label
            - prob_{label_name} for each label
        If return_dataframe=False: Tuple of (true_labels, predicted_labels, predicted_probabilities) as numpy arrays
    """
    import pandas as pd

    if return_dataframe and label_names is None:
        from utils.label_utils import label_names as get_label_names

        label_names = get_label_names()

    model.eval()

    # Check if model is multi-sequence
    is_multi_seq = is_multi_sequence_model(model)

    all_patient_ids = []
    all_ivd_labels = []
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Generating {dataset_name} predictions")
        for batch in pbar:
            # Prepare model input (multi-sequence format with IVD levels)
            model_input, labels = prepare_model_input(
                batch, device, is_multi_seq, model=model
            )

            # Forward pass
            if is_multi_seq:
                sequences, sequence_available, ivd_levels = model_input
                # Check if model supports IVD encoding
                from models.pathology_model import SimpleMultiSequenceFusion

                if isinstance(model, SimpleMultiSequenceFusion):
                    # SimpleMultiSequenceFusion supports IVD levels
                    output = model(sequences, sequence_available, ivd_levels)
                else:
                    # Other multi-sequence models support return_attention_weights
                    import inspect

                    forward_signature = inspect.signature(model.forward)
                    if "return_attention_weights" in forward_signature.parameters:
                        output = model(
                            sequences,
                            ivd_levels,
                            sequence_available,
                            return_attention_weights=False,
                        )
                    else:
                        output = model(sequences, ivd_levels, sequence_available)
                logits = output[0] if isinstance(output, tuple) else output
            else:
                output = model(model_input)
                logits = output[0] if isinstance(output, tuple) else output

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_patient_ids.extend(batch["patient_id"])
            all_ivd_labels.extend(batch["ivd_label"])
            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

    # Combine results
    all_targets = np.vstack(all_targets)
    all_pred_probs = np.vstack(all_probs)
    all_pred_labels = np.vstack(all_preds)

    if return_dataframe:
        # Create DataFrame
        data = {
            "patient_id": all_patient_ids,
            "ivd_label": all_ivd_labels,
        }

        # Add true labels
        for i, label_name in enumerate(label_names):
            data[f"true_{label_name}"] = all_targets[:, i]

        # Add predicted labels
        for i, label_name in enumerate(label_names):
            data[f"pred_{label_name}"] = all_pred_labels[:, i]

        # Add predicted probabilities
        for i, label_name in enumerate(label_names):
            data[f"pred_prob_{label_name}"] = all_pred_probs[:, i]

        df = pd.DataFrame(data)
        return df
    else:
        return all_targets, all_pred_labels, all_pred_probs


def optimize_and_save_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    label_names: List[str],
    weights_dir: Path,
    thresholds_config: Dict,
    thresholds_save_path: Path,
    best_epoch: int,
    best_val_f1: float,
) -> Optional[Dict[str, float]]:
    """
    Optimize thresholds on validation set and save them.

    Args:
        model: Trained model
        val_loader: Validation dataloader
        device: PyTorch device
        label_names: List of label names
        weights_dir: Directory to save thresholds
        thresholds_config: Threshold optimization configuration
        best_epoch: Best epoch number
        best_val_f1: Best validation F1 score
        timestamp: Experiment timestamp
        project_root: Project root directory

    Returns:
        Dictionary of optimal thresholds or None if optimization disabled
    """
    optimize_thresholds = thresholds_config.get("optimize", True)

    if not optimize_thresholds:
        print("Threshold optimization disabled in config")
        return None

    print("=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)

    model.eval()

    # Check if model is multi-sequence
    is_multi_seq = is_multi_sequence_model(model)

    # Generate predictions on validation set
    val_true_labels = []
    val_pred_probs = []

    with torch.no_grad():
        for batch in val_loader:
            # Prepare model input (multi-sequence format)
            model_input, labels = prepare_model_input(
                batch, device, is_multi_seq, model=model
            )

            # Forward pass
            if is_multi_seq:
                sequences, sequence_available, ivd_levels = model_input
                # Check if model supports IVD encoding
                from models.pathology_model import SimpleMultiSequenceFusion

                if isinstance(model, SimpleMultiSequenceFusion):
                    # SimpleMultiSequenceFusion supports IVD levels and doesn't have return_attention_weights
                    if (
                        hasattr(model, "use_ivd_encoding")
                        and model.use_ivd_encoding
                        and ivd_levels is not None
                    ):
                        output = model(sequences, sequence_available, ivd_levels)
                    else:
                        output = model(sequences, sequence_available)
                else:
                    # Other multi-sequence models
                    if ivd_levels is not None:
                        output = model(
                            sequences,
                            ivd_levels,
                            sequence_available,
                            return_attention_weights=False,
                        )
                    else:
                        output = model(
                            sequences,
                            sequence_available,
                            return_attention_weights=False,
                        )
                logits = output[0] if isinstance(output, tuple) else output
            else:
                output = model(model_input)
                logits = output[0] if isinstance(output, tuple) else output

            probs = torch.sigmoid(logits).cpu().numpy()

            val_true_labels.append(labels.cpu().numpy())
            val_pred_probs.append(probs)

    val_true_labels = np.vstack(val_true_labels)
    val_pred_probs = np.vstack(val_pred_probs)

    # Find optimal thresholds
    metric = thresholds_config.get("metric", "f1")
    optimal_thresholds, threshold_metrics = find_optimal_thresholds(
        val_true_labels, val_pred_probs, label_names, metric=metric, return_metrics=True
    )

    print(f"\nOptimal thresholds (metric: {metric}):")
    for label_name, threshold in optimal_thresholds.items():
        metrics = threshold_metrics[label_name]
        print(
            f"  {label_name:25s}: {threshold:.3f} (F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f})"
        )

    # Save thresholds
    thresholds_save_path = thresholds_save_path.resolve()

    save_thresholds(
        optimal_thresholds,
        thresholds_save_path,
        metadata={
            "metric": metric,
            "threshold_metrics": threshold_metrics,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
        },
    )

    # Log to wandb (let wandb handle step management automatically)
    log_metrics({"thresholds": optimal_thresholds}, step=None)

    print(f"\n✓ Thresholds saved to: {thresholds_save_path}")

    return optimal_thresholds
