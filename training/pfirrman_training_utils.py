"""
Training utilities for Pfirrmann grade classification models.

This module provides reusable functions for:
- Configuration loading and architecture override
- Wandb initialization with architecture-specific setup
- Model creation and verification
- Dataset creation (multi-sequence only)
- Training and validation loops (multi-class)
- Metrics calculation (accuracy, per-class F1, confusion matrix)
- Prediction generation
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

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
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from utils.dataset import get_default_transforms

try:
    from utils.pfirrman_dataset import MultiSequencePfirrmannDataset

    PFIRRMAN_DATASET_AVAILABLE = True
except ImportError:
    PFIRRMAN_DATASET_AVAILABLE = False
    MultiSequencePfirrmannDataset = None
from utils.wandb_utils import init_wandb, log_metrics
from models.pfirrman_model import create_pfirrman_model

# Check if multi-sequence model is available
try:
    from models.multi_sequence_fusion_model import MultiSequenceMultiScaleFusion

    MULTI_SEQUENCE_AVAILABLE = True
except ImportError:
    MULTI_SEQUENCE_AVAILABLE = False
    MultiSequenceMultiScaleFusion = None


def is_pfirrman_multi_sequence_model(model: nn.Module) -> bool:
    """Check if model is a Pfirrmann multi-sequence fusion model."""
    from models.pfirrman_model import MultiSequencePfirrmannFusion

    return isinstance(model, MultiSequencePfirrmannFusion)


def prepare_pfirrman_model_input(
    batch: Dict, device: torch.device, model: Optional[nn.Module] = None
) -> Tuple:
    """
    Prepare model input from batch for Pfirrmann multi-sequence models.

    Args:
        batch: Data batch from dataloader (must have 'sequences' key)
        device: PyTorch device
        model: Optional model instance to get expected sequences from

    Returns:
        Tuple of (model_input, labels) where model_input is (sequences_dict, sequence_available_dict)
    """
    # Get grade labels (already 0-indexed: 0-4 for grades 1-5)
    labels = batch["grade"].to(device)

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
        raise ValueError(
            "Batch must contain 'sequences' key. "
            "Pfirrmann datasets use MultiSequencePfirrmannDataset which provides 'sequences' in the batch."
        )

    model_input = (sequences, sequence_available if sequence_available else None)

    return model_input, labels


def load_config_and_override_architecture(
    config_path: Path,
    target_architecture: str,
    architecture_options: List[str],
    experiment_prefix: str = "pfirrman_multiclass",
) -> Tuple[Dict, Dict, str, str]:
    """
    Load configuration and override architecture to ensure correct model is used.

    Args:
        config_path: Path to wandb_config.yaml
        target_architecture: Target architecture name (e.g., 'vit_base_patch16_224')
        architecture_options: List of valid architecture options
        experiment_prefix: Prefix for experiment name

    Returns:
        Tuple of (config, hyperparams, model_config, experiment_name, timestamp)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get Pfirrmann-specific config or use default
    if "pfirrman" in config.get("hyperparameters", {}):
        hyperparams = config["hyperparameters"]["pfirrman"]
    else:
        # Fallback to general hyperparameters
        hyperparams = config.get("hyperparameters", {})

    model_config = hyperparams.get("model", {})

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
            "pfirrman",
            "multi-class",
            architecture_family,
            model_config["architecture"],
        ],
        "notes": config["run"].get("notes", "Pfirrmann grade classification training"),
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


def create_pfirrman_datasets(
    manifest_path: Path,
    project_root: Path,
    data_config: Dict,
    multi_sequence_config: Optional[Dict] = None,
) -> Tuple:
    """
    Create train, validation, and test datasets for Pfirrmann classification.

    Always uses multi-sequence datasets.

    Args:
        manifest_path: Path to pfirrman_training_manifest.csv
        project_root: Project root directory
        data_config: Data configuration dictionary
        multi_sequence_config: Configuration for multi-sequence dataset

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if not PFIRRMAN_DATASET_AVAILABLE:
        raise ImportError(
            "Pfirrmann multi-sequence dataset is required but not available. Check imports."
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
    DatasetClass = MultiSequencePfirrmannDataset
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

    # Print grade distribution
    print("\nPfirrmann Grade Distribution (Train Set):")
    grade_dist = train_dataset.get_grade_distribution()
    for grade in range(1, 6):
        count = grade_dist.get(grade, 0)
        pct = (count / len(train_dataset) * 100) if len(train_dataset) > 0 else 0
        print(f"  Grade {grade}: {count} ({pct:.1f}%)")

    print(f"\n✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def compute_class_weights(
    dataset: MultiSequencePfirrmannDataset, smoothing: float = 0.1
) -> torch.Tensor:
    """
    Compute class weights for Pfirrmann grades using inverse frequency.

    Args:
        dataset: Pfirrmann dataset
        smoothing: Smoothing factor to prevent extreme weights (0.0 to 1.0)

    Returns:
        Class weights tensor of shape [5] (for grades 1-5, indexed 0-4)
    """
    grade_dist = dataset.get_grade_distribution()
    total_samples = len(dataset)

    # Count samples per class (0-indexed: 0-4 for grades 1-5)
    class_counts = torch.zeros(5, dtype=torch.float32)
    for grade in range(1, 6):
        class_counts[grade - 1] = float(grade_dist.get(grade, 0))

    # Compute inverse frequency weights
    # Add smoothing to prevent division by zero and extreme weights
    class_weights = total_samples / (
        5.0 * (class_counts + smoothing * total_samples / 5.0)
    )

    # Normalize weights (optional: make mean = 1.0)
    class_weights = class_weights / class_weights.mean()

    return class_weights


def create_pfirrman_model_with_verification(
    model_config: Dict, device: torch.device, hyperparams: Optional[Dict] = None
) -> nn.Module:
    """
    Create Pfirrmann model with verification.

    Args:
        model_config: Model configuration dictionary
        device: PyTorch device
        hyperparams: Optional hyperparameters dictionary

    Returns:
        Pfirrmann model
    """
    # Extract multi-sequence fusion parameters
    multi_seq_config = (hyperparams or {}).get("multi_sequence", {})
    model_kwargs = {
        "sequences": multi_seq_config.get(
            "sequences", ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]
        ),
        # Note: num_classes and pretrained are passed explicitly below, not in kwargs
        "in_channels": model_config.get("in_channels", 1),
        "feature_dims": multi_seq_config.get("feature_dims", [256, 512, 1024]),
        "fusion_dim": multi_seq_config.get("fusion_dim", 512),
        "attention_heads": multi_seq_config.get("attention", {}).get("num_heads", 8),
        "attention_dropout": multi_seq_config.get("attention", {}).get("dropout", 0.1),
        "head_hidden_dim": multi_seq_config.get("heads", {}).get("hidden_dim", 256),
        "head_dropout": multi_seq_config.get("heads", {}).get("dropout", 0.3),
    }

    model = create_pfirrman_model(
        architecture=model_config["architecture"],
        num_classes=model_config.get("num_classes", 5),
        pretrained=model_config.get("pretrained", True),
        **model_kwargs,
    )

    model = model.to(device)

    # Verify model architecture
    print(f"\n✓ Model created: {model_config['architecture']}")
    print(f"✓ Model type: {type(model).__name__}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"{'=' * 80}\n")

    # Log model info to wandb
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


def calculate_pfirrman_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-class Pfirrmann classification.

    Args:
        y_true: True class indices [N] (0-4 for grades 1-5)
        y_pred: Predicted class indices [N] (0-4 for grades 1-5)
        y_proba: Predicted probabilities [N, 5] (optional, required for AUROC/AUPRC)
        class_names: List of class names (default: ['Grade_1', 'Grade_2', 'Grade_3', 'Grade_4', 'Grade_5'])

    Returns:
        Dictionary of metrics
    """
    if class_names is None:
        class_names = [f"Grade_{i + 1}" for i in range(5)]

    metrics = {}

    # Overall accuracy
    metrics["overall/accuracy"] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    auroc_scores = []
    auprc_scores = []

    for i, class_name in enumerate(class_names):
        # Convert to binary for per-class metrics
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        metrics[f"{class_name}/precision"] = precision_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f"{class_name}/recall"] = recall_score(
            y_true_binary, y_pred_binary, zero_division=0
        )
        metrics[f"{class_name}/f1"] = f1_score(
            y_true_binary, y_pred_binary, zero_division=0
        )

        # AUROC and AUPRC (one-vs-rest) - require probability scores
        if y_proba is not None:
            try:
                # AUROC: one-vs-rest
                auroc = roc_auc_score(y_true_binary, y_proba[:, i])
                metrics[f"{class_name}/auroc"] = auroc
                auroc_scores.append(auroc)
            except ValueError:
                # Handle case where only one class present (no ROC curve)
                metrics[f"{class_name}/auroc"] = 0.0
                auroc_scores.append(0.0)

            try:
                # AUPRC: one-vs-rest
                auprc = average_precision_score(y_true_binary, y_proba[:, i])
                metrics[f"{class_name}/auprc"] = auprc
                auprc_scores.append(auprc)
            except ValueError:
                # Handle case where only one class present
                metrics[f"{class_name}/auprc"] = 0.0
                auprc_scores.append(0.0)
        else:
            # If no probabilities provided, set to None/0
            metrics[f"{class_name}/auroc"] = None
            metrics[f"{class_name}/auprc"] = None

    # Macro-averaged metrics
    metrics["overall/macro_precision"] = np.mean(
        [metrics[f"{cn}/precision"] for cn in class_names]
    )
    metrics["overall/macro_recall"] = np.mean(
        [metrics[f"{cn}/recall"] for cn in class_names]
    )
    metrics["overall/macro_f1"] = np.mean([metrics[f"{cn}/f1"] for cn in class_names])

    # Macro-averaged AUROC and AUPRC
    if y_proba is not None and len(auroc_scores) > 0:
        metrics["overall/macro_auroc"] = np.mean(auroc_scores)
        metrics["overall/macro_auprc"] = np.mean(auprc_scores)
    else:
        metrics["overall/macro_auroc"] = None
        metrics["overall/macro_auprc"] = None

    # Weighted-averaged metrics (accounts for class imbalance)
    metrics["overall/weighted_precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["overall/weighted_recall"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["overall/weighted_f1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Multi-class AUROC (ovr - one-vs-rest)
    if y_proba is not None:
        try:
            metrics["overall/multiclass_auroc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["overall/multiclass_auroc"] = None

    # Confusion matrix (for logging)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    metrics["confusion_matrix"] = cm.tolist()  # Convert to list for JSON serialization

    return metrics


def train_epoch_pfirrman(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch (Pfirrmann multi-class).

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: PyTorch device
        epoch: Current epoch number
        gradient_clip: Gradient clipping value (0.0 to disable)

    Returns:
        Tuple of (epoch_loss, metrics_dict)
    """
    model.train()
    running_loss = 0.0

    all_preds = []
    all_probs = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Prepare model input (multi-sequence format)
        model_input, labels = prepare_pfirrman_model_input(batch, device, model=model)

        # Forward pass
        optimizer.zero_grad()
        sequences, sequence_available = model_input
        logits = model(sequences, sequence_available)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Statistics
        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

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
    all_preds = np.concatenate(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)

    epoch_loss = running_loss / len(dataloader)
    metrics = calculate_pfirrman_metrics(all_targets, all_preds, all_probs)

    return epoch_loss, metrics


def validate_epoch_pfirrman(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate for one epoch (Pfirrmann multi-class).

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: PyTorch device
        epoch: Current epoch number

    Returns:
        Tuple of (epoch_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")
        for batch in pbar:
            # Prepare model input (multi-sequence format)
            model_input, labels = prepare_pfirrman_model_input(
                batch, device, model=model
            )

            # Forward pass
            sequences, sequence_available = model_input
            logits = model(sequences, sequence_available)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate epoch metrics
    all_preds = np.concatenate(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)

    epoch_loss = running_loss / len(dataloader)
    metrics = calculate_pfirrman_metrics(all_targets, all_preds, all_probs)

    return epoch_loss, metrics


def generate_pfirrman_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str = "dataset",
    return_dataframe: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], "pd.DataFrame"]:
    """
    Generate predictions on a dataset (Pfirrmann multi-class).

    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: PyTorch device
        dataset_name: Name of the dataset (for logging)
        return_dataframe: If True, return DataFrame with metadata (default: True)

    Returns:
        If return_dataframe=True: pandas DataFrame with columns:
            - patient_id, ivd_label
            - true_grade (1-5)
            - pred_grade (1-5)
            - prob_grade_1, prob_grade_2, ..., prob_grade_5
        If return_dataframe=False: Tuple of (true_labels, predicted_labels, predicted_probabilities) as numpy arrays
    """
    import pandas as pd

    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []
    all_patient_ids = []
    all_ivd_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Generating predictions [{dataset_name}]")
        for batch in pbar:
            # Prepare model input
            model_input, labels = prepare_pfirrman_model_input(
                batch, device, model=model
            )

            # Forward pass
            sequences, sequence_available = model_input
            logits = model(sequences, sequence_available)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

            # Collect metadata
            if "patient_id" in batch:
                all_patient_ids.extend(batch["patient_id"])
            if "ivd_label" in batch:
                all_ivd_labels.extend(batch["ivd_label"])

    # Concatenate all results
    all_preds = np.concatenate(all_preds)
    all_probs = np.vstack(all_probs)
    all_targets = np.concatenate(all_targets)

    if return_dataframe:
        # Convert to DataFrame
        df = pd.DataFrame(
            {
                "patient_id": all_patient_ids[: len(all_targets)],
                "ivd_label": all_ivd_labels[: len(all_targets)],
                "true_grade": all_targets + 1,  # Convert back to 1-5
                "pred_grade": all_preds + 1,  # Convert back to 1-5
            }
        )

        # Add probability columns
        for i in range(5):
            df[f"prob_grade_{i + 1}"] = all_probs[:, i]

        return df
    else:
        # Return as arrays (0-indexed: 0-4 for grades 1-5)
        return all_targets, all_preds, all_probs
