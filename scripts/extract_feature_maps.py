#!/usr/bin/env python3
"""
Script to extract and visualize feature maps from trained pathology classification models.

This script loads a trained model and extracts feature maps from the backbone
for specified patients, saving them as PNG images for visualization.

Usage:
    python scripts/extract_feature_maps.py --checkpoint outputs/pathology_model/runs/efficientnet_b1/weights/best_model.pth --num_patients 1
    python scripts/extract_feature_maps.py --checkpoint outputs/pathology_model/runs/resnet18/weights/best_model.pth --patient_id P001 --split test
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

# Configure loguru logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.pathology_model import SimpleMultiSequenceFusion
from utils.multi_sequence_dataset import MultiSequencePathologyDataset
from utils.dataset import get_default_transforms


class FeatureExtractor:
    """Extract feature maps from model backbone layers."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.features = {}
        self.hooks = []

    def register_hooks(self, backbone_type: str):
        """Register forward hooks to capture feature maps from backbone layers."""
        self.clear_hooks()

        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone

            if backbone_type.startswith("resnet"):
                # Register hooks for ResNet layers
                layers = {
                    "layer1": backbone.layer1,
                    "layer2": backbone.layer2,
                    "layer3": backbone.layer3,
                    "layer4": backbone.layer4,
                }
            elif backbone_type.startswith("efficientnet") or backbone_type.startswith(
                "densenet"
            ):
                # Register hooks for EfficientNet/DenseNet features
                if hasattr(backbone, "features"):
                    features = backbone.features
                    num_layers = len(features)

                    # Select early, middle, and late layers
                    layers = {
                        "early_features": features[: num_layers // 3],
                        "mid_features": features[num_layers // 3 : 2 * num_layers // 3],
                        "late_features": features[2 * num_layers // 3 :],
                    }
            elif backbone_type.startswith("vit"):
                # ViT doesn't have conv layers, skip
                logger.warning(
                    "Vision Transformer doesn't have convolutional feature maps to visualize"
                )
                return
            else:
                logger.warning(f"Unknown backbone type: {backbone_type}")
                return

            # Register hooks
            for name, layer in layers.items():
                hook = layer.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)
                logger.info(f"Registered hook for layer: {name}")
        else:
            logger.error("Model doesn't have 'backbone' attribute")

    def _get_activation(self, name: str):
        """Create hook function to capture activations."""

        def hook(module, input, output):
            self.features[name] = output.detach()

        return hook

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}

    def extract_features(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract features for a batch of data."""
        self.features = {}

        # Move batch to device
        if isinstance(batch, dict):
            images = batch["images"].to(self.device)
            ivd_labels = batch.get("ivd_labels")
            if ivd_labels is not None:
                ivd_labels = ivd_labels.to(self.device)
        else:
            images = batch.to(self.device)
            ivd_labels = None

        # Forward pass
        with torch.no_grad():
            if ivd_labels is not None:
                _ = self.model(images, ivd_labels=ivd_labels)
            else:
                _ = self.model(images)

        return self.features


def save_feature_maps(
    features: Dict[str, torch.Tensor],
    save_dir: Path,
    patient_id: str,
    sequence_names: List[str],
    max_channels: int = 16,
):
    """
    Save feature maps as PNG images.

    Args:
        features: Dictionary of layer_name -> feature tensor [B, C, H, W]
        save_dir: Directory to save images
        patient_id: Patient identifier
        sequence_names: List of sequence names for this batch
        max_channels: Maximum number of channels to visualize per layer
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, feature_map in features.items():
        # feature_map shape: [batch_size, channels, height, width]
        batch_size, num_channels, height, width = feature_map.shape

        logger.info(f"Processing {layer_name}: {feature_map.shape}")

        # Process each sample in batch (each sequence)
        for batch_idx in range(min(batch_size, len(sequence_names))):
            sequence_name = (
                sequence_names[batch_idx]
                if batch_idx < len(sequence_names)
                else f"seq_{batch_idx}"
            )
            layer_dir = save_dir / patient_id / sequence_name / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)

            # Get feature map for this sample
            sample_features = feature_map[batch_idx]  # [C, H, W]

            # Save individual channels
            channels_to_save = min(num_channels, max_channels)
            for ch_idx in range(channels_to_save):
                channel_data = sample_features[ch_idx].cpu().numpy()

                # Normalize to 0-255
                channel_min = channel_data.min()
                channel_max = channel_data.max()
                if channel_max > channel_min:
                    channel_normalized = (
                        (channel_data - channel_min) / (channel_max - channel_min) * 255
                    ).astype(np.uint8)
                else:
                    channel_normalized = np.zeros_like(channel_data, dtype=np.uint8)

                # Save as grayscale PNG
                img = Image.fromarray(channel_normalized, mode="L")
                img_path = layer_dir / f"channel_{ch_idx:03d}.png"
                img.save(img_path)

            # Create RGB composite from first 3 channels
            if num_channels >= 3:
                rgb_channels = []
                for ch_idx in range(3):
                    channel_data = sample_features[ch_idx].cpu().numpy()
                    channel_min = channel_data.min()
                    channel_max = channel_data.max()
                    if channel_max > channel_min:
                        channel_normalized = (
                            (channel_data - channel_min)
                            / (channel_max - channel_min)
                            * 255
                        ).astype(np.uint8)
                    else:
                        channel_normalized = np.zeros_like(channel_data, dtype=np.uint8)
                    rgb_channels.append(channel_normalized)

                rgb_image = np.stack(rgb_channels, axis=-1)
                img = Image.fromarray(rgb_image, mode="RGB")
                img_path = layer_dir / "rgb_composite.png"
                img.save(img_path)

            # Create grid visualization
            grid_size = int(np.ceil(np.sqrt(channels_to_save)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            axes = axes.flatten() if channels_to_save > 1 else [axes]

            for ch_idx in range(channels_to_save):
                channel_data = sample_features[ch_idx].cpu().numpy()
                axes[ch_idx].imshow(channel_data, cmap="viridis")
                axes[ch_idx].set_title(f"Ch {ch_idx}", fontsize=8)
                axes[ch_idx].axis("off")

            # Hide unused subplots
            for ch_idx in range(channels_to_save, len(axes)):
                axes[ch_idx].axis("off")

            plt.tight_layout()
            grid_path = layer_dir / "feature_grid.png"
            plt.savefig(grid_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.success(
                f"Saved feature maps for {patient_id}/{sequence_name}/{layer_name}"
            )


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration
    model_config = checkpoint.get("model_config", {})
    hyperparams = checkpoint.get("hyperparams", {})
    backbone = checkpoint.get("backbone", "resnet18")

    # Get sequences from checkpoint
    sequences = checkpoint.get("sequences", ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"])

    # Create model
    if "simple_multi_sequence_fusion" not in hyperparams:
        hyperparams["simple_multi_sequence_fusion"] = {}
    hyperparams["simple_multi_sequence_fusion"]["backbone_type"] = backbone
    hyperparams["simple_multi_sequence_fusion"]["sequences"] = sequences

    # Ensure model config has required fields
    if "multi_sequence" not in model_config:
        model_config["multi_sequence"] = {}
    model_config["multi_sequence"]["use_ivd_encoding"] = checkpoint.get(
        "ivd_encoding_enabled", False
    )
    model_config["multi_sequence"]["ivd_embedding_dim"] = checkpoint.get(
        "ivd_embedding_dim", 16
    )

    num_labels = model_config.get("num_labels", 5)
    use_ivd_encoding = model_config["multi_sequence"].get("use_ivd_encoding", False)
    ivd_embedding_dim = model_config["multi_sequence"].get("ivd_embedding_dim", 16)

    logger.info(
        f"Model config: backbone={backbone}, num_labels={num_labels}, sequences={sequences}"
    )
    logger.info(f"IVD encoding: {use_ivd_encoding}, embedding_dim: {ivd_embedding_dim}")

    # Create model
    model = SimpleMultiSequenceFusion(
        backbone_type=backbone,
        num_labels=num_labels,
        sequences=sequences,
        use_ivd_encoding=use_ivd_encoding,
    )

    # Load state dict with strict=False to handle architecture mismatches
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        logger.success("Loaded checkpoint with strict mode")
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {str(e)[:200]}")
        logger.info("Attempting non-strict loading...")

        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}..."
            )

        # Check if critical backbone keys are missing
        backbone_keys = [k for k in missing_keys if "backbone" in k.lower()]
        if backbone_keys:
            logger.error(
                "Critical backbone keys are missing! Model may not work correctly."
            )
            logger.error(f"Missing backbone keys: {backbone_keys[:5]}...")
        else:
            logger.success(
                "Loaded checkpoint with non-strict mode (backbone weights intact)"
            )

    model.eval()

    logger.success(
        f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}"
    )

    return model, backbone, sequences, use_ivd_encoding


def main():
    parser = argparse.ArgumentParser(
        description="Extract and visualize feature maps from trained pathology models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features for 1 patient from validation set
  python scripts/extract_feature_maps.py --checkpoint outputs/pathology_model/runs/efficientnet_b1/weights/best_model.pth --num_patients 1
  
  # Extract features for specific patient
  python scripts/extract_feature_maps.py --checkpoint outputs/pathology_model/runs/resnet18/weights/best_model.pth --patient_id P001 --split test
  
  # Extract features for 5 patients from training set
  python scripts/extract_feature_maps.py --checkpoint outputs/pathology_model/runs/densenet121/weights/best_model.pth --num_patients 5 --split train
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )

    parser.add_argument(
        "--num_patients",
        type=int,
        default=1,
        help="Number of patients to extract features for (default: 1)",
    )

    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Specific patient ID to extract features for (overrides --num_patients)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to use (default: val)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/mri_phen/pipeline_figs",
        help="Output directory for feature maps (default: ../data/mri_phen/pipeline_figs)",
    )

    parser.add_argument(
        "--max_channels",
        type=int,
        default=16,
        help="Maximum number of channels to visualize per layer (default: 16)",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.success(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model, backbone, sequences, use_ivd_encoding = load_checkpoint(
        checkpoint_path, device
    )

    # Create feature extractor
    extractor = FeatureExtractor(model, device)
    extractor.register_hooks(backbone)

    # Load dataset
    data_dir = Path("../data/mri_phen/processed")
    manifest_path = data_dir / "pathology_training_manifest.csv"

    if not manifest_path.exists():
        logger.error(f"Training manifest not found: {manifest_path}")
        sys.exit(1)

    # Load manifest to get split info
    manifest_df = pd.read_csv(manifest_path)
    split_df = manifest_df[manifest_df["split"] == args.split]

    if len(split_df) == 0:
        logger.error(f"No samples found for split: {args.split}")
        sys.exit(1)

    logger.info(f"Found {len(split_df)} samples in {args.split} split")

    # Create dataset
    transform = get_default_transforms(
        img_size=224,
        augment=False,  # No augmentation for feature extraction
    )

    dataset = MultiSequencePathologyDataset(
        manifest_df=split_df,
        sequences=sequences,
        transform=transform,
        use_ivd_labels=use_ivd_encoding,
    )

    # Select patients to process
    if args.patient_id:
        # Find specific patient
        patient_indices = [
            i for i, row in split_df.iterrows() if row["patient_id"] == args.patient_id
        ]
        if not patient_indices:
            logger.error(f"Patient ID not found: {args.patient_id}")
            sys.exit(1)
        indices = patient_indices[:1]  # Take first occurrence
        logger.info(f"Processing patient: {args.patient_id}")
    else:
        # Take first N patients
        unique_patients = split_df["patient_id"].unique()
        num_patients = min(args.num_patients, len(unique_patients))
        selected_patients = unique_patients[:num_patients]
        indices = [
            i
            for i, row in split_df.iterrows()
            if row["patient_id"] in selected_patients
        ]
        logger.info(f"Processing {num_patients} patients: {selected_patients.tolist()}")

    # Create subset dataset
    subset_dataset = Subset(dataset, indices)
    dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving feature maps to: {output_dir}")

    # Extract and save features
    logger.info(f"{'=' * 80}")
    logger.info("Starting feature extraction...")
    logger.info(f"{'=' * 80}")

    for batch_idx, batch in enumerate(dataloader):
        # Get patient info
        dataset_idx = indices[batch_idx]
        patient_row = split_df.iloc[dataset_idx]
        patient_id = patient_row["patient_id"]
        ivd_level = patient_row["ivd_level"]

        logger.info(f"\nProcessing patient: {patient_id}, IVD: {ivd_level}")

        # Extract features
        features = extractor.extract_features(batch)

        if not features:
            logger.warning(f"No features extracted for patient {patient_id}")
            continue

        # Save feature maps
        save_feature_maps(
            features=features,
            save_dir=output_dir,
            patient_id=f"{patient_id}_{ivd_level}",
            sequence_names=sequences,
            max_channels=args.max_channels,
        )

    # Clean up
    extractor.clear_hooks()

    logger.success(f"\n{'=' * 80}")
    logger.success("Feature extraction completed!")
    logger.success(f"Feature maps saved to: {output_dir}")
    logger.success(f"{'=' * 80}")


if __name__ == "__main__":
    main()
