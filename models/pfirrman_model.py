"""
Model architectures for Pfirrmann grade classification.

This module provides configurable models for predicting Pfirrmann grades (1-5)
using multi-sequence fusion.

All models use multi-sequence fusion architecture with different backbone types:
- ResNet (resnet18, resnet34, resnet50)
- EfficientNet (efficientnet_b0, efficientnet_b1, efficientnet_b2)
- DenseNet (densenet121, densenet169, densenet201)
- Vision Transformer (vit_base_patch16_224, vit_large_patch16_224)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class MultiSequencePfirrmannFusion(nn.Module):
    """
    Multi-sequence fusion model for Pfirrmann grade classification.

    Reuses MultiSequenceMultiScaleFusion architecture but with:
    - Single classifier head (not class-specific) for multi-class classification
    - 5 output classes (grades 1-5)
    - Softmax output for CrossEntropyLoss
    """

    def __init__(
        self,
        sequences: List[str] = ["sag_t2", "ax_t2", "sag_stir"],
        num_classes: int = 5,
        backbone_type: str = "densenet121",
        pretrained: bool = True,
        in_channels: int = 1,
        feature_dims: List[int] = [256, 512, 1024],
        fusion_dim: int = 512,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
    ):
        """
        Initialize MultiSequencePfirrmannFusion model.

        Args:
            sequences: List of sequence names to use
            num_classes: Number of output classes (default: 5 for grades 1-5)
            backbone_type: Backbone type for sequence encoders ('resnet18', 'resnet34', 'resnet50',
                          'densenet121', 'densenet169', 'densenet201',
                          'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                          'vit_base_patch16_224', 'vit_large_patch16_224')
            pretrained: Whether to use pretrained backbones
            in_channels: Number of input channels (1 for grayscale)
            feature_dims: Feature dimensions for each scale [scale1, scale2, scale3]
            fusion_dim: Dimension for fused features
            attention_heads: Number of attention heads
            attention_dropout: Dropout rate for attention
            head_hidden_dim: Hidden dimension for classifier head
            head_dropout: Dropout rate for classifier head
        """
        super(MultiSequencePfirrmannFusion, self).__init__()

        self.sequences = [seq.lower() for seq in sequences]
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.feature_dims = feature_dims
        self.fusion_dim = fusion_dim

        # Import components from multi_sequence_fusion_model
        from models.multi_sequence_fusion_model import (
            SequenceEncoder,
            CrossSequenceAttention,
            MultiScaleFusion,
        )

        # Sequence encoders (one per sequence)
        self.sequence_encoders = nn.ModuleDict(
            {
                seq: SequenceEncoder(
                    sequence_name=seq,
                    backbone_type=backbone_type,
                    pretrained=pretrained,
                    in_channels=in_channels,
                    feature_dims=feature_dims,
                )
                for seq in self.sequences
            }
        )

        # Cross-sequence attention for each scale (standard attention, not pathology-specific)
        self.scale_attentions = nn.ModuleList(
            [
                CrossSequenceAttention(
                    feature_dim=feature_dims[i],
                    num_heads=attention_heads,
                    dropout=attention_dropout,
                )
                for i in range(len(feature_dims))
            ]
        )

        # Multi-scale fusion
        self.multi_scale_fusion = MultiScaleFusion(
            feature_dims=feature_dims,
            output_dim=fusion_dim,
        )

        # Single classifier head for multi-class classification
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim // 2, num_classes),  # 5 classes for grades 1-5
        )

        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        sequences: Dict[str, torch.Tensor],
        sequence_available: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: Dict mapping sequence names to input tensors [B, C, H, W]
            sequence_available: Optional dict mapping sequence names to availability flags [B]

        Returns:
            Logits [B, num_classes] (no softmax applied - use with CrossEntropyLoss)
        """
        # Extract features using base model (up to fusion)
        # We need to manually extract features since base_model expects to output logits
        # Let's extract features directly from the fusion layer

        # Determine batch size and device
        non_none_sequences = [v for v in sequences.values() if v is not None]
        if not non_none_sequences:
            raise ValueError("All sequences are None - cannot determine batch size")
        batch_size = non_none_sequences[0].size(0)
        device = non_none_sequences[0].device

        # Extract multi-scale features from each sequence
        all_scale_features = {
            scale_idx: {} for scale_idx in range(len(self.feature_dims))
        }

        for seq_name in self.sequences:
            if seq_name in sequences and sequences[seq_name] is not None:
                # Encode sequence
                scale_features = self.sequence_encoders[seq_name](sequences[seq_name])

                # Store features for each scale
                for scale_idx, features in enumerate(scale_features):
                    # Global average pooling to get [B, feature_dim]
                    if features.dim() > 2:
                        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
                        features = features.view(features.size(0), -1)
                    all_scale_features[scale_idx][seq_name] = features

        # Apply cross-sequence attention at each scale
        attended_features = []
        for scale_idx in range(len(self.feature_dims)):
            scale_feat_dict = all_scale_features[scale_idx]

            # Get availability flags for this scale
            scale_available = None
            if sequence_available is not None:
                scale_available = sequence_available

            # Apply attention
            if scale_feat_dict:
                attended = self.scale_attentions[scale_idx](
                    scale_feat_dict, scale_available
                )
                attended_features.append(attended)
            else:
                # No sequences available - create zero features
                feature_dim = self.feature_dims[scale_idx]
                attended = torch.zeros(batch_size, feature_dim, device=device)
                attended_features.append(attended)

        # Multi-scale fusion
        fused_features = self.multi_scale_fusion(attended_features)

        # Classify
        logits = self.classifier(fused_features)

        return logits


def create_pfirrman_model(
    architecture: str,
    num_classes: int = 5,
    pretrained: bool = True,
    sequences: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Pfirrmann classification model.

    Args:
        architecture: Architecture name (e.g., 'resnet18', 'densenet121', 'efficientnet_b0', 'vit_base_patch16_224')
                     This determines the backbone type used in multi-sequence fusion
        num_classes: Number of output classes (default: 5 for grades 1-5)
        pretrained: Whether to use pretrained weights
        sequences: List of sequence names (default: ['SAG_T2', 'AX_T2', 'SAG_STIR'])
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Pfirrmann classification model

    Example:
        >>> model = create_pfirrman_model('densenet121', num_classes=5)
        >>> model = create_pfirrman_model('resnet18', sequences=['SAG_T2', 'AX_T2'])
    """
    # Default sequences
    if sequences is None:
        sequences = ["SAG_T2", "AX_T2", "SAG_STIR"]

    # Map architecture to backbone type
    # All architectures use multi-sequence fusion, architecture name determines backbone
    backbone_type = architecture.lower()

    # Validate backbone type
    valid_backbones = [
        "resnet18",
        "resnet34",
        "resnet50",
        "densenet121",
        "densenet169",
        "densenet201",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
    ]

    if backbone_type not in valid_backbones:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Must be one of: {valid_backbones}"
        )

    # Create model
    model = MultiSequencePfirrmannFusion(
        sequences=sequences,
        num_classes=num_classes,
        backbone_type=backbone_type,
        pretrained=pretrained,
        **kwargs,
    )

    return model
