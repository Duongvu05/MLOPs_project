"""
Model architectures for multi-label pathology classification.

This module provides configurable models for predicting
4 independent pathology labels (disc_herniation, disc_bulging, spondylolisthesis, disc_narrowing).

Supported architectures:
- ResNet (resnet18, resnet34, resnet50)
- EfficientNet (efficientnet_b0, efficientnet_b1, efficientnet_b2)
- DenseNet (densenet121, densenet169, densenet201)
- Vision Transformer (vit_base_patch16_224, vit_large_patch16_224)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal, Union

try:
    from torchvision.models import (
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        EfficientNet_B0_Weights,
        EfficientNet_B1_Weights,
        EfficientNet_B2_Weights,
    )

    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from models.multi_sequence_fusion_model import MultiSequenceMultiScaleFusion

    MULTI_SEQUENCE_AVAILABLE = True
except ImportError:
    MULTI_SEQUENCE_AVAILABLE = False
    MultiSequenceMultiScaleFusion = None

try:
    from models.transformer_fusion_model import TransformerHierarchicalFusion

    TRANSFORMER_FUSION_AVAILABLE = True
except ImportError:
    TRANSFORMER_FUSION_AVAILABLE = False
    TransformerHierarchicalFusion = None


class PathologyResNet(nn.Module):
    """
    ResNet-based model for multi-label pathology classification.

    Outputs 4 independent logits (one per pathology type):
    - disc_herniation
    - disc_bulging
    - spondylolisthesis
    - disc_narrowing

    Architecture:
    - Backbone: ResNet18/34/50 (configurable)
    - Final layer: Linear(backbone_features, 4) for 4 binary outputs
    - No activation in forward (returns logits for BCEWithLogitsLoss)
    """

    def __init__(
        self,
        architecture: Literal["resnet18", "resnet34", "resnet50"] = "resnet18",
        num_labels: int = 4,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize PathologyResNet model.

        Args:
            architecture: ResNet architecture to use ('resnet18', 'resnet34', or 'resnet50')
            num_labels: Number of output labels (default: 4 for pathology types)
            pretrained: Whether to use ImageNet pretrained weights
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate before classifier (0.0 to disable, default: 0.5)
        """
        super(PathologyResNet, self).__init__()

        self.architecture = architecture
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        # Load pretrained ResNet backbone
        if architecture == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif architecture == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif architecture == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. Choose from 'resnet18', 'resnet34', 'resnet50'"
            )

        # Modify first layer for grayscale input if needed
        if in_channels == 1:
            # Replace first conv layer to accept 1 channel instead of 3
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            # Initialize with pretrained weights (average across RGB channels)
            if pretrained:
                with torch.no_grad():
                    backbone.conv1.weight.data = original_conv.weight.data.mean(
                        dim=1, keepdim=True
                    )
        elif in_channels == 3:
            # Keep original conv1 for RGB
            pass
        else:
            raise ValueError(
                f"Unsupported in_channels: {in_channels}. Use 1 (grayscale) or 3 (RGB)"
            )

        # Remove final fully connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Add dropout for regularization
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Add custom classifier for multi-label output
        self.classifier = nn.Linear(feature_dim, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Logits tensor of shape [batch_size, num_labels]
            (No sigmoid applied - use with BCEWithLogitsLoss)
        """
        # Extract features
        features = self.backbone(x)

        # Flatten
        features = features.view(features.size(0), -1)

        # Apply dropout
        features = self.dropout(features)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions (with sigmoid).

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Probability tensor of shape [batch_size, num_labels] (values in [0, 1])
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions (with threshold).

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            threshold: Threshold for binary classification (default: 0.5)

        Returns:
            Binary predictions tensor of shape [batch_size, num_labels] (values 0 or 1)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


class PathologyEfficientNet(nn.Module):
    """
    EfficientNet-based model for multi-label pathology classification.

    EfficientNet is known for its efficiency and strong performance on medical imaging tasks.
    It uses compound scaling to balance depth, width, and resolution.

    Outputs 4 independent logits (one per pathology type):
    - disc_herniation
    - disc_bulging
    - spondylolisthesis
    - disc_narrowing
    """

    def __init__(
        self,
        architecture: Literal[
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"
        ] = "efficientnet_b0",
        num_labels: int = 4,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize PathologyEfficientNet model.

        Args:
            architecture: EfficientNet architecture to use
            num_labels: Number of output labels (default: 4 for pathology types)
            pretrained: Whether to use ImageNet pretrained weights
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate before classifier (0.0 to disable, default: 0.5)
        """
        super(PathologyEfficientNet, self).__init__()

        if not EFFICIENTNET_AVAILABLE:
            raise ImportError(
                "EfficientNet models require torchvision >= 0.13.0. Install with: pip install torchvision>=0.13.0"
            )

        self.architecture = architecture
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        # Load pretrained EfficientNet backbone
        if architecture == "efficientnet_b0":
            if pretrained:
                backbone = efficientnet_b0(
                    weights=EfficientNet_B0_Weights.IMAGENET1K_V1
                )
            else:
                backbone = efficientnet_b0(weights=None)
            feature_dim = 1280
        elif architecture == "efficientnet_b1":
            if pretrained:
                backbone = efficientnet_b1(
                    weights=EfficientNet_B1_Weights.IMAGENET1K_V1
                )
            else:
                backbone = efficientnet_b1(weights=None)
            feature_dim = 1280
        elif architecture == "efficientnet_b2":
            if pretrained:
                backbone = efficientnet_b2(
                    weights=EfficientNet_B2_Weights.IMAGENET1K_V1
                )
            else:
                backbone = efficientnet_b2(weights=None)
            feature_dim = 1408
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. Choose from 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'"
            )

        # Modify first layer for grayscale input if needed
        if in_channels == 1:
            # Replace first conv layer to accept 1 channel instead of 3
            original_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            # Initialize with pretrained weights (average across RGB channels)
            if pretrained:
                with torch.no_grad():
                    backbone.features[0][
                        0
                    ].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        elif in_channels == 3:
            # Keep original conv for RGB
            pass
        else:
            raise ValueError(
                f"Unsupported in_channels: {in_channels}. Use 1 (grayscale) or 3 (RGB)"
            )

        # Remove final classifier layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Add dropout for regularization
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Add custom classifier for multi-label output
        self.classifier = nn.Linear(feature_dim, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.backbone(x)

        # EfficientNet outputs [B, C, H, W], need to pool
        if features.dim() == 4:
            # Global average pooling
            features = features.mean(dim=[2, 3])

        # Flatten if needed
        features = features.view(features.size(0), -1)

        # Apply dropout
        features = self.dropout(features)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (with sigmoid)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions (with threshold)."""
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


class PathologyDenseNet(nn.Module):
    """
    DenseNet-based model for multi-label pathology classification.

    DenseNet uses dense connections between layers, promoting feature reuse
    and reducing the number of parameters. Good for medical imaging tasks.

    Outputs 4 independent logits (one per pathology type):
    - disc_herniation
    - disc_bulging
    - spondylolisthesis
    - disc_narrowing
    """

    def __init__(
        self,
        architecture: Literal[
            "densenet121", "densenet169", "densenet201"
        ] = "densenet121",
        num_labels: int = 4,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize PathologyDenseNet model.

        Args:
            architecture: DenseNet architecture to use
            num_labels: Number of output labels (default: 4 for pathology types)
            pretrained: Whether to use ImageNet pretrained weights
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate before classifier (0.0 to disable, default: 0.5)
        """
        super(PathologyDenseNet, self).__init__()

        self.architecture = architecture
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        # Load pretrained DenseNet backbone
        if architecture == "densenet121":
            backbone = models.densenet121(pretrained=pretrained)
            feature_dim = 1024
        elif architecture == "densenet169":
            backbone = models.densenet169(pretrained=pretrained)
            feature_dim = 1664
        elif architecture == "densenet201":
            backbone = models.densenet201(pretrained=pretrained)
            feature_dim = 1920
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. Choose from 'densenet121', 'densenet169', 'densenet201'"
            )

        # Modify first layer for grayscale input if needed
        if in_channels == 1:
            # Replace first conv layer to accept 1 channel instead of 3
            original_conv = backbone.features.conv0
            backbone.features.conv0 = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            # Initialize with pretrained weights (average across RGB channels)
            if pretrained:
                with torch.no_grad():
                    backbone.features.conv0.weight.data = (
                        original_conv.weight.data.mean(dim=1, keepdim=True)
                    )
        elif in_channels == 3:
            # Keep original conv for RGB
            pass
        else:
            raise ValueError(
                f"Unsupported in_channels: {in_channels}. Use 1 (grayscale) or 3 (RGB)"
            )

        # Remove final classifier layer
        # DenseNet has features and classifier, we only need features
        self.backbone = backbone.features

        # Add dropout for regularization
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Add custom classifier for multi-label output
        self.classifier = nn.Linear(feature_dim, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.backbone(x)

        # DenseNet features output [B, C, H, W], need global pooling
        # Use adaptive average pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))

        # Flatten
        features = features.view(features.size(0), -1)

        # Apply dropout
        features = self.dropout(features)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (with sigmoid)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions (with threshold)."""
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


class SimpleMultiSequenceFusion(nn.Module):
    """
    Simple multi-sequence fusion model for pathology classification.

    Uses pretrained backbones (ResNet, DenseNet, EfficientNet) with direct concatenation fusion.

    Architecture:
    1. Pretrained backbone for each sequence -> Global average pooling -> B,C (C depends on backbone)
    2. Concatenate all sequence features directly -> B,sum(C_i) where C_i is feature dim per sequence
    3. Concatenate with IVD level embedding (if enabled) -> B,sum(C_i)+16
    4. Separate feedforward networks for each pathology (4 separate heads)

    Supported backbones:
    - ResNet: resnet18 (512 features), resnet34 (512), resnet50 (2048)
    - DenseNet: densenet121 (1024), densenet169 (1664), densenet201 (1920)
    - EfficientNet: efficientnet_b0 (1280), efficientnet_b1 (1280), efficientnet_b2 (1408)

    Outputs 4 independent logits (one per pathology type):
    - disc_herniation
    - disc_bulging
    - spondylolisthesis
    - disc_narrowing
    """

    def __init__(
        self,
        sequences=["sag_t2", "ax_t2", "sag_stir"],
        num_labels=4,
        backbone_type="densenet121",
        pretrained=True,
        in_channels=1,
        projection_dim=10,
        hidden_dim=128,
        dropout_rate=0.3,
        use_ivd_encoding=True,
        ivd_encoding_mode="positional",
    ):
        """
        Initialize SimpleMultiSequenceFusion model with pretrained backbone.

        Args:
            sequences: List of sequence names
            num_labels: Number of pathology labels (default: 4)
            backbone_type: Backbone architecture ('resnet18/34/50', 'densenet121/169/201', 'efficientnet_b0/b1/b2')
            pretrained: Whether to use pretrained ImageNet weights
            in_channels: Input channels (1 for grayscale)
            projection_dim: Dimension to project each sequence to (default: 10)
            hidden_dim: Hidden dimension for feedforward networks (default: 128)
            dropout_rate: Dropout rate (default: 0.3)
            use_ivd_encoding: Whether to use IVD level encoding (default: True)
            ivd_encoding_mode: IVD encoding mode - "positional" (nn.Embedding) or "label" (direct values, default: "positional")
        """
        super(SimpleMultiSequenceFusion, self).__init__()

        self.sequences = sequences
        self.num_sequences = len(sequences)
        self.num_labels = num_labels
        self.backbone_type = backbone_type
        self.projection_dim = (
            projection_dim  # Keep for backward compatibility but not used
        )
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_ivd_encoding = use_ivd_encoding
        self.ivd_encoding_mode = ivd_encoding_mode

        # Validate encoding mode
        if self.use_ivd_encoding:
            if self.ivd_encoding_mode not in ["positional", "label"]:
                raise ValueError(
                    f"ivd_encoding_mode must be 'positional' or 'label', got '{self.ivd_encoding_mode}'"
                )

        # IVD level embedding (only for positional encoding mode)
        if self.use_ivd_encoding and self.ivd_encoding_mode == "positional":
            self.ivd_embedding = nn.Embedding(
                6, 16
            )  # 6 vocab size (0-5), 16 embedding dim

        # Create sequence encoders and track feature dimensions
        self.sequence_encoders = nn.ModuleDict()
        self.feature_dims = {}  # Track feature dimension for each sequence

        for seq_name in sequences:
            # Create pretrained backbone for each sequence
            backbone, feature_dim = self._create_backbone(
                backbone_type, pretrained, in_channels
            )
            self.sequence_encoders[seq_name] = backbone
            self.feature_dims[seq_name] = feature_dim

        # Concatenated feature dimension (sum of all backbone feature dims)
        concat_dim = sum(self.feature_dims.values())
        if self.use_ivd_encoding:
            if self.ivd_encoding_mode == "positional":
                concat_dim += 16  # Add IVD embedding dimension
            elif self.ivd_encoding_mode == "label":
                concat_dim += 1  # Add single IVD level feature (normalized to [0, 1])

        # Separate feedforward networks for each pathology
        self.pathology_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(concat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim // 2, 1),  # Binary output for each pathology
                )
                for _ in range(num_labels)
            ]
        )

        # Initialize weights
        self._init_weights()

    def _create_backbone(self, backbone_type, pretrained, in_channels):
        """Create backbone and return backbone + feature dimension."""
        if backbone_type.startswith("resnet"):
            if backbone_type == "resnet18":
                backbone = models.resnet18(pretrained=pretrained)
                feature_dim = 512
            elif backbone_type == "resnet34":
                backbone = models.resnet34(pretrained=pretrained)
                feature_dim = 512
            elif backbone_type == "resnet50":
                backbone = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet: {backbone_type}")

            # Modify first conv for grayscale if needed
            if in_channels == 1:
                original_conv = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    in_channels=1,
                    out_channels=original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias is not None,
                )
                if pretrained:
                    with torch.no_grad():
                        backbone.conv1.weight.data = original_conv.weight.data.mean(
                            dim=1, keepdim=True
                        )

            # Remove final FC layer, keep features + adaptive pool
            backbone = nn.Sequential(
                *list(backbone.children())[:-1]  # Remove final FC layer
            )

        elif backbone_type.startswith("densenet"):
            if backbone_type == "densenet121":
                backbone = models.densenet121(pretrained=pretrained)
                feature_dim = 1024
            elif backbone_type == "densenet169":
                backbone = models.densenet169(pretrained=pretrained)
                feature_dim = 1664
            elif backbone_type == "densenet201":
                backbone = models.densenet201(pretrained=pretrained)
                feature_dim = 1920
            else:
                raise ValueError(f"Unsupported DenseNet: {backbone_type}")

            # Modify first conv for grayscale if needed
            if in_channels == 1:
                original_conv = backbone.features.conv0
                backbone.features.conv0 = nn.Conv2d(
                    in_channels=1,
                    out_channels=original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias is not None,
                )
                if pretrained:
                    with torch.no_grad():
                        backbone.features.conv0.weight.data = (
                            original_conv.weight.data.mean(dim=1, keepdim=True)
                        )

            # Use features + adaptive pool
            backbone = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)))

        elif backbone_type.startswith("efficientnet") and EFFICIENTNET_AVAILABLE:
            if backbone_type == "efficientnet_b0":
                backbone = efficientnet_b0(
                    weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
                )
                feature_dim = 1280
            elif backbone_type == "efficientnet_b1":
                backbone = efficientnet_b1(
                    weights=EfficientNet_B1_Weights.DEFAULT if pretrained else None
                )
                feature_dim = 1280
            elif backbone_type == "efficientnet_b2":
                backbone = efficientnet_b2(
                    weights=EfficientNet_B2_Weights.DEFAULT if pretrained else None
                )
                feature_dim = 1408
            else:
                raise ValueError(f"Unsupported EfficientNet: {backbone_type}")

            # Modify first conv for grayscale if needed
            if in_channels == 1:
                original_conv = backbone.features[0][0]
                backbone.features[0][0] = nn.Conv2d(
                    in_channels=1,
                    out_channels=original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias is not None,
                )
                if pretrained:
                    with torch.no_grad():
                        backbone.features[0][
                            0
                        ].weight.data = original_conv.weight.data.mean(
                            dim=1, keepdim=True
                        )

            # Use features + adaptive pool
            backbone = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)))

        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        return backbone, feature_dim

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sequences, sequence_available=None, ivd_levels=None):
        """
        Forward pass.

        Args:
            sequences: Dict of sequence tensors {seq_name: tensor [B, C, H, W]}
            sequence_available: Optional dict of availability masks {seq_name: bool_tensor [B]}
            ivd_levels: Optional tensor of IVD levels [B] (values 1-5) for embedding

        Returns:
            logits: Tensor of shape [B, num_labels] with logits for each pathology
        """
        # Determine batch size
        non_none_sequences = [v for v in sequences.values() if v is not None]
        if non_none_sequences:
            batch_size = non_none_sequences[0].size(0)
        elif ivd_levels is not None:
            batch_size = ivd_levels.size(0)
        elif sequence_available is not None:
            # Get batch size from any sequence availability tensor
            for av in sequence_available.values():
                if av is not None:
                    batch_size = av.size(0)
                    break
            else:
                raise ValueError(
                    "Cannot determine batch size: all sequences are None and no availability info"
                )
        else:
            raise ValueError(
                "Cannot determine batch size: all sequences are None, no ivd_levels, and no sequence_available"
            )
        sequence_features = []

        # Process each sequence
        for seq_name in self.sequences:
            if seq_name in sequences and sequences[seq_name] is not None:
                # Get sequence input
                seq_input = sequences[seq_name]  # [B, C, H, W]

                # Extract features via backbone + adaptive pooling -> [B, C, 1, 1]
                seq_features = self.sequence_encoders[seq_name](seq_input)

                # Flatten to [B, feature_dim]
                seq_features = seq_features.view(batch_size, -1)

                # Handle missing sequences via availability mask
                if sequence_available is not None and seq_name in sequence_available:
                    # Zero out features for missing sequences
                    available_mask = (
                        sequence_available[seq_name].float().unsqueeze(1)
                    )  # [B, 1]
                    seq_features = seq_features * available_mask

                sequence_features.append(seq_features)
            else:
                # Missing sequence - use zero features with correct dimension
                feature_dim = self.feature_dims[seq_name]
                zero_features = torch.zeros(
                    batch_size, feature_dim, device=next(self.parameters()).device
                )
                sequence_features.append(zero_features)

        # Concatenate all sequence features -> [B, sum(feature_dims)]
        fused_features = torch.cat(
            sequence_features, dim=1
        )  # [B, sum of all feature dims]

        # Add IVD level encoding if enabled
        if self.use_ivd_encoding and ivd_levels is not None:
            # Ensure IVD levels are in valid range (1-5), clamp to avoid errors
            ivd_levels_clamped = torch.clamp(ivd_levels, 1, 5).long()

            if self.ivd_encoding_mode == "positional":
                # Positional encoding: use nn.Embedding to get learned representation
                ivd_embedded = self.ivd_embedding(ivd_levels_clamped)  # [B, 16]
                fused_features = torch.cat([fused_features, ivd_embedded], dim=1)
            elif self.ivd_encoding_mode == "label":
                # Label encoding: normalize IVD levels to [0, 1] and concatenate as single feature
                # IVD levels are 1-5, so normalize to (ivd_level - 1) / 4 = [0, 1]
                ivd_normalized = (ivd_levels_clamped.float() - 1.0) / 4.0  # [B]
                ivd_normalized = ivd_normalized.unsqueeze(1)  # [B, 1]
                fused_features = torch.cat([fused_features, ivd_normalized], dim=1)
        elif self.use_ivd_encoding:
            # If IVD encoding is enabled but no levels provided, use zero features
            device = next(self.parameters()).device
            if self.ivd_encoding_mode == "positional":
                zero_ivd = torch.zeros(batch_size, 16, device=device)
            elif self.ivd_encoding_mode == "label":
                zero_ivd = torch.zeros(batch_size, 1, device=device)
            fused_features = torch.cat([fused_features, zero_ivd], dim=1)

        # Apply separate feedforward for each pathology
        pathology_logits = []
        for head in self.pathology_heads:
            logit = head(fused_features).squeeze(1)  # [B, 1] -> [B]
            pathology_logits.append(logit)

        # Stack to get [B, num_labels]
        logits = torch.stack(pathology_logits, dim=1)

        return logits

    def predict_proba(self, sequences, sequence_available=None):
        """Get probability predictions (with sigmoid)."""
        with torch.no_grad():
            logits = self.forward(sequences, sequence_available)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, sequences, sequence_available=None, threshold=0.5):
        """Get binary predictions (with threshold)."""
        probs = self.predict_proba(sequences, sequence_available)
        return (probs >= threshold).float()


class PathologyVisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) based model for multi-label pathology classification.

    Vision Transformer uses self-attention mechanisms to capture long-range dependencies
    in images. It's particularly effective for medical imaging tasks where global context matters.

    Outputs 4 independent logits (one per pathology type):
    - disc_herniation
    - disc_bulging
    - spondylolisthesis
    - disc_narrowing

    Architecture:
    - Image patches: 16x16 patches from 224x224 input (14x14 = 196 patches)
    - Positional embeddings: Learnable positional embeddings
    - Transformer encoder: Multi-head self-attention + MLP blocks
    - Classification head: Linear layer for multi-label output
    """

    def __init__(
        self,
        architecture: Literal[
            "vit_base_patch16_224", "vit_large_patch16_224"
        ] = "vit_base_patch16_224",
        num_labels: int = 4,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        """
        Initialize PathologyVisionTransformer model.

        Args:
            architecture: ViT architecture to use
            num_labels: Number of output labels (default: 4 for pathology types)
            pretrained: Whether to use pretrained weights (uses timm if available)
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            dropout_rate: Dropout rate before classifier (0.0 to disable, default: 0.5)
            img_size: Input image size (default: 224)
            patch_size: Patch size for image splitting (default: 16)
        """
        super(PathologyVisionTransformer, self).__init__()

        self.architecture = architecture
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.img_size = img_size
        self.patch_size = patch_size

        # Use timm if available (recommended)
        if TIMM_AVAILABLE and pretrained:
            try:
                # Load pretrained ViT from timm
                if architecture == "vit_base_patch16_224":
                    self.backbone = timm.create_model(
                        "vit_base_patch16_224", pretrained=True, num_classes=0
                    )
                    embed_dim = 768
                elif architecture == "vit_large_patch16_224":
                    self.backbone = timm.create_model(
                        "vit_large_patch16_224", pretrained=True, num_classes=0
                    )
                    embed_dim = 1024
                else:
                    raise ValueError(f"Unsupported architecture: {architecture}")

                # Modify first layer for grayscale input if needed
                if in_channels == 1:
                    # Replace first conv layer to accept 1 channel instead of 3
                    original_conv = self.backbone.patch_embed.proj
                    self.backbone.patch_embed.proj = nn.Conv2d(
                        in_channels=1,
                        out_channels=original_conv.out_channels,
                        kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride,
                        padding=original_conv.padding,
                        bias=original_conv.bias is not None,
                    )
                    # Initialize with pretrained weights (average across RGB channels)
                    if pretrained:
                        with torch.no_grad():
                            self.backbone.patch_embed.proj.weight.data = (
                                original_conv.weight.data.mean(dim=1, keepdim=True)
                            )

                # Use timm's feature extraction
                self.use_timm = True
                self.embed_dim = embed_dim

            except Exception as e:
                print(
                    f"Warning: Could not load timm model: {e}. Using custom implementation."
                )
                self.use_timm = False
                self._build_custom_vit(
                    embed_dim=768 if architecture == "vit_base_patch16_224" else 1024
                )
        else:
            # Use custom implementation
            self.use_timm = False
            embed_dim = 768 if architecture == "vit_base_patch16_224" else 1024
            self._build_custom_vit(embed_dim=embed_dim)

        # Add dropout for regularization
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Add custom classifier for multi-label output
        self.classifier = nn.Linear(self.embed_dim, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _build_custom_vit(self, embed_dim: int = 768):
        """Build custom ViT implementation when timm is not available."""
        num_patches = (self.img_size // self.patch_size) ** 2

        # Patch embedding: convert image to patches
        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Transformer encoder
        num_layers = 12 if "base" in self.architecture else 24
        num_heads = 12 if "base" in self.architecture else 16

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Logits tensor of shape [batch_size, num_labels]
            (No sigmoid applied - use with BCEWithLogitsLoss)
        """
        if self.use_timm:
            # Use timm's forward pass
            features = self.backbone(x)  # [B, embed_dim]
        else:
            # Custom ViT forward pass
            B = x.shape[0]

            # Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
            x = self.patch_embed(x)  # [B, embed_dim, H', W']
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

            # Add class token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

            # Add positional embedding
            x = x + self.pos_embed

            # Transformer encoder
            x = self.transformer(x)  # [B, num_patches+1, embed_dim]

            # Use class token for classification
            x = self.norm(x)
            features = x[:, 0]  # [B, embed_dim] - use class token

        # Apply dropout
        features = self.dropout(features)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (with sigmoid)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions (with threshold)."""
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


def create_pathology_model(
    architecture: Union[
        str,
        Literal[
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
            "multi_sequence_fusion",
            "transformer_fusion",
            "adaptive_attention",
        ],
    ] = "resnet18",
    num_labels: int = 4,
    pretrained: bool = True,
    in_channels: int = 1,
    dropout_rate: float = 0.5,
    **kwargs,
):
    """
    Factory function to create pathology classification model.

    BREAKING CHANGE: All architectures now use multi-sequence fusion by default.
    The architecture name determines the backbone_type used in SequenceEncoder.

    Supports multiple architectures (all use multi-sequence fusion):
    - ResNet: resnet18, resnet34, resnet50 (uses ResNet backbone in SequenceEncoder)
    - EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2 (uses EfficientNet backbone)
    - DenseNet: densenet121, densenet169, densenet201 (uses DenseNet backbone)
    - Vision Transformer: vit_base_patch16_224, vit_large_patch16_224 (uses ViT backbone)
    - Multi-Sequence Fusion: multi_sequence_fusion (explicit multi-sequence, configurable backbone)
    - Transformer Hierarchical Fusion: transformer_fusion
    - Adaptive Multi-Scale Attention Network: adaptive_attention

    Args:
        architecture: Model architecture to use
        num_labels: Number of output labels (default: 4)
        pretrained: Whether to use ImageNet pretrained weights
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        dropout_rate: Dropout rate before classifier (0.0 to disable, default: 0.5)
        **kwargs: Additional arguments for specific architectures
            - For multi_sequence_fusion:
              - sequences: List of sequence names (default: ['sag_t2', 'sag_t1', 'ax_t2', 'sag_stir'])
              - feature_dims: Feature dimensions for each scale (default: [256, 512, 1024])
              - fusion_dim: Dimension for fused features (default: 512)
              - attention_heads: Number of attention heads (default: 8)
              - attention_dropout: Dropout rate for attention (default: 0.1)
              - head_hidden_dim: Hidden dimension for class-specific heads (default: 256)
              - head_dropout: Dropout rate for heads (default: 0.3)
            - For transformer_fusion:
              - sequences: List of sequence names (default: ['sag_t2', 'ax_t2', 'sag_stir'])
              - feature_dim: Feature dimension for transformer (default: 512)
              - num_transformer_layers: Number of transformer layers in encoder (default: 2)
              - num_attention_heads: Number of attention heads (default: 8)
              - attention_dropout: Dropout rate for attention (default: 0.1)
              - head_hidden_dim: Hidden dimension for class-specific heads (default: 256)
              - head_dropout: Dropout rate for heads (default: 0.3)

    Returns:
        Model instance (PathologyResNet, PathologyEfficientNet, PathologyDenseNet,
                       PathologyVisionTransformer, or MultiSequenceMultiScaleFusion)

    Example:
        >>> model = create_pathology_model(architecture='efficientnet_b0', pretrained=True)
        >>> model = create_pathology_model(architecture='densenet121', pretrained=True)
        >>> model = create_pathology_model(architecture='vit_base_patch16_224', pretrained=True)
        >>> model = create_pathology_model(architecture='multi_sequence_fusion', sequences=['sag_t2', 'ax_t2'])
    """
    # BREAKING CHANGE: All architectures now use multi-sequence fusion
    # Architecture name determines the backbone_type used in SequenceEncoder
    if (
        architecture.startswith("resnet")
        or architecture.startswith("efficientnet")
        or architecture.startswith("densenet")
        or architecture.startswith("vit")
    ):
        if not MULTI_SEQUENCE_AVAILABLE:
            raise ImportError(
                "Multi-sequence fusion model requires multi_sequence_fusion_model module"
            )

        # Get sequences from kwargs or use default
        sequences = kwargs.get("sequences", ["sag_t2", "ax_t2", "sag_stir"])
        feature_dims = kwargs.get("feature_dims", [256, 512, 1024])
        fusion_dim = kwargs.get("fusion_dim", 512)
        attention_heads = kwargs.get("attention_heads", 8)
        attention_dropout = kwargs.get("attention_dropout", 0.1)
        head_hidden_dim = kwargs.get("head_hidden_dim", 256)
        head_dropout = kwargs.get("head_dropout", 0.3)
        # Enhanced parameters
        use_pathology_specific_attention = kwargs.get(
            "use_pathology_specific_attention", False
        )
        use_clinical_priors = kwargs.get("use_clinical_priors", True)
        prior_strength = kwargs.get("prior_strength", 0.3)

        # Use architecture name as backbone_type
        backbone_type = architecture

        return MultiSequenceMultiScaleFusion(
            sequences=sequences,
            num_labels=num_labels,
            backbone_type=backbone_type,
            pretrained=pretrained,
            in_channels=in_channels,
            feature_dims=feature_dims,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            use_pathology_specific_attention=use_pathology_specific_attention,
            use_clinical_priors=use_clinical_priors,
            prior_strength=prior_strength,
        )
    elif architecture == "multi_sequence_fusion":
        if not MULTI_SEQUENCE_AVAILABLE:
            raise ImportError(
                "Multi-sequence fusion model requires multi_sequence_fusion_model module"
            )
        # Get sequences from kwargs or use default
        sequences = kwargs.get("sequences", ["sag_t2", "ax_t2", "sag_stir"])
        backbone_type = kwargs.get("backbone_type", "densenet121")
        feature_dims = kwargs.get("feature_dims", [256, 512, 1024])
        fusion_dim = kwargs.get("fusion_dim", 512)
        attention_heads = kwargs.get("attention_heads", 8)
        attention_dropout = kwargs.get("attention_dropout", 0.1)
        head_hidden_dim = kwargs.get("head_hidden_dim", 256)
        head_dropout = kwargs.get("head_dropout", 0.3)
        # Enhanced parameters
        use_pathology_specific_attention = kwargs.get(
            "use_pathology_specific_attention", False
        )
        use_clinical_priors = kwargs.get("use_clinical_priors", True)
        prior_strength = kwargs.get("prior_strength", 0.3)

        return MultiSequenceMultiScaleFusion(
            sequences=sequences,
            num_labels=num_labels,
            backbone_type=backbone_type,
            pretrained=pretrained,
            in_channels=in_channels,
            feature_dims=feature_dims,
            fusion_dim=fusion_dim,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            use_pathology_specific_attention=use_pathology_specific_attention,
            use_clinical_priors=use_clinical_priors,
            prior_strength=prior_strength,
        )
    elif architecture == "transformer_fusion":
        if not TRANSFORMER_FUSION_AVAILABLE:
            raise ImportError(
                "Transformer fusion model requires transformer_fusion_model module"
            )
        # Get sequences from kwargs or use default
        sequences = kwargs.get("sequences", ["sag_t2", "ax_t2", "sag_stir"])
        feature_dim = kwargs.get("feature_dim", 512)
        num_transformer_layers = kwargs.get("num_transformer_layers", 2)
        num_attention_heads = kwargs.get("num_attention_heads", 8)
        attention_dropout = kwargs.get("attention_dropout", 0.1)
        head_hidden_dim = kwargs.get("head_hidden_dim", 256)
        head_dropout = kwargs.get("head_dropout", 0.3)

        return TransformerHierarchicalFusion(
            sequences=sequences,
            num_labels=num_labels,
            pretrained=pretrained,
            in_channels=in_channels,
            feature_dim=feature_dim,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
        )
    elif architecture == "adaptive_attention":
        from models.adaptive_attention_model import AdaptiveMultiScaleAttentionNetwork

        # Get backbone from kwargs or use default
        backbone_name = kwargs.get("backbone", "efficientnet_b0")
        attention_dim = kwargs.get("attention_dim", 256)
        num_attention_heads = kwargs.get("num_attention_heads", 8)
        attention_dropout = kwargs.get("attention_dropout", 0.1)
        fusion_method = kwargs.get(
            "fusion_method", "adaptive"
        )  # 'adaptive', 'concat', 'weighted'

        return AdaptiveMultiScaleAttentionNetwork(
            backbone_name=backbone_name,
            num_labels=num_labels,
            pretrained=pretrained,
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            attention_dim=attention_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            fusion_method=fusion_method,
        )
    elif architecture == "simple_multi_sequence_fusion":
        # Get sequences from kwargs or use default
        sequences = kwargs.get("sequences", ["sag_t2", "ax_t2", "sag_stir"])
        backbone_type = kwargs.get("backbone_type", "densenet121")
        projection_dim = kwargs.get("projection_dim", 10)
        hidden_dim = kwargs.get("hidden_dim", 128)
        use_ivd_encoding = kwargs.get("use_ivd_encoding", True)

        return SimpleMultiSequenceFusion(
            sequences=sequences,
            num_labels=num_labels,
            backbone_type=backbone_type,
            pretrained=pretrained,
            in_channels=in_channels,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            use_ivd_encoding=use_ivd_encoding,
        )
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Choose from: resnet18/34/50, efficientnet_b0/b1/b2, densenet121/169/201, "
            f"vit_base_patch16_224/vit_large_patch16_224, multi_sequence_fusion, transformer_fusion, adaptive_attention, simple_multi_sequence_fusion"
        )


# For backward compatibility
PathologyModel = PathologyResNet
