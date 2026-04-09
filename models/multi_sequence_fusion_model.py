"""
Multi-sequence multi-scale fusion model for pathology classification.

This module implements a model that fuses features from multiple MRI sequences
at multiple scales to improve performance, especially on rare pathology classes.

Enhanced version includes pathology-specific attention mechanisms that learn
different sequence importance weights for each pathology type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import math

try:
    from models.pathology_aware_attention import PathologyAwareCrossSequenceAttention

    PATHOLOGY_AWARE_ATTENTION_AVAILABLE = True
except ImportError:
    PATHOLOGY_AWARE_ATTENTION_AVAILABLE = False
    PathologyAwareCrossSequenceAttention = None


class SequenceEncoder(nn.Module):
    """
    Encoder for a single MRI sequence with multi-scale feature extraction.

    Supports multiple backbone types (ResNet, DenseNet, EfficientNet, ViT) with
    Feature Pyramid Network (FPN) approach to extract features at multiple scales.
    """

    def __init__(
        self,
        sequence_name: str,
        backbone_type: str,
        pretrained: bool = True,
        in_channels: int = 1,
        feature_dims: List[int] = [256, 512, 1024],
    ):
        """
        Initialize SequenceEncoder.

        Args:
            sequence_name: Name of the sequence (for identification)
            backbone_type: Backbone type ('resnet18', 'resnet34', 'resnet50',
                          'densenet121', 'densenet169', 'densenet201',
                          'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                          'vit_base_patch16_224', 'vit_large_patch16_224')
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels (1 for grayscale)
            feature_dims: Feature dimensions for each scale [scale1, scale2, scale3]
        """
        super(SequenceEncoder, self).__init__()

        self.sequence_name = sequence_name
        self.backbone_type = backbone_type
        self.feature_dims = feature_dims

        # Load backbone based on type
        if backbone_type.startswith("resnet"):
            self._build_resnet_backbone(backbone_type, pretrained, in_channels)
        elif backbone_type.startswith("densenet"):
            self._build_densenet_backbone(backbone_type, pretrained, in_channels)
        elif backbone_type.startswith("efficientnet"):
            self._build_efficientnet_backbone(backbone_type, pretrained, in_channels)
        elif backbone_type.startswith("vit"):
            self._build_vit_backbone(backbone_type, pretrained, in_channels)
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")

        # Projection layers to match desired feature dimensions
        self.scale1_proj = nn.Conv2d(self.scale1_channels, feature_dims[0], 1)
        self.scale2_proj = nn.Conv2d(self.scale2_channels, feature_dims[1], 1)
        self.scale3_proj = nn.Conv2d(self.scale3_channels, feature_dims[2], 1)

        # Batch normalization for each scale
        self.scale1_bn = nn.BatchNorm2d(feature_dims[0])
        self.scale2_bn = nn.BatchNorm2d(feature_dims[1])
        self.scale3_bn = nn.BatchNorm2d(feature_dims[2])

    def _build_resnet_backbone(
        self, backbone_type: str, pretrained: bool, in_channels: int
    ):
        """Build ResNet backbone and extract multi-scale features."""
        if backbone_type == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            self.scale1_channels = 64
            self.scale2_channels = 128
            self.scale3_channels = 256
        elif backbone_type == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            self.scale1_channels = 64
            self.scale2_channels = 128
            self.scale3_channels = 256
        elif backbone_type == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            self.scale1_channels = 256
            self.scale2_channels = 512
            self.scale3_channels = 1024
        else:
            raise ValueError(f"Unsupported ResNet architecture: {backbone_type}")

        # Modify first layer for grayscale input
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

        # Extract layers for multi-scale features
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.backbone_type_name = "resnet"

    def _build_densenet_backbone(
        self, backbone_type: str, pretrained: bool, in_channels: int
    ):
        """Build DenseNet backbone and extract multi-scale features."""
        if backbone_type == "densenet121":
            backbone = models.densenet121(pretrained=pretrained)
            self.scale1_channels = 256
            self.scale2_channels = 512
            self.scale3_channels = 1024
        elif backbone_type == "densenet169":
            backbone = models.densenet169(pretrained=pretrained)
            self.scale1_channels = 256
            self.scale2_channels = 512
            self.scale3_channels = 1664
        elif backbone_type == "densenet201":
            backbone = models.densenet201(pretrained=pretrained)
            self.scale1_channels = 256
            self.scale2_channels = 512
            self.scale3_channels = 1920
        else:
            raise ValueError(f"Unsupported DenseNet architecture: {backbone_type}")

        # Modify first layer for grayscale input
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

        # Extract feature layers at different scales
        self.features = backbone.features
        self.backbone_type_name = "densenet"

    def _build_efficientnet_backbone(
        self, backbone_type: str, pretrained: bool, in_channels: int
    ):
        """Build EfficientNet backbone and extract multi-scale features."""
        try:
            from torchvision.models import (
                efficientnet_b0,
                efficientnet_b1,
                efficientnet_b2,
            )
            from torchvision.models import (
                EfficientNet_B0_Weights,
                EfficientNet_B1_Weights,
                EfficientNet_B2_Weights,
            )
        except ImportError:
            raise ImportError("EfficientNet models require torchvision >= 0.13.0")

        if backbone_type == "efficientnet_b0":
            if pretrained:
                backbone = efficientnet_b0(
                    weights=EfficientNet_B0_Weights.IMAGENET1K_V1
                )
            else:
                backbone = efficientnet_b0(weights=None)
            self.scale1_channels = 40
            self.scale2_channels = 112
            self.scale3_channels = 320
        elif backbone_type == "efficientnet_b1":
            if pretrained:
                backbone = efficientnet_b1(
                    weights=EfficientNet_B1_Weights.IMAGENET1K_V1
                )
            else:
                backbone = efficientnet_b1(weights=None)
            self.scale1_channels = 40
            self.scale2_channels = 112
            self.scale3_channels = 320
        elif backbone_type == "efficientnet_b2":
            if pretrained:
                backbone = efficientnet_b2(
                    weights=EfficientNet_B2_Weights.IMAGENET1K_V1
                )
            else:
                backbone = efficientnet_b2(weights=None)
            self.scale1_channels = 48
            self.scale2_channels = 120
            self.scale3_channels = 352
        else:
            raise ValueError(f"Unsupported EfficientNet architecture: {backbone_type}")

        # Modify first layer for grayscale input
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
                    ].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        self.features = backbone.features
        self.backbone_type_name = "efficientnet"

    def _build_vit_backbone(
        self, backbone_type: str, pretrained: bool, in_channels: int
    ):
        """Build ViT backbone - note: ViT requires special handling for multi-scale."""
        try:
            import timm
        except ImportError:
            raise ImportError("ViT models require timm. Install with: pip install timm")

        if backbone_type == "vit_base_patch16_224":
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrained, num_classes=0
            )
            self.scale1_channels = 768
            self.scale2_channels = 768
            self.scale3_channels = 768
        elif backbone_type == "vit_large_patch16_224":
            self.backbone = timm.create_model(
                "vit_large_patch16_224", pretrained=pretrained, num_classes=0
            )
            self.scale1_channels = 1024
            self.scale2_channels = 1024
            self.scale3_channels = 1024
        else:
            raise ValueError(f"Unsupported ViT architecture: {backbone_type}")

        # Modify first layer for grayscale input
        if in_channels == 1:
            original_conv = self.backbone.patch_embed.proj
            self.backbone.patch_embed.proj = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    self.backbone.patch_embed.proj.weight.data = (
                        original_conv.weight.data.mean(dim=1, keepdim=True)
                    )

        self.backbone_type_name = "vit"

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from input sequence.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            List of feature tensors at 3 scales:
            - scale1: [B, feature_dims[0], H/4, W/4] (early features)
            - scale2: [B, feature_dims[1], H/8, W/8] (mid-level features)
            - scale3: [B, feature_dims[2], H/16, W/16] (high-level features)
        """
        if self.backbone_type_name == "resnet":
            return self._forward_resnet(x)
        elif self.backbone_type_name == "densenet":
            return self._forward_densenet(x)
        elif self.backbone_type_name == "efficientnet":
            return self._forward_efficientnet(x)
        elif self.backbone_type_name == "vit":
            return self._forward_vit(x)
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone_type_name}")

    def _forward_resnet(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from ResNet."""
        features = []

        # Initial conv and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer1 (scale1)
        x = self.layer1(x)
        scale1 = self.scale1_bn(self.scale1_proj(x))
        features.append(scale1)

        # Layer2 (scale2)
        x = self.layer2(x)
        scale2 = self.scale2_bn(self.scale2_proj(x))
        features.append(scale2)

        # Layer3 (scale3)
        x = self.layer3(x)
        scale3 = self.scale3_bn(self.scale3_proj(x))
        features.append(scale3)

        return features

    def _forward_densenet(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from DenseNet."""
        features = []

        # Forward through DenseNet features
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        # DenseBlock1 + Transition1 (scale1)
        x = self.features.denseblock1(x)
        scale1 = self.scale1_bn(self.scale1_proj(x))
        features.append(scale1)

        x = self.features.transition1(x)

        # DenseBlock2 + Transition2 (scale2)
        x = self.features.denseblock2(x)
        scale2 = self.scale2_bn(self.scale2_proj(x))
        features.append(scale2)

        x = self.features.transition2(x)

        # DenseBlock3 + Transition3
        x = self.features.denseblock3(x)
        x = self.features.transition3(x)

        # DenseBlock4 (scale3)
        x = self.features.denseblock4(x)
        scale3 = self.scale3_bn(self.scale3_proj(x))
        features.append(scale3)

        return features

    def _forward_efficientnet(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from EfficientNet."""
        features = []

        # Forward through EfficientNet features
        # Scale 1: After MBConv block 3 (index 3)
        for i in range(4):
            x = self.features[i](x)
        scale1 = self.scale1_bn(self.scale1_proj(x))
        features.append(scale1)

        # Scale 2: After MBConv block 5 (index 5)
        for i in range(4, 6):
            x = self.features[i](x)
        scale2 = self.scale2_bn(self.scale2_proj(x))
        features.append(scale2)

        # Scale 3: After MBConv block 7 (index 7)
        for i in range(6, 8):
            x = self.features[i](x)
        scale3 = self.scale3_bn(self.scale3_proj(x))
        features.append(scale3)

        return features

    def _forward_vit(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from ViT."""
        features = []

        # ViT forward pass
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)

        # Extract features from different transformer blocks
        # Scale 1: After block 3
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i == 2:  # After block 3 (0-indexed)
                # Reshape to spatial format for projection
                B, N, C = x.shape
                H = W = int(N**0.5)
                x_reshaped = (
                    x[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
                )  # Remove cls token
                scale1 = self.scale1_bn(self.scale1_proj(x_reshaped))
                features.append(scale1)

        # Scale 2: After block 6
        for i, block in enumerate(self.backbone.blocks[3:], start=3):
            x = block(x)
            if i == 5:  # After block 6
                B, N, C = x.shape
                H = W = int(N**0.5)
                x_reshaped = x[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
                scale2 = self.scale2_bn(self.scale2_proj(x_reshaped))
                features.append(scale2)

        # Scale 3: After block 9 (or final)
        for i, block in enumerate(self.backbone.blocks[6:], start=6):
            x = block(x)
            if i == 8 or i == len(self.backbone.blocks) - 1:  # After block 9 or final
                B, N, C = x.shape
                H = W = int(N**0.5)
                x_reshaped = x[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
                scale3 = self.scale3_bn(self.scale3_proj(x_reshaped))
                features.append(scale3)
                break

        return features


class CrossSequenceAttention(nn.Module):
    """
    Cross-sequence attention mechanism to fuse features from different sequences.

    Uses self-attention to learn which sequences are most informative
    for each pathology type.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize CrossSequenceAttention.

        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossSequenceAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, (
            "feature_dim must be divisible by num_heads"
        )

        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        sequence_features: Dict[str, torch.Tensor],
        sequence_available: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Apply cross-sequence attention.

        Args:
            sequence_features: Dict mapping sequence names to feature tensors [B, feature_dim]
            sequence_available: Optional dict mapping sequence names to availability flags [B]

        Returns:
            Attended features [B, feature_dim]
        """
        # Stack sequence features
        sequence_names = sorted(sequence_features.keys())
        num_sequences = len(sequence_names)

        if num_sequences == 0:
            raise ValueError("No sequence features provided")

        # Stack: [B, num_sequences, feature_dim]
        stacked_features = torch.stack(
            [sequence_features[name] for name in sequence_names], dim=1
        )
        batch_size = stacked_features.size(0)

        # Apply masking if sequence availability is provided
        if sequence_available is not None:
            # Create mask: [B, num_sequences]
            # Ensure all sequence names have corresponding availability flags
            batch_size = stacked_features.size(0)
            device = stacked_features.device
            mask_list = []
            for name in sequence_names:
                if name in sequence_available:
                    mask_list.append(sequence_available[name])
                else:
                    # Key missing - default to unavailable
                    mask_list.append(
                        torch.zeros(batch_size, device=device, dtype=torch.bool)
                    )
            mask = torch.stack(mask_list, dim=1)
            # Expand for attention: [B, 1, num_sequences]
            mask = mask.unsqueeze(1).float()
        else:
            mask = None

        # Compute Q, K, V
        Q = self.q_proj(stacked_features)  # [B, num_sequences, feature_dim]
        K = self.k_proj(stacked_features)  # [B, num_sequences, feature_dim]
        V = self.v_proj(stacked_features)  # [B, num_sequences, feature_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_sequences, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, num_heads, num_sequences, head_dim]
        K = K.view(batch_size, num_sequences, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.view(batch_size, num_sequences, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, num_heads, num_sequences, num_sequences]

        # Apply mask if provided
        if mask is not None:
            # Expand mask for all heads
            mask_expanded = mask.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # [B, num_heads, 1, num_sequences]
            # Mask out unavailable sequences
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(
            attn_weights, V
        )  # [B, num_heads, num_sequences, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # [B, num_sequences, num_heads, head_dim]
        attn_output = attn_output.view(
            batch_size, num_sequences, self.feature_dim
        )  # [B, num_sequences, feature_dim]

        # Aggregate across sequences (weighted average)
        # Use attention weights to weight the sequences
        attn_weights_avg = attn_weights.mean(
            dim=1
        )  # [B, num_sequences, num_sequences] -> average over heads
        sequence_weights = attn_weights_avg.mean(
            dim=1
        )  # [B, num_sequences] -> average over query positions

        # Weighted sum of attended features
        fused_features = (attn_output * sequence_weights.unsqueeze(-1)).sum(
            dim=1
        )  # [B, feature_dim]

        # Output projection
        output = self.out_proj(fused_features)
        output = self.norm(output)

        return output


class MultiScaleFusion(nn.Module):
    """
    Fuse features from multiple scales using learned weights.
    """

    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 512,
    ):
        """
        Initialize MultiScaleFusion.

        Args:
            feature_dims: Feature dimensions for each scale
            output_dim: Output feature dimension
        """
        super(MultiScaleFusion, self).__init__()

        self.feature_dims = feature_dims
        self.num_scales = len(feature_dims)

        # Project each scale to output dimension
        self.scale_projections = nn.ModuleList(
            [nn.Linear(dim, output_dim) for dim in feature_dims]
        )

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

        # Layer normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features.

        Args:
            scale_features: List of feature tensors [B, feature_dim_i] for each scale

        Returns:
            Fused features [B, output_dim]
        """
        # Project each scale to output dimension
        projected_features = []
        for i, (features, proj) in enumerate(
            zip(scale_features, self.scale_projections)
        ):
            # Global average pooling if needed (in case features are 2D)
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)

            projected = proj(features)
            projected_features.append(projected)

        # Stack: [B, num_scales, output_dim]
        stacked = torch.stack(projected_features, dim=1)

        # Weighted combination
        weights = F.softmax(self.scale_weights, dim=0)
        fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)  # [B, output_dim]

        # Normalize
        fused = self.norm(fused)

        return fused


class ClassSpecificHead(nn.Module):
    """
    Class-specific classification head for each pathology type.

    Allows different architectures/weights for different classes,
    which is especially useful for rare classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Initialize ClassSpecificHead.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(ClassSpecificHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Binary classification per label
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.head(x)


class MultiSequenceMultiScaleFusion(nn.Module):
    """
    Multi-sequence multi-scale fusion model for pathology classification.

    Architecture:
    1. Sequence Encoders: Extract multi-scale features from each sequence
    2. Cross-Sequence Attention: Fuse features across sequences
    3. Multi-Scale Fusion: Combine features from different scales
    4. Class-Specific Heads: Separate head for each pathology type
    """

    def __init__(
        self,
        sequences: List[str] = ["sag_t2", "ax_t2", "sag_stir"],
        num_labels: int = 4,
        backbone_type: str = "densenet121",
        pretrained: bool = True,
        in_channels: int = 1,
        feature_dims: List[int] = [256, 512, 1024],
        fusion_dim: int = 512,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
        use_pathology_specific_attention: bool = False,
        use_clinical_priors: bool = True,
        prior_strength: float = 0.3,
    ):
        """
        Initialize MultiSequenceMultiScaleFusion model.

        Args:
            sequences: List of sequence names to use
            num_labels: Number of output labels (default: 4)
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
            head_hidden_dim: Hidden dimension for class-specific heads
            head_dropout: Dropout rate for heads
            use_pathology_specific_attention: Whether to use pathology-aware attention
            use_clinical_priors: Whether to initialize with clinical knowledge (if pathology-aware)
            prior_strength: Strength of clinical prior (0.0-1.0, if pathology-aware)
        """
        super(MultiSequenceMultiScaleFusion, self).__init__()

        self.sequences = [seq.lower() for seq in sequences]
        self.num_labels = num_labels
        self.backbone_type = backbone_type
        self.feature_dims = feature_dims
        self.fusion_dim = fusion_dim
        self.use_pathology_specific_attention = use_pathology_specific_attention

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

        # Cross-sequence attention for each scale
        if use_pathology_specific_attention and PATHOLOGY_AWARE_ATTENTION_AVAILABLE:
            # Use pathology-aware attention
            self.scale_attentions = nn.ModuleList(
                [
                    PathologyAwareCrossSequenceAttention(
                        feature_dim=feature_dims[i],
                        num_pathologies=num_labels,
                        sequence_names=self.sequences,
                        num_heads=attention_heads,
                        dropout=attention_dropout,
                        use_clinical_priors=use_clinical_priors,
                        prior_strength=prior_strength,
                    )
                    for i in range(len(feature_dims))
                ]
            )
        else:
            # Use standard attention (backward compatibility)
            if use_pathology_specific_attention:
                import warnings

                warnings.warn(
                    "Pathology-aware attention not available, using standard attention"
                )
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

        # Class-specific heads
        self.class_heads = nn.ModuleList(
            [
                ClassSpecificHead(
                    input_dim=fusion_dim,
                    hidden_dim=head_hidden_dim,
                    dropout=head_dropout,
                )
                for _ in range(num_labels)
            ]
        )

    def forward(
        self,
        sequences: Dict[str, torch.Tensor],
        sequence_available: Optional[Dict[str, torch.Tensor]] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[int, Dict[int, torch.Tensor]]]]:
        """
        Forward pass.

        Args:
            sequences: Dict mapping sequence names to input tensors [B, C, H, W]
            sequence_available: Optional dict mapping sequence names to availability flags [B]
            return_attention_weights: Whether to return attention weights for interpretability

        Returns:
            Tuple of:
            - Logits [B, num_labels]
            - Attention weights dict {scale_idx: {pathology_idx: [B, num_sequences]}} (if return_attention_weights=True)
        """
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
                        features = F.adaptive_avg_pool2d(features, (1, 1))
                        features = features.view(batch_size, -1)
                    all_scale_features[scale_idx][seq_name] = features
            else:
                # Missing sequence - create zero features
                for scale_idx in range(len(self.feature_dims)):
                    all_scale_features[scale_idx][seq_name] = torch.zeros(
                        batch_size, self.feature_dims[scale_idx], device=device
                    )

        # Apply cross-sequence attention at each scale
        attended_scale_features = []
        all_attention_weights = {} if return_attention_weights else None

        for scale_idx in range(len(self.feature_dims)):
            scale_seq_features = all_scale_features[scale_idx]

            # Get availability for this scale
            if sequence_available is not None:
                scale_available = sequence_available
            else:
                scale_available = {
                    seq: torch.ones(batch_size, device=device, dtype=torch.bool)
                    for seq in self.sequences
                }

            # Apply attention
            if (
                self.use_pathology_specific_attention
                and PATHOLOGY_AWARE_ATTENTION_AVAILABLE
            ):
                # Pathology-aware attention returns attention weights
                attended, attn_weights = self.scale_attentions[scale_idx](
                    scale_seq_features,
                    scale_available,
                    return_attention_weights=return_attention_weights,
                )
                if return_attention_weights and attn_weights is not None:
                    all_attention_weights[scale_idx] = attn_weights
            else:
                # Standard attention
                attended = self.scale_attentions[scale_idx](
                    scale_seq_features, scale_available
                )

            attended_scale_features.append(attended)

        # Fuse multi-scale features
        fused_features = self.multi_scale_fusion(attended_scale_features)

        # Class-specific predictions
        logits_list = []
        for head in self.class_heads:
            logit = head(fused_features)
            logits_list.append(logit)

        # Concatenate: [B, num_labels]
        logits = torch.cat(logits_list, dim=1)

        if return_attention_weights:
            return logits, all_attention_weights
        else:
            return logits, None

    def get_attention_weights(
        self,
        sequences: Dict[str, torch.Tensor],
        sequence_available: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Get attention weights for interpretability.

        Args:
            sequences: Dict mapping sequence names to input tensors [B, C, H, W]
            sequence_available: Optional dict mapping sequence names to availability flags [B]

        Returns:
            Dictionary with attention weights:
            {
                'scale_0': {pathology_idx: [B, num_sequences]},
                'scale_1': {pathology_idx: [B, num_sequences]},
                ...
            }
        """
        if not self.use_pathology_specific_attention:
            return {}

        _, attention_weights = self.forward(
            sequences, sequence_available, return_attention_weights=True
        )

        if attention_weights is None:
            return {}

        # Convert to more readable format
        result = {}
        for scale_idx, path_weights in attention_weights.items():
            scale_key = f"scale_{scale_idx}"
            result[scale_key] = {}

            # Map pathology indices to names
            pathology_names = [
                "disc_herniation",
                "disc_bulging",
                "spondylolisthesis",
                "disc_narrowing",
            ]
            for path_idx, weights in path_weights.items():
                if path_idx < len(pathology_names):
                    result[scale_key][pathology_names[path_idx]] = weights

        return result

    def predict_proba(
        self,
        sequences: Dict[str, torch.Tensor],
        sequence_available: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get probability predictions (with sigmoid)."""
        with torch.no_grad():
            logits, _ = self.forward(
                sequences, sequence_available, return_attention_weights=False
            )
            probs = torch.sigmoid(logits)
        return probs

    def predict(
        self,
        sequences: Dict[str, torch.Tensor],
        sequence_available: Optional[Dict[str, torch.Tensor]] = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Get binary predictions (with threshold)."""
        probs = self.predict_proba(sequences, sequence_available)
        return (probs >= threshold).float()
