"""
Adaptive Multi-Scale Attention Network (AMAN) for Pathology Classification.

This module implements an advanced architecture that combines:
- Multi-scale feature extraction
- Channel and spatial attention mechanisms
- Adaptive feature fusion
- Residual connections for better gradient flow

The architecture is designed to improve performance on multi-label pathology
classification by focusing on relevant features at multiple scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Literal

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


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) - focuses on 'what' features are important."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) - focuses on 'where' features are important."""

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel and spatial attention."""

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply channel attention first
        out = x * self.channel_attention(x)
        # Then apply spatial attention
        out = out * self.spatial_attention(out)
        return out


class MultiScaleFeatureExtractor(nn.Module):
    """Extracts features at multiple scales using adaptive pooling."""

    def __init__(self, in_channels: int, feature_dim: int = 256):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim

        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d(1)  # Global
        self.pool2 = nn.AdaptiveAvgPool2d(2)  # 2x2
        self.pool4 = nn.AdaptiveAvgPool2d(4)  # 4x4

        # Feature projection
        self.proj1 = nn.Linear(in_channels, feature_dim)
        self.proj2 = nn.Linear(in_channels * 4, feature_dim)  # 2x2 = 4
        self.proj4 = nn.Linear(in_channels * 16, feature_dim)  # 4x4 = 16

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Extract multi-scale features
        f1 = self.pool1(x).view(B, C)  # [B, C]
        f2 = self.pool2(x).view(B, C * 4)  # [B, C*4]
        f4 = self.pool4(x).view(B, C * 16)  # [B, C*16]

        # Project to same dimension
        f1 = self.proj1(f1)  # [B, feature_dim]
        f2 = self.proj2(f2)  # [B, feature_dim]
        f4 = self.proj4(f4)  # [B, feature_dim]

        # Stack and normalize
        features = torch.stack([f1, f2, f4], dim=1)  # [B, 3, feature_dim]
        features = self.norm(features)

        return features


class AdaptiveFusion(nn.Module):
    """Adaptive feature fusion using attention mechanism."""

    def __init__(
        self,
        feature_dim: int,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(AdaptiveFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.num_heads = num_heads

        # Multi-head self-attention for scale fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Layer norm and feed-forward
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, multi_scale_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_scale_features: [B, num_scales, feature_dim]

        Returns:
            fused_features: [B, feature_dim]
        """
        B = multi_scale_features.shape[0]

        # Apply learnable scale weights
        weighted_features = multi_scale_features * self.scale_weights.unsqueeze(
            0
        ).unsqueeze(-1)

        # Self-attention for adaptive fusion
        attn_out, _ = self.attention(
            weighted_features, weighted_features, weighted_features
        )
        attn_out = self.norm1(attn_out + weighted_features)  # Residual connection

        # Feed-forward network
        ffn_out = self.ffn(attn_out)
        fused = self.norm2(attn_out + ffn_out)  # Residual connection

        # Global average pooling across scales
        fused_features = fused.mean(dim=1)  # [B, feature_dim]

        return fused_features


class AdaptiveMultiScaleAttentionNetwork(nn.Module):
    """
    Adaptive Multi-Scale Attention Network (AMAN) for multi-label pathology classification.

    Architecture:
    1. Backbone CNN (EfficientNet/DenseNet/ResNet) for feature extraction
    2. CBAM (Channel + Spatial Attention) for feature refinement
    3. Multi-scale feature extraction at different resolutions
    4. Adaptive fusion of multi-scale features using self-attention
    5. Classifier head with dropout for multi-label prediction

    Key innovations:
    - Multi-scale feature extraction captures both local and global patterns
    - Attention mechanisms focus on relevant features
    - Adaptive fusion learns optimal combination of scales
    - Residual connections improve gradient flow
    """

    def __init__(
        self,
        backbone_name: Literal[
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "densenet121",
            "densenet169",
            "resnet50",
        ] = "efficientnet_b0",
        num_labels: int = 4,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
        attention_dim: int = 256,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        fusion_method: Literal["adaptive", "concat", "weighted"] = "adaptive",
    ):
        """
        Initialize Adaptive Multi-Scale Attention Network.

        Args:
            backbone_name: Backbone CNN architecture
            num_labels: Number of output labels
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels (1 for grayscale)
            dropout_rate: Dropout rate before classifier
            attention_dim: Dimension for attention mechanisms
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout rate for attention layers
            fusion_method: Feature fusion method ('adaptive', 'concat', 'weighted')
        """
        super(AdaptiveMultiScaleAttentionNetwork, self).__init__()

        self.backbone_name = backbone_name
        self.num_labels = num_labels
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.attention_dim = attention_dim
        self.fusion_method = fusion_method

        # Build backbone
        self.backbone, self.backbone_features = self._build_backbone(
            backbone_name, pretrained, in_channels
        )

        # Add CBAM attention after backbone
        self.cbam = CBAM(self.backbone_features, reduction=16)

        # Multi-scale feature extraction
        self.multi_scale_extractor = MultiScaleFeatureExtractor(
            in_channels=self.backbone_features, feature_dim=attention_dim
        )

        # Adaptive fusion
        if fusion_method == "adaptive":
            self.fusion = AdaptiveFusion(
                feature_dim=attention_dim,
                num_scales=3,
                num_heads=num_attention_heads,
                dropout=attention_dropout,
            )
            classifier_input_dim = attention_dim
        elif fusion_method == "concat":
            classifier_input_dim = attention_dim * 3
            self.fusion = None
        elif fusion_method == "weighted":
            self.scale_weights = nn.Parameter(torch.ones(3) / 3)
            classifier_input_dim = attention_dim
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Dropout
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Classifier
        self.classifier = nn.Linear(classifier_input_dim, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _build_backbone(self, backbone_name: str, pretrained: bool, in_channels: int):
        """Build backbone CNN and return it with feature dimension."""
        if backbone_name.startswith("efficientnet"):
            if not EFFICIENTNET_AVAILABLE:
                raise ImportError("EfficientNet requires torchvision >= 0.13.0")

            if backbone_name == "efficientnet_b0":
                if pretrained:
                    backbone = efficientnet_b0(
                        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
                    )
                else:
                    backbone = efficientnet_b0(weights=None)
                feature_dim = 1280
            elif backbone_name == "efficientnet_b1":
                if pretrained:
                    backbone = efficientnet_b1(
                        weights=EfficientNet_B1_Weights.IMAGENET1K_V1
                    )
                else:
                    backbone = efficientnet_b1(weights=None)
                feature_dim = 1280
            elif backbone_name == "efficientnet_b2":
                if pretrained:
                    backbone = efficientnet_b2(
                        weights=EfficientNet_B2_Weights.IMAGENET1K_V1
                    )
                else:
                    backbone = efficientnet_b2(weights=None)
                feature_dim = 1408
            else:
                raise ValueError(f"Unsupported EfficientNet: {backbone_name}")

            # Modify first layer for grayscale
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

            # Remove classifier
            backbone = nn.Sequential(*list(backbone.children())[:-1])

        elif backbone_name.startswith("densenet"):
            if backbone_name == "densenet121":
                backbone = models.densenet121(pretrained=pretrained)
                feature_dim = 1024
            elif backbone_name == "densenet169":
                backbone = models.densenet169(pretrained=pretrained)
                feature_dim = 1664
            else:
                raise ValueError(f"Unsupported DenseNet: {backbone_name}")

            # Modify first layer for grayscale
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

            # Use features only
            backbone = backbone.features

        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048

            # Modify first layer for grayscale
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

            # Remove classifier
            backbone = nn.Sequential(*list(backbone.children())[:-1])

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        return backbone, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            logits: [B, num_labels]
        """
        # Backbone feature extraction
        features = self.backbone(x)  # [B, C, H', W']

        # Apply CBAM attention
        features = self.cbam(features)  # [B, C, H', W']

        # Multi-scale feature extraction
        multi_scale_features = self.multi_scale_extractor(
            features
        )  # [B, 3, attention_dim]

        # Adaptive fusion
        if self.fusion_method == "adaptive":
            fused = self.fusion(multi_scale_features)  # [B, attention_dim]
        elif self.fusion_method == "concat":
            fused = multi_scale_features.view(
                multi_scale_features.size(0), -1
            )  # [B, 3*attention_dim]
        elif self.fusion_method == "weighted":
            # Weighted average
            weights = F.softmax(self.scale_weights, dim=0)
            fused = (multi_scale_features * weights.unsqueeze(0).unsqueeze(-1)).sum(
                dim=1
            )  # [B, attention_dim]

        # Dropout
        fused = self.dropout(fused)

        # Classifier
        logits = self.classifier(fused)  # [B, num_labels]

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
