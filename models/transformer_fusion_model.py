"""
Transformer-based Hierarchical Attention Fusion Model for pathology classification.

This model uses transformer encoders with hierarchical attention mechanisms to fuse
features from multiple MRI sequences, with special focus on improving performance
for rare pathology classes.

Key features:
- Transformer-based sequence encoders with positional encoding
- Hierarchical attention (sequence-level, scale-level, feature-level)
- Class-specific attention heads for rare classes
- Residual connections throughout
- Multi-scale feature fusion
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence order."""

    def __init__(self, d_model: int, max_len: int = 10):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model]
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerSequenceEncoder(nn.Module):
    """
    Encoder for a single MRI sequence using CNN backbone + Transformer.
    Extracts multi-scale features and encodes them with transformer.
    """

    def __init__(
        self,
        sequence_name: str,
        pretrained: bool = True,
        in_channels: int = 1,
        feature_dim: int = 512,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(TransformerSequenceEncoder, self).__init__()

        self.sequence_name = sequence_name
        self.feature_dim = feature_dim

        # CNN backbone (DenseNet-121) for feature extraction
        backbone = models.densenet121(pretrained=pretrained)

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

        self.backbone = backbone.features

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection to feature_dim
        self.projection = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder for sequence-level features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, feature_dim] - encoded sequence features
        """
        # Extract features with CNN
        features = self.backbone(x)

        # Global pooling: [B, 1024, H', W'] -> [B, 1024]
        features = self.global_pool(features).flatten(1)

        # Project to feature_dim: [B, 1024] -> [B, feature_dim]
        features = self.projection(features)

        return features


class HierarchicalAttentionFusion(nn.Module):
    """
    Hierarchical attention mechanism for fusing multiple sequences.

    Implements:
    1. Sequence-level attention: Which sequences are most important?
    2. Feature-level attention: Which features are most discriminative?
    3. Class-specific attention: Different attention for each pathology class
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_sequences: int = 4,
        num_labels: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(HierarchicalAttentionFusion, self).__init__()

        self.feature_dim = feature_dim
        self.num_sequences = num_sequences
        self.num_labels = num_labels

        # Sequence-level attention: [B, num_sequences, feature_dim] -> [B, feature_dim]
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.sequence_norm = nn.LayerNorm(feature_dim)

        # Feature-level attention: Self-attention on fused features
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feature_norm = nn.LayerNorm(feature_dim)

        # Class-specific attention heads for rare classes
        self.class_attention_heads = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=feature_dim,
                    num_heads=num_heads // 2,  # Smaller heads for class-specific
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_labels)
            ]
        )
        self.class_norms = nn.ModuleList(
            [nn.LayerNorm(feature_dim) for _ in range(num_labels)]
        )

        # Learnable query for sequence attention
        self.sequence_query = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Learnable queries for class-specific attention
        self.class_queries = nn.Parameter(torch.randn(num_labels, 1, feature_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(
        self,
        sequence_features: Dict[str, torch.Tensor],
        sequence_available: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Fuse sequence features using hierarchical attention.

        Args:
            sequence_features: Dict mapping sequence names to [B, feature_dim]
            sequence_available: Optional dict mapping sequence names to [B] availability flags

        Returns:
            [B, feature_dim] - fused features
        """
        batch_size = next(iter(sequence_features.values())).size(0)
        device = next(iter(sequence_features.values())).device

        # Stack sequences: [B, num_sequences, feature_dim]
        sequence_names = sorted(sequence_features.keys())
        stacked_features = torch.stack(
            [sequence_features[name] for name in sequence_names], dim=1
        )

        # Create mask for missing sequences
        if sequence_available is not None:
            # Ensure sequence_available has all required keys
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
            mask = torch.stack(mask_list, dim=1)  # [B, num_sequences]
            # Invert mask for attention (True = attend, False = mask out)
            attention_mask = ~mask.unsqueeze(1)  # [B, 1, num_sequences]
        else:
            attention_mask = None

        # 1. Sequence-level attention
        # Use learnable query to attend to sequences
        query = self.sequence_query.expand(batch_size, -1, -1)  # [B, 1, feature_dim]
        attended_seq, _ = self.sequence_attention(
            query,
            stacked_features,
            stacked_features,
            key_padding_mask=attention_mask.squeeze(1)
            if attention_mask is not None
            else None,
        )
        attended_seq = self.sequence_norm(attended_seq + query)  # Residual
        attended_seq = attended_seq.squeeze(1)  # [B, feature_dim]

        # 2. Feature-level self-attention
        # Treat features as sequence and apply self-attention
        features_as_seq = attended_seq.unsqueeze(1)  # [B, 1, feature_dim]
        attended_features, _ = self.feature_attention(
            features_as_seq, features_as_seq, features_as_seq
        )
        attended_features = self.feature_norm(attended_features + features_as_seq)
        attended_features = attended_features.squeeze(1)  # [B, feature_dim]

        # 3. Class-specific attention (for rare classes)
        # Each class gets its own attention head
        class_attended = []
        for label_idx in range(self.num_labels):
            class_query = self.class_queries[label_idx].expand(
                batch_size, -1, -1
            )  # [B, 1, feature_dim]
            features_for_attention = attended_features.unsqueeze(
                1
            )  # [B, 1, feature_dim]
            class_att, _ = self.class_attention_heads[label_idx](
                class_query, features_for_attention, features_for_attention
            )
            class_att = self.class_norms[label_idx](class_att + class_query)
            class_attended.append(class_att.squeeze(1))  # [B, feature_dim]

        # Combine class-specific features
        class_features = torch.stack(
            class_attended, dim=1
        )  # [B, num_labels, feature_dim]
        # Average pooling across classes (could also use weighted sum)
        class_fused = class_features.mean(dim=1)  # [B, feature_dim]

        # 4. Final fusion with residual connection
        combined = attended_features + class_fused  # Residual connection
        fused = self.fusion(combined)
        fused = fused + combined  # Another residual connection

        return fused


class TransformerHierarchicalFusion(nn.Module):
    """
    Transformer-based Hierarchical Attention Fusion Model.

    Architecture:
    1. Transformer-based sequence encoders (one per sequence)
    2. Hierarchical attention fusion (sequence-level, feature-level, class-specific)
    3. Class-specific prediction heads
    """

    def __init__(
        self,
        sequences: List[str] = ["sag_t2", "ax_t2", "sag_stir"],
        num_labels: int = 4,
        pretrained: bool = True,
        in_channels: int = 1,
        feature_dim: int = 512,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.3,
        attention_dropout: float = 0.1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
    ):
        super(TransformerHierarchicalFusion, self).__init__()

        self.sequences = [s.lower() for s in sequences]
        self.num_labels = num_labels
        self.feature_dim = feature_dim

        # Sequence encoders (one per sequence)
        self.sequence_encoders = nn.ModuleDict(
            {
                seq_name: TransformerSequenceEncoder(
                    sequence_name=seq_name,
                    pretrained=pretrained,
                    in_channels=in_channels,
                    feature_dim=feature_dim,
                    num_transformer_layers=num_transformer_layers,
                    num_heads=num_attention_heads,
                    dropout=attention_dropout,
                )
                for seq_name in self.sequences
            }
        )

        # Hierarchical attention fusion
        self.fusion = HierarchicalAttentionFusion(
            feature_dim=feature_dim,
            num_sequences=len(self.sequences),
            num_labels=num_labels,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

        # Class-specific prediction heads
        self.class_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, head_hidden_dim),
                    nn.LayerNorm(head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(head_hidden_dim, head_hidden_dim // 2),
                    nn.LayerNorm(head_hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(head_hidden_dim // 2, 1),
                )
                for _ in range(num_labels)
            ]
        )

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
            Logits [B, num_labels]
        """
        # Encode each sequence
        sequence_features = {}
        for seq_name in self.sequences:
            if seq_name in sequences and sequences[seq_name] is not None:
                # Encode sequence
                features = self.sequence_encoders[seq_name](sequences[seq_name])
                sequence_features[seq_name] = features
            else:
                # Missing sequence - create zero features
                non_none_sequences = [v for v in sequences.values() if v is not None]
                if not non_none_sequences:
                    raise ValueError(
                        "All sequences are None - cannot determine batch size"
                    )
                batch_size = non_none_sequences[0].size(0)
                device = non_none_sequences[0].device
                sequence_features[seq_name] = torch.zeros(
                    batch_size, self.feature_dim, device=device
                )

        # Fuse sequences with hierarchical attention
        fused_features = self.fusion(sequence_features, sequence_available)

        # Class-specific predictions
        logits_list = []
        for head in self.class_heads:
            logit = head(fused_features)
            logits_list.append(logit)

        # Concatenate: [B, num_labels]
        logits = torch.cat(logits_list, dim=1)

        return logits
