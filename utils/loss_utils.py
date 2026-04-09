"""
Enhanced loss functions for multi-label and multi-class classification with class imbalance.

This module provides:
- Multi-label loss functions: Focal Loss, Asymmetric Loss
- Multi-class loss functions: Weighted CrossEntropyLoss, Focal Loss for multi-class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification.

    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Where:
    - p_t is the predicted probability for the true class
    - alpha balances importance of positive/negative examples
    - gamma focuses learning on hard examples

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for each class (list of length num_classes).
                   If None, uses uniform weighting.
            gamma: Focusing parameter (gamma >= 0). Higher gamma focuses more on hard examples.
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Logits tensor of shape [batch_size, num_classes]
            targets: Binary labels tensor of shape [batch_size, num_classes]

        Returns:
            Loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate BCE component
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Alpha for positive class
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Addresses class imbalance by applying different penalties for
    false positives and false negatives.

    Reference: Ridnik et al. "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        """
        Initialize Asymmetric Loss.

        Args:
            gamma_neg: Focusing parameter for negative examples (false positives)
            gamma_pos: Focusing parameter for positive examples (false negatives)
            clip: Probability clipping threshold
            eps: Small epsilon for numerical stability
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Logits tensor of shape [batch_size, num_classes]
            targets: Binary labels tensor of shape [batch_size, num_classes]

        Returns:
            Loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Clip probabilities for numerical stability
        probs = torch.clamp(probs, min=self.clip, max=1 - self.clip)

        # Calculate asymmetric loss components
        # For positive targets (false negatives are penalized more)
        pos_loss = (
            -targets * torch.log(probs + self.eps) * (1 - probs) ** self.gamma_pos
        )

        # For negative targets (false positives are penalized more)
        neg_loss = (
            -(1 - targets) * torch.log(1 - probs + self.eps) * probs**self.gamma_neg
        )

        # Combine losses
        loss = pos_loss + neg_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLossWithPosWeights(nn.Module):
    """
    Focal Loss combined with positive class weights for multi-label classification.

    Combines the benefits of focal loss (focusing on hard examples) with
    positive class weights (addressing class imbalance).
    """

    def __init__(
        self, pos_weights: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"
    ):
        """
        Initialize Focal Loss with positive weights.

        Args:
            pos_weights: Positive class weights tensor of shape [num_classes]
            gamma: Focusing parameter (gamma >= 0)
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
        """
        super(FocalLossWithPosWeights, self).__init__()
        self.pos_weights = pos_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Logits tensor of shape [batch_size, num_classes]
            targets: Binary labels tensor of shape [batch_size, num_classes]

        Returns:
            Loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate BCE with pos_weights
        if self.pos_weights.device != inputs.device:
            self.pos_weights = self.pos_weights.to(inputs.device)

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weights, reduction="none"
        )

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CorrelationAwareFocalLoss(nn.Module):
    """
    Correlation-Aware Focal Loss for multi-label classification.

    Extends Focal Loss to consider label correlations in multi-label settings.
    When labels co-occur frequently, the loss adapts to account for these correlations,
    improving performance on correlated pathology types.

    Key features:
    - Focal loss mechanism for handling class imbalance
    - Label correlation matrix learned from training data
    - Adaptive weighting based on label co-occurrence
    - Positive class weights for rare classes
    """

    def __init__(
        self,
        pos_weights: torch.Tensor,
        gamma: float = 2.0,
        correlation_weight: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Initialize Correlation-Aware Focal Loss.

        Args:
            pos_weights: Positive class weights tensor of shape [num_classes]
            gamma: Focusing parameter (gamma >= 0). Higher gamma focuses more on hard examples.
            correlation_weight: Weight for correlation term (0.0 to 1.0)
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
        """
        super(CorrelationAwareFocalLoss, self).__init__()
        self.pos_weights = pos_weights
        self.gamma = gamma
        self.correlation_weight = correlation_weight
        self.reduction = reduction

        # Correlation matrix will be learned/updated during training
        self.register_buffer("label_correlation", None)

    def update_correlation_matrix(self, targets: torch.Tensor, momentum: float = 0.1):
        """
        Update label correlation matrix from batch targets.

        Args:
            targets: Binary labels tensor [batch_size, num_classes]
            momentum: Momentum for exponential moving average
        """
        # Compute correlation: how often labels co-occur
        batch_size = targets.size(0)
        num_classes = targets.size(1)

        # Normalize targets to [0, 1]
        targets_norm = targets.float()

        # Compute correlation: E[target_i * target_j] for all pairs
        correlation = (
            torch.matmul(targets_norm.t(), targets_norm) / batch_size
        )  # [num_classes, num_classes]

        # Update with momentum
        if self.label_correlation is None:
            self.label_correlation = correlation
        else:
            self.label_correlation = (
                momentum * correlation + (1 - momentum) * self.label_correlation
            )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Logits tensor of shape [batch_size, num_classes]
            targets: Binary labels tensor of shape [batch_size, num_classes]

        Returns:
            Loss value
        """
        # Update correlation matrix
        self.update_correlation_matrix(targets)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Base BCE loss with positive weights
        if self.pos_weights.device != inputs.device:
            self.pos_weights = self.pos_weights.to(inputs.device)

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weights, reduction="none"
        )

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Correlation-aware adjustment
        if self.correlation_weight > 0.0 and self.label_correlation is not None:
            # For each sample, compute correlation penalty
            # If labels are correlated and model predicts one but not the other, add penalty
            batch_size = targets.size(0)
            num_classes = targets.size(1)

            # Get predicted labels (threshold at 0.5)
            pred_labels = (probs >= 0.5).float()

            # Compute correlation penalty for each label pair
            correlation_penalty = torch.zeros_like(focal_loss)

            for i in range(num_classes):
                for j in range(num_classes):
                    if (
                        i != j and self.label_correlation[i, j] > 0.3
                    ):  # Only for correlated pairs
                        # If true labels are correlated but predictions disagree
                        true_correlated = (targets[:, i] * targets[:, j]).float()
                        pred_disagree = torch.abs(pred_labels[:, i] - pred_labels[:, j])

                        # Penalty when true labels are correlated but predictions disagree
                        penalty = (
                            true_correlated
                            * pred_disagree
                            * self.label_correlation[i, j]
                        )
                        correlation_penalty[:, i] += penalty * self.correlation_weight

            # Add correlation penalty to loss
            focal_loss = focal_loss + correlation_penalty

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for multi-class classification with class imbalance.

    Computes CrossEntropyLoss with class weights to handle imbalanced datasets.
    Typically uses inverse frequency weights to give more importance to rare classes.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Weighted Cross-Entropy Loss.

        Args:
            class_weights: Class weights tensor of shape [num_classes]
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor (0.0 to 1.0)
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Logits tensor of shape [batch_size, num_classes]
            targets: Class indices tensor of shape [batch_size] (long tensor, values 0 to num_classes-1)

        Returns:
            Loss value
        """
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)

        # Use PyTorch's CrossEntropyLoss with weights
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class FocalLossMultiClass(nn.Module):
    """
    Focal Loss for multi-class classification.

    Adapts Focal Loss to multi-class setting using softmax instead of sigmoid.
    Focuses learning on hard examples, especially useful for imbalanced datasets.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Focal Loss for multi-class.

        Args:
            alpha: Weighting factor for each class (list of length num_classes).
                   If None, uses uniform weighting.
            gamma: Focusing parameter (gamma >= 0). Higher gamma focuses more on hard examples.
            reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor (0.0 to 1.0)
        """
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Logits tensor of shape [batch_size, num_classes]
            targets: Class indices tensor of shape [batch_size] (long tensor, values 0 to num_classes-1)

        Returns:
            Loss value
        """
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            num_classes = inputs.size(1)
            # Convert targets to one-hot
            targets_one_hot = F.one_hot(targets, num_classes).float()
            # Apply label smoothing
            targets_one_hot = (
                1 - self.label_smoothing
            ) * targets_one_hot + self.label_smoothing / num_classes
        else:
            # Convert targets to one-hot
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes).float()

        # Get probability of true class
        p_t = (probs * targets_one_hot).sum(dim=1)  # [batch_size]

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate cross-entropy component
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # [batch_size]

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Get alpha for each sample based on true class
            alpha_t = self.alpha[targets]  # [batch_size]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss_function(
    loss_type: str = "bce_with_logits",
    pos_weights: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: Optional[List[float]] = None,
    focal_gamma: float = 2.0,
    asym_gamma_neg: float = 4.0,
    asym_gamma_pos: float = 1.0,
    correlation_weight: float = 0.5,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    task_type: str = "multi_label",  # 'multi_label' or 'multi_class'
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: Type of loss
            Multi-label: 'bce_with_logits', 'focal', 'focal_with_weights', 'asymmetric', 'correlation_aware_focal'
            Multi-class: 'cross_entropy', 'weighted_ce', 'focal_multiclass'
        pos_weights: Positive class weights (for multi-label losses)
        class_weights: Class weights (for multi-class losses)
        focal_alpha: Alpha weights for focal loss
        focal_gamma: Gamma parameter for focal loss
        asym_gamma_neg: Gamma_neg for asymmetric loss
        asym_gamma_pos: Gamma_pos for asymmetric loss
        correlation_weight: Weight for correlation term in correlation_aware_focal
        label_smoothing: Label smoothing factor (for multi-class losses)
        reduction: Reduction method ('mean', 'sum', 'none')
        task_type: Task type ('multi_label' or 'multi_class')

    Returns:
        Loss function module
    """
    if task_type == "multi_class":
        # Multi-class classification losses
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(
                reduction=reduction, label_smoothing=label_smoothing
            )

        elif loss_type == "weighted_ce":
            if class_weights is None:
                raise ValueError("class_weights required for weighted_ce")
            return WeightedCrossEntropyLoss(
                class_weights=class_weights,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )

        elif loss_type == "focal_multiclass":
            return FocalLossMultiClass(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )

        else:
            raise ValueError(
                f"Unknown multi-class loss type: {loss_type}. "
                "Choose from: 'cross_entropy', 'weighted_ce', 'focal_multiclass'"
            )

    else:
        # Multi-label classification losses
        if loss_type == "bce_with_logits":
            if pos_weights is not None:
                return nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction=reduction)
            else:
                return nn.BCEWithLogitsLoss(reduction=reduction)

        elif loss_type == "focal":
            return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)

        elif loss_type == "focal_with_weights":
            if pos_weights is None:
                raise ValueError("pos_weights required for focal_with_weights")
            return FocalLossWithPosWeights(
                pos_weights=pos_weights, gamma=focal_gamma, reduction=reduction
            )

        elif loss_type == "asymmetric":
            return AsymmetricLoss(
                gamma_neg=asym_gamma_neg, gamma_pos=asym_gamma_pos, reduction=reduction
            )

        elif loss_type == "correlation_aware_focal":
            if pos_weights is None:
                raise ValueError("pos_weights required for correlation_aware_focal")
            return CorrelationAwareFocalLoss(
                pos_weights=pos_weights,
                gamma=focal_gamma,
                correlation_weight=correlation_weight,
                reduction=reduction,
            )

        else:
            raise ValueError(
                f"Unknown multi-label loss type: {loss_type}. "
                "Choose from: 'bce_with_logits', 'focal', 'focal_with_weights', 'asymmetric', 'correlation_aware_focal'"
            )
