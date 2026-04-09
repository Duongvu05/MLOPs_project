"""
Test-Time Augmentation (TTA) utilities for improving model predictions.

TTA applies multiple augmentations to test images and averages the predictions,
which can significantly improve model performance.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Dict, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader

from training.pathology_training_utils import (
    is_multi_sequence_model,
)


def get_tta_transforms(
    input_size: Tuple[int, int] = (224, 224),
    num_augmentations: int = 8,
    normalization: str = "imagenet",
    in_channels: int = 1,
) -> List[transforms.Compose]:
    """
    Generate list of augmentation transforms for TTA.

    Args:
        input_size: Target image size (height, width)
        num_augmentations: Number of augmentation strategies (default: 8)
        normalization: Normalization method ('imagenet', 'grayscale', or 'none')
        in_channels: Number of input channels

    Returns:
        List of transform compositions for TTA
    """
    transform_list = []

    # Base transforms (resize, to tensor, normalize)
    base_transforms = [
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ]

    if normalization == "imagenet":
        if in_channels == 1:
            base_transforms.append(transforms.Normalize(mean=[0.449], std=[0.226]))
        else:
            base_transforms.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
    elif normalization == "grayscale":
        base_transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    base_transform = transforms.Compose(base_transforms)

    # 1. Original (no augmentation)
    transform_list.append(base_transform)

    # 2. Horizontal flip
    transform_list.append(
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
            ]
            + (base_transforms[2:] if len(base_transforms) > 2 else [])
        )
    )

    # 3. Vertical flip
    transform_list.append(
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
            ]
            + (base_transforms[2:] if len(base_transforms) > 2 else [])
        )
    )

    # 4. Rotation +5 degrees
    transform_list.append(
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
            ]
            + (base_transforms[2:] if len(base_transforms) > 2 else [])
        )
    )

    # 5. Rotation -5 degrees
    transform_list.append(
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomRotation(degrees=-5),
                transforms.ToTensor(),
            ]
            + (base_transforms[2:] if len(base_transforms) > 2 else [])
        )
    )

    # 6. Slight affine (translation)
    transform_list.append(
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
            ]
            + (base_transforms[2:] if len(base_transforms) > 2 else [])
        )
    )

    # 7. Slight scale
    transform_list.append(
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
                transforms.ToTensor(),
            ]
            + (base_transforms[2:] if len(base_transforms) > 2 else [])
        )
    )

    # 8. Combined: horizontal flip + slight rotation
    transform_list.append(
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(degrees=3),
                transforms.ToTensor(),
            ]
            + (base_transforms[2:] if len(base_transforms) > 2 else [])
        )
    )

    # Return requested number of augmentations
    return transform_list[:num_augmentations]


def apply_tta_to_batch(
    model: nn.Module,
    batch: Dict,
    tta_transforms: List[transforms.Compose],
    device: torch.device,
    is_multi_seq: bool = True,
) -> torch.Tensor:
    """
    Apply TTA to a batch and return averaged predictions.

    BREAKING CHANGE: All models now use multi-sequence fusion. This function
    always expects multi-sequence input format.

    Args:
        model: Trained model (can be ensemble or single model)
        batch: Data batch from dataloader (must have 'sequences' key)
        tta_transforms: List of augmentation transforms
        device: PyTorch device
        is_multi_seq: Whether model is multi-sequence (should always be True now)

    Returns:
        Averaged probabilities [B, num_labels]
    """
    model.eval()
    all_probs = []

    # BREAKING CHANGE: All models now use multi-sequence
    if not is_multi_seq:
        raise ValueError(
            "All models now use multi-sequence fusion. is_multi_seq must be True."
        )

    with torch.no_grad():
        for transform in tta_transforms:
            # Apply augmentation to multi-sequence batch
            # For multi-sequence, apply to each sequence in batch
            augmented_sequences = {}
            batch_size = (
                batch["sequences"][list(batch["sequences"].keys())[0]].size(0)
                if batch["sequences"]
                else 1
            )

            for seq_name, seq_tensor in batch["sequences"].items():
                if seq_tensor is not None:
                    # seq_tensor is [B, C, H, W]
                    augmented_batch = []
                    for i in range(batch_size):
                        # Convert to PIL for augmentation

                        img_tensor = seq_tensor[i].cpu()  # [C, H, W]
                        # Denormalize if needed (assuming ImageNet normalization)
                        img_tensor = img_tensor * 0.226 + 0.449
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        img = transforms.ToPILImage()(img_tensor)

                        # Apply augmentation
                        augmented = transform(img)
                        augmented_batch.append(augmented)

                    augmented_sequences[seq_name] = torch.stack(augmented_batch).to(
                        device
                    )
                else:
                    augmented_sequences[seq_name] = None

            # Get prediction - use prepare_model_input for consistency
            from training.pathology_training_utils import prepare_model_input

            model_input, _ = prepare_model_input(
                {
                    "sequences": augmented_sequences,
                    "sequence_available": batch.get("sequence_available"),
                },
                device,
                is_multi_seq,
                model=model,
            )

            # BREAKING CHANGE: All models now use multi-sequence
            # Always use multi-sequence format
            sequences, sequence_available = model_input
            output = model(
                sequences, sequence_available, return_attention_weights=False
            )
            # Handle tuple return (logits, attention_weights) or just logits
            logits = output[0] if isinstance(output, tuple) else output

            probs = torch.sigmoid(logits)
            all_probs.append(probs)

    # Average probabilities across all augmentations
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)

    return avg_probs


def tta_predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tta_transforms: Optional[List[transforms.Compose]] = None,
    num_augmentations: int = 8,
    input_size: Tuple[int, int] = (224, 224),
    normalization: str = "imagenet",
    in_channels: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate TTA predictions on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: PyTorch device
        tta_transforms: Optional pre-computed TTA transforms
        num_augmentations: Number of augmentations if tta_transforms not provided
        input_size: Image size for TTA transforms
        normalization: Normalization method
        in_channels: Number of input channels

    Returns:
        Tuple of (true_labels, predicted_probabilities) as numpy arrays
    """
    if tta_transforms is None:
        tta_transforms = get_tta_transforms(
            input_size=input_size,
            num_augmentations=num_augmentations,
            normalization=normalization,
            in_channels=in_channels,
        )

    is_multi_seq = is_multi_sequence_model(model)

    all_true_labels = []
    all_pred_probs = []

    from tqdm import tqdm

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="TTA Predictions"):
            # Get true labels
            labels = batch["labels"].to(device)
            all_true_labels.append(labels.cpu().numpy())

            # Apply TTA
            probs = apply_tta_to_batch(
                model=model,
                batch=batch,
                tta_transforms=tta_transforms,
                device=device,
                is_multi_seq=is_multi_seq,
            )

            all_pred_probs.append(probs.cpu().numpy())

    all_true_labels = np.vstack(all_true_labels)
    all_pred_probs = np.vstack(all_pred_probs)

    return all_true_labels, all_pred_probs
