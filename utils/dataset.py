"""
PyTorch Dataset classes for MRI Phenotyping tasks.

This module provides Dataset classes for:
1. Pathology Detection (Multi-label classification)
2. Pfirrman Grade Prediction (Multi-class classification)
"""

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pydicom
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MixupDataset(Dataset):
    """
    Dataset wrapper that applies Mixup augmentation.

    Mixup creates new samples by mixing two samples and their labels:
    x_mix = λ * x_i + (1 - λ) * x_j
    y_mix = λ * y_i + (1 - λ) * y_j

    Where λ ~ Beta(α, α) and α is a hyperparameter.
    """

    def __init__(self, dataset: Dataset, alpha: float = 0.4, mix_prob: float = 0.5):
        """
        Initialize MixupDataset.

        Args:
            dataset: Base dataset to apply Mixup to
            alpha: Mixup alpha parameter (default: 0.4 for medical images)
            mix_prob: Probability of applying Mixup to a sample (default: 0.5)
        """
        self.dataset = dataset
        self.alpha = alpha
        self.mix_prob = mix_prob

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get item with potential Mixup.

        Args:
            idx: Sample index

        Returns:
            Dictionary with mixed image and labels
        """
        # Get base sample
        sample = self.dataset[idx]

        # Decide whether to apply Mixup
        if np.random.random() < self.mix_prob and self.alpha > 0:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            # Get another random sample
            idx2 = np.random.randint(0, len(self.dataset))
            sample2 = self.dataset[idx2]

            # Mix images
            if "image" in sample:
                sample["image"] = lam * sample["image"] + (1 - lam) * sample2["image"]
            elif "sequences" in sample:
                # Mix each sequence
                mixed_sequences = {}
                for seq_name in sample["sequences"].keys():
                    if seq_name in sample2["sequences"]:
                        seq1 = sample["sequences"][seq_name]
                        seq2 = sample2["sequences"][seq_name]
                        if seq1 is not None and seq2 is not None:
                            mixed_sequences[seq_name] = lam * seq1 + (1 - lam) * seq2
                        elif seq1 is not None:
                            mixed_sequences[seq_name] = seq1
                        elif seq2 is not None:
                            mixed_sequences[seq_name] = seq2
                        else:
                            mixed_sequences[seq_name] = None
                    else:
                        mixed_sequences[seq_name] = sample["sequences"][seq_name]
                sample["sequences"] = mixed_sequences

            # Mix labels (for multi-label classification)
            if "labels" in sample:
                sample["labels"] = (
                    lam * sample["labels"] + (1 - lam) * sample2["labels"]
                )

        return sample


class PathologyDataset(Dataset):
    """
    Dataset class for multi-label pathology detection.

    Loads DICOM images and returns multi-label format:
    - Image: torch.Tensor of shape [C, H, W]
    - Labels: torch.Tensor of shape [4] with values [0, 1, 0, 1] etc.
      representing [disc_herniation, disc_bulging, spondylolisthesis, disc_narrowing]
    """

    # Label order (must match this order consistently)
    LABEL_NAMES = [
        "disc_herniation",
        "disc_bulging",
        "spondylolisthesis",
        "disc_narrowing",
    ]

    def __init__(
        self,
        manifest_path: Union[str, Path],
        project_root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        preferred_sequence: str = "SAG_T2",
        return_binary: bool = False,
    ):
        """
        Initialize PathologyDataset.

        Args:
            manifest_path: Path to pathology_training_manifest.csv
            project_root: Root directory of the project (for resolving relative paths)
            split: Dataset split to use ('train', 'val', 'test', or None for all)
            transform: Optional torchvision transforms to apply
            preferred_sequence: Preferred sequence type (default: 'SAG_T2')
            return_binary: If True, also return binary label for compatibility
        """
        self.manifest_path = Path(manifest_path)
        self.project_root = (
            Path(project_root)
            if project_root
            else self.manifest_path.parent.parent.parent
        )
        self.split = split
        self.transform = transform
        self.preferred_sequence = preferred_sequence
        self.return_binary = return_binary

        # Load manifest
        self.df = pd.read_csv(self.manifest_path)

        # Filter by split if specified
        if split is not None:
            self.df = self.df[self.df["dataset_split"] == split].reset_index(drop=True)

        print(f"Loaded {len(self.df)} samples from {self.manifest_path}")
        if split:
            print(f"  Split: {split}")

        # Parse pathology_details JSON
        self.df["pathology_details_parsed"] = self.df["pathology_details"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        # Parse dicom_file_paths JSON
        self.df["dicom_paths_parsed"] = self.df["dicom_file_paths"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

        # Parse sequence_types JSON
        self.df["sequence_types_parsed"] = self.df["sequence_types"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def _get_middle_slice(
        self,
        dicom_paths: List[str],
        sequence_types: List[str],
        preferred_sequence: str = "SAG_T2",
    ) -> Optional[str]:
        """
        Get middle slice from preferred sequence, or first available sequence.

        Args:
            dicom_paths: List of DICOM file paths
            sequence_types: List of sequence types (same order as dicom_paths)
            preferred_sequence: Preferred sequence type

        Returns:
            Path to middle slice DICOM file, or None if not found
        """
        if not dicom_paths or not sequence_types:
            return None

        # Map standardized sequence names to directory patterns
        sequence_patterns = {
            "SAG_T2": ["Sag T2", "SAG T2", "SagT2", "sag T2", "SAG_T2"],
            "SAG_T1": ["Sag T1", "SAG T1", "SagT1", "sag T1", "SAG_T1"],
            "AX_T2": ["Ax T2", "AX T2", "Axial T2", "AxT2", "AX_T2"],
            "SAG_STIR": ["Sag Stir", "SAG STIR", "Sag Stir", "sag stir", "SAG_STIR"],
            "COR_STIR": ["Cor Stir", "COR STIR", "cor stir", "COR_STIR"],
        }

        # Try preferred sequence first
        patterns = sequence_patterns.get(preferred_sequence, [preferred_sequence])
        for path, seq_type in zip(dicom_paths, sequence_types):
            path_str = str(path)
            seq_str = str(seq_type)

            # Check if sequence matches preferred
            for pattern in patterns:
                if pattern in path_str or pattern in seq_str:
                    # Get all matching paths for this sequence
                    matching_paths = [
                        p
                        for p, s in zip(dicom_paths, sequence_types)
                        if pattern in str(p) or pattern in str(s)
                    ]
                    if matching_paths:
                        # Return middle slice
                        return matching_paths[len(matching_paths) // 2]

        # If not found, return middle slice of any sequence
        if dicom_paths:
            return dicom_paths[len(dicom_paths) // 2]

        return None

    def _load_dicom_image(self, dicom_path: str) -> np.ndarray:
        """
        Load and normalize a DICOM image.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Normalized image array (0-1 range)
        """
        try:
            # Resolve path relative to project root
            if not Path(dicom_path).is_absolute():
                full_path = self.project_root / dicom_path
            else:
                full_path = Path(dicom_path)

            if not full_path.exists():
                raise FileNotFoundError(f"DICOM file not found: {full_path}")

            # Read DICOM
            ds = pydicom.dcmread(str(full_path))
            pixel_array = ds.pixel_array.astype(np.float32)

            # Normalize to 0-1 range
            if pixel_array.max() > pixel_array.min():
                pixel_array = (pixel_array - pixel_array.min()) / (
                    pixel_array.max() - pixel_array.min()
                )
            else:
                pixel_array = np.zeros_like(pixel_array)

            return pixel_array

        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            # Return black image as fallback
            return np.zeros((256, 256), dtype=np.float32)

    def _parse_pathology_labels(self, pathology_details: Dict) -> torch.Tensor:
        """
        Parse pathology_details dictionary to 4-element tensor.

        Args:
            pathology_details: Dictionary with pathology flags

        Returns:
            torch.Tensor of shape [4] with values [0, 1, 0, 1] etc.
        """
        labels = torch.zeros(4, dtype=torch.float32)

        for i, label_name in enumerate(self.LABEL_NAMES):
            labels[i] = float(pathology_details.get(label_name, 0))

        return labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - 'image': torch.Tensor [C, H, W] (normalized 0-1)
                - 'labels': torch.Tensor [4] (multi-label format)
                - 'pathology_binary': torch.Tensor [1] (optional, for compatibility)
                - 'patient_id': str
                - 'ivd_label': str
        """
        row = self.df.iloc[idx]

        # Get DICOM paths and sequences
        dicom_paths = row["dicom_paths_parsed"]
        sequence_types = row["sequence_types_parsed"]

        # Select middle slice from preferred sequence
        selected_path = self._get_middle_slice(
            dicom_paths, sequence_types, self.preferred_sequence
        )

        if selected_path is None:
            # Fallback: use first available path
            selected_path = dicom_paths[0] if dicom_paths else None

        # Load DICOM image
        image = self._load_dicom_image(selected_path)

        # Convert to PIL Image for transforms
        # Normalize to 0-255 range for PIL
        image_pil = Image.fromarray((image * 255).astype(np.uint8))

        # Apply transforms
        if self.transform:
            image_pil = self.transform(image_pil)

        # Convert to tensor if not already
        if not isinstance(image_pil, torch.Tensor):
            # Convert PIL to tensor
            image_tensor = transforms.ToTensor()(image_pil)
        else:
            image_tensor = image_pil

        # Ensure single channel (grayscale) -> [1, H, W]
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.shape[0] == 3:
            # If RGB, convert to grayscale (take mean)
            image_tensor = image_tensor.mean(dim=0, keepdim=True)

        # Parse pathology labels
        pathology_details = row["pathology_details_parsed"]
        labels = self._parse_pathology_labels(pathology_details)

        # Prepare return dictionary
        sample = {
            "image": image_tensor,
            "labels": labels,
            "patient_id": str(row["patient_id"]),
            "ivd_label": str(row["ivd_label"]),
        }

        # Add binary label for compatibility if requested
        if self.return_binary:
            sample["pathology_binary"] = torch.tensor(
                float(row["pathology_binary"]), dtype=torch.float32
            )

        return sample


def get_default_transforms(
    mode: str = "train",
    input_size: Tuple[int, int] = (224, 224),
    normalization: str = "imagenet",
    in_channels: int = 1,
    augmentation_strength: str = "medium",
) -> transforms.Compose:
    """
    Get default data augmentation transforms.

    Args:
        mode: 'train' for training transforms (with augmentation) or 'val'/'test' for validation
        input_size: Target image size (height, width)
        normalization: Normalization method ('imagenet', 'grayscale', or 'none')
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        augmentation_strength: 'light', 'medium', or 'strong' (default: 'medium')

    Returns:
        torchvision.transforms.Compose
    """
    transform_list = []

    # Resize
    transform_list.append(transforms.Resize(input_size))

    # Data augmentation for training
    if mode == "train":
        if augmentation_strength == "light":
            # Light augmentation (original)
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )
        elif augmentation_strength == "medium":
            # Medium augmentation (enhanced)
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                    ),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1
                    ),
                ]
            )
        elif augmentation_strength == "strong":
            # Strong augmentation (aggressive)
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),  # For medical images
                    transforms.RandomRotation(degrees=20),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10
                    ),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
                        p=0.3,
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown augmentation_strength: {augmentation_strength}")

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Normalization
    if normalization == "imagenet":
        if in_channels == 1:
            # For grayscale images, use ImageNet mean/std for single channel
            # (average of RGB channels: mean=0.449, std=0.226)
            transform_list.append(transforms.Normalize(mean=[0.449], std=[0.226]))
        else:
            # ImageNet normalization for RGB (3 channels)
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.406, 0.456], std=[0.229, 0.224, 0.225]
                )
            )
    elif normalization == "grayscale":
        # Standard grayscale normalization (mean=0.5, std=0.5)
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    elif normalization == "none":
        # No normalization (images already in 0-1 range)
        pass

    return transforms.Compose(transform_list)
