"""
Multi-sequence dataset class for pathology classification.

This module provides a dataset class that loads multiple MRI sequences
per sample for multi-sequence fusion models.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from utils.dataset import PathologyDataset


class MultiSequencePathologyDataset(PathologyDataset):
    """
    Dataset class for multi-label pathology detection with multiple sequences.

    Extends PathologyDataset to load multiple MRI sequences per sample:
    - SAG_T2: Sagittal T2-weighted (primary)
    - SAG_T1: Sagittal T1-weighted
    - AX_T2: Axial T2-weighted
    - SAG_STIR: Sagittal STIR

    Returns dictionary with all available sequences and their availability flags.
    """

    # Standard sequence names
    SEQUENCE_NAMES = ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]

    def __init__(
        self,
        manifest_path: Union[str, Path],
        project_root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        sequences: List[str] = None,
        handle_missing: str = "zero_pad",
        return_binary: bool = False,
    ):
        """
        Initialize MultiSequencePathologyDataset.

        Args:
            manifest_path: Path to pathology_training_manifest.csv
            project_root: Root directory of the project
            split: Dataset split to use ('train', 'val', 'test', or None for all)
            transform: Optional torchvision transforms to apply
            sequences: List of sequence types to load (default: ['SAG_T2', 'SAG_T1', 'AX_T2', 'SAG_STIR'])
            handle_missing: How to handle missing sequences ('zero_pad', 'mask', or 'skip')
            return_binary: If True, also return binary label for compatibility
        """
        # Initialize parent class
        super().__init__(
            manifest_path=manifest_path,
            project_root=project_root,
            split=split,
            transform=transform,
            preferred_sequence="SAG_T2",  # Default, but we'll load multiple
            return_binary=return_binary,
        )

        self.sequences = sequences if sequences else self.SEQUENCE_NAMES
        self.handle_missing = handle_missing

        # Validate sequences
        for seq in self.sequences:
            if seq not in self.SEQUENCE_NAMES:
                raise ValueError(
                    f"Unknown sequence: {seq}. Must be one of {self.SEQUENCE_NAMES}"
                )

    def _get_sequence_slices(
        self, dicom_paths: List[str], sequence_types: List[str], sequence_name: str
    ) -> Optional[List[str]]:
        """
        Get all slices for a specific sequence type.

        Args:
            dicom_paths: List of DICOM file paths
            sequence_types: List of sequence types (same order as dicom_paths)
            sequence_name: Sequence type to extract (e.g., 'SAG_T2')

        Returns:
            List of DICOM paths for the specified sequence, or None if not found
        """
        if not dicom_paths or not sequence_types:
            return None

        # Map standardized sequence names to directory patterns
        sequence_patterns = {
            "SAG_T2": ["Sag T2", "SAG T2", "SagT2", "sag T2", "SAG_T2", "SAGT2"],
            "SAG_T1": ["Sag T1", "SAG T1", "SagT1", "sag T1", "SAG_T1", "SAGT1"],
            "AX_T2": ["Ax T2", "AX T2", "Axial T2", "AxT2", "AX_T2", "Ax T2'"],
            "SAG_STIR": [
                "Sag Stir",
                "SAG STIR",
                "Sag Stir",
                "sag stir",
                "SAG_STIR",
                "Sag Sitr",
            ],
            "COR_STIR": ["Cor Stir", "COR STIR", "cor stir", "COR_STIR"],
        }

        patterns = sequence_patterns.get(sequence_name, [sequence_name])
        matching_paths = []

        for path, seq_type in zip(dicom_paths, sequence_types):
            path_str = str(path)
            seq_str = str(seq_type)

            for pattern in patterns:
                if pattern in path_str or pattern in seq_str:
                    matching_paths.append(path)
                    break

        return matching_paths if matching_paths else None

    def _select_representative_slice(self, dicom_paths: List[str]) -> Optional[str]:
        """
        Select a representative slice from a sequence.

        Strategy: Use middle slice (most representative of the spine region).

        Args:
            dicom_paths: List of DICOM file paths for a sequence

        Returns:
            Path to selected slice, or None if empty
        """
        if not dicom_paths:
            return None

        # Sort paths to ensure consistent ordering
        sorted_paths = sorted(dicom_paths)

        # Return middle slice
        return sorted_paths[len(sorted_paths) // 2]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with multiple sequences.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - 'sequences': Dict[str, torch.Tensor] - One tensor per sequence [C, H, W]
                - 'sequence_available': Dict[str, bool] - Availability flags
                - 'labels': torch.Tensor [4] - Multi-label format
                - 'pathology_binary': torch.Tensor [1] - Optional binary label
                - 'patient_id': str
                - 'ivd_label': str - IVD label as string (1-5)
                - 'ivd_levels': torch.Tensor [scalar] - IVD level as integer tensor for embedding (1-5)
        """
        row = self.df.iloc[idx]

        # Get DICOM paths and sequences
        dicom_paths = row["dicom_paths_parsed"]
        sequence_types = row["sequence_types_parsed"]

        # Load all requested sequences
        sequences_dict = {}
        availability_dict = {}

        for seq_name in self.sequences:
            # Get slices for this sequence
            sequence_paths = self._get_sequence_slices(
                dicom_paths, sequence_types, seq_name
            )

            if sequence_paths:
                # Select representative slice
                selected_path = self._select_representative_slice(sequence_paths)

                if selected_path:
                    # Load DICOM image
                    image = self._load_dicom_image(selected_path)

                    # Convert to PIL Image for transforms
                    image_pil = Image.fromarray((image * 255).astype(np.uint8))

                    # Apply transforms
                    if self.transform:
                        image_pil = self.transform(image_pil)

                    # Convert to tensor if not already
                    if not isinstance(image_pil, torch.Tensor):
                        image_tensor = transforms.ToTensor()(image_pil)
                    else:
                        image_tensor = image_pil

                    # Ensure single channel (grayscale) -> [1, H, W]
                    if image_tensor.dim() == 2:
                        image_tensor = image_tensor.unsqueeze(0)
                    elif image_tensor.shape[0] == 3:
                        # If RGB, convert to grayscale (take mean)
                        image_tensor = image_tensor.mean(dim=0, keepdim=True)

                    sequences_dict[seq_name.lower()] = image_tensor
                    availability_dict[seq_name.lower()] = True
                else:
                    # Sequence found but no valid slice
                    sequences_dict[seq_name.lower()] = None
                    availability_dict[seq_name.lower()] = False
            else:
                # Sequence not found
                sequences_dict[seq_name.lower()] = None
                availability_dict[seq_name.lower()] = False

        # Handle missing sequences
        if self.handle_missing == "zero_pad":
            # Create zero tensor with same shape as first available sequence
            reference_shape = None
            for seq_tensor in sequences_dict.values():
                if seq_tensor is not None:
                    reference_shape = seq_tensor.shape
                    break

            if reference_shape is None:
                # No sequences available, create default shape
                reference_shape = (1, 224, 224)

            for seq_name in self.sequences:
                seq_key = seq_name.lower()
                if sequences_dict[seq_key] is None:
                    sequences_dict[seq_key] = torch.zeros(
                        reference_shape, dtype=torch.float32
                    )

        elif self.handle_missing == "mask":
            # Keep as None, model will handle masking
            pass

        # Parse pathology labels
        pathology_details = row["pathology_details_parsed"]
        labels = self._parse_pathology_labels(pathology_details)

        # Build return dictionary
        result = {
            "sequences": sequences_dict,
            "sequence_available": availability_dict,
            "labels": labels,
            "patient_id": str(row["patient_id"]),
            "ivd_label": str(row["ivd_label"]),
            "ivd_levels": torch.tensor(
                int(row["ivd_label"]), dtype=torch.long
            ),  # Add integer IVD level for embedding
        }

        if self.return_binary:
            result["pathology_binary"] = torch.tensor(
                [row["pathology_binary"]], dtype=torch.float32
            )

        return result

    def get_sequence_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about sequence availability in the dataset.

        Returns:
            Dictionary with availability counts per sequence
        """
        stats = {seq: {"available": 0, "missing": 0} for seq in self.sequences}

        for idx in range(len(self)):
            sample = self.__getitem__(idx)
            availability = sample["sequence_available"]

            for seq_name in self.sequences:
                seq_key = seq_name.lower()
                if availability.get(seq_key, False):
                    stats[seq_name]["available"] += 1
                else:
                    stats[seq_name]["missing"] += 1

        # Calculate percentages
        total = len(self)
        for seq_name in self.sequences:
            stats[seq_name]["available_pct"] = (
                (stats[seq_name]["available"] / total * 100) if total > 0 else 0
            )
            stats[seq_name]["missing_pct"] = (
                (stats[seq_name]["missing"] / total * 100) if total > 0 else 0
            )

        return stats
