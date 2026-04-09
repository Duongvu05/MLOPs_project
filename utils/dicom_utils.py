"""
DICOM Metadata Extraction and Patient Index Utilities

This module provides utilities for:
1. Extracting DICOM metadata from all images
2. Creating unified patient index
3. Standardizing sequence naming
"""

import os
import pandas as pd
import pydicom
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from loguru import logger
import sys

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


# Sequence name standardization mapping
SEQUENCE_NAME_MAP = {
    # Axial T2
    "AX T2": "AX_T2",
    "Ax T2": "AX_T2",
    "Axial T2": "AX_T2",
    "AX T2'": "AX_T2",
    # Sagittal T1
    "Sag T1": "SAG_T1",
    "Sag t1": "SAG_T1",
    "SagT1": "SAG_T1",
    "SAg T1": "SAG_T1",
    "SAG T1": "SAG_T1",
    "sag T1": "SAG_T1",
    "sag t1": "SAG_T1",
    "SAGT1": "SAG_T1",
    # Sagittal T2
    "Sag T2": "SAG_T2",
    "SagT2": "SAG_T2",
    "SAg T2": "SAG_T2",
    "SAG T2": "SAG_T2",
    "sag T2": "SAG_T2",
    "SAG T2 STIR": "SAG_T2_STIR",
    # Sagittal STIR
    "Sag Stir": "SAG_STIR",
    "Sag Stir": "SAG_STIR",
    "Sag Sitr": "SAG_STIR",
    "Sag Stir": "SAG_STIR",
    "SAg Stir": "SAG_STIR",
    "SAG STIR": "SAG_STIR",
    "sag stir": "SAG_STIR",
    "Sag Sitr": "SAG_STIR",
    # Coronal STIR
    "Cor Stir": "COR_STIR",
    "Cor stir": "COR_STIR",
    "COR STIR": "COR_STIR",
    # Coronal T2
    "COR T2": "COR_T2",
    "Cor T2": "COR_T2",
}


def standardize_sequence_name(seq_name: str) -> str:
    """
    Standardize sequence name to canonical form.

    Args:
        seq_name: Original sequence name

    Returns:
        Standardized sequence name
    """
    seq_name = seq_name.strip()

    # Try direct mapping first
    if seq_name in SEQUENCE_NAME_MAP:
        return SEQUENCE_NAME_MAP[seq_name]

    # Try case-insensitive matching
    seq_lower = seq_name.lower()
    for key, value in SEQUENCE_NAME_MAP.items():
        if key.lower() == seq_lower:
            return value

    # If no match, return original with spaces replaced by underscores and uppercase
    return seq_name.replace(" ", "_").upper()


def extract_dicom_metadata(dicom_path: str) -> Dict:
    """
    Extract metadata from a single DICOM file.

    Args:
        dicom_path: Path to DICOM file

    Returns:
        Dictionary containing relevant DICOM metadata
    """
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

        metadata = {
            "file_path": str(dicom_path),
            "file_name": os.path.basename(dicom_path),
            "patient_id": getattr(ds, "PatientID", None),
            "patient_name": str(getattr(ds, "PatientName", "")),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "study_time": str(getattr(ds, "StudyTime", "")),
            "modality": getattr(ds, "Modality", None),
            "series_description": getattr(ds, "SeriesDescription", ""),
            "sequence_name": getattr(ds, "SeriesDescription", "").strip(),
            "slice_thickness": getattr(ds, "SliceThickness", None),
            "pixel_spacing": str(getattr(ds, "PixelSpacing", "")),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "bits_allocated": getattr(ds, "BitsAllocated", None),
            "bits_stored": getattr(ds, "BitsStored", None),
            "instance_number": getattr(ds, "InstanceNumber", None),
            "image_position_patient": str(getattr(ds, "ImagePositionPatient", "")),
            "image_orientation_patient": str(
                getattr(ds, "ImageOrientationPatient", "")
            ),
            "manufacturer": getattr(ds, "Manufacturer", ""),
            "manufacturer_model_name": getattr(ds, "ManufacturerModelName", ""),
            "magnetic_field_strength": getattr(ds, "MagneticFieldStrength", None),
            "scanning_sequence": str(getattr(ds, "ScanningSequence", "")),
            "sequence_variant": str(getattr(ds, "SequenceVariant", "")),
            "slice_location": getattr(ds, "SliceLocation", None),
        }

        return metadata
    except Exception as e:
        return {
            "file_path": str(dicom_path),
            "file_name": os.path.basename(dicom_path),
            "error": str(e),
        }


def extract_all_dicom_metadata(data_root: str, output_file: str) -> pd.DataFrame:
    """
    Extract metadata from all DICOM files in the dataset.

    Args:
        data_root: Root directory containing all datasets
        output_file: Path to save metadata CSV

    Returns:
        DataFrame containing all DICOM metadata
    """
    data_root = Path(data_root)
    all_metadata = []

    # Define dataset directories
    datasets = [
        ("T27.7.25", data_root / "cstl"),
        ("T8", data_root / "CSTL T8"),
        (
            "T9",
            data_root / "Du lieu MRI cot song thang 9" / "Du lieu MRI cot song thang 9",
        ),
    ]

    for dataset_name, dataset_path in datasets:
        if not dataset_path.exists():
            logger.warning(f"{dataset_path} does not exist, skipping...")
            continue

        logger.info(f"Processing {dataset_name} dataset...")
        dicom_files = list(dataset_path.rglob("*.dcm"))
        logger.info(f"Found {len(dicom_files)} DICOM files")

        for dicom_file in tqdm(dicom_files, desc=f"Extracting {dataset_name}"):
            metadata = extract_dicom_metadata(str(dicom_file))
            metadata["dataset"] = dataset_name

            # Extract patient directory name
            relative_path = dicom_file.relative_to(dataset_path)
            path_parts = relative_path.parts
            if len(path_parts) >= 1:
                metadata["patient_directory"] = path_parts[0]

            # Extract sequence directory name
            if len(path_parts) >= 2:
                original_seq_name = path_parts[1]
                metadata["sequence_directory"] = original_seq_name
                metadata["sequence_name_standardized"] = standardize_sequence_name(
                    original_seq_name
                )

            all_metadata.append(metadata)

    # Create DataFrame
    df = pd.DataFrame(all_metadata)

    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.success(f"Metadata saved to {output_file}")
    logger.info(f"Total records: {len(df)}")

    return df


def create_unified_patient_index(
    metadata_df: pd.DataFrame, annotations_dir: str, output_file: str
) -> pd.DataFrame:
    """
    Create unified patient index combining DICOM metadata and annotations.

    Args:
        metadata_df: DataFrame with DICOM metadata
        annotations_dir: Directory containing annotation files
        output_file: Path to save unified patient index

    Returns:
        DataFrame with unified patient index
    """
    annotations_dir = Path(annotations_dir)

    # Load annotation files
    annotation_files = {
        "T27.7.25": annotations_dir / "MRI CSTL Grade - 27.7.25.csv",
        "T8": annotations_dir / "radiological_gradings - T8.csv",
        "T9": annotations_dir / "radiological_gradings - T9.xls",
    }

    # Load and process annotations
    all_annotations = []

    for dataset_name, annot_file in annotation_files.items():
        if not annot_file.exists():
            logger.warning(f"{annot_file} does not exist, skipping...")
            continue

        logger.info(f"Loading annotations from {dataset_name}...")

        if annot_file.suffix == ".xls":
            df = pd.read_excel(annot_file)
            # Handle mixed Modic types in T9
            if "Modic" in df.columns:
                df["Modic_original"] = df["Modic"].copy()
                # Convert mixed types to numeric (handle "1&2")
                df["Modic"] = pd.to_numeric(df["Modic"], errors="coerce")
                df["Modic_mixed"] = (
                    df["Modic_original"].astype(str).str.contains("&", na=False)
                )
        else:
            df = pd.read_csv(annot_file)

        df["dataset"] = dataset_name
        all_annotations.append(df)

    if not all_annotations:
        logger.warning("No annotation files found!")
        return pd.DataFrame()

    annotations_df = pd.concat(all_annotations, ignore_index=True)

    # Create patient-level summary from metadata
    logger.info("Creating patient-level summary from DICOM metadata...")
    patient_metadata = (
        metadata_df.groupby(["dataset", "patient_directory"])
        .agg(
            {
                "file_path": "count",
                "sequence_name_standardized": lambda x: list(set(x)),
                "study_date": "first",
                "patient_id": "first",
                "manufacturer": "first",
                "magnetic_field_strength": "first",
            }
        )
        .reset_index()
    )

    patient_metadata.columns = [
        "dataset",
        "patient_directory",
        "num_dicom_files",
        "sequences_available",
        "study_date",
        "patient_id_dicom",
        "manufacturer",
        "magnetic_field_strength",
    ]

    # Create patient-level summary from annotations
    logger.info("Creating patient-level summary from annotations...")
    patient_annotations = (
        annotations_df.groupby(["dataset", "Patient ID"])
        .agg(
            {
                "IVD label": "count",
                "Pfirrman grade": ["mean", "min", "max"],
                "Modic": lambda x: (x != 0).sum() if hasattr(x, "__len__") else 0,
                "Disc herniation": "sum",
                "Disc bulging": "sum",
                "Spondylolisthesis": "sum",
                "Disc narrowing": "sum",
            }
        )
        .reset_index()
    )

    patient_annotations.columns = [
        "dataset",
        "patient_id_annotation",
        "num_ivd_records",
        "pfirrman_mean",
        "pfirrman_min",
        "pfirrman_max",
        "num_modic_changes",
        "num_disc_herniation",
        "num_disc_bulging",
        "num_spondylolisthesis",
        "num_disc_narrowing",
    ]

    # Try to match patients between DICOM and annotations
    # This is approximate since naming conventions may differ
    logger.info("Matching patients between DICOM and annotations...")

    # Create unified index
    unified_index = []

    # Process each dataset separately
    for dataset in ["T27.7.25", "T8", "T9"]:
        meta_subset = patient_metadata[patient_metadata["dataset"] == dataset].copy()
        annot_subset = patient_annotations[
            patient_annotations["dataset"] == dataset
        ].copy()

        for _, meta_row in meta_subset.iterrows():
            # Handle sequences_available properly, converting any non-string items to strings and filtering out NaN values
            sequences = meta_row["sequences_available"]
            if isinstance(sequences, list):
                # Convert all items to string and filter out NaN values
                sequences_str = [
                    str(seq)
                    for seq in sequences
                    if pd.notna(seq) and str(seq).lower() != "nan"
                ]
                sequences_available = (
                    ", ".join(sequences_str) if sequences_str else "No sequences"
                )
            else:
                sequences_available = (
                    str(sequences)
                    if pd.notna(sequences) and str(sequences).lower() != "nan"
                    else "No sequences"
                )

            patient_entry = {
                "dataset": dataset,
                "patient_directory": meta_row["patient_directory"],
                "patient_id_dicom": meta_row["patient_id_dicom"],
                "num_dicom_files": meta_row["num_dicom_files"],
                "sequences_available": sequences_available,
                "study_date": meta_row["study_date"],
                "manufacturer": meta_row["manufacturer"],
                "magnetic_field_strength": meta_row["magnetic_field_strength"],
            }

            # Try to find matching annotation
            # This is a simple matching - can be improved
            if len(annot_subset) > 0:
                # Use first annotation for now (better matching can be implemented)
                patient_entry["has_annotations"] = True
                # Add annotation summary if available
                if len(annot_subset) > 0:
                    annot_match = annot_subset.iloc[0]
                    patient_entry["patient_id_annotation"] = annot_match[
                        "patient_id_annotation"
                    ]
                    patient_entry["num_ivd_records"] = annot_match["num_ivd_records"]
                    patient_entry["pfirrman_mean"] = annot_match["pfirrman_mean"]
                    patient_entry["pfirrman_max"] = annot_match["pfirrman_max"]

            unified_index.append(patient_entry)

    unified_df = pd.DataFrame(unified_index)

    # Save unified index
    unified_df.to_csv(output_file, index=False)
    logger.success(f"Unified patient index saved to {output_file}")
    logger.info(f"Total patients: {len(unified_df)}")

    return unified_df


def standardize_sequences_in_annotations(annotations_dir: str, output_dir: str) -> None:
    """
    Standardize sequence naming and handle special cases (like mixed Modic types).

    Args:
        annotations_dir: Directory containing annotation files
        output_dir: Directory to save standardized annotations
    """
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = {
        "T27.7.25": annotations_dir / "MRI CSTL Grade - 27.7.25.csv",
        "T8": annotations_dir / "radiological_gradings - T8.csv",
        "T9": annotations_dir / "radiological_gradings - T9.xls",
    }

    for dataset_name, annot_file in annotation_files.items():
        if not annot_file.exists():
            logger.warning(f"{annot_file} does not exist, skipping...")
            continue

        logger.info(f"Standardizing {dataset_name} annotations...")

        # Load annotations
        if annot_file.suffix == ".xls":
            df = pd.read_excel(annot_file)
        else:
            df = pd.read_csv(annot_file)

        # Handle T9 special cases
        if dataset_name == "T9":
            # Handle mixed Modic types
            if "Modic" in df.columns:
                df["Modic_original"] = df["Modic"].copy()

                # Create separate columns for analysis
                df["Modic_numeric"] = pd.to_numeric(
                    df["Modic"], errors="coerce"
                ).fillna(0)
                df["Modic_has_mixed"] = (
                    df["Modic_original"].astype(str).str.contains("&", na=False)
                )

                # For binary classification, mark mixed as having Modic changes
                df["Modic_binary"] = (df["Modic_numeric"] != 0) | df["Modic_has_mixed"]

            # Handle disc bulging outlier (value 4)
            if "Disc bulging" in df.columns:
                # Replace 4 with 1 (treat as present)
                df.loc[df["Disc bulging"] == 4, "Disc bulging"] = 1
                df["Disc bulging_outlier_fixed"] = df["Disc bulging"] == 4

        # Add dataset column
        df["dataset"] = dataset_name

        # Save standardized version
        output_file = output_dir / f"{dataset_name}_standardized.csv"
        df.to_csv(output_file, index=False)
        logger.success(f"Saved to {output_file}")

        # Create summary
        logger.info(f"{dataset_name} Summary:")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Patients: {df['Patient ID'].nunique()}")
        if "Modic" in df.columns:
            if "Modic_has_mixed" in df.columns:
                logger.info(f"  Mixed Modic types: {df['Modic_has_mixed'].sum()}")
            logger.info(f"  Modic distribution: {df['Modic'].value_counts().to_dict()}")


def main():
    """Main function to run all three steps."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract DICOM metadata and create patient index"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data/mri_phen/raw/PhenikaaMed_Dataset",
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="../data/mri_phen/raw/PhenikaaMed_Dataset",
        help="Directory containing annotation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/mri_phen/processed",
        help="Output directory for processed files",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Step 1: Extracting DICOM metadata...")
    logger.info("=" * 80)
    metadata_file = output_dir / "dicom_metadata.csv"
    metadata_df = extract_all_dicom_metadata(args.data_root, str(metadata_file))

    logger.info("=" * 80)
    logger.info("Step 2: Creating unified patient index...")
    logger.info("=" * 80)
    patient_index_file = output_dir / "unified_patient_index.csv"
    patient_index_df = create_unified_patient_index(
        metadata_df, args.annotations_dir, str(patient_index_file)
    )

    logger.info("=" * 80)
    logger.info("Step 3: Standardizing sequences and handling special cases...")
    logger.info("=" * 80)
    standardized_dir = output_dir / "standardized_annotations"
    standardize_sequences_in_annotations(args.annotations_dir, str(standardized_dir))

    logger.success("=" * 80)
    logger.success("All steps completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
