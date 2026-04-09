"""
Data Preprocessing Utilities for MRI Phenotyping

This module provides utilities for:
1. Loading and merging unified annotations
2. Linking DICOM files to IVD-level annotations
3. Creating training manifests for Pfirrman grade and pathology detection tasks
4. Generating train/val/test splits with stratification
5. Validating data links and generating statistics
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm
from loguru import logger
import sys
from sklearn.model_selection import train_test_split

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def load_unified_annotations(annotations_dir: str) -> pd.DataFrame:
    """
    Load and merge all standardized annotation files.

    Args:
        annotations_dir: Directory containing standardized annotation files

    Returns:
        DataFrame with all annotations merged
    """
    annotations_dir = Path(annotations_dir)

    annotation_files = {
        "T27.7.25": annotations_dir / "T27.7.25_standardized.csv",
        "T8": annotations_dir / "T8_standardized.csv",
        "T9": annotations_dir / "T9_standardized.csv",
    }

    all_annotations = []

    for dataset_name, annot_file in annotation_files.items():
        if not annot_file.exists():
            logger.warning(f"{annot_file} does not exist, skipping...")
            continue

        logger.info(f"Loading {dataset_name} annotations...")
        df = pd.read_csv(annot_file)

        # Ensure dataset column is set
        if "dataset" not in df.columns:
            df["dataset"] = dataset_name

        all_annotations.append(df)

    if not all_annotations:
        raise ValueError("No annotation files found!")

    # Merge all annotations
    unified_annotations = pd.concat(all_annotations, ignore_index=True)

    # Convert Patient ID to numeric for matching
    unified_annotations["Patient ID"] = pd.to_numeric(
        unified_annotations["Patient ID"], errors="coerce"
    )

    logger.success(f"Loaded {len(unified_annotations)} total annotation records")
    logger.info(f"  Unique patients: {unified_annotations['Patient ID'].nunique()}")
    logger.info(
        f"  Unique IVD labels: {sorted(unified_annotations['IVD label'].unique())}"
    )

    return unified_annotations


def merge_dicom_with_annotations(
    dicom_metadata_path: str, annotations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Link DICOM files to IVD-level annotations by Patient ID.

    Args:
        dicom_metadata_path: Path to dicom_metadata.csv
        annotations_df: DataFrame with unified annotations

    Returns:
        DataFrame with DICOM files linked to each IVD level
    """
    logger.info("Loading DICOM metadata...")
    dicom_metadata = pd.read_csv(dicom_metadata_path)

    # Convert Patient ID to numeric for matching
    dicom_metadata["patient_id"] = pd.to_numeric(
        dicom_metadata["patient_id"], errors="coerce"
    )

    logger.info(f"Loaded {len(dicom_metadata)} DICOM file records")
    logger.info(f"  Unique patients: {dicom_metadata['patient_id'].nunique()}")

    # Group DICOM files by patient
    logger.info("Grouping DICOM files by patient...")
    patient_dicom = (
        dicom_metadata.groupby("patient_id")
        .agg(
            {
                "file_path": list,
                "sequence_name_standardized": list,
                "dataset": "first",
                "study_date": "first",
            }
        )
        .reset_index()
    )

    patient_dicom.columns = [
        "patient_id",
        "dicom_file_paths",
        "sequence_types",
        "dataset",
        "study_date",
    ]

    # Merge annotations with DICOM files
    logger.info("Merging annotations with DICOM files...")
    merged_df = annotations_df.merge(
        patient_dicom, left_on="Patient ID", right_on="patient_id", how="left"
    )

    # Check for unmatched annotations
    unmatched = merged_df[merged_df["patient_id"].isna()]
    if len(unmatched) > 0:
        logger.warning(
            f"{len(unmatched)} annotation records could not be matched to DICOM files"
        )
        logger.warning(f"  Unique patients: {unmatched['Patient ID'].nunique()}")

    # Filter to only matched records
    matched_df = merged_df[merged_df["patient_id"].notna()].copy()

    logger.success(f"Successfully matched {len(matched_df)} records")
    logger.info(f"  Unique patients: {matched_df['Patient ID'].nunique()}")

    return matched_df


def create_pathology_binary_label(row: pd.Series) -> int:
    """
    Create binary pathology label (1 if any pathology present, 0 otherwise).

    Pathology is considered present if any of:
    - Disc herniation == 1
    - Disc bulging == 1
    - Spondylolisthesis == 1
    - Disc narrowing == 1

    Args:
        row: DataFrame row with pathology columns

    Returns:
        Binary label (0 or 1)
    """
    pathology_flags = [
        row.get("Disc herniation", 0),
        row.get("Disc bulging", 0),
        row.get("Spondylolisthesis", 0),
        row.get("Disc narrowing", 0),
    ]

    # Convert to numeric and handle any non-numeric values
    pathology_values = []
    for val in pathology_flags:
        try:
            num_val = float(val) if pd.notna(val) else 0
            pathology_values.append(1 if num_val >= 1 else 0)
        except (ValueError, TypeError):
            pathology_values.append(0)

    # Binary: 1 if any pathology present
    return 1 if any(pathology_values) else 0


def create_sequence_groups(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group DICOM files by patient and sequence type for multi-sequence models.
    Also create pathology binary label and pathology details.

    Args:
        merged_df: DataFrame with merged annotations and DICOM files

    Returns:
        DataFrame with one row per IVD level, including grouped DICOM files
    """
    logger.info("Creating sequence groups and pathology labels...")

    # Filter to lumbar spine only (IVD labels 1-5)
    original_count = len(merged_df)
    merged_df = merged_df[merged_df["IVD label"].isin([1, 2, 3, 4, 5])].copy()
    filtered_count = len(merged_df)
    logger.info(
        f"Filtered to lumbar spine (IVD labels 1-5): {original_count} → {filtered_count} records"
    )
    logger.info(
        f"  Removed {original_count - filtered_count} records with IVD labels outside 1-5 range"
    )

    result_rows = []

    for idx, row in tqdm(
        merged_df.iterrows(), total=len(merged_df), desc="Processing IVD levels"
    ):
        patient_id = row["Patient ID"]
        ivd_label = row["IVD label"]

        # Get DICOM files for this patient
        dicom_files = row.get("dicom_file_paths", [])
        sequence_types = row.get("sequence_types", [])

        # Handle case where dicom_files might be a string (JSON) or list
        if isinstance(dicom_files, str):
            try:
                dicom_files = json.loads(dicom_files)
            except:
                dicom_files = []
        elif not isinstance(dicom_files, list):
            # Handle NaN, None, or other types
            try:
                if pd.isna(dicom_files):
                    dicom_files = []
                else:
                    dicom_files = []
            except (TypeError, ValueError):
                dicom_files = []

        if isinstance(sequence_types, str):
            try:
                sequence_types = json.loads(sequence_types)
            except:
                sequence_types = []
        elif not isinstance(sequence_types, list):
            # Handle NaN, None, or other types
            try:
                if pd.isna(sequence_types):
                    sequence_types = []
                else:
                    sequence_types = []
            except (TypeError, ValueError):
                sequence_types = []

        # Ensure lists (final check)
        if not isinstance(dicom_files, list):
            dicom_files = []
        if not isinstance(sequence_types, list):
            sequence_types = []

        # Create pathology binary label
        pathology_binary = create_pathology_binary_label(row)

        # Create pathology details dictionary
        pathology_details = {
            "disc_herniation": int(row.get("Disc herniation", 0) or 0),
            "disc_bulging": int(row.get("Disc bulging", 0) or 0),
            "spondylolisthesis": int(row.get("Spondylolisthesis", 0) or 0),
            "disc_narrowing": int(row.get("Disc narrowing", 0) or 0),
        }

        # Get Pfirrman grade
        pfirrman_grade = row.get("Pfirrman grade", None)
        if pd.notna(pfirrman_grade):
            pfirrman_grade = int(float(pfirrman_grade))
        else:
            pfirrman_grade = None

        result_row = {
            "patient_id": int(float(patient_id)) if pd.notna(patient_id) else None,
            "ivd_label": int(float(ivd_label)) if pd.notna(ivd_label) else None,
            "dicom_file_paths": json.dumps(dicom_files)
            if dicom_files
            else json.dumps([]),
            "sequence_types": json.dumps(sequence_types)
            if sequence_types
            else json.dumps([]),
            "num_dicom_files": len(dicom_files),
            "num_sequences": len(set(sequence_types)),
            "pfirrman_grade": pfirrman_grade,
            "pathology_binary": pathology_binary,
            "pathology_details": json.dumps(pathology_details),
            "dataset": row.get("dataset", ""),
            "study_date": row.get("study_date", ""),
            # Include individual pathology columns for reference
            "disc_herniation": pathology_details["disc_herniation"],
            "disc_bulging": pathology_details["disc_bulging"],
            "spondylolisthesis": pathology_details["spondylolisthesis"],
            "disc_narrowing": pathology_details["disc_narrowing"],
            "modic": row.get("Modic", 0),
            "up_endplate": row.get("UP endplate", 0),
            "low_endplate": row.get("LOW endplate", 0),
        }

        result_rows.append(result_row)

    result_df = pd.DataFrame(result_rows)

    logger.success(f"Created {len(result_df)} IVD-level records")
    logger.info(
        f"  Records with DICOM files: {(result_df['num_dicom_files'] > 0).sum()}"
    )
    logger.info(
        f"  Records without DICOM files: {(result_df['num_dicom_files'] == 0).sum()}"
    )

    return result_df


def validate_data_links(merged_df: pd.DataFrame) -> Dict:
    """
    Validate data links between annotations and DICOM files.

    Args:
        merged_df: DataFrame with merged annotations and DICOM files

    Returns:
        Dictionary with validation statistics
    """
    validation_stats = {
        "total_ivd_records": len(merged_df),
        "records_with_dicom": (merged_df["num_dicom_files"] > 0).sum(),
        "records_without_dicom": (merged_df["num_dicom_files"] == 0).sum(),
        "unique_patients": merged_df["patient_id"].nunique(),
        "patients_with_dicom": merged_df[merged_df["num_dicom_files"] > 0][
            "patient_id"
        ].nunique(),
        "patients_without_dicom": merged_df[merged_df["num_dicom_files"] == 0][
            "patient_id"
        ].nunique(),
        "pfirrman_grade_distribution": merged_df["pfirrman_grade"]
        .value_counts()
        .to_dict(),
        "pathology_binary_distribution": merged_df["pathology_binary"]
        .value_counts()
        .to_dict(),
        "sequence_coverage": {},
    }

    # Check sequence coverage
    all_sequences = set()
    for seq_list_str in merged_df["sequence_types"]:
        if pd.notna(seq_list_str):
            try:
                seq_list = json.loads(seq_list_str)
                all_sequences.update(seq_list)
            except:
                pass

    for seq in all_sequences:
        count = 0
        for seq_list_str in merged_df["sequence_types"]:
            if pd.notna(seq_list_str):
                try:
                    seq_list = json.loads(seq_list_str)
                    if seq in seq_list:
                        count += 1
                except:
                    pass
        validation_stats["sequence_coverage"][seq] = {
            "patients_with_sequence": count,
            "percentage": (count / len(merged_df) * 100) if len(merged_df) > 0 else 0,
        }

    return validation_stats


def create_train_val_test_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_by: List[str] = ["pfirrman_grade", "dataset"],
) -> pd.DataFrame:
    """
    Create stratified train/val/test splits at patient level.

    Args:
        df: DataFrame with IVD-level records
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        stratify_by: Columns to stratify by (default: pfirrman_grade and dataset)

    Returns:
        DataFrame with 'dataset_split' column added
    """
    logger.info("Creating train/val/test splits...")

    # Get unique patients with their characteristics
    patient_features = (
        df.groupby("patient_id")
        .agg(
            {
                "pfirrman_grade": lambda x: (
                    int(x.mode()[0]) if len(x.mode()) > 0 else None
                ),
                "pathology_binary": lambda x: (
                    int(x.mode()[0]) if len(x.mode()) > 0 else 0
                ),
                "dataset": "first",
            }
        )
        .reset_index()
    )

    patient_features.columns = [
        "patient_id",
        "patient_pfirrman",
        "patient_pathology",
        "patient_dataset",
    ]

    # Create stratification column
    patient_features["stratify_col"] = (
        patient_features["patient_pfirrman"].astype(str)
        + "_"
        + patient_features["patient_dataset"].astype(str)
    )

    # First split: train+val vs test
    stratify_col = (
        patient_features["stratify_col"] if "pfirrman_grade" in stratify_by else None
    )

    train_val_patients, test_patients = train_test_split(
        patient_features["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Second split: train vs val (adjust val_size for remaining proportion)
    train_val_features = patient_features[
        patient_features["patient_id"].isin(train_val_patients)
    ]
    remaining_size = len(train_val_features)
    val_size_adjusted = val_size / (1 - test_size)

    train_patients, val_patients = train_test_split(
        train_val_features["patient_id"],
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_features["stratify_col"]
        if stratify_col is not None
        else None,
    )

    # Assign splits to patients
    patient_splits = {}
    for pid in train_patients:
        patient_splits[pid] = "train"
    for pid in val_patients:
        patient_splits[pid] = "val"
    for pid in test_patients:
        patient_splits[pid] = "test"

    # Assign splits to IVD-level records
    df = df.copy()
    df["dataset_split"] = df["patient_id"].map(patient_splits)

    # Report split statistics
    logger.info("Split statistics:")
    for split in ["train", "val", "test"]:
        split_df = df[df["dataset_split"] == split]
        logger.info(f"  {split}:")
        logger.info(f"    Patients: {split_df['patient_id'].nunique()}")
        logger.info(f"    IVD records: {len(split_df)}")
        if "pfirrman_grade" in split_df.columns:
            logger.info(
                f"    Pfirrman distribution: {split_df['pfirrman_grade'].value_counts().sort_index().to_dict()}"
            )
        if "pathology_binary" in split_df.columns:
            logger.info(
                f"    Pathology positive: {split_df['pathology_binary'].sum()} ({split_df['pathology_binary'].mean() * 100:.1f}%)"
            )

    return df


def generate_task_statistics(df: pd.DataFrame, output_path: str):
    """
    Generate statistics for both tasks and save to CSV.

    Args:
        df: DataFrame with IVD-level records and splits
        output_path: Path to save statistics CSV
    """
    logger.info("Generating task statistics...")

    stats_rows = []

    # Overall statistics
    stats_rows.append(
        {"metric": "total_records", "value": len(df), "split": "all", "task": "both"}
    )
    stats_rows.append(
        {
            "metric": "total_patients",
            "value": df["patient_id"].nunique(),
            "split": "all",
            "task": "both",
        }
    )

    # Task 1: Pfirrman Grade Statistics
    for split in ["all", "train", "val", "test"]:
        if split == "all":
            split_df = df
        else:
            split_df = df[df["dataset_split"] == split]

        if len(split_df) == 0:
            continue

        # Pfirrman grade distribution
        pfirrman_dist = split_df["pfirrman_grade"].value_counts().sort_index()
        for grade, count in pfirrman_dist.items():
            stats_rows.append(
                {
                    "metric": f"pfirrman_grade_{int(grade)}",
                    "value": int(count),
                    "split": split,
                    "task": "pfirrman",
                }
            )

        stats_rows.append(
            {
                "metric": "pfirrman_mean",
                "value": float(split_df["pfirrman_grade"].mean()),
                "split": split,
                "task": "pfirrman",
            }
        )

        # Task 2: Pathology Statistics
        pathology_pos = split_df["pathology_binary"].sum()
        pathology_neg = len(split_df) - pathology_pos

        stats_rows.append(
            {
                "metric": "pathology_positive",
                "value": int(pathology_pos),
                "split": split,
                "task": "pathology",
            }
        )
        stats_rows.append(
            {
                "metric": "pathology_negative",
                "value": int(pathology_neg),
                "split": split,
                "task": "pathology",
            }
        )
        stats_rows.append(
            {
                "metric": "pathology_prevalence",
                "value": float(split_df["pathology_binary"].mean()),
                "split": split,
                "task": "pathology",
            }
        )

    # Save statistics
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_path, index=False)
    logger.success(f"Statistics saved to {output_path}")

    return stats_df


def main():
    """Main function to run the preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess MRI phenotyping data")
    parser.add_argument(
        "--dicom_metadata",
        type=str,
        default="../data/mri_phen/processed/dicom_metadata.csv",
        help="Path to DICOM metadata CSV",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="../data/mri_phen/processed/standardized_annotations",
        help="Directory containing standardized annotation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/mri_phen/processed",
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="Test set proportion"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.15, help="Validation set proportion"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for splits"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MRI Phenotyping Data Preprocessing Pipeline")
    logger.info("=" * 80)

    # Step 1: Load unified annotations
    logger.info("=" * 80)
    logger.info("Step 1: Loading unified annotations")
    logger.info("=" * 80)
    annotations_df = load_unified_annotations(args.annotations_dir)

    # Step 2: Merge with DICOM metadata
    logger.info("=" * 80)
    logger.info("Step 2: Merging with DICOM metadata")
    logger.info("=" * 80)
    merged_df = merge_dicom_with_annotations(args.dicom_metadata, annotations_df)

    # Step 3: Create sequence groups and pathology labels
    logger.info("=" * 80)
    logger.info("Step 3: Creating sequence groups and pathology labels")
    logger.info("=" * 80)
    processed_df = create_sequence_groups(merged_df)

    # Step 4: Create train/val/test splits
    logger.info("=" * 80)
    logger.info("Step 4: Creating train/val/test splits")
    logger.info("=" * 80)
    processed_df = create_train_val_test_splits(
        processed_df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    # Step 5: Validate data links
    logger.info("=" * 80)
    logger.info("Step 5: Validating data links")
    logger.info("=" * 80)
    validation_stats = validate_data_links(processed_df)

    # Step 6: Generate task statistics
    logger.info("=" * 80)
    logger.info("Step 6: Generating task statistics")
    logger.info("=" * 80)
    stats_df = generate_task_statistics(
        processed_df, output_dir / "task_statistics.csv"
    )

    # Step 7: Save training manifests
    logger.info("=" * 80)
    logger.info("Step 7: Saving training manifests")
    logger.info("=" * 80)

    # Pfirrman manifest
    pfirrman_manifest = processed_df[
        [
            "patient_id",
            "ivd_label",
            "dicom_file_paths",
            "sequence_types",
            "pfirrman_grade",
            "dataset_split",
            "dataset",
        ]
    ].copy()
    pfirrman_manifest_path = output_dir / "pfirrman_training_manifest.csv"
    pfirrman_manifest.to_csv(pfirrman_manifest_path, index=False)
    logger.success(f"Pfirrman manifest saved to {pfirrman_manifest_path}")
    logger.info(f"  Records: {len(pfirrman_manifest)}")

    # Pathology manifest
    pathology_manifest = processed_df[
        [
            "patient_id",
            "ivd_label",
            "dicom_file_paths",
            "sequence_types",
            "pathology_binary",
            "pathology_details",
            "dataset_split",
            "dataset",
        ]
    ].copy()
    pathology_manifest_path = output_dir / "pathology_training_manifest.csv"
    pathology_manifest.to_csv(pathology_manifest_path, index=False)
    logger.success(f"Pathology manifest saved to {pathology_manifest_path}")
    logger.info(f"  Records: {len(pathology_manifest)}")

    logger.success("=" * 80)
    logger.success("Preprocessing completed successfully!")
    logger.success("=" * 80)
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("  - pfirrman_training_manifest.csv")
    logger.info("  - pathology_training_manifest.csv")
    logger.info("  - task_statistics.csv")


if __name__ == "__main__":
    main()
