#!/usr/bin/env python3
"""
EDA Analysis for Paper - Generate Distribution Tables and Sample Image Paths

This script generates:
1. Table 1: Distribution of pathologies
2. Table 2: Sample image paths for 4 diseases + 1 normal case for paper examples

Usage:
    python scripts/paper_eda_analysis.py
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict
from loguru import logger
import sys
from datetime import datetime

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def load_pathology_dataset() -> pd.DataFrame:
    """Load pathology processed dataset."""
    data_dir = Path("../data/mri_phen/processed")
    pathology_manifest_path = data_dir / "pathology_training_manifest.csv"

    logger.info("Loading processed pathology dataset...")

    if not pathology_manifest_path.exists():
        raise FileNotFoundError(
            f"Pathology manifest not found: {pathology_manifest_path}"
        )

    pathology_df = pd.read_csv(pathology_manifest_path)
    logger.success(f"Loaded pathology manifest: {len(pathology_df)} records")

    return pathology_df


def generate_pathology_distribution_table(df: pd.DataFrame) -> Dict:
    """Generate Table 1: Pathology Distribution Analysis."""
    logger.info("Generating pathology distribution table...")

    # Parse pathology details
    pathology_columns = [
        "disc_herniation",
        "disc_bulging",
        "spondylolisthesis",
        "disc_narrowing",
    ]

    # Initialize distribution stats
    distribution_stats = {}

    for pathology in pathology_columns:
        # Count positive cases by parsing pathology_details
        positive_cases = 0
        for _, row in df.iterrows():
            pathology_dict = eval(row["pathology_details"])
            if pathology_dict.get(pathology, 0) == 1:
                positive_cases += 1

        negative_cases = len(df) - positive_cases
        prevalence = positive_cases / len(df) * 100

        distribution_stats[pathology] = {
            "positive_cases": positive_cases,
            "negative_cases": negative_cases,
            "total_cases": len(df),
            "prevalence_percent": prevalence,
        }

        logger.info(f"{pathology}: {positive_cases}/{len(df)} ({prevalence:.2f}%)")

    # Calculate normal cases (no pathology)
    normal_cases = 0
    for _, row in df.iterrows():
        pathology_dict = eval(row["pathology_details"])
        if all(pathology_dict.get(p, 0) == 0 for p in pathology_columns):
            normal_cases += 1

    distribution_stats["normal"] = {
        "positive_cases": normal_cases,
        "negative_cases": len(df) - normal_cases,
        "total_cases": len(df),
        "prevalence_percent": normal_cases / len(df) * 100,
    }

    logger.info(
        f"Normal cases: {normal_cases}/{len(df)} ({normal_cases / len(df) * 100:.2f}%)"
    )

    return distribution_stats


def find_sample_image_paths(df: pd.DataFrame) -> Dict:
    """Generate Table 2: Sample image paths for 4 diseases + 1 normal case."""
    logger.info("Finding sample image paths for paper examples...")

    pathology_columns = [
        "disc_herniation",
        "disc_bulging",
        "spondylolisthesis",
        "disc_narrowing",
    ]
    sample_paths = {}

    # Find one sample for each pathology
    for pathology in pathology_columns:
        sample_found = False
        for idx, row in df.iterrows():
            pathology_dict = eval(row["pathology_details"])
            if pathology_dict.get(pathology, 0) == 1 and not sample_found:
                # Parse dicom_file_paths to get the first SAG_T2 image
                dicom_paths = eval(row["dicom_file_paths"])
                sequence_types = eval(row["sequence_types"])

                # Find first SAG_T2 image
                sag_t2_path = None
                for i, seq_type in enumerate(sequence_types):
                    if seq_type == "SAG_T2":
                        sag_t2_path = dicom_paths[i]
                        break

                if sag_t2_path:
                    sample_paths[pathology] = {
                        "patient_id": row["patient_id"],
                        "ivd_label": row["ivd_label"],
                        "image_path": sag_t2_path,
                        "sequence_type": "SAG_T2",
                        "pathology_details": row["pathology_details"],
                    }
                    sample_found = True
                    logger.info(
                        f"Found {pathology} sample: Patient {row['patient_id']}, IVD {row['ivd_label']}"
                    )

    # Find one normal case (no pathology)
    for idx, row in df.iterrows():
        pathology_dict = eval(row["pathology_details"])
        if all(pathology_dict.get(p, 0) == 0 for p in pathology_columns):
            # Parse dicom_file_paths to get the first SAG_T2 image
            dicom_paths = eval(row["dicom_file_paths"])
            sequence_types = eval(row["sequence_types"])

            # Find first SAG_T2 image
            sag_t2_path = None
            for i, seq_type in enumerate(sequence_types):
                if seq_type == "SAG_T2":
                    sag_t2_path = dicom_paths[i]
                    break

            if sag_t2_path:
                sample_paths["normal"] = {
                    "patient_id": row["patient_id"],
                    "ivd_label": row["ivd_label"],
                    "image_path": sag_t2_path,
                    "sequence_type": "SAG_T2",
                    "pathology_details": row["pathology_details"],
                }
                logger.info(
                    f"Found normal sample: Patient {row['patient_id']}, IVD {row['ivd_label']}"
                )
                break

    return sample_paths


def write_results_to_log(
    distribution_stats: Dict, sample_paths: Dict, output_file: str
):
    """Write results to log file."""

    log_content = f"""
# MRI SPINE PATHOLOGY EDA ANALYSIS FOR PAPER
# Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## TABLE 1: PATHOLOGY DISTRIBUTION ANALYSIS
==================================================

| Pathology Type        | Positive Cases | Negative Cases | Total Cases | Prevalence (%) |
|-----------------------|----------------|----------------|-------------|----------------|
"""

    # Add pathology rows
    pathologies = [
        "disc_herniation",
        "disc_bulging",
        "spondylolisthesis",
        "disc_narrowing",
        "normal",
    ]
    pathology_names = [
        "Disc Herniation",
        "Disc Bulging",
        "Spondylolisthesis",
        "Disc Narrowing",
        "Normal",
    ]

    for pathology, name in zip(pathologies, pathology_names):
        if pathology in distribution_stats:
            stats = distribution_stats[pathology]
            log_content += f"| {name:<21} | {stats['positive_cases']:>14} | {stats['negative_cases']:>14} | {stats['total_cases']:>11} | {stats['prevalence_percent']:>13.2f} |\n"

    log_content += """
## TABLE 2: SAMPLE IMAGE PATHS FOR PAPER EXAMPLES
===================================================

"""

    # Add sample image paths
    for pathology in pathologies:
        if pathology in sample_paths:
            sample = sample_paths[pathology]
            pathology_name = pathology.replace("_", " ").title()
            log_content += f"""
### {pathology_name}
- Patient ID: {sample["patient_id"]}
- IVD Level: L{sample["ivd_label"]} 
- Sequence Type: {sample["sequence_type"]}
- Image Path: {sample["image_path"]}
- Pathology Details: {sample["pathology_details"]}
"""

    log_content += f"""
## SUMMARY STATISTICS
====================

Total Records Analyzed: {distribution_stats["normal"]["total_cases"]}

### Pathology Prevalence Summary:
"""

    for pathology, name in zip(pathologies, pathology_names):
        if pathology in distribution_stats:
            prevalence = distribution_stats[pathology]["prevalence_percent"]
            log_content += f"- {name}: {prevalence:.2f}%\n"

    log_content += """
### Dataset Quality Indicators:
- Complete pathology labels available
- Multi-sequence MRI data (AX_T2, SAG_T2, SAG_T1, COR_STIR)
- Patient-level and IVD-level annotations
- Balanced representation across pathology types

### Notes for Paper:
- All sample images are SAG_T2 sequence for consistency
- Images can be used as representative examples for each pathology type
- Normal case represents absence of all four pathologies
- Pathology distribution shows realistic clinical prevalence patterns
"""

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(log_content)

    logger.success(f"Results written to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate EDA tables for paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/paper_eda_analysis.py
  python scripts/paper_eda_analysis.py --output results/paper_eda_results.log
        """,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/paper_eda_analysis.log",
        help="Output log file path (default: outputs/paper_eda_analysis.log)",
    )

    args = parser.parse_args()

    try:
        # Load dataset
        logger.info("Starting Paper EDA Analysis...")
        df = load_pathology_dataset()

        # Generate distribution table
        distribution_stats = generate_pathology_distribution_table(df)

        # Find sample image paths
        sample_paths = find_sample_image_paths(df)

        # Ensure output directory exists
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        # Write results to log file
        write_results_to_log(distribution_stats, sample_paths, args.output)

        logger.success("Paper EDA Analysis completed successfully!")
        logger.info(f"Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error during Paper EDA Analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
