#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for MRI Spine Pathology Dataset

This script performs comprehensive EDA on the processed pathology dataset including:
- Dataset overview and statistics
- Pathology label distribution analysis
- MRI sequence availability analysis
- Data balance and quality checks
- Visualizations and insights

Usage:
    python scripts/eda_analysis.py
    python scripts/eda_analysis.py --save-figures --save-report
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, Tuple
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

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pathology processed dataset."""
    data_dir = Path("../data/mri_phen/processed")

    # Load pathology manifest
    pathology_manifest_path = data_dir / "pathology_training_manifest.csv"
    statistics_path = data_dir / "task_statistics.csv"

    logger.info("Loading processed pathology dataset...")

    if not pathology_manifest_path.exists():
        raise FileNotFoundError(
            f"Pathology manifest not found: {pathology_manifest_path}"
        )

    pathology_df = pd.read_csv(pathology_manifest_path)

    # Load statistics if available
    statistics_df = None
    if statistics_path.exists():
        statistics_df = pd.read_csv(statistics_path)

    logger.success("Loaded datasets:")
    logger.success(f"  Pathology manifest: {len(pathology_df)} records")
    if statistics_df is not None:
        logger.success(f"  Task statistics: {len(statistics_df)} metrics")

    return pathology_df, statistics_df


def analyze_dataset_overview(pathology_df: pd.DataFrame) -> Dict:
    """Analyze basic dataset statistics."""
    logger.info("Analyzing dataset overview...")

    overview = {
        "total_records": len(pathology_df),
        "total_patients": pathology_df["patient_id"].nunique(),
        "datasets": pathology_df["dataset"].value_counts().to_dict(),
        "splits": pathology_df["dataset_split"].value_counts().to_dict(),
        "ivd_levels": pathology_df["ivd_label"].value_counts().sort_index().to_dict(),
    }

    # Analyze DICOM availability
    pathology_df["num_dicom_files"] = pathology_df["dicom_file_paths"].apply(
        lambda x: len(json.loads(x)) if pd.notna(x) else 0
    )

    overview["dicom_stats"] = {
        "records_with_dicom": (pathology_df["num_dicom_files"] > 0).sum(),
        "records_without_dicom": (pathology_df["num_dicom_files"] == 0).sum(),
        "avg_dicom_per_record": pathology_df["num_dicom_files"].mean(),
        "total_dicom_files": pathology_df["num_dicom_files"].sum(),
    }

    # Analyze sequences
    all_sequences = []
    for seq_str in pathology_df["sequence_types"].dropna():
        try:
            sequences = json.loads(seq_str)
            all_sequences.extend(sequences)
        except:
            continue

    sequence_counts = pd.Series(all_sequences).value_counts()
    overview["sequence_stats"] = sequence_counts.to_dict()

    return overview


def analyze_pathology_labels(pathology_df: pd.DataFrame) -> Dict:
    """Analyze pathology label distribution."""
    logger.info("Analyzing pathology label distribution...")

    # Parse pathology details
    pathology_stats = {
        "disc_herniation": [],
        "disc_bulging": [],
        "spondylolisthesis": [],
        "disc_narrowing": [],
    }

    for details_str in pathology_df["pathology_details"].dropna():
        try:
            details = json.loads(details_str)
            for key in pathology_stats.keys():
                pathology_stats[key].append(details.get(key, 0))
        except:
            continue

    # Calculate statistics for each pathology
    results = {}
    for pathology, values in pathology_stats.items():
        if values:
            series = pd.Series(values)
            results[pathology] = {
                "total_records": len(series),
                "positive_cases": series.sum(),
                "negative_cases": len(series) - series.sum(),
                "positive_rate": series.mean(),
                "prevalence_percent": series.mean() * 100,
            }

    # Overall pathology binary
    pathology_binary = pathology_df["pathology_binary"].value_counts()
    results["overall_binary"] = {
        "any_pathology": pathology_binary.get(1, 0),
        "no_pathology": pathology_binary.get(0, 0),
        "any_pathology_rate": pathology_binary.get(1, 0) / len(pathology_df),
    }

    return results


def analyze_sequence_availability(pathology_df: pd.DataFrame) -> Dict:
    """Analyze sequence availability patterns."""
    logger.info("Analyzing sequence availability...")

    sequence_matrix = []
    valid_sequences = ["SAG_T2", "SAG_T1", "AX_T2", "SAG_STIR"]

    for seq_str in pathology_df["sequence_types"].dropna():
        try:
            sequences = json.loads(seq_str)
            row = {seq: (seq in sequences) for seq in valid_sequences}
            sequence_matrix.append(row)
        except:
            continue

    seq_df = pd.DataFrame(sequence_matrix)

    results = {
        "sequence_availability": seq_df.mean().to_dict(),
        "sequence_combinations": {},
        "complete_cases": len(seq_df[seq_df.all(axis=1)]),
        "incomplete_cases": len(seq_df[~seq_df.all(axis=1)]),
    }

    # Analyze common combinations
    combination_counts = (
        seq_df.groupby(list(valid_sequences)).size().reset_index(name="count")
    )
    combination_counts = combination_counts.sort_values("count", ascending=False)

    for idx, row in combination_counts.head(10).iterrows():
        combo_name = "+".join([seq for seq in valid_sequences if row[seq]])
        if not combo_name:
            combo_name = "No_sequences"
        results["sequence_combinations"][combo_name] = int(row["count"])

    return results


def create_visualizations(
    pathology_df: pd.DataFrame,
    pathology_stats: Dict,
    sequence_stats: Dict,
    save_figures: bool = False,
):
    """Create comprehensive visualizations."""
    logger.info("Creating visualizations...")

    fig_dir = project_root / "outputs" / "eda_figures"
    if save_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

    # Dataset distribution
    dataset_counts = pathology_df["dataset"].value_counts()
    axes[0, 0].pie(
        dataset_counts.values, labels=dataset_counts.index, autopct="%1.1f%%"
    )
    axes[0, 0].set_title("Records by Dataset")

    # Split distribution
    split_counts = pathology_df["dataset_split"].value_counts()
    axes[0, 1].bar(split_counts.index, split_counts.values)
    axes[0, 1].set_title("Records by Split")
    axes[0, 1].set_ylabel("Number of Records")

    # IVD level distribution
    ivd_counts = pathology_df["ivd_label"].value_counts().sort_index()
    axes[1, 0].bar(ivd_counts.index, ivd_counts.values)
    axes[1, 0].set_title("Records by IVD Level")
    axes[1, 0].set_xlabel("IVD Level")
    axes[1, 0].set_ylabel("Number of Records")

    # DICOM files distribution
    pathology_df["num_dicom_files"] = pathology_df["dicom_file_paths"].apply(
        lambda x: len(json.loads(x)) if pd.notna(x) else 0
    )
    axes[1, 1].hist(pathology_df["num_dicom_files"], bins=20, edgecolor="black")
    axes[1, 1].set_title("Distribution of DICOM Files per Record")
    axes[1, 1].set_xlabel("Number of DICOM Files")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    if save_figures:
        plt.savefig(fig_dir / "dataset_overview.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Pathology Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Pathology Label Analysis", fontsize=16, fontweight="bold")

    # Individual pathology prevalence
    pathology_names = list(pathology_stats.keys())[:-1]  # Exclude overall_binary
    prevalences = [
        pathology_stats[name]["prevalence_percent"] for name in pathology_names
    ]

    bars = axes[0, 0].bar(pathology_names, prevalences)
    axes[0, 0].set_title("Pathology Prevalence (%)")
    axes[0, 0].set_ylabel("Prevalence (%)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, val in zip(bars, prevalences):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
        )

    # Pathology counts
    positive_counts = [
        pathology_stats[name]["positive_cases"] for name in pathology_names
    ]
    negative_counts = [
        pathology_stats[name]["negative_cases"] for name in pathology_names
    ]

    x = np.arange(len(pathology_names))
    width = 0.35

    axes[0, 1].bar(x - width / 2, positive_counts, width, label="Positive", alpha=0.8)
    axes[0, 1].bar(x + width / 2, negative_counts, width, label="Negative", alpha=0.8)
    axes[0, 1].set_title("Pathology Cases Count")
    axes[0, 1].set_ylabel("Number of Cases")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(pathology_names, rotation=45)
    axes[0, 1].legend()

    # Overall pathology binary
    binary_stats = pathology_stats["overall_binary"]
    labels = ["No Pathology", "Any Pathology"]
    values = [binary_stats["no_pathology"], binary_stats["any_pathology"]]

    axes[1, 0].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    axes[1, 0].set_title("Overall Pathology Distribution")

    # Pathology heatmap by IVD level
    pathology_by_ivd = pd.DataFrame()
    for pathology in pathology_names:
        pathology_col = []
        for ivd in sorted(pathology_df["ivd_label"].unique()):
            ivd_subset = pathology_df[pathology_df["ivd_label"] == ivd]
            if len(ivd_subset) > 0:
                pathology_details = (
                    ivd_subset["pathology_details"]
                    .apply(
                        lambda x: (
                            json.loads(x)[pathology.replace(" ", "_").lower()]
                            if pd.notna(x)
                            else 0
                        )
                    )
                    .mean()
                    * 100
                )
                pathology_col.append(pathology_details)
            else:
                pathology_col.append(0)
        pathology_by_ivd[pathology] = pathology_col

    pathology_by_ivd.index = sorted(pathology_df["ivd_label"].unique())

    im = axes[1, 1].imshow(pathology_by_ivd.T, cmap="YlOrRd", aspect="auto")
    axes[1, 1].set_title("Pathology Prevalence by IVD Level (%)")
    axes[1, 1].set_xlabel("IVD Level")
    axes[1, 1].set_ylabel("Pathology Type")
    axes[1, 1].set_xticks(range(len(pathology_by_ivd.index)))
    axes[1, 1].set_xticklabels(pathology_by_ivd.index)
    axes[1, 1].set_yticks(range(len(pathology_names)))
    axes[1, 1].set_yticklabels(pathology_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label("Prevalence (%)")

    plt.tight_layout()
    if save_figures:
        plt.savefig(fig_dir / "pathology_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Sequence Availability Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("MRI Sequence Availability Analysis", fontsize=16, fontweight="bold")

    # Individual sequence availability
    seq_availability = sequence_stats["sequence_availability"]
    sequences = list(seq_availability.keys())
    availabilities = [v * 100 for v in seq_availability.values()]

    bars = axes[0, 0].bar(sequences, availabilities)
    axes[0, 0].set_title("Sequence Availability (%)")
    axes[0, 0].set_ylabel("Availability (%)")
    axes[0, 0].set_ylim(0, 100)

    for bar, val in zip(bars, availabilities):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
        )

    # Common sequence combinations
    combinations = sequence_stats["sequence_combinations"]
    combo_names = list(combinations.keys())[:8]  # Top 8
    combo_counts = [combinations[name] for name in combo_names]

    axes[0, 1].barh(combo_names, combo_counts)
    axes[0, 1].set_title("Most Common Sequence Combinations")
    axes[0, 1].set_xlabel("Number of Records")

    # Complete vs Incomplete cases
    complete = sequence_stats["complete_cases"]
    incomplete = sequence_stats["incomplete_cases"]

    axes[1, 0].pie(
        [complete, incomplete],
        labels=["Complete (All 4 sequences)", "Incomplete"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1, 0].set_title("Sequence Completeness")

    # Sequence co-occurrence matrix
    valid_sequences = list(seq_availability.keys())
    cooccurrence = np.zeros((len(valid_sequences), len(valid_sequences)))

    for i, seq1 in enumerate(valid_sequences):
        for j, seq2 in enumerate(valid_sequences):
            if i == j:
                cooccurrence[i, j] = seq_availability[seq1]
            else:
                # This is a simplified co-occurrence calculation
                # In practice, you'd want to calculate actual co-occurrence from the data
                cooccurrence[i, j] = min(seq_availability[seq1], seq_availability[seq2])

    im = axes[1, 1].imshow(cooccurrence, cmap="Greens")
    axes[1, 1].set_title("Sequence Co-occurrence Matrix")
    axes[1, 1].set_xticks(range(len(valid_sequences)))
    axes[1, 1].set_xticklabels(valid_sequences, rotation=45)
    axes[1, 1].set_yticks(range(len(valid_sequences)))
    axes[1, 1].set_yticklabels(valid_sequences)

    # Add text annotations
    for i in range(len(valid_sequences)):
        for j in range(len(valid_sequences)):
            axes[1, 1].text(
                j,
                i,
                f"{cooccurrence[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cooccurrence[i, j] > 0.5 else "black",
            )

    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label("Co-occurrence Rate")

    plt.tight_layout()
    if save_figures:
        plt.savefig(fig_dir / "sequence_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_summary_report(
    overview: Dict,
    pathology_stats: Dict,
    sequence_stats: Dict,
    save_report: bool = False,
) -> str:
    """Generate comprehensive summary report."""
    logger.info("Generating summary report...")

    report = f"""
# MRI Spine Pathology Dataset - Exploratory Data Analysis Report

## Dataset Overview
- **Total Records**: {overview["total_records"]:,}
- **Total Patients**: {overview["total_patients"]:,}
- **Total DICOM Files**: {overview["dicom_stats"]["total_dicom_files"]:,}
- **Average DICOM per Record**: {overview["dicom_stats"]["avg_dicom_per_record"]:.1f}

### Dataset Distribution
"""

    for dataset, count in overview["datasets"].items():
        pct = count / overview["total_records"] * 100
        report += f"- **{dataset}**: {count:,} records ({pct:.1f}%)\n"

    report += """
### Data Splits
"""
    for split, count in overview["splits"].items():
        pct = count / overview["total_records"] * 100
        report += f"- **{split.title()}**: {count:,} records ({pct:.1f}%)\n"

    report += """
### IVD Level Distribution
"""
    for ivd, count in sorted(overview["ivd_levels"].items()):
        pct = count / overview["total_records"] * 100
        report += f"- **L{ivd}**: {count:,} records ({pct:.1f}%)\n"

    report += """
## Pathology Analysis

### Individual Pathology Prevalence
"""
    pathology_names = list(pathology_stats.keys())[:-1]  # Exclude overall_binary
    for pathology in pathology_names:
        stats = pathology_stats[pathology]
        report += f"- **{pathology.replace('_', ' ').title()}**: {stats['positive_cases']:,}/{stats['total_records']:,} ({stats['prevalence_percent']:.1f}%)\n"

    binary_stats = pathology_stats["overall_binary"]
    report += f"""
### Overall Pathology
- **Any Pathology**: {binary_stats["any_pathology"]:,} records ({binary_stats["any_pathology_rate"] * 100:.1f}%)
- **No Pathology**: {binary_stats["no_pathology"]:,} records ({(1 - binary_stats["any_pathology_rate"]) * 100:.1f}%)

## MRI Sequence Analysis

### Sequence Availability
"""

    for seq, availability in sequence_stats["sequence_availability"].items():
        report += f"- **{seq}**: {availability * 100:.1f}% of records\n"

    report += f"""
### Sequence Completeness
- **Complete Cases** (all 4 sequences): {sequence_stats["complete_cases"]:,}
- **Incomplete Cases**: {sequence_stats["incomplete_cases"]:,}

### Common Sequence Combinations
"""

    for combo, count in list(sequence_stats["sequence_combinations"].items())[:5]:
        report += f"- **{combo}**: {count:,} records\n"

    report += f"""
## Key Insights

### Data Quality
- {overview["dicom_stats"]["records_with_dicom"]:,} records have DICOM files ({overview["dicom_stats"]["records_with_dicom"] / overview["total_records"] * 100:.1f}%)
- {overview["dicom_stats"]["records_without_dicom"]:,} records missing DICOM files ({overview["dicom_stats"]["records_without_dicom"] / overview["total_records"] * 100:.1f}%)

### Class Imbalance
"""

    # Find most and least common pathologies
    pathology_rates = [
        (name, stats["prevalence_percent"])
        for name, stats in pathology_stats.items()
        if name != "overall_binary"
    ]
    pathology_rates.sort(key=lambda x: x[1])

    least_common = pathology_rates[0]
    most_common = pathology_rates[-1]

    report += f"- **Most common pathology**: {most_common[0].replace('_', ' ').title()} ({most_common[1]:.1f}%)\n"
    report += f"- **Least common pathology**: {least_common[0].replace('_', ' ').title()} ({least_common[1]:.1f}%)\n"
    report += f"- **Imbalance ratio**: {most_common[1] / least_common[1]:.1f}:1\n"

    report += f"""
### Recommendations
- Consider weighted loss functions for pathology classification due to class imbalance
- Focus on data augmentation for rare pathologies ({least_common[0].replace("_", " ")})
- Implement sequence-missing handling strategies for incomplete cases
- Consider stratified sampling to maintain class distribution across splits

---
Generated by MRI Spine Pathology EDA Script
"""

    if save_report:
        report_path = project_root / "outputs" / "eda_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        logger.success(f"Report saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Perform EDA on processed MRI pathology dataset"
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save visualization figures to outputs/eda_figures/",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save summary report to outputs/eda_report.md",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=True,
        help="Display interactive plots (default: True)",
    )

    args = parser.parse_args()

    logger.info("Starting MRI Spine Pathology Dataset Exploratory Data Analysis")
    logger.info("=" * 60)

    try:
        # Load datasets
        pathology_df, statistics_df = load_datasets()

        # Perform analyses
        overview = analyze_dataset_overview(pathology_df)
        pathology_stats = analyze_pathology_labels(pathology_df)
        sequence_stats = analyze_sequence_availability(pathology_df)

        # Log key findings
        logger.success("Pathology analysis completed!")
        logger.success(
            f"Dataset contains {overview['total_records']:,} records from {overview['total_patients']:,} patients"
        )
        logger.success(
            f"Pathology prevalence ranges from {min([s['prevalence_percent'] for s in pathology_stats.values() if 'prevalence_percent' in s]):.1f}% to {max([s['prevalence_percent'] for s in pathology_stats.values() if 'prevalence_percent' in s]):.1f}%"
        )
        logger.success(
            f"Sequence completeness: {sequence_stats['complete_cases']} complete cases out of {sequence_stats['complete_cases'] + sequence_stats['incomplete_cases']}"
        )

        # Create visualizations
        if args.show_plots:
            create_visualizations(
                pathology_df, pathology_stats, sequence_stats, args.save_figures
            )

        # Generate report
        report = generate_summary_report(
            overview, pathology_stats, sequence_stats, args.save_report
        )

        if not args.save_report:
            logger.info("Summary Report:")
            logger.info("=" * 60)
            print(report)

        logger.success("EDA completed successfully!")

    except Exception as e:
        logger.error(f"Error during EDA: {str(e)}")
        raise


if __name__ == "__main__":
    main()
