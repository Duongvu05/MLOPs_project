#!/usr/bin/env python3
"""
Fetch results from multiple WandB projects and save to a comprehensive log file.

This script fetches results from all sequence combination projects and creates
a detailed comparison report.

Usage:
    python scripts/fetch_all_wandb_projects.py --entity YOUR_ENTITY --output results_log.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import wandb
from loguru import logger
from datetime import datetime
import json

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def get_recent_projects(api: wandb.Api, entity: str, limit: int = 19) -> List[str]:
    """
    Get list of recent projects for an entity.

    Args:
        api: wandb API instance
        entity: wandb entity/username
        limit: Maximum number of projects to return

    Returns:
        List of project names
    """
    try:
        logger.info(f"Fetching projects for entity: {entity}")

        # Get all projects for the entity
        projects = api.projects(entity)
        projects_list = list(projects)

        if not projects_list:
            logger.warning(f"No projects found for entity: {entity}")
            return []

        logger.info(f"Found {len(projects_list)} total projects")

        # Simply return first N projects (already in some order from API)
        # WandB API doesn't expose updated_at for Project objects directly
        project_names = [p.name for p in projects_list[:23]]

        logger.success(f"Selected {len(project_names)} projects")
        for i, name in enumerate(project_names, 1):
            logger.info(f"  {i}. {name}")

        return project_names

    except Exception as e:
        logger.error(f"Failed to fetch projects: {e}")
        return []


def get_wandb_api():
    """Initialize and return wandb API instance."""
    try:
        api = wandb.Api()
        logger.success("Connected to wandb API")
        return api
    except Exception as e:
        logger.error(f"Failed to connect to wandb API: {e}")
        logger.error("Make sure you're logged in: wandb login")
        sys.exit(1)


def fetch_project_results(api: wandb.Api, entity: str, project_name: str) -> tuple:
    """
    Fetch all results for a project.

    Args:
        api: wandb API instance
        entity: wandb entity/username
        project_name: Name of the project

    Returns:
        Tuple of (summary dict, list of all run results)
    """
    project_path = f"{entity}/{project_name}"

    try:
        runs = api.runs(project_path)
        runs_list = list(runs)

        if not runs_list:
            return (
                {
                    "project_name": project_name,
                    "num_runs": 0,
                    "status": "No runs found",
                },
                [],
            )

        # Extract metrics from all runs by downloading table files
        all_metrics = []
        for run in runs_list:
            try:
                summary = run.summary._json_dict
                config = run.config

                # Get table file path
                eval_results_info = summary.get("Evaluation Results", {})
                table_path = eval_results_info.get("path", "")

                if not table_path:
                    continue

                # Download and parse table file
                try:
                    file = run.file(table_path)
                    file_content = file.download(replace=True, root="/tmp/")

                    with open(file_content.name, "r") as f:
                        table_data = json.load(f)

                    # Extract metrics from table
                    columns = table_data.get("columns", [])
                    data = table_data.get("data", [])

                    if not data or not columns:
                        continue

                    # Get first row of data (should be only one row per run)
                    row = data[0]

                    # Create dict from columns and data
                    metrics_dict = dict(zip(columns, row))

                    # Extract relevant metrics
                    val_macro_f1 = metrics_dict.get("Macro F1", 0)

                    if val_macro_f1 == 0:
                        continue

                    metrics = {
                        "project_name": project_name,
                        "run_name": run.name,
                        "run_id": run.id,
                        "state": run.state,
                        "backbone": metrics_dict.get(
                            "Backbone", config.get("backbone", "unknown")
                        ),
                        "ivd_encoding_enabled": config.get(
                            "simple_multi_sequence_fusion", {}
                        ).get(
                            "use_ivd_encoding",
                            config.get("multi_sequence", {}).get(
                                "use_ivd_encoding", False
                            ),
                        ),
                        "ivd_encoding_mode": config.get(
                            "simple_multi_sequence_fusion", {}
                        ).get(
                            "ivd_encoding_mode",
                            config.get("multi_sequence", {}).get(
                                "ivd_encoding_mode", "unknown"
                            ),
                        ),
                        "val_macro_f1": val_macro_f1,
                        "val_micro_f1": metrics_dict.get("Micro F1", 0),
                        "val_weighted_f1": metrics_dict.get("Weighted F1", 0),
                        "val_subset_accuracy": metrics_dict.get("Subset Accuracy", 0),
                        "val_macro_roc_auc": metrics_dict.get("Macro ROC-AUC", 0),
                        "val_macro_auprc": metrics_dict.get("Macro AUPRC", 0),
                        "val_macro_precision": metrics_dict.get("Macro Precision", 0),
                        "val_macro_recall": metrics_dict.get("Macro Recall", 0),
                    }
                    all_metrics.append(metrics)

                except Exception as e:
                    logger.warning(f"Failed to fetch table for run {run.name}: {e}")
                    continue

            except Exception as e:
                logger.warning(f"Failed to extract metrics from run {run.id}: {e}")
                continue

        if not all_metrics:
            return (
                {
                    "project_name": project_name,
                    "num_runs": len(runs_list),
                    "status": "No valid metrics found",
                },
                [],
            )

        # Calculate statistics
        df = pd.DataFrame(all_metrics)

        # Find best run
        best_run = df.loc[df["val_macro_f1"].idxmax()]

        summary = {
            "project_name": project_name,
            "num_runs": len(runs_list),
            "completed_runs": len(df[df["state"] == "finished"]),
            "best_run_name": best_run["run_name"],
            "best_backbone": best_run["backbone"],
            "best_ivd_mode": best_run["ivd_encoding_mode"]
            if best_run["ivd_encoding_enabled"]
            else "disabled",
            "best_val_macro_f1": best_run["val_macro_f1"],
            "best_val_micro_f1": best_run.get("val_micro_f1", 0),
            "best_val_weighted_f1": best_run.get("val_weighted_f1", 0),
            "best_val_subset_acc": best_run["val_subset_accuracy"],
            "best_val_macro_roc_auc": best_run.get("val_macro_roc_auc", 0),
            "best_val_macro_precision": best_run.get("val_macro_precision", 0),
            "best_val_macro_recall": best_run.get("val_macro_recall", 0),
            "avg_val_macro_f1": df["val_macro_f1"].mean(),
            "std_val_macro_f1": df["val_macro_f1"].std(),
            "min_val_macro_f1": df["val_macro_f1"].min(),
            "max_val_macro_f1": df["val_macro_f1"].max(),
            "status": "success",
        }

        return (summary, all_metrics)

    except Exception as e:
        logger.error(f"Failed to fetch project {project_name}: {e}")
        return (
            {"project_name": project_name, "num_runs": 0, "status": f"Error: {str(e)}"},
            [],
        )


def write_log_header(log_file, entity: str):
    """Write log file header."""
    log_file.write("=" * 100 + "\n")
    log_file.write("WANDB PROJECTS RESULTS SUMMARY\n")
    log_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Entity: {entity}\n")
    log_file.write("=" * 100 + "\n\n")


def write_project_section(log_file, summary: Dict, index: int):
    """Write a project section to log file."""
    log_file.write(f"\n{'=' * 100}\n")
    log_file.write(f"PROJECT {index}: {summary['project_name']}\n")
    log_file.write(f"{'=' * 100}\n")

    if summary["status"] != "success":
        log_file.write(f"Status: {summary['status']}\n")
        log_file.write(f"Number of runs: {summary['num_runs']}\n")
        return

    log_file.write(f"Total runs: {summary['num_runs']}\n")
    log_file.write(f"Completed runs: {summary['completed_runs']}\n")
    log_file.write("\n")

    log_file.write("BEST RUN RESULTS:\n")
    log_file.write("-" * 80 + "\n")
    log_file.write(f"  Run name:           {summary['best_run_name']}\n")
    log_file.write(f"  Backbone:           {summary['best_backbone']}\n")
    log_file.write(f"  IVD Encoding Mode:  {summary['best_ivd_mode']}\n")
    log_file.write(f"  Val Macro F1:       {summary['best_val_macro_f1']:.4f}\n")
    log_file.write(f"  Val Micro F1:       {summary.get('best_val_micro_f1', 0):.4f}\n")
    log_file.write(
        f"  Val Weighted F1:    {summary.get('best_val_weighted_f1', 0):.4f}\n"
    )
    log_file.write(f"  Val Subset Acc:     {summary['best_val_subset_acc']:.4f}\n")
    log_file.write(
        f"  Val Macro ROC-AUC:  {summary.get('best_val_macro_roc_auc', 0):.4f}\n"
    )
    log_file.write(
        f"  Val Macro Precision:{summary.get('best_val_macro_precision', 0):.4f}\n"
    )
    log_file.write(
        f"  Val Macro Recall:   {summary.get('best_val_macro_recall', 0):.4f}\n"
    )
    log_file.write("\n")

    log_file.write("OVERALL STATISTICS:\n")
    log_file.write("-" * 80 + "\n")
    log_file.write(f"  Average Macro F1:   {summary['avg_val_macro_f1']:.4f}\n")
    log_file.write(f"  Std Dev Macro F1:   {summary['std_val_macro_f1']:.4f}\n")
    log_file.write("\n")


def write_comparison_table(log_file, all_summaries: List[Dict]):
    """Write comparison table of all projects."""
    log_file.write("\n" + "=" * 100 + "\n")
    log_file.write("COMPARISON TABLE - ALL PROJECTS\n")
    log_file.write("=" * 100 + "\n\n")

    # Filter successful projects
    successful = [s for s in all_summaries if s["status"] == "success"]

    if not successful:
        log_file.write("No successful projects to compare.\n")
        return

    # Sort by best macro F1
    successful.sort(key=lambda x: x["best_val_macro_f1"], reverse=True)

    # Write header
    log_file.write(
        f"{'Rank':<6} {'Project Name':<50} {'Best F1':<10} {'Avg F1':<10} {'Runs':<8}\n"
    )
    log_file.write("-" * 100 + "\n")

    # Write rows
    for i, summary in enumerate(successful, 1):
        project_name = summary["project_name"][:48]  # Truncate long names
        log_file.write(
            f"{i:<6} {project_name:<50} "
            f"{summary['best_val_macro_f1']:<10.4f} "
            f"{summary['avg_val_macro_f1']:<10.4f} "
            f"{summary['num_runs']:<8}\n"
        )

    log_file.write("\n")
    log_file.write(f"Best performing project: {successful[0]['project_name']}\n")
    log_file.write(f"Best Macro F1: {successful[0]['best_val_macro_f1']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch results from all WandB sequence combination projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--entity",
        type=str,
        default="11230526-business-ai-lab",
        required=False,
        help="wandb entity/username",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/wandb_projects_results.log",
        help="Output log file path (default: outputs/wandb_projects_results.log)",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/wandb_projects_results.csv",
        help="Output CSV file path for summary (default: outputs/wandb_projects_results.csv)",
    )

    parser.add_argument(
        "--all-runs-csv",
        type=str,
        default="outputs/wandb_all_runs_results.csv",
        help="Output CSV file for all runs (default: outputs/wandb_all_runs_results.csv)",
    )

    parser.add_argument(
        "--projects",
        type=str,
        nargs="+",
        default=None,
        help="Specific project names to fetch (optional)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of recent projects to fetch (default: 15)",
    )

    args = parser.parse_args()

    # Initialize wandb API
    api = get_wandb_api()

    # Get projects - either specified or recent ones
    if args.projects:
        projects = args.projects
        logger.info(f"Using specified projects: {len(projects)} projects")
    else:
        logger.info(f"Fetching {args.limit} most recent projects...")
        projects = get_recent_projects(api, args.entity, args.limit)

        if not projects:
            logger.error("No projects found")
            return

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch results from all projects
    all_summaries = []
    all_runs_data = []  # Store all individual run results

    logger.info(f"\nFetching results from {len(projects)} projects...")

    with open(output_path, "w", encoding="utf-8") as log_file:
        # Write header
        write_log_header(log_file, args.entity)

        # Fetch and write each project
        for i, project_name in enumerate(projects, 1):
            logger.info(f"[{i}/{len(projects)}] Fetching: {project_name}")

            summary, runs_data = fetch_project_results(api, args.entity, project_name)
            all_summaries.append(summary)
            all_runs_data.extend(runs_data)  # Collect all run data

            # Write to log file
            write_project_section(log_file, summary, i)

            # Progress update
            if summary["status"] == "success":
                logger.success(
                    f"  ✓ Best F1: {summary['best_val_macro_f1']:.4f} "
                    f"({summary['num_runs']} runs, collected {len(runs_data)} valid runs)"
                )
            else:
                logger.warning(f"  ⚠ {summary['status']}")

        # Write comparison table
        write_comparison_table(log_file, all_summaries)

    logger.success(f"Results saved to: {output_path}")

    # Save summary to CSV
    csv_path = Path(args.csv)
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(csv_path, index=False)
    logger.success(f"Summary CSV saved to: {csv_path}")

    # Save all runs to CSV
    all_runs_csv_path = Path(args.all_runs_csv)
    if all_runs_data:
        df_all_runs = pd.DataFrame(all_runs_data)
        df_all_runs.to_csv(all_runs_csv_path, index=False)
        logger.success(
            f"All runs CSV saved to: {all_runs_csv_path} ({len(all_runs_data)} runs)"
        )
    else:
        logger.warning("No run data to save")

    # Print summary
    successful = [s for s in all_summaries if s["status"] == "success"]
    logger.info("\nSummary:")
    logger.info(f"  Total projects: {len(projects)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed: {len(projects) - len(successful)}")

    if successful:
        best = max(successful, key=lambda x: x["best_val_macro_f1"])
        logger.info("\nBest overall result:")
        logger.info(f"  Project: {best['project_name']}")
        logger.info(f"  Macro F1: {best['best_val_macro_f1']:.4f}")
        logger.info(f"  Backbone: {best['best_backbone']}")
        logger.info(f"  IVD Mode: {best['best_ivd_mode']}")


if __name__ == "__main__":
    main()
