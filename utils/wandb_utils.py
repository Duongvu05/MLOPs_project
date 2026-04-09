"""
Weights & Biases (wandb) utility functions for experiment tracking.

This module provides helper functions for initializing wandb runs,
logging metrics, saving artifacts, and managing experiment tracking.
"""

import os
import wandb
from typing import Dict, Any, Optional
from pathlib import Path


def init_wandb(
    project_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    resume: Optional[str] = None,
    mode: str = "online",
    dir: Optional[str] = None,
) -> wandb.Run:
    """
    Initialize a wandb run with project configuration.

    Args:
        project_name: Name of the wandb project
        experiment_name: Name of the experiment/run
        config: Dictionary of hyperparameters and configuration
        entity: Wandb entity/team name (optional)
        tags: List of tags for organizing runs (optional)
        notes: Notes/description for the run (optional)
        resume: Resume mode ("allow", "must", "never", or run_id)
        mode: Wandb mode ("online", "offline", "disabled")
        dir: Directory to store wandb files (defaults to outputs/)

    Returns:
        Initialized wandb.Run object
    """
    if dir is None:
        # Default to outputs directory
        dir = str(Path(__file__).parent.parent / "outputs")

    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
        entity=entity,
        tags=tags,
        notes=notes,
        resume=resume,
        mode=mode,
        dir=dir,
    )

    return run


def log_metrics(
    metrics: Dict[str, float], step: Optional[int] = None, commit: bool = True
):
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of metric names and values
        step: Step/epoch number (optional)
        commit: Whether to commit the metrics immediately
    """
    wandb.log(metrics, step=step, commit=commit)


def log_model_checkpoint(
    model_path: str,
    artifact_name: str,
    artifact_type: str = "model",
    aliases: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_on_error: bool = True,
):
    """
    Save model checkpoint as wandb artifact.

    Args:
        model_path: Path to the model checkpoint file
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (default: "model")
        aliases: List of aliases for the artifact (e.g., ["latest", "best"])
        metadata: Additional metadata for the artifact
        skip_on_error: If True, skip logging on errors (e.g., disk full) instead of raising

    Returns:
        True if successful, False if skipped due to error
    """
    try:
        artifact = wandb.Artifact(
            name=artifact_name, type=artifact_type, metadata=metadata or {}
        )
        artifact.add_file(model_path)

        if aliases:
            wandb.log_artifact(artifact, aliases=aliases)
        else:
            wandb.log_artifact(artifact)
        return True
    except OSError as e:
        if e.errno == 28:  # No space left on device
            if skip_on_error:
                print(
                    f"⚠ Warning: Skipping wandb artifact logging due to disk space issue: {e}"
                )
                print(f"  Model checkpoint saved locally at: {model_path}")
                return False
            else:
                raise
        else:
            # Re-raise other OSErrors
            raise
    except Exception as e:
        if skip_on_error:
            print(f"⚠ Warning: Failed to log model checkpoint to wandb: {e}")
            print(f"  Model checkpoint saved locally at: {model_path}")
            return False
        else:
            raise


def log_data_artifact(
    data_path: str,
    artifact_name: str,
    artifact_type: str = "dataset",
    description: Optional[str] = None,
    skip_on_error: bool = True,
):
    """
    Log dataset or data file as wandb artifact.

    Args:
        data_path: Path to the data file or directory
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (default: "dataset")
        description: Description of the artifact
        skip_on_error: If True, skip logging on errors (e.g., disk full) instead of raising

    Returns:
        True if successful, False if skipped due to error
    """
    try:
        artifact = wandb.Artifact(
            name=artifact_name, type=artifact_type, description=description
        )

        if os.path.isdir(data_path):
            artifact.add_dir(data_path)
        else:
            artifact.add_file(data_path)

        wandb.log_artifact(artifact)
        return True
    except OSError as e:
        if e.errno == 28:  # No space left on device
            if skip_on_error:
                print(
                    f"⚠ Warning: Skipping wandb artifact logging due to disk space issue: {e}"
                )
                print(f"  Data artifact saved locally at: {data_path}")
                return False
            else:
                raise
        else:
            # Re-raise other OSErrors
            raise
    except Exception as e:
        if skip_on_error:
            print(f"⚠ Warning: Failed to log data artifact to wandb: {e}")
            print(f"  Data artifact saved locally at: {data_path}")
            return False
        else:
            raise


def log_config_file(config_path: str):
    """
    Log configuration file to wandb.

    Args:
        config_path: Path to the configuration file
    """
    wandb.save(config_path, base_path=os.path.dirname(config_path))


def log_predictions(
    predictions: Any,
    targets: Any,
    class_names: Optional[list] = None,
    table_name: str = "predictions",
):
    """
    Log predictions as wandb table for visualization.

    Args:
        predictions: Model predictions (numpy array or list)
        targets: Ground truth targets (numpy array or list)
        class_names: List of class names for classification tasks
        table_name: Name for the wandb table
    """
    import numpy as np

    # Convert to numpy if needed
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    # Create table columns
    columns = ["target", "prediction"]
    if class_names:
        columns.extend([f"class_{i}" for i in range(len(class_names))])

    # Create table data
    data = []
    for i in range(len(predictions)):
        row = [int(targets[i]), int(predictions[i])]
        if class_names:
            # Add one-hot or probability distribution if available
            pass  # Extend based on your prediction format
        data.append(row)

    table = wandb.Table(columns=columns, data=data)
    wandb.log({table_name: table})


def log_image(image, caption: Optional[str] = None, name: Optional[str] = None):
    """
    Log image to wandb.

    Args:
        image: Image array, PIL Image, or path to image file
        caption: Caption for the image
        name: Name/key for the image in wandb
    """
    wandb.log({name or "image": wandb.Image(image, caption=caption)})


def finish_run():
    """Finish and close the current wandb run."""
    wandb.finish()


def get_wandb_config() -> Dict[str, Any]:
    """
    Get the current wandb run configuration.

    Returns:
        Dictionary of configuration parameters
    """
    return dict(wandb.config) if wandb.run else {}


def set_wandb_config(key: str, value: Any):
    """
    Update wandb config during run (for dynamic hyperparameters).

    Args:
        key: Configuration key
        value: Configuration value
    """
    if wandb.run:
        wandb.config.update({key: value}, allow_val_change=True)
