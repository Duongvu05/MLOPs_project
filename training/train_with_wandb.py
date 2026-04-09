"""
Example training script with wandb integration.

This is a template showing how to integrate wandb into your training pipeline.
Customize this based on your specific model and training requirements.

NOTE: This is a template file. You need to:
1. Implement your actual model, dataloaders, and training logic
2. Install required packages: pip install wandb torch pyyaml
3. Customize hyperparameters in configs/wandb_config.yaml
"""

import yaml
from pathlib import Path
import torch

# Import wandb utilities
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.wandb_utils import (
    init_wandb,
    log_metrics,
    log_model_checkpoint,
    log_config_file,
    finish_run,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log batch metrics (optional, can be done less frequently)
        if batch_idx % 10 == 0:
            batch_metrics = {
                "train/batch_loss": loss.item(),
                "train/batch_acc": 100.0 * correct / total,
            }
            log_metrics(batch_metrics, step=epoch * len(dataloader) + batch_idx)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "wandb_config.yaml"
    config = load_config(str(config_path))

    # Extract wandb settings
    wandb_settings = {
        "project_name": config["project"]["name"],
        "experiment_name": config["run"]["name"],
        "entity": config["project"].get("entity"),
        "tags": config["run"].get("tags", []),
        "notes": config["run"].get("notes", ""),
        "mode": config["run"].get("mode", "online"),
    }

    # Combine all hyperparameters
    hyperparameters = {
        **config.get("hyperparameters", {}),
        "config_file": str(config_path),
    }

    # Initialize wandb
    run = init_wandb(
        project_name=wandb_settings["project_name"],
        experiment_name=wandb_settings["experiment_name"],
        config=hyperparameters,
        entity=wandb_settings["entity"],
        tags=wandb_settings["tags"],
        notes=wandb_settings["notes"],
        mode=wandb_settings["mode"],
    )

    # Log configuration file
    if config.get("artifacts", {}).get("log_configs", True):
        log_config_file(str(config_path))

    # Setup device
    device = torch.device(
        hyperparameters.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # TODO: Initialize model, dataloaders, etc. (customize based on your setup)
    # Example (uncomment and customize):
    # from models.your_model import YourModel
    # from utils.dataset import YourDataset
    #
    # model = YourModel(**hyperparameters["model"])
    # train_dataset = YourDataset(...)
    # val_dataset = YourDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=hyperparameters["training"]["batch_size"], ...)
    # val_loader = DataLoader(val_dataset, batch_size=hyperparameters["training"]["batch_size"], ...)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["training"]["learning_rate"])

    # Placeholder - replace with your actual model and dataloaders
    model = None
    train_loader = None
    val_loader = None
    criterion = None
    optimizer = None

    if model is None:
        raise NotImplementedError(
            "Please implement your model, dataloaders, and training setup"
        )

    # Training loop
    best_val_acc = 0.0
    num_epochs = hyperparameters["training"]["num_epochs"]

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        }
        log_metrics(epoch_metrics, step=epoch)

        # Save checkpoint
        if config.get("artifacts", {}).get("log_model_checkpoints", True):
            checkpoint_dir = Path(
                config["paths"]["checkpoint_dir"].format(
                    experiment_name=wandb_settings["experiment_name"], timestamp=run.id
                )
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )

            # Log best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if config.get("artifacts", {}).get("log_best_model", True):
                    log_model_checkpoint(
                        str(checkpoint_path),
                        artifact_name=f"{wandb_settings['experiment_name']}_best",
                        aliases=["latest", "best"],
                        metadata={"epoch": epoch, "val_acc": val_acc},
                    )

    # Finish wandb run
    finish_run()
    print("Training completed!")


if __name__ == "__main__":
    main()
