#!/usr/bin/env python3
"""
Clean up wandb staging directory to free up disk space.

This script removes old wandb artifact staging files that may be taking up space.
Use with caution - only removes staging files, not actual artifacts.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta


def get_dir_size(path):
    """Get total size of directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except:
                    pass
    except:
        pass
    return total


def cleanup_wandb_staging(days_old=7, dry_run=True):
    """
    Clean up wandb staging directory.

    Args:
        days_old: Remove files older than this many days (default: 7)
        dry_run: If True, only show what would be deleted without actually deleting
    """
    wandb_staging = Path.home() / ".local" / "share" / "wandb" / "artifacts" / "staging"

    if not wandb_staging.exists():
        print(f"Wandb staging directory not found: {wandb_staging}")
        return

    # Get initial size
    initial_size = get_dir_size(wandb_staging)
    print(f"Wandb staging directory: {wandb_staging}")
    print(f"Initial size: {initial_size / (1024**3):.2f} GB")
    print(f"\nLooking for files older than {days_old} days...")

    cutoff_time = datetime.now() - timedelta(days=days_old)
    deleted_count = 0
    deleted_size = 0

    # Walk through staging directory
    for item in wandb_staging.iterdir():
        try:
            # Get file/directory modification time
            mtime = datetime.fromtimestamp(item.stat().st_mtime)

            if mtime < cutoff_time:
                size = get_dir_size(item) if item.is_dir() else item.stat().st_size
                deleted_size += size
                deleted_count += 1

                if dry_run:
                    print(
                        f"  Would delete: {item.name} ({size / (1024**2):.2f} MB, modified {mtime.strftime('%Y-%m-%d')})"
                    )
                else:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print(f"  Deleted: {item.name} ({size / (1024**2):.2f} MB)")
        except Exception as e:
            print(f"  Error processing {item.name}: {e}")

    if dry_run:
        print(
            f"\n[DRY RUN] Would delete {deleted_count} items ({deleted_size / (1024**3):.2f} GB)"
        )
        print("Run with dry_run=False to actually delete files")
    else:
        final_size = get_dir_size(wandb_staging)
        print(f"\nDeleted {deleted_count} items ({deleted_size / (1024**3):.2f} GB)")
        print(f"Final size: {final_size / (1024**3):.2f} GB")
        print(f"Freed: {(initial_size - final_size) / (1024**3):.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean up wandb staging directory")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Remove files older than N days (default: 7)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default: dry run)",
    )

    args = parser.parse_args()

    cleanup_wandb_staging(days_old=args.days, dry_run=not args.execute)
