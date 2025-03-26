#!/usr/bin/env python3

import os
import re
import glob
import shutil
import argparse

def remove_optimizer_from_old_checkpoints(checkpoints_dir, exclude_steps):
    """
    Removes the subdirectory that starts with 'global_step' from all but the most
    recent checkpoint in the given directory.
    """
    # Pattern to match folders named 'checkpoint-XYZ'
    pattern = os.path.join(checkpoints_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(pattern)

    # Function to extract the numeric portion from a directory name like 'checkpoint-116'
    def extract_step_number(path):
        match = re.search(r"checkpoint-(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else -1

    # Sort checkpoint directories by their numeric step, ascending
    checkpoint_dirs = sorted(checkpoint_dirs, key=extract_step_number)

    # If there's at least one checkpoint, keep the last one fully intact
    # and remove optimizer states from the others
    if len(checkpoint_dirs) > 1:
        for old_dir in checkpoint_dirs[:-1]:
            # Find subdirectories starting with 'global_step'
            for item in os.listdir(old_dir):
                if item.startswith("global_step") and (exclude_steps is None or (int(item.replace("global_step", "")) not in exclude_steps)):
                    path_to_remove = os.path.join(old_dir, item)
                    if os.path.isdir(path_to_remove):
                        print(f"Removing {path_to_remove}")
                        shutil.rmtree(path_to_remove, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(
        description="Remove optimizer subdirectories from old Hugging Face checkpoints."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Path to the directory containing 'checkpoint-*' folders."
    )
    parser.add_argument(
        "--exclude_steps",
        type=int,
        nargs="+",
        required=False,
        help="Steps to exclude from removal."
    )
    args = parser.parse_args()
    remove_optimizer_from_old_checkpoints(args.checkpoints_dir, args.exclude_steps)

if __name__ == "__main__":
    main()