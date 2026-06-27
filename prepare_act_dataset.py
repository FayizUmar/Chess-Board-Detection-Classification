"""
Merges all valid chess-teleop LeRobot v3.0 recordings in ./recordings/ into a
single dataset at ./act_dataset/ ready for ACT training with lerobot_train.

Uses LeRobot's built-in aggregate_datasets utility (lerobot>=0.5.1).

Usage:
    python prepare_act_dataset.py
    python prepare_act_dataset.py --recordings-dir ./recordings --output-dir ./act_dataset

Empty recordings (0 episodes) are skipped automatically.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

logging.basicConfig(level=logging.INFO, format="%(message)s")


def find_valid_recordings(recordings_dir: Path) -> list[Path]:
    valid = []
    for rec in sorted(recordings_dir.iterdir()):
        if not rec.is_dir() or rec.name.startswith("."):
            continue
        info_path = rec / "meta" / "info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        if info["total_episodes"] == 0 or info["total_frames"] == 0:
            print(f"  skip {rec.name} (empty)")
            continue
        print(f"  use  {rec.name}: {info['total_episodes']} ep, {info['total_frames']} frames")
        valid.append(rec)
    return valid


def main(recordings_dir: Path, output_dir: Path):
    print(f"\nScanning recordings in: {recordings_dir}")
    valid = find_valid_recordings(recordings_dir)

    if not valid:
        print("No valid recordings found.")
        return

    total_ep = sum(
        json.load(open(r / "meta" / "info.json"))["total_episodes"] for r in valid
    )
    total_fr = sum(
        json.load(open(r / "meta" / "info.json"))["total_frames"] for r in valid
    )
    print(f"\n{len(valid)} recordings -> {total_ep} episodes, {total_fr} frames")
    print(f"Output: {output_dir}\n")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    # aggregate_datasets needs a repo_id per source (used only as a key internally)
    repo_ids = [f"local/{rec.name}" for rec in valid]
    roots = valid

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id="chess_pick_place",
        roots=roots,
        aggr_root=output_dir,
    )

    print(f"\nDataset ready at: {output_dir.resolve()}")
    print("\nTo train ACT, run:")
    print("  lerobot-train \\")
    print("    --policy.type=act \\")
    print("    --dataset.repo_id=chess_pick_place \\")
    print(f"    --dataset.root={output_dir.resolve()} \\")
    print("    --dataset.image_transforms.enable=true \\")
    print("    --output_dir=./act_output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ACT training dataset from chess teleop recordings"
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory containing all recording folders (default: ./recordings)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("act_dataset"),
        help="Output directory for merged dataset (default: ./act_dataset)",
    )
    args = parser.parse_args()
    main(args.recordings_dir, args.output_dir)
