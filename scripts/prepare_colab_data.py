"""Package processed training data into a zip for Google Colab upload.

Creates bci_training_data.zip containing only the processed .npy/.txt files
(sub-1/ directory + trial_index will be rebuilt on Colab).

Usage:
    python scripts/prepare_colab_data.py
    python scripts/prepare_colab_data.py --output my_data.zip
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package BCI training data for Colab")
    parser.add_argument(
        "--data-dir", type=str,
        default="./data/willett_handwriting",
        help="Path to willett_handwriting directory",
    )
    parser.add_argument(
        "--output", type=str,
        default="./bci_training_data.zip",
        help="Output zip file path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sub_dir = data_dir / "sub-1"

    if not sub_dir.exists():
        print(f"ERROR: {sub_dir} not found. Run data download first.")
        raise SystemExit(1)

    output_path = Path(args.output)
    print(f"Packaging {sub_dir} into {output_path}...")

    n_files = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(sub_dir):
            for f in files:
                filepath = Path(root) / f
                arcname = str(Path("willett_handwriting") / filepath.relative_to(data_dir))
                zf.write(filepath, arcname)
                n_files += 1

    size_mb = output_path.stat().st_size / 1e6
    print(f"Done! {n_files} files, {size_mb:.1f} MB")
    print(f"Upload {output_path} to Google Drive for Colab training.")


if __name__ == "__main__":
    main()
