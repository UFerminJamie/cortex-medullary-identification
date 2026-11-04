"""Copy selected inference JSON files from one directory to another."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SUFFIXES = ("_refined_heatmap.json", "_mask_gt.json")


def copy_matching_files(
    input_dir: Path, out_dir: Path, suffixes: tuple[str, ...]
) -> tuple[int, int]:
    """Copy top-level JSON files ending in a requested suffix.

    Existing destination files are left unchanged. Returns the copied and
    skipped file counts.
    """
    input_dir = input_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if input_dir == out_dir:
        raise ValueError("Input and output directories must be different")

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for source in sorted(input_dir.glob("*.json")):
        if not source.is_file() or not source.name.endswith(suffixes):
            continue
        destination = out_dir / source.name
        if destination.exists():
            skipped += 1
            continue
        shutil.copy2(source, destination)
        copied += 1
    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing JSON files")
    parser.add_argument("out_dir", type=Path, help="Directory to receive matching files")
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=DEFAULT_SUFFIXES,
        metavar="SUFFIX",
        help=(
            "Filename endings to copy (default: %(default)s). Provide one or more, "
            "for example --suffixes _refined_heatmap.json _mask_gt.json"
        ),
    )
    args = parser.parse_args()

    try:
        copied, skipped = copy_matching_files(args.input_dir, args.out_dir, tuple(args.suffixes))
    except (NotADirectoryError, ValueError) as error:
        parser.error(str(error))
    print(f"Copied {copied} file(s) to {args.out_dir}; skipped {skipped} existing file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Run with defaults:
# python utils/copy_inference_jsons.py /path/to/input_dir /path/to/out_dir
