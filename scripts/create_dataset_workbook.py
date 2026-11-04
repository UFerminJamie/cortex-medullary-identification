"""Create a WSI workbook by recursively finding files with one extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "sample_name",
    "folder_path",
    "split",
    "num_cortex_tiles",
    "num_medullary_tiles",
]


def create_workbook(wsi_dir: Path, output: Path, extension: str = ".svs") -> int:
    wsi_dir = wsi_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if not wsi_dir.is_dir():
        raise NotADirectoryError(f"WSI directory not found: {wsi_dir}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() != ".xlsx":
        raise ValueError("Output filename must end in .xlsx")

    suffix = extension if extension.startswith(".") else f".{extension}"
    paths = sorted(
        (path for path in wsi_dir.rglob("*")
         if path.is_file() and path.suffix.lower() == suffix.lower()),
        key=lambda path: str(path).casefold(),
    )
    if not paths:
        raise ValueError(f"No {suffix} files found recursively in {wsi_dir}")

    rows = [
        {
            "sample_name": path.name,
            "folder_path": str(path),
            "split": "holdout",
            "num_cortex_tiles": 0.0,
            "num_medullary_tiles": 0.0,
        }
        for path in paths
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_excel(output, index=False, engine="openpyxl")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wsi_dir", type=Path,
                        help="Directory to search recursively for WSI files")
    parser.add_argument("output", type=Path,
                        help="Output workbook filename (must end in .xlsx)")
    parser.add_argument("--extension", default=".svs",
                        help="File extension to include (default: .svs; accepts svs or .svs)")
    args = parser.parse_args()
    count = create_workbook(args.wsi_dir, args.output, args.extension)
    print(f"Saved {args.output}: {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
