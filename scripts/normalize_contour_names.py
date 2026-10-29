"""Replace semicolons in TRIDENT contour GeoJSON filenames with underscores."""

from __future__ import annotations

import argparse
from pathlib import Path


def normalize_contour_filenames(contour_dir: Path) -> list[tuple[Path, Path]]:
    """Rename semicolon-containing GeoJSON files without overwriting outputs."""
    contour_dir = contour_dir.expanduser().resolve()
    if not contour_dir.is_dir():
        return []

    renames = [
        (path, path.with_name(path.name.replace(";", "_")))
        for path in sorted(contour_dir.glob("*.geojson"))
        if ";" in path.name
    ]
    targets = [target for _, target in renames]
    if len(targets) != len(set(targets)):
        raise FileExistsError("Multiple contour filenames normalize to the same name")
    for source, target in renames:
        if target.exists():
            raise FileExistsError(f"Cannot rename {source}: target already exists: {target}")

    completed: list[tuple[Path, Path]] = []
    for source, target in renames:
        source.rename(target)
        completed.append((source, target))
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contour_dir", type=Path,
                        help="TRIDENT contours_geojson directory")
    args = parser.parse_args()
    renames = normalize_contour_filenames(args.contour_dir)
    print(f"Renamed {len(renames)} contour GeoJSON file(s) in {args.contour_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
