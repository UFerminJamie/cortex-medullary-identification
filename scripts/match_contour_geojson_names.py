"""Match TRIDENT contour GeoJSON names to their corresponding contour files."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_JOB_DIR = Path(
    "/orange/pinaki.sarder/j.fermin/cort_med_wsi/"
    "trident_processed_final/IU401_529"
)


def _pair_key(stem: str) -> str:
    """Use TRIDENT/Fiona's semicolon normalization to identify each pair."""
    return stem.replace(";", "_")


def plan_renames(job_dir: Path) -> list[tuple[Path, Path]]:
    """Plan GeoJSON renames, failing before changes if pairs are ambiguous."""
    contour_dir = job_dir / "contours"
    geojson_dir = job_dir / "contours_geojson"
    if not contour_dir.is_dir():
        raise FileNotFoundError(f"Contour directory does not exist: {contour_dir}")
    if not geojson_dir.is_dir():
        raise FileNotFoundError(f"GeoJSON directory does not exist: {geojson_dir}")

    contours = sorted(path for path in contour_dir.iterdir() if path.is_file())
    geojson_files = sorted(
        path for path in geojson_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".geojson"
    )

    def unique_by_pair_key(paths: list[Path], label: str) -> dict[str, Path]:
        indexed: dict[str, Path] = {}
        for path in paths:
            key = _pair_key(path.stem)
            if key in indexed:
                raise ValueError(
                    f"Ambiguous {label} pair key {key!r}: "
                    f"{indexed[key].name!r} and {path.name!r}"
                )
            indexed[key] = path
        return indexed

    contour_by_key = unique_by_pair_key(contours, "contour")
    geojson_by_key = unique_by_pair_key(geojson_files, "GeoJSON")
    missing = sorted(contour_by_key.keys() - geojson_by_key.keys())
    extra = sorted(geojson_by_key.keys() - contour_by_key.keys())
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing GeoJSON for: {', '.join(missing)}")
        if extra:
            details.append(f"GeoJSON without contour: {', '.join(extra)}")
        raise ValueError("Contour/GeoJSON pairs do not match (" + "; ".join(details) + ")")

    renames = []
    for key, contour in contour_by_key.items():
        source = geojson_by_key[key]
        target = geojson_dir / f"{contour.stem}.geojson"
        if source != target:
            renames.append((source, target))

    targets = [target for _, target in renames]
    if len(targets) != len(set(targets)):
        raise FileExistsError("Multiple contour pairs would produce the same GeoJSON name")
    for source, target in renames:
        if target.exists() and target != source:
            raise FileExistsError(f"Cannot rename {source}: target already exists: {target}")
    return renames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job_dir", nargs="?", type=Path, default=DEFAULT_JOB_DIR,
        help=f"TRIDENT job directory containing contours/ and contours_geojson/ (default: {DEFAULT_JOB_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show planned renames without changing filenames",
    )
    args = parser.parse_args()

    renames = plan_renames(args.job_dir.expanduser())
    if args.dry_run:
        for source, target in renames:
            print(f"{source.name} -> {target.name}")
        print(f"Would rename {len(renames)} GeoJSON file(s)")
        return 0

    for source, target in renames:
        source.rename(target)
        print(f"{source.name} -> {target.name}")
    print(f"Renamed {len(renames)} GeoJSON file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
