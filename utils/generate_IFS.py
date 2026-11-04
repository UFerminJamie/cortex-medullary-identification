"""Generate HistomicsUI annotations for interstitial fractional space (IFS)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely.validation import explain_validity


COLOR_RGB = "144, 238, 144"  # CSS light green
MASK_ALPHA = 0.35
CONTOUR_FILL_ALPHA = 0.2
FTU_SUFFIXES = ("_tubules", "_muscular_vessels")


def _polygon_parts(geometry: Any):
    """Yield polygon parts from polygonal geometries and geometry collections."""
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _histomics_polygon(element: dict[str, Any]) -> Polygon | None:
    if element.get("type") != "polyline" or element.get("closed") is not True:
        return None
    points = element.get("points", [])
    if len(points) < 3:
        return None
    exterior = [(float(point[0]), float(point[1])) for point in points]
    if len(set(exterior)) < 3:
        return None
    holes = [
        [(float(point[0]), float(point[1])) for point in ring]
        for ring in element.get("holes", []) if len(ring) >= 3
    ]
    polygon = Polygon(exterior, holes)
    return polygon if polygon.area > 0 else None


def _document_geometries(document: Any) -> list[Any]:
    """Read polygon geometries from GeoJSON or HistomicsUI annotation JSON."""
    if isinstance(document, list):
        geometries = []
        for row in document:
            geometries.extend(_document_geometries(row))
        return geometries
    if not isinstance(document, dict):
        raise ValueError("Annotation JSON must contain an object or list of objects")
    if isinstance(document.get("document"), dict):
        document = document["document"]

    kind = document.get("type")
    if kind == "FeatureCollection":
        geometries = []
        for feature in document.get("features", []):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                continue
            geometry = feature.get("geometry")
            if isinstance(geometry, dict) and geometry.get("type") in ("Polygon", "MultiPolygon"):
                geometries.append(shape(geometry))
        return geometries
    if kind == "Feature":
        geometry = document.get("geometry")
        if isinstance(geometry, dict) and geometry.get("type") in ("Polygon", "MultiPolygon"):
            return [shape(geometry)]
        return []
    if kind in ("Polygon", "MultiPolygon"):
        return [shape(document)]

    annotation = document.get("annotation", document)
    elements = annotation.get("elements") if isinstance(annotation, dict) else None
    if not isinstance(elements, list):
        raise ValueError("Expected GeoJSON polygons or HistomicsUI polyline elements")
    return [
        polygon for element in elements if isinstance(element, dict)
        if (polygon := _histomics_polygon(element)) is not None
    ]


def _load_geometries(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as stream:
        geometries = _document_geometries(json.load(stream))
    for geometry in geometries:
        if not geometry.is_empty and not geometry.is_valid:
            raise ValueError(f"Invalid polygon in {path}: {explain_validity(geometry)}")
    return [geometry for geometry in geometries if not geometry.is_empty]


def _union(geometries: list[Any]) -> Any | None:
    if not geometries:
        return None
    return unary_union(geometries)


def _index_ftu_files(ftu_dir: Path) -> dict[str, list[Path]]:
    """Index matching tubule and muscular-vessel JSONs by slide stem."""
    index: dict[str, list[Path]] = {}
    for path in sorted(ftu_dir.rglob("*.json")):
        for suffix in FTU_SUFFIXES:
            if path.stem.endswith(suffix):
                sample_stem = path.stem[:-len(suffix)]
                index.setdefault(sample_stem, []).append(path)
                break
    return index


def _make_row(sample_stem: str, geometry: Any, *, layer: str, alpha: float,
              source_geojson: Path, ftu_files: list[Path], ftu_count: int) -> list[dict]:
    elements = []
    for polygon in _polygon_parts(geometry):
        elements.append({
            "type": "polyline",
            "id": uuid4().hex[:24],
            "points": [[float(x), float(y), 0] for x, y in polygon.exterior.coords],
            "closed": True,
            "holes": [
                [[float(x), float(y), 0] for x, y in ring.coords]
                for ring in polygon.interiors
            ],
            "fillColor": f"rgba({COLOR_RGB}, {alpha:g})",
            "lineColor": f"rgba({COLOR_RGB}, 1)",
            "lineWidth": 2,
            "user": {"class": "interstitial fractional space"},
        })
    label = "mask" if layer == "ifs_mask" else "contours"
    return [{
        "name": f"interstitial fractional space {label}",
        "description": "Interstitial fractional space in level-zero WSI coordinates",
        "attributes": {
            "sample_name": sample_stem,
            "class": "interstitial fractional space",
            "coordinate_system": "level-zero WSI pixels",
            "layer": layer,
            "fill_alpha": alpha,
            "source_geojson": source_geojson.name,
            "ftu_count": ftu_count,
            "ftu_sources": [path.name for path in ftu_files],
        },
        "elements": elements,
    }]


def _write_json(path: Path, payload: list[dict], *, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"  Skipped existing output: {path}")
        return False
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"  Wrote {path}")
    return True


def generate_ifs(geojson_dir: Path, ftu_dir: Path, output_dir: Path, *,
                 mode: str = "both", overwrite: bool = False) -> tuple[int, int]:
    geojson_dir = geojson_dir.expanduser().resolve()
    ftu_dir = ftu_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not geojson_dir.is_dir():
        raise NotADirectoryError(f"GeoJSON directory does not exist: {geojson_dir}")
    if not ftu_dir.is_dir():
        raise NotADirectoryError(f"FTU directory does not exist: {ftu_dir}")
    if output_dir in (geojson_dir, ftu_dir):
        raise ValueError("Output directory must be separate from the input directories")
    if mode not in {"both", "mask", "contours"}:
        raise ValueError("mode must be 'both', 'mask', or 'contours'")

    ftu_index = _index_ftu_files(ftu_dir)
    geojson_by_stem = {
        path.stem: path for path in sorted(geojson_dir.glob("*.geojson"))
    }
    if not geojson_by_stem:
        print(f"No .geojson files found in {geojson_dir}")
        return 0, 0

    geojson_stems = set(geojson_by_stem)
    ftu_stems = set(ftu_index)
    matched_stems = sorted(geojson_stems & ftu_stems)
    unmatched_geojson = sorted(geojson_stems - ftu_stems)
    unmatched_ftus = sorted(ftu_stems - geojson_stems)
    print(
        f"Found {len(matched_stems)} matched slide(s); "
        f"{len(unmatched_geojson)} GeoJSON(s) and {len(unmatched_ftus)} FTU stem(s) unmatched"
    )
    for sample_stem in unmatched_geojson:
        print(f"SKIP {sample_stem}: no matching FTU JSON files")
    for sample_stem in unmatched_ftus:
        files = ", ".join(path.name for path in ftu_index[sample_stem])
        print(f"SKIP FTU JSON(s) for {sample_stem}: no matching GeoJSON ({files})")
    if not matched_stems:
        print("No matched GeoJSON and FTU inputs; no outputs generated")
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    for sample_stem in matched_stems:
        source_geojson = geojson_by_stem[sample_stem]
        ftu_files = ftu_index[sample_stem]
        print(f"{sample_stem}: {len(ftu_files)} matching FTU JSON file(s)")
        ftu_geometries = []
        ftu_count = 0
        try:
            for ftu_path in ftu_files:
                geometries = _load_geometries(ftu_path)
                ftu_count += len(geometries)
                ftu_geometries.extend(geometries)
                print(f"  {ftu_path}: {len(geometries)} FTU annotation(s)")
            print(f"  Total FTUs: {ftu_count}")
            if ftu_count == 0:
                print(f"SKIP {sample_stem}: matching FTU files contain no valid polygons")
                failed += 1
                continue

            tissue_geometries = _load_geometries(source_geojson)
            tissue = _union(tissue_geometries)
            if tissue is None:
                raise ValueError("GeoJSON contains no polygon geometries")

            ftu_union = _union(ftu_geometries)
            if ftu_union is not None:
                tissue = tissue.difference(ftu_union)
            tissue = _union(list(_polygon_parts(tissue)))
            if tissue is None:
                raise ValueError("Subtracting FTUs left no polygonal IFS geometry")

            outputs = []
            if mode in ("both", "mask"):
                outputs.append((
                    output_dir / f"{sample_stem}_IFS_mask.json",
                    _make_row(sample_stem, tissue, layer="ifs_mask", alpha=MASK_ALPHA,
                              source_geojson=source_geojson, ftu_files=ftu_files,
                              ftu_count=ftu_count),
                ))
            if mode in ("both", "contours"):
                outputs.append((
                    output_dir / f"{sample_stem}_IFS.json",
                    _make_row(sample_stem, tissue, layer="ifs_contours",
                              alpha=CONTOUR_FILL_ALPHA, source_geojson=source_geojson,
                              ftu_files=ftu_files, ftu_count=ftu_count),
                ))
            for output_path, payload in outputs:
                written += _write_json(output_path, payload, overwrite=overwrite)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failed += 1
            print(f"ERROR {sample_stem}: {error}")
    return written, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geojson_dir", type=Path, help="Directory of slide <stem>.geojson files")
    parser.add_argument("ftu_dir", type=Path, help="Directory containing FTU JSON annotations")
    parser.add_argument("output_dir", type=Path, help="Directory for generated IFS JSON files")
    parser.add_argument(
        "--mode", choices=("both", "mask", "contours"), default="both",
        help="Generate both outputs, only the mask, or only contours (default: both)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing IFS output JSON files",
    )
    args = parser.parse_args()
    try:
        written, failed = generate_ifs(
            args.geojson_dir, args.ftu_dir, args.output_dir,
            mode=args.mode, overwrite=args.overwrite,
        )
    except (NotADirectoryError, ValueError) as error:
        parser.error(str(error))
    print(f"Finished: wrote {written} output file(s); {failed} slide(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

# Example:
# python utils/generate_IFS.py /path/to/contours_geojson /path/to/ftu_jsons /path/to/ifs_jsons
