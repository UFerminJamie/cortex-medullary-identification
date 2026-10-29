"""Fill holes and prune small components in a DSA mask prediction JSON."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as PlotPolygon
from shapely.geometry import Polygon
from shapely.ops import unary_union


COLORS = {"cortical": "#00bcd4", "medullary": "#ff69b4"}
DEFAULT_MIN_AREA = 32_768.0


def _polygons(geometry: Any):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _polygons(part)


def _row_class(row: dict) -> str | None:
    value = row.get("attributes", {}).get("class")
    if value in COLORS:
        return value
    name = row.get("name", "").lower()
    return next((class_name for class_name in COLORS if class_name in name), None)


def _row_geometry(row: dict):
    polygons = []
    for element in row.get("elements", []):
        points = element.get("points")
        if element.get("type") != "polyline" or not points or element.get("closed") is False:
            continue
        exterior = [(float(point[0]), float(point[1])) for point in points]
        holes = [
            [(float(point[0]), float(point[1])) for point in ring]
            for ring in element.get("holes", []) if len(ring) >= 4
        ]
        if len(exterior) >= 4:
            polygons.append(Polygon(exterior, holes))
    valid = [polygon for polygon in polygons if polygon.is_valid and not polygon.is_empty]
    return unary_union(valid) if valid else None


def refine_geometry(geometry: Any, min_area: float):
    """Fill all interior holes; keep the largest part and parts above min_area."""
    if geometry is None or geometry.is_empty:
        return None, {"input_components": 0, "kept_components": 0, "largest_area": 0.0}

    hole_filled = unary_union([Polygon(part.exterior) for part in _polygons(geometry)])
    components = list(_polygons(hole_filled))
    if not components:
        return None, {"input_components": 0, "kept_components": 0, "largest_area": 0.0}

    largest = max(components, key=lambda part: part.area)
    kept = [part for part in components if part is largest or part.area >= min_area]
    refined = unary_union(kept)
    return refined, {
        "input_components": len(components),
        "kept_components": len(kept),
        "largest_area": float(largest.area),
        "removed_components": len(components) - len(kept),
    }


def _make_elements(geometry: Any, class_name: str, old_elements: list[dict]) -> list[dict]:
    template = old_elements[0] if old_elements else {}
    fill_color = template.get(
        "fillColor",
        "rgba(0, 255, 255, 0.35)" if class_name == "cortical"
        else "rgba(255, 105, 180, 0.35)",
    )
    line_color = template.get("lineColor", fill_color)
    user = copy.deepcopy(template.get("user", {}))
    user["class"] = class_name
    elements = []
    for polygon in _polygons(geometry):
        elements.append({
            "type": "polyline",
            "id": uuid4().hex[:24],
            "points": [[float(x), float(y), 0] for x, y in polygon.exterior.coords],
            "closed": True,
            "holes": [],
            "fillColor": fill_color,
            "lineColor": line_color,
            "lineWidth": template.get("lineWidth", 2),
            "user": copy.deepcopy(user),
        })
    return elements


def _draw_geometry(ax, geometry: Any, color: str) -> None:
    for polygon in _polygons(geometry):
        ax.add_patch(PlotPolygon(
            list(polygon.exterior.coords), closed=True, facecolor=color,
            edgecolor=color, linewidth=0.25, alpha=0.8,
        ))
        for ring in polygon.interiors:
            ax.add_patch(PlotPolygon(
                list(ring.coords), closed=True, facecolor="white",
                edgecolor="white", linewidth=0,
            ))


def save_comparison(original: dict[str, Any], refined: dict[str, Any], path: Path) -> None:
    all_geometries = [*original.values(), *refined.values()]
    bounds = [
        geometry.bounds for geometry in all_geometries
        if geometry is not None and not geometry.is_empty
    ]
    if not bounds:
        raise ValueError("No polygon geometry found in the input JSON")
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    pad_x = max((max_x - min_x) * 0.02, 1)
    pad_y = max((max_y - min_y) * 0.02, 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, layer, title in ((axes[0], original, "Original"), (axes[1], refined, "Refined")):
        for class_name in ("medullary", "cortical"):
            _draw_geometry(ax, layer.get(class_name), COLORS[class_name])
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(max_y + pad_y, min_y - pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("Level-zero WSI x (pixels)")
        ax.set_ylabel("Level-zero WSI y (pixels)")
    axes[0].legend(handles=[
        Patch(facecolor=COLORS["cortical"], label="Cortical"),
        Patch(facecolor=COLORS["medullary"], label="Medullary"),
    ], loc="upper right")
    fig.suptitle("Cortex and medulla mask postprocessing")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def postprocess(input_path: Path, output_json: Path, output_plot: Path, min_area: float) -> dict:
    document = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError("Expected a DSA annotation JSON list with cortical and medullary rows")

    original: dict[str, Any] = {name: None for name in COLORS}
    for row in document:
        class_name = _row_class(row)
        if class_name is None:
            continue
        geometry = _row_geometry(row)
        if geometry is not None:
            original[class_name] = (
                geometry if original[class_name] is None
                else unary_union([original[class_name], geometry])
            )

    refined: dict[str, Any] = {}
    stats = {}
    for class_name in COLORS:
        refined[class_name], stats[class_name] = refine_geometry(original[class_name], min_area)

    output_document = copy.deepcopy(document)
    emitted_classes: set[str] = set()
    for row in output_document:
        class_name = _row_class(row)
        if class_name is None:
            continue
        if class_name not in emitted_classes:
            row["elements"] = _make_elements(refined[class_name], class_name, row.get("elements", []))
            emitted_classes.add(class_name)
        else:
            row["elements"] = []
        row.setdefault("attributes", {})["postprocessing"] = {
            "holes_filled": True,
            "minimum_component_area_level0_px2": min_area,
            "largest_component_kept": True,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_document, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    save_comparison(original, refined, output_plot)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Input <sample>_mask_prediction.json")
    parser.add_argument("--output-json", type=Path, help="Refined JSON (default: add _postprocessed before .json)")
    parser.add_argument("--output-plot", type=Path, help="Comparison PNG (default: add _postprocessed_comparison before .png)")
    parser.add_argument(
        "--min-area", type=float, default=DEFAULT_MIN_AREA,
        help=("Minimum area in level-zero WSI pixel^2 for non-largest components "
              f"(default: {DEFAULT_MIN_AREA:g})"),
    )
    args = parser.parse_args()
    if not args.input_json.is_file():
        parser.error(f"Input file does not exist: {args.input_json}")
    if args.min_area < 0:
        parser.error("--min-area must be nonnegative")

    stem = args.input_json.with_suffix("")
    output_json = args.output_json or stem.with_name(f"{stem.name}_postprocessed.json")
    output_plot = args.output_plot or stem.with_name(f"{stem.name}_postprocessed_comparison.png")
    existing = [path for path in (output_json, output_plot) if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing output(s): " + ", ".join(map(str, existing)))
    stats = postprocess(args.input_json, output_json, output_plot, args.min_area)
    print(f"Refined JSON: {output_json}")
    print(f"Comparison image: {output_plot}")
    for class_name, values in stats.items():
        print(f"{class_name}: {values}")


if __name__ == "__main__":
    main()
