"""Run an H5 SVC model on a workbook split and write spatial prediction maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
from shapely.ops import unary_union

from .gt_visualization import _polygons
from .h5_svm import (FEATURE_ROOT, KPMP, decision_scores, load_model, load_tiles,
                     safe_name, workbook_rows)
from .tile_annotations import ANNOTATION_FILES, load_annotation_geometry


DEFAULT_ANNOTATION_DIR = Path(
    "/orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed/annotations")
DEFAULT_TISSUE_MASK_DIR = Path(
    "/orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed_final/KPMP/contours_geojson")


def _polygonal(geometry: Any | None) -> Any | None:
    """Drop line and point remnants produced by polygon intersections."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    if geometry.geom_type == "GeometryCollection":
        parts = [_polygonal(part) for part in geometry.geoms]
        parts = [part for part in parts if part is not None]
        return unary_union(parts) if parts else None
    return None


def aggregate(coords: np.ndarray, scores: np.ndarray, patch: int, stride: int,
              sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Put overlapping predictions on source-grid cells; medullary wins ties/conflicts."""
    if patch % stride:
        raise ValueError("Patch size must be a multiple of source stride for exact aggregation")
    origin = coords.min(axis=0)
    if np.any((coords - origin) % stride):
        raise ValueError("Tile coordinates do not lie on the H5 source grid")
    cell_xy = ((coords - origin) // stride).astype(np.int64)
    factor = patch // stride
    width = int(cell_xy[:, 0].max() + factor)
    height = int(cell_xy[:, 1].max() + factor)
    raw = np.full((height, width), -np.inf, dtype=np.float32)
    seen = np.zeros((height, width), dtype=bool)
    conflict = np.zeros((height, width), dtype=bool)
    med_seen = np.zeros((height, width), dtype=bool)
    cortex_seen = np.zeros((height, width), dtype=bool)
    is_med = scores >= 0
    for dy in range(factor):
        for dx in range(factor):
            yy, xx = cell_xy[:, 1] + dy, cell_xy[:, 0] + dx
            np.maximum.at(raw, (yy, xx), scores)
            seen[yy, xx] = True
            med_seen[yy[is_med], xx[is_med]] = True
            cortex_seen[yy[~is_med], xx[~is_med]] = True
    conflict[:] = med_seen & cortex_seen
    # Gaussian smoothing ignores unobserved cells by smoothing numerator and coverage separately.
    clipped = np.where(seen, raw, 0)
    smooth = gaussian_filter(clipped, sigma=sigma) / np.maximum(
        gaussian_filter(seen.astype(np.float32), sigma=sigma), 1e-6)
    smooth[conflict] = np.maximum(smooth[conflict], 1e-6)
    return origin, raw, smooth, seen


def heatmap_payload(sample_name: str, mode: str, coords: np.ndarray, scores: np.ndarray,
                    tile_width: int, tile_height: int | None = None, *,
                    layer: str = "heatmap", origin_level0: np.ndarray | None = None,
                    source_stride: int | None = None,
                    tissue_geometry: Any | None = None) -> list[dict]:
    """Build separate DSA rows of signed scores clipped to available tissue."""
    coords = np.asarray(coords)
    scores = np.asarray(scores, dtype=np.float32)
    tile_height = tile_width if tile_height is None else tile_height
    if (coords.ndim != 2 or coords.shape[1] != 2 or scores.shape != (len(coords),)
            or not np.isfinite(coords).all() or not np.equal(coords, np.floor(coords)).all()
            or not np.isfinite(scores).all() or tile_width <= 0 or tile_height <= 0):
        raise ValueError("Heatmap tiles need finite integer coordinates, scores, and positive dimensions")
    if origin_level0 is None:
        origin_level0 = coords.min(axis=0) if len(coords) else np.array([0, 0])
    origin_level0 = np.asarray(origin_level0, dtype=np.int64)
    if origin_level0.shape != (2,):
        raise ValueError("Heatmap origin must be an (x, y) level-zero coordinate")

    tile_geometries = []
    retained = np.ones(len(coords), dtype=bool)
    for index, (x, y) in enumerate(coords):
        footprint = box(float(x), float(y), float(x + tile_width), float(y + tile_height))
        clipped = footprint if tissue_geometry is None else footprint.intersection(tissue_geometry)
        tile_geometries.append((footprint, clipped))
        tile_geometries[-1] = (footprint, _polygonal(clipped))
        retained[index] = tile_geometries[-1][1] is not None
    cortex = (scores < 0) & retained
    medullary = (scores >= 0) & retained
    maxima = {
        "cortical": float(np.max(np.abs(scores[cortex]))) if cortex.any() else 0.0,
        "medullary": float(np.max(np.abs(scores[medullary]))) if medullary.any() else 0.0,
    }
    documents = []
    for class_name, selected in (
        ("cortical", cortex),
        ("medullary", medullary),
    ):
        elements = []
        for index in np.flatnonzero(selected):
            x, y = (int(value) for value in coords[index])
            score = float(scores[index])
            maximum = maxima[class_name]
            scaled_score = abs(score) / maximum if maximum else 0.0
            footprint, clipped = tile_geometries[index]
            user = {
                "class": class_name,
                "decision_score": score,
                "scaled_score": scaled_score,
                "tile_x": x,
                "tile_y": y,
                "tile_width": tile_width,
                "tile_height": tile_height,
            }
            fill_rgb = "0, 255, 255" if class_name == "cortical" else "255, 105, 180"
            if clipped.equals(footprint):
                elements.append({
                    "type": "rectangle",
                    "id": uuid4().hex[:24],
                    "center": [x + tile_width / 2, y + tile_height / 2, 0],
                    "width": tile_width,
                    "height": tile_height,
                    "rotation": 0,
                    "fillColor": f"rgba({fill_rgb}, {scaled_score:.8g})",
                    "lineColor": "rgba(0, 0, 0, 0)",
                    "lineWidth": 0,
                    "user": user,
                })
            else:
                for polygon in _polygons(clipped):
                    elements.append(_polygon_element(
                        polygon, class_name, fill_rgb, scaled_score, user=user))
        documents.append({
            "name": f"{class_name} {layer.replace('_', ' ')}",
            "description": (
                f"{class_name.capitalize()} tiles with signed SVC decision scores; "
                "opacity is scaled by the maximum score magnitude within this class"
            ),
            "attributes": {
                "sample_name": sample_name,
                "tile_mode": mode,
                "layer": layer,
                "class": class_name,
                "origin_level0": origin_level0.tolist(),
                "cell_size_level0": source_stride,
                "source_stride_level0": source_stride,
                "tile_width_level0": tile_width,
                "tile_height_level0": tile_height,
                "classes": {"0": "cortex", "1": "medullary"},
                "colors": {"0": "#00FFFF", "1": "#FF69B4"},
                "scoreColumn": "decision_score",
                "decision_threshold": 0,
                "maxMagnitude": maxima,
                "numberOfTiles": int(retained.sum()),
                "classTileCount": int(selected.sum()),
                "params": {
                    "sampleName": sample_name,
                    "tileSize": tile_width,
                    "tileWidth": tile_width,
                    "tileHeight": tile_height,
                    "scoreColumn": "decision_score",
                    "layer": layer,
                    "numberOfTiles": int(retained.sum()),
                    "maxMagnitude": maxima,
                },
            },
            "elements": elements,
        })
    return documents


def _polygon_element(polygon: Any, class_name: str, fill_rgb: str, alpha: float,
                     *, user: dict[str, Any] | None = None) -> dict:
    points = [[float(x), float(y), 0] for x, y in polygon.exterior.coords]
    holes = [
        [[float(x), float(y), 0] for x, y in ring.coords]
        for ring in polygon.interiors
    ]
    return {
        "type": "polyline",
        "id": uuid4().hex[:24],
        "points": points,
        "closed": True,
        "holes": holes,
        "fillColor": f"rgba({fill_rgb}, {alpha:g})",
        "lineColor": f"rgba({fill_rgb}, 1)",
        "lineWidth": 2,
        "user": {"class": class_name, **(user or {})},
    }


def contour_payload(sample_name: str, cortex: Any | None, medullary: Any | None,
                    alpha: float = 0.35, *,
                    source_status: dict[str, str] | None = None,
                    tissue_geometry: Any | None = None,
                    layer: str = "mask_gt") -> list[dict]:
    """Build separate DSA contour rows, clipped to tissue with medulla precedence."""
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Contour alpha must be between 0 and 1")
    cortical_export = cortex
    if (cortex is not None and not cortex.is_empty and medullary is not None
            and not medullary.is_empty):
        cortical_export = cortex.difference(medullary)
    if tissue_geometry is not None:
        if cortical_export is not None:
            cortical_export = _polygonal(cortical_export.intersection(tissue_geometry))
        if medullary is not None:
            medullary = _polygonal(medullary.intersection(tissue_geometry))
    source_status = source_status or {}
    documents = []
    for class_name, geometry, fill_rgb, source_key in (
        ("cortical", cortical_export, "0, 255, 255", "cortex"),
        ("medullary", medullary, "255, 105, 180", "medullary"),
    ):
        elements = []
        for polygon in _polygons(geometry):
            elements.append(_polygon_element(polygon, class_name, fill_rgb, alpha))
        documents.append({
            "name": (f"{class_name} mask gt" if layer == "mask_gt"
                     else f"{class_name} {layer.replace('_', ' ')}"),
            "description": f"{class_name.capitalize()} {layer.replace('_', ' ')} in level-zero WSI coordinates",
            "attributes": {
                "sample_name": sample_name,
                "class": class_name,
                "coordinate_system": "level-zero WSI pixels",
                "source_status": source_status.get(source_key, "empty"),
                "fill_alpha": alpha,
                "layer": layer,
                **({"decision_threshold": 0, "prediction_source": "heatmap"}
                   if layer == "mask_prediction" else {}),
            },
            "elements": elements,
        })
    return documents


def load_sample_contours(annotation_dir: Path, sample_name: str) -> tuple[dict, dict]:
    """Load source GT geometries while retaining missing versus empty file state."""
    sample_dir = annotation_dir / sample_name
    paths = {name: sample_dir / filename for name, filename in ANNOTATION_FILES.items()}
    geometries = {name: load_annotation_geometry(path) for name, path in paths.items()}
    statuses = {
        name: ("missing" if not path.is_file() else
               "empty" if geometries[name] is None or geometries[name].is_empty else "present")
        for name, path in paths.items()
    }
    return geometries, statuses


def load_tissue_geometry(tissue_mask_dir: Path, sample_name: str) -> Any:
    """Load one slide's GeoJSON tissue footprint from the configured directory."""
    sample_name = Path(str(sample_name)).name
    # A slide stem can itself end in a suffix such as ".pyr". Try it intact
    # before removing a possible WSI extension from a filename.
    path = None
    for stem in dict.fromkeys((sample_name, Path(sample_name).stem)):
        safe_stem = safe_name(stem)
        candidates = [
            tissue_mask_dir / f"{stem}.geojson",
            tissue_mask_dir / f"{safe_stem}.geojson",
            tissue_mask_dir / f"{stem}_contours.geojson",
            tissue_mask_dir / f"{safe_stem}_contours.geojson",
            tissue_mask_dir / f"{stem}.json",
            tissue_mask_dir / stem / "contours.geojson",
            tissue_mask_dir / safe_stem / "contours.geojson",
            tissue_mask_dir / stem / "tissue.geojson",
            tissue_mask_dir / safe_stem / "tissue.geojson",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None and tissue_mask_dir.is_dir():
            names = (stem, safe_stem)
            matches = sorted(candidate for candidate in tissue_mask_dir.rglob("*.geojson")
                             if any(candidate.stem == name or
                                    candidate.stem.startswith((f"{name}.", f"{name}_"))
                                    for name in names))
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Multiple tissue GeoJSON files match {stem}: {matches}")
        if path is not None:
            break
    if path is None:
        raise FileNotFoundError(
            f"No tissue GeoJSON found for {sample_name} in {tissue_mask_dir}")
    geometry = load_annotation_geometry(path)
    if geometry is None or geometry.is_empty:
        raise ValueError(f"Tissue GeoJSON has no polygon geometry: {path}")
    return geometry


def threshold_grid_geometry(scores: np.ndarray, seen: np.ndarray, origin: np.ndarray,
                            stride: int, tissue_geometry: Any, *, medullary: bool) -> Any:
    """Polygonize thresholded source-grid cells, then clip them to tissue."""
    selected = np.asarray(seen, dtype=bool) & (
        np.asarray(scores) >= 0 if medullary else np.asarray(scores) < 0)
    x0, y0 = (int(value) for value in origin)
    runs = []
    for row_index, row in enumerate(selected):
        padded = np.r_[False, row, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, stop in zip(changes[::2], changes[1::2]):
            runs.append(box(x0 + int(start) * stride, y0 + row_index * stride,
                            x0 + int(stop) * stride, y0 + (row_index + 1) * stride))
    if not runs:
        return None
    return _polygonal(unary_union(runs).intersection(tissue_geometry))


def _save_json_new(path: Path, payload: list[dict]) -> None:
    """Create a JSON export only if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        try:
            json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        except BaseException:
            path.unlink(missing_ok=True)
            raise


def save_sample_exports(output_dir: Path, sample_name: str,
                        refined_heatmap: list[dict], mask_prediction: list[dict],
                        mask_gt: list[dict] | None = None, *,
                        raw_heatmap: list[dict]) -> list[Path]:
    """Write DSA exports for a sample stem without replacing existing files."""
    sample_name = Path(str(sample_name)).name
    stem = safe_name(sample_name)
    payloads = [
        (output_dir / f"{stem}_raw_heatmap.json", raw_heatmap),
        (output_dir / f"{stem}_refined_heatmap.json", refined_heatmap),
        (output_dir / f"{stem}_mask_prediction.json", mask_prediction),
    ]
    if mask_gt is not None:
        payloads.append((output_dir / f"{stem}_mask_gt.json", mask_gt))
    existing = [path for path, _ in payloads if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing DSA export(s): "
                              + ", ".join(map(str, existing)))
    for path, payload in payloads:
        _save_json_new(path, payload)
    return [path for path, _ in payloads]


def map_payload(sample_name: str, mode: str, origin: np.ndarray, stride: int,
                scores: np.ndarray, seen: np.ndarray, *, mask: bool,
                layer: str) -> dict | list[dict]:
    """Build a grid-data mask or two class-separated score annotations."""
    if scores.shape != seen.shape or not seen.any() or not np.isfinite(scores[seen]).all():
        raise ValueError("Prediction map needs a nonempty, finite observed grid")
    attributes = {"sample_name": sample_name, "tile_mode": mode, "layer": layer,
                  "origin_level0": origin.tolist(), "cell_size_level0": stride,
                  "classes": {"0": "cortex", "1": "medullary"},
                  "decision_threshold": 0}
    if mask:
        values = (scores >= 0).astype(np.float64)
        elements = [{"type": "griddata", "id": uuid4().hex[:24], "interpretation": "choropleth",
                     "origin": [float(origin[0]) + stride / 2,
                                float(origin[1]) + stride / 2, 0],
                     "dx": stride, "dy": stride, "gridWidth": scores.shape[1],
                     "values": np.where(seen, values, -1).ravel().tolist(),
                     "colorRange": ["rgba(0,0,255,0.5)", "rgba(255,0,0,0.5)"],
                     "rangeValues": [0, 1], "normalizeRange": False,
                     "minColor": "rgba(0,0,0,0)"}]
        description = "0=cortex, 1=medullary; unobserved cells are transparent"
    else:
        yy, xx = np.nonzero(seen)
        coords = np.column_stack((origin[0] + xx * stride, origin[1] + yy * stride))
        return heatmap_payload(
            sample_name, mode, coords, scores[yy, xx], stride,
            layer=layer, origin_level0=origin, source_stride=stride,
        )
    return {"name": f"{sample_name} {layer.replace('_', ' ')}",
            "description": description, "attributes": attributes,
            "elements": elements}


def save_visualizations(h5_path: Path, stem: str, origin: np.ndarray, stride: int,
                        raw: np.ndarray, refined: np.ndarray, seen: np.ndarray,
                        output_dir: Path, annotation_dir: Path | None = None,
                        tissue_geometry: Any | None = None,
                        mask_alpha: float = 0.35) -> None:
    """Save raw/refined score overlays and their thresholded class masks."""
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.patches import Patch
    from PIL import Image

    thumbnail_path = h5_path.parents[2] / "thumbnails" / f"{h5_path.stem}.jpg"
    if not thumbnail_path.is_file():
        raise FileNotFoundError(f"Thumbnail not found: {thumbnail_path}")
    with h5py.File(h5_path, "r") as handle:
        attrs = handle["coords"].attrs
        width = int(attrs.get("level0_width", handle.attrs.get("level0_width", 0)))
        height = int(attrs.get("level0_height", handle.attrs.get("level0_height", 0)))
    if width <= 0 or height <= 0:
        raise ValueError(f"H5 coords need positive level0_width and level0_height: {h5_path}")
    with Image.open(thumbnail_path) as source:
        thumbnail = np.asarray(source.convert("RGB"))

    # Both extents are in level-zero pixels, so sparse/nonzero-origin grids
    # align with the thumbnail and unobserved cells leave it visible.
    slide_extent = (0, width, height, 0)
    grid_extent = (int(origin[0]), int(origin[0] + raw.shape[1] * stride),
                   int(origin[1] + raw.shape[0] * stride), int(origin[1]))
    limit = max(float(np.max(np.abs(raw[seen]))),
                float(np.max(np.abs(refined[seen]))), 1e-6)
    norm = Normalize(vmin=-limit, vmax=limit)
    score_cmap = plt.get_cmap("coolwarm").copy()
    score_cmap.set_bad(alpha=0)
    mask_cmap = ListedColormap(["#00ffff", "#ff69b4"])
    mask_cmap.set_bad(alpha=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    contours = []
    if annotation_dir is not None:
        sample_dir = annotation_dir / h5_path.stem
        for name, color in (("cortex", "#00ffff"), ("medullary", "#ff69b4")):
            geometry = load_annotation_geometry(sample_dir / ANNOTATION_FILES[name])
            for polygon in _polygons(geometry):
                contours.extend((np.asarray(ring.coords), color)
                                for ring in (polygon.exterior, *polygon.interiors))

    visible_cells = seen.copy()
    if tissue_geometry is not None:
        for row_index, column_index in zip(*np.nonzero(seen)):
            cell = box(origin[0] + column_index * stride,
                       origin[1] + row_index * stride,
                       origin[0] + (column_index + 1) * stride,
                       origin[1] + (row_index + 1) * stride)
            visible_cells[row_index, column_index] = (
                cell.intersection(tissue_geometry).area > 0)
    maps = [("heatmap", raw, False), ("refined_heatmap", refined, False),
            ("mask_h", (raw >= 0).astype(np.int8), True),
            ("mask_rh", (refined >= 0).astype(np.int8), True)]
    for name, values, is_mask in maps:
        fig, ax = plt.subplots(figsize=(12, 10))
        try:
            ax.imshow(thumbnail, extent=slide_extent, origin="upper")
            visible = np.ma.array(values, mask=~visible_cells)
            if is_mask:
                ax.imshow(visible, extent=grid_extent, origin="upper",
                          interpolation="nearest", cmap=mask_cmap,
                          norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), alpha=mask_alpha)
                ax.legend(handles=[Patch(color="#00ffff", label="Cortex (< 0)"),
                                   Patch(color="#ff69b4", label="Medullary (>= 0)")],
                          loc="lower right")
            else:
                overlay = ax.imshow(visible, extent=grid_extent, origin="upper",
                                    interpolation="nearest", cmap=score_cmap,
                                    norm=norm, alpha=0.6)
                fig.colorbar(overlay, ax=ax, label="Signed SVC decision score",
                             fraction=0.046, pad=0.04)
            for points, color in contours:
                ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5,
                        zorder=3)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{h5_path.stem} {name.replace('_', ' ')}")
            fig.savefig(output_dir / f"{stem}_{name}.png", dpi=160,
                        bbox_inches="tight")
        finally:
            plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", type=Path, required=True)
    p.add_argument("--h5-dir", type=Path, default=FEATURE_ROOT)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--split", required=True, help="Workbook split, e.g. holdout")
    p.add_argument("--output-dir", type=Path, default=KPMP / "inference_jsons")
    p.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False,
                   help="Save heatmap and thresholded-mask PNG overlays")
    p.add_argument("--visualization-dir", type=Path,
                   default=KPMP / "prediction_visualization")
    p.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR,
                   help="Folder with per-slide cortical_gt.json and medullary_gt.json files")
    p.add_argument("--tissue-mask-dir", type=Path, default=DEFAULT_TISSUE_MASK_DIR,
                   help="Folder with per-slide tissue contour GeoJSON files")
    p.add_argument("--contour-alpha", type=float, default=0.35,
                   help="Fill opacity for GT and predicted DSA masks (default: 0.35)")
    p.add_argument("--tile-mode", choices=("all", "no-overlap"), default="all",
                   help="Tiles to predict and visualize (default: all)")
    p.add_argument("--gaussian-sigma", type=float, default=1.0,
                   help="Gaussian sigma measured in source-grid cells")
    p.add_argument("--batch-size", type=int, default=65536)
    args = p.parse_args()
    if args.gaussian_sigma < 0:
        p.error("--gaussian-sigma must be nonnegative")
    if not np.isfinite(args.contour_alpha) or not 0 <= args.contour_alpha <= 1:
        p.error("--contour-alpha must be between 0 and 1")
    model = load_model(args.model)
    group_name = f"{args.model.stem}_{args.gaussian_sigma}"
    output_dir = args.output_dir / group_name
    visualization_dir = args.visualization_dir / group_name
    expected_mode = model["metadata"].get("tile_mode")
    mode = args.tile_mode
    if expected_mode and mode != expected_mode:
        print(f"NOTE: using {mode} tiles for inference; model was trained with {expected_mode} tiles",
              flush=True)
    rows = workbook_rows(args.xlsx, args.h5_dir, args.split, skip_missing=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows.itertuples(index=False):
        sample = load_tiles(row.h5_path, mode, require_labels=False)
        assert sample is not None
        sample_name = Path(str(row.sample_name)).stem
        if sample.features.shape[1] != model["metadata"]["feature_dimension"]:
            raise ValueError(f"Feature dimension mismatch for {sample_name}")
        scores = decision_scores(model, sample.features, args.batch_size)
        origin, raw, refined, seen = aggregate(sample.coords, scores, sample.patch,
                                                sample.source_stride, args.gaussian_sigma)
        stem = safe_name(sample_name)
        contours, source_status = load_sample_contours(args.annotation_dir, sample_name)
        tissue_geometry = load_tissue_geometry(args.tissue_mask_dir, sample_name)
        raw_heatmap = heatmap_payload(
            sample_name, mode, sample.coords, scores, sample.patch,
            layer="raw_heatmap", origin_level0=origin,
            source_stride=sample.source_stride, tissue_geometry=tissue_geometry,
        )
        yy, xx = np.nonzero(seen)
        cell_coords = np.column_stack((origin[0] + xx * sample.source_stride,
                                       origin[1] + yy * sample.source_stride))
        refined_heatmap = heatmap_payload(
            sample_name, mode, cell_coords, refined[yy, xx], sample.source_stride,
            layer="refined_heatmap", origin_level0=origin,
            source_stride=sample.source_stride, tissue_geometry=tissue_geometry,
        )
        mask_gt = contour_payload(
            sample_name, contours["cortex"], contours["medullary"], args.contour_alpha,
            source_status=source_status, tissue_geometry=tissue_geometry,
            layer="mask_gt",
        )
        mask_prediction = contour_payload(
            sample_name,
            threshold_grid_geometry(raw, seen, origin, sample.source_stride,
                                    tissue_geometry, medullary=False),
            threshold_grid_geometry(raw, seen, origin, sample.source_stride,
                                    tissue_geometry, medullary=True),
            args.contour_alpha,
            source_status={"cortex": "derived", "medullary": "derived"},
            tissue_geometry=tissue_geometry, layer="mask_prediction",
        )
        has_gt_sources = any(status != "missing" for status in source_status.values())
        save_sample_exports(output_dir, sample_name, refined_heatmap,
                            mask_prediction, mask_gt if has_gt_sources else None,
                            raw_heatmap=raw_heatmap)
        if args.visualize:
            save_visualizations(row.h5_path, stem, origin, sample.source_stride,
                                raw, refined, seen, visualization_dir, args.annotation_dir,
                                tissue_geometry=tissue_geometry,
                                mask_alpha=args.contour_alpha)
        json_count = 4 if has_gt_sources else 3
        output_types = (f"{json_count} JSON files and 4 PNG overlays"
                        if args.visualize else f"{json_count} JSON files")
        print(f"Saved {output_types} for {sample_name} ({int(seen.sum()):,} cells)", flush=True)


if __name__ == "__main__":
    main()
