"""Run an H5 SVC model on a workbook split and write spatial prediction maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .h5_svm import (FEATURE_ROOT, KPMP, decision_scores, load_model, load_tiles,
                        safe_name, save_json, workbook_rows)


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


def map_payload(sample_name: str, mode: str, origin: np.ndarray, stride: int,
                scores: np.ndarray, seen: np.ndarray, *, mask: bool) -> dict:
    yy, xx = np.nonzero(seen)
    values = scores[yy, xx]
    labels = (values >= 0).astype(np.int8)
    if mask:
        cells = [[int(origin[0] + x * stride), int(origin[1] + y * stride), int(label)]
                 for x, y, label in zip(xx, yy, labels)]
    else:
        med = 1 / (1 + np.exp(-np.clip(values, -30, 30)))
        cells = [[int(origin[0] + x * stride), int(origin[1] + y * stride),
                  round(float(1 - probability), 6), round(float(probability), 6), int(label)]
                 for x, y, probability, label in zip(xx, yy, med, labels)]
    return {"sample_name": sample_name, "tile_mode": mode, "origin_level0": origin.tolist(),
            "cell_size_level0": stride, "classes": {"0": "cortex", "1": "medullary"},
            "colors": {"0": "#0000FF", "1": "#FF0000"}, "decision_threshold": 0,
            "columns": ["x", "y", "class_id"] if mask else
                       ["x", "y", "cortex_score", "medullary_score", "class_id"],
            "cells": cells}


def save_visualizations(h5_path: Path, stem: str, origin: np.ndarray, stride: int,
                        raw: np.ndarray, refined: np.ndarray, seen: np.ndarray,
                        output_dir: Path) -> None:
    """Overlay source-grid decisions on the full-slide TRIDENT thumbnail."""
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
    mask_cmap = ListedColormap(["#0000ff", "#ff0000"])
    mask_cmap.set_bad(alpha=0)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, values in (("heatmap", raw), ("refined_heatmap", refined),
                         ("mask", (refined >= 0).astype(np.int8))):
        fig, ax = plt.subplots(figsize=(12, 10))
        try:
            ax.imshow(thumbnail, extent=slide_extent, origin="upper")
            visible = np.ma.array(values, mask=~seen)
            if name == "mask":
                ax.imshow(visible, extent=grid_extent, origin="upper",
                          interpolation="nearest", cmap=mask_cmap,
                          norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), alpha=0.5)
                ax.legend(handles=[Patch(color="#0000ff", label="Cortex (< 0)"),
                                   Patch(color="#ff0000", label="Medullary (>= 0)")],
                          loc="lower right")
            else:
                overlay = ax.imshow(visible, extent=grid_extent, origin="upper",
                                    interpolation="nearest", cmap=score_cmap,
                                    norm=norm, alpha=0.6)
                fig.colorbar(overlay, ax=ax, label="Signed SVC decision score",
                             fraction=0.046, pad=0.04)
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
                   help="Save heatmap, refined heatmap, and transparent mask PNG overlays")
    p.add_argument("--visualization-dir", type=Path,
                   default=KPMP / "prediction_visualization")
    p.add_argument("--tile-mode", choices=("all", "no-overlap"), default="all",
                   help="Tiles to predict and visualize (default: all)")
    p.add_argument("--gaussian-sigma", type=float, default=1.0,
                   help="Gaussian sigma measured in source-grid cells")
    p.add_argument("--batch-size", type=int, default=32768)
    args = p.parse_args()
    if args.gaussian_sigma < 0:
        p.error("--gaussian-sigma must be nonnegative")
    model = load_model(args.model)
    expected_mode = model["metadata"].get("tile_mode")
    mode = args.tile_mode
    if expected_mode and mode != expected_mode:
        print(f"NOTE: using {mode} tiles for inference; model was trained with {expected_mode} tiles",
              flush=True)
    rows = workbook_rows(args.xlsx, args.h5_dir, args.split, skip_missing=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows.itertuples(index=False):
        sample = load_tiles(row.h5_path, mode, require_labels=False)
        assert sample is not None
        if sample.features.shape[1] != model["metadata"]["feature_dimension"]:
            raise ValueError(f"Feature dimension mismatch for {row.sample_name}")
        scores = decision_scores(model, sample.features, args.batch_size)
        origin, raw, refined, seen = aggregate(sample.coords, scores, sample.patch,
                                                sample.source_stride, args.gaussian_sigma)
        stem = safe_name(row.h5_name)
        save_json(args.output_dir / f"{stem}_heatmap.json",
                  map_payload(row.sample_name, mode, origin, sample.source_stride, raw, seen, mask=False))
        save_json(args.output_dir / f"{stem}_refined_heatmap.json",
                  map_payload(row.sample_name, mode, origin, sample.source_stride, refined, seen, mask=False))
        save_json(args.output_dir / f"{stem}_mask.json",
                  map_payload(row.sample_name, mode, origin, sample.source_stride, refined, seen, mask=True))
        if args.visualize:
            save_visualizations(row.h5_path, stem, origin, sample.source_stride,
                                raw, refined, seen, args.visualization_dir)
        output_types = "three JSON maps and three PNG overlays" if args.visualize else "three JSON maps"
        print(f"Saved {output_types} for {row.sample_name} ({int(seen.sum()):,} cells)", flush=True)


if __name__ == "__main__":
    main()
