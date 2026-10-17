"""Shared H5, GPU SVC, metrics, and spatial output helpers (no PCA)."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score

KPMP = Path("/orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed_final/KPMP")
FEATURE_ROOT = KPMP / "40x_512px_384px_overlap"
LABELS = ("cortex", "medullary")
CS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1, 2, 3, 4, 5)


def feature_index(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    folders = [root] if root.name.startswith("features_") else sorted(root.glob("features_*"))
    paths = [path for folder in folders for path in folder.glob("*.h5")]
    if not paths:
        raise FileNotFoundError(f"No H5 files under features_* in {root}")
    index: dict[str, Path] = {}
    for path in paths:
        if path.stem in index:
            raise ValueError(f"Duplicate sample {path.stem} in {index[path.stem]} and {path}; point to one features_* folder")
        index[path.stem] = path
    return index


def workbook_rows(xlsx: Path, root: Path, split: str | None = None, *,
                  skip_missing: bool = False) -> pd.DataFrame:
    frame = pd.read_excel(xlsx, dtype={"sample_name": str}, engine="openpyxl", keep_default_na=False)
    required = {"sample_name", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{xlsx} needs columns {sorted(required)}")
    if split is not None:
        frame = frame[frame.split.astype("string").str.strip().str.lower() == split.strip().lower()].copy()
    if frame.empty:
        raise ValueError(f"No samples for split {split!r} in {xlsx}")
    frame["h5_name"] = frame.sample_name.map(lambda value: Path(str(value)).stem)
    if frame.h5_name.duplicated().any():
        raise ValueError(f"Duplicate sample names: {frame.loc[frame.h5_name.duplicated(), 'h5_name'].tolist()}")
    index = feature_index(root)
    missing = sorted(set(frame.h5_name) - set(index))
    if missing:
        if not skip_missing:
            raise FileNotFoundError(f"{len(missing)} samples lack H5 features under {root}: {missing[:20]}")
        print(f"SKIP {len(missing)} samples without H5 features: {missing}", flush=True)
        frame = frame[frame.h5_name.isin(index)].copy()
        if frame.empty:
            raise ValueError(f"No H5 features remain for split {split!r}")
    frame["h5_path"] = frame.h5_name.map(index)
    return frame


def h5_geometry(handle: h5py.File) -> tuple[int, int]:
    attrs = handle["coords"].attrs
    patch = int(attrs.get("patch_size_level0", attrs.get("patch_size", 0)))
    source_patch = int(attrs.get("patch_size", patch))
    overlap = int(attrs.get("overlap", 0))
    if patch <= 0 or source_patch <= 0 or not 0 <= overlap < source_patch:
        raise ValueError(f"Invalid H5 tile geometry in {handle.filename}")
    stride = (source_patch - overlap) * patch / source_patch
    if not np.isclose(stride, round(stride)):
        raise ValueError(f"Non-integral level-zero stride in {handle.filename}")
    return patch, round(stride)


def tile_selection(coords: np.ndarray, patch: int, source_stride: int, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones(len(coords), dtype=bool)
    if mode != "no-overlap":
        raise ValueError(f"Unknown tile mode: {mode}")
    if patch % source_stride:
        raise ValueError(f"Cannot select exact no-overlap grid: patch={patch}, source stride={source_stride}")
    origin = coords.min(axis=0)
    if np.any((coords - origin) % source_stride):
        raise ValueError("H5 coordinates are not on one regular source grid")
    return np.all((coords - origin) % patch == 0, axis=1)


@dataclass
class TileData:
    features: np.ndarray
    labels: np.ndarray | None
    coords: np.ndarray
    patch: int
    source_stride: int
    total: int


def load_tiles(path: Path, mode: str, *, require_labels: bool) -> TileData | None:
    with h5py.File(path, "r") as handle:
        if "features" not in handle or "coords" not in handle:
            raise ValueError(f"Missing features or coords in {path}")
        coords = np.asarray(handle["coords"][:], dtype=np.int64)
        features = handle["features"]
        if features.ndim != 2 or coords.shape != (len(features), 2):
            raise ValueError(f"Invalid feature/coordinate shapes in {path}")
        patch, stride = h5_geometry(handle)
        selected = tile_selection(coords, patch, stride, mode)
        labels = None
        if "labels" in handle:
            raw = np.asarray(handle["labels"][:])
            if raw.shape != (len(features),) or not np.isin(raw, (0, 1, 2)).all():
                raise ValueError(f"Invalid labels in {path}")
            if require_labels:
                selected &= np.isin(raw, (1, 2))
                labels = raw[selected].astype(np.int8) - 1
        elif require_labels:
            print(f"SKIP {path.stem}: H5 has no labels dataset", flush=True)
            return None
        indices = np.flatnonzero(selected)
        if not len(indices):
            if require_labels:
                print(f"SKIP {path.stem}: no labeled tiles after {mode} selection", flush=True)
                return None
            raise ValueError(f"No tiles selected from {path}")
        data = np.asarray(features[indices], dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite features in {path}")
    print(f"{mode} | {path.stem}: used {len(indices):,} / {len(coords):,} total H5 tiles"
          + (f" | cortex={int((labels == 0).sum()):,} medullary={int((labels == 1).sum()):,}" if labels is not None else ""), flush=True)
    return TileData(data, labels, coords[indices], patch, stride, len(coords))


def load_split(rows: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    arrays, labels = [], []
    for row in rows.itertuples(index=False):
        sample = load_tiles(row.h5_path, mode, require_labels=True)
        if sample is not None:
            arrays.append(sample.features)
            labels.append(sample.labels)
    if not arrays:
        raise ValueError("No labeled tiles in the selected split")
    dimensions = {array.shape[1] for array in arrays}
    if len(dimensions) != 1:
        raise ValueError(f"Mixed feature dimensions: {dimensions}")
    return np.concatenate(arrays), np.concatenate(labels)


def gpu_modules():
    try:
        import cupy as cp
        from cuml.preprocessing import StandardScaler
        from cuml.svm import SVC
        if cp.cuda.runtime.getDeviceCount() != 1:
            raise RuntimeError("Exactly one visible GPU is required")
        return cp, StandardScaler, SVC
    except Exception as error:
        raise RuntimeError("Use one GPU compute node with the project RAPIDS/CuPy environment") from error


def fit_svc(x: np.ndarray, y: np.ndarray, c: float, kernel: str, cache_mb: int):
    cp, StandardScaler, SVC = gpu_modules()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training needs both cortex and medullary labels")
    scaler = StandardScaler(output_type="cupy")
    scaled = scaler.fit_transform(cp.asarray(x, dtype=cp.float32))
    svc = SVC(C=c, kernel=kernel, gamma="scale", class_weight="balanced",
              cache_size=cache_mb, output_type="cupy")
    svc.fit(scaled, cp.asarray(y, dtype=cp.int32))
    del scaled
    cp.get_default_memory_pool().free_all_blocks()
    return {"scaler": scaler, "svc": svc, "metadata": {"C": c, "kernel": kernel,
            "feature_dimension": x.shape[1], "class_names": LABELS, "pca": False}}


def decision_scores(model: dict, x: np.ndarray, batch_size: int) -> np.ndarray:
    cp, _, _ = gpu_modules()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    scores = []
    for start in range(0, len(x), batch_size):
        batch = cp.asarray(x[start:start + batch_size], dtype=cp.float32)
        scaled = model["scaler"].transform(batch)
        scores.append(cp.asnumpy(model["svc"].decision_function(scaled)).reshape(-1))
    return np.concatenate(scores).astype(np.float32)


def metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= 0).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=(0, 1))
    result = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
              "mcc": matthews_corrcoef(y, pred), "auc": roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan,
              "tiles": len(y), "confusion_matrix": cm.tolist()}
    for i, name in enumerate(LABELS):
        tp, fn, fp, tn = cm[i, i], cm[i, 1-i], cm[1-i, i], cm[1-i, 1-i]
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        result[f"{name}_accuracy"] = (tp + tn) / cm.sum()
        result[f"{name}_balanced_accuracy"] = (sensitivity + specificity) / 2
        result[f"{name}_sensitivity"] = sensitivity
        result[f"{name}_specificity"] = specificity
        result[f"{name}_auc"] = roc_auc_score(y == i, scores if i == 1 else -scores) if len(np.unique(y)) == 2 else np.nan
    return result


def metric_row(prefix: str, result: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in result.items() if key != "confusion_matrix"}


def save_cm(path: Path, result: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(np.asarray(result["confusion_matrix"]), display_labels=LABELS).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_model(path: Path, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream, pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> dict:
    gpu_modules()
    with path.open("rb") as stream:
        model = pickle.load(stream)
    if not {"scaler", "svc", "metadata"}.issubset(model):
        raise ValueError(f"Invalid SVC model: {path}")
    return model


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def c_name(c: float) -> str:
    return f"{c:g}"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, separators=(",", ":"))
