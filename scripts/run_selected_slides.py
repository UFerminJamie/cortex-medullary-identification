"""HiPerGator job body: preprocess every slide, then infer each successful slide."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

GEOMETRY = "40x_512px_384px_overlap"


def write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_manifest(manifest: dict) -> None:
    slides = manifest.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("Manifest needs at least one slide")
    if not str(manifest.get("model_path", "")).endswith(".pkl"):
        raise ValueError("A shared binary SVC .pkl model is required")
    item_ids, paths = set(), set()
    for slide in slides:
        for key in ("item_id", "folder_id", "slide_path", "job_dir"):
            if not slide.get(key):
                raise ValueError(f"Slide is missing {key}")
        if slide["item_id"] in item_ids or slide["slide_path"] in paths:
            raise ValueError("Duplicate slide item or path")
        item_ids.add(slide["item_id"])
        paths.add(slide["slide_path"])
        if Path(slide["slide_path"]).suffix.lower() not in (".svs", ".tif", ".tiff", ".ndpi"):
            raise ValueError(f"Unsupported slide type: {slide['slide_path']}")
        if not Path(slide["job_dir"]).is_absolute():
            raise ValueError("HiPerGator job_dir must be absolute")


def run(manifest: dict, project_dir: Path, status_path: Path,
        execute=subprocess.run) -> dict:
    from .h5_svm import load_model, safe_name
    validate_manifest(manifest)
    model = Path(manifest["model_path"])
    statuses = {slide["item_id"]: {"state": "queued", "name": Path(slide["slide_path"]).name}
                for slide in manifest["slides"]}
    result = {"slides": statuses, "phase": "preprocessing"}
    write_status(status_path, result)
    try:
        if not model.is_file():
            raise FileNotFoundError(f"Model checkpoint missing: {model}")
        checkpoint = load_model(model)
        metadata = checkpoint["metadata"]
        if tuple(metadata.get("class_names", ())) != ("cortex", "medullary") or metadata.get("pca") is not False:
            raise ValueError("Checkpoint must be the repository's binary cortex/medullary SVC without PCA")
        if not isinstance(metadata.get("feature_dimension"), int) or metadata["feature_dimension"] <= 0:
            raise ValueError("Checkpoint has no valid feature dimension")
        if metadata.get("tile_mode", "all") != "all":
            raise ValueError("Managed inference requires a checkpoint trained with all overlapping tiles")
    except Exception as error:
        result.update(phase="complete", state="error")
        for item in statuses.values():
            item.update(state="error", phase="checkpoint", error=str(error))
        write_status(status_path, result)
        return result
    for slide in manifest["slides"]:
        item = statuses[slide["item_id"]]
        item["state"] = "preprocessing"
        write_status(status_path, result)
        try:
            execute(["bash", str(project_dir / "run_single_slide.sh"), slide["slide_path"],
                     slide["job_dir"]], check=True)
            stem = Path(slide["slide_path"]).stem
            feature = Path(slide["job_dir"]) / GEOMETRY / "features_uni_v1" / f"{stem}.h5"
            if not feature.is_file():
                raise FileNotFoundError(f"TRIDENT feature file missing: {feature}")
            item["state"] = "preprocessed"
        except (subprocess.CalledProcessError, OSError) as error:
            item.update(state="error", phase="preprocessing", error=str(error))
        write_status(status_path, result)
    result["phase"] = "inference"
    write_status(status_path, result)
    for slide in manifest["slides"]:
        item = statuses[slide["item_id"]]
        if item["state"] != "preprocessed":
            continue
        stem = Path(slide["slide_path"]).stem
        feature = Path(slide["job_dir"]) / GEOMETRY / "features_uni_v1" / f"{stem}.h5"
        output = Path(slide["job_dir"]) / "inference_jsons"
        item["state"] = "inference"
        write_status(status_path, result)
        try:
            execute([sys.executable, "-m", "scripts.inference", "--h5-file", str(feature),
                     "--model", str(model), "--sample-name", stem, "--output-dir", str(output),
                     "--flat-output", "--no-visualize", "--tissue-mask-dir",
                     str(Path(slide["job_dir"]) / "contours_geojson"), "--annotation-dir",
                     str(Path(slide["job_dir"]) / "annotations"),
                     "--gaussian-sigma", str(manifest.get("gaussian_sigma", 1.0))],
                    cwd=project_dir, check=True)
            refined = output / f"{safe_name(stem)}_refined_heatmap.json"
            if not refined.is_file():
                raise FileNotFoundError(f"Refined heatmap missing: {refined}")
            item.update(state="complete", refined_heatmap=str(refined))
        except (subprocess.CalledProcessError, OSError) as error:
            item.update(state="error", phase="inference", error=str(error))
        write_status(status_path, result)
    result["phase"] = "complete"
    result["state"] = "error" if any(item["state"] == "error" for item in statuses.values()) else "complete"
    write_status(status_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    status = run(manifest, Path(os.environ.get("PROJECT_DIR", Path.cwd())),
                 args.manifest.with_name("status.json"))
    return 0 if status["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
