"""Search GPU SVC kernels/C values on labeled H5 tiles without PCA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .h5_svm import (CS, FEATURE_ROOT, KPMP, c_name, decision_scores, fit_svc,
                        load_split, metric_row, metrics, save_cm, save_json,
                        timestamp, workbook_rows)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", type=Path, default=Path("dataset_split.xlsx"))
    p.add_argument("--h5-dir", type=Path, default=FEATURE_ROOT,
                   help="Trident job folder or one features_* folder")
    p.add_argument("--tile-mode", choices=("all", "no-overlap"), default="all")
    p.add_argument("--include-val", action="store_true", help="Refit each candidate on train+val after validation scoring")
    p.add_argument("--output-root", type=Path, default=KPMP / "grid_search_results")
    p.add_argument("--cache-mb", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=65536)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.cache_mb <= 0:
        raise ValueError("--cache-mb must be positive")
    output = args.output_root / timestamp()
    output.mkdir(parents=True, exist_ok=False)
    print(f"Tile mode: {args.tile_mode}; output: {output}", flush=True)
    train_x, train_y = load_split(workbook_rows(args.xlsx, args.h5_dir, "train"), args.tile_mode)
    val_x, val_y = load_split(workbook_rows(args.xlsx, args.h5_dir, "val"), args.tile_mode)
    test_x, test_y = load_split(workbook_rows(args.xlsx, args.h5_dir, "test"), args.tile_mode)
    if len({train_x.shape[1], val_x.shape[1], test_x.shape[1]}) != 1:
        raise ValueError("Train, val, and test feature dimensions differ")
    if args.include_val:
        fit_x = np.concatenate((train_x, val_x))
        fit_y = np.concatenate((train_y, val_y))
    else:
        fit_x, fit_y = train_x, train_y
    records = []
    for kernel in ("linear", "rbf"):
    # for kernel in ["rbf"]:
        for c in CS:
            name = f"svc_{c_name(c)}_{kernel}"
            print(f"Fitting {name} (validation fit)", flush=True)
            model = fit_svc(train_x, train_y, c, kernel, args.cache_mb)
            val_result = metrics(val_y, decision_scores(model, val_x, args.batch_size))
            if args.include_val:
                del model
                print(f"Refitting {name} on train+val", flush=True)
                model = fit_svc(fit_x, fit_y, c, kernel, args.cache_mb)
            train_result = metrics(fit_y, decision_scores(model, fit_x, args.batch_size))
            test_result = metrics(test_y, decision_scores(model, test_x, args.batch_size))
            for split, result in (("train", train_result), ("val", val_result), ("test", test_result)):
                save_cm(output / "confusion_matrices" / f"{name}_{split}.png", result, f"{name} {split}")
            records.append({"C": c, "kernel": kernel, "tile_mode": args.tile_mode,
                            "fitted_splits": "train+val" if args.include_val else "train",
                            "validation_fitted_splits": "train",
                            **metric_row("train", train_result), **metric_row("val", val_result),
                            **metric_row("test", test_result)})
            print(f"{name}: val balanced accuracy={val_result['balanced_accuracy']:.4f}; "
                  f"test balanced accuracy={test_result['balanced_accuracy']:.4f}", flush=True)
            del model
    result = pd.DataFrame(records).sort_values(["val_balanced_accuracy", "val_mcc"], ascending=False)
    result.to_excel(output / "grid_search_results.xlsx", index=False)
    best = result.iloc[0]
    save_json(output / "best_hyperparameters.json", {"C": float(best.C), "kernel": str(best.kernel),
              "selection": "highest validation balanced accuracy, then validation MCC",
              "tile_mode": args.tile_mode, "include_val": args.include_val,
              "xlsx": str(args.xlsx.resolve()), "h5_dir": str(args.h5_dir.resolve())})
    print(f"Best validation model: C={best.C:g}, kernel={best.kernel}; results: {output}", flush=True)


if __name__ == "__main__":
    main()
