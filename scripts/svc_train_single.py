"""Train one standardized GPU SVC from labeled Trident H5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .h5_svm import (FEATURE_ROOT, KPMP, c_name, decision_scores, fit_svc,
                        load_split, metric_row, metrics, save_cm, save_model,
                        workbook_rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", type=Path, default=Path("dataset_split.xlsx"))
    p.add_argument("--h5-dir", type=Path, default=FEATURE_ROOT)
    p.add_argument("--C", type=float, required=True)
    p.add_argument("--kernel", choices=("linear", "rbf"), required=True)
    p.add_argument("--tile-mode", choices=("all", "no-overlap"), default="all")
    p.add_argument("--include-val", action="store_true")
    p.add_argument("--model-dir", type=Path, default=KPMP / "output/model_weights")
    p.add_argument("--cache-mb", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=65536)
    args = p.parse_args()
    if args.C <= 0 or args.cache_mb <= 0:
        p.error("--C and --cache-mb must be positive")
    print(f"Tile mode: {args.tile_mode}", flush=True)
    train_rows = workbook_rows(args.xlsx, args.h5_dir, "train")
    if args.include_val:
        import pandas as pd
        train_rows = pd.concat((train_rows, workbook_rows(args.xlsx, args.h5_dir, "val")), ignore_index=True)
    train_x, train_y = load_split(train_rows, args.tile_mode)
    test_x, test_y = load_split(workbook_rows(args.xlsx, args.h5_dir, "test"), args.tile_mode)
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training and test feature dimensions differ")
    model = fit_svc(train_x, train_y, args.C, args.kernel, args.cache_mb)
    model["metadata"].update({"tile_mode": args.tile_mode, "fitted_splits": ["train", "val"] if args.include_val else ["train"],
                               "xlsx": str(args.xlsx.resolve()), "h5_dir": str(args.h5_dir.resolve())})
    name = f"svc_{c_name(args.C)}_{args.kernel}"
    path = args.model_dir / f"{name}.pkl"
    save_model(path, model)
    train = metrics(train_y, decision_scores(model, train_x, args.batch_size))
    test = metrics(test_y, decision_scores(model, test_x, args.batch_size))
    save_cm(args.model_dir / f"{name}_train_confusion_matrix.png", train, f"{name} train")
    save_cm(args.model_dir / f"{name}_test_confusion_matrix.png", test, f"{name} test")
    import pandas as pd
    pd.DataFrame([{"C": args.C, "kernel": args.kernel, "tile_mode": args.tile_mode,
                   **metric_row("train", train), **metric_row("test", test)}]).to_excel(
        args.model_dir / f"{name}_metrics.xlsx", index=False)
    print(f"Saved {path}; test balanced accuracy={test['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
