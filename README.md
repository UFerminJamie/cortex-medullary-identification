# Cortex and medullary WSI classification

This project uses RAPIDS cuML, accelerated by CuPy, to train a GPU-based support vector machine (SVM) that classifies kidney whole-slide image tiles as cortical or medullary. It maps tile-level predictions back to slide coordinates to generate spatial heatmaps of cortical and medullary tissue.

This workflow uses [TRIDENT](https://github.com/mahmoodlab/TRIDENT) to segment tissue and extract tile embeddings, labels tiles from annotations, creates slide-level train/validation/test splits, and trains a GPU SVC. Run commands from this repository on HiPerGator.

## Installation


```bash
git clone https://github.com/UFerminJamie/cortex-medullary-identification.git
cd cortex-medullary-identification
```

```bash
git clone https://github.com/mahmoodlab/TRIDENT.git TRIDENT
```

```bash
sbatch setup_cort_med_env.sh
```

```bash
module load conda
```

```bash
conda activate /blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu
```

The setup job creates `cort_med_gpu` directly with Python 3.10, CUDA 12.9 PyTorch, the pinned cuML/CuPy and SVC dependencies in `requirements-svc.txt`, and the local `TRIDENT/` checkout. It does not use `rapids_svc`. The launchers use this environment by default; `CORT_MED_CONDA_ENV` overrides its path, including during setup. Submit the setup job on a GPU node to run its final import and device check.

## Process WSIs with TRIDENT

Set these paths for your data. `JOB_DIR` is reused by every later stage.

```bash
WSI_DIR=/path/to/wsis
```

```bash
JOB_DIR=/path/to/trident_processed
```

```bash
H5_DIR="$JOB_DIR/40x_512px_384px_overlap"
```

```bash
sbatch run_batch_slide.sh "$WSI_DIR" "$JOB_DIR"
```

The launcher runs `seg → coords → feat`, searches nested `.svs` folders, and defaults to HEST, UNI-v1, 40×, 512-pixel patches, and 384-pixel overlap. UNI-v1 requires access to its gated Hugging Face weights. Completed slide outputs in the same `JOB_DIR` are skipped on rerun. See the [TRIDENT README](https://github.com/mahmoodlab/TRIDENT) for model access and other processing options.

Contour GeoJSON filenames replace semicolons with underscores before they are
written because GDAL/Fiona treats semicolons as path separators. The launcher
also normalizes legacy filenames after TRIDENT finishes.

To make existing `contours_geojson` filenames exactly match the corresponding
`contours` filenames except for the extension, run:

```bash
python -m scripts.match_contour_geojson_names "$JOB_DIR"
```

Use `--dry-run` before the job directory to preview planned renames. This can
restore semicolons in the GeoJSON filenames to match the contour filenames.

Useful flags after the two paths:

- `--task seg`, `--task coords`, or `--task feat`: run one stage instead of `all`.
- `--segmenter otsu`: use Otsu tissue segmentation instead of HEST.
- `--patch_encoder uni_v1`: choose the patch encoder; keep the same encoder for all SVC samples.
- `--mag`, `--patch_size`, `--overlap`: change tile geometry; update `H5_DIR` to match.

## Label tiles

Create one annotation folder per WSI stem, containing `cortical_gt.json` and/or `medullary_gt.json` with level-zero polygons. The labels are `0` unlabeled, `1` cortex, and `2` medullary.

```bash
ANNOTATION_DIR=/path/to/annotations
```

```bash
sbatch batch_labelling.sh "$ANNOTATION_DIR" "$JOB_DIR"
```

The job adds `labels` to matching `features_uni_v1/*.h5` files and writes `batch_labelling_report.csv`. Review that report before splitting. The launcher also saves GT thumbnail overlays by default.

For optional H5 and tile-label checks, run:

```bash
python -m scripts.inspect_h5 "$H5_DIR/features_uni_v1/SAMPLE.h5"
```

```bash
python -m scripts.visualize_tile_labels "$H5_DIR/features_uni_v1/SAMPLE.h5" --overlap 0
```

## Create train, validation, and test splits

Build a real Excel workbook named `dataset_split.xlsx` from the labeled feature H5 files. Keep it in `JOB_DIR` so an existing workbook in this repository is not replaced:

```bash
SPLIT_XLSX="$JOB_DIR/dataset_split.xlsx"
```

```bash
python -m scripts.split_h5 --job-dir "$JOB_DIR" --features-dir "$H5_DIR/features_uni_v1" --output "$SPLIT_XLSX"
```

The generator assigns approximately 70%/15%/15% of labeled WSIs to `train`, `val`, and `test` with seed 42. Review and edit the workbook so slides from the same patient stay in one split. Save it as an actual `.xlsx` file, not CSV text renamed to `.xlsx`.

Workbook columns:

- `sample_name` **(required)**: unique WSI stem or filename whose stem matches a feature H5 file.
- `split` **(required)**: exactly `train`, `val`, or `test` for this workflow.
- `feature_path`, `cortex_tiles`, `medullary_tiles` **(generated, optional)**: audit columns; the H5 SVC loader finds features under `--h5-dir`.
- `folder_path` **(optional)**: original WSI location; ignored by the H5 SVC loader.

Flags:

- `--features-dir`: select one `features_*` folder if the TRIDENT job has multiple encoders or tile geometries.
- `--seed`: change the reproducible slide shuffle; default is `42`.
- `--output`: write `.xlsx` for the commands below, or `.csv` for other tools.

To build `dataset_all_split.xlsx` from all available KPMP WSIs with matching
`features_uni_v1/*.h5` files, run:

```bash
python -m scripts.dataset_all_split
```

Flags:

- `--wsi-dirs DIR [DIR ...]` — Use different WSI directories.
- `--h5-dir DIR`** — Use a different feature H5 folder. It must match the encoder and tile geometry used for inference.
- `--source FILE` — Use a different source workbook.
- `--output FILE` — Set the output workbook path.

To create a workbook inventory of WSI files under one directory, including
nested folders, run:

```bash
python -m scripts.create_dataset_workbook /path/to/wsis dataset_IU401_529.xlsx
```

The file extension defaults to `.svs`. Use `--extension` to select another
extension, with or without the leading dot (for example, `--extension tif`).
The output workbook lists `sample_name`, `folder_path`, `split`, and tile-count
columns; newly listed files start as `holdout` with zero tile counts. The script
refuses to overwrite an existing output workbook.

## GPU SVC grid search

Search linear and RBF kernels and several `C` values using training data for fitting and validation data for selection:

```bash
sbatch svc_grid_search.sh --xlsx "$SPLIT_XLSX" --h5-dir "$H5_DIR" --tile-mode all --output-root "$JOB_DIR/grid_search_results"
```

Results go to `JOB_DIR/grid_search_results/<timestamp>/`. Read `best_hyperparameters.json` for the chosen `C` and kernel. 
Flags:
- `--tile-mode all`: use every H5 tile; use `no-overlap` for a sparse, nonoverlapping grid.
- `--include-val`: also refit each candidate on train plus validation after its validation score is recorded.
- `--output-root`: choose where search results and confusion matrices are saved.

## Train and evaluate the final SVC

Use the `C` and kernel selected by the grid search, or choose them before looking at test results. The following example uses `C=2` and `rbf`:

```bash
sbatch svc_train_single.sh --xlsx "$SPLIT_XLSX" --h5-dir "$H5_DIR" --C 2 --kernel rbf --tile-mode all --include-val --model-dir "$JOB_DIR/output/model_weights"
```

The model fits on train plus validation, then evaluates once on test. It saves `svc_2_rbf.pkl`, a metrics workbook, and train/test confusion matrix PNGs under `--model-dir/svc_2_rbf/`. Use this model with `inference.sh` or `python -m scripts.inference`.

Flags:

- `--C` and `--kernel`: the selected SVC hyperparameters; kernel is `linear` or `rbf`.
- `--tile-mode`: use the same `all` or `no-overlap` choice as grid search.
- `--include-val`: add validation slides to the final fit after model selection.
- `--model-dir`: parent directory for a model-named folder containing the model and evaluation files.

## Predict and visualize

Use the saved H5 SVC model for a workbook split. This example visualizes the test slides:

```bash
sbatch inference.sh --xlsx "$SPLIT_XLSX" --h5-dir "$H5_DIR" --model "$JOB_DIR/output/model_weights/svc_2_rbf/svc_2_rbf.pkl" --split test --output-dir "$JOB_DIR/inference_jsons" --visualization-dir "$JOB_DIR/prediction_visualization"
```

### Inference outputs

Outputs are saved under `--output-dir/<model-stem>_<sigma>/` (for example, `svc_2_rbf_1.0/`). Filenames use the workbook sample stem.

Inference writes:

- `<sample>_raw_heatmap.json` — One signed SVC score per tile. Negative is cortical; zero or positive is medullary. Includes a `raw_heatmap` layer.
- `<sample>_refined_heatmap.json` — Gaussian-smoothed scores on the source grid.
- `<sample>_mask_prediction.json` — Cortex and medulla polygons thresholded at zero from the raw scores.
- `<sample>_mask_gt.json` — Ground-truth polygons, written when at least one matching GT source file exists.

The refined heatmap JSON contains two annotations named `cortical heatmap` and
`medullary heatmap`.

Cortical annotations and masks use RGB `(12, 123, 220)` (`#0C7BDC`); medullary
annotations and masks use RGB `(255, 194, 10)` (`#FFC20A`).

Flags:
- `--visualize` — generates PNGs to `$JOB_DIR/predictions_visualization`:
  - `<sample>_heatmap.png`
  - `<sample>_refined_heatmap.png`
  - `<sample>_mask_h.png` — Raw heatmap thresholded at zero.
  - `<sample>_mask_rh.png` — Smoothed heatmap thresholded at zero.
- `--tissue-mask-dir` — Tissue GeoJSON directory. Defaults to `$JOB_DIR/contours_geojson`.
- `--annotation-dir` — Directory containing GT JSON annotation files.
- `--contour-alpha` — Fill opacity for mask JSONs and PNG masks; default `0.35`.

### Refine a mask prediction

Fill polygon holes, keep each class's largest connected component, and remove
other components below the minimum area (default: 32,768 level-zero WSI pixels²).
The script writes a separate refined JSON and an original/refined comparison PNG:

```bash
python -m scripts.postprocess_mask_prediction \
  /path/to/<sample>_mask_prediction.json \
  --min-area 32768
```

Use `--output-json` and `--output-plot` to choose output paths. Component area is
measured in level-zero WSI pixel².

### Generate interstitial fractional space annotations

For each `<sample>.geojson`, subtract every matching FTU JSON whose filename
starts with `<sample>_` (for example, tubules, muscular vessels, glomeruli, and
capillaries). Submit the CPU job with the tissue GeoJSON directory, FTU JSON
directory, and output folder:

```bash
sbatch generate_IFS.sh \
  /orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed_final/KPMP/contours_geojson \
  /path/to/ftu_jsons \
  /path/to/ifs_jsons \
  --tolerance 1.0
```

This writes only `<sample>_IFS.json` as a HistomicsUI contour annotation named
`IFS`, using light green with 0.5 fill opacity. Existing output files are skipped
unless `--overwrite` is provided. Slides without matching FTU polygons are
logged and skipped.
Only exact stem matches between a GeoJSON and at least one FTU JSON are read
for generation; unmatched GeoJSONs and FTU files are logged and skipped.
`--tolerance` applies topology-preserving Douglas-Peucker simplification in
level-zero WSI pixels; it defaults to `1.0`. Use `--tolerance 0` to disable it.

## Transfer annotation JSON files from HiPerGator to Athena

Use the [HPG-ATHENA-Transfer](https://github.com/UFerminJamie/HPG-ATHENA-Transfer) tool to attach annotation JSON documents from HiPerGator to their matching slide items in Athena. It reads the JSON files on HiPerGator and sends their annotation contents through Athena’s Girder API.




