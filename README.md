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

## Optional GPU SVC grid search

Search linear and RBF kernels and several `C` values using training data for fitting and validation data for selection:

```bash
sbatch svc_grid_search.sh --xlsx "$SPLIT_XLSX" --h5-dir "$H5_DIR" --tile-mode all --output-root "$JOB_DIR/grid_search_results"
```

Results go to `JOB_DIR/grid_search_results/<timestamp>/`. Read `best_hyperparameters.json` for the chosen `C` and kernel. The test split must not guide that choice.

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

Inference writes raw heatmap, refined heatmap, and mask JSON files per sample under `--output-dir/svc_2_rbf_1.0/`, and thumbnail-overlay PNGs under `--visualization-dir/svc_2_rbf_1.0/` for the default Gaussian sigma of 1.0. The four first-pass PNGs are `<sample>_heatmap.png`, `<sample>_refined_heatmap.png`, `<sample>_aggregate.png`, and `<sample>_mask.png`; two additional PNGs show the second SVC's results. The subfolder name combines the `.pkl` filename stem and Gaussian sigma.

Heatmaps show signed decision scores. For each source-grid cell, the raw score is the maximum score among overlapping tiles, so any medullary prediction wins a cortex/medullary conflict. The aggregate PNG thresholds this unsmoothed score at zero; the translucent mask thresholds the refined score after Gaussian smoothing. During visualization, inference also fits a separate SVC for each slide using first-pass scores below `-0.4` as cortex and above `0.4` as medullary. It scores every tile again and saves `<sample>_refinement.png` (second-pass class map at zero) and `<sample>_heatmap_refinement.png` (second-pass signed score map). The second SVC uses the saved model's `C` and kernel. If either confident class is absent, these two PNGs are skipped for that slide. This is self-training on the model's own predictions, so incorrect first-pass labels may carry into the second pass. When matching annotation JSONs exist, all PNGs also show cortical contours in cyan and medullary contours in pink. Missing annotations leave the PNGs without GT contours.

Flags:

- `--split`: select a workbook split; use a separate holdout workbook for new slides.
- `--tile-mode all`: predict every H5 tile by default; `no-overlap` selects a sparse grid.
- `--gaussian-sigma`: smoothing width in source-grid cells; default is `1`.
- `--refinement-threshold`: positive and negative first-pass score cutoffs for second-SVC training; default is `0.4`.
- `--refinement-cache-mb`: GPU cache size for the per-slide SVC; default is `8192`.
- `--annotation-dir`: parent of per-slide annotation folders; defaults to `/orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed/annotations`.
- `--no-visualize`: skip PNG overlays and the per-slide refinement fit.
- `--output-dir` and `--visualization-dir`: choose parent directories for model-and-sigma-named JSON and PNG folders.
