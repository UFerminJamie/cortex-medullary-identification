# CorticalMedullaryIdentification for DSA

This Slicer CLI Web image runs kidney tissue segmentation, UNI-v1 tile embedding, cuML SVC classification, and cortical/medullary heatmap generation inside the DSA worker container. It does not connect to HiPerGator. The repository's separate training workflow still uses HiPerGator.

## Build

The image is Linux `amd64` and starts from a slim Python base, then builds a Conda environment at `/opt/conda/envs/cort_med_gpu` using the Python 3.10, CUDA 12.9 PyTorch, pinned cuML/CuPy, SVC requirements, and TRIDENT setup used by this repository's HiPerGator `cort_med_gpu` environment. This avoids adding a separate `nvidia/cuda` base image; CUDA runtime libraries required by PyTorch and cuML are installed with the Python packages. The HiPerGator path (`/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu`) is host-only and is not copied into Docker. The image contains only the UNI-v1 patch encoder weights and HEST tissue segmentation checkpoint used by this workflow. Both Hugging Face repositories require applicable access approval. Provide a Hugging Face token as a BuildKit secret; the token is not copied into the image. The build fails if either checkpoint cannot be downloaded. Do not put a token in a build argument or the repository.

```bash
docker build --platform linux/amd64 --secret id=hf_token,src=hf_token \
  -f plugin/Dockerfile -t cortex-medulla-identification:0.5.0 .
```

The Conda environment and model checkpoints still require substantial disk space. The image's UNI and HEST checkpoints are inside `/opt/models` and registered in TRIDENT's `local_ckpts.json`; no foundation-model path is a CLI argument. Only the trained binary SVC `.pkl` checkpoint is selected in the panel. Review the upstream model terms before distributing an image that contains the weights.

## DSA worker requirements

The Docker worker host needs an NVIDIA GPU and NVIDIA Container Toolkit. After building the plugin image, confirm the worker can expose its GPU with `docker run --rm --gpus all --entrypoint nvidia-smi cortex-medulla-identification:0.5.0`. This uses the plugin image instead of pulling a separate CUDA base image for the check. The image label signals Girder Worker to use the NVIDIA runtime. Its CLI metadata requests **one GPU, 8 CPUs, 64 GiB memory, and 8 GiB shared memory**. These Docker limits do not reserve resources on a cluster or add a GPU to a CPU-only host. Route this CLI to a GPU-capable Girder Worker and size its concurrent job capacity for these limits. The image requires a compatible NVIDIA driver for CUDA 12.9.

On the Docker daemon host, create `/etc/cort-med-worker/config.json` based on `config.example.json` and `/etc/cort-med-worker/girder-api-key`. Make the directory and files readable to UID 1000. The CLI mounts it read-only at `/run/cort-med`. Set an HTTPS Girder API URL, allowed slide folder IDs, and approved SHA-256 digests for the SVC `.pkl` files. The digest allowlist is mandatory because loading pickle can execute code. The Girder service account needs read access to slides and checkpoints and annotation/write access to the approved slide folders. Restrict who can launch the imported CLI in DSA.

Import the built image under an immutable tag through Slicer CLI Web's Docker image admin page. Verify the Athena deployment's Girder Worker and `slicer_cli_web` versions support `docker-params` for volumes, device requests, CPU, memory, and shared memory. This checkout cannot inspect Athena's installed versions. Use `docker run --rm cortex-medulla-identification:0.5.0 --list_cli` and `docker run --rm cortex-medulla-identification:0.5.0 CorticalMedullaryIdentification --xml` to inspect the CLI before import; these commands do not require a GPU.

## Panel and results

The IO panel has one **Model Checkpoint** file and a multiple-resource **Input Image** field. HistomicsUI can populate the input from the open slide; select additional slides to process them together in one DSA job. The image parameter accepts a comma-separated list of Girder file IDs, which the runner validates and maps back to their source items. The advanced **Tile Extraction** controls set magnification, patch size, and overlap in pixels. Defaults are 40×, 512 × 512 pixels, and 384 pixels of overlap (75% of the patch width). Overlap is an absolute pixel count, not a percentage; it must be smaller than patch size.

Each job downloads the selected files into a temporary container directory, verifies the checkpoint SHA-256, and processes all slides in three ordered phases: HEST tissue segmentation, patch coordinates and UNI-v1 embeddings for every slide; SVC inference for every successfully preprocessed slide; then attach each slide's `*_refined_heatmap.json` as native annotations on its matching source item. The run log records progress across these phases. After inference has been attempted for every slide, the runner removes its TRIDENT segmentation, coordinate, tile, and feature directories. It uploads only the refined JSON, status, and manifest for successful slides. The temporary source slides, model copy, and local results are removed when the container exits. A forced container termination also discards its writable layer.

The output item is created under the source slide's DSA folder in `40x_512px_384px_overlap`. Existing annotations are replaced only when their name and plugin ownership markers match the same slide; unrelated annotations remain. A failed slide is recorded in the job log and status; successful slides in a mixed job are still published.

## Verification still needed on Athena

Run a real single-slide and batch submission on a GPU worker. Confirm the worker honors the GPU and resource metadata, the Girder mount and HTTPS API are reachable, the selected checkpoint digest and metadata match, the heatmaps display in HistomicsUI, and a completed or canceled job leaves no preprocessing files on the worker. No GPU inference or Athena integration test was run on this development machine.
