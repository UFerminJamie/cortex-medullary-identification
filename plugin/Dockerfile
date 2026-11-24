# syntax=docker/dockerfile:1.7
FROM python:3.10-slim-bookworm AS dependencies

LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV DEBIAN_FRONTEND=noninteractive \
    CORT_MED_CONDA_ENV=/opt/conda/envs/cort_med_gpu \
    PATH=/opt/conda/envs/cort_med_gpu/bin:/opt/conda/bin:${PATH} \
    PROJECT_DIR=/opt/cort-med \
    TRIDENT_DIR=/opt/TRIDENT \
    CORT_MED_CONTAINER=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bash libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 \
    libxrender1 libvips42 openslide-tools \
    && rm -rf /var/lib/apt/lists/*

ARG MINIFORGE_VERSION=24.11.3-2
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
      -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda create -p /opt/conda/envs/cort_med_gpu -c conda-forge --override-channels python=3.10 pip -y \
    && conda clean --all -y

COPY requirements-svc.txt /opt/cort-med/requirements-svc.txt
RUN python -m pip install torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cu129 \
    && python -m pip install -r /opt/cort-med/requirements-svc.txt \
      --extra-index-url https://pypi.nvidia.com

ARG TRIDENT_COMMIT=e38c9b65abb0862599dc88ad519562840538d22d
RUN git clone https://github.com/mahmoodlab/TRIDENT.git /opt/TRIDENT \
    && git -C /opt/TRIDENT checkout "${TRIDENT_COMMIT}" \
    && python -m pip install -e /opt/TRIDENT girder-client==5.0.20 \
    && python -m pip check

FROM dependencies AS final
ENV PYTHONPATH=/opt/cort-med
# HEST loads a complete checkpoint; avoid fetching an unused TorchVision backbone.
RUN sed -i 's/deeplabv3_resnet50(weights=None)/deeplabv3_resnet50(weights=None, weights_backbone=None)/' \
    /opt/TRIDENT/trident/segmentation_models/load.py \
    && grep -q 'deeplabv3_resnet50(weights=None, weights_backbone=None)' \
       /opt/TRIDENT/trident/segmentation_models/load.py
COPY plugin/download_model_weights.py /opt/cort-med/plugin/download_model_weights.py
RUN --mount=type=secret,id=hf_token,required=true \
    HF_HUB_OFFLINE=0 python /opt/cort-med/plugin/download_model_weights.py

COPY scripts/ /opt/cort-med/scripts/
COPY run_single_slide.sh /opt/cort-med/run_single_slide.sh
COPY plugin/entrypoint.py plugin/selection.py plugin/results.py \
     plugin/CorticalMedullaryIdentification.xml /opt/cort-med/plugin/
RUN useradd --uid 1000 --create-home worker \
    && mkdir -p /opt/models \
    && chown -R worker:worker /opt/cort-med /opt/models \
    && chmod -R a+rX /opt/TRIDENT
WORKDIR /opt/cort-med
USER worker
RUN /opt/conda/envs/cort_med_gpu/bin/python /opt/cort-med/plugin/entrypoint.py --list_cli \
    && /opt/conda/envs/cort_med_gpu/bin/python /opt/cort-med/plugin/entrypoint.py \
       CorticalMedullaryIdentification --xml >/dev/null \
    && bash -n /opt/cort-med/run_single_slide.sh
ENTRYPOINT ["python", "/opt/cort-med/plugin/entrypoint.py"]
