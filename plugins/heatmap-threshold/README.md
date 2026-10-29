# Cortex–Medulla Threshold

A Girder/HistomicsUI plugin for interactively thresholding cortex and medulla heatmaps. It adds **Analyses → cortex-medulla-threshold** to the slide viewer. Thresholding runs in a browser Web Worker; it does not submit a Slicer CLI job.

Saving creates merged cortex and medulla contour annotations with threshold and source-heatmap provenance. It does not create raster masks or change the source heatmaps.

## Install into DSA with `provision.yaml`

This package is a Girder plugin. Its `setup.py` registers the `threshold_mask` Girder entry point, and `MANIFEST.in` includes the browser client files. The worker and its `polygon-clipping` dependency are installed as package data in DSA's static directory.

Add these top-level settings to the DSA instance's `provision.yaml`:

```yaml
pip:
  - "--upgrade git+https://github.com/UFerminJamie/cortex-medullary-identification.git@dsa_plugin/heatmap-threshold#subdirectory=plugins/heatmap-threshold"
rebuild-client: True
```

Keep the existing `dsarchive/dsa_common` image in the Compose file. Recreate the Girder service so DSA installs the package and builds the client:

```sh
DSA_USER="$(id -u):$(id -g)" docker compose up -d --force-recreate girder
```

The public GitHub repository must be reachable from the Girder container. Pin the URL to a commit SHA instead of the branch when you need a repeatable deployment.

## Package layout

- `setup.py` declares Python dependencies and the `girder.plugin` entry point.
- `girder_threshold_mask/__init__.py` defines the plugin and points Girder at `web_client/`.
- `girder_threshold_mask/web_client/` contains the HistomicsUI panel, worker, and the vendored `polygon-clipping` browser asset.
- `MANIFEST.in` includes the client files in the pip distribution.
- `setup.py` installs the worker assets under `share/girder/static/built/plugins/threshold_mask`, matching the URL used by the client.

The Dockerfile in this folder is an optional way to build a derived DSA image. It is not needed for the `provision.yaml` installation above.
