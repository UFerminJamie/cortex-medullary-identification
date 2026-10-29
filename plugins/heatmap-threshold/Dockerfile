ARG DSA_BASE_IMAGE=dsarchive/dsa_common
FROM ${DSA_BASE_IMAGE}

COPY . /opt/threshold_mask_plugin
RUN pip install --no-cache-dir -e /opt/threshold_mask_plugin && \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false \
    NPM_CONFIG_LOGLEVEL=error \
    NPM_CONFIG_PROGRESS=false \
    NPM_CONFIG_PREFER_OFFLINE=true \
    girder build && \
    npm install --prefix /tmp/threshold-v1-worker polygon-clipping@0.15.7 --no-audit --no-fund && \
    mkdir -p /opt/venv/share/girder/static/built/plugins/threshold_mask && \
    cp /opt/threshold_mask_plugin/girder_threshold_mask/web_client/contour-worker.js /opt/venv/share/girder/static/built/plugins/threshold_mask/contour-worker.js && \
    cp /tmp/threshold-v1-worker/node_modules/polygon-clipping/dist/polygon-clipping.umd.min.js /opt/venv/share/girder/static/built/plugins/threshold_mask/polygon-clipping.umd.min.js && \
    find /opt -xdev -name node_modules -type d -prune -exec rm -rf {} + && \
    find /opt -name package-lock.json -type f -delete

WORKDIR /opt/digital_slide_archive
