FROM python:3.11.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends openssh-client=1:9.2p1-2+deb12u10 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir girder-client==5.0.20

WORKDIR /opt/cort-med/plugin
COPY plugin/entrypoint.py plugin/selection.py plugin/results.py plugin/CorticalMedullaryIdentification.xml ./
RUN useradd --uid 1000 --create-home worker && chown -R worker:worker /opt/cort-med
USER worker
ENTRYPOINT ["python", "/opt/cort-med/plugin/entrypoint.py"]
