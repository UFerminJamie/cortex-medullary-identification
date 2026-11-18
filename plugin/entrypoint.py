"""Slicer CLI Web adapter. Only staging and Slurm control run in this image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from uuid import uuid4

from selection import select_items, validate_checkpoints
from results import attach_refined_heatmap, upload_result

TASK = "CorticalMedullaryIdentification"
STOP = False


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).stem)


def on_stop(_signal, _frame):
    global STOP
    STOP = True


class SshSlurm:
    def __init__(self, settings):
        required = ("CORT_MED_SSH_HOST", "CORT_MED_SSH_USER", "CORT_MED_SSH_KEY_FILE",
                    "CORT_MED_SSH_KNOWN_HOSTS", "CORT_MED_REMOTE_PROJECT_DIR",
                    "CORT_MED_REMOTE_OUTPUT_ROOT", "CORT_MED_SLURM_ACCOUNT",
                    "CORT_MED_SLURM_QOS")
        missing = [key for key in required if not settings.get(key)]
        if missing:
            raise ValueError("Worker configuration missing: " + ", ".join(missing))
        for key in ("CORT_MED_SSH_HOST", "CORT_MED_SSH_USER", "CORT_MED_SLURM_ACCOUNT",
                    "CORT_MED_SLURM_QOS"):
            if not re.fullmatch(r"[A-Za-z0-9_.@-]+", settings[key]):
                raise ValueError(f"Invalid worker setting: {key}")
        self.host = f"{settings['CORT_MED_SSH_USER']}@{settings['CORT_MED_SSH_HOST']}"
        self.project = PurePosixPath(settings["CORT_MED_REMOTE_PROJECT_DIR"])
        self.output = PurePosixPath(settings["CORT_MED_REMOTE_OUTPUT_ROOT"])
        if not self.project.is_absolute() or not self.output.is_absolute():
            raise ValueError("HiPerGator project and output roots must be absolute")
        for path in (self.project, self.output):
            if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(path)):
                raise ValueError("HiPerGator path contains unsupported characters")
        for key in ("CORT_MED_SSH_KEY_FILE", "CORT_MED_SSH_KNOWN_HOSTS"):
            if not Path(settings[key]).is_file():
                raise ValueError(f"Worker secret mount unavailable: {key}")
        self.options = ["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                        "-o", "StrictHostKeyChecking=yes", "-o",
                        f"UserKnownHostsFile={settings['CORT_MED_SSH_KNOWN_HOSTS']}",
                        "-i", settings["CORT_MED_SSH_KEY_FILE"]]
        self.account = settings["CORT_MED_SLURM_ACCOUNT"]
        self.qos = settings["CORT_MED_SLURM_QOS"]
        self.approved_models = {value.strip().lower() for value in
                                settings.get("CORT_MED_APPROVED_MODEL_SHA256", "").split(",")
                                if value.strip()}
        self.allowed_folders = {value.strip() for value in
                                settings.get("CORT_MED_ALLOWED_FOLDER_IDS", "").split(",")
                                if value.strip()}

    def command(self, argv, *, check=True) -> str:
        remote = " ".join(shlex.quote(str(arg)) for arg in argv)
        result = subprocess.run(["ssh", *self.options, self.host, remote],
                                text=True, capture_output=True)
        if check and result.returncode:
            raise RuntimeError(f"HiPerGator command failed ({argv[0]}): {result.stderr.strip()[-400:]}")
        return result.stdout.strip()

    def copy_to(self, local: Path, remote: PurePosixPath) -> None:
        subprocess.run(["scp", *self.options, str(local), f"{self.host}:{remote}"],
                       check=True, capture_output=True)

    def copy_from(self, remote: PurePosixPath, local: Path) -> None:
        subprocess.run(["scp", *self.options, f"{self.host}:{remote}", str(local)],
                       check=True, capture_output=True)

    def submit(self, manifest: PurePosixPath, root: PurePosixPath) -> str:
        result = self.command(["sbatch", "--parsable", f"--account={self.account}",
                               f"--qos={self.qos}", "--time=24:00:00", "--mem=64G",
                               "--cpus-per-task=8", "--gres=gpu:l4:1",
                               f"--chdir={self.project}",
                               f"--output={root}/slurm-%j.out",
                               f"--error={root}/slurm-%j.err",
                               self.project / "run_selected_slides.sh", manifest])
        job_id = result.split(";", 1)[0]
        if not job_id.isdecimal():
            raise RuntimeError("sbatch did not return a numeric job ID")
        return job_id

    def wait(self, job_id: str, root: PurePosixPath, local_status: Path) -> dict:
        last = ""
        last_summary = None
        while True:
            if STOP:
                self.command(["scancel", job_id])
                raise InterruptedError(f"Canceled Slurm job {job_id}")
            state = self.command(["squeue", "-h", "-j", job_id, "-o", "%T"])
            if state != last:
                print(f"Slurm {job_id}: {state or 'left queue'}", flush=True)
                last = state
            try:
                self.copy_from(root / "status.json", local_status)
                status = json.loads(local_status.read_text(encoding="utf-8"))
                summary = [(item_id, entry["state"])
                           for item_id, entry in status.get("slides", {}).items()]
                if summary != last_summary:
                    print(f"Per-slide status: {summary}", flush=True)
                    last_summary = summary
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                status = {}
            if not state:
                accounting = self.command(["sacct", "-n", "-P", "-j", job_id,
                                           "--format=State"], check=False).splitlines()
                if status.get("phase") == "complete":
                    return status
                raise RuntimeError(f"Slurm job {job_id} ended without complete status: {accounting[:2]}")
            time.sleep(30)


def execute(args, client, remote: SshSlurm) -> dict:
    item_ids = [item.strip() for item in args.inputImage.split(",") if item.strip()]
    slides = select_items(client, item_ids, args.inputFolder or None, args.itemFilter)
    validate_checkpoints(client, args.corticalModelCheckpoint, args.medullaryModelCheckpoint)
    if not remote.allowed_folders:
        raise ValueError("Worker has no administrator-approved input folder IDs")
    for slide in slides:
        if slide["folder_id"] not in remote.allowed_folders:
            raise PermissionError(f"Slide folder {slide['folder_id']} is not enabled for this plugin")
        if not client.get(f"annotation/folder/{slide['folder_id']}/create"):
            raise PermissionError(f"Annotation creation denied in folder {slide['folder_id']}")
        client.createFolder(slide["folder_id"], "40x_512px_384px_overlap",
                            parentType="folder", reuseExisting=True)
    print("Matched slide items:", ", ".join(f"{item['name']} ({item['item_id']})" for item in slides),
          flush=True)
    run_id = uuid4().hex
    root = remote.output / run_id
    remote.command(["mkdir", "-m", "700", "-p", root])
    manifest = {"run_id": run_id, "model_file_id": args.corticalModelCheckpoint,
                "cortical_model_file_id": args.corticalModelCheckpoint,
                "medullary_model_file_id": args.medullaryModelCheckpoint,
                "gaussian_sigma": args.gaussianSigma, "geometry": "40x_512px_384px_overlap",
                "slides": []}
    with tempfile.TemporaryDirectory(prefix="cort-med-stage-") as directory:
        local = Path(directory)
        model = local / "model.pkl"
        client.downloadFile(args.corticalModelCheckpoint, str(model))
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if not remote.approved_models or digest not in remote.approved_models:
            raise PermissionError("Selected checkpoint SHA-256 is not on the administrator allowlist")
        remote.copy_to(model, root / "model.pkl")
        manifest["model_path"] = str(root / "model.pkl")
        for slide in slides:
            slide_root = root / slide["folder_id"] / slide["item_id"]
            remote.command(["mkdir", "-m", "700", "-p", slide_root])
            staged_name = f"{safe_stem(slide['filename'])}{Path(slide['filename']).suffix.lower()}"
            local_slide = local / f"{slide['item_id']}_{staged_name}"
            client.downloadFile(slide["file_id"], str(local_slide))
            remote.copy_to(local_slide, slide_root / staged_name)
            local_slide.unlink()
            manifest["slides"].append({"item_id": slide["item_id"],
                                       "folder_id": slide["folder_id"],
                                       "slide_file_id": slide["file_id"],
                                       "slide_path": str(slide_root / staged_name),
                                       "job_dir": str(slide_root / "trident")})
        manifest_file = local / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        remote.copy_to(manifest_file, root / "manifest.json")
        job_id = remote.submit(root / "manifest.json", root)
        print(f"HiPerGator Slurm job ID: {job_id}", flush=True)
        manifest["slurm_job_id"] = job_id
        manifest["model_sha256"] = digest
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status_file = local / "status.json"
        status = remote.wait(job_id, root, status_file)
        for stream in ("out", "err"):
            log = remote.command(["tail", "-n", "100", root / f"slurm-{job_id}.{stream}"], check=False)
            if log:
                print(f"Slurm {stream} log tail:\n{log}", flush=True)
        for slide in slides:
            entry = status.get("slides", {}).get(slide["item_id"], {})
            if entry.get("state") != "complete":
                print(f"FAILED {slide['name']}: {entry.get('error', 'No result')}", flush=True)
                continue
            stem = safe_stem(slide["filename"])
            remote_path = root / slide["folder_id"] / slide["item_id"] / "trident" / "inference_jsons" / f"{stem}_refined_heatmap.json"
            refined = local / f"{slide['item_id']}_refined_heatmap.json"
            remote.copy_from(remote_path, refined)
            annotation_ids = attach_refined_heatmap(client, slide["item_id"], refined)
            public_name = local / f"{stem}_refined_heatmap.json"
            refined.rename(public_name)
            result = upload_result(client, slide["folder_id"], slide["item_id"],
                                   run_id, public_name, status_file, manifest_file)
            print(f"ATTACHED {slide['name']}: annotations {annotation_ids}; output item {result['itemId']}",
                  flush=True)
            public_name.unlink()
        print(f"Slurm log: {root}/slurm-{job_id}.out", flush=True)
        if status.get("state") != "complete":
            raise RuntimeError(f"Slurm job {job_id} had per-slide failures")
        return status


def main() -> int:
    arguments = sys.argv[1:]
    if arguments == ["--list_cli"]:
        print(json.dumps({TASK: {"type": "python", "docker-params": {"volumes": {
            "/etc/cort-med-worker": {"bind": "/run/cort-med", "mode": "ro"}}}}}))
        return 0
    if arguments and arguments[0] == TASK:
        arguments = arguments[1:]
    if arguments == ["--xml"]:
        print((Path(__file__).with_name(f"{TASK}.xml")).read_text(encoding="utf-8"))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corticalModelCheckpoint", required=True)
    parser.add_argument("--medullaryModelCheckpoint", required=True)
    parser.add_argument("--inputImage", default="")
    parser.add_argument("--inputFolder", default="")
    parser.add_argument("--itemFilter", default=".*")
    parser.add_argument("--gaussianSigma", type=float, default=1.0)
    parser.add_argument("--girderApiUrl", default="")
    args = parser.parse_args(arguments)
    if not 0 <= args.gaussianSigma <= 10:
        parser.error("--gaussianSigma must be between 0 and 10")
    config_file = Path(os.environ.get("CORT_MED_CONFIG_FILE", "/run/cort-med/config.json"))
    if not config_file.is_file():
        parser.error("Worker configuration mount is unavailable")
    settings = json.loads(config_file.read_text(encoding="utf-8"))
    settings.update({key: value for key, value in os.environ.items() if key.startswith("CORT_MED_")})
    settings.setdefault("CORT_MED_SSH_KEY_FILE", "/run/cort-med/ssh-key")
    settings.setdefault("CORT_MED_SSH_KNOWN_HOSTS", "/run/cort-med/known_hosts")
    api_url = settings.get("CORT_MED_GIRDER_API_URL", "") or args.girderApiUrl
    parsed_url = urlparse(api_url)
    if (parsed_url.scheme != "https" or not parsed_url.netloc or parsed_url.username or
            parsed_url.password or parsed_url.query or parsed_url.fragment):
        parser.error("A configured HTTPS Girder API URL is required")
    key_file = Path(settings.get("CORT_MED_GIRDER_API_KEY_FILE", "/run/cort-med/girder-api-key"))
    if not key_file.is_file():
        parser.error("Worker Girder API key secret mount is unavailable")
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    try:
        from girder_client import GirderClient
        client = GirderClient(apiUrl=api_url)
        try:
            client.authenticate(apiKey=key_file.read_text(encoding="utf-8").strip())
        except Exception as error:
            raise RuntimeError("Girder service authentication failed") from error
        execute(args, client, SshSlurm(settings))
    except Exception as error:
        print(f"CorticalMedullaryIdentification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
