"""Non-GPU contract tests for DSA selection, orchestration and annotation retries."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))
from results import attach_refined_heatmap, upload_result
from selection import select_items, validate_checkpoints
from entrypoint import SshSlurm, safe_stem
from scripts.run_selected_slides import GEOMETRY, run, validate_manifest


class FakeGirder:
    def __init__(self):
        self.items = {
            "slide1": {"_id": "slide1", "folderId": "folderA", "name": "A.svs",
                       "largeImage": {"fileId": "file1"}},
            "slide2": {"_id": "slide2", "folderId": "folderA", "name": "B.svs",
                       "largeImage": {"fileId": "file2"}},
        }
        self.files = {name: {"_id": name, "itemId": f"slide{index}",
                             "name": f"{'A' if index == 1 else 'B'}.svs"}
                      for index, name in enumerate(("file1", "file2"), 1)}
        self.files["model"] = {"_id": "model", "name": "svc.pkl"}
        self.files["other"] = {"_id": "other", "name": "other.pkl"}
        self.annotations = []
        self.created = []

    def getFile(self, file_id):
        return self.files[file_id]

    def getItem(self, item_id):
        return self.items[item_id]

    def listItem(self, folder_id):
        return list(self.items.values()) if folder_id == "folderA" else []

    def get(self, path, parameters=None):
        if path == "annotation":
            return self.annotations
        return True

    def put(self, path, json):
        self.created.append(("update", path, json))
        return {"_id": path.rsplit("/", 1)[-1]}

    def post(self, path, parameters=None, json=None):
        self.created.append(("create", path, json))
        return {"_id": f"ann{len(self.created)}"}

    def createFolder(self, parent, name, **_kwargs):
        return {"_id": f"{parent}/{name}"}

    def createItem(self, parent, name, **_kwargs):
        return {"_id": f"{parent}/{name}"}

    def uploadFileToItem(self, item_id, path, **_kwargs):
        self.created.append(("upload", item_id, Path(path).name))


class SelectionTests(unittest.TestCase):
    def test_file_and_folder_filter_use_actual_slide_names(self):
        client = FakeGirder()
        self.assertEqual([row["item_id"] for row in select_items(client, ["file1"], None, r"^A\.svs$")],
                         ["slide1"])
        self.assertEqual([row["item_id"] for row in select_items(client, [], "folderA", r"B\.svs$")],
                         ["slide2"])
        with self.assertRaisesRegex(ValueError, "matched no"):
            select_items(client, ["file1"], None, r"^Z")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_items(client, ["file1"], "folderA", ".*")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_items(client, ["file1", "file1"], None, ".*")
        with self.assertRaisesRegex(ValueError, "Invalid item filter"):
            select_items(client, ["file1"], None, "[")

    def test_model_and_slide_validation(self):
        client = FakeGirder()
        self.assertEqual(validate_checkpoints(client, "model", "model")["name"], "svc.pkl")
        with self.assertRaisesRegex(ValueError, "same shared"):
            validate_checkpoints(client, "model", "other")
        client.items["slide1"]["largeImage"] = {}
        with self.assertRaisesRegex(ValueError, "not a DSA large image"):
            select_items(client, ["file1"], None, ".*")


class OrchestrationTests(unittest.TestCase):
    def test_all_preprocessing_finishes_before_first_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pkl"
            model.write_bytes(b"fake")
            slides = []
            for index in (1, 2):
                slide = root / f"sample{index}.svs"
                slide.write_bytes(b"fake")
                slides.append({"item_id": f"item{index}", "folder_id": "folderA",
                               "slide_path": str(slide), "job_dir": str(root / f"job{index}")})
            manifest = {"model_path": str(model), "slides": slides}
            calls = []

            def execute(argv, **_kwargs):
                calls.append(argv)
                if argv[0] == "bash":
                    job = Path(argv[3])
                    feature = job / GEOMETRY / "features_uni_v1" / (Path(argv[2]).stem + ".h5")
                    feature.parent.mkdir(parents=True)
                    feature.write_bytes(b"fake")
                else:
                    output = Path(argv[argv.index("--output-dir") + 1])
                    output.mkdir(parents=True)
                    (output / (argv[argv.index("--sample-name") + 1] + "_refined_heatmap.json")).write_text("[]")

            fake_h5 = types.SimpleNamespace(load_model=lambda _path: {"metadata": {
                "class_names": ("cortex", "medullary"), "pca": False,
                "feature_dimension": 1024, "tile_mode": "all"}}, safe_name=lambda name: name)
            with patch.dict(sys.modules, {"scripts.h5_svm": fake_h5}):
                status = run(manifest, root, root / "status.json", execute=execute)
            self.assertEqual([call[0] for call in calls], ["bash", "bash", sys.executable, sys.executable])
            self.assertEqual(status["state"], "complete")
            self.assertEqual({entry["state"] for entry in status["slides"].values()}, {"complete"})

    def test_manifest_rejects_duplicate_items_and_missing_model(self):
        slide = {"item_id": "a", "folder_id": "f", "slide_path": "/tmp/a.svs",
                 "job_dir": "/tmp/job"}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest({"model_path": "/tmp/m.pkl", "slides": [slide, slide]})

    def test_preprocessing_failure_reports_item_and_skips_its_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pkl"
            model.write_bytes(b"fake")
            manifest = {"model_path": str(model), "slides": [
                {"item_id": "item1", "folder_id": "folderA", "slide_path": str(root / "bad.svs"),
                 "job_dir": str(root / "job")}]}

            def fail(argv, **_kwargs):
                raise OSError("preprocessing failed")

            fake_h5 = types.SimpleNamespace(load_model=lambda _path: {"metadata": {
                "class_names": ("cortex", "medullary"), "pca": False,
                "feature_dimension": 1024, "tile_mode": "all"}}, safe_name=lambda name: name)
            with patch.dict(sys.modules, {"scripts.h5_svm": fake_h5}):
                result = run(manifest, root, root / "status.json", execute=fail)
            self.assertEqual(result["state"], "error")
            self.assertEqual(result["slides"]["item1"]["phase"], "preprocessing")
            self.assertIn("preprocessing failed", result["slides"]["item1"]["error"])


class ResultTests(unittest.TestCase):
    def test_attach_replaces_only_owned_annotations(self):
        client = FakeGirder()
        client.annotations = [
            {"_id": "unrelated", "annotation": {"name": "cortical heatmap"}},
            {"_id": "owned", "annotation": {"name": "medullary heatmap", "attributes": {
                "sourcePlugin": "CorticalMedullaryIdentification", "sourceItemId": "slide1"}}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_refined_heatmap.json"
            path.write_text(json.dumps([{"name": "cortical heatmap", "elements": []},
                                        {"name": "medullary heatmap", "elements": []}]))
            ids = attach_refined_heatmap(client, "slide1", path)
            self.assertEqual(len(ids), 2)
            self.assertEqual([entry[0] for entry in client.created], ["create", "update"])
            status = Path(directory) / "status.json"
            status.write_text("{}")
            result = upload_result(client, "folderA", "slide1", "run1", path, status)
            self.assertIn("folderA/40x_512px_384px_overlap", result["folderId"])


class SchedulerTests(unittest.TestCase):
    def test_safe_name_and_exact_slurm_request(self):
        self.assertEqual(safe_stem("case/18;142 PAS.svs"), "18_142_PAS")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "key").write_text("test")
            (root / "hosts").write_text("test")
            settings = {"CORT_MED_SSH_HOST": "login.example.edu", "CORT_MED_SSH_USER": "worker",
                        "CORT_MED_SSH_KEY_FILE": str(root / "key"),
                        "CORT_MED_SSH_KNOWN_HOSTS": str(root / "hosts"),
                        "CORT_MED_REMOTE_PROJECT_DIR": "/blue/project/repo",
                        "CORT_MED_REMOTE_OUTPUT_ROOT": "/blue/project/results",
                        "CORT_MED_SLURM_ACCOUNT": "group", "CORT_MED_SLURM_QOS": "group"}
            remote = SshSlurm(settings)
            commands = []

            def command(argv, **_kwargs):
                commands.append(argv)
                return "12345"

            remote.command = command
            self.assertEqual(remote.submit(remote.output / "manifest.json", remote.output), "12345")
            self.assertIn("--gres=gpu:l4:1", commands[0])
            self.assertIn("--cpus-per-task=8", commands[0])
            self.assertIn("--mem=64G", commands[0])
            self.assertIn("--time=24:00:00", commands[0])


if __name__ == "__main__":
    unittest.main()
