"""Small CPU checks for grid selection, metrics, and medullary precedence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from scripts.inference import aggregate, map_payload, save_visualizations
from scripts.h5_svm import load_tiles, metrics


class H5SvmWorkflowTests(unittest.TestCase):
    def test_h5_no_overlap_uses_metadata_and_keeps_labels_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            with h5py.File(path, "w") as handle:
                coords = handle.create_dataset("coords", data=np.array([[100, 200], [356, 200], [612, 200], [100, 456], [612, 712]]))
                coords.attrs["patch_size"] = 512
                coords.attrs["patch_size_level0"] = 512
                coords.attrs["overlap"] = 256
                handle.create_dataset("features", data=np.arange(10, dtype=np.float32).reshape(5, 2))
                handle.create_dataset("labels", data=np.array([1, 2, 2, 0, 1]))
            sample = load_tiles(path, "no-overlap", require_labels=True)
            assert sample is not None
            np.testing.assert_array_equal(sample.coords, [[100, 200], [612, 200], [612, 712]])
            np.testing.assert_array_equal(sample.labels, [0, 1, 0])
            self.assertEqual(sample.total, 5)
            all_tiles = load_tiles(path, "all", require_labels=False)
            assert all_tiles is not None
            self.assertEqual(len(all_tiles.coords), 5)

    def test_metrics_and_overlap_priority(self):
        result = metrics(np.array([0, 0, 1, 1]), np.array([-1, 2, -2, 3], dtype=np.float32))
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["mcc"], 0)
        self.assertEqual(result["cortex_sensitivity"], 0.5)
        coords = np.array([[0, 0], [256, 0]], dtype=np.int64)
        origin, raw, refined, seen = aggregate(coords, np.array([-2, 1], dtype=np.float32), 512, 256, 1)
        self.assertGreaterEqual(raw[0, 1], 0)
        self.assertGreaterEqual(refined[0, 1], 0)
        payload = map_payload("sample.svs", "all", origin, 256, refined, seen, mask=True)
        self.assertIn([256, 0, 1], payload["cells"])

    def test_prediction_visualizations_use_thumbnail_and_write_three_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "KPMP"
            feature_dir = job / "40x_512px_256px_overlap" / "features_test"
            feature_dir.mkdir(parents=True)
            h5_path = feature_dir / "sample.h5"
            with h5py.File(h5_path, "w") as handle:
                coords = handle.create_dataset("coords", data=np.array([[100, 100]]))
                coords.attrs["level0_width"] = 400
                coords.attrs["level0_height"] = 400
            thumbnail_dir = job / "thumbnails"
            thumbnail_dir.mkdir()
            Image.new("RGB", (100, 100), "white").save(thumbnail_dir / "sample.jpg")
            output = job / "prediction_visualization"
            raw = np.array([[-2, 2], [0, 1]], dtype=np.float32)
            refined = np.array([[-1, -0.1], [0, 0.5]], dtype=np.float32)
            seen = np.array([[True, True], [False, True]])
            save_visualizations(h5_path, "sample", np.array([100, 100]), 100,
                                raw, refined, seen, output)
            for suffix in ("heatmap", "refined_heatmap", "mask"):
                with Image.open(output / f"sample_{suffix}.png") as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 500)
                    self.assertGreater(image.height, 500)


if __name__ == "__main__":
    unittest.main()
