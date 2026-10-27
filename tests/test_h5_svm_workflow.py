"""Small CPU checks for grid selection, metrics, and medullary precedence."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import matplotlib
import numpy as np
from PIL import Image, ImageChops
from shapely.geometry import MultiPolygon, box

from scripts.convert_inference_json import legacy_to_annotation
from scripts.inference import (aggregate, contour_payload, heatmap_payload,
                               load_sample_contours, load_tissue_geometry, map_payload,
                               save_sample_exports, save_visualizations,
                               threshold_grid_geometry)
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
        payload = map_payload("sample.svs", "all", origin, 256, refined, seen,
                              mask=True, layer="mask")
        grid = payload["elements"][0]
        self.assertEqual(grid["type"], "griddata")
        self.assertEqual(grid["origin"], [128, 128, 0])
        self.assertEqual(grid["gridWidth"], 3)
        self.assertEqual(grid["values"][1], 1)

    def test_athena_json_keeps_signed_scores_and_hides_unobserved_cells(self):
        scores = np.array([[-2, 0, 2], [1, 9, -1]], dtype=np.float32)
        seen = np.array([[True, True, True], [True, False, True]])
        origin = np.array([100, 200])
        heatmap = map_payload("sample.svs", "all", origin, 100, scores, seen,
                              mask=False, layer="heatmap")
        mask = map_payload("sample.svs", "all", origin, 100, scores, seen,
                           mask=True, layer="mask")
        self.assertEqual([document["name"] for document in heatmap],
                         ["cortical heatmap", "medullary heatmap"])
        cortical, medullary = heatmap
        self.assertEqual(len(cortical["elements"]), 2)
        self.assertEqual(cortical["elements"][0]["center"], [150, 250, 0])
        self.assertEqual(cortical["elements"][0]["user"]["decision_score"], -2)
        self.assertEqual(cortical["elements"][1]["user"]["tile_x"], 300)
        self.assertEqual(len(medullary["elements"]), 3)
        self.assertEqual(medullary["elements"][0]["user"]["decision_score"], 0)
        self.assertEqual(medullary["elements"][1]["user"]["decision_score"], 2)
        self.assertEqual(mask["elements"][0]["values"], [0, 1, 1, 1, -1, 0])
        self.assertEqual(mask["elements"][0]["minColor"], "rgba(0,0,0,0)")
        json.dumps(heatmap, allow_nan=False)
        json.dumps(mask, allow_nan=False)

    def test_legacy_cell_maps_convert_without_rerunning_inference(self):
        metadata = {"sample_name": "sample.svs", "tile_mode": "all",
                    "origin_level0": [100, 200], "cell_size_level0": 100}
        mask = legacy_to_annotation({**metadata, "cells": [[100, 200, 0],
                                                        [300, 200, 1]]}, "mask")
        self.assertEqual(mask["elements"][0]["values"], [0, -1, 1])
        heatmap = legacy_to_annotation({**metadata, "cells": [[100, 200, 0.8, 0.2, 0],
                                                           [300, 200, 0.2, 0.8, 1]]}, "heatmap")
        self.assertEqual([document["name"] for document in heatmap],
                         ["cortical heatmap", "medullary heatmap"])
        self.assertLess(heatmap[0]["elements"][0]["user"]["decision_score"], 0)
        self.assertGreater(heatmap[1]["elements"][0]["user"]["decision_score"], 0)

    def test_small_rgba_alpha_uses_fixed_decimal_notation(self):
        alpha = 4.43172e-07
        payload = heatmap_payload(
            "sample", "all", np.array([[0, 0], [100, 0]]),
            np.array([1.0, alpha]), 100,
        )
        medullary = payload[1]["elements"]
        tiny_element = next(element for element in medullary
                            if element["user"]["tile_x"] == 100)
        expected = "rgba(255, 105, 180, 0.00000044317201)"
        self.assertEqual(tiny_element["fillColor"], expected)
        self.assertNotIn("e", tiny_element["fillColor"].lower())

        clipped = contour_payload("sample", box(0, 0, 100, 100), None, alpha)
        self.assertEqual(clipped[0]["elements"][0]["fillColor"],
                         "rgba(0, 255, 255, 0.000000443172)")

    def test_dsa_heatmap_and_contour_exports_keep_geometry_metadata_and_unique_ids(self):
        coords = np.array([[100, 200], [400, 200], [700, 400]], dtype=np.int64)
        scores = np.array([-4, 2, 1], dtype=np.float32)
        heatmap = heatmap_payload("sample", "all", coords, scores, 256, 128,
                                  origin_level0=np.array([100, 200]), source_stride=128)
        raw_heatmap = heatmap_payload(
            "sample", "all", coords, scores, 256, 128,
            layer="raw_heatmap", origin_level0=np.array([100, 200]),
            source_stride=128,
        )
        refined_heatmap = heatmap_payload(
            "sample", "all", coords, scores / 2, 256, 128,
            layer="refined_heatmap", origin_level0=np.array([100, 200]),
            source_stride=128,
        )
        cortex = box(0, 0, 100, 100)
        medullary = MultiPolygon([box(20, 20, 40, 40), box(120, 120, 140, 140)])
        contours = contour_payload("sample", cortex, medullary, 0.35,
                                   source_status={"cortex": "present", "medullary": "present"})
        predictions = contour_payload("sample", cortex, medullary, 0.35,
                                      source_status={"cortex": "derived", "medullary": "derived"},
                                      layer="mask_prediction")

        with tempfile.TemporaryDirectory() as directory:
            paths = save_sample_exports(Path(directory), "case/sample", heatmap,
                                        refined_heatmap, predictions, contours,
                                        raw_heatmap=raw_heatmap)
            heatmap_path = paths[0]
            contour_path = Path(directory) / "sample_mask_gt.json"
            self.assertEqual(len(paths), 5)
            self.assertEqual(heatmap_path.name, "sample_heatmap.json")
            self.assertEqual(contour_path.name, "sample_mask_gt.json")
            parsed_heatmap = json.loads(heatmap_path.read_text(encoding="utf-8"))
            parsed_raw = json.loads((Path(directory) / "sample_raw_heatmap.json").read_text(encoding="utf-8"))
            parsed_refined = json.loads((Path(directory) / "sample_refined_heatmap.json").read_text(encoding="utf-8"))
            parsed_contours = json.loads(contour_path.read_text(encoding="utf-8"))
            parsed_predictions = json.loads((Path(directory) / "sample_mask_prediction.json").read_text(encoding="utf-8"))

        self.assertEqual([document["name"] for document in parsed_heatmap],
                         ["cortical heatmap", "medullary heatmap"])
        self.assertEqual([document["name"] for document in parsed_raw],
                         ["cortical raw heatmap", "medullary raw heatmap"])
        self.assertEqual(parsed_raw[0]["attributes"]["layer"], "raw_heatmap")
        self.assertEqual(parsed_raw[0]["elements"][0]["user"]["decision_score"], -4)
        self.assertEqual([document["name"] for document in parsed_contours],
                         ["cortical", "medullary"])
        self.assertEqual([document["name"] for document in parsed_predictions],
                         ["cortical mask prediction", "medullary mask prediction"])
        cortical_tile = parsed_heatmap[0]["elements"][0]
        self.assertEqual(cortical_tile["type"], "rectangle")
        self.assertEqual(cortical_tile["center"], [228, 264, 0])
        self.assertEqual((cortical_tile["width"], cortical_tile["height"]), (256, 128))
        self.assertEqual(cortical_tile["user"]["class"], "cortical")
        self.assertEqual(cortical_tile["user"]["decision_score"], -4)
        self.assertEqual(cortical_tile["user"]["scaled_score"], 1)
        self.assertEqual(parsed_heatmap[0]["attributes"]["scoreColumn"], "decision_score")
        medullary_tiles = parsed_heatmap[1]["elements"]
        self.assertEqual([element["user"]["scaled_score"] for element in medullary_tiles],
                         [1, 0.5])
        self.assertTrue(all(element["fillColor"].startswith("rgba(255, 105, 180, ")
                            for element in medullary_tiles))
        self.assertEqual(parsed_contours[0]["elements"][0]["fillColor"],
                         "rgba(0, 255, 255, 0.35)")
        self.assertEqual(parsed_contours[0]["elements"][0]["lineColor"],
                         "rgba(0, 255, 255, 1)")
        self.assertEqual(parsed_contours[0]["elements"][0]["user"]["class"], "cortical")
        self.assertEqual(len(parsed_contours[0]["elements"][0]["holes"]), 1)
        cortical_points = parsed_contours[0]["elements"][0]["points"]
        self.assertEqual((min(point[0] for point in cortical_points),
                          max(point[0] for point in cortical_points),
                          min(point[1] for point in cortical_points),
                          max(point[1] for point in cortical_points)), (0, 100, 0, 100))
        self.assertEqual(parsed_contours[1]["elements"][0]["fillColor"],
                         "rgba(255, 105, 180, 0.35)")
        self.assertEqual(parsed_contours[1]["elements"][0]["user"]["class"], "medullary")
        self.assertEqual(len(parsed_contours[1]["elements"]), 2)
        self.assertTrue(all(
            len(point) == 3 and point[2] == 0
            for document in parsed_contours
            for element in document["elements"]
            for ring in [element["points"], *element["holes"]]
            for point in ring
        ))
        all_elements = [element for document in (*parsed_heatmap, *parsed_raw,
                                                  *parsed_refined,
                                                  *parsed_contours,
                                                  *parsed_predictions)
                        for element in document["elements"]]
        ids = [element["id"] for element in all_elements]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{24}", element_id) for element_id in ids))

    def test_empty_heatmap_class_and_missing_versus_empty_contour_sources(self):
        heatmap = heatmap_payload("sample", "all", np.array([[1, 2], [3, 4]]),
                                  np.array([-1, -2]), 64)
        self.assertEqual([len(document["elements"]) for document in heatmap], [2, 0])
        with tempfile.TemporaryDirectory() as directory:
            annotation_dir = Path(directory)
            sample_dir = annotation_dir / "sample"
            sample_dir.mkdir()
            (sample_dir / "cortical_gt.json").write_text(
                json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
            geometries, statuses = load_sample_contours(annotation_dir, "sample")
            self.assertIsNone(geometries["cortex"])
            self.assertIsNone(geometries["medullary"])
            self.assertEqual(statuses, {"cortex": "empty", "medullary": "missing"})
            contours = contour_payload("sample", geometries["cortex"],
                                       geometries["medullary"], source_status=statuses)
        self.assertEqual([document["name"] for document in contours], ["cortical", "medullary"])
        self.assertEqual([document["attributes"]["source_status"] for document in contours],
                         ["empty", "missing"])
        self.assertEqual([document["elements"] for document in contours], [[], []])

    def test_sample_exports_refuse_to_replace_either_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contour_path = output / "sample_mask_prediction.json"
            contour_path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_sample_exports(output, "sample", [], [], [], raw_heatmap=[])
            self.assertFalse((output / "sample_heatmap.json").exists())
            self.assertEqual(contour_path.read_text(encoding="utf-8"), "preserve")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            raw_path = output / "sample.pyr_raw_heatmap.json"
            raw_path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_sample_exports(output, "sample.pyr", [], [], [], raw_heatmap=[])
            self.assertFalse((output / "sample.pyr_heatmap.json").exists())
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "preserve")

    def test_raw_heatmap_preserves_overlapping_tile_scores_and_clips_to_tissue(self):
        coords = np.array([[0, 0], [50, 0], [200, 0]])
        scores = np.array([-2, 3, 7], dtype=np.float32)
        tissue = box(25, 10, 125, 90)
        raw = heatmap_payload("sample", "all", coords, scores, 100,
                              layer="raw_heatmap", tissue_geometry=tissue)
        cortical, medullary = raw
        self.assertEqual([len(row["elements"]) for row in raw], [1, 1])
        self.assertEqual(cortical["elements"][0]["user"]["decision_score"], -2)
        self.assertEqual(medullary["elements"][0]["user"]["decision_score"], 3)
        self.assertEqual([row["attributes"]["numberOfTiles"] for row in raw], [2, 2])
        for row in raw:
            element = row["elements"][0]
            self.assertEqual(element["type"], "polyline")
            self.assertTrue(all(25 <= x <= 125 and 10 <= y <= 90
                                for x, y, _ in element["points"]))

    def test_thresholded_prediction_contours_keep_only_tissue_and_respect_zero(self):
        scores = np.array([[-1, 0, 1], [-1, -1, 1]], dtype=np.float32)
        seen = np.ones_like(scores, dtype=bool)
        tissue = box(100, 0, 300, 200)
        cortex = threshold_grid_geometry(scores, seen, np.array([0, 0]), 100,
                                        tissue, medullary=False)
        medullary = threshold_grid_geometry(scores, seen, np.array([0, 0]), 100,
                                            tissue, medullary=True)
        self.assertAlmostEqual(cortex.area, 10000)
        self.assertAlmostEqual(medullary.area, 30000)
        self.assertTrue(tissue.covers(cortex))
        self.assertTrue(tissue.covers(medullary))
        self.assertEqual(cortex.intersection(medullary).area, 0)

    def test_tissue_mask_geojson_is_loaded_by_sample_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample_tissue.geojson").write_text(
                json.dumps({"type": "Polygon", "coordinates": [[[0, 0], [20, 0],
                           [20, 20], [0, 20], [0, 0]]]}), encoding="utf-8")
            tissue = load_tissue_geometry(root, "case/sample.svs")
        self.assertEqual(tissue.area, 400)

    def test_tissue_mask_preserves_dotted_slide_stem_and_matches_dot_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            polygon = {"type": "Polygon", "coordinates": [[[0, 0], [20, 0],
                       [20, 20], [0, 20], [0, 0]]]}
            smaller = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0],
                       [10, 10], [0, 10], [0, 0]]]}
            (root / "Image_CKD_100_PAS.geojson").write_text(
                json.dumps(smaller), encoding="utf-8")
            (root / "Image_CKD_100_PAS.pyr.noimagej.geojson").write_text(
                json.dumps(polygon), encoding="utf-8")
            for sample in ("Image_CKD_100_PAS.pyr",
                           "case/Image_CKD_100_PAS.pyr.tiff"):
                with self.subTest(sample=sample):
                    self.assertEqual(load_tissue_geometry(root, sample).area, 400)

    def test_heatmap_and_gt_contours_are_clipped_to_tissue_and_gt_export_can_be_omitted(self):
        tissue = box(50, 20, 100, 80)
        heatmap = heatmap_payload("sample", "all", np.array([[0, 0]]),
                                  np.array([-2]), 100, tissue_geometry=tissue)
        clipped_tile = heatmap[0]["elements"][0]
        self.assertEqual(clipped_tile["type"], "polyline")
        self.assertEqual(clipped_tile["user"]["decision_score"], -2)
        tile_points = clipped_tile["points"]
        self.assertEqual((min(point[0] for point in tile_points),
                          max(point[0] for point in tile_points),
                          min(point[1] for point in tile_points),
                          max(point[1] for point in tile_points)), (50, 100, 20, 80))
        edge_only = heatmap_payload("sample", "all", np.array([[0, 0]]),
                                    np.array([-2]), 100,
                                    tissue_geometry=box(100, 0, 200, 100))
        self.assertEqual(edge_only[0]["elements"], [])
        gt = contour_payload("sample", box(0, 0, 100, 100), None, 0.2,
                             tissue_geometry=tissue)
        self.assertEqual(gt[0]["elements"][0]["fillColor"], "rgba(0, 255, 255, 0.2)")
        with tempfile.TemporaryDirectory() as directory:
            save_sample_exports(Path(directory), "sample", heatmap, heatmap, [], None,
                                raw_heatmap=heatmap)
            self.assertFalse((Path(directory) / "sample_mask_gt.json").exists())
            self.assertTrue((Path(directory) / "sample_raw_heatmap.json").exists())
            self.assertTrue((Path(directory) / "sample_mask_prediction.json").exists())

    def test_prediction_visualizations_use_thumbnail_and_write_four_pngs(self):
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
            annotation_dir = job / "annotations"
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # Load pyplot before patching Axes.plot.
            with patch("matplotlib.axes.Axes.plot") as plot:
                save_visualizations(h5_path, "sample", np.array([100, 100]), 100,
                                    raw, refined, seen, output, annotation_dir)
                plot.assert_not_called()
            for suffix in ("heatmap", "refined_heatmap", "mask_h", "mask_rh"):
                with Image.open(output / f"sample_{suffix}.png") as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 500)
                    self.assertGreater(image.height, 500)
            with Image.open(output / "sample_mask_rh.png") as image:
                mask_without_gt = image.convert("RGB").copy()
            for suffix in ("aggregate", "mask", "refinement", "heatmap_refinement"):
                self.assertFalse((output / f"sample_{suffix}.png").exists())

            sample_dir = annotation_dir / "sample"
            sample_dir.mkdir(parents=True)
            for filename, x in (("cortical_gt.json", 20), ("medullary_gt.json", 220)):
                polygon = {"type": "Polygon", "coordinates": [[[x, 30], [x + 80, 30],
                            [x + 80, 90], [x, 30]]]}
                (sample_dir / filename).write_text(json.dumps(polygon), encoding="utf-8")
            with patch("matplotlib.axes.Axes.plot") as plot:
                save_visualizations(h5_path, "sample", np.array([100, 100]), 100,
                                    raw, refined, seen, output, annotation_dir)
            self.assertEqual(plot.call_count, 8)
            self.assertEqual([call.kwargs["color"] for call in plot.call_args_list],
                             ["#00ffff", "#ff69b4"] * 4)
            np.testing.assert_array_equal(plot.call_args_list[0].args[0], [20, 100, 100, 20])
            np.testing.assert_array_equal(plot.call_args_list[0].args[1], [30, 30, 90, 30])
            save_visualizations(h5_path, "sample", np.array([100, 100]), 100,
                                raw, refined, seen, output, annotation_dir)
            with Image.open(output / "sample_mask_rh.png") as image:
                difference = ImageChops.difference(mask_without_gt, image.convert("RGB"))
                self.assertIsNotNone(difference.getbbox())


if __name__ == "__main__":
    unittest.main()
