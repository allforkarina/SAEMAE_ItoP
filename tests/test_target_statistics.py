"""Run on the Linux server: python -m unittest discover -s tests -v."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from tools.target_statistics import (
    Rules, compute_attributes, generate_dataset, load_amplitude, make_prompt,
)


def write_target(path, amplitude):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.zeros(amplitude.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
    raw["real"] = amplitude
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("pfa_patch", data=raw)
        dataset.attrs["MATLAB_class"] = np.bytes_("single")


class AttributeTests(unittest.TestCase):
    def test_uniform_image_does_not_invent_a_dominant_region(self):
        attrs = compute_attributes(np.full((512, 512), 7.0))
        self.assertEqual(attrs["dominant_region"], "no_single_region")
        self.assertEqual(attrs["concentration_level"], "low")
        self.assertEqual(attrs["texture_level"], "low")
        self.assertEqual(attrs["quantile_contrast"], 0.0)
        self.assertAlmostEqual(sum(map(sum, attrs["power_grid_3x3"])), 1.0)

    def test_unequal_grid_cell_sizes_do_not_imply_concentration(self):
        attrs = compute_attributes(np.ones((4, 4)))
        self.assertEqual(attrs["dominant_region"], "no_single_region")

    def test_isolated_scatterer_is_concentrated_in_upper_right(self):
        amplitude = np.zeros((12, 12))
        amplitude[1, 10] = 50
        attrs = compute_attributes(amplitude)
        self.assertEqual(attrs["dominant_region"], "upper_right")
        self.assertEqual(attrs["brightest_power_share"], 1.0)
        self.assertEqual(attrs["concentration_level"], "high")
        self.assertIn("upper-right", make_prompt(attrs))

    def test_equal_regions_do_not_arbitrarily_choose_one(self):
        amplitude = np.zeros((12, 12))
        amplitude[1, 1] = amplitude[10, 10] = 8
        attrs = compute_attributes(amplitude)
        self.assertEqual(attrs["dominant_region"], "no_single_region")

    def test_global_gain_does_not_change_relative_attributes_or_prompt(self):
        amplitude = np.arange(1, 145, dtype=float).reshape(12, 12)
        first = compute_attributes(amplitude)
        scaled = compute_attributes(amplitude * 1e100)
        self.assertEqual(make_prompt(first), make_prompt(scaled))
        for name in ("quantile_contrast", "brightest_power_share", "texture_gradient_mean"):
            self.assertAlmostEqual(first[name], scaled[name])
        np.testing.assert_allclose(first["power_grid_3x3"], scaled["power_grid_3x3"])

    def test_checkerboard_has_more_local_variation_than_smooth_ramp(self):
        smooth = np.tile(np.linspace(0, 1, 60), (60, 1))
        checkerboard = np.indices((60, 60)).sum(axis=0) % 2
        self.assertEqual(compute_attributes(smooth)["texture_level"], "low")
        self.assertEqual(compute_attributes(checkerboard)["texture_level"], "high")

    def test_zero_image_has_finite_attributes_and_no_scattering_claim(self):
        attrs = compute_attributes(np.zeros((12, 12)))
        self.assertEqual(attrs["dominant_region"], "no_single_region")
        self.assertEqual(attrs["concentration_level"], "undefined")
        self.assertIn("zero amplitude", make_prompt(attrs))
        json.dumps(attrs, allow_nan=False)

    def test_invalid_images_and_rules_are_rejected(self):
        for amplitude in (
            np.full((6, 6), np.nan), np.full((6, 6), np.inf),
            np.full((6, 6), -1), np.zeros((2, 6)), np.zeros((6, 6, 1)),
        ):
            with self.subTest(shape=amplitude.shape):
                with self.assertRaises(ValueError):
                    compute_attributes(amplitude)
        with self.assertRaises(ValueError):
            compute_attributes(np.ones((6, 6)), Rules(top_fraction=0))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.amplitude = np.zeros((12, 12))
        self.amplitude[1, 10] = 5
        write_target(self.root / "images/sample.mat", self.amplitude)
        self.records = [{
            "source": "echo/sample.mat", "target": "images/sample.mat",
            "prompt": "Echo to image", "split": "train",
        }]
        self.manifest = self.root / "pairs.json"
        self.manifest.write_text(json.dumps(self.records), encoding="utf-8")

    def test_complex_fields_and_explicit_orientation(self):
        path = self.root / "images/complex.mat"
        raw = np.zeros((6, 9), dtype=[("real", "<f4"), ("imag", "<f4")])
        raw["real"][0, 8] = 3
        raw["imag"][0, 8] = 4
        with h5py.File(path, "w") as handle:
            handle.create_dataset("pfa_patch", data=raw)
        native, orientation = load_amplitude(path)
        self.assertEqual(native.shape, (6, 9))
        self.assertEqual(native[0, 8], 5)
        self.assertFalse(orientation["transpose"])
        transformed, _ = load_amplitude(path, transpose=True)
        self.assertEqual(transformed.shape, (9, 6))
        self.assertEqual(compute_attributes(transformed)["dominant_region"], "lower_left")
        flipped, _ = load_amplitude(path, flip_horizontal=True, flip_vertical=True)
        self.assertEqual(flipped[5, 0], 5)

    def test_end_to_end_preserves_original_and_creates_preview(self):
        original_bytes = self.manifest.read_bytes()
        output = self.root / "generated.json"
        previews = self.root / "previews"
        self.assertEqual(generate_dataset(self.manifest, self.root, output, preview_dir=previews), 1)
        result = json.loads(output.read_text(encoding="utf-8"))[0]
        for key in ("source", "target", "split"):
            self.assertEqual(result[key], self.records[0][key])
        self.assertEqual(result["image_attributes"]["dominant_region"], "upper_right")
        self.assertFalse(result["image_attributes"]["uses_sarmae"])
        self.assertTrue(result["prompt"].startswith("Echo to image. "))
        self.assertEqual(self.manifest.read_bytes(), original_bytes)
        self.assertTrue((previews / "000000_sample.png").is_file())

    def test_refuses_to_overwrite_input_or_existing_output(self):
        with self.assertRaises(ValueError):
            generate_dataset(self.manifest, self.root, self.manifest)
        output = self.root / "existing.json"
        output.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            generate_dataset(self.manifest, self.root, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_missing_or_invalid_target_stops_without_writing_json(self):
        for target in ("missing.mat", "images/real_only.mat"):
            with self.subTest(target=target):
                with h5py.File(self.root / "images/real_only.mat", "w") as handle:
                    handle.create_dataset("pfa_patch", data=np.ones((12, 12)))
                self.manifest.write_text(json.dumps([
                    self.records[0], {"source": "echo/b.mat", "target": target},
                ]), encoding="utf-8")
                output = self.root / "failed.json"
                with self.assertRaisesRegex(ValueError, "Record 1"):
                    generate_dataset(self.manifest, self.root, output)
                self.assertFalse(output.exists())

    def test_limit_outputs_only_selected_records_in_original_order(self):
        records = [dict(self.records[0], sample_id=i) for i in range(3)]
        self.manifest.write_text(json.dumps(records), encoding="utf-8")
        output = self.root / "subset.json"
        generate_dataset(self.manifest, self.root, output, limit=2)
        results = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([row["sample_id"] for row in results], [0, 1])


if __name__ == "__main__":
    unittest.main()
