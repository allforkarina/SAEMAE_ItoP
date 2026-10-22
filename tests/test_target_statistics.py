"""Run on the Linux server: python -m unittest discover -s tests -v."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from tools.target_statistics import (
    Rules, build_pairs, compute_attributes, generate_dataset, load_amplitude, make_prompt,
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

    def test_numeric_prompt_contains_three_decimal_statistics_in_grid_order(self):
        attrs = compute_attributes(np.ones((100, 100)))
        attrs.update(
            dominant_region="upper_right", quantile_contrast=0.43749,
            power_grid_3x3=[[0.08, 0.10, 0.22], [0.06, 0.14, 0.12], [0.07, 0.09, 0.12]],
            brightest_power_share=0.28749, texture_gradient_mean=0.08349,
        )
        prompt = make_prompt(attrs)
        self.assertIn("upper-right", prompt)
        self.assertIn("Amplitude contrast: 0.437.", prompt)
        self.assertIn("[0.080, 0.100, 0.220, 0.060, 0.140, 0.120, 0.070, 0.090, 0.120]", prompt)
        self.assertIn("Brightest 1.000% pixel power share: 0.287.", prompt)
        self.assertIn("Local amplitude variation: 0.083.", prompt)
        self.assertEqual(attrs["quantile_contrast"], 0.43749)

    def test_prompt_reports_actual_brightest_pixel_fraction_after_rounding_count(self):
        attrs = compute_attributes(np.ones((3, 3)))
        self.assertIn("Brightest 11.111% pixel power share: 0.111.", make_prompt(attrs))

    def test_same_coarse_categories_can_produce_different_numeric_prompts(self):
        amplitude = np.tile(np.linspace(0.1, 1, 60), (60, 1))
        first = compute_attributes(amplitude)
        second = compute_attributes(amplitude ** 1.1)
        for key in ("dominant_region", "contrast_level", "concentration_level", "texture_level"):
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(make_prompt(first), make_prompt(second))

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
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(set(result), {"source", "target", "prompt"})
        self.assertEqual(set(metadata), {"source", "target", "image_attributes"})
        for key in ("source", "target"):
            self.assertEqual(result[key], self.records[0][key])
            self.assertEqual(metadata[key], result[key])
        self.assertEqual(metadata["image_attributes"]["dominant_region"], "upper_right")
        self.assertFalse(metadata["image_attributes"]["uses_sarmae"])
        self.assertTrue(result["prompt"].startswith("Echo to image. "))
        self.assertEqual(self.manifest.read_bytes(), original_bytes)
        self.assertTrue((previews / "000000_sample.png").is_file())

    def test_refuses_to_overwrite_original_input(self):
        original_bytes = self.manifest.read_bytes()
        with self.assertRaises(ValueError):
            generate_dataset(self.manifest, self.root, self.manifest)
        self.assertEqual(self.manifest.read_bytes(), original_bytes)

    def test_metadata_cannot_overwrite_input_or_prompt_output(self):
        output = self.root / "generated.json"
        for metadata in (self.manifest, output):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    generate_dataset(self.manifest, self.root, output, metadata_json=metadata)
                self.assertFalse(output.exists())

    def test_existing_output_is_overwritten_with_updated_prompt(self):
        output = self.root / "existing.json"
        metadata = self.root / "metadata.json"
        output.write_text(json.dumps(self.records), encoding="utf-8")
        metadata.write_text("previous metadata", encoding="utf-8")
        generate_dataset(self.manifest, self.root, output)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn("Amplitude contrast:", result[0]["prompt"])
        self.assertEqual(result[0]["source"], self.records[0]["source"])
        self.assertEqual(result[0]["target"], self.records[0]["target"])
        self.assertNotIn("image_attributes", result[0])
        details = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(details[0]["source"], result[0]["source"])
        self.assertIn("image_attributes", details[0])
        self.assertNotIn("prompt", details[0])

    def test_failed_generation_preserves_existing_output(self):
        output = self.root / "existing.json"
        metadata = self.root / "metadata.json"
        output.write_text("previous output", encoding="utf-8")
        metadata.write_text("previous metadata", encoding="utf-8")
        self.manifest.write_text(json.dumps([
            self.records[0], {"source": "echo/missing.mat", "target": "missing.mat"},
        ]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Record 1"):
            generate_dataset(self.manifest, self.root, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous output")
        self.assertEqual(metadata.read_text(encoding="utf-8"), "previous metadata")

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
                self.assertFalse((self.root / "metadata.json").exists())

    def test_limit_outputs_only_selected_records_in_original_order(self):
        records = [dict(self.records[0], source=f"echo/sample_{i}.mat") for i in range(3)]
        self.manifest.write_text(json.dumps(records), encoding="utf-8")
        output = self.root / "subset.json"
        generate_dataset(self.manifest, self.root, output, limit=2)
        results = json.loads(output.read_text(encoding="utf-8"))
        expected = ["echo/sample_0.mat", "echo/sample_1.mat"]
        self.assertEqual([row["source"] for row in results], expected)
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual([row["source"] for row in metadata], expected)


class DirectoryPairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.echo_dir = self.root / "echo_5_new"
        self.image_dir = self.root / "image_5_new"
        self.echo_dir.mkdir()
        self.image_dir.mkdir()

    def add_pair(self, name):
        # Echo content is deliberately not a MAT file: only its paired path is used.
        (self.echo_dir / name).touch()
        write_target(self.image_dir / name, np.ones((12, 12)))

    def test_matching_is_sorted_and_paths_are_relative_to_data_root(self):
        self.add_pair("b.mat")
        self.add_pair("a.mat")
        (self.echo_dir / "notes.txt").write_text("not a sample", encoding="utf-8")
        self.assertEqual(build_pairs(self.echo_dir, self.image_dir, self.root), [
            {"source": "echo_5_new/a.mat", "target": "image_5_new/a.mat"},
            {"source": "echo_5_new/b.mat", "target": "image_5_new/b.mat"},
        ])

    def test_cli_generates_prompts_from_directories_without_input_json(self):
        self.add_pair("b.mat")
        self.add_pair("a.mat")
        output = self.root / "outputs/dataset_pair.json"
        script = Path(__file__).resolve().parents[1] / "tools/target_statistics.py"
        result = subprocess.run([
            sys.executable, str(script), "--data-root", str(self.root),
            "--output-json", str(output),
        ], check=True, capture_output=True, text=True)
        records = json.loads(output.read_text(encoding="utf-8"))
        metadata = json.loads((output.parent / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("Wrote 2 records", result.stdout)
        self.assertEqual([row["source"] for row in records], [
            "echo_5_new/a.mat", "echo_5_new/b.mat",
        ])
        self.assertTrue(all(row["prompt"].startswith("Echo to image.") for row in records))
        self.assertTrue(all(set(row) == {"source", "target", "prompt"} for row in records))
        self.assertEqual([(row["source"], row["target"]) for row in metadata],
                         [(row["source"], row["target"]) for row in records])
        self.assertTrue(all(row["image_attributes"]["baseline"] == "target_statistics_v1"
                            for row in metadata))

    def test_explicit_directories_and_pilot_limit(self):
        self.add_pair("a.mat")
        self.add_pair("b.mat")
        output = self.root / "pilot.json"
        metadata = self.root / "audit/metadata_pilot.json"
        generate_dataset(data_root=self.root, output_json=output,
                         echo_dir=self.echo_dir, image_dir=self.image_dir, limit=1,
                         metadata_json=metadata)
        records = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target"], "image_5_new/a.mat")
        details = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["target"], records[0]["target"])
        self.assertFalse((self.root / "metadata.json").exists())

    def test_unmatched_files_on_either_side_are_reported_without_output(self):
        self.add_pair("a.mat")
        (self.echo_dir / "echo_only.mat").touch()
        write_target(self.image_dir / "image_only.mat", np.ones((12, 12)))
        output = self.root / "failed.json"
        with self.assertRaisesRegex(ValueError, "missing images=1.*missing echoes=1"):
            generate_dataset(data_root=self.root, output_json=output)
        self.assertFalse(output.exists())
        self.assertFalse((self.root / "metadata.json").exists())

    def test_empty_and_missing_directories_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No matching .mat pairs"):
            build_pairs(self.echo_dir, self.image_dir, self.root)
        with self.assertRaisesRegex(ValueError, "directory does not exist"):
            build_pairs(self.root / "missing", self.image_dir, self.root)

    def test_existing_manifest_cannot_be_combined_with_directory_mode(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            generate_dataset(input_json=self.root / "pairs.json", data_root=self.root,
                             output_json=self.root / "out.json", echo_dir=self.echo_dir)


if __name__ == "__main__":
    unittest.main()
