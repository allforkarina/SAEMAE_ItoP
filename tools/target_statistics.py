"""Generate offline, target-derived statistical prompts from MATLAB 7.3 images."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import textwrap

import h5py
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data/dyn/capella/output/dataset_hkk_pfa_512")
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "dataset_pair.json"


REGIONS = (
    "upper_left", "upper_center", "upper_right",
    "middle_left", "center", "middle_right",
    "lower_left", "lower_center", "lower_right",
)


@dataclass(frozen=True)
class Rules:
    """Initial descriptive cutoffs, to be calibrated by sample inspection."""

    contrast_low: float = 0.25
    contrast_high: float = 0.60
    dominant_min_share: float = 0.20
    dominant_min_ratio: float = 1.25
    top_fraction: float = 0.01
    concentration_low: float = 0.10
    concentration_high: float = 0.25
    texture_low: float = 0.05
    texture_high: float = 0.15

    def validate(self):
        for low, high, name in (
            (self.contrast_low, self.contrast_high, "contrast"),
            (self.concentration_low, self.concentration_high, "concentration"),
            (self.texture_low, self.texture_high, "texture"),
        ):
            if not 0 <= low < high <= 1:
                raise ValueError(f"{name} cutoffs must satisfy 0 <= low < high <= 1")
        if not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        if not 0 < self.dominant_min_share <= 1:
            raise ValueError("dominant_min_share must be in (0, 1]")
        if not np.isfinite(self.dominant_min_ratio) or self.dominant_min_ratio <= 1:
            raise ValueError("dominant_min_ratio must be finite and greater than 1")


def load_amplitude(path, mat_key="pfa_patch", transpose=False,
                   flip_vertical=False, flip_horizontal=False):
    """Read the confirmed compound real/imag layout; do not infer orientation."""
    with h5py.File(path, "r") as handle:
        dataset = handle[mat_key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{mat_key!r} is not an HDF5 dataset")
        if dataset.ndim != 2 or min(dataset.shape) < 3:
            raise ValueError(f"Expected a 2-D image at least 3x3; got {dataset.shape}")
        raw = dataset[()]

    if not raw.dtype.names or not {"real", "imag"}.issubset(raw.dtype.names):
        raise ValueError(f"Expected compound real/imag fields; got {raw.dtype}")
    real = np.asarray(raw["real"], dtype=np.float64)
    imag = np.asarray(raw["imag"], dtype=np.float64)
    amplitude = np.hypot(real, imag)
    if not np.isfinite(amplitude).all():
        raise ValueError("Image contains non-finite amplitude values")
    stored_shape = list(amplitude.shape)
    if transpose:
        amplitude = amplitude.T
    if flip_vertical:
        amplitude = amplitude[::-1, :]
    if flip_horizontal:
        amplitude = amplitude[:, ::-1]
    orientation = {
        "basis": "hdf5_array",
        "transpose": transpose,
        "flip_vertical": flip_vertical,
        "flip_horizontal": flip_horizontal,
        "row_zero": "top",
        "column_zero": "left",
        "geographic_orientation": "unknown",
        "stored_shape": stored_shape,
    }
    return amplitude, orientation


def _level(value, low, high):
    return "low" if value < low else "high" if value >= high else "moderate"


def compute_attributes(amplitude, rules=None):
    rules = rules or Rules()
    rules.validate()
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2 or min(amplitude.shape) < 3:
        raise ValueError("Amplitude must be a 2-D image at least 3x3")
    if not np.isfinite(amplitude).all() or np.any(amplitude < 0):
        raise ValueError("Amplitude must be finite and nonnegative")

    peak = float(amplitude.max())
    normalized = amplitude / peak if peak > 0 else np.zeros_like(amplitude)
    percentiles = (1, 10, 50, 90, 99)
    quantiles = np.percentile(normalized, percentiles)
    q01, q10, _, q90, q99 = quantiles
    denominator = float(q90 + q10)
    contrast = float((q90 - q10) / denominator) if denominator > 0 else 0.0

    # Squaring after peak normalization avoids unnecessary overflow. Fractions
    # are identical to abs(z)**2 fractions, without implying calibrated power.
    power = normalized ** 2
    total_power = float(power.sum())
    row_groups = np.array_split(np.arange(amplitude.shape[0]), 3)
    col_groups = np.array_split(np.arange(amplitude.shape[1]), 3)
    grid = np.array([
        [power[np.ix_(rows, cols)].sum() for cols in col_groups]
        for rows in row_groups
    ], dtype=np.float64)
    grid = grid / total_power if total_power > 0 else np.zeros((3, 3))
    shares = grid.ravel()
    ranked = np.sort(shares)
    dominant_index = int(np.argmax(shares))
    dominant_share = float(shares[dominant_index])
    cell_sizes = np.outer([len(x) for x in row_groups], [len(x) for x in col_groups])
    densities = (grid / cell_sizes).ravel()
    # Unequal cell areas must not make a uniform image appear localized.
    other_density = float(np.delete(densities, dominant_index).max())
    distinct = (
        total_power > 0
        and dominant_share >= rules.dominant_min_share
        and dominant_share >= rules.dominant_min_ratio * ranked[-2]
        and densities[dominant_index] >= rules.dominant_min_ratio * other_density
    )

    count = max(1, int(np.ceil(power.size * rules.top_fraction)))
    brightest = np.partition(power.ravel(), power.size - count)[-count:]
    top_share = float(brightest.sum() / total_power) if total_power > 0 else 0.0
    concentration = _level(top_share, rules.concentration_low, rules.concentration_high)

    span = float(q99 - q01)
    texture_image = np.clip((normalized - q01) / span, 0, 1) if span > 0 else normalized
    texture = float(0.5 * (
        np.abs(np.diff(texture_image, axis=0)).mean()
        + np.abs(np.diff(texture_image, axis=1)).mean()
    ))

    return {
        "baseline": "target_statistics_v1",
        "information_source": "target_image",
        "uses_sarmae": False,
        "shape": list(amplitude.shape),
        "amplitude_quantiles": {
            f"p{p:02d}": float(q * peak) for p, q in zip(percentiles, quantiles)
        },
        "peak_amplitude": peak,
        "zero_amplitude": peak == 0,
        "quantile_contrast": contrast,
        "contrast_level": _level(contrast, rules.contrast_low, rules.contrast_high),
        "power_grid_3x3": grid.tolist(),
        "dominant_region": REGIONS[dominant_index] if distinct else "no_single_region",
        "largest_region_power_share": dominant_share,
        "brightest_pixel_count": count,
        "brightest_pixel_fraction": count / power.size,
        "brightest_power_share": top_share,
        "concentration_level": concentration if total_power > 0 else "undefined",
        "texture_gradient_mean": texture,
        "texture_level": _level(texture, rules.texture_low, rules.texture_high),
        "texture_normalization": "p01_p99_clipped" if span > 0 else "peak_amplitude",
        "rules": asdict(rules),
    }


def make_prompt(attributes):
    if attributes["zero_amplitude"]:
        return "Echo to image. The target image has zero amplitude throughout."
    region = attributes["dominant_region"]
    if region == "no_single_region":
        spatial_text = "No single region dominates the image power on a 3-by-3 grid."
    else:
        label = region.replace("_", "-")
        spatial_text = f"Relative image power is strongest in the {label} region."
    grid_text = ", ".join(
        f"{share:.3f}" for row in attributes["power_grid_3x3"] for share in row
    )
    brightest_percent = attributes["brightest_pixel_fraction"] * 100
    return " ".join((
        "Echo to image.",
        spatial_text,
        f"Amplitude contrast: {attributes['quantile_contrast']:.3f}.",
        "Power distribution on the 3x3 grid, from top-left to bottom-right: "
        f"[{grid_text}].",
        f"Brightest {brightest_percent:.3f}% pixel power share: "
        f"{attributes['brightest_power_share']:.3f}.",
        f"Local amplitude variation: {attributes['texture_gradient_mean']:.3f}.",
    ))


def save_preview(amplitude, attributes, prompt, path):
    """Render relative amplitude in [-60, 0] dB, with the same grid orientation."""
    peak = attributes["peak_amplitude"]
    if peak > 0:
        decibels = 20 * np.log10(np.maximum(amplitude / peak, 1e-3))
        pixels = np.round((decibels + 60) / 60 * 255).astype(np.uint8)
    else:
        pixels = np.zeros(amplitude.shape, dtype=np.uint8)
    image = Image.fromarray(pixels).convert("RGB")
    # Keep native aspect ratio; previews need not have the training input size.
    image.thumbnail((768, 768))
    width, height = image.size
    canvas_width = max(640, width)
    lines = textwrap.wrap(prompt, width=max(40, canvas_width // 7))
    canvas = Image.new("RGB", (canvas_width, height + 55 + 16 * len(lines)), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    row_edges = np.cumsum([0] + [len(x) for x in np.array_split(np.arange(amplitude.shape[0]), 3)])
    col_edges = np.cumsum([0] + [len(x) for x in np.array_split(np.arange(amplitude.shape[1]), 3)])
    ys = row_edges * height / amplitude.shape[0]
    xs = col_edges * width / amplitude.shape[1]
    for x in xs[1:-1]:
        draw.line((int(x), 0, int(x), height - 1), fill="yellow")
    for y in ys[1:-1]:
        draw.line((0, int(y), width - 1, int(y)), fill="yellow")
    for i, share in enumerate(np.asarray(attributes["power_grid_3x3"]).ravel()):
        row, col = divmod(i, 3)
        position = (int(xs[col]) + 3, int(ys[row]) + 3)
        label = f"{REGIONS[i]}: {share:.1%}"
        box = draw.textbbox(position, label)
        draw.rectangle(box, fill="black")
        draw.text(position, label, fill="yellow")
    draw.text((5, height + 5), "Target statistics; HDF5 array coordinates; display: -60 to 0 dB", fill="black")
    for i, line in enumerate(lines):
        draw.text((5, height + 30 + 16 * i), line, fill="black")
    canvas.save(path)


def build_pairs(echo_dir, image_dir, data_root):
    """Pair direct-child .mat files by exact filename in deterministic order."""
    echo_dir, image_dir, data_root = (
        Path(path).resolve() for path in (echo_dir, image_dir, data_root)
    )
    for directory in (echo_dir, image_dir):
        if not directory.is_dir():
            raise ValueError(f"Data directory does not exist: {directory}")
    sources = {path.name: path for path in echo_dir.glob("*.mat") if path.is_file()}
    targets = {path.name: path for path in image_dir.glob("*.mat") if path.is_file()}
    missing_images = sorted(sources.keys() - targets.keys())
    missing_echoes = sorted(targets.keys() - sources.keys())
    if missing_images or missing_echoes:
        raise ValueError(
            "Unmatched .mat files: "
            f"missing images={len(missing_images)} {missing_images[:5]}; "
            f"missing echoes={len(missing_echoes)} {missing_echoes[:5]}"
        )
    if not sources:
        raise ValueError("No matching .mat pairs found in the specified directories")
    return [
        {
            "source": sources[name].relative_to(data_root).as_posix(),
            "target": targets[name].relative_to(data_root).as_posix(),
        }
        for name in sorted(sources)
    ]


def generate_dataset(input_json=None, data_root=DEFAULT_DATA_ROOT,
                     output_json=DEFAULT_OUTPUT_JSON, rules=None, mat_key="pfa_patch",
                     limit=None, preview_dir=None, preview_count=12, transpose=False,
                     flip_vertical=False, flip_horizontal=False, echo_dir=None,
                     image_dir=None):
    """Preserve records and order. Errors stop the run; no placeholder samples."""
    rules = rules or Rules()
    rules.validate()
    data_root, output_json = Path(data_root), Path(output_json)
    input_json = Path(input_json) if input_json is not None else None
    if input_json is not None and (echo_dir is not None or image_dir is not None):
        raise ValueError("Use either input_json or echo/image directories, not both")
    if input_json is not None and output_json.resolve() == input_json.resolve():
        raise ValueError("Output must differ from the original input JSON")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if preview_count < 0:
        raise ValueError("preview_count must be nonnegative")
    if input_json is not None:
        with input_json.open(encoding="utf-8") as handle:
            records = json.load(handle)
    else:
        records = build_pairs(
            echo_dir if echo_dir is not None else data_root / "echo_5_new",
            image_dir if image_dir is not None else data_root / "image_5_new",
            data_root,
        )
        print(f"Matched {len(records)} echo/image pairs", file=sys.stderr)
    if not isinstance(records, list) or not records:
        raise ValueError("Input JSON must contain a nonempty list of data pairs")
    selected = records if limit is None else records[:limit]
    output = []
    preview_dir = Path(preview_dir) if preview_dir is not None else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(selected):
        target = record.get("target") if isinstance(record, dict) else None
        if not isinstance(target, str) or not target or not isinstance(record.get("source"), str):
            raise ValueError(f"Record {index} must contain string source and target paths")
        path = data_root / target
        try:
            amplitude, orientation = load_amplitude(
                path, mat_key, transpose, flip_vertical, flip_horizontal,
            )
            attributes = compute_attributes(amplitude, rules)
            attributes.update(mat_key=mat_key, orientation=orientation)
            prompt = make_prompt(attributes)
            result = dict(record)
            result.update(prompt=prompt, image_attributes=attributes)
            output.append(result)
            if preview_dir is not None and index < preview_count:
                save_preview(amplitude, attributes, prompt, preview_dir / f"{index:06d}_{path.stem}.png")
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Record {index}, target {target!r}: {error}") from error
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(selected)}", file=sys.stderr)

    # Process and serialize before opening the destination, preserving any
    # previous output if reading, computation, or serialization fails.
    text = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Wrote {len(output)} records to {output_json}")
    return len(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, help="Optional existing manifest; otherwise pair directories")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--echo-dir", type=Path, help="Default: DATA_ROOT/echo_5_new")
    parser.add_argument("--image-dir", type=Path, help="Default: DATA_ROOT/image_5_new")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--mat-key", default="pfa_patch")
    parser.add_argument("--limit", type=int, help="Process only the first N records; writes a subset JSON")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    for name, default in asdict(Rules()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args()
    options = vars(args).copy()
    rules = Rules(**{name: options.pop(name) for name in asdict(Rules())})
    try:
        generate_dataset(rules=rules, **options)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
