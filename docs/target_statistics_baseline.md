# Target statistics prompt baseline

## Scope and data flow

This independent CPU tool pairs echo and image files, loads each **target** MATLAB
7.3 image, computes descriptive statistics, and writes one `prompt` and
`image_attributes` entry per pair to the project file `outputs/dataset_pair.json`.
It does not load echo contents, SARMAE, model weights, or an LLM.
These are target-derived training annotations; generation from echo at inference
is outside this baseline. Downstream consumption of JSON prompts is not yet verified.

The confirmed data root is
`/data/dyn/capella/output/dataset_hkk_pfa_512`. Dataset `pfa_patch` is a
`512x512` compound array with float32 `real` and `imag` fields. Read amplitude
as `hypot(real, imag)` in float64. The original files are opened read-only.
The CLI accepts another HDF5 dataset path via `--mat-key`.

Default directories:

- Echo: `/data/dyn/capella/output/dataset_hkk_pfa_512/echo_5_new`
- Image: `/data/dyn/capella/output/dataset_hkk_pfa_512/image_5_new`

Pair direct-child `*.mat` files by exact filename, then sort by filename. Missing
counterparts on either side stop the run with counts and example names, even for
a pilot. Nested directories are not scanned. JSON paths remain relative to the
data root, for example `echo_5_new/x_+00550_y_-00250.mat` and
`image_5_new/x_+00550_y_-00250.mat`. Both directories must be inside the data root.
Override them with `--data-root`, `--echo-dir`, and `--image-dir` if needed.

Existing manifests remain supported through `--input-json`; that mode preserves
all other fields and the original record order. It cannot be combined with
`--echo-dir` or `--image-dir`. No input JSON is needed for the default workflow.

## Coordinate convention

Default: preserve the HDF5 array exactly, with row zero at the top and column
zero at the left. These labels are **image-array positions**, not geographic
directions. Orientation relative to the unavailable downstream loader remains
unverified. Optional transforms run in this order: `--transpose`,
`--flip-vertical`, `--flip-horizontal`. All transforms and the stored shape are
recorded in `image_attributes.orientation`; previews use the same orientation.

## Attributes and initial rules

Let `A = abs(z)` and `U = A/max(A)` for a nonzero image. Use `P = U**2` for
relative power fractions; these match fractions from `A**2` and are not calibrated
backscatter or physical radiometry. Global positive gain does not change the
relative attributes or prompt. Raw amplitude quantiles and peak are also retained.

| Attribute | Definition | Initial attribute-level cutoffs |
|---|---|---|
| Contrast | `(p90(U)-p10(U))/(p90(U)+p10(U))`; zero if denominator is zero | Low `<0.25`, high `>=0.60`, otherwise moderate |
| Spatial distribution | Split rows/columns into 3 nearly equal groups; sum `P` in each cell and divide by total | Name the largest cell only if its share is `>=0.20` and `>=1.25` times the runner-up; its power per pixel must also exceed every other cell by that ratio |
| Concentration | Power in exactly `ceil(0.01*N)` brightest pixels divided by total power | Low `<0.10`, high `>=0.25`, otherwise moderate |
| Local variation | Clip `(U-p01)/(p99-p01)` to `[0,1]`; average mean absolute adjacent differences along both axes | Low `<0.05`, high `>=0.15`, otherwise moderate |

If `p99 == p01`, use `U` for local variation. Percentile contrast describes the
central amplitude distribution: a sparse bright outlier can coexist with low
central contrast. Spatial dominance and pixel concentration are separate measures.
Tied regions are not arbitrarily named. Uniform images have no dominant region;
zero images receive an explicit zero-amplitude prompt. Non-finite, negative,
non-2-D, or smaller-than-3x3 amplitude inputs are rejected.

All cutoffs are **uncalibrated initial rules**, not established SAR thresholds.
They are exposed as CLI options and recorded per sample under `rules`.
Gradient statistics describe local variation, not object classes, texture causes,
or physical speckle parameters. No directional-structure claims are generated.

## Numerical prompt format

For nonzero images, prompts combine a short spatial description with a fixed-order
numerical summary: quantile contrast, nine grid power shares in row-major order,
brightest-pixel power share, and mean local amplitude variation. All summary
numbers use three decimal places. `image_attributes` retains the unrounded
statistics and the categorical levels for inspection. The brightest-pixel
percentage uses the actual selected pixel count, including ceiling effects.

Illustrative values, not a measured sample:

```text
Echo to image. Relative image power is strongest in the upper-right region.
Amplitude contrast: 0.437.
Power distribution on the 3x3 grid, from top-left to bottom-right:
[0.080, 0.100, 0.220, 0.060, 0.140, 0.120, 0.070, 0.090, 0.120].
Brightest 1.000% pixel power share: 0.287.
Local amplitude variation: 0.083.
```

The JSON prompt is a single line. Rounding can make the displayed grid shares
sum to slightly above or below one. Numerical summaries preserve differences
within categorical bins but do not guarantee unique prompts. Zero-amplitude
images retain the dedicated zero-amplitude prompt. Check the downstream text
encoder's length limit before using these prompts for model training.

## Run on the Linux server

Run these commands from the server repository checkout, not the local Windows machine.
First synchronize the `main` branch and inspect the commit being tested:

```bash
git pull --ff-only origin main
git rev-parse HEAD
conda activate my_py310
bash run/01_setup.sh
bash run/02_test.sh
```

If this is the first server checkout, clone the user's repository first:

```bash
cd /data/dyn
git -c core.sshCommand="ssh -i /data/dyn/.ssh/id_ed25519_sarmae -o IdentitiesOnly=yes" \
  clone git@github.com:allforkarina/SARMAE_ItoP.git
cd SARMAE_ItoP
git config --local core.sshCommand \
  "ssh -i /data/dyn/.ssh/id_ed25519_sarmae -o IdentitiesOnly=yes"
```

The dependency file matches the confirmed server versions: NumPy 2.2.6,
h5py 3.16.0, and Pillow 12.0.0. SciPy is not used for these MATLAB 7.3 files.
No GPU or PyTorch dependency is needed for this baseline.
The scripts locate the project root themselves, so they can also be invoked by
absolute path from another working directory. They use the active `python`;
set `PYTHON_BIN=/home/dyn/.conda/envs/my_py310/bin/python` to select it explicitly.

### Pilot: first 12 pairs

`--limit` takes the first N sorted pairs, not a random or representative sample.
Choose diverse records for subsequent review if these are adjacent patches.

```bash
bash run/03_pilot.sh
```

The pilot writes `outputs/dataset_pair_pilot.json` and PNGs under
`outputs/previews/`. It contains only the selected pairs. Each PNG shows relative
amplitude in `[-60,0]` dB, grid-cell power shares, and the generated prompt.
Statistics use linear amplitude/power, not the display transform.

### Full dataset: after pilot review

```bash
bash run/04_generate_dataset.sh
```

This writes all pairs to **`outputs/dataset_pair.json`**. The project `outputs/`
directory is ignored by Git. Additional CLI options can be appended to either
generation script, for example `bash run/03_pilot.sh --limit 24` or
`bash run/04_generate_dataset.sh --output-json /tmp/dataset_pair_v2.json`.
Direct invocation `python tools/target_statistics.py` uses the same full-dataset
defaults. For an existing manifest, use `--input-json /path/to/pairs.json`.

Existing output JSON files are overwritten on successful regeneration, including
`outputs/dataset_pair.json` and the pilot JSON. An input manifest cannot also be
the output path. Read/schema failures
stop the run with the record index and target path; failed samples are not replaced
or silently dropped. No output JSON is opened until all selected records have
been processed and serialized, so earlier processing failures preserve existing
output content. Interruption during the final disk write can still leave an
incomplete output; rerun generation to replace it.
Previews already written before a failure remain available for inspection.
Pair lists and output records are held in memory; MAT images are read one at a time.

## Validation and user calibration

Synthetic server tests cover isolated/tied scatterers, uniform and zero images,
gain invariance, local variation, compound complex reading, explicit orientation,
JSON preservation, preview creation, directory pairing, CLI execution, unmatched
file reporting, numerical prompt formatting, output replacement, and preservation
of previous output on processing failure. `run/02_test.sh` also checks Bash syntax
for all run scripts. These checks do not establish
that the cutoffs are suitable for the real dataset or that prompts improve reconstruction.

For the pilot, check count/order, unchanged source/target values, plausible grid
shares summing to one for nonzero images, and agreement between each prompt and
its displayed target. Compare several smooth, dense, and concentrated samples.
As a hands-on calibration step, change one cutoff using a CLI option, for example
`--dominant-min-share 0.30`, write to a new output path, and inspect which regional
claims disappear and why. The ratio-to-runner-up rule still applies.

Status: implementation prepared; local execution and tests intentionally not run.
Server synthetic tests, real-image visual review, user calibration, and downstream
effect evaluation are pending. Synchronization to GitHub is a code handoff, not
evidence that the server or model has passed validation.
