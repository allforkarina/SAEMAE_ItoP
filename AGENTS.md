# Repository Guidelines

## Project Structure & Module Organization

- `SARMAE_Pretrain/`: pretraining and classification entry points, training engines, and `models_vit.py`.
- `SARMAE_Pretrain/models/`: SAR MAE backbone and SAR–optical patch alignment. `dataset/` handles mixed samples and transforms; `util/` contains noise, checkpoint, and training helpers.
- `SARMAE_Fintune/Detection/` and `SARMAE_Fintune/Segmentation/`: OpenMMLab backbone adaptations, datasets, and task configurations. These are integration files, not complete framework checkouts.
- `Figs/`: README illustrations.
- `tools/target_statistics.py`: independent MATLAB 7.3 target-statistics prompt baseline; see `docs/target_statistics_baseline.md` for formulas and server commands.
- `tools/cphd_parameters.py`: standard-library CPHD 1.0/1.1 XML parameter extraction and product-context rendering; `tests/test_cphd_parameters.py` covers synthetic CPHD integration.
- `tests/test_target_statistics.py`: synthetic baseline tests using standard-library `unittest`; run on the server only.
- `run/`: Bash entry points for dependency setup, tests, pilot previews, and full JSON generation. `outputs/` holds generated artifacts and is ignored by Git.

## Local–GitHub–Server Workflow

1. Read and edit code in the local Windows checkout, `D:\Files\Projects\PythonProjects\SARMAE`. Local file inspection and Git operations are allowed; all project execution, dependency installation, syntax checks, tests, benchmarks, and model-effect validation belong on the Linux server.
2. After substantial project changes, create a focused Git commit and push it to the user's confirmed GitHub repository and branch. This synchronization is a standing requirement; do not stop at preparing a diff. Stage only relevant files and exclude datasets, checkpoints, generated features, credentials, and unrelated user changes.
3. Provide the branch, commit hash, and server commands. The user synchronizes the server checkout through GitHub, then runs validation in `my_py310`. Use `git pull --ff-only` on the matching branch and `git rev-parse HEAD` to identify the revision being tested.
4. Review the user's server output, adjust locally, and repeat the commit/push/pull cycle. A successful push means GitHub synchronization only; server synchronization and validation require their own evidence. Label unexecuted checks as pending.

The current local branch is `main`; `origin` points to the user's confirmed GitHub repository, `git@github.com:allforkarina/SARMAE_ItoP.git`. Use this repository for commit/push synchronization. The original source repository is `git@github.com:MiliLab/SARMAE.git`. The server checkout path remains unconfirmed; a new clone under `/data/dyn` defaults to `/data/dyn/SARMAE_ItoP`. Use the actual server location when providing synchronization commands.

The server's dedicated SSH key is `/data/dyn/.ssh/id_ed25519_sarmae`. Use it explicitly for cloning and configure `core.sshCommand` locally in the server checkout for subsequent Git operations; do not change global SSH settings or commit key material.

The baseline pairs same-named `.mat` files directly under `/data/dyn/capella/output/dataset_hkk_pfa_512/echo_5_new` and `image_5_new`. It writes aligned project-local JSON lists: `outputs/dataset_pair.json` contains only `source`, `target`, `prompt`; `outputs/metadata.json` contains only `source`, `target`, `image_attributes`. Retain data-root-relative paths and identical pair order. Existing manifests remain an optional input; their additional fields are not copied to outputs.

Prompts combine a spatial description with contrast, nine grid shares, brightest-pixel power share, and local variation formatted to three decimal places. Preserve unrounded `image_attributes` in metadata only. Both output JSON files may be overwritten after successful processing; never overwrite an input manifest. Reuse existing `run/` scripts; pilot files are `dataset_pair_pilot.json` and `metadata_pilot.json` to keep them separate from full outputs.

With `--cphd-file`, append one acquisition paragraph to every patch prompt and save structured parameters to `outputs/cphd_parameters.json`, without a separate prompt field. At the user's request, `run/03_pilot.sh` and `run/04_generate_dataset.sh` hardcode `/data/dyn/capella/data/CAPELLA_C02_SP_CPHD_HH_20201201165018_20201201165040.cphd`; invoke them without CPHD arguments. This version reads only the selected product and shares its parameters across all pairs; equality with the other products and patch-to-product correspondence remain unverified. Require `--cphd-channel` for multiple channels. Preserve Hz/seconds/meters/degrees in parameter field names; distinguish saved bandwidth from transmitted RF bandwidth and product reference geometry from patch geometry. Do not infer PRF or image resolution. No new dependencies are needed. The pilot script fixes `outputs/cphd_parameters_pilot.json`; full generation fixes `outputs/cphd_parameters.json`. All output paths must differ from each other and original inputs. Validation stays on the server.

## Server Configuration Reference

Read this section whenever configuration or dependency decisions arise. Use this server baseline rather than the local Python environment. Treat README versions as original reproduction settings; target the confirmed server environment unless the user approves a different target. Update this reference when new server output establishes a change.

Confirmed from user-provided output on **2026-10-22**:

| Component | Configuration |
|---|---|
| Host / architecture | `dyn@kb304`, Linux `x86_64` |
| OS / kernel | Ubuntu 22.04.5 LTS / `6.8.0-138-generic` |
| glibc / GCC / G++ | 2.35 / 11.4.0 / 11.4.0 |
| CPU | 2 × AMD EPYC 9V74, 160 physical cores, 320 logical threads, 2 NUMA nodes |
| RAM / swap | 503 GiB / 2 GiB |
| GPU | 4 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM / compute capability | 97,887 MiB (94.97 GiB) per GPU / 12.0 |
| NVIDIA driver | `580.178.04`; `nvidia-smi` reports CUDA 13.0 |
| CUDA compiler | `nvcc` not found in PATH; toolkit installation is unconfirmed |
| Conda / environment | 25.11.1 / `my_py310` |
| Python executable | `/home/dyn/.conda/envs/my_py310/bin/python` |
| Python / pip | 3.10.19 / 25.3 |
| PyTorch / torchvision / torchaudio | `2.9.0+cu130` / `0.24.0+cu130` / `2.9.0+cu130` |
| PyTorch CUDA / cuDNN | 13.0 / reported version `91300` |
| GPU detection | CUDA available; 4 visible GPUs; compiled architectures include `sm_120` |
| NumPy / Pillow | 2.2.6 / 12.0.0 |
| Dependency consistency | `python -m pip check` passed in the captured environment |

Not installed in that environment: `timm`, `scipy`, `opencv-python`, `mmcv`, `mmcv-full`, `mmcv-lite`, `mmengine`, `mmdet`, `mmrotate`, `mmsegmentation`, `transformers`, `huggingface-hub`, `safetensors`, `einops`, `matplotlib`, and `tensorboard`. Select and document dependencies for the actual task; confirm backbone/checkpoint compatibility instead of relying on the missing-`timm` fallback.

Subsequent server inspection confirmed `h5py==3.16.0` and a working SciPy import (version not recorded). Targets are MATLAB 7.3 HDF5 files: dataset `pfa_patch`, shape `(512, 512)`, compound float32 `real`/`imag` fields. The statistics baseline uses NumPy, h5py, and Pillow only; its dependency file is `requirements-target-statistics.txt`. Default orientation preserves the HDF5 array with no transpose or flips; geographic orientation and alignment with the downstream loader remain unverified.

Resource snapshot, not a reservation: `/data` was 7.3 T total, 94% used, with 492 G available; RAM available was 337 GiB; GPU utilization was 99–100%. `CUDA_VISIBLE_DEVICES` was unset. Recheck availability on the server before choosing GPUs or sizing feature exports. Driver CUDA support, PyTorch's CUDA runtime, and the CUDA compiler are separate configuration facts. Model execution has not yet been validated for this repository on the server.

## Build, Test, and Development Commands

Server only, after installing required dependencies:

```bash
conda activate my_py310
python -m pip check
cd SARMAE_Pretrain
python train_mae_contrastive.py --help
python main_finetune.py --help
```

`pip check` checks declared dependency consistency. The entry points expose pretraining and classification options. Follow README launch examples with actual dataset and checkpoint paths. Detection and segmentation commands run inside their corresponding OpenMMLab installations. No repository-wide build command or training dependency manifest is provided.

For the independent statistics baseline, run on the server from the repository root:

```bash
bash run/01_setup.sh
bash run/02_test.sh
bash run/03_pilot.sh
# After reviewing the pilot:
bash run/04_generate_dataset.sh
```

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `PascalCase` classes. Follow surrounding code and preserve framework registry names and checkpoint keys. Keep dataset-specific settings in task configurations. No formatter or linter configuration is currently provided.

## Testing Guidelines

The statistics baseline has `unittest` coverage under `tests/`; no repository-wide coverage threshold or model-training test suite is established. For behavioral changes, provide server commands and expected results. Check checkpoint-key matching, input normalization, tensor shapes, patch ordering, and finite outputs where relevant. Clearly label pending server checks.

## Commit & Pull Request Guidelines

History uses short descriptive subjects, such as `Update README.md`; no Conventional Commits requirement exists. Keep commits focused. PRs should explain the change, affected tasks, environment changes, validation commands and results, and related issues when applicable. Include visual evidence for changes to model outputs.

## Agent-Specific Instructions

Before project-building or behavior-changing implementation, use `grill-me` as the sole clarification gate for requirements, constraints, edge cases, and success criteria. Other task-specific skills may assist execution without adding clarification gates. Honor read-only review requests and avoid unrelated changes.
