# Crosswalk Detector

Semester project for detecting pedestrian crossings in Swiss aerial imagery.

Live dataset explorer: https://crosswalk-detector.vercel.app

## What This Project Does

The project trains and evaluates a custom deep-learning model that detects crosswalks in small aerial image tiles.

The current model is **CrossMaskNet**, a compact U-Net-style CNN segmentation model written in PyTorch and trained from scratch. It predicts a crosswalk mask first, then derives the final `crosswalk` / `no_crosswalk` classification from the mask coverage.

SAM3 is used only to bootstrap pseudo-labels for the dataset. SAM3 is not the final model and is not used during CrossMaskNet inference.

## Current State

- Model: `CrossMaskNet v4`
- Architecture: custom CNN encoder-decoder with skip connections
- Input: RGB image plus road-context channel
- Output: one-channel crosswalk segmentation mask
- Classification rule: mask coverage above threshold means `crosswalk`
- Static review frontend: deployed on Vercel
- Public URL: https://crosswalk-detector.vercel.app

The latest measured test results for CrossMaskNet v4 are:

| Metric | Value |
|---|---:|
| Accuracy | `0.961165` |
| Precision | `0.957746` |
| Recall | `0.966825` |
| Positive Dice | `0.794408` |
| Positive IoU | `0.658937` |

## Repository Structure

- `src/crosswalk_detector/`: Python training, inference, dataset, and pipeline code
- `web/`: dataset explorer and review frontend
- `scripts/`: local dataset export, review export, and batch-processing helpers
- `docs/`: project report draft, dataset notes, and design decisions
- `tests/`: automated checks
- `configs/`: local run configuration files
- `datasets/`, `results/`, `review-exports/`, `logs/`: local generated artifacts
- `.local/release/crossmasknet-v4/`: local release copy of the current model checkpoint and metrics

Large datasets, generated masks, logs, and model checkpoints should normally stay out of Git. The compressed static dataset metadata used by the deployed frontend lives under `web/public/static-datasets/`.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/DotNaos/fs26-crosswalk-detector.git
cd fs26-crosswalk-detector
```

Using `uv` (recommended):

```bash
uv sync
uv run download-images --count 20 --positive-ratio 0.5 --output-dir data/input/crossmask-images
uv run test --input-dir data/input/crossmask-images --output-dir data/predictions/my-run --positive-threshold 0.005
uv run dataset
uv run train
uv run test
```

Using Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m crosswalk_detector.workflow download-images --count 20 --positive-ratio 0.5 --output-dir data/input/crossmask-images
python -m crosswalk_detector.workflow test --input-dir data/input/crossmask-images --output-dir data/predictions/my-run --positive-threshold 0.005
python -m crosswalk_detector.workflow dataset
python -m crosswalk_detector.workflow train
python -m crosswalk_detector.workflow test
```

Common options:

```bash
uv run train --epochs 8 --batch-size 64
uv run dataset --skip-raw-cache --positive-limit 20 --negative-ratio 1 \
  --export /tmp/crosswalk-dataset-smoke --rebuild-export
uv run train --skip-raw-cache --positive-limit 20 --negative-ratio 1 \
  --epochs 1 --batch-size 2 \
  --export /tmp/crosswalk-train-smoke \
  --model-output /tmp/crosswalk-model-smoke \
  --rebuild-export
uv run test --model-root models/crossmask/sam3-500k-road-channel-v4
uv run test --input-dir path/to/images --output-dir data/predictions/my-run --positive-threshold 0.005
uv run download-images --count 20 --positive-ratio 0.5 --output-dir data/input/demo-images
```

The normal workflow commands are:

| Command | What it does |
|---|---|
| `uv run download-images` | Downloads a small folder of example input images that can be passed to `test --input-dir`. |
| `uv run dataset` | Downloads the public dataset metadata if needed, downloads the required raw aerial scenes, and prepares the local training images and masks. |
| `uv run train` | Ensures the dataset exists, then trains CrossMaskNet and writes a checkpoint plus metrics. |
| `uv run test` | Downloads the public checkpoint if needed and prints the stored test metrics for the default model. With `--input-dir`, it classifies new images into output folders. |

SAM3 is only needed if you want to run the labeling pipeline yourself. The
normal `dataset`, `train`, and `test` commands use the released labels and do
not require SAM3.

All options are optional. If you omit them, the commands use these defaults:

| Option | Default | What it means |
|---|---|---|
| `--profile` | `default` | Selects the prepared project configuration. |
| `--dataset` | `web/public/static-datasets/sam3-500k-masks-v1` | Metadata dataset location for `dataset` and `train`. |
| `--export` | `data/processed/crossmask/sam3-500k-road-channel-v4` | Where `dataset` writes prepared training images and masks, and where `train` reads them. |
| `--model-output` | `models/crossmask/sam3-500k-road-channel-v4` | Where `train` writes model checkpoints and metrics. |
| `--model-root` | `models/crossmask/sam3-500k-road-channel-v4` | Where `test` reads model checkpoints and metrics. |
| `--input-dir` | not set | Folder of new images for `test` to classify. Supports common image files such as JPG and PNG. |
| `--output-dir` | `data/predictions/crossmask-test` | Where `test --input-dir` writes classified images, overlays, and summary files. |
| `--positive-threshold` | `0.005` | Minimum mask coverage for `test --input-dir` to classify an image as `positive`. Images below this value go to `negative/`. Lower values put more images in `positive/`; higher values put more images in `negative/`. |
| `--no-overlays` | off | Skips writing overlay images for positive predictions. |
| `--count` | `20` | Total number of example images written by `download-images`. |
| `--positive-ratio` | `0.5` | Share of downloaded examples that come from positive source labels. `0.5` with `--count 20` means 10 positive and 10 negative examples. |
| `--positive-count` / `--negative-count` | not set | Explicit class counts for `download-images` if you do not want to use `--count` and `--positive-ratio`. |
| `--epochs` | `8` | Number of full passes over the training data. `8` means the model sees the prepared training set eight times. |
| `--batch-size` | `64` | Number of image/mask pairs processed in one training step. Use `16` or `32` if the machine runs out of memory. |
| `--image-size` | `128` | Training crop size. `128` means each crop is resized to 128 x 128 pixels. |
| `--limit-scenes` | no limit | Limits the raw scene prefetch step. The export can still download a missing scene later if it needs it. |
| `--positive-limit` | `2500` | Maximum number of crosswalk examples used when preparing the local training export. |
| `--negative-ratio` | `1.0` | Number of no-crosswalk examples per crosswalk example. `1.0` means a balanced export. |
| `--min-confidence` | `0.4` | Minimum released label confidence accepted for training examples. |
| `--min-mask-coverage` | `0.01` | Minimum mask area required for a positive crosswalk example. |
| `--workers` | `4` | Parallel download workers for raw aerial scenes. |
| `--num-workers` | `2` | Data-loading workers used while training. |
| `--learning-rate` | `0.001` | Training step size for the optimizer. |
| `--base-channels` | `24` | Width of the CrossMaskNet feature layers. Larger values make the model heavier. |
| `--road-channel` | off | Adds a road-context input channel during training. |
| `--skip-raw-cache` | off | Skips the full scene prefetch and downloads only scenes needed while building the export. |
| `--rebuild-export` | off | Recreates the prepared dataset export even if it already exists. |
| `--seed` | `7` | Keeps the train/test split and sampling repeatable. |

The raw image cache is written to `data/raw/sam3-500k-masks-v1/wms-mosaics/`.

`uv run dataset` restores the public dataset metadata if needed, caches the raw
swisstopo scene images locally, and prepares the training export.

`uv run train` reuses the prepared export when it already exists. If it is
missing, `train` prepares it first and then trains CrossMaskNet.

`uv run test` restores the public model checkpoint if needed and prints the
stored held-out test metrics for the default model.

To classify new image files, put them in one folder and run:

```bash
uv run test --input-dir path/to/images --output-dir data/predictions/my-run --positive-threshold 0.005
```

`test` copies input images into the output folders. It does not move or delete
the original files in `--input-dir`.

If you need a ready-made input folder first, run:

```bash
uv run download-images --count 20 --positive-ratio 0.5 --output-dir data/input/crossmask-images
```

Then classify those images:

```bash
uv run test --input-dir data/input/crossmask-images --output-dir data/predictions/my-run --positive-threshold 0.005
```

This writes:

| Output | Content |
|---|---|
| `positive/` | Images classified as crosswalk. |
| `negative/` | Images classified as no-crosswalk. |
| `positive_overlays/` | Positive images with the predicted mask drawn on top. |
| `summary.json` | Full machine-readable run summary. |
| `predictions.csv` | Compact table of image paths, decisions, and scores. |

Useful training options:

```bash
uv run train --epochs 8 --batch-size 64
uv run train --skip-raw-cache --positive-limit 20 --negative-ratio 1 \
  --epochs 1 --batch-size 2 \
  --export /tmp/crosswalk-train-smoke \
  --model-output /tmp/crosswalk-model-smoke \
  --rebuild-export
```

The second command is a small smoke run. It checks the download, dataset export,
and training path without building the full default dataset.

All generated data stays in local ignored folders such as `data/` and `models/`.
The raw aerial images are not committed to Git.

## Local Development

Python dependencies are managed with `uv`.

```bash
uv sync
uv run --with pytest python -m pytest
```

The frontend lives in `web/` and uses Bun.

```bash
cd web
bun install
bun run dev
```

For a static-only frontend build, use:

```bash
cd web
VITE_CROSSWALK_STATIC_ONLY=1 bun run build
```

## Download Project Assets

The larger dataset metadata package and the trained model checkpoint are stored
as GitHub Release assets, not in git.

```bash
python3 scripts/download_submission_assets.py
```

This downloads:

- static 500k dataset metadata to `web/public/static-datasets/`;
- CrossMaskNet v4 checkpoint and metrics to
  `models/crossmask/sam3-500k-road-channel-v4/`.

The raw aerial images are intentionally not stored in Git or in the release
archive. They are downloaded reproducibly from swisstopo when a training export
needs them. To prefetch the raw scene cache before preparing or testing a
training run:

```bash
uv run crosswalk-pipeline download-raw-scenes \
  --dataset web/public/static-datasets/sam3-500k-masks-v1
```

For the 500k metadata dataset this caches 489 source scenes under
`data/raw/sam3-500k-masks-v1/wms-mosaics/`. A five-scene measurement on the
current WMS source estimates this cache at about 641 MiB. Storing the same
coverage as 500k separate 25m image requests would be much larger, around
7.7 GiB from the same sample.

Release links:

- Dataset metadata: https://github.com/DotNaos/fs26-crosswalk-detector/releases/tag/submission-dataset-v1
- Model checkpoint: https://github.com/DotNaos/fs26-crosswalk-detector/releases/tag/crossmasknet-v4

## Vercel Deployment

The production frontend is deployed from this repository to:

https://crosswalk-detector.vercel.app

The root `vercel.json` is required because the web app lives in `web/`, while the repository root contains the project as a whole. It tells Vercel to:

- install dependencies inside `web/`;
- build the static frontend with `VITE_CROSSWALK_STATIC_ONLY=1`;
- publish `web/dist`;
- rewrite routes back to `index.html` for the single-page app.

## Main Documentation

- Project report draft: `docs/project-report-draft.md`
- Project decisions: `docs/project-decisions.md`
- Static dataset deployment: `docs/static-dataset-deployment.md`
- SAM3 dataset data model: `docs/sam3-100k-data-model.md`
- SAM3 runbook: `docs/sam3-100k-runbook.md`
