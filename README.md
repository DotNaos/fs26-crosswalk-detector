# Crosswalk Detector

Semester project for detecting pedestrian crossings in Swiss aerial imagery.

Live dataset explorer: https://crosswalk-detector.vercel.app

The dataset is provided with the project as public metadata, not as committed
raw aerial images. The raw Swisstopo images are downloaded reproducibly when
needed by `uv run dataset`, `uv run train`, or `uv run test`. The helper script
`scripts/download_submission_assets.py` restores the released dataset metadata
and model checkpoint into the expected local folders.

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

Requirements: `git` and `uv` must already be installed.

Copy and paste this into PowerShell, macOS Terminal, or a Linux terminal:

```bash
git clone https://github.com/DotNaos/fs26-crosswalk-detector.git
cd fs26-crosswalk-detector
uv sync
uv run test --count 20 --positive-ratio 0.5 --output-dir data/predictions/release-check --positive-threshold 0.005
```

This creates:

| Path | Content |
|---|---|
| `data/input/crossmask-images/` | 20 downloaded example images: 10 source-positive and 10 source-negative. |
| `data/predictions/release-check/positive/` | Images classified as crosswalk by the released checkpoint. |
| `data/predictions/release-check/negative/` | Images classified as no-crosswalk by the released checkpoint. |
| `data/predictions/release-check/positive_overlays/` | Positive images with the predicted mask drawn on top. |
| `data/predictions/release-check/summary.json` | Full machine-readable run summary. |
| `data/predictions/release-check/predictions.csv` | Compact table of image paths, decisions, and scores. |

To train a fresh model and test that exact model, run this after setup:

```bash
uv run train
uv run test --model-root models/crossmask/local-run --count 20 --output-dir data/predictions/local-run --positive-threshold 0.005
```

The default training run is intentionally small. It shows progress, writes
checkpoints, and is meant to finish in about two minutes on a normal machine.
For a larger run, increase the training options:

```bash
uv run train --positive-limit 2500 --epochs 8 --batch-size 64 --base-channels 24 --num-workers 2 --max-train-seconds 0 --prefetch-raw-cache
```

To test more input images, increase `--count`:

```bash
uv run test --count 200 --positive-ratio 0.5 --output-dir data/predictions/release-check-200 --positive-threshold 0.005
```

The normal workflow commands are:

| Command | What it does |
|---|---|
| `uv run download-images` | Downloads a small folder of example input images that can be passed to `test --input-dir`. |
| `uv run dataset` | Downloads the public dataset metadata if needed, downloads the required raw aerial scenes, and prepares the local training images and masks. |
| `uv run train` | Ensures the dataset exists, then trains CrossMaskNet and writes a checkpoint plus metrics. |
| `uv run test` | Downloads the public checkpoint and example input images if needed, then classifies images into output folders. With `--metrics-only`, it only prints stored model metrics. |

SAM3 is only needed if you want to run the labeling pipeline yourself. The
normal `dataset`, `train`, and `test` commands use the released labels and do
not require SAM3.

All options are optional. If you omit them, the commands use these defaults:

| Option | Default | What it means |
|---|---|---|
| `--profile` | `default` | Selects the prepared project configuration. |
| `--dataset` | `web/public/static-datasets/sam3-500k-masks-v1` | Metadata dataset location for `dataset` and `train`. |
| `--export` | `data/processed/crossmask/local-run` for `train`; `data/processed/crossmask/sam3-500k-road-channel-v4` for `dataset` | Where prepared training images and masks are written. |
| `--model-output` | `models/crossmask/local-run` | Where `train` writes model checkpoints and metrics. |
| `--model-root` | `models/crossmask/sam3-500k-road-channel-v4` | Where `test` reads model checkpoints and metrics. |
| `--input-dir` | not set | Folder of your own images for `test` to classify. If omitted, `test` downloads example images automatically. |
| `--download-dir` | `data/input/crossmask-images` | Where `test` writes automatically downloaded input images. |
| `--output-dir` | `data/predictions/crossmask-test` | Where `test` writes classified images, overlays, and summary files. |
| `--positive-threshold` | `0.005` | Minimum mask coverage for `test` to classify an image as `positive`. Images below this value go to `negative/`. Lower values put more images in `positive/`; higher values put more images in `negative/`. |
| `--no-overlays` | off | Skips writing overlay images for positive predictions. |
| `--count` | `20` | Total number of example images downloaded by `test` when `--input-dir` is omitted, or by `download-images`. |
| `--positive-ratio` | `0.5` | Share of downloaded examples that come from positive source labels. `0.5` with `--count 20` means 10 positive and 10 negative examples. |
| `--positive-count` / `--negative-count` | not set | Explicit class counts for downloaded examples if you do not want to use `--count` and `--positive-ratio`. |
| `--epochs` | `1` | Number of full passes over the training data. Increase this for a stronger run. |
| `--batch-size` | `4` | Number of image/mask pairs processed in one training step. Increase this on stronger machines. |
| `--image-size` | `128` | Training crop size. `128` means each crop is resized to 128 x 128 pixels. |
| `--limit-scenes` | no limit | Limits the raw scene prefetch step. The export can still download a missing scene later if it needs it. |
| `--positive-limit` | `30` for `train`; `2500` for `dataset` | Maximum number of crosswalk examples used when preparing the local training export. |
| `--negative-ratio` | `1.0` | Number of no-crosswalk examples per crosswalk example. `1.0` means a balanced export. |
| `--min-confidence` | `0.4` | Minimum released label confidence accepted for training examples. |
| `--min-mask-coverage` | `0.01` | Minimum mask area required for a positive crosswalk example. |
| `--workers` | `4` | Parallel download workers for raw aerial scenes. |
| `--num-workers` | `0` | Data-loading workers used while training. Increase this on stronger machines. |
| `--learning-rate` | `0.001` | Training step size for the optimizer. |
| `--base-channels` | `4` | Width of the CrossMaskNet feature layers. Larger values make the model heavier and stronger. |
| `--max-train-seconds` | `120` | Soft training time budget in seconds. `0` disables the limit. |
| `--road-channel` | off | Adds a road-context input channel during training. |
| `--skip-raw-cache` | on for `train`; off for `dataset` | Skips the full scene prefetch and downloads only scenes needed while building the export. |
| `--prefetch-raw-cache` | off | For `train`, downloads the raw scene cache before building the export. This is slower but useful for larger runs. |
| `--no-progress` | off | Disables live progress output during `download-images`, `train`, or `test --input-dir`. |
| `--metrics-only` | off | For `test`, prints stored model metrics without downloading or classifying images. |
| `--rebuild-export` | off | Recreates the prepared dataset export even if it already exists. |
| `--seed` | `7` | Keeps the train/test split and sampling repeatable. |

The raw image cache is written to `data/raw/sam3-500k-masks-v1/wms-mosaics/`.

`uv run dataset` restores the public dataset metadata if needed, caches the raw
swisstopo scene images locally, and prepares the training export.

`uv run train` reuses the prepared export when it already exists. If it is
missing, `train` prepares it first and then trains CrossMaskNet.

During training, checkpoints are written to the model output folder:

| File | Content |
|---|---|
| `checkpoint_epoch_001.pt` | Model weights after epoch 1. Later epochs use `002`, `003`, and so on. |
| `crossmasknet_latest.pt` | Most recent checkpoint. |
| `crossmasknet_best.pt` | Best checkpoint selected from the run. |
| `metrics.json` | Metrics, settings, and checkpoint paths for the run. |

`uv run test` restores the public model checkpoint if needed, downloads input
images when `--input-dir` is omitted, and classifies those images.

To classify new image files, put them in one folder and run:

```bash
uv run test --input-dir path/to/images --output-dir data/predictions/my-run --positive-threshold 0.005
```

`test` copies input images into the output folders. It does not move or delete
the original files in `--input-dir`.

If you want to only create an input folder without running the model, run:

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

- Final project report: `docs/project-report.md`
- Exam notes: `docs/exam-notes.md`
- Project report draft: `docs/project-report-draft.md`
- Project decisions: `docs/project-decisions.md`
- Static dataset deployment: `docs/static-dataset-deployment.md`
- SAM3 dataset data model: `docs/sam3-100k-data-model.md`
- SAM3 runbook: `docs/sam3-100k-runbook.md`
