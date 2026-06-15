# Project Decisions

This document keeps the current public project decisions. Older pilot plans,
local infrastructure notes, and intermediate runbooks were removed from the
submitted repository because they no longer describe the normal workflow.

## Submission Scope

- The submitted model is `CrossMaskNet`, a custom PyTorch segmentation model.
- `SAM3` was used to generate pseudo-labels and masks for dataset construction.
- `SAM3` is not used during training or inference in the submitted workflow.
- The normal commands are `uv run dataset`, `uv run download-images`,
  `uv run train`, and `uv run test`.

## Command Surface

- `uv run dataset` restores the public metadata release, downloads raw
  Swisstopo scenes when needed, and prepares a local CrossMaskNet export.
- `uv run download-images` creates a small input folder for manual or scripted
  inference checks.
- `uv run train` prepares a small default export if needed, trains CrossMaskNet,
  and writes checkpoints under `models/crossmask/`.
- `uv run test` restores the released checkpoint when needed, downloads example
  inputs when no input folder is provided, and writes classified output folders.
- Without `uv`, the same commands are available through
  `python -m crosswalk_detector.workflow <command>`.

## Dataset Storage

- Raw aerial images are not committed to Git.
- The public dataset release stores metadata, labels, masks, scene references,
  and download information.
- Raw Swisstopo imagery is downloaded reproducibly when the dataset export,
  training run, or test run needs image pixels.
- This keeps the repository small while still making the dataset reconstructable.

## Public Assets

- Dataset metadata release: `submission-dataset-v1`
- Dataset id: `sam3-500k-masks-v1`
- Model checkpoint release: `crossmasknet-v4`
- Restore script: `scripts/download_submission_assets.py`

The restore script downloads the released metadata package into
`web/public/static-datasets/` and the released model checkpoint into
`models/crossmask/sam3-500k-road-channel-v4/`.

The release assets are intentionally outside Git so that the repository remains
small and can be cloned quickly.

## Model Design

- CrossMaskNet predicts a crosswalk mask from aerial image tiles.
- The binary positive/negative decision is derived from predicted mask coverage.
- The model is intentionally small enough to train and run on normal hardware.
- Road-context information is used to reduce visually plausible false positives
  away from roads.

## Training Defaults

- The default quick training run uses a small export so the workflow can be
  verified quickly.
- Larger runs can be configured with command-line parameters such as
  `--positive-limit`, `--negative-ratio`, `--epochs`, and `--batch-size`.
- Training writes checkpoints and metrics under `models/crossmask/`.

## Inference Output

- `uv run test` can classify automatically downloaded examples or a custom
  input directory.
- Images are copied into `positive/` and `negative/` output folders.
- Positive predictions also get overlay images in `positive_overlays/`.
- The run writes `summary.json` and `predictions.csv` for inspection.

## Repository Hygiene

- Do not commit raw datasets, generated exports, checkpoints, logs, local caches,
  or machine-specific configuration.
- Keep public setup instructions in `README.md`.
- Keep final submission documentation in `docs/project-report.md`.
- Keep operational restore details in `docs/static-dataset-deployment.md`.
- Keep old plans, local runbooks, and machine-specific notes out of the final
  public documentation.
