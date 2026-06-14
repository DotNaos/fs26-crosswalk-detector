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

These results are measured against the generated held-out test split. A human-reviewed validation subset is still recommended before making strong real-world accuracy claims.

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

## Local Development

Python dependencies are managed with `uv`.

```bash
uv sync
uv run pytest
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

## Download Submission Assets

The larger dataset metadata package and the trained model checkpoint are stored
as GitHub Release assets, not in git.

```bash
python3 scripts/download_submission_assets.py
```

This downloads:

- static 500k dataset metadata to `web/public/static-datasets/`;
- CrossMaskNet v4 checkpoint and metrics to
  `models/crossmask/sam3-500k-road-channel-v4/`.

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

## Notes

- Do not commit large raw datasets, generated model runs, or private machine-specific artifacts.
- Keep project decisions and important experiment results documented in `docs/`.
- The final report should clearly state that SAM3 generated pseudo-labels, while CrossMaskNet is the submitted custom model.
