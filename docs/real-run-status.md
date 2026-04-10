# Real Run Status

This file captures the current verified state of the real crosswalk pipeline.

## Dataset

- Run: `real-v1`
- Export: `real-balanced-256`
- Selected tiles: `256`
- Class balance:
  - `128` crosswalk
  - `128` no_crosswalk
- Split balance:
  - `train`: `90 / 90`
  - `val`: `19 / 19`
  - `test`: `19 / 19`

Main files:

- `data/processed/real-v1/exports/real-balanced-256/tiles.json`
- `data/processed/real-v1/exports/real-balanced-256/labels.csv`
- `data/processed/real-v1/exports/real-balanced-256/summary.json`

## Labeling Pipeline

- Source imagery comes from large swisstopo WMS mosaics over several Swiss city centers.
- Each scene is cut into real `25 m x 25 m` tiles.
- Auto-labeling uses:
  - CLIP zero-shot scoring
  - a local stripe-based heuristic
  - a harder negative selection that prefers road-like negatives over water, fields, and rooftops
- The dataset stays auditable because every tile remains in `tiles.json`, even when it is dropped.

## Review UI

The Bun + React review app lives in `web/`.

Verified behavior:

- it loads the real dataset
- it renders the tile board
- it lazy-loads tile images
- it exposes a local API for tile updates
- updates can write back to `tiles.json` and regenerate `labels.csv`

Default local URLs:

- API: `http://127.0.0.1:8787`
- UI: `http://localhost:5173` or the next free Vite port

## Baseline Model

- Model: `mobilenet_v3_small`
- Latest verified test accuracy on the harder 256-image dataset: `0.763158`
- Best validation accuracy in the latest run: `0.868421`

Artifacts:

- `models/real-v1/real-balanced-256/mobilenet_v3_small.pt`
- `models/real-v1/real-balanced-256/metrics.json`

## Manual QA Notes

Review sheets were generated under:

- `data/processed/real-v1/exports/real-balanced-256/review/crosswalk-sample-sheet.png`
- `data/processed/real-v1/exports/real-balanced-256/review/no_crosswalk-sample-sheet.png`
- `data/processed/real-v1/exports/real-balanced-256/review/test-errors-sheet.png`

Observed from manual spot checks:

- the negative class is now much more road-heavy than the earlier easy-negative build
- the positive class still contains some obvious false positives and rooftop-like stripe patterns
- the current review UI is therefore an important part of cleaning the dataset further

## Hugging Face Check

The user asked about `facebook/sam3.1`.

What is verifiably available on Hugging Face right now is the `facebook/sam2.1-*` family, for example:

- `facebook/sam2.1-hiera-tiny`
- `facebook/sam2.1-hiera-small`
- `facebook/sam2.1-hiera-base-plus`
- `facebook/sam2.1-hiera-large`

These are segmentation models, not simple drop-in binary tile classifiers.
They are relevant for stronger review or segmentation-assisted labeling, but the current baseline remains the MobileNet tile classifier.
