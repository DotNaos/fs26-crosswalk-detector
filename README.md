# Crosswalk Detector

Private workspace for the FS26 crosswalk classification project.

## Goal

Train and evaluate a model that detects whether a 25 m x 25 m aerial image contains a pedestrian crossing.

## Structure

- `src/crosswalk_detector/`: reusable code
- `tests/`: automated checks
- `notebooks/`: experiments
- `data/`: local datasets
- `models/`: saved model artifacts
- `docs/`: project notes and plans

## Getting Started

1. Install dependencies with `uv sync`.
2. Put the Hugging Face token into `.env` as `HF_TOKEN=...`.
3. Run tests with `uv run pytest`.
4. Build the real 256-image dataset with `uv run crosswalk-pipeline build-real-dataset --run-name real-v1 --name real-balanced-256 --target-per-class 128`.
5. Train the baseline classifier with `uv run crosswalk-pipeline train-mobilenet --run-name real-v1 --name real-balanced-256 --epochs 6`.

## Review App

The local label review app lives in `web/` and runs on Bun.

From that directory:

1. Install dependencies with `bun install`.
2. Start the API and frontend together with `bun run dev`.
3. Open the local review UI in the browser.

The review app defaults to `real-v1/real-balanced-256`.
It reads `tiles.json`, shows the tile map, and writes label corrections back to both `tiles.json` and `labels.csv`.

## Current Run

- Real dataset export: `data/processed/real-v1/exports/real-balanced-256`
- Review sheets: `data/processed/real-v1/exports/real-balanced-256/review/`
- Model artifacts: `models/real-v1/real-balanced-256/`
- Review UI: `web/`

## Notes

- Keep large datasets and model weights out of git.
- Keep project decisions in `docs/project-decisions.md`.
- Current dataset outline: `docs/dataset-strategy.md`
- Real run status and evaluation notes: `docs/real-run-status.md`
- Data access and sizing decisions are summarized in `docs/project-decisions.md`.
- The real build uses only unique raw tiles. It does not pad buckets with rotations or duplicates.
