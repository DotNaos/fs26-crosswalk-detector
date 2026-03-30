# Crosswalk Detector

Private workspace for the FS26 image classification project.

## Goal

Train and evaluate a model that detects whether a 25 m x 25 m aerial image contains a pedestrian crossing.

## Structure

- `src/crosswalk_detector/`: reusable code
- `tests/`: automated checks
- `notebooks/`: experiments
- `data/`: local datasets
- `models/`: saved model artifacts

## Getting Started

1. Install dependencies with `uv sync`.
2. Run tests with `uv run pytest`.

## Notes

- Keep large datasets and model weights out of git.
- Record dataset and experiment decisions in the README or a report once work starts.
