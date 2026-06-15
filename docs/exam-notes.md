# Exam Notes

This file maps the expected project requirements to the current repository. It
is a compact preparation guide; the full setup instructions live in
`README.md`, and the concise report lives in `docs/project-report.md`.

## Required Submission

The mandatory submission consists of:

| Required item | Where it is in this project |
|---|---|
| Created and used dataset | Public dataset metadata release, restored by `uv run dataset` or automatically by `uv run train` and `uv run test`. |
| Code of the used and described models | Python code in `src/crosswalk_detector/`, especially the CrossMaskNet training and inference code. |

The repository also contains:

- reproducible setup and command documentation in `README.md`;
- the final short report in `docs/project-report.md`;
- release links for the public dataset metadata and model checkpoint;
- a static dataset explorer for inspecting the released metadata.

The optional project report should be no longer than two A4 pages. The grading
help sheet names five report topics, each worth 4 points:

| Topic | What to document for this project |
|---|---|
| Dataset key facts | Source imagery, tile size, label source, class counts, train/validation/test split, and data quality. |
| Model architecture | CrossMaskNet, its encoder-decoder structure, skip connections, mask output, and classification rule. |
| Optimizations | Hard negatives, road-context channel, confidence filters, threshold calibration, and training settings. |
| Results | Accuracy, precision, recall, Dice, IoU, and how to interpret the tradeoff. |
| Personal insights | What was learned about data quality, pseudo-labels, hard negatives, and explainable predictions. |

## Project In One Minute

The task is to decide whether a `25 m x 25 m` Swiss aerial image tile contains a
pedestrian crossing.

This project solves that as a segmentation-first classification task:

```text
Swisstopo aerial image tile
        |
        v
CrossMaskNet predicts a crosswalk mask
        |
        v
Mask coverage is compared to a threshold
        |
        v
Final class: positive or negative
```

The final submitted model is CrossMaskNet. SAM3 was used to generate
pseudo-labels for the dataset, but SAM3 is not used when training or running
CrossMaskNet.

## Dataset Facts

| Fact | Value |
|---|---:|
| Public metadata dataset | `sam3-500k-masks-v1` |
| Total selected tiles | `500,000` |
| Source-positive tiles | `8,815` |
| Source-negative tiles | `491,185` |
| Metadata shards | `489` |
| Source scenes | `489` |
| Cities covered | `14` |

The model was trained on a prepared CrossMaskNet export derived from that larger
metadata dataset:

| Split | Crosswalk | No crosswalk | Total |
|---|---:|---:|---:|
| Train | `1,600` | `3,406` | `5,006` |
| Validation | `189` | `193` | `382` |
| Test | `211` | `201` | `412` |
| Total | `2,000` | `3,800` | `5,800` |

Important explanation: the large dataset stores metadata and label information.
The raw aerial image scenes are downloaded reproducibly from swisstopo when a
local training export needs them. They are not committed to Git.

## Model Architecture

CrossMaskNet is a custom compact U-Net-style convolutional neural network written
in PyTorch and trained from scratch.

The important architecture points are:

| Part | Explanation |
|---|---|
| Input | RGB aerial tile plus a road-context channel. |
| Encoder | Compresses the image and learns higher-level visual features. |
| Decoder | Upsamples the compressed features back to image resolution. |
| Skip connections | Bring fine image details from encoder layers into decoder layers. |
| Output | One-channel crosswalk probability mask. |
| Classification | If predicted mask coverage is above the threshold, the image is classified as positive. Otherwise it is negative. |

This fits the assignment because the final output is still a binary
classification, but the model also produces a visible mask that makes the
decision easier to inspect.

## Optimizations

The most important improvements were:

| Optimization | Why it matters |
|---|---|
| Hard negatives | Added difficult no-crosswalk examples such as roof patterns, parking markings, shadows, and road markings. This reduced false positives. |
| Road-context channel | Gives the model extra information about plausible road areas, which improves mask quality. |
| Confidence and mask filters | Keeps low-quality pseudo-labels out of the training export. |
| Dice-style segmentation loss | Helps with small crosswalk regions because most pixels in an aerial tile are background. |
| Threshold calibration | Controls the precision/recall tradeoff for the final positive/negative decision. |

## Current Results

The current documented checkpoint is CrossMaskNet v4.

| Metric | Value | Meaning |
|---|---:|---|
| Accuracy | `0.961165` | Overall share of correct positive/negative decisions. |
| Precision | `0.957746` | When the model predicts positive, how often that is correct. |
| Recall | `0.966825` | Of all positive examples, how many the model finds. |
| Positive Dice | `0.794408` | Mask overlap quality on positive examples. |
| Positive IoU | `0.658937` | Stricter mask overlap metric on positive examples. |

The default positive threshold is `0.005`. Lower thresholds classify more images
as positive. Higher thresholds classify more images as negative.

## What To Be Able To Explain

These are the core points to understand:

| Question | Short answer |
|---|---|
| What is the dataset? | Swiss aerial image tiles of `25 m x 25 m`, with generated labels for crosswalk/no-crosswalk and masks for positives. |
| What is the model? | CrossMaskNet, a small custom PyTorch segmentation model trained from scratch. |
| Why segmentation if the task is classification? | The mask gives a visible reason for the decision; the final binary class is derived from the mask. |
| What did SAM3 do? | It generated pseudo-labels and masks for building the dataset. It is not the final model. |
| What improved the model most? | Better data, especially hard negatives, plus road context and threshold tuning. |
| What do the results mean? | The model correctly classifies about 96 percent of the held-out test examples and also produces usable masks. |
| What is the main tradeoff? | Precision versus recall: stricter thresholds reduce false positives but can miss weaker crosswalks. |

## Files To Use While Writing The Report

| File | Purpose |
|---|---|
| `README.md` | Quick start, commands, defaults, release links, and current metrics. |
| `docs/project-report.md` | Final concise project report. |
| `docs/static-dataset-deployment.md` | How the public dataset and checkpoint are restored. |
| `docs/project-decisions.md` | Current project decisions and rationale. |
| `docs/README.md` | Overview of the documentation folder. |
| `datasets/README.md` | What belongs in the dataset folder and what stays out of Git. |
| `web/README.md` | How the dataset explorer frontend is built and run locally. |
