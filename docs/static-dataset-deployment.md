# Static Dataset Deployment

The dataset inspector can run without the Ubuntu inference server by using the
compressed static metadata under `web/public/static-datasets`.

## Current Static Dataset

- Dataset: `sam3-500k-masks-v1`
- Rows: `500,000`
- Static path: `web/public/static-datasets/sam3-500k-masks-v1`
- Browser manifest: `web/public/static-datasets/index.json`
- Format: compressed JSONL shards (`*.jsonl.gz`)
- Release: https://github.com/DotNaos/fs26-crosswalk-detector/releases/tag/submission-dataset-v1

The static frontend is read-only. It can inspect tiles, filters, clusters, and
label vote history, but it cannot start CrossMaskNet inference runs or write
human review labels. Those actions still require the Ubuntu backend.

## Downloading Release Assets

From a fresh checkout, restore the public dataset metadata and model checkpoint
with:

```bash
python3 scripts/download_submission_assets.py
```

The script verifies SHA-256 checksums, extracts the static dataset metadata into
`web/public/static-datasets/`, and stores CrossMaskNet v4 under
`models/crossmask/sam3-500k-road-channel-v4/`.

## Vercel Deployment

The repository root contains `vercel.json`, which builds the web app from
`web/` and serves `web/dist`.

Production URL:

https://crosswalk-detector.vercel.app

The `.vercelignore` file keeps local training data, dependencies, and result
folders out of the deployment upload. Only the web app and static dataset
metadata are deployed.

## Updating The Static Dataset

Generate a static export from a metadata dataset:

```bash
python scripts/export_static_metadata_dataset.py \
  --dataset datasets/sam3-500k-masks-v1 \
  --output web/public/static-datasets/sam3-500k-masks-v1 \
  --overwrite
```

Then update `web/public/static-datasets/index.json` so the dataset entry points
to `/static-datasets/sam3-500k-masks-v1`.

## Model Checkpoint Release

The trained CrossMaskNet v4 checkpoint is stored as a public GitHub Release
asset, not inside git:

https://github.com/DotNaos/fs26-crosswalk-detector/releases/tag/crossmasknet-v4

Release assets:

- `crossmasknet_best.pt`
- `metrics.json`
- `road-filter-metrics.json`
- `SHA256SUMS`
