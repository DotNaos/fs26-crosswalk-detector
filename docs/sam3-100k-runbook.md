# SAM3 100k Runbook

This runbook describes the intended operational flow. The metadata scaffold,
job builder, merge step, validator, and training export commands exist. The
actual SAM3 scan still needs a CUDA host such as Mercury.

## Preconditions

- Mercury is reachable over SSH.
- Slurm can allocate an A100 GPU.
- The remote repo has Python dependencies installed.
- `sam3` is installed from the Git dependency in `pyproject.toml`.
- Local secrets stay in ignored `.env` files or 1Password, not in git.
- The metadata dataset folder exists under `datasets/sam3-100k-v1/`.

Existing direct scan command:

```bash
python -m crosswalk_detector.cli run-scan-job \
  --job-file jobs/example.json \
  --output results/example.result.json \
  --progress
```

Existing Slurm wrapper:

```bash
sbatch cluster/scan-job.slurm jobs/example.json results/example.result.json
```

## Step 1: Prepare Metadata Scaffold

Existing command:

```bash
python -m crosswalk_detector.cli prepare-sam3-metadata-dataset \
  --config configs/sam3-100k.toml \
  --dataset datasets/sam3-100k-v1
```

Expected output:

- `datasets/sam3-100k-v1/dataset.json`
- `datasets/sam3-100k-v1/label-sources.json`
- `datasets/sam3-100k-v1/sources/swisstopo.json`
- scene and perimeter files
- many `tiles.jsonl` shards

Existing validation command:

```bash
python -m crosswalk_detector.cli validate-metadata-dataset \
  --dataset datasets/sam3-100k-v1
```

Checks:

- no image files under `datasets/sam3-100k-v1`
- row count is exactly `100,000`
- every row has reconstruction metadata
- every row has a deterministic `image_id`
- every shard listed in `dataset.json` exists
- split counts match the config

## Step 2: Smoke Run on Mercury

Existing command:

```bash
python -m crosswalk_detector.cli build-sam3-shard-jobs \
  --dataset datasets/sam3-100k-v1 \
  --limit-shards 1 \
  --output jobs/sam3-100k-smoke
```

Submit the first job:

```bash
sbatch cluster/scan-job.slurm \
  jobs/sam3-100k-smoke/shard-0000.json \
  results/sam3-100k-smoke/shard-0000.result.json
```

Monitor:

```bash
squeue -u "$USER"
sacct -j <job-id>
```

Success criteria:

- CUDA is visible.
- SAM3 loads.
- The job writes a result JSON.
- The result contains one SAM3 vote per processed image.
- A few restored image crops visually match the metadata.

## Step 3: Merge Smoke Result

Existing command:

```bash
python -m crosswalk_detector.cli merge-sam3-metadata-dataset \
  --dataset datasets/sam3-100k-v1 \
  --results results/sam3-100k-smoke \
  --write
```

Expected behavior:

- append or replace SAM3 votes for matching `image_id`
- keep previous votes from other sources
- recompute `resolved_label`
- leave human votes untouched
- mark low-confidence or conflicting rows as `disputed`

Validation:

```bash
python -m crosswalk_detector.cli validate-metadata-dataset \
  --dataset datasets/sam3-100k-v1
```

Then open the review UI and inspect the smoke shard through the metadata paging
API.

## Step 4: Full Shard Queue

Existing command:

```bash
python -m crosswalk_detector.cli build-sam3-shard-jobs \
  --dataset datasets/sam3-100k-v1 \
  --output jobs/sam3-100k-v1
```

Submit either through the web remote controller or with Slurm job arrays. The
job files are ordinary `run-scan-job` payloads, so each result can be written
with `cluster/scan-job.slurm`.

## Step 5: Review and Training Export

Review first:

- inspect low-confidence positives
- inspect low-confidence negatives
- inspect rows where road overlay suggests a crossing but SAM3 says no
- inspect random samples from both classes
- add human labels where needed

Existing export command:

```bash
python -m crosswalk_detector.cli export-training-dataset \
  --dataset datasets/sam3-100k-v1 \
  --output data/exports/sam3-100k-v1-reviewed \
  --materialize-images
```

Export should write images outside git and keep a manifest that points back to
the exact metadata dataset.

## Failure Handling

If SAM3 fails to load:

- stop the full queue
- fix the remote environment
- rerun the smoke shard

If Swisstopo image reconstruction fails:

- do not mark rows as `drop` automatically
- record the retrieval error in result metadata
- keep the row so the source reference can be fixed

If the merge finds duplicate votes from the same source:

- keep only the newest vote for the same `source_id` and `image_id`
- preserve old results in the raw result folder outside git

If row counts change unexpectedly:

- stop before committing metadata
- rerun validation
- compare `dataset.json` shard counts with actual JSONL line counts

## Current Code Gap

Already present:

- SAM3.1 scan backend dependency and scan job command
- Slurm wrapper for one scan job
- metadata JSONL paging API in the web server
- TypeScript metadata dataset types
- Python metadata dataset validator and label priority resolver

Still needed:

- 100k scene/perimeter config
- metadata scaffold generator
- SAM3 shard job builder
- merge command for SAM3 votes into JSONL
- metadata review UI
- final training export from resolved metadata
