# Slurm Batch Workflow

This repo now supports an offline scan workflow for shared GPU servers.

## What the UI exports

The review UI can export a `*-scan-job.json` file for the currently selected map circle.

That file contains:

- dataset run/export names
- scene metadata
- tile geometry for the selected area
- the threshold used for labeling

The cluster job does not need the browser. It only needs that JSON file plus internet access to pull the Swiss aerial scene and the Hugging Face models.

## What the cluster runner does

The batch runner command is:

```bash
python -m crosswalk_detector.cli run-scan-job \
  --job-file path/to/job.json \
  --output path/to/result.json \
  --progress
```

The result file contains:

- the original job payload
- scanner metadata
- per-tile labels and scores
- a small summary block

## Slurm usage

Submit with:

```bash
sbatch cluster/scan-job.slurm jobs/scene-job.json results/scene-result.json
```

Useful commands:

```bash
squeue -u "$USER"
sacct -j <job-id>
scancel <job-id>
```

## Import back into the UI

Use the import control in the review map for the same scene.

The imported result is loaded back as review suggestions, so you can inspect it before applying it to the local dataset.
