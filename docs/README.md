# Documentation Overview

This folder contains the final project documentation. Older planning notes,
local infrastructure runbooks, and exploratory drafts were removed because they
no longer describe the normal submission workflow.

## Recommended Reading Order

| File | Purpose |
|---|---|
| `../README.md` | Main setup guide, quick start, commands, options, release links, and current metrics. |
| `project-report.md` | Concise final project report covering dataset, model, optimizations, results, and insights. |
| `exam-notes.md` | Short preparation notes mapping the project to the expected exam/report topics. |
| `project-decisions.md` | Current decisions about scope, dataset storage, model design, and repository hygiene. |
| `static-dataset-deployment.md` | Details for restoring release assets and serving the static dataset explorer. |

## Submission Coverage

The documentation covers:

- what the project does and which model is submitted;
- how the dataset is distributed without committing raw images;
- how to restore public dataset metadata and the released checkpoint;
- how to run the normal `dataset`, `download-images`, `train`, and `test`
  commands;
- where generated local data and checkpoints are written;
- which results are currently documented for CrossMaskNet v4;
- how SAM3 was used for pseudo-labeling while CrossMaskNet remains the submitted
  model.
