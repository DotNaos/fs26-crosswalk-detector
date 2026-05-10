# Dataset and Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable pipeline that creates high-quality crosswalk labels, trains the course model, and proves quality with geographic holdout evaluation.

**Architecture:** Use SAM3.1 for pseudo-labeling, but never trust pseudo-labels blindly. Store every tile with source area, pseudo-label score, review state, and split group; train only from reviewed or high-confidence audited labels; evaluate on cities/areas that were not used during training.

**Tech Stack:** Python, PyTorch, SAM3.1, swisstopo imagery, existing review UI, remote GPU for batch inference and training, local laptop for UI review and metadata checks.

---

## Files and Responsibilities

- Modify `src/crosswalk_detector/scan_backend.py`: keep SAM3.1 scoring deterministic and expose confidence metadata.
- Modify `src/crosswalk_detector/dataset.py`: export labels with review state, area id, city id, and split group.
- Modify `src/crosswalk_detector/cli.py`: add commands for pilot dataset generation, audit sampling, split creation, and training runs.
- Modify `src/crosswalk_detector/train_mobilenet.py`: make training consume the curated dataset manifest and report metrics per split group.
- Create `src/crosswalk_detector/audit.py`: create spot-check queues and compute audited label accuracy.
- Create `src/crosswalk_detector/splits.py`: assign train/validation/test by geographic area instead of random tile.
- Create `tests/test_audit.py`: verify spot-check sampling and accuracy calculation.
- Create `tests/test_splits.py`: verify geographic split rules.
- Update `docs/project-decisions.md`: record the final dataset quality gates.
- Update `docs/real-run-status.md`: record pilot runs and evaluation results.

## Milestone 1: Make Label Quality Measurable

- [ ] **Step 1: Add audit metadata to exported labels**

Add fields to every tile record:

```json
{
  "pseudo_label": "crosswalk",
  "pseudo_score": 0.736,
  "review_label": null,
  "final_label": "crosswalk",
  "review_state": "unreviewed",
  "city_id": "zurich",
  "area_id": "zurich-main-station-01",
  "split_group": "zurich-main-station-01"
}
```

- [ ] **Step 2: Add audit queues**

Create three queues per area:

```text
positive_audit: random sample from high-confidence crosswalk labels
negative_audit: random sample from high-confidence no_crosswalk labels
uncertain_review: all labels near the threshold
```

- [ ] **Step 3: Define quality gates**

For a dataset batch to be accepted:

```text
positive audit precision >= 95%
negative audit precision >= 95%
uncertain tiles reviewed manually
no train/validation/test leakage by area
```

## Milestone 2: Build a Small Reliable Pilot Dataset

- [ ] **Step 1: Select pilot areas**

Use 6 to 9 areas total:

```text
3 dense city-center areas
3 normal urban road areas
2 to 3 outer/suburban areas
```

Use different Swiss cities early, not only Zürich.

- [ ] **Step 2: Generate tiles**

Target:

```text
1,500 to 2,000 final labeled tiles
roughly 20% to 35% crosswalk after candidate filtering
```

- [ ] **Step 3: Run SAM3.1 pre-labeling**

Use the existing SAM3.1 backend and keep these values in metadata:

```text
prompt
score
mask coverage
box overlap
detection count
```

- [ ] **Step 4: Review only the right tiles**

Review:

```text
100% uncertain tiles
at least 10% high-confidence positives
at least 5% high-confidence negatives
all obvious UI-selected failure clusters
```

## Milestone 3: Train the First Real Baseline

- [ ] **Step 1: Create geographic splits**

Split by `area_id`, not by random tile:

```text
train: 70%
validation: 15%
test: 15%
```

No neighboring tiles from the same area may appear in both train and test.

- [ ] **Step 2: Train a simple baseline**

Start with one model only:

```text
ResNet-18 or MobileNetV3
binary classification
image size 224 or 256
balanced batches
early stopping on validation F1
```

- [ ] **Step 3: Report metrics that matter**

Every run must report:

```text
accuracy
precision
recall
F1
confusion matrix
metrics per city
metrics per area type
top false positives
top false negatives
```

## Milestone 4: Scale Only After Quality Holds

- [ ] **Step 1: Expand area count**

Only expand after the pilot test set is acceptable.

Next target:

```text
4,000 to 6,000 labeled tiles
at least 4 cities/towns
mixed city center, normal urban, suburban, and road-heavy negative areas
```

- [ ] **Step 2: Active learning loop**

Use the trained model to find:

```text
high-confidence model/SAM disagreement
low-confidence model predictions
new areas with unusual road markings
```

Send only those into manual review.

## Verification Checklist

- [ ] A batch can be generated from scratch from area definitions.
- [ ] Every tile has source, score, review, and split metadata.
- [ ] Audit precision is computed before training.
- [ ] Training refuses unreviewed low-confidence labels.
- [ ] Test split is geographically separate.
- [ ] The final report includes confusion matrix and failure examples.

## Recommended Execution Order

1. Add audit and split metadata.
2. Generate one small pilot batch locally or from cached results.
3. Review/audit that batch in the UI.
4. Train the first baseline.
5. Inspect failures.
6. Expand to the larger dataset only after the first test split is credible.
