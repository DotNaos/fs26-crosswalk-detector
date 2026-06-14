# Crosswalk Detector TODO

Last updated: 2026-05-18

## Current State

- The 100k SAM3 metadata dataset exists as `datasets/sam3-100k-v1`.
- The filtered 100k results produced about 2.7k crosswalk positives.
- The final 500k mask-overlay review gallery is available at `http://100.93.7.26:8789/`.
- First own-model baseline is trained:
  - model: `CrossMaskNet`, a small from-scratch segmentation model.
  - remote model path: `/home/oli/codex-runs/crosswalk-detector-sam3/models/crossmask/sam3-500k-balanced-4k-v1/crossmasknet_best.pt`
  - remote metrics path: `/home/oli/codex-runs/crosswalk-detector-sam3/models/crossmask/sam3-500k-balanced-4k-v1/metrics.json`
  - visual prediction gallery: `http://100.93.7.26:8791/`
  - training set: 4,000 samples, balanced between 2,000 SAM3 crosswalk masks and 2,000 no-crosswalk negatives.
  - test result: positive Dice 0.763920, positive IoU 0.618018, image accuracy 0.917476, precision 0.870293, recall 0.985782.
  - observed failure mode: the model over-predicts blue mask blobs on roofs, vegetation, parking areas, shadows, and tile edges. It has learned useful mask localization, but it still needs hard negatives and stronger context filtering before it is reliable.
- CrossMaskNet v2 is trained and compared against v1:
  - remote model path: `/home/oli/codex-runs/crosswalk-detector-sam3/models/crossmask/sam3-500k-hard-v2/crossmasknet_best.pt`
  - visual prediction gallery: `http://100.93.7.26:8792/`
  - v1/v2 review galleries now use transparent overlays, so the underlying image remains visible.
  - v2 dataset: v1's 4,000-sample export plus 1,400 repeated hard-negative training rows mined from v1's highest false-positive scores.
  - same held-out test split as v1: 211 crosswalk, 201 no-crosswalk.
  - v2 test result: positive Dice 0.768770, positive IoU 0.624392, image accuracy 0.944175, precision 0.915929, recall 0.981043.
  - compared with v1: precision improved by 0.045636, image accuracy by 0.026699, positive Dice by 0.004850, while recall dropped slightly by 0.004739.
  - decision: hard negatives help, but they are not enough alone. The next improvement should add road-context filtering and human-reviewed hard negatives, because remaining false positives still include roofs, vegetation, parking/road-edge structures, and generic bright markings.
- CrossMaskNet v3 is trained and compared against v1/v2:
  - remote model path: `/home/oli/codex-runs/crosswalk-detector-sam3/models/crossmask/sam3-500k-road-hard-v3/crossmasknet_best.pt`
  - visual prediction gallery: `http://100.93.7.26:8793/`
  - v3 dataset: v1's 4,000-sample export plus 1,800 repeated hard-negative training rows mined from v2's highest false-positive scores.
  - same held-out test split as v1/v2: 211 crosswalk, 201 no-crosswalk.
  - v3 raw test result: positive Dice 0.791034, positive IoU 0.654306, image accuracy 0.966019, precision 0.971292, recall 0.962085.
  - v3 with road-context post-filter: positive Dice 0.781053, positive IoU 0.640761, image accuracy 0.968447, precision 0.980583, recall 0.957346.
  - decision: v3 raw is the best default model because it has the best mask quality. The road-context filter is useful as a high-precision mode, but it can trim true positives, so it should not be the only training/evaluation truth yet.
- CrossMaskNet v4 is trained with road-context as a fourth input channel:
  - remote model path: `/home/oli/codex-runs/crosswalk-detector-sam3/models/crossmask/sam3-500k-road-channel-v4/crossmasknet_best.pt`
  - visual prediction gallery: `http://100.93.7.26:8794/`
  - v4 dataset: v3's 5,800-sample export plus precomputed road masks saved in `road_masks/` and referenced by `road_path` in the manifest.
  - same held-out test split as v1/v2/v3: 211 crosswalk, 201 no-crosswalk.
  - v4 raw test result: positive Dice 0.794408, positive IoU 0.658937, image accuracy 0.961165, precision 0.957746, recall 0.966825.
  - v4 with road-context post-filter: positive Dice 0.775852, positive IoU 0.633789, image accuracy 0.958738, precision 0.961905, recall 0.957346.
  - decision: v4 is the best segmentation model so far by positive Dice and IoU, but v3 raw is still the better default for image-level decisions. The explicit road input helped mask quality more than false-positive suppression.
- Final 500k run result, checked on 2026-05-17 23:10 Europe/Zurich:
  - 489 of 489 shards finished.
  - 500,000 tiles scanned.
  - 8,811 raw SAM3 crosswalk candidates with masks.
  - 6,284 candidates at confidence >= 0.4.
  - 5,173 candidates at confidence >= 0.5.
  - 3,044 candidates at confidence >= 0.6.
  - Mask consistency is clean: no missing mask files and no masks attached to `no_crosswalk` labels.
  - The exported gallery contains 1,000 rows at confidence >= 0.4; the first row has score 0.815766 and the last row has score 0.686114.
  - Visual checks passed at the top and bottom of the gallery: images load, green tile outlines show the selected tile, and blue overlays show the SAM3 pseudo-mask.
  - Decision: do not start final own-model training yet. The 500k run is useful, but it is below the 10k high-confidence positive target, so the next dataset step should be a second focused candidate run in high-yield city/intersection areas.
- The new mask-capable 500k dataset exists on the Ubuntu laptop:
  - remote path: `/home/oli/codex-runs/crosswalk-detector-sam3/datasets/sam3-500k-masks-v1`
  - result path: `/home/oli/codex-runs/crosswalk-detector-sam3/results/sam3-500k-masks-v1`
  - review server session: `tmux attach -t crosswalk-review-1000-server`
- SAM3 is used as a dataset generator and benchmark, not as the final semester-work model.
- The final model must be our own architecture, trained by us, with its own explanation and evaluation.

## Active Goal

CrossMaskNet v4 with road-context input is trained, evaluated, visually reviewed, and compared against v1/v2/v3.

## Next Steps

1. Add human-reviewed hard negatives:
   - roofs
   - vegetation
   - parking lots
   - shadows
   - tile-edge artifacts
   - road markings that are not crosswalks
2. Calibrate the image-level decision threshold separately from mask training:
   - current fixed mask-coverage threshold `0.005` gives v4 accuracy 0.961165, precision 0.957746, recall 0.966825.
   - threshold `0.010` reduces false positives from 9 to 6 and raises precision to 0.971014, but recall drops to 0.952607.
   - choose a default based on whether the review UI should favor high recall or fewer false positives.
3. Add mask-quality metrics to the review/export tooling:
   - mask coverage
   - empty/tiny mask flag
   - edge-touching mask flag
   - road-context score
4. Create a second focused candidate run around high-yield cities, dense intersections, and road areas that produced real crosswalks in the first 500k scan.
5. Keep the confidence threshold at `>= 0.4` for review, but prioritize the strongest items first and sample lower scores mainly for hard negatives.
6. Add human review workflow for:
   - confirmed crosswalk
   - rejected crosswalk
   - adjusted mask
   - hard negative category
7. Continue the own-model track:
   - keep v3 raw as the current image-level default
   - keep v4 as the current best mask-quality model
   - add human-reviewed hard negatives before training v5
   - export visual failure galleries after each run
   - evaluate against SAM3 labels, human-reviewed labels, and hard negatives

## Quality Gates

- Do not train the semester model until masks are exported and visually checked.
- Do not count raw SAM3 guesses as final positives without at least review sampling.
- Keep hard negatives as first-class examples.
- Keep SAM3 out of the final model architecture; it is only a data generator and baseline.
- Mask artifacts must only be attached to `crosswalk` labels, never to `no_crosswalk`.
- Before reporting a UI or dataset export as done, open it and inspect it visually.

## Useful Commands

```bash
ssh os-yoga-unix-personal
cd /home/oli/codex-runs/crosswalk-detector-sam3
tmux attach -t crosswalk-review-1000-server
tail -n 120 logs/sam3-500k-masks-scan.log
tail -n 120 logs/sam3-500k-masks-postprocess.log
```

```bash
.venv/bin/python -m crosswalk_detector.cli validate-metadata-dataset \
  --dataset datasets/sam3-500k-masks-v1 \
  --max-errors 50
```

```bash
.venv/bin/python -m crosswalk_detector.cli summarize-sam3-results \
  --results results/sam3-500k-masks-v1 \
  --expected-shards 489
```

```bash
.venv/bin/python scripts/export_crosswalk_review.py \
  --dataset datasets/sam3-500k-masks-v1 \
  --output review-exports/sam3-500k-crosswalk-1000 \
  --limit 1000 \
  --min-confidence 0.4
```
