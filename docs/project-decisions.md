# Project Decisions

This file is the current source of truth for the project plan.
It captures the main decisions made so far, the rationale behind them, and the expected data volume.

## Problem Definition

- Task: classify a `25 m x 25 m` aerial tile as either `crosswalk` or `no_crosswalk`.
- The final project model should solve the course task directly as a binary classification problem.
- Segmentation is useful for pre-labeling and analysis, but not the final task itself.

## Dataset Strategy

- Build the dataset semi-automatically instead of labeling everything by hand.
- Start from several seed cities instead of a single city.
- Use city-center areas only as a bootstrap step because they likely contain more crossings.
- Expand later into outer districts and additional cities to reduce center bias.
- Split train, validation, and test by geographic area, not by random tile shuffle.
- Keep only two final labels:
  - `crosswalk`
  - `no_crosswalk`
- Preserve confidence scores and review flags in metadata so the dataset stays auditable.

## Raw Data Sources

### Imagery

- Primary image source: `SWISSIMAGE` from swisstopo.
- Preferred coordinate system: `EPSG:2056`.
- Expected resolution in many urban areas such as Zurich: `10 cm`.
- At `10 cm`, a `25 m x 25 m` tile corresponds to about `250 x 250` pixels.

### Map Context

- Use street and city context only to guide sampling and prioritization.
- Suitable context sources can include road geometry, city boundaries, and candidate search areas.
- These context layers are helper inputs, not the final ground-truth labels.

## API and Download Decision

- For fast prototyping, use `WMTS`.
- For serious batch processing and larger scans, use `STAC` and the downloadable `COG` files.
- Do not rely on large-scale WMTS scraping for the real pipeline.

Reasoning:

- `WMTS` is convenient for testing small areas and validating the pipeline.
- `STAC` plus `COG` is better for repeatable bulk processing, local caching, and GPU batch jobs.
- swisstopo explicitly indicates that high-intensity automated retrieval should not be done as if the visualization service were a bulk download backend.

## Scanning and Sampling Decision

- Do not start with a brute-force scan of all of Zurich.
- Start with a pilot over a small number of selected areas in several cities.
- Discretize each search area into `25 m x 25 m` task tiles.
- Use controlled sampling at first instead of exhaustively scanning every possible window.
- Bias sampling toward road-heavy areas so the negative examples stay relevant.
- If a broader scanner is used later, make it staged:
  1. candidate generation
  2. stronger vision pass on candidates
  3. deduplication
  4. manual audit only where needed

## Labeling Decision

- Use a weak-label pipeline first:
  1. location and map heuristics
  2. general-purpose vision detection or segmentation
  3. confidence-based bucket assignment
  4. small manual review
- Manual review should focus on:
  - uncertain cases
  - random spot checks from likely positives
  - random spot checks from likely negatives
- The first version of the dataset does not need perfect labels.
- The first version does need a reproducible and inspectable labeling process.

## Model Decision

### Pre-labeling Model

- Use a scriptable general-purpose vision pipeline to accelerate label creation.
- A good fit is an open-vocabulary detector plus segmentation, for example:
  - `Grounding DINO`
  - `SAM 2`
- This is for candidate discovery and pseudo-labeling, not the final project model.

### Final Course Model

- Start with a small image classifier as the project baseline.
- Good baseline candidates:
  - `EfficientNet-B0`
  - `ResNet-18`
  - `ConvNeXt-Tiny`
- The final model should stay easy to train, explain, and evaluate against the course rubric.

## Compute Decision

- Laptop use:
  - local development
  - sampling logic
  - metadata handling
  - small tests
  - visual spot checks
- Remote GPU use:
  - batch pseudo-labeling
  - large inference runs
  - model training
  - repeated experiments
- Preferred heavy-compute targets:
  - dedicated GPU server
  - Kaggle
  - Google Colab

## Data Volume Expectations

### Official SWISSIMAGE Download Size

According to the official SWISSIMAGE product information:

- one `1 km x 1 km` tile at `10 cm` resolution is about `55 MB`
- full-country coverage is about `42,700` tiles
- full-country coverage at `10 cm` is about `2.4 TB`

This means the project must plan around area-based subsets, not the whole country.

### Raw Imagery Size by Area

At `10 cm`, a useful planning estimate is:

| Area | Approx. raw SWISSIMAGE size |
| --- | --- |
| `5 km²` | `~275 MB` |
| `10 km²` | `~550 MB` |
| `25 km²` | `~1.4 GB` |
| `50 km²` | `~2.75 GB` |
| `90 km²` | `~4.95 GB` |

The `90 km²` row is a rough Zurich-scale planning estimate, not an official city-boundary measurement.

### Tile Counts

For the actual course tile size:

- one `25 m x 25 m` tile is about `250 x 250` pixels at `10 cm`
- per `1 km²`, a non-overlapping grid yields `1,600` task tiles

If a sliding-window scanner is used instead of a non-overlapping grid:

| Window step | Approx. windows per `1 km²` |
| --- | --- |
| `25 m` | `1,600` |
| `10 m` | `9,604` |
| `5 m` | `38,416` |

This is why exhaustive dense scanning should be delayed until the pipeline is proven on smaller areas.

### Exported Training Dataset Size

The exported training set will usually be much smaller than the raw imagery source.

Reasonable first targets:

- pilot dataset: `1,000` to `2,000` labeled tiles total
- good working dataset: `4,000` to `6,000` labeled tiles total
- later expansion: `10,000+` labeled tiles if the pipeline is stable

Storage estimate for exported tile images depends on format and compression, but as a practical planning range:

- `5,000` compressed tile images will usually stay well below the raw-source size of a city-scale download
- `10,000` compressed tile images are still manageable on a laptop if the originals remain outside git

The real storage pressure comes from the source imagery and repeated scan outputs, not from a few thousand final training tiles.

## Operational Guidance

- Keep raw imagery outside git.
- Keep exported datasets outside git or in ignored local paths.
- Keep only metadata, documentation, and lightweight code in the repository.
- Treat the first milestone as a pipeline milestone, not a scale milestone.
- Success for the first milestone means:
  - raw imagery can be retrieved reproducibly
  - tiles can be generated reproducibly
  - pseudo-labels can be created reproducibly
  - a small labeled dataset can be exported
  - a baseline classifier can be trained and evaluated

## MVP Plan

Start with the smallest end-to-end version that proves the full workflow.

### Scope

- use only `2` to `3` cities
- define only `1` small search area per city at the start
- keep the first pass to roughly `1,000` to `2,000` final labeled tiles total
- use a simple train, validation, and test split by area
- train only one baseline classifier first
- keep raw source imagery for the first local run below about `1 GB`

### First End-to-End Run

1. download a small amount of raw imagery
2. generate `25 m x 25 m` tiles
3. run weak pre-labeling
4. manually review only a small uncertain subset
5. export the final labeled dataset
6. train one baseline model
7. evaluate and inspect obvious failure cases

### Local-First Debug Profile

The first real execution should run on the laptop, not on a remote GPU service.

Purpose:

- debug the code locally
- confirm the folder structure and metadata flow
- confirm that tile generation works
- confirm that pre-labeling works at small scale
- confirm that one baseline model can train end to end

Practical limit:

- cap the initial raw imagery download at about `1 GB`

At the current planning estimate of about `55 MB` per `1 km²` at `10 cm`, this means:

- roughly `18 km²` is the hard upper bound
- a safer target is about `10` to `15 km²` total for the first local pass

Recommended shape of the first local run:

- `2` cities at about `5 km²` each, or
- `3` cities at about `3` to `4 km²` each

This is large enough to exercise the whole pipeline but still small enough to rerun locally without too much friction.

### Why This Comes Before Colab or a GPU Server

- local debugging is faster than debugging through remote notebooks
- failures in download, preprocessing, metadata, and export are easier to inspect locally
- the first run is about correctness, not speed
- only after the local pipeline is stable should larger scans and heavier batch jobs move to remote compute

### Current Local Fetch Choice

For the very first real imagery pull, use a minimal official WMTS fetch:

- source layer: `ch.swisstopo.swissimage-product`
- coordinate system: `3857`
- zoom level: `20`
- fetch shape: `5 x 5` tiles per city center
- current seed cities:
  - Zurich
  - Winterthur
  - Chur

Reasoning:

- this is the simplest reliable way to get real project imagery onto the laptop
- the files are small and fast to redownload
- one WMTS tile at this zoom is already close enough to the task scale for local debugging
- it avoids building the full bulk-download pipeline before the first end-to-end run exists

Current result of the first fetch:

- `75` tiles downloaded
- total download size: about `1.28 MB`
- each tile image is `256 x 256` pixels
- raw tile location:
  - `data/raw/local-debug-v1/wmts-3857-z20/`
- fetch manifest:
  - `data/processed/local-debug-v1/manifests/wmts-debug-tiles.csv`

### Current First Labeling Pass

The first labeling pass for the `75` downloaded WMTS tiles was done as a fast manual seed review.

Current result:

- `5` tiles labeled `crosswalk`
- `70` tiles labeled `no_crosswalk`
- current positive cities:
  - Zurich: `3`
  - Chur: `2`
  - Winterthur: `0`

Important interpretation:

- the positive labels are high-confidence positives
- the negative labels are only seed negatives for the MVP run
- these negatives should be revisited later with model-assisted review or a second manual pass

Artifacts:

- review manifest:
  - `data/processed/local-debug-v1/reviews/manual-review-v1.csv`
- review summary:
  - `data/processed/local-debug-v1/reviews/manual-review-v1-summary.md`

### Current Dataset Size Rule

Use a target size per class instead of thinking in terms of one-off balancing.

Working rule:

- define a target cap for each class
- treat `crosswalk` and `no_crosswalk` as two separate capped silos
- inside each class, treat `train`, `val`, and `test` as their own capped sub-silos
- keep collecting new reviewed samples until both silos reach the target
- never overfill either silo beyond the cap
- if one class is currently underfilled, the export uses only the filled amount that both classes can match
- if one split is underfilled for one class, that split is capped for both classes at the lower filled amount

Current local debug target:

- `64` images per class
- split targets inside each class:
  - `train: 45`
  - `val: 10`
  - `test: 9`

Current result:

- available positives: `5`
- available negatives: `70`
- exportable balanced size right now: `5` per class
- current split fill:
  - `train: 3`
  - `val: 0`
  - `test: 2`
- current balanced export size: `10` images total

Artifacts:

- balanced export manifest:
  - `data/processed/local-debug-v1/exports/balanced-seed-v1/labels.csv`
- balanced export images:
  - `data/processed/local-debug-v1/exports/balanced-seed-v1/images/`

This is the correct rule for the current stage because it gives the project a stable target shape.
As more positives are found later, the positive silo fills up and the negative export simply grows with it until both caps are reached.

### Current Full Build Status

The old build that reached `64 / 64` by rotating and varying seed tiles is no longer accepted.
The current full build now uses only unique raw images.

Current raw-only output:

- total dataset size: `62`
- class balance:
  - `31` crosswalk
  - `31` no_crosswalk
- split balance inside each class:
  - `train: 12`
  - `val: 10`
  - `test: 9`

Artifacts:

- raw-only dataset manifest:
  - `data/processed/local-debug-v1/exports/full-capped-raw-only-v1/labels.csv`
- raw-only dataset summary:
  - `data/processed/local-debug-v1/exports/full-capped-raw-only-v1/summary.json`
- raw-only dataset images:
  - `data/processed/local-debug-v1/exports/full-capped-raw-only-v1/images/`

### How The Current Full Build Is Classified

The current full build uses two stages:

1. bootstrap fitting
   - reviewed labels are used only to fit a tiny centroid-based classifier
   - they are not duplicated into synthetic data
2. automatic assignment on raw imagery
   - every exported item is a unique raw tile
   - the classifier ranks tiles by crosswalk probability
   - the pipeline fills positive and negative buckets from opposite ends of that ranking

Important:

- the build no longer creates rotated, mirrored, or color-shifted copies
- the build can only fill buckets up to the amount of unique raw imagery available in each split
- with the current local raw pool there are only `25` raw tiles per split, so `train` cannot exceed `12 / 12`
- reaching the full `64 / 64` target now requires fetching more real raw imagery, especially for `train`

### What Counts as Done for Version 1

Version 1 is successful if:

- the entire pipeline runs once from raw imagery to evaluation
- the output dataset is small but clean enough to train on
- the model trains without infrastructure problems
- the evaluation produces believable results
- the weak points of the pipeline are visible for the next iteration

### What Not to Do in Version 1

- do not start with all of Zurich
- do not aim for perfect labels
- do not tune many models at once
- do not over-optimize the scanner before the baseline exists
- do not treat dataset size as the main goal

## External References

- SWISSIMAGE product page:
  - <https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10>
- SWISSIMAGE product info with file sizes:
  - <https://www.swisstopo.admin.ch/dam/de/sd-web/WchyQCcLkyd9/Produktinfo_SWISSIMAGE10cm_DE.pdf>
- WMTS documentation:
  - <https://docs.geo.admin.ch/visualize-data/wmts.html>
- STAC overview:
  - <https://docs.geo.admin.ch/download-data/stac-api/overview.html>
- GeoAdmin API FAQ:
  - <https://api3.geo.admin.ch/api/faq/index.html>
- 2025 request-limit update:
  - <https://www.geo.admin.ch/en/change-limits-fair-use>
