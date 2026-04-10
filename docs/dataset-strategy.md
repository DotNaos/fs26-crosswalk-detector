# Dataset Strategy

## Goal

Build a city-scale dataset for crosswalk detection with as little manual labeling as possible.

The current high-level decisions for data source, APIs, model roles, compute usage, and data volume are tracked in `docs/project-decisions.md`.

## Core Idea

Use a staged pipeline:

1. Start from a small number of seed cities.
2. Prefer candidate areas near city centers at the beginning because crosswalk density is usually higher there.
3. Discretize the city area into tiles.
4. Sample tiles with controlled randomness instead of scanning every square meter immediately.
5. Use heuristics and a general-purpose vision model to pre-classify tiles.
6. Keep only a small manual review step for uncertain cases and for spot checks.
7. Train a dedicated project model on the resulting dataset.

## Initial Working Assumption

The first pass should not try to cover all of Zurich or all available cities exhaustively.
Instead, it should generate a useful and diverse first dataset quickly.

## Proposed Pipeline

### 1. Choose Cities

Start with a handful of cities with different visual structure:

- one large city
- one medium city
- one smaller town

This should reduce the risk that the model only learns one urban style.

### 2. Define Seed Areas

For each city:

- take an approximate city center
- define one or more initial search areas around that center
- expand later once the pipeline works

The center-first strategy is only meant as a bootstrap step, not as the final sampling policy.

### 3. Discretize the Map

Split the search area into tiles that match the project task:

- tile size: 25 m x 25 m

Use a grid over the area and track tile IDs, coordinates, city name, and source image metadata.

### 4. Controlled Sampling

Do not scan every tile immediately.

Instead:

- sample a random subset of tiles from the grid
- bias the sample toward road-heavy areas
- later increase coverage as the pipeline improves

This creates useful "noise" in the data without forcing a full brute-force city scan on day one.

### 5. Pre-Classification

For each sampled tile:

- run heuristics based on map context if available
- run a general-purpose vision model or detector
- optionally refine with segmentation

Output one of three states:

- likely crosswalk
- likely no crosswalk
- uncertain

### 6. Small Human Review

Manual review should be limited to:

- uncertain tiles
- random spot checks from the likely positive bucket
- random spot checks from the likely negative bucket

This keeps the manual load small while still giving a basic quality check.

### 7. Label Construction

Build the training labels from the reviewed and pre-classified buckets:

- `crosswalk`
- `no_crosswalk`

Keep uncertainty flags and confidence values in the metadata so the dataset remains auditable.

### 8. Train and Evaluate

Train a dedicated project model on the resulting dataset.

Evaluation should avoid leakage between near-identical neighboring tiles.
Splits should therefore be done by larger geographic units, not by pure random tile shuffle.

## Important Risks

### City-Center Bias

If the first dataset is built mostly from centers, the model may overfit to:

- dense urban road markings
- specific road materials
- specific camera viewpoints

Mitigation:

- add outer districts later
- add different cities early

### Duplicate or Near-Duplicate Tiles

Neighboring tiles can be visually almost identical.

Mitigation:

- deduplicate by location
- split train, validation, and test by area rather than by random tile

### Weak Labels

Automatic labels will contain mistakes.

Mitigation:

- keep confidence scores
- review uncertain cases
- measure quality by spot checks

## Recommended First Version

First build a small but diverse pilot dataset:

- a few cities
- a few search areas per city
- random but road-aware tile sampling
- lightweight manual review only where needed

The goal of the first version is not perfect coverage.
The goal is a stable pipeline that can later scale outward "like a virus" from a few seed areas into more of each city.

## Next Step

Implement the first pipeline version in this order:

1. city and area definition
2. tile grid generation
3. tile sampling
4. pre-classification
5. review queue
6. dataset export
