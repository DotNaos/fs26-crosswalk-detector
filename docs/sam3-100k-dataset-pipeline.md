# SAM3 100k Dataset Pipeline

This document is the working plan for the larger metadata-only crosswalk dataset.
The order is intentional:

1. write the pipeline and data contract down in Markdown
2. align the TypeScript and Python data models with that contract
3. implement the generator, remote SAM3 run, merge step, and review UI

## Target

- Dataset id: `sam3-100k-v1`
- Final size: `100,000` image metadata rows
- Tile size: `25 m x 25 m`
- Final labels: `crosswalk`, `no_crosswalk`, or `drop`
- Primary automatic label source: `SAM3.1`
- Highest-priority label source: human review
- Git contents: metadata, docs, and code only
- Not in git: images, thumbnails, raw imagery, caches, ZIPs, and archives

The dataset must be reconstructable from metadata. A small Python script plus the
metadata folder should be enough to download or crop the exact images again from
Swisstopo source imagery.

## Pipeline Diagram

```text
+----------------------+     +------------------------+
| Scene/perimeter plan | --> | Swisstopo source refs  |
| cities, splits, area |     | COG/STAC, bbox, CRS    |
+----------------------+     +-----------+------------+
                                      |
                                      v
+----------------------+     +------------------------+
| Road overlay/grid    | --> | Metadata scaffold      |
| road cells, scores   |     | JSONL shards, no imgs  |
+----------------------+     +-----------+------------+
                                      |
                                      v
+----------------------+     +------------------------+
| Remote job builder   | --> | Mercury Slurm + SAM3   |
| shard jobs, smoke    |     | model votes per image  |
+----------------------+     +-----------+------------+
                                      |
                                      v
+----------------------+     +------------------------+
| Result collector     | --> | Review UI              |
| merge votes, resolve |     | endless gallery/table  |
+----------------------+     +-----------+------------+
                                      |
                                      v
                            +------------------------+
                            | Training export        |
                            | reviewed/resolved rows |
                            +------------------------+
```

## Stages

### 1. Scene and Perimeter Planning

The generator starts from scene definitions. A scene is a Swisstopo-reconstructable
area with a city, split, bounding box, tile size, and source imagery reference.

For `800 m x 800 m` scenes with non-overlapping `25 m` tiles:

- one scene has `32 x 32 = 1,024` images
- the current `configs/real-dataset-10k.toml` has `28` scenes
- that is `28,672` possible images before filtering
- `100,000` images need about `98` such scenes before filtering

This means the 100k run needs a larger scene plan than the current 10k config.

### 2. Swisstopo Reconstruction Metadata

Every row must keep enough source information to rebuild the image:

- source product, for example `SWISSIMAGE`
- source access method, preferably `STAC` plus downloadable `COG`
- source asset id or URL
- CRS, expected `EPSG:2056`
- scene bbox and tile bbox
- tile row, column, tile size, and crop pixels
- local cache key or relative restored image path

The metadata row is the durable object. The image file is disposable cache.

### 3. Road Overlay

Road data is not the final label. It is used for sampling, priority, and later
auditing.

Each image row should point to the road overlay cell or perimeter used when it was
created. The row should not duplicate a large geometry. It stores a small
`road_overlay_ref` with identifiers and compact scores.

### 4. Metadata Scaffold

Before SAM3 runs, we create the folder tree and JSONL shards with empty or
placeholder label votes. The structure is:

```text
datasets/
  sam3-100k-v1/
    dataset.json
    label-sources.json
    sources/
      swisstopo.json
    scenes/
      <scene-id>/
        scene.json
        road-overlay.json
        perimeters/
          <perimeter-id>/
            perimeter.json
            tiles.jsonl
```

JSONL is used for image rows because it can be paged, streamed, filtered, and
diffed without loading one giant JSON array.

### 5. Remote SAM3 Labeling

The first real automatic label source is SAM3.1. Older heuristics and CLIP-like
signals can remain useful for sampling or diagnostics, but they should not be the
authoritative automatic label for this dataset.

Remote execution shape:

1. build small smoke jobs from one or two shards
2. submit to Mercury with Slurm
3. confirm CUDA and SAM3 load correctly
4. collect model votes back into JSON result files
5. merge model votes into the JSONL rows
6. run the full shard queue only after the smoke result looks correct

### 6. Label Resolution

The row stores all label votes. It does not assume a fixed number of models.

Resolution rules:

1. a human vote wins over model votes
2. otherwise the highest-priority source wins
3. if later needed, weighted voting can be enabled without changing the row shape

This keeps the dataset auditable. We can always see which model said what, when a
human overrode it, and which label was used for training.

### 7. Review UI

The review UI should be a fast gallery or endless table first, not map-first.

Needed filters:

- city
- split
- scene
- perimeter
- resolved label
- review state
- selected for training
- source disagreement
- low confidence

Needed actions:

- mark as `crosswalk`
- mark as `no_crosswalk`
- mark as `drop`
- mark reviewed
- select or unselect for training
- show source image metadata and road overlay scores

### 8. Training Export

Training export should read only resolved metadata rows and then materialize
images outside git.

Recommended first export policy:

- use reviewed human rows directly
- use high-confidence SAM3 rows only after spot checks
- exclude `drop`
- split by scene or perimeter, not by random tile
- keep an export manifest with the exact metadata dataset version

## Scale Estimates

At `25 m x 25 m`, a non-overlapping grid has `1,600` images per `1 km2`.

Switzerland-scale planning:

| Area basis | Approx. non-overlapping images |
| --- | ---: |
| `41,285 km2` country area | `66.1 million` |
| `42,700` SWISSIMAGE 1 km tiles | `68.3 million` |

Dense sliding windows are much larger:

| Step size | Images per `1 km2` | Switzerland-scale images |
| --- | ---: | ---: |
| `25 m` | `1,600` | `68.3 million` |
| `10 m` | `9,604` | `410.1 million` |
| `5 m` | `38,416` | `1.64 billion` |

Metadata size depends on row verbosity. Compact JSONL rows with reconstruction,
road overlay reference, one SAM3 vote, and resolution fields should be planned at
roughly `2 KB` to `4 KB` each before git compression.

| Row count | Approx. compact JSONL size |
| ---: | ---: |
| `1,000` | `2 MB` to `4 MB` |
| `100,000` | `200 MB` to `400 MB` |
| `1,000,000` | `2 GB` to `4 GB` |

For git, `100,000` rows is acceptable only if sharded. Keep each shard below
about `5,000` rows, and preferably smaller when the rows are verbose.

## Finishing Criteria Before Full GPU Run

- The dataset scaffold can be generated without image files in git.
- Every row has Swisstopo reconstruction metadata.
- Every row has a road overlay reference when road context exists.
- `dataset.json` lists all shards and row counts.
- The review UI can page through the metadata on demand.
- A Mercury smoke job creates SAM3 votes for a small shard.
- The merge step preserves all votes and resolves labels deterministically.
- Existing web and metadata tests pass.
