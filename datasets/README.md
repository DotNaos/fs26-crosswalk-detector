# Dataset Metadata

This folder is for git-friendly dataset metadata only.

Do not commit generated image files here. The dataset images should be restored from Swisstopo source metadata or cached outside git.

See also:

- `docs/sam3-100k-dataset-pipeline.md`
- `docs/sam3-100k-data-model.md`
- `docs/sam3-100k-runbook.md`

## Layout

```text
datasets/
  <dataset-id>/
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

`dataset.json` is the index. It lists all JSONL shards and their relative paths.

`tiles.jsonl` stores one image metadata object per line. Keep shards small enough for readable diffs and on-demand loading. For the 100k SAM3 dataset, use perimeter or scene/perimeter shards rather than one large file.

## Rules

- JSONL rows are append/update friendly and easier to diff than one huge JSON array.
- YAML is intentionally avoided for tile rows.
- Images, thumbnails, generated caches, ZIPs, and archives are ignored by git.
- Every image row must include Swisstopo reconstruction metadata and the road overlay reference when available.
