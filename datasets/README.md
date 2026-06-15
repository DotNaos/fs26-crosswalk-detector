# Dataset Metadata

This folder is for git-friendly dataset metadata only.

Do not commit generated image files here. The dataset images should be restored from Swisstopo source metadata or cached outside git.

The final public dataset is distributed as a GitHub Release archive and restored
to `web/public/static-datasets/` by `uv run dataset`, `uv run train`,
`uv run test`, or `python3 scripts/download_submission_assets.py`. The raw
Swisstopo images are downloaded locally only when a command needs image pixels.

See also:

- `docs/static-dataset-deployment.md`
- `docs/project-decisions.md`

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

`tiles.jsonl` stores one image metadata object per line. Keep shards small enough for readable diffs and on-demand loading.

## Rules

- JSONL rows are append/update friendly and easier to diff than one huge JSON array.
- YAML is intentionally avoided for tile rows.
- Images, thumbnails, generated caches, ZIPs, and archives are ignored by git.
- Every image row must include Swisstopo reconstruction metadata and the road overlay reference when available.
