# SAM3 100k Data Model

This is the Markdown contract for the metadata dataset. Code should follow this
shape unless a later document changes it deliberately.

## Main Concepts

- `Dataset`: one versioned metadata dataset, for example `sam3-100k-v1`
- `Scene`: a reconstructable Swisstopo area with one split
- `Perimeter`: a smaller shardable area inside a scene
- `Image row`: one `25 m x 25 m` tile image metadata object
- `Label source`: a model or human that can vote on an image
- `Label vote`: one source's decision for one image
- `Resolved label`: the label currently used by the dataset
- `Road overlay ref`: compact link from the image row to the road grid data

## Label Resolution

There is no fixed limit on how many models can label one image.

Each source has a `priority`. Higher priority wins when no human override exists.
Human labels have the highest priority.

Suggested first priorities:

| Source | Kind | Priority |
| --- | --- | ---: |
| `human:<user>` | `human` | `1000` |
| `sam3.1` | `model` | `100` |
| `legacy-heuristic` | `model` | `10` |

Resolution modes:

- `human_override`: a human vote selected the final label
- `priority`: the highest-priority non-human source selected the final label
- `weighted_vote`: future mode for combining several model votes

## TypeScript Contract

```ts
export type LabelDecision = "crosswalk" | "no_crosswalk" | "drop";

export type LabelSourceKind = "model" | "human";

export type LabelSource = {
  source_id: string;
  kind: LabelSourceKind;
  priority: number;
  display_name: string;
  version?: string;
};

export type ImageLabelVote = {
  vote_id: string;
  source: LabelSource;
  decision: LabelDecision;
  confidence?: number;
  created_at: string;
  metadata?: Record<string, unknown>;
};

export type ResolvedLabel = {
  decision: LabelDecision;
  source_id: string;
  source_kind: LabelSourceKind;
  resolved_by: "human_override" | "priority" | "weighted_vote";
  confidence?: number;
  updated_at: string;
};

export type SwisstopoSourceRef = {
  provider: "swisstopo";
  product: "SWISSIMAGE";
  access: "stac-cog" | "wmts";
  crs: "EPSG:2056";
  asset_id?: string;
  asset_url?: string;
  acquisition_year?: number;
  resolution_m?: number;
};

export type TileReconstruction = {
  source_scene_id: string;
  row: number;
  col: number;
  tile_size_m: number;
  tile_bbox_mercator: [number, number, number, number];
  crop_px: {
    left: number;
    top: number;
    width: number;
    height: number;
  };
  relative_path: string;
};

export type RoadOverlayRef = {
  overlay_id: string;
  perimeter_id: string;
  cell_id: string;
  surface_ratio: number;
  road_pixel_ratio?: number;
  line_density_score?: number;
  cluster_score?: number;
  tile_road_coverage?: number;
  nearest_road_distance_m?: number;
};

export type DatasetImage = {
  image_id: string;
  tile_id: string;

  scene_id: string;
  source_scene_id: string;
  perimeter_id: string;
  city: string;
  split: "train" | "val" | "test";

  row: number;
  col: number;
  bbox_mercator: [number, number, number, number];

  swisstopo: SwisstopoSourceRef;
  reconstruction: TileReconstruction;
  road_overlay_ref?: RoadOverlayRef;

  labels: ImageLabelVote[];

  resolved_label: ResolvedLabel;

  review_state: "unreviewed" | "reviewed" | "disputed" | "dropped";
  selected_for_training: boolean;

  image_path?: string;
  thumbnail_path?: string;
};
```

The current web type is named `MetadataDatasetImage`. It should stay compatible
with this `DatasetImage` contract. If names differ, the field meanings should not.

## Dataset Index

`dataset.json` is the small file the UI reads first.

```ts
export type MetadataDatasetShard = {
  shard_id: string;
  path: string;
  tile_count: number;
  scene_id?: string;
  perimeter_id?: string;
};

export type MetadataDatasetIndex = {
  format: "crosswalk-jsonl-v1";
  dataset_id: string;
  display_name?: string;
  run_name: string;
  export_name: string;
  tile_count: number;
  selected_count: number;
  shard_target_count: number;
  shards: MetadataDatasetShard[];
};
```

## Example JSONL Row

Each line in `tiles.jsonl` is one compact JSON object.

```json
{"image_id":"sam3-100k-v1:zurich-nw:p0007:r12:c08","tile_id":"zurich-nw-p0007-r12-c08","scene_id":"zurich-nw","source_scene_id":"swissimage-2024-zurich-nw","perimeter_id":"p0007","city":"Zurich","split":"train","row":12,"col":8,"bbox_mercator":[2683125,1247525,2683150,1247550],"swisstopo":{"provider":"swisstopo","product":"SWISSIMAGE","access":"stac-cog","crs":"EPSG:2056","asset_id":"swissimage-2024-2683-1247","asset_url":"https://example.invalid/swissimage/cog.tif","acquisition_year":2024,"resolution_m":0.1},"reconstruction":{"source_scene_id":"swissimage-2024-zurich-nw","row":12,"col":8,"tile_size_m":25,"tile_bbox_mercator":[2683125,1247525,2683150,1247550],"crop_px":{"left":3000,"top":2000,"width":250,"height":250},"relative_path":"images/zurich-nw/p0007/r12-c08.jpg"},"road_overlay_ref":{"overlay_id":"roads-v1-zurich-nw","perimeter_id":"p0007","cell_id":"zurich-nw-p0007-r12-c08","surface_ratio":0.44,"road_pixel_ratio":0.31,"line_density_score":0.62,"cluster_score":0.71,"tile_road_coverage":0.38},"labels":[{"vote_id":"sam3.1:2026-05-15T10:00:00Z:zurich-nw-p0007-r12-c08","source":{"source_id":"sam3.1","kind":"model","priority":100,"display_name":"SAM3.1","version":"sam3@main"},"decision":"crosswalk","confidence":0.87,"created_at":"2026-05-15T10:00:00Z","metadata":{"prompt":"marked pedestrian crossing in aerial image","mask_count":2,"coverage":0.018}}],"resolved_label":{"decision":"crosswalk","source_id":"sam3.1","source_kind":"model","resolved_by":"priority","confidence":0.87,"updated_at":"2026-05-15T10:00:00Z"},"review_state":"unreviewed","selected_for_training":true}
```

## Postgres Mapping

The JSONL format should also map cleanly to Postgres later:

- `datasets`
- `scenes`
- `perimeters`
- `dataset_images`
- `label_sources`
- `image_label_votes`
- `resolved_labels`
- `road_overlay_refs`
- `swisstopo_sources`

The JSONL row can be imported as one document first, then normalized into these
tables when querying or collaborative review needs it.

## Git Rules

- Commit `dataset.json`, source manifests, scene metadata, perimeter metadata,
  road overlay summaries, and `tiles.jsonl`.
- Do not commit image files.
- Do not commit thumbnails.
- Do not commit source COGs.
- Do not commit generated ZIPs.
- Keep JSONL shards small enough that reviewing a diff remains possible.
