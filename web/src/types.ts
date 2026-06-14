export type TileBBoxMercator = [number, number, number, number] | Record<string, number> | null;

export type DatasetScene = {
  scene_id: string;
  city: string;
  split: string;
  tile_count: number;
  min_row?: number;
  max_row?: number;
  min_col?: number;
  max_col?: number;
  [key: string]: unknown;
};

export type DatasetTile = {
  tile_id: string;
  scene_id: string;
  city: string;
  split: string;
  row: number;
  col: number;
  relative_path: string;
  image_path: string;
  bbox_mercator: TileBBoxMercator;
  clip_probability: number;
  heuristic_probability: number;
  combined_probability: number;
  predicted_label: string;
  label: string;
  selected: boolean;
  status: string;
  review_source: string;
  [key: string]: unknown;
};

export type DatasetContract = {
  display_name?: string;
  run_name: string;
  export_name: string;
  target_per_class?: number;
  split_targets?: Record<string, number>;
  scenes: DatasetScene[];
  tiles: DatasetTile[];
  [key: string]: unknown;
};

export type DatasetSummary = {
  display_name?: string;
  run_name: string;
  export_name: string;
  target_per_class?: number | null;
  split_targets?: Record<string, number>;
  total_tiles: number;
  selected_tiles: number;
  selected_crosswalk: number;
  selected_no_crosswalk: number;
  dropped_tiles: number;
  label_counts?: Record<string, number>;
  scenes: DatasetScene[];
};

export type ScenePayload = {
  summary: DatasetSummary;
  scene: DatasetScene;
  tiles: DatasetTile[];
};

export type DatasetViewportCluster = {
  id: string;
  bbox_mercator: [number, number, number, number];
  center_mercator: [number, number];
  total: number;
  crosswalk: number;
  no_crosswalk: number;
  labels: Record<string, number>;
  splits: Record<string, number>;
  cities: Record<string, number>;
};

export type DatasetViewportPayload = {
  summary: DatasetSummary;
  bbox_mercator: [number, number, number, number];
  zoom: number;
  mode: "tiles" | "clusters";
  total_matching: number;
  returned_tiles: number;
  returned_clusters: number;
  tiles: DatasetTile[];
  clusters: DatasetViewportCluster[];
};

export type RealDatasetConfig = {
  display_name?: string;
  run_name: string;
  export_name: string;
  target_per_class: number;
  tile_size_m: number;
  split_ratios: Record<string, number>;
  selection: {
    positive_min_combined: number;
    positive_min_road_surface: number;
    positive_min_heuristic: number;
    negative_max_combined: number;
    negative_positive_penalty: number;
  };
  scenes: Array<{
    scene_id: string;
    city: string;
    split: string;
    latitude: number;
    longitude: number;
    size_m: number;
    image_px: number;
    autopilot_rank?: number;
    autopilot_score?: number;
    autopilot_city_id?: string;
    autopilot_cell_id?: string;
  }>;
  autopilot?: unknown;
};

export type DatasetListEntry = {
  display_name?: string;
  run_name: string;
  export_name: string;
  tile_count: number;
  selected_count: number;
  path: string;
};

export type LabelDecision = "crosswalk" | "no_crosswalk" | "drop";

export type LabelSourceKind = "model" | "human";

export type LabelSource = {
  source_id: string;
  kind: LabelSourceKind;
  priority: number;
  display_name: string;
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
  cell_id: string;
  perimeter_id: string;
  surface_ratio: number;
  road_pixel_ratio?: number;
  line_density_score?: number;
  cluster_score?: number;
  tile_road_coverage?: number;
  nearest_road_distance_m?: number;
};

export type MetadataDatasetImage = {
  image_id: string;
  tile_id: string;
  source_scene_id: string;
  perimeter_id: string;
  scene_id: string;
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

export type MetadataTilePage = {
  dataset: MetadataDatasetIndex;
  rows: MetadataDatasetImage[];
  next_cursor: string | null;
  returned_count: number;
};

export type TileUpdate = {
  label: string;
  selected: boolean;
  combined_probability?: number;
  predicted_label?: string;
  review_source?: string;
};

export type TileBatchUpdate = TileUpdate & {
  tile_id: string;
};

export type BrowserLabelSuggestion = {
  tile_id: string;
  label: "crosswalk" | "no_crosswalk";
  score: number;
  peak: number;
  coverage: number;
  prompt: string;
  selected: boolean;
  review_source: string;
};

export type MapBasemap = "osm" | "swisstopo" | "roads";

export type RoadClusterCell = {
  id: string;
  row: number;
  col: number;
  sizeM: number;
  surfaceRatio: number;
  densityScore: number;
  roadPixelRatio?: number;
  lineDensityScore?: number;
  clusterScore?: number;
  bboxMercator: [number, number, number, number];
};

export type RoadGridLine = {
  id: string;
  positions: [[number, number], [number, number]];
};

export type RoadClusterGrid = {
  sourceLayer: string;
  method: string;
  zoom: number;
  cellSizeM: number;
  bboxMercator: [number, number, number, number];
  surfaceThreshold: number;
  surfaceCoverage: number;
  cells: RoadClusterCell[];
};

export type SceneReviewState = {
  scan_radius: number;
  scan_delay_ms: number;
  scanned_tile_ids: string[];
};

export type ReviewState = {
  selected_scene_id?: string;
  selected_tile_id?: string;
  map_zoom?: number;
  scenes: Record<string, SceneReviewState>;
};

export type RemoteControllerConfig = {
  server_id?: string;
  server_name?: string;
  host: string;
  username: string;
  port: number;
  repo_path: string;
  execution_mode: RemoteExecutionMode;
  sbatch_script_path: string;
  direct_run_command: string;
  partition: string;
  time_limit: string;
  poll_interval_seconds: number;
};

export type RemoteExecutionMode = "slurm" | "direct";

export type RemoteServerOption = {
  id: string;
  label: string;
  kind?: "ssh" | "fake";
  hostname?: string;
  host: string;
  username: string;
  port: number;
  repo_path: string;
  execution_mode?: RemoteExecutionMode;
  sbatch_script_path?: string;
  direct_run_command?: string;
  partition?: string;
  time_limit?: string;
};

export type RemoteControllerSnapshot = {
  config: RemoteControllerConfig;
  server_options: RemoteServerOption[];
  selected_server_id: string | null;
  connected: boolean;
  password_configured: boolean;
  sshpass_available: boolean;
  expect_available: boolean;
  password_transport_available: boolean;
  tmux_available: boolean;
  remote_home: string | null;
  remote_hostname: string | null;
  last_error: string | null;
};

export type RemoteJobStatus =
  | "idle"
  | "bootstrapping"
  | "syncing"
  | "submitting"
  | "queued"
  | "running"
  | "completed"
  | "cancelled"
  | "failed";

export type RemoteScanJobRecord = {
  id: string;
  scene_id: string;
  scene_label: string;
  tile_count: number;
  created_at: string;
  updated_at: string;
  tmux_session: string;
  log_tmux_session: string | null;
  status: RemoteJobStatus;
  execution_mode: RemoteExecutionMode;
  remote_state: string | null;
  slurm_job_id: string | null;
  error: string | null;
  result_available: boolean;
  log_tail: string[];
  summary: {
    total: number;
    crosswalk: number;
    no_crosswalk: number;
  } | null;
  live_results: Record<string, BrowserLabelSuggestion>;
  live_scanned_tile_ids: string[];
};
