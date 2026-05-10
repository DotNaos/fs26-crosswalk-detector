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
  scenes: DatasetScene[];
};

export type ScenePayload = {
  summary: DatasetSummary;
  scene: DatasetScene;
  tiles: DatasetTile[];
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
