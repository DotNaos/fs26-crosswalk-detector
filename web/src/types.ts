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
  run_name: string;
  export_name: string;
  target_per_class?: number;
  split_targets?: Record<string, number>;
  scenes: DatasetScene[];
  tiles: DatasetTile[];
  [key: string]: unknown;
};

export type DatasetSummary = {
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
  }>;
};

export type DatasetListEntry = {
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

export type MapBasemap = "osm" | "swisstopo";

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
