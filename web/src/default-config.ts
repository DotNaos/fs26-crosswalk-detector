import type { RealDatasetConfig } from "./types";

export const DEFAULT_BROWSER_CONFIG: RealDatasetConfig = {
  run_name: "browser-v1",
  export_name: "crosswalk-browser-dataset",
  target_per_class: 128,
  tile_size_m: 25,
  split_ratios: {
    train: 0.7,
    val: 0.15,
    test: 0.15,
  },
  selection: {
    positive_min_combined: 0.6,
    positive_min_road_surface: -0.02,
    positive_min_heuristic: 0.5,
    negative_max_combined: 0.55,
    negative_positive_penalty: 0.35,
  },
  scenes: [
    { scene_id: "zurich-center", city: "Zurich", split: "train", latitude: 47.3769, longitude: 8.5417, size_m: 800, image_px: 2048 },
    { scene_id: "basel-center", city: "Basel", split: "train", latitude: 47.5596, longitude: 7.5886, size_m: 800, image_px: 2048 },
    { scene_id: "bern-center", city: "Bern", split: "train", latitude: 46.948, longitude: 7.4474, size_m: 800, image_px: 2048 },
    { scene_id: "winterthur-center", city: "Winterthur", split: "val", latitude: 47.499, longitude: 8.7241, size_m: 800, image_px: 2048 },
    { scene_id: "lucerne-center", city: "Lucerne", split: "val", latitude: 47.0502, longitude: 8.3093, size_m: 800, image_px: 2048 },
    { scene_id: "chur-center", city: "Chur", split: "test", latitude: 46.8508, longitude: 9.5329, size_m: 800, image_px: 2048 },
    { scene_id: "st-gallen-center", city: "St. Gallen", split: "test", latitude: 47.4245, longitude: 9.3767, size_m: 800, image_px: 2048 },
  ],
};
