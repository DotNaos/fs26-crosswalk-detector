import { DEFAULT_BROWSER_CONFIG } from "./default-config";
import { clearSceneRasterCache, ensureSceneTileImages } from "./scene-imagery";
import { latLngToMercator } from "./utils";
import type { DatasetContract, DatasetListEntry, DatasetScene, DatasetSummary, DatasetTile, RealDatasetConfig, ReviewState, ScenePayload, TileBatchUpdate, TileUpdate } from "./types";

const CONFIG_KEY = "crosswalk.browser.config.v1";
const REVIEW_STATE_KEY = "crosswalk.browser.review-state.v1";
const TILE_OVERRIDES_KEY = "crosswalk.browser.tile-overrides.v1";

type TileOverride = Partial<DatasetTile>;
type TileOverrideMap = Record<string, TileOverride>;

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown) {
  window.localStorage.setItem(key, JSON.stringify(value));
}

function baseSceneFromConfig(scene: RealDatasetConfig["scenes"][number], tileSizeM: number): DatasetScene {
  const grid = Math.round(scene.size_m / tileSizeM);
  return {
    scene_id: scene.scene_id,
    city: scene.city,
    split: scene.split,
    tile_count: grid * grid,
    min_row: 0,
    max_row: grid - 1,
    min_col: 0,
    max_col: grid - 1,
    latitude: scene.latitude,
    longitude: scene.longitude,
    size_m: scene.size_m,
    image_px: scene.image_px,
  };
}

function buildBaseTiles(config: RealDatasetConfig): DatasetTile[] {
  return config.scenes.flatMap((scene) => {
    const center = latLngToMercator(scene.latitude, scene.longitude);
    const half = scene.size_m / 2;
    const tileSize = config.tile_size_m;
    const grid = Math.round(scene.size_m / tileSize);

    return Array.from({ length: grid * grid }, (_, index) => {
      const row = Math.floor(index / grid);
      const col = index % grid;
      const minX = center.x - half + col * tileSize;
      const maxX = minX + tileSize;
      const maxY = center.y + half - row * tileSize;
      const minY = maxY - tileSize;
      const tileId = `${scene.scene_id}:r${String(row).padStart(2, "0")}:c${String(col).padStart(2, "0")}`;

      return {
        tile_id: tileId,
        scene_id: scene.scene_id,
        city: scene.city,
        split: scene.split,
        row,
        col,
        relative_path: `${scene.scene_id}/r${row}-c${col}.jpg`,
        image_path: "",
        bbox_mercator: [minX, minY, maxX, maxY],
        clip_probability: 0,
        heuristic_probability: 0,
        combined_probability: 0,
        predicted_label: "unknown",
        label: "unknown",
        selected: false,
        status: "pending",
        review_source: "browser-local",
      } satisfies DatasetTile;
    });
  });
}

function mergeTile(tile: DatasetTile, overrides: TileOverrideMap): DatasetTile {
  return {
    ...tile,
    ...(overrides[tile.tile_id] ?? {}),
    image_path: tile.image_path,
  };
}

function datasetTiles(config: RealDatasetConfig, overrides: TileOverrideMap) {
  return buildBaseTiles(config).map((tile) => mergeTile(tile, overrides));
}

function buildSummary(config: RealDatasetConfig, tiles: DatasetTile[]): DatasetSummary {
    const scenes = config.scenes.map((scene) => baseSceneFromConfig(scene, config.tile_size_m));
  return {
    run_name: config.run_name,
    export_name: config.export_name,
    target_per_class: config.target_per_class,
    split_targets: {},
    total_tiles: tiles.length,
    selected_tiles: tiles.filter((tile) => tile.selected).length,
    selected_crosswalk: tiles.filter((tile) => tile.selected && tile.label === "crosswalk").length,
    selected_no_crosswalk: tiles.filter((tile) => tile.selected && tile.label === "no_crosswalk").length,
    dropped_tiles: tiles.filter((tile) => !tile.selected).length,
    scenes,
  };
}

function currentOverrides() {
  return readJson<TileOverrideMap>(TILE_OVERRIDES_KEY, {});
}

function currentConfig() {
  return readJson<RealDatasetConfig>(CONFIG_KEY, DEFAULT_BROWSER_CONFIG);
}

export function listBrowserDatasets(): DatasetListEntry[] {
  const config = currentConfig();
  const tiles = datasetTiles(config, currentOverrides());
  return [
    {
      run_name: config.run_name,
      export_name: config.export_name,
      tile_count: tiles.length,
      selected_count: tiles.filter((tile) => tile.selected).length,
      path: "browser-local",
    },
  ];
}

export function loadBrowserConfig() {
  return currentConfig();
}

export function saveBrowserConfig(config: RealDatasetConfig) {
  const previous = currentConfig();
  writeJson(CONFIG_KEY, config);
  if (previous.tile_size_m !== config.tile_size_m || previous.scenes.length !== config.scenes.length) {
    writeJson(TILE_OVERRIDES_KEY, {});
    writeJson(REVIEW_STATE_KEY, { scenes: {} } satisfies ReviewState);
    clearSceneRasterCache();
  }
  return config;
}

export function loadBrowserReviewState() {
  return readJson<ReviewState>(REVIEW_STATE_KEY, { scenes: {} });
}

export function saveBrowserReviewState(state: ReviewState) {
  writeJson(REVIEW_STATE_KEY, state);
  return state;
}

export function loadBrowserDatasetMeta() {
  const config = currentConfig();
  return buildSummary(config, datasetTiles(config, currentOverrides()));
}

export function loadBrowserDataset(): DatasetContract {
  const config = currentConfig();
  const overrides = currentOverrides();
  return {
    run_name: config.run_name,
    export_name: config.export_name,
    target_per_class: config.target_per_class,
    split_targets: {},
    scenes: config.scenes.map((scene) => baseSceneFromConfig(scene, config.tile_size_m)),
    tiles: datasetTiles(config, overrides),
  };
}

function applySceneImagePaths(tiles: DatasetTile[], imageUrls?: Map<string, string>) {
  if (!imageUrls) return tiles;
  return tiles.map((tile) => ({
    ...tile,
    image_path: imageUrls.get(tile.tile_id) ?? tile.image_path,
  }));
}

export async function hydrateBrowserSceneImages(sceneId: string) {
  const config = currentConfig();
  const overrides = currentOverrides();
  const scene = config.scenes.find((entry) => entry.scene_id === sceneId);
  if (!scene) {
    throw new Error(`Unknown scene: ${sceneId}`);
  }
  const tiles = datasetTiles(config, overrides).filter((tile) => tile.scene_id === sceneId);
  return ensureSceneTileImages(scene, tiles);
}

function persistOverrides(next: TileOverrideMap) {
  writeJson(TILE_OVERRIDES_KEY, next);
}

export async function loadBrowserScene(sceneId: string, options?: { includeImages?: boolean }): Promise<ScenePayload> {
  const config = currentConfig();
  const overrides = currentOverrides();
  const scene = config.scenes.find((entry) => entry.scene_id === sceneId);
  if (!scene) {
    throw new Error(`Unknown scene: ${sceneId}`);
  }

  const tiles = datasetTiles(config, overrides).filter((tile) => tile.scene_id === sceneId);
  const imageUrls = options?.includeImages ? await ensureSceneTileImages(scene, tiles) : undefined;
  return {
    summary: buildSummary(config, datasetTiles(config, overrides)),
    scene: baseSceneFromConfig(scene, config.tile_size_m),
    tiles: applySceneImagePaths(tiles, imageUrls),
  };
}

export async function updateBrowserTile(tileId: string, update: TileUpdate) {
  const overrides = currentOverrides();
  overrides[tileId] = {
    ...(overrides[tileId] ?? {}),
    ...update,
    status: update.selected ? "labeled" : "dropped",
  };
  persistOverrides(overrides);
  const sceneId = tileId.split(":")[0];
  return loadBrowserScene(sceneId);
}

export async function updateBrowserTiles(sceneId: string, updates: TileBatchUpdate[]) {
  const overrides = currentOverrides();
  for (const update of updates) {
    overrides[update.tile_id] = {
      ...(overrides[update.tile_id] ?? {}),
      ...update,
      status: update.selected ? "labeled" : "dropped",
    };
  }
  persistOverrides(overrides);
  return loadBrowserScene(sceneId);
}
