import { DEFAULT_BROWSER_CONFIG } from "./default-config";
import { buildAutopilotPlan } from "./autopilot-planner";
import type { AutopilotPlan } from "./autopilot-planner";
import { clearSceneRasterCache, ensureSceneTileImages } from "./scene-imagery";
import { latLngToMercator } from "./utils";
import type {
  DatasetContract,
  DatasetListEntry,
  DatasetScene,
  DatasetSummary,
  DatasetTile,
  RealDatasetConfig,
  ReviewState,
  ScenePayload,
  TileBatchUpdate,
  TileUpdate,
} from "./types";

const DATASETS_KEY = "crosswalk.browser.datasets.v2";
const LEGACY_CONFIG_KEY = "crosswalk.browser.config.v1";
const LEGACY_REVIEW_STATE_KEY = "crosswalk.browser.review-state.v1";
const LEGACY_TILE_OVERRIDES_KEY = "crosswalk.browser.tile-overrides.v1";

type TileOverride = Partial<DatasetTile>;
type TileOverrideMap = Record<string, TileOverride>;

type BrowserStoredDataset = {
  config: RealDatasetConfig;
  reviewState: ReviewState;
  tileOverrides: TileOverrideMap;
  created_at: string;
};

type CreateBrowserDatasetInput = {
  name: string;
  sceneId: string;
};

type CreateAutopilotDatasetInput = {
  targetPositiveCount: number;
  maxPanels?: number;
  perimeterBudget?: number;
  plan?: AutopilotPlan;
  name?: string;
};

const IMAGE_SOURCE_RUN_NAME = "real-v1";

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

function slugify(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");
}

function normalizeDisplayName(config: RealDatasetConfig) {
  return config.display_name?.trim() || config.export_name;
}

function normalizeReviewState(reviewState?: ReviewState): ReviewState {
  return reviewState && typeof reviewState === "object" ? reviewState : { scenes: {} };
}

function normalizeAutopilotConfig(config: RealDatasetConfig): RealDatasetConfig {
  const plan = config.autopilot as { mode?: string; coarseGrid?: { rows?: number }; maxPanels?: number; sceneBudget?: number } | undefined;
  if (plan?.mode !== "swiss-lowres-urban-grid" || (plan.coarseGrid?.rows ?? 0) >= 40) {
    return config;
  }
  const nextPlan = buildAutopilotPlan({
    targetPositiveCount: config.target_per_class,
    maxPanels: plan.maxPanels,
    perimeterBudget: plan.sceneBudget,
  });
  return {
    ...config,
    target_per_class: nextPlan.targetPositiveCount,
    tile_size_m: nextPlan.tileSizeM,
    scenes: nextPlan.scenes,
    autopilot: nextPlan,
  };
}

function normalizeStoredDataset(stored: BrowserStoredDataset): BrowserStoredDataset {
  const config = normalizeAutopilotConfig(stored.config);
  return {
    config: {
      ...config,
      display_name: normalizeDisplayName(config),
    },
    reviewState: normalizeReviewState(stored.reviewState),
    tileOverrides: stored.tileOverrides ?? {},
    created_at: stored.created_at ?? new Date().toISOString(),
  };
}

function buildLegacyDataset(): BrowserStoredDataset {
  const legacyConfig = readJson<RealDatasetConfig>(LEGACY_CONFIG_KEY, DEFAULT_BROWSER_CONFIG);
  return normalizeStoredDataset({
    config: {
      ...legacyConfig,
      display_name: normalizeDisplayName(legacyConfig),
    },
    reviewState: readJson<ReviewState>(LEGACY_REVIEW_STATE_KEY, { scenes: {} }),
    tileOverrides: readJson<TileOverrideMap>(LEGACY_TILE_OVERRIDES_KEY, {}),
    created_at: new Date().toISOString(),
  });
}

function loadStoredDatasets(): BrowserStoredDataset[] {
  const stored = readJson<BrowserStoredDataset[] | null>(DATASETS_KEY, null);
  if (stored?.length) {
    return stored.map(normalizeStoredDataset);
  }
  const migrated = [buildLegacyDataset()];
  writeJson(DATASETS_KEY, migrated);
  return migrated;
}

function saveStoredDatasets(datasets: BrowserStoredDataset[]) {
  writeJson(DATASETS_KEY, datasets.map(normalizeStoredDataset));
}

function datasetId(runName: string, exportName: string) {
  return `${runName}::${exportName}`;
}

function findStoredDataset(runName?: string, exportName?: string): BrowserStoredDataset {
  const datasets = loadStoredDatasets();
  if (runName && exportName) {
    const match = datasets.find((entry) => entry.config.run_name === runName && entry.config.export_name === exportName);
    if (match) return match;
  }
  return datasets[0] ?? buildLegacyDataset();
}

function updateStoredDataset(
  runName: string,
  exportName: string,
  updater: (stored: BrowserStoredDataset) => BrowserStoredDataset,
): BrowserStoredDataset {
  const datasets = loadStoredDatasets();
  const index = datasets.findIndex((entry) => entry.config.run_name === runName && entry.config.export_name === exportName);
  if (index < 0) {
    throw new Error(`Unknown browser dataset: ${runName}/${exportName}`);
  }
  const next = normalizeStoredDataset(updater(datasets[index]));
  datasets[index] = next;
  saveStoredDatasets(datasets);
  return next;
}

function baseSceneFromConfig(scene: RealDatasetConfig["scenes"][number], tileSizeM: number): DatasetScene {
  const grid = Math.round(scene.size_m / tileSizeM);
  return {
    ...scene,
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
      const relativePath = `${scene.scene_id}/r${String(row).padStart(2, "0")}-c${String(col).padStart(2, "0")}.jpg`;

      return {
        tile_id: tileId,
        scene_id: scene.scene_id,
        city: scene.city,
        split: scene.split,
        row,
        col,
        relative_path: relativePath,
        image_path: `/assets/processed/${IMAGE_SOURCE_RUN_NAME}/tiles/${relativePath}`,
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
    display_name: normalizeDisplayName(config),
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

function applySceneImagePaths(tiles: DatasetTile[], imageUrls?: Map<string, string>) {
  if (!imageUrls) return tiles;
  return tiles.map((tile) => ({
    ...tile,
    image_path: imageUrls.get(tile.tile_id) ?? tile.image_path,
  }));
}

function uniqueExportName(baseName: string, datasets: BrowserStoredDataset[]) {
  const baseSlug = slugify(baseName) || "dataset";
  let candidate = baseSlug;
  let counter = 2;
  while (datasets.some((entry) => entry.config.export_name === candidate)) {
    candidate = `${baseSlug}-${counter}`;
    counter += 1;
  }
  return candidate;
}

export function listBrowserDatasets(): DatasetListEntry[] {
  return loadStoredDatasets()
    .map((stored) => {
      const tiles = datasetTiles(stored.config, stored.tileOverrides);
      return {
        display_name: normalizeDisplayName(stored.config),
        run_name: stored.config.run_name,
        export_name: stored.config.export_name,
        tile_count: tiles.length,
        selected_count: tiles.filter((tile) => tile.selected).length,
        path: `browser-local:${datasetId(stored.config.run_name, stored.config.export_name)}`,
      } satisfies DatasetListEntry;
    })
    .sort((left, right) => left.display_name?.localeCompare(right.display_name ?? "") ?? 0);
}

export function createBrowserDataset({ name, sceneId }: CreateBrowserDatasetInput): DatasetListEntry {
  const trimmedName = name.trim();
  if (!trimmedName) {
    throw new Error("Dataset name is required.");
  }
  const scene = DEFAULT_BROWSER_CONFIG.scenes.find((entry) => entry.scene_id === sceneId);
  if (!scene) {
    throw new Error("Choose a valid area for the new dataset.");
  }

  const datasets = loadStoredDatasets();
  const exportName = uniqueExportName(trimmedName, datasets);
  const config: RealDatasetConfig = {
    ...DEFAULT_BROWSER_CONFIG,
    display_name: trimmedName,
    export_name: exportName,
    scenes: [scene],
  };

  const stored: BrowserStoredDataset = normalizeStoredDataset({
    config,
    reviewState: {
      selected_scene_id: scene.scene_id,
      scenes: {},
    },
    tileOverrides: {},
    created_at: new Date().toISOString(),
  });

  saveStoredDatasets([...datasets, stored]);
  clearSceneRasterCache();

  const tiles = datasetTiles(stored.config, stored.tileOverrides);
  return {
    display_name: normalizeDisplayName(stored.config),
    run_name: stored.config.run_name,
    export_name: stored.config.export_name,
    tile_count: tiles.length,
    selected_count: 0,
    path: `browser-local:${datasetId(stored.config.run_name, stored.config.export_name)}`,
  };
}

export function createBrowserAutopilotDataset({ targetPositiveCount, maxPanels, perimeterBudget, plan: providedPlan, name }: CreateAutopilotDatasetInput): DatasetListEntry {
  const plan = providedPlan ?? buildAutopilotPlan({ targetPositiveCount, maxPanels, perimeterBudget });
  const datasets = loadStoredDatasets();
  const displayName = name?.trim() || `Autopilot ${plan.targetPositiveCount} positives`;
  const exportName = uniqueExportName(displayName, datasets);
  const firstScene = plan.scenes[0];
  if (!firstScene) {
    throw new Error("Autopilot could not create any search scenes.");
  }

  const config: RealDatasetConfig = {
    ...DEFAULT_BROWSER_CONFIG,
    display_name: displayName,
    export_name: exportName,
    target_per_class: plan.targetPositiveCount,
    tile_size_m: plan.tileSizeM,
    scenes: plan.scenes,
    autopilot: plan,
  };

  const stored: BrowserStoredDataset = normalizeStoredDataset({
    config,
    reviewState: {
      selected_scene_id: firstScene.scene_id,
      scenes: {},
    },
    tileOverrides: {},
    created_at: new Date().toISOString(),
  });

  saveStoredDatasets([...datasets, stored]);
  clearSceneRasterCache();

  const tiles = datasetTiles(stored.config, stored.tileOverrides);
  return {
    display_name: normalizeDisplayName(stored.config),
    run_name: stored.config.run_name,
    export_name: stored.config.export_name,
    tile_count: tiles.length,
    selected_count: 0,
    path: `browser-local:${datasetId(stored.config.run_name, stored.config.export_name)}`,
  };
}

export function loadBrowserConfig(runName?: string, exportName?: string) {
  return findStoredDataset(runName, exportName).config;
}

export function saveBrowserConfig(runName: string, exportName: string, config: RealDatasetConfig) {
  const previous = findStoredDataset(runName, exportName);
  const next = updateStoredDataset(runName, exportName, (stored) => ({
    ...stored,
    config: {
      ...config,
      run_name: runName,
      export_name: exportName,
      display_name: normalizeDisplayName(config),
    },
    reviewState:
      previous.config.tile_size_m !== config.tile_size_m || previous.config.scenes.length !== config.scenes.length
        ? { scenes: {} }
        : stored.reviewState,
    tileOverrides:
      previous.config.tile_size_m !== config.tile_size_m || previous.config.scenes.length !== config.scenes.length
        ? {}
        : stored.tileOverrides,
  }));

  if (previous.config.tile_size_m !== config.tile_size_m || previous.config.scenes.length !== config.scenes.length) {
    clearSceneRasterCache();
  }

  return next.config;
}

export function loadBrowserReviewState(runName?: string, exportName?: string) {
  return findStoredDataset(runName, exportName).reviewState;
}

export function saveBrowserReviewState(runName: string, exportName: string, state: ReviewState) {
  return updateStoredDataset(runName, exportName, (stored) => ({
    ...stored,
    reviewState: normalizeReviewState(state),
  })).reviewState;
}

export function loadBrowserDatasetMeta(runName?: string, exportName?: string) {
  const stored = findStoredDataset(runName, exportName);
  return buildSummary(stored.config, datasetTiles(stored.config, stored.tileOverrides));
}

export function loadBrowserDataset(runName?: string, exportName?: string): DatasetContract {
  const stored = findStoredDataset(runName, exportName);
  return {
    display_name: normalizeDisplayName(stored.config),
    run_name: stored.config.run_name,
    export_name: stored.config.export_name,
    target_per_class: stored.config.target_per_class,
    split_targets: {},
    scenes: stored.config.scenes.map((scene) => baseSceneFromConfig(scene, stored.config.tile_size_m)),
    tiles: datasetTiles(stored.config, stored.tileOverrides),
  };
}

export async function hydrateBrowserSceneImages(sceneId: string, runName?: string, exportName?: string) {
  const stored = findStoredDataset(runName, exportName);
  const scene = stored.config.scenes.find((entry) => entry.scene_id === sceneId);
  if (!scene) {
    throw new Error(`Unknown scene: ${sceneId}`);
  }
  const tiles = datasetTiles(stored.config, stored.tileOverrides).filter((tile) => tile.scene_id === sceneId);
  return ensureSceneTileImages(scene, tiles);
}

export async function loadBrowserScene(
  sceneId: string,
  runName?: string,
  exportName?: string,
  options?: { includeImages?: boolean },
): Promise<ScenePayload> {
  const stored = findStoredDataset(runName, exportName);
  const scene = stored.config.scenes.find((entry) => entry.scene_id === sceneId);
  if (!scene) {
    throw new Error(`Unknown scene: ${sceneId}`);
  }

  const tiles = datasetTiles(stored.config, stored.tileOverrides).filter((tile) => tile.scene_id === sceneId);
  const imageUrls = options?.includeImages ? await ensureSceneTileImages(scene, tiles) : undefined;
  return {
    summary: buildSummary(stored.config, datasetTiles(stored.config, stored.tileOverrides)),
    scene: baseSceneFromConfig(scene, stored.config.tile_size_m),
    tiles: applySceneImagePaths(tiles, imageUrls),
  };
}

export async function updateBrowserTile(runName: string, exportName: string, tileId: string, update: TileUpdate) {
  updateStoredDataset(runName, exportName, (stored) => ({
    ...stored,
    tileOverrides: {
      ...stored.tileOverrides,
      [tileId]: {
        ...(stored.tileOverrides[tileId] ?? {}),
        ...update,
        status: update.selected ? "labeled" : "dropped",
      },
    },
  }));
  const sceneId = tileId.split(":")[0];
  return loadBrowserScene(sceneId, runName, exportName);
}

export async function updateBrowserTiles(runName: string, exportName: string, sceneId: string, updates: TileBatchUpdate[]) {
  updateStoredDataset(runName, exportName, (stored) => {
    const tileOverrides = { ...stored.tileOverrides };
    for (const update of updates) {
      tileOverrides[update.tile_id] = {
        ...(tileOverrides[update.tile_id] ?? {}),
        ...update,
        status: update.selected ? "labeled" : "dropped",
      };
    }
    return {
      ...stored,
      tileOverrides,
    };
  });
  return loadBrowserScene(sceneId, runName, exportName);
}
