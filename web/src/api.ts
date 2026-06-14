import {
  createBrowserDataset,
  createBrowserAutopilotDataset,
  listBrowserDatasets,
  hydrateBrowserSceneImages,
  loadBrowserConfig,
  loadBrowserDataset,
  loadBrowserDatasetMeta,
  loadBrowserReviewState,
  loadBrowserScene,
  saveBrowserConfig,
  saveBrowserReviewState,
  updateBrowserTile,
  updateBrowserTiles,
} from "./browser-workspace";
import type {
  BrowserLabelSuggestion,
  DatasetContract,
  DatasetListEntry,
  DatasetSummary,
  DatasetTile,
  DatasetViewportPayload,
  RealDatasetConfig,
  ReviewState,
  RoadClusterGrid,
  ScenePayload,
  TileBatchUpdate,
  TileUpdate,
} from "./types";
import type { MapValidationSnapshot } from "./map-validation";
import type { AutopilotPlan } from "./autopilot-planner";
import { listStaticDatasetEntries, loadStaticDatasetSummary, loadStaticDatasetViewport } from "./static-dataset";

const VALIDATION_STATE_PREFIX = "crosswalk.validation.state";
const VALIDATION_ARTIFACT_PREFIX = "crosswalk.validation.artifact";
const STATIC_ONLY = import.meta.env.VITE_CROSSWALK_STATIC_ONLY === "1";

export async function listDatasets(): Promise<DatasetListEntry[]> {
  if (STATIC_ONLY) return listStaticDatasetEntries();
  try {
    return await parseJson<DatasetListEntry[]>(await fetch("/api/exports"));
  } catch {
    const staticDatasets = await listStaticDatasetEntries();
    if (staticDatasets.length) return staticDatasets;
    return listBrowserDatasets();
  }
}

export async function createDataset(name: string, sceneId: string): Promise<DatasetListEntry> {
  return createBrowserDataset({ name, sceneId });
}

export async function createAutopilotDataset(
  targetPositiveCount: number,
  name?: string,
  maxPanels?: number,
  perimeterBudget?: number,
): Promise<DatasetListEntry> {
  const plan = await loadAutopilotPlan(targetPositiveCount, maxPanels, perimeterBudget);
  return createBrowserAutopilotDataset({ targetPositiveCount, name, maxPanels, perimeterBudget, plan });
}

export async function loadAutopilotPlan(targetPositiveCount: number, maxPanels = 8, perimeterBudget = 72) {
  const params = new URLSearchParams({
    targetPositiveCount: String(targetPositiveCount),
    maxPanels: String(maxPanels),
    perimeterBudget: String(perimeterBudget),
  });
  const response = await fetch(`/api/autopilot/plan?${params}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const plan = (await response.json()) as Partial<AutopilotPlan>;
  if (plan.mode !== "swiss-lowres-urban-grid" || !Array.isArray(plan.scenes)) {
    throw new Error("Invalid autopilot plan response.");
  }
  return plan as AutopilotPlan;
}

export async function loadRoadClusterGrid(
  bboxMercator: [number, number, number, number],
  zoom: number,
  options: { signal?: AbortSignal } = {},
): Promise<RoadClusterGrid> {
  const params = new URLSearchParams({
    bbox: bboxMercator.join(","),
    zoom: String(zoom),
  });
  const response = await fetch(`/api/autopilot/road-clusters?${params}`, { signal: options.signal });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as RoadClusterGrid;
}

export async function loadDataset(_runName?: string, _exportName?: string): Promise<DatasetContract> {
  try {
    return await parseJson<DatasetContract>(await fetch(datasetUrl("/api/dataset", _runName, _exportName)));
  } catch {
    return loadBrowserDataset(_runName, _exportName);
  }
}

export async function loadDatasetMeta(_runName?: string, _exportName?: string): Promise<DatasetSummary> {
  if (STATIC_ONLY && _runName && _exportName) return loadStaticDatasetSummary(_runName, _exportName);
  try {
    return await parseJson<DatasetSummary>(await fetch(datasetUrl("/api/dataset-meta", _runName, _exportName)));
  } catch {
    if (_runName && _exportName) {
      try {
        return await loadStaticDatasetSummary(_runName, _exportName);
      } catch {
        // Fall through to browser workspace fixtures.
      }
    }
    return loadBrowserDatasetMeta(_runName, _exportName);
  }
}

export async function loadScene(_runName: string, _exportName: string, sceneId: string): Promise<ScenePayload> {
  try {
    const url = datasetUrl("/api/scene", _runName, _exportName);
    url.searchParams.set("scene", sceneId);
    return await parseJson<ScenePayload>(await fetch(url));
  } catch {
    return loadBrowserScene(sceneId, _runName, _exportName);
  }
}

export async function ensureSceneImages(_runName: string, _exportName: string, sceneId: string) {
  return hydrateBrowserSceneImages(sceneId, _runName, _exportName);
}

export async function updateTile(_runName: string, _exportName: string, tileId: string, update: TileUpdate): Promise<ScenePayload> {
  try {
    return await parseJson<ScenePayload>(
      await fetch(datasetUrl(`/api/tiles/${encodeURIComponent(tileId)}`, _runName, _exportName), {
        body: JSON.stringify(update),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      }),
    );
  } catch {
    return updateBrowserTile(_runName, _exportName, tileId, update);
  }
}

export async function updateTiles(
  _runName: string,
  _exportName: string,
  sceneId: string,
  updates: TileBatchUpdate[],
): Promise<ScenePayload> {
  return updateBrowserTiles(_runName, _exportName, sceneId, updates);
}

export async function loadConfig(runName?: string, exportName?: string): Promise<RealDatasetConfig> {
  try {
    return await parseJson<RealDatasetConfig>(await fetch(datasetUrl("/api/config", runName, exportName)));
  } catch {
    return loadBrowserConfig(runName, exportName);
  }
}

export async function saveConfig(runName: string, exportName: string, config: RealDatasetConfig): Promise<RealDatasetConfig> {
  return saveBrowserConfig(runName, exportName, config);
}

export async function loadReviewState(_runName?: string, _exportName?: string): Promise<ReviewState> {
  try {
    return await parseJson<ReviewState>(await fetch(datasetUrl("/api/review-state", _runName, _exportName)));
  } catch {
    return loadBrowserReviewState(_runName, _exportName);
  }
}

export async function saveReviewState(_runName: string, _exportName: string, state: ReviewState): Promise<ReviewState> {
  try {
    return await parseJson<ReviewState>(
      await fetch(datasetUrl("/api/review-state", _runName, _exportName), {
        body: JSON.stringify(state),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      }),
    );
  } catch {
    return saveBrowserReviewState(_runName, _exportName, state);
  }
}

export async function loadDatasetViewport(
  runName: string,
  exportName: string,
  input: {
    bboxMercator: [number, number, number, number];
    city?: string;
    label?: string;
    limit?: number;
    split?: string;
    zoom: number;
  },
  options: { signal?: AbortSignal } = {},
): Promise<DatasetViewportPayload> {
  if (STATIC_ONLY) return loadStaticDatasetViewport(runName, exportName, input);
  const url = datasetUrl("/api/dataset/viewport", runName, exportName);
  url.searchParams.set("bbox", input.bboxMercator.join(","));
  url.searchParams.set("zoom", String(input.zoom));
  url.searchParams.set("limit", String(input.limit ?? 1400));
  if (input.city && input.city !== "all") url.searchParams.set("city", input.city);
  if (input.label && input.label !== "all") url.searchParams.set("label", input.label);
  if (input.split && input.split !== "all") url.searchParams.set("split", input.split);
  try {
    return await parseJson<DatasetViewportPayload>(await fetch(url, { signal: options.signal }));
  } catch (reason) {
    if (options.signal?.aborted) throw reason;
    return loadStaticDatasetViewport(runName, exportName, input);
  }
}

export async function runCrossMaskOnTiles(input: {
  exportName: string;
  maxTiles?: number;
  runName: string;
  threshold?: number;
  tiles: DatasetTile[];
}) {
  return parseJson<{
    predictions: Array<{ confidence: number; decision: "crosswalk" | "no_crosswalk"; mask_coverage: number; mask_score: number; tile_id: string }>;
    run_id: string;
    summary: { crosswalk: number; no_crosswalk: number; total: number };
    updated_tiles: DatasetTile[];
  }>(
    await fetch("/api/crossmask/run", {
      body: JSON.stringify({
        export_name: input.exportName,
        max_tiles: input.maxTiles,
        run_name: input.runName,
        threshold: input.threshold,
        tiles: input.tiles,
      }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    }),
  );
}

export async function pushMapValidationState(snapshot: MapValidationSnapshot): Promise<void> {
  window.localStorage.setItem(
    `${VALIDATION_STATE_PREFIX}:${snapshot.validationRunId}:${snapshot.validationCase ?? "default"}`,
    JSON.stringify(snapshot),
  );
}

export async function fetchMapValidationState(validationRunId: string, validationCase: string) {
  const raw = window.localStorage.getItem(`${VALIDATION_STATE_PREFIX}:${validationRunId}:${validationCase}`);
  return raw ? (JSON.parse(raw) as MapValidationSnapshot) : null;
}

export async function saveMapValidationArtifact(validationRunId: string, validationCase: string, name: string, artifact: unknown) {
  const key = `${VALIDATION_ARTIFACT_PREFIX}:${validationRunId}:${validationCase}:${name}`;
  window.localStorage.setItem(key, JSON.stringify(artifact));
  return { ok: true };
}

export function suggestionsToBatchUpdates(suggestions: Record<string, BrowserLabelSuggestion>): TileBatchUpdate[] {
  return Object.values(suggestions).map((suggestion) => ({
    tile_id: suggestion.tile_id,
    label: suggestion.label,
    selected: suggestion.selected,
    combined_probability: suggestion.score,
    predicted_label: suggestion.label,
    review_source: suggestion.review_source,
  }));
}

function datasetUrl(path: string, runName?: string, exportName?: string) {
  const url = new URL(path, window.location.href);
  if (runName) url.searchParams.set("run", runName);
  if (exportName) url.searchParams.set("export", exportName);
  return url;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}
