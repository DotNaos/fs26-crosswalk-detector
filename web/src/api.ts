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
  RealDatasetConfig,
  ReviewState,
  ScenePayload,
  TileBatchUpdate,
  TileUpdate,
} from "./types";
import type { MapValidationSnapshot } from "./map-validation";
import type { AutopilotPlan } from "./autopilot-planner";

const VALIDATION_STATE_PREFIX = "crosswalk.validation.state";
const VALIDATION_ARTIFACT_PREFIX = "crosswalk.validation.artifact";

export async function listDatasets(): Promise<DatasetListEntry[]> {
  return listBrowserDatasets();
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

export async function loadDataset(_runName?: string, _exportName?: string): Promise<DatasetContract> {
  return loadBrowserDataset(_runName, _exportName);
}

export async function loadDatasetMeta(_runName?: string, _exportName?: string): Promise<DatasetSummary> {
  return loadBrowserDatasetMeta(_runName, _exportName);
}

export async function loadScene(_runName: string, _exportName: string, sceneId: string): Promise<ScenePayload> {
  return loadBrowserScene(sceneId, _runName, _exportName);
}

export async function ensureSceneImages(_runName: string, _exportName: string, sceneId: string) {
  return hydrateBrowserSceneImages(sceneId, _runName, _exportName);
}

export async function updateTile(_runName: string, _exportName: string, tileId: string, update: TileUpdate): Promise<ScenePayload> {
  return updateBrowserTile(_runName, _exportName, tileId, update);
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
  return loadBrowserConfig(runName, exportName);
}

export async function saveConfig(runName: string, exportName: string, config: RealDatasetConfig): Promise<RealDatasetConfig> {
  return saveBrowserConfig(runName, exportName, config);
}

export async function loadReviewState(_runName?: string, _exportName?: string): Promise<ReviewState> {
  return loadBrowserReviewState(_runName, _exportName);
}

export async function saveReviewState(_runName: string, _exportName: string, state: ReviewState): Promise<ReviewState> {
  return saveBrowserReviewState(_runName, _exportName, state);
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
