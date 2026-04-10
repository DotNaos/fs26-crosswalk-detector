import {
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

const VALIDATION_STATE_PREFIX = "crosswalk.validation.state";
const VALIDATION_ARTIFACT_PREFIX = "crosswalk.validation.artifact";

export async function listDatasets(): Promise<DatasetListEntry[]> {
  return listBrowserDatasets();
}

export async function loadDataset(_runName?: string, _exportName?: string): Promise<DatasetContract> {
  return loadBrowserDataset();
}

export async function loadDatasetMeta(_runName?: string, _exportName?: string): Promise<DatasetSummary> {
  return loadBrowserDatasetMeta();
}

export async function loadScene(_runName: string, _exportName: string, sceneId: string): Promise<ScenePayload> {
  return loadBrowserScene(sceneId);
}

export async function ensureSceneImages(_runName: string, _exportName: string, sceneId: string) {
  return hydrateBrowserSceneImages(sceneId);
}

export async function updateTile(_runName: string, _exportName: string, tileId: string, update: TileUpdate): Promise<ScenePayload> {
  return updateBrowserTile(tileId, update);
}

export async function updateTiles(
  _runName: string,
  _exportName: string,
  sceneId: string,
  updates: TileBatchUpdate[],
): Promise<ScenePayload> {
  return updateBrowserTiles(sceneId, updates);
}

export async function loadConfig(): Promise<RealDatasetConfig> {
  return loadBrowserConfig();
}

export async function saveConfig(config: RealDatasetConfig): Promise<RealDatasetConfig> {
  return saveBrowserConfig(config);
}

export async function loadReviewState(_runName?: string, _exportName?: string): Promise<ReviewState> {
  return loadBrowserReviewState();
}

export async function saveReviewState(_runName: string, _exportName: string, state: ReviewState): Promise<ReviewState> {
  return saveBrowserReviewState(state);
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
