import type { ReviewState, SceneReviewState } from "./types";

export const DEFAULT_SCAN_RADIUS = 4;
export const DEFAULT_SCAN_DELAY_MS = 24;

export function defaultSceneReviewState(): SceneReviewState {
  return {
    scan_radius: DEFAULT_SCAN_RADIUS,
    scan_delay_ms: DEFAULT_SCAN_DELAY_MS,
    scanned_tile_ids: [],
  };
}

export function normalizeSceneReviewState(value?: Partial<SceneReviewState> | null): SceneReviewState {
  return {
    scan_radius: Number(value?.scan_radius ?? DEFAULT_SCAN_RADIUS) || DEFAULT_SCAN_RADIUS,
    scan_delay_ms: Number(value?.scan_delay_ms ?? DEFAULT_SCAN_DELAY_MS) || DEFAULT_SCAN_DELAY_MS,
    scanned_tile_ids: Array.isArray(value?.scanned_tile_ids) ? value.scanned_tile_ids.filter(Boolean) : [],
  };
}

export function normalizeReviewState(value?: Partial<ReviewState> | null): ReviewState {
  const scenes = Object.fromEntries(
    Object.entries(value?.scenes ?? {}).map(([sceneId, sceneState]) => [sceneId, normalizeSceneReviewState(sceneState)]),
  );

  return {
    selected_scene_id: value?.selected_scene_id,
    selected_tile_id: value?.selected_tile_id,
    map_zoom: typeof value?.map_zoom === "number" ? value.map_zoom : undefined,
    scenes,
  };
}

export function sceneReviewStateFor(reviewState: ReviewState | undefined, sceneId: string | undefined): SceneReviewState {
  if (!sceneId) return defaultSceneReviewState();
  return normalizeSceneReviewState(reviewState?.scenes?.[sceneId]);
}
