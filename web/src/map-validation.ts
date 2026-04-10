import { normalizeReviewState } from "./review-state";
import type { ReviewState, SceneReviewState } from "./types";

export type ValidationInteractionPhase = "idle" | "wheel-zoom" | "pinch-zoom" | "pan" | "fit" | "scan";

export type ValidationPoint = { x: number; y: number };

export type ValidationBounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

export type ValidationCameraState = {
  center: [number, number];
  zoom: number;
  bounds: [[number, number], [number, number]] | null;
  viewport: { width: number; height: number };
  pageScale: number;
};

export type ValidationRenderedTile = {
  tileId: string;
  selected: boolean;
  active: boolean;
  projectedBounds: ValidationBounds;
};

export type ValidationInvariant = {
  id: string;
  pass: boolean;
  detail: string;
};

export type ValidationTraceEvent = {
  type: string;
  at: number;
  phase: ValidationInteractionPhase;
  zoom: number;
  center: [number, number];
  selectedTileId?: string;
  activeTileId?: string;
  anchor?: ValidationPoint | null;
  metrics?: Record<string, number | string | boolean | null>;
};

export type ValidationVerdict = {
  name: string;
  pass: boolean;
  detail: string;
  metrics?: Record<string, number>;
};

export type ValidationCaseStatus = "idle" | "running" | "completed" | "failed";

export type ValidationCaseName =
  | "idle-stability"
  | "zoom-sequence"
  | "circle-coverage"
  | "pan-sequence"
  | "fit-sequence"
  | "scan-sequence"
  | "persistence-sequence";

export type MapValidationSnapshot = {
  validationRunId: string;
  validationCase: ValidationCaseName | null;
  status: ValidationCaseStatus;
  updatedAt: number;
  sceneId?: string;
  selectedTileId?: string;
  activeTileId?: string;
  scanIndex: number;
  orderedScanTileIds: string[];
  scannedTileIds: string[];
  interactionPhase: ValidationInteractionPhase;
  lastInputSource: string;
  camera: ValidationCameraState;
  selectedTileBounds: ValidationBounds | null;
  activeTileBounds: ValidationBounds | null;
  scanCircleBounds: ValidationBounds | null;
  zoomAnchor: ValidationPoint | null;
  renderedTiles: ValidationRenderedTile[];
  counters: {
    componentRenders: number;
    overlayRecomputes: number;
    cameraUpdates: number;
    pageZoomBlocks: number;
    consoleErrors: number;
  };
  invariants: ValidationInvariant[];
  verdicts: ValidationVerdict[];
  traces: ValidationTraceEvent[];
  persistedReviewChecksum: string;
  persistedReviewState: ReviewState;
};

export function checksum(value: unknown) {
  const json = JSON.stringify(value);
  let hash = 0;
  for (let index = 0; index < json.length; index += 1) {
    hash = (hash * 31 + json.charCodeAt(index)) | 0;
  }
  return `${json.length}:${Math.abs(hash)}`;
}

export function sameSceneReviewState(a: SceneReviewState, b: SceneReviewState) {
  return (
    a.scan_radius === b.scan_radius &&
    a.scan_delay_ms === b.scan_delay_ms &&
    a.scanned_tile_ids.length === b.scanned_tile_ids.length &&
    a.scanned_tile_ids.every((tileId, index) => tileId === b.scanned_tile_ids[index])
  );
}

export function sameReviewState(a: ReviewState, b: ReviewState) {
  const sceneIdsA = Object.keys(a.scenes).sort();
  const sceneIdsB = Object.keys(b.scenes).sort();
  return (
    a.selected_scene_id === b.selected_scene_id &&
    a.selected_tile_id === b.selected_tile_id &&
    a.map_zoom === b.map_zoom &&
    sceneIdsA.length === sceneIdsB.length &&
    sceneIdsA.every((sceneId, index) => sceneId === sceneIdsB[index] && sameSceneReviewState(a.scenes[sceneId], b.scenes[sceneId]))
  );
}

export function boundsErrorPx(a: ValidationBounds | null, b: ValidationBounds | null) {
  if (!a || !b) return Number.POSITIVE_INFINITY;
  return Math.max(
    Math.abs(a.minX - b.minX),
    Math.abs(a.minY - b.minY),
    Math.abs(a.maxX - b.maxX),
    Math.abs(a.maxY - b.maxY),
  );
}

export function pointDistancePx(a: ValidationPoint | null, b: ValidationPoint | null) {
  if (!a || !b) return Number.POSITIVE_INFINITY;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function limitTrace(events: ValidationTraceEvent[], max = 240) {
  return events.slice(-max);
}

export function validationClassSuffix(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

export function buildGlobalInvariants(snapshot: {
  sceneId?: string;
  selectedTileId?: string;
  activeTileId?: string;
  orderedScanTileIds: string[];
  scannedTileIds: string[];
  renderedTiles: ValidationRenderedTile[];
  camera: ValidationCameraState;
  persistedReviewState: ReviewState;
  counters: MapValidationSnapshot["counters"];
}) {
  const uniqueRendered = new Set(snapshot.renderedTiles.map((tile) => tile.tileId));
  const uniqueOrdered = new Set(snapshot.orderedScanTileIds);
  const scannedPrefix = snapshot.orderedScanTileIds.slice(0, snapshot.scannedTileIds.length);
  const normalizedPersisted = normalizeReviewState(snapshot.persistedReviewState);

  const invariants: ValidationInvariant[] = [
    {
      id: "I1",
      pass: !snapshot.selectedTileId || Boolean(snapshot.sceneId),
      detail: "Selected tile requires a selected scene.",
    },
    {
      id: "I2",
      pass:
        Number.isFinite(snapshot.camera.zoom) &&
        snapshot.camera.center.every((value) => Number.isFinite(value)) &&
        snapshot.camera.viewport.width > 0 &&
        snapshot.camera.viewport.height > 0,
      detail: "Camera state must be finite and viewport positive.",
    },
    {
      id: "I3",
      pass: uniqueRendered.size === snapshot.renderedTiles.length,
      detail: "Rendered tile rectangles must be unique.",
    },
    {
      id: "I4",
      pass: snapshot.renderedTiles.every(
        (tile) =>
          Number.isFinite(tile.projectedBounds.minX) &&
          Number.isFinite(tile.projectedBounds.minY) &&
          Number.isFinite(tile.projectedBounds.maxX) &&
          Number.isFinite(tile.projectedBounds.maxY),
      ),
      detail: "Rendered tiles must have finite projected bounds.",
    },
    {
      id: "I5",
      pass: !snapshot.selectedTileId || uniqueRendered.has(snapshot.selectedTileId) || uniqueOrdered.has(snapshot.selectedTileId),
      detail: "Selected tile must belong to the selected scene overlay or scan ordering.",
    },
    {
      id: "I6",
      pass: scannedPrefix.every((tileId, index) => tileId === snapshot.scannedTileIds[index]),
      detail: "Scan progress must be a prefix of the ordered scan tiles.",
    },
    {
      id: "I7",
      pass: checksum(snapshot.persistedReviewState) === checksum(normalizedPersisted),
      detail: "Persisted review state must already be normalized.",
    },
    {
      id: "I8",
      pass: snapshot.counters.consoleErrors === 0,
      detail: "Canvas must not emit console errors during validation runs.",
    },
    {
      id: "I9",
      pass: Math.abs(snapshot.camera.pageScale - 1) < 0.0001,
      detail: "Browser page zoom must remain unchanged inside the canvas.",
    },
  ];

  return invariants;
}
