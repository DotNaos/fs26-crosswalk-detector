import { useEffect, useMemo, useRef, useState } from "react";
import { point, type LatLngExpression, type Map as LeafletMap } from "leaflet";
import { pushMapValidationState, saveMapValidationArtifact } from "./api";
import { buildGlobalInvariants, boundsErrorPx, checksum, limitTrace, pointDistancePx, validationClassSuffix, type MapValidationSnapshot, type ValidationBounds, type ValidationCaseName, type ValidationCaseStatus, type ValidationInteractionPhase, type ValidationPoint, type ValidationRenderedTile, type ValidationTraceEvent, type ValidationVerdict } from "./map-validation";
import type { DatasetScene, DatasetTile, ReviewState } from "./types";
import { bboxToLatLngBounds, bboxToMercatorRect, circleIntersectsMercatorRect, sceneLatLngBounds } from "./utils";

type ValidationCounters = MapValidationSnapshot["counters"];

type UseMapValidationRuntimeArgs = {
  enabled: boolean;
  validationRunId: string;
  validationCase: string | null;
  map: LeafletMap | null;
  mapShell: HTMLDivElement | null;
  scenes: DatasetScene[];
  sceneTiles: DatasetTile[];
  sceneId?: string;
  selectedTileId?: string;
  activeTileId?: string;
  selectedTile?: DatasetTile;
  activeTile?: DatasetTile;
  footprintTiles: DatasetTile[];
  orderedScanTiles: DatasetTile[];
  footprintCenterMercator: { x: number; y: number } | null;
  footprintRadiusM: number;
  scannedTileIds: string[];
  interactionPhase: ValidationInteractionPhase;
  lastInputSource: string;
  zoomAnchor: ValidationPoint | null;
  persistedReviewChecksum: string;
  persistedReviewState: ReviewState;
  counters: ValidationCounters;
  onSelectScene: (sceneId: string) => void;
  onStartScan: () => void;
  onPauseScan: () => void;
  onResetScan: () => void;
};

type ScenarioMeasurements = {
  startedAt: number;
  zoomAnchor?: ValidationPoint | null;
  zoomWorld?: LatLngExpression;
  zoomTileId?: string;
  zoomAnchorDrifts: number[];
  zoomOverlayErrors: number[];
  panCenters: Array<[number, number]>;
  fitEvents: number;
  scanSequence: string[];
  persistenceExpected?: {
    sceneId?: string;
    selectedTileId?: string;
    checksum: string;
    scannedTileIds: string[];
  };
};

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function nextFrame() {
  return new Promise<number>((resolve) => window.requestAnimationFrame(resolve));
}

function domBounds(container: HTMLElement, selector: string) {
  const element = container.querySelector<SVGElement>(selector);
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  const containerRect = container.getBoundingClientRect();
  return {
    minX: rect.left - containerRect.left,
    minY: rect.top - containerRect.top,
    maxX: rect.right - containerRect.left,
    maxY: rect.bottom - containerRect.top,
  } satisfies ValidationBounds;
}

function projectedBounds(map: LeafletMap, tile: DatasetTile | undefined) {
  if (!tile) return null;
  const bounds = bboxToLatLngBounds(tile.bbox_mercator);
  if (!bounds) return null;
  const northWest = map.latLngToContainerPoint(bounds[0]);
  const southEast = map.latLngToContainerPoint(bounds[1]);
  return {
    minX: Math.min(northWest.x, southEast.x),
    minY: Math.min(northWest.y, southEast.y),
    maxX: Math.max(northWest.x, southEast.x),
    maxY: Math.max(northWest.y, southEast.y),
  } satisfies ValidationBounds;
}

function tileSelector(tileId: string) {
  return `.validation-tile-${validationClassSuffix(tileId)}`;
}

function readRenderedTiles(container: HTMLElement, tiles: DatasetTile[]) {
  return tiles
    .map((tile) => {
      const bounds = domBounds(container, tileSelector(tile.tile_id));
      if (!bounds) return null;
      return {
        tileId: tile.tile_id,
        selected: false,
        active: false,
        projectedBounds: bounds,
      } satisfies ValidationRenderedTile;
    })
    .filter(Boolean) as ValidationRenderedTile[];
}

function caseName(value: string | null): ValidationCaseName | null {
  return value === "idle-stability" || value === "zoom-sequence" || value === "circle-coverage" || value === "pan-sequence" || value === "fit-sequence" || value === "scan-sequence" || value === "persistence-sequence" ? value : null;
}

export function useMapValidationRuntime({
  enabled,
  validationRunId,
  validationCase,
  map,
  mapShell,
  scenes,
  sceneTiles,
  sceneId,
  selectedTileId,
  activeTileId,
  selectedTile,
  activeTile,
  footprintTiles,
  orderedScanTiles,
  footprintCenterMercator,
  footprintRadiusM,
  scannedTileIds,
  interactionPhase,
  lastInputSource,
  zoomAnchor,
  persistedReviewChecksum,
  persistedReviewState,
  counters,
  onSelectScene,
  onStartScan,
  onPauseScan,
  onResetScan,
}: UseMapValidationRuntimeArgs) {
  const [status, setStatus] = useState<ValidationCaseStatus>("idle");
  const [verdicts, setVerdicts] = useState<ValidationVerdict[]>([]);
  const [snapshot, setSnapshot] = useState<MapValidationSnapshot | null>(null);
  const statusRef = useRef<ValidationCaseStatus>("idle");
  const verdictsRef = useRef<ValidationVerdict[]>([]);
  const tracesRef = useRef<ValidationTraceEvent[]>([]);
  const consoleErrorsRef = useRef(0);
  const measurementsRef = useRef<ScenarioMeasurements>({ startedAt: 0, zoomAnchorDrifts: [], zoomOverlayErrors: [], panCenters: [], fitEvents: 0, scanSequence: [] });
  const lastSignatureRef = useRef("");
  const lastSnapshotSignatureRef = useRef("");
  const activeRunRef = useRef<string | null>(null);
  const snapshotRef = useRef<MapValidationSnapshot | null>(null);
  const persistedPhaseRef = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const originalError = console.error;
    console.error = (...args: unknown[]) => {
      consoleErrorsRef.current += 1;
      originalError(...args);
    };
    return () => {
      console.error = originalError;
    };
  }, [enabled]);

  const validationCaseName = useMemo(() => caseName(validationCase), [validationCase]);

  useEffect(() => {
    if (!enabled || !map || !mapShell) return;

    const camera = {
      center: [map.getCenter().lat, map.getCenter().lng] as [number, number],
      zoom: map.getZoom(),
      bounds: map.getBounds()
        ? ([
            [map.getBounds().getSouthWest().lat, map.getBounds().getSouthWest().lng],
            [map.getBounds().getNorthEast().lat, map.getBounds().getNorthEast().lng],
          ] as [[number, number], [number, number]])
        : null,
      viewport: {
        width: map.getContainer().clientWidth,
        height: map.getContainer().clientHeight,
      },
      pageScale: window.visualViewport?.scale ?? 1,
    };

    const renderedTiles = readRenderedTiles(map.getContainer(), footprintTiles).map((tile) => ({
      ...tile,
      selected: tile.tileId === selectedTileId,
      active: tile.tileId === activeTileId,
    }));

    const nextSnapshot: MapValidationSnapshot = {
      validationRunId,
      validationCase: validationCaseName,
      status: statusRef.current,
      updatedAt: Date.now(),
      sceneId,
      selectedTileId,
      activeTileId,
      scanIndex: scannedTileIds.length,
      orderedScanTileIds: orderedScanTiles.map((tile) => tile.tile_id),
      scannedTileIds,
      interactionPhase,
      lastInputSource,
      camera,
      selectedTileBounds: selectedTileId ? domBounds(map.getContainer(), tileSelector(selectedTileId)) : null,
      activeTileBounds: activeTileId ? domBounds(map.getContainer(), tileSelector(activeTileId)) : null,
      scanCircleBounds: domBounds(map.getContainer(), ".validation-scan-circle"),
      zoomAnchor,
      renderedTiles,
      counters: {
        ...counters,
        consoleErrors: consoleErrorsRef.current,
      },
      invariants: [],
      verdicts: verdictsRef.current,
      traces: tracesRef.current,
      persistedReviewChecksum,
      persistedReviewState,
    };

    nextSnapshot.invariants = buildGlobalInvariants({
      sceneId,
      selectedTileId,
      activeTileId,
      orderedScanTileIds: nextSnapshot.orderedScanTileIds,
      scannedTileIds,
      renderedTiles,
      camera,
      persistedReviewState,
      counters: nextSnapshot.counters,
    });

    const signature = checksum({
      sceneId,
      selectedTileId,
      activeTileId,
      scanIndex: scannedTileIds.length,
      interactionPhase,
      zoom: camera.zoom,
      center: camera.center,
      verdicts: verdictsRef.current,
      consoleErrors: consoleErrorsRef.current,
    });

    const snapshotSignature = checksum({
      sceneId,
      selectedTileId,
      activeTileId,
      scanIndex: scannedTileIds.length,
      interactionPhase,
      lastInputSource,
      zoomAnchor,
      camera,
      renderedTiles: renderedTiles.map((tile) => ({
        tileId: tile.tileId,
        selected: tile.selected,
        active: tile.active,
        projectedBounds: tile.projectedBounds,
      })),
      counters: {
        cameraUpdates: nextSnapshot.counters.cameraUpdates,
        pageZoomBlocks: nextSnapshot.counters.pageZoomBlocks,
        consoleErrors: nextSnapshot.counters.consoleErrors,
      },
      persistedReviewChecksum,
      status: statusRef.current,
      verdicts: verdictsRef.current,
    });

    snapshotRef.current = nextSnapshot;
    if (snapshotSignature !== lastSnapshotSignatureRef.current) {
      lastSnapshotSignatureRef.current = snapshotSignature;
      setSnapshot(nextSnapshot);
    }

    if (signature !== lastSignatureRef.current) {
      tracesRef.current = limitTrace([
        ...tracesRef.current,
        {
          type: "snapshot",
          at: nextSnapshot.updatedAt,
          phase: interactionPhase,
          zoom: camera.zoom,
          center: camera.center,
          selectedTileId,
          activeTileId,
          anchor: zoomAnchor,
          metrics: {
            scanIndex: scannedTileIds.length,
            renderedTiles: renderedTiles.length,
          },
        },
      ]);
      lastSignatureRef.current = signature;
    }
  }, [
    activeTileId,
    counters.cameraUpdates,
    counters.overlayRecomputes,
    enabled,
    footprintTiles,
    interactionPhase,
    lastInputSource,
    map,
    mapShell,
    orderedScanTiles,
    persistedReviewChecksum,
    persistedReviewState,
    validationCaseName,
    validationRunId,
    scannedTileIds,
    sceneId,
    selectedTileId,
    zoomAnchor,
    counters.pageZoomBlocks,
  ]);

  useEffect(() => {
    if (!enabled || !snapshot) return;
    void pushMapValidationState(snapshot);
  }, [enabled, snapshot]);

  function complete(nextVerdicts: ValidationVerdict[], artifact?: Record<string, unknown>) {
    const nextStatus = nextVerdicts.every((verdict) => verdict.pass) ? "completed" : "failed";
    statusRef.current = nextStatus;
    verdictsRef.current = nextVerdicts;
    setVerdicts(nextVerdicts);
    setStatus(nextStatus);
    if (snapshotRef.current) {
      const completedSnapshot = {
        ...snapshotRef.current,
        status: nextStatus,
        verdicts: nextVerdicts,
      } satisfies MapValidationSnapshot;
      snapshotRef.current = completedSnapshot;
      setSnapshot(completedSnapshot);
      void pushMapValidationState(completedSnapshot);
    }
    if (validationCaseName) {
      void saveMapValidationArtifact(validationRunId, validationCaseName, "verdicts", nextVerdicts);
      if (artifact) {
        void saveMapValidationArtifact(validationRunId, validationCaseName, "artifact", artifact);
      }
    }
  }

  useEffect(() => {
    if (!enabled || !map || !validationCaseName || !snapshotRef.current || !sceneId) return;
    const needsGridData = validationCaseName === "zoom-sequence" || validationCaseName === "circle-coverage" || validationCaseName === "scan-sequence";
    if (needsGridData && (!footprintTiles.length || !orderedScanTiles.length)) {
      return;
    }
    const runKey = `${validationRunId}:${validationCaseName}`;
    if (activeRunRef.current === runKey || status === "running") return;
    activeRunRef.current = runKey;
    statusRef.current = "running";
    verdictsRef.current = [];
    setStatus("running");
    setVerdicts([]);
    tracesRef.current = [];
    measurementsRef.current = {
      startedAt: Date.now(),
      zoomAnchorDrifts: [],
      zoomOverlayErrors: [],
      panCenters: [],
      fitEvents: 0,
      scanSequence: [],
    };

    const waitForGridData = async (minimumTiles = 1) => {
      for (let attempt = 0; attempt < 30; attempt += 1) {
        const currentCount = snapshotRef.current?.orderedScanTileIds.length ?? 0;
        if (currentCount >= minimumTiles && footprintTiles.length >= minimumTiles) return true;
        await wait(150);
      }
      return false;
    };

    const run = async () => {
      if (validationCaseName === "idle-stability") {
        const startCounters = { ...snapshotRef.current!.counters };
        await wait(10_000);
        const current = snapshotRef.current;
        if (!current) return;
        complete(
          [
            {
              name: "idle-recursion",
              pass: current.counters.consoleErrors === startCounters.consoleErrors,
              detail: current.counters.consoleErrors === startCounters.consoleErrors ? "No console recursion during idle." : "Console errors increased during idle.",
              metrics: { consoleErrors: current.counters.consoleErrors },
            },
            {
              name: "idle-camera-stability",
              pass: current.counters.cameraUpdates - startCounters.cameraUpdates <= 6,
              detail:
                current.counters.cameraUpdates - startCounters.cameraUpdates <= 6
                  ? "Camera settled while idle."
                  : "Camera kept updating without input.",
              metrics: { cameraUpdates: current.counters.cameraUpdates - startCounters.cameraUpdates },
            },
          ],
          { snapshot: current },
        );
        return;
      }

      if (validationCaseName === "zoom-sequence") {
        const ready = await waitForGridData();
        if (!ready) {
          complete([{ name: "zoom-availability", pass: false, detail: "Grid data was not ready before the zoom validation started." }]);
          return;
        }
        const targetTile = orderedScanTiles[0] ?? footprintTiles[0] ?? selectedTile;
        const anchor = zoomAnchor ?? { x: map.getContainer().clientWidth * 0.52, y: map.getContainer().clientHeight * 0.42 };
        const worldPoint = map.containerPointToLatLng([anchor.x, anchor.y]);
        measurementsRef.current.zoomAnchor = anchor;
        measurementsRef.current.zoomWorld = worldPoint;
        measurementsRef.current.zoomTileId = targetTile?.tile_id;
        for (const delta of [0.5, 0.5, -0.5, -0.5]) {
          map.setZoomAround(point(anchor.x, anchor.y), map.getZoom() + delta, { animate: false });
          await nextFrame();
          await wait(90);
          const currentAnchor = map.latLngToContainerPoint(worldPoint);
          measurementsRef.current.zoomAnchorDrifts.push(pointDistancePx(anchor, { x: currentAnchor.x, y: currentAnchor.y }));
          const actual = targetTile ? domBounds(map.getContainer(), tileSelector(targetTile.tile_id)) : null;
          const expected = projectedBounds(map, targetTile);
          const error = boundsErrorPx(actual, expected);
          if (Number.isFinite(error)) {
            measurementsRef.current.zoomOverlayErrors.push(error);
          }
        }
        await wait(180);
        const anchorDrifts = measurementsRef.current.zoomAnchorDrifts;
        const overlayErrors = measurementsRef.current.zoomOverlayErrors;
        complete(
          [
            {
              name: "zoom-overlay",
              pass: overlayErrors.length > 0 && overlayErrors.every((value) => value <= 2) && (overlayErrors.at(-1) ?? Number.POSITIVE_INFINITY) <= 1,
              detail:
                overlayErrors.length > 0 && overlayErrors.every((value) => value <= 2) && (overlayErrors.at(-1) ?? Number.POSITIVE_INFINITY) <= 1
                  ? "Overlay stayed aligned during zoom."
                  : overlayErrors.length === 0
                    ? "No measurable overlay bounds were available for the target tile."
                    : "Overlay drift exceeded validation threshold during zoom.",
              metrics: { maxOverlayErrorPx: Math.max(...overlayErrors, 0), finalOverlayErrorPx: overlayErrors.at(-1) ?? 0 },
            },
            {
              name: "zoom-anchor",
              pass: anchorDrifts.every((value) => value <= 3) && (anchorDrifts.at(-1) ?? Number.POSITIVE_INFINITY) <= 1,
              detail:
                anchorDrifts.every((value) => value <= 3) && (anchorDrifts.at(-1) ?? Number.POSITIVE_INFINITY) <= 1
                  ? "Zoom anchor stayed stable."
                  : "Zoom anchor drifted beyond threshold.",
              metrics: { maxAnchorDriftPx: Math.max(...anchorDrifts, 0), finalAnchorDriftPx: anchorDrifts.at(-1) ?? 0 },
            },
          ],
          { anchorDrifts, overlayErrors },
        );
        return;
      }

      if (validationCaseName === "circle-coverage") {
        const ready = await waitForGridData();
        if (!ready) {
          complete([{ name: "circle-coverage", pass: false, detail: "Grid data was not ready before the coverage validation started." }]);
          return;
        }
        if (!footprintCenterMercator) {
          complete([{ name: "circle-coverage", pass: false, detail: "No footprint center available for the circle coverage validation." }]);
          return;
        }
        const expectedTileIds = sceneTiles
          .filter((tile) => {
            const rect = bboxToMercatorRect(tile.bbox_mercator);
            return rect ? circleIntersectsMercatorRect(rect, footprintCenterMercator.x, footprintCenterMercator.y, footprintRadiusM) : false;
          })
          .map((tile) => tile.tile_id)
          .sort();
        const actualTileIds = footprintTiles.map((tile) => tile.tile_id).sort();
        const expectedSet = new Set(expectedTileIds);
        const actualSet = new Set(actualTileIds);
        const missing = expectedTileIds.filter((tileId) => !actualSet.has(tileId));
        const extra = actualTileIds.filter((tileId) => !expectedSet.has(tileId));
        complete(
          [
            {
              name: "circle-coverage",
              pass: missing.length === 0 && extra.length === 0,
              detail:
                missing.length === 0 && extra.length === 0
                  ? "The active scan tiles exactly match the set of tiles whose rectangles intersect the scan circle."
                  : "The active scan tiles do not match the rectangle-circle intersection set.",
              metrics: {
                missingCount: missing.length,
                extraCount: extra.length,
                expectedCount: expectedTileIds.length,
                actualCount: actualTileIds.length,
              },
            },
          ],
          { missing, extra, expectedTileIds, actualTileIds },
        );
        return;
      }

      if (validationCaseName === "pan-sequence") {
        for (const delta of [
          [48, 28],
          [42, -18],
          [-36, 22],
        ]) {
          map.panBy(delta as [number, number], { animate: false });
          await nextFrame();
          measurementsRef.current.panCenters.push([map.getCenter().lat, map.getCenter().lng]);
          await wait(70);
        }
        const expectedCenter = measurementsRef.current.panCenters.at(-1);
        await wait(180);
        const finalCenter = [map.getCenter().lat, map.getCenter().lng] as [number, number];
        const drift = expectedCenter ? Math.hypot(finalCenter[0] - expectedCenter[0], finalCenter[1] - expectedCenter[1]) : Number.POSITIVE_INFINITY;
        complete(
          [
            {
              name: "pan-no-snapback",
              pass: drift <= 0.000001,
              detail: drift <= 0.000001 ? "Pan ended exactly at the last derived center." : "Pan snapped after release.",
              metrics: { settleDrift: drift },
            },
          ],
          { centers: measurementsRef.current.panCenters, finalCenter },
        );
        return;
      }

      if (validationCaseName === "fit-sequence") {
        const alternate = scenes.find((entry) => entry.scene_id !== sceneId) ?? scenes[0];
        if (!alternate) {
          complete([{ name: "fit-availability", pass: false, detail: "No alternate scene available for fit validation." }]);
          return;
        }
        onSelectScene(alternate.scene_id);
        await wait(500);
        const currentBounds = map.getBounds();
        const expectedBounds = sceneLatLngBounds(alternate);
        const pass = expectedBounds
          ? currentBounds.contains(expectedBounds[0]) && currentBounds.contains(expectedBounds[1])
          : false;
        complete(
          [
            {
              name: "fit-contains-scene",
              pass,
              detail: pass ? "Fit landed on the requested scene bounds." : "Fit did not contain the full scene bounds.",
            },
          ],
          { alternateSceneId: alternate.scene_id, mapBounds: currentBounds.toBBoxString(), expectedBounds },
        );
        return;
      }

      if (validationCaseName === "scan-sequence") {
        onResetScan();
        await wait(60);
        onStartScan();
        const targetSteps = Math.min(12, orderedScanTiles.length);
        while ((snapshotRef.current?.scanIndex ?? 0) < targetSteps) {
          await wait(50);
          const currentActive = snapshotRef.current?.activeTileId;
          if (currentActive) {
            measurementsRef.current.scanSequence.push(currentActive);
          }
        }
        onPauseScan();
        const prefix = orderedScanTiles.slice(0, targetSteps).map((tile) => tile.tile_id);
        const sequence = measurementsRef.current.scanSequence.filter((tileId, index, array) => tileId !== array[index - 1]).slice(0, prefix.length);
        complete(
          [
            {
              name: "scan-order",
              pass: prefix.every((tileId, index) => sequence[index] === tileId),
              detail:
                prefix.every((tileId, index) => sequence[index] === tileId)
                  ? "Scan advanced in deterministic row-major order."
                  : "Scan order deviated from the expected prefix.",
            },
          ],
          { expectedPrefix: prefix, observedSequence: sequence },
        );
        return;
      }

      if (validationCaseName === "persistence-sequence") {
        const sessionKey = `map-validation:${validationRunId}:${validationCaseName}`;
        const phase = sessionStorage.getItem(sessionKey);
        if (!phase) {
          onResetScan();
          await wait(50);
          onStartScan();
          await wait(220);
          onPauseScan();
          const expected = {
            sceneId,
            selectedTileId,
            checksum: snapshotRef.current?.persistedReviewChecksum ?? persistedReviewChecksum,
            scannedTileIds: snapshotRef.current?.scannedTileIds ?? scannedTileIds,
          };
          measurementsRef.current.persistenceExpected = expected;
          sessionStorage.setItem(sessionKey, JSON.stringify(expected));
          window.location.reload();
          return;
        }

        if (persistedPhaseRef.current !== phase) {
          persistedPhaseRef.current = phase;
          const expected = JSON.parse(phase) as NonNullable<ScenarioMeasurements["persistenceExpected"]>;
          sessionStorage.removeItem(sessionKey);
          complete(
            [
              {
                name: "persistence-roundtrip",
                pass:
                  expected.sceneId === sceneId &&
                  expected.selectedTileId === selectedTileId &&
                  persistedReviewChecksum === expected.checksum &&
                  checksum(scannedTileIds) === checksum(expected.scannedTileIds),
                detail:
                  expected.sceneId === sceneId &&
                  expected.selectedTileId === selectedTileId &&
                  persistedReviewChecksum === expected.checksum &&
                  checksum(scannedTileIds) === checksum(expected.scannedTileIds)
                    ? "Saved review state restored after reload."
                    : "Reloaded state did not match the saved review state.",
              },
            ],
            { expected, restored: { sceneId, selectedTileId, scannedTileIds, checksum: persistedReviewChecksum } },
          );
        }
      }
    };

    void run();
  }, [
    enabled,
    map,
    onPauseScan,
    onResetScan,
    onSelectScene,
    onStartScan,
    orderedScanTiles,
    persistedReviewChecksum,
    persistedReviewState,
    validationCaseName,
    validationRunId,
    sceneId,
    scenes,
    selectedTile,
    selectedTileId,
    snapshot,
    status,
    zoomAnchor,
    sceneTiles,
    footprintCenterMercator,
    footprintRadiusM,
    footprintTiles,
  ]);

  return {
    snapshot,
    status,
  };
}
