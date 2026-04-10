import type { Dispatch, SetStateAction } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Map as LeafletMap } from "leaflet";
import { Circle, CircleMarker, MapContainer, Rectangle, TileLayer } from "react-leaflet";
import type { BrowserCrosswalkLabelerHandle } from "../hooks/useBrowserCrosswalkLabeler";
import { buildScanBatchJob, exportScanBatchJob, readScanBatchResultFile } from "../scan-batch";
import { validationClassSuffix, type ValidationInteractionPhase, type ValidationPoint } from "../map-validation";
import { MAP_BASEMAPS } from "../map-basemaps";
import { normalizeSceneReviewState } from "../review-state";
import type { BrowserLabelSuggestion, DatasetSummary, DatasetTile, MapBasemap, ReviewState, SceneReviewState } from "../types";
import { useMapValidationRuntime } from "../useMapValidationRuntime";
import {
  bboxAverageSizeM,
  bboxCenterLatLng,
  bboxToMercatorRect,
  bboxToLatLngBounds,
  circleIntersectsMercatorRect,
  formatProbability,
  mercatorToLatLng,
  sceneLabel,
  sceneLatLngBounds,
  sceneMercatorCenter,
  sortScenes,
  tileTone,
} from "../utils";
import { MapEventBridge } from "./MapEventBridge";
import { MapCamera } from "./MapCamera";
import { MapScanControls } from "./MapScanControls";
import { MapValidationPanel } from "./MapValidationPanel";
const DEFAULT_MAP_CENTER: [number, number] = [46.8, 8.25];
const GRID_ZOOM_THRESHOLD = 16;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function dashArrayForZoom(zoom: number, on: number, off: number, minimum = 2) {
  const scale = clamp(0.34 + (zoom - GRID_ZOOM_THRESHOLD) * 0.32, 0.22, 1.8);
  const dashOn = Math.max(minimum, Math.round(on * scale));
  const dashOff = Math.max(minimum, Math.round(off * scale));
  return `${dashOn} ${dashOff}`;
}

type SceneMapProps = {
  summary?: DatasetSummary | null;
  selectedSceneId?: string;
  focusSceneId?: string;
  sceneTiles: DatasetTile[];
  selectedTileId?: string;
  mapZoom: number;
  reviewState: SceneReviewState;
  mapDebug?: boolean;
  validationCase?: string | null;
  validationRunId: string;
  persistedReviewChecksum: string;
  persistedReviewState: ReviewState;
  basemap: MapBasemap;
  labeler: BrowserCrosswalkLabelerHandle;
  browserSuggestions: Record<string, BrowserLabelSuggestion>;
  sceneTilesReady: boolean;
  sceneImagesReady: boolean;
  onSuggestionsChange: Dispatch<SetStateAction<Record<string, BrowserLabelSuggestion>>>;
  onError: (message: string | null) => void;
  onEnsureSceneTiles: () => Promise<void>;
  onEnsureSceneImages: () => Promise<Map<string, string>>;
  onZoomChange: (zoom: number) => void;
  onReviewStateChange: (nextState: SceneReviewState) => void;
  onBasemapChange: (next: MapBasemap) => void;
  onSelectScene: (sceneId: string) => void;
  onSelectTile: (tile: DatasetTile) => void;
};

export function SceneMap({
  summary,
  selectedSceneId,
  focusSceneId,
  sceneTiles,
  selectedTileId,
  mapZoom,
  reviewState,
  mapDebug = false,
  validationCase,
  validationRunId,
  persistedReviewChecksum,
  persistedReviewState,
  basemap,
  labeler,
  browserSuggestions,
  sceneTilesReady,
  sceneImagesReady,
  onSuggestionsChange,
  onError,
  onEnsureSceneTiles,
  onEnsureSceneImages,
  onZoomChange,
  onReviewStateChange,
  onBasemapChange,
  onSelectScene,
  onSelectTile,
}: SceneMapProps) {
  const [scanRunning, setScanRunning] = useState(false);
  const [scanQueued, setScanQueued] = useState(false);
  const [scanPreparing, setScanPreparing] = useState(false);
  const [runtimeScanCount, setRuntimeScanCount] = useState<number | null>(null);
  const [mapInstance, setMapInstance] = useState<LeafletMap | null>(null);
  const [interactionPhase, setInteractionPhase] = useState<ValidationInteractionPhase>("idle");
  const [lastInputSource, setLastInputSource] = useState("initial");
  const [zoomAnchor, setZoomAnchor] = useState<ValidationPoint | null>(null);
  const [cameraUpdates, setCameraUpdates] = useState(0);
  const [pageZoomBlocks, setPageZoomBlocks] = useState(0);
  const mapShellRef = useRef<HTMLDivElement | null>(null);
  const lastPersistedCountRef = useRef(0);
  const scanStartedAtRef = useRef<number | null>(null);
  const scanAbortRef = useRef(false);
  const liveSuggestionsRef = useRef<Record<string, BrowserLabelSuggestion>>({});
  const liveSuggestionsFlushRef = useRef<number | null>(null);
  const renderCountRef = useRef(0);
  const overlayRecomputeRef = useRef(0);
  renderCountRef.current += 1;

  const scenes = useMemo(() => summary?.scenes.slice().sort(sortScenes) ?? [], [summary]);
  const selectedScene = scenes.find((scene) => scene.scene_id === selectedSceneId);
  const focusedScene = scenes.find((scene) => scene.scene_id === (focusSceneId ?? selectedSceneId));
  const selectedBounds = useMemo(() => (focusedScene ? sceneLatLngBounds(focusedScene) : null), [focusedScene?.scene_id, focusedScene?.latitude, focusedScene?.longitude, focusedScene?.size_m]);
  const mapCenter = useMemo<[number, number]>(() => (selectedScene?.latitude && selectedScene?.longitude ? [Number(selectedScene.latitude), Number(selectedScene.longitude)] : DEFAULT_MAP_CENTER), [selectedScene?.latitude, selectedScene?.longitude]);
  const effectiveMapZoom = mapInstance?.getZoom() ?? mapZoom;
  const showGrid = Boolean(selectedScene && sceneTiles.length && effectiveMapZoom >= GRID_ZOOM_THRESHOLD);
  const sceneReviewState = useMemo(() => normalizeSceneReviewState(reviewState), [reviewState]);
  const [liveSuggestions, setLiveSuggestions] = useState<Record<string, BrowserLabelSuggestion>>({});
  const displaySuggestions = useMemo(() => ({ ...browserSuggestions, ...liveSuggestions }), [browserSuggestions, liveSuggestions]);

  const gridStats = useMemo(() => {
    if (!sceneTiles.length) return null;
    const rows = sceneTiles.map((tile) => tile.row);
    const cols = sceneTiles.map((tile) => tile.col);
    const minRow = Math.min(...rows);
    const maxRow = Math.max(...rows);
    const minCol = Math.min(...cols);
    const maxCol = Math.max(...cols);
    const centerRow = (minRow + maxRow) / 2;
    const centerCol = (minCol + maxCol) / 2;
    return { minRow, maxRow, minCol, maxCol, centerRow, centerCol };
  }, [sceneTiles]);

  const averageTileSizeM = useMemo(() => sceneTiles.reduce((sum, tile) => sum + bboxAverageSizeM(tile.bbox_mercator), 0) / Math.max(1, sceneTiles.length), [sceneTiles]);
  const footprintCenterMercator = useMemo(() => {
    if (selectedScene) {
      return sceneMercatorCenter(selectedScene);
    }
    if (!gridStats || !sceneTiles.length) return null;
    const centerCandidates = sceneTiles.filter(
      (tile) => Math.abs(tile.row - gridStats.centerRow) <= 0.5 && Math.abs(tile.col - gridStats.centerCol) <= 0.5,
    );
    if (!centerCandidates.length) return null;
    const rects = centerCandidates.map((tile) => bboxToMercatorRect(tile.bbox_mercator)).filter(Boolean);
    if (!rects.length) return null;
    const minX = Math.min(...rects.map((rect) => rect!.minX));
    const minY = Math.min(...rects.map((rect) => rect!.minY));
    const maxX = Math.max(...rects.map((rect) => rect!.maxX));
    const maxY = Math.max(...rects.map((rect) => rect!.maxY));
    return { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  }, [gridStats, sceneTiles, selectedScene]);
  const footprintRadiusM = useMemo(() => Math.max(averageTileSizeM * sceneReviewState.scan_radius, averageTileSizeM * 1.5), [averageTileSizeM, sceneReviewState.scan_radius]);
  const footprintTiles = useMemo(() => {
    if (!footprintCenterMercator) return [];
    return sceneTiles.filter((tile) => {
      const rect = bboxToMercatorRect(tile.bbox_mercator);
      return rect ? circleIntersectsMercatorRect(rect, footprintCenterMercator.x, footprintCenterMercator.y, footprintRadiusM) : false;
    });
  }, [footprintCenterMercator, footprintRadiusM, sceneTiles]);

  const footprintTileIds = useMemo(() => new Set(footprintTiles.map((tile) => tile.tile_id)), [footprintTiles]);
  const orderedScanTiles = useMemo(() => footprintTiles.slice().sort((a, b) => a.row - b.row || a.col - b.col || a.tile_id.localeCompare(b.tile_id)), [footprintTiles]);
  const persistedScannedTileIds = useMemo(() => sceneReviewState.scanned_tile_ids.filter((tileId) => footprintTileIds.has(tileId)), [footprintTileIds, sceneReviewState.scanned_tile_ids]);
  const scannedTileIds = useMemo(() => {
    if (runtimeScanCount == null) {
      return persistedScannedTileIds;
    }
    return orderedScanTiles.slice(0, runtimeScanCount).map((tile) => tile.tile_id);
  }, [orderedScanTiles, persistedScannedTileIds, runtimeScanCount]);
  const scannedTileIdSet = useMemo(() => new Set(scannedTileIds), [scannedTileIds]);
  const footprintCenter = useMemo<[number, number] | null>(() => (footprintCenterMercator ? mercatorToLatLng(footprintCenterMercator.x, footprintCenterMercator.y) : null), [footprintCenterMercator]);
  const scanFocusBounds = useMemo(() => (footprintCenterMercator ? bboxToLatLngBounds([footprintCenterMercator.x - footprintRadiusM, footprintCenterMercator.y - footprintRadiusM, footprintCenterMercator.x + footprintRadiusM, footprintCenterMercator.y + footprintRadiusM]) : selectedBounds), [footprintCenterMercator, footprintRadiusM, selectedBounds]);
  const scanCircleDashArray = useMemo(() => dashArrayForZoom(effectiveMapZoom, 10, 8), [effectiveMapZoom]);
  const droppedTileDashArray = useMemo(() => dashArrayForZoom(effectiveMapZoom, 6, 8), [effectiveMapZoom]);
  const activeTileDashArray = useMemo(() => dashArrayForZoom(effectiveMapZoom, 6, 6), [effectiveMapZoom]);
  const basemapConfig = MAP_BASEMAPS[basemap];
  const derivedTileSizeM = useMemo(() => Math.max(1, Math.round(averageTileSizeM || 25)), [averageTileSizeM]);

  useEffect(() => {
    setScanRunning(false);
    setScanQueued(false);
    setScanPreparing(false);
    setRuntimeScanCount(null);
    scanAbortRef.current = false;
    liveSuggestionsRef.current = {};
    setLiveSuggestions({});
    lastPersistedCountRef.current = 0;
    scanStartedAtRef.current = null;
    setInteractionPhase("idle");
  }, [selectedSceneId]);

  useEffect(() => () => {
    if (liveSuggestionsFlushRef.current != null) {
      window.cancelAnimationFrame(liveSuggestionsFlushRef.current);
    }
  }, []);

  useEffect(() => {
    const node = mapShellRef.current;
    if (!node) return;

    const handleWheel = (event: WheelEvent) => {
      const rect = node.getBoundingClientRect();
      setZoomAnchor({ x: event.clientX - rect.left, y: event.clientY - rect.top });
      setLastInputSource(event.ctrlKey ? "pinch-gesture" : "wheel");
      if (event.ctrlKey) {
        event.preventDefault();
        setPageZoomBlocks((current) => current + 1);
      }
    };

    const handlePointerDown = () => {
      setLastInputSource("pointer-drag");
    };

    node.addEventListener("wheel", handleWheel, { passive: false });
    node.addEventListener("pointerdown", handlePointerDown);
    return () => {
      node.removeEventListener("wheel", handleWheel);
      node.removeEventListener("pointerdown", handlePointerDown);
    };
  }, []);

  useEffect(() => {
    if (runtimeScanCount == null) return;
    const currentCount = runtimeScanCount;
    const shouldPersist = currentCount > 0 && (currentCount - lastPersistedCountRef.current >= 8 || !scanRunning);
    if (!shouldPersist) return;

    lastPersistedCountRef.current = currentCount;
    onReviewStateChange({
      ...sceneReviewState,
      scanned_tile_ids: orderedScanTiles.slice(0, currentCount).map((tile) => tile.tile_id),
    });
  }, [onReviewStateChange, orderedScanTiles, runtimeScanCount, scanRunning, sceneReviewState]);

  const scanCounts = useMemo(() => {
    const scannedTiles = orderedScanTiles.filter((tile) => scannedTileIdSet.has(tile.tile_id));
    return {
      total: orderedScanTiles.length,
      scanned: scannedTiles.length,
      crosswalk: scannedTiles.filter((tile) => (displaySuggestions[tile.tile_id]?.label ?? tile.predicted_label) === "crosswalk").length,
      noCrosswalk: scannedTiles.filter((tile) => (displaySuggestions[tile.tile_id]?.label ?? tile.predicted_label) === "no_crosswalk").length,
    };
  }, [displaySuggestions, orderedScanTiles, scannedTileIdSet]);

  const activeTile = useMemo(() => {
    if (!scanRunning) return undefined;
    if (labeler.progress.currentTileId) {
      return orderedScanTiles.find((tile) => tile.tile_id === labeler.progress.currentTileId);
    }
    const activeIndex = Math.min(runtimeScanCount ?? 0, Math.max(orderedScanTiles.length - 1, 0));
    return orderedScanTiles[activeIndex];
  }, [labeler.progress.currentTileId, orderedScanTiles, runtimeScanCount, scanRunning]);
  const activeTileBounds = useMemo(() => (activeTile ? bboxToLatLngBounds(activeTile.bbox_mercator) : null), [activeTile?.bbox_mercator, activeTile?.tile_id]);
  const activeTileCenter = useMemo(() => (activeTile ? bboxCenterLatLng(activeTile.bbox_mercator) : null), [activeTile?.bbox_mercator, activeTile?.tile_id]);
  const liveScanStep = scanCounts.scanned + (scanRunning && activeTile ? 1 : 0);
  const activeSuggestion = activeTile ? displaySuggestions[activeTile.tile_id] : undefined;
  const activeSummary = scanPreparing
    ? sceneImagesReady
      ? "Loading the Python scan backend and preparing first results"
      : "Loading scene tiles before scan"
    : activeTile
      ? activeSuggestion
        ? `Active ${activeTile.relative_path} · ${activeSuggestion.label} · ${formatProbability(activeSuggestion.score)}`
        : `Classifying ${activeTile.relative_path}`
      : null;
  const selectedTile = useMemo(() => sceneTiles.find((tile) => tile.tile_id === selectedTileId) ?? orderedScanTiles[0] ?? sceneTiles[0], [orderedScanTiles, sceneTiles, selectedTileId]);

  const flushLiveSuggestions = useCallback((force = false) => {
    if (!force && liveSuggestionsFlushRef.current != null) {
      return;
    }
    const commit = () => {
      liveSuggestionsFlushRef.current = null;
      setLiveSuggestions({ ...liveSuggestionsRef.current });
    };
    if (force) {
      if (liveSuggestionsFlushRef.current != null) {
        window.cancelAnimationFrame(liveSuggestionsFlushRef.current);
        liveSuggestionsFlushRef.current = null;
      }
      commit();
      return;
    }
    liveSuggestionsFlushRef.current = window.requestAnimationFrame(commit);
  }, []);

  useEffect(() => {
    overlayRecomputeRef.current += 1;
  }, [footprintTiles, orderedScanTiles, scannedTileIds.length, selectedTile?.tile_id, activeTile?.tile_id]);

  function setScanRadius(nextRadius: number) {
    onReviewStateChange({
      ...sceneReviewState,
      scan_radius: clamp(nextRadius, 2, 12),
    });
  }

  function setScanDelay(nextDelay: number) {
    onReviewStateChange({
      ...sceneReviewState,
      scan_delay_ms: clamp(nextDelay, 8, 400),
    });
  }

  const runActualScan = useCallback(async () => {
    if (!orderedScanTiles.length) return;
    scanAbortRef.current = false;
    setScanQueued(false);
    setScanPreparing(true);
    setInteractionPhase("scan");
    lastPersistedCountRef.current = 0;
    scanStartedAtRef.current = null;
    setRuntimeScanCount(0);
    liveSuggestionsRef.current = {};
    setLiveSuggestions({});
    onError(null);
    onSuggestionsChange((current) => {
      const next = { ...current };
      for (const tile of orderedScanTiles) {
        delete next[tile.tile_id];
      }
      return next;
    });
    onReviewStateChange({
      ...sceneReviewState,
      scanned_tile_ids: [],
    });
    try {
      await labeler.ensureReady();
      setScanPreparing(false);
      setScanRunning(true);
      await labeler.runSceneLabeling(orderedScanTiles, labeler.defaultPromptText, 0.32, {
        shouldAbort: () => scanAbortRef.current,
        onSuggestion: (suggestion, index) => {
          const nextCount = index + 1;
          setRuntimeScanCount(nextCount);
          liveSuggestionsRef.current[suggestion.tile_id] = suggestion;
          if (nextCount === 1 || nextCount % 4 === 0 || nextCount === orderedScanTiles.length) {
            flushLiveSuggestions(nextCount === orderedScanTiles.length);
          }
          if (nextCount - lastPersistedCountRef.current >= 8 || nextCount === orderedScanTiles.length) {
            lastPersistedCountRef.current = nextCount;
            onReviewStateChange({
              ...sceneReviewState,
              scanned_tile_ids: orderedScanTiles.slice(0, nextCount).map((tile) => tile.tile_id),
            });
          }
        },
      });
    } catch (reason) {
      onError(String(reason));
    } finally {
      setScanPreparing(false);
      flushLiveSuggestions(true);
      onSuggestionsChange((current) => ({
        ...current,
        ...liveSuggestionsRef.current,
      }));
      scanAbortRef.current = false;
      setScanRunning(false);
      setInteractionPhase("idle");
      scanStartedAtRef.current = null;
    }
  }, [flushLiveSuggestions, labeler, onError, onReviewStateChange, onSuggestionsChange, orderedScanTiles, sceneReviewState]);

  useEffect(() => {
    if (!scanQueued || scanRunning || !orderedScanTiles.length) return;
    void runActualScan();
  }, [orderedScanTiles.length, runActualScan, scanQueued, scanRunning]);

  async function handleStartScan() {
    onError(null);
    if (!sceneTilesReady) {
      setScanQueued(true);
      try {
        await onEnsureSceneTiles();
      } catch (reason) {
        setScanQueued(false);
        onError(String(reason));
        return;
      }
    }
    if (mapInstance && scanFocusBounds && effectiveMapZoom < GRID_ZOOM_THRESHOLD) {
      setScanQueued(true);
      setInteractionPhase("fit");
      setLastInputSource("auto-focus");
      mapInstance.fitBounds(scanFocusBounds, { padding: [24, 24], animate: false });
      return;
    }
    if (!showGrid || !orderedScanTiles.length) {
      setScanQueued(false);
      return;
    }
    void runActualScan();
  }

  function handleResetScan() {
    scanAbortRef.current = true;
    setScanRunning(false);
    setScanQueued(false);
    setScanPreparing(false);
    setInteractionPhase("idle");
    scanStartedAtRef.current = null;
    setRuntimeScanCount(0);
    lastPersistedCountRef.current = 0;
    onSuggestionsChange((current) => {
      const next = { ...current };
      for (const tile of orderedScanTiles) {
        delete next[tile.tile_id];
      }
      return next;
    });
    onReviewStateChange({
      ...sceneReviewState,
      scanned_tile_ids: [],
    });
  }

  function handleExportBatchJob() {
    if (!summary || !selectedScene || !orderedScanTiles.length) {
      onError("Select a scene with a visible scan area before exporting a batch job.");
      return;
    }
    onError(null);
    const job = buildScanBatchJob({
      summary,
      scene: selectedScene,
      tileSizeM: derivedTileSizeM,
      scanRadiusTiles: sceneReviewState.scan_radius,
      threshold: 0.32,
      promptsText: labeler.defaultPromptText,
      tiles: orderedScanTiles,
    });
    exportScanBatchJob(job);
  }

  async function handleImportBatchResult() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json,.json";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const imported = await readScanBatchResultFile(file);
        if (selectedScene && imported.job.scene.scene_id !== selectedScene.scene_id) {
          throw new Error(
            `This result is for ${imported.job.scene.scene_id}, but the map is showing ${selectedScene.scene_id}.`,
          );
        }
        onSuggestionsChange((current) => ({
          ...current,
          ...imported.results,
        }));
        onError(null);
      } catch (reason) {
        onError(String(reason));
      }
    };
    input.click();
  }

  const handleMapReady = useCallback((map: LeafletMap) => {
    setMapInstance((current) => (current === map ? current : map));
  }, []);

  const handleCameraUpdate = useCallback(() => {
    setCameraUpdates((current) => current + 1);
  }, []);

  const handleInteractionPhase = useCallback((phase: ValidationInteractionPhase) => {
    setInteractionPhase((current) => (scanRunning && phase === "idle" ? "scan" : current === phase ? current : phase));
  }, [scanRunning]);

  const { snapshot: validationSnapshot } = useMapValidationRuntime({
    enabled: mapDebug,
    validationRunId,
    validationCase: validationCase ?? null,
    map: mapInstance,
    mapShell: mapShellRef.current,
    scenes,
    sceneTiles,
    sceneId: selectedSceneId,
    selectedTileId,
    activeTileId: activeTile?.tile_id,
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
    counters: {
      componentRenders: renderCountRef.current,
      overlayRecomputes: overlayRecomputeRef.current,
      cameraUpdates,
      pageZoomBlocks,
      consoleErrors: 0,
    },
    onSelectScene,
    onStartScan: handleStartScan,
    onPauseScan: () => {
      scanAbortRef.current = true;
      setScanRunning(false);
      setInteractionPhase("idle");
    },
    onResetScan: handleResetScan,
  });

  return (
    <section className="panel map-panel">
      <MapScanControls
        activeSummary={activeSummary}
        basemap={basemap}
        crosswalkCount={scanCounts.crosswalk}
        liveScanStep={liveScanStep}
        mapZoom={effectiveMapZoom}
        noCrosswalkCount={scanCounts.noCrosswalk}
        onBasemapChange={onBasemapChange}
        onPauseScan={() => {
          scanAbortRef.current = true;
          setScanRunning(false);
          setScanPreparing(false);
          setInteractionPhase("idle");
        }}
        scanPreparing={scanPreparing}
        sceneTilesReady={sceneTilesReady}
        sceneImagesReady={sceneImagesReady}
        onResetScan={handleResetScan}
        onScanDelayChange={setScanDelay}
        onScanRadiusChange={setScanRadius}
        onStartScan={handleStartScan}
        onExportBatchJob={handleExportBatchJob}
        onImportBatchResult={handleImportBatchResult}
        scanDelay={sceneReviewState.scan_delay_ms}
        scanQueued={scanQueued}
        scanRadius={sceneReviewState.scan_radius}
        scanRunning={scanRunning}
        scannedCount={scanCounts.scanned}
        sceneLabel={selectedScene ? sceneLabel(selectedScene) : "World overview"}
        showGrid={showGrid}
        totalTilesInCircle={scanCounts.total}
      />

      <div className="map-shell map-shell-large" ref={mapShellRef}>
        <MapContainer
          className="leaflet-map"
          center={mapCenter}
          zoom={mapZoom}
          scrollWheelZoom
          preferCanvas={!mapDebug}
          zoomAnimation={false}
          fadeAnimation={false}
          markerZoomAnimation={false}
          inertia
          inertiaDeceleration={2400}
          inertiaMaxSpeed={1800}
          zoomSnap={0.15}
          zoomDelta={0.3}
          wheelPxPerZoomLevel={150}
          wheelDebounceTime={24}
        >
          <TileLayer
            attribution={basemapConfig.attribution}
            crossOrigin="anonymous"
            key={basemap}
            maxZoom={basemapConfig.maxZoom}
            keepBuffer={6}
            updateWhenIdle
            updateWhenZooming={false}
            url={basemapConfig.url}
          />
          <MapEventBridge
            onMapReady={handleMapReady}
            onZoomChange={onZoomChange}
            onCameraUpdate={handleCameraUpdate}
            onInteractionPhase={handleInteractionPhase}
          />
          <MapCamera bounds={selectedBounds} focusKey={focusSceneId ?? selectedSceneId} onFitPhase={handleInteractionPhase} />

          {scenes.map((scene) => {
            const latitude = Number(scene.latitude ?? 0);
            const longitude = Number(scene.longitude ?? 0);
            const isSelected = scene.scene_id === selectedSceneId;
            return (
              <CircleMarker
                key={scene.scene_id}
                center={[latitude, longitude]}
                radius={isSelected ? 10 : 7}
                pathOptions={{
                  color: isSelected ? "#f5c05d" : "#29c37d",
                  weight: isSelected ? 3 : 2,
                  fillOpacity: isSelected ? 0.74 : 0.45,
                }}
                eventHandlers={{ click: () => onSelectScene(scene.scene_id) }}
              />
            );
          })}

          {showGrid && footprintCenter ? (
            <>
              <Circle
                center={footprintCenter}
                radius={footprintRadiusM * 1.02}
                pathOptions={{
                  color: "rgba(255, 246, 214, 0.9)",
                  weight: 8,
                  opacity: 0.34,
                  fillColor: "rgba(245, 192, 93, 0.1)",
                  fillOpacity: 0.1,
                  className: "scan-circle-glow",
                }}
              />
              <Circle
                center={footprintCenter}
                radius={footprintRadiusM}
                pathOptions={{
                  color: "rgba(245, 192, 93, 0.98)",
                  weight: 3,
                  fillColor: "rgba(245, 192, 93, 0.08)",
                  fillOpacity: 0.08,
                  dashArray: scanCircleDashArray,
                  className: "scan-circle-ring validation-scan-circle",
                }}
              />
              <CircleMarker
                center={footprintCenter}
                radius={7}
                pathOptions={{
                  color: "#fff2c2",
                  weight: 2,
                  fillColor: "#f5c05d",
                  fillOpacity: 0.95,
                  className: "scan-circle-center",
                }}
              />
            </>
          ) : null}

          {showGrid
            ? footprintTiles.map((tile) => {
                const bounds = bboxToLatLngBounds(tile.bbox_mercator);
                if (!bounds) return null;
                const isScanned = scannedTileIdSet.has(tile.tile_id);
                const tone = tileTone(tile);
                const isSelectedTile = tile.tile_id === selectedTileId;
                const isActive = tile.tile_id === activeTile?.tile_id;
                const browserSuggestion = displaySuggestions[tile.tile_id];
                const displayTone =
                  browserSuggestion?.label === "crosswalk"
                    ? "crosswalk"
                    : browserSuggestion?.label === "no_crosswalk"
                      ? "no_crosswalk"
                      : tone;
                const color =
                  displayTone === "crosswalk" ? "#29c37d" : displayTone === "no_crosswalk" ? "#ff6b6b" : "rgba(141, 161, 184, 0.8)";

                return (
                  <Rectangle
                    key={tile.tile_id}
                    bounds={bounds}
                    pathOptions={{
                      className: `validation-tile validation-tile-${validationClassSuffix(tile.tile_id)}${isSelectedTile ? " validation-selected-tile" : ""}${isActive ? " validation-active-tile" : ""}`,
                      color: isSelectedTile ? "#f5c05d" : isActive ? "#f5c05d" : color,
                      weight: isSelectedTile || isActive ? 4 : browserSuggestion ? 3 : isScanned ? 3 : 2,
                      fillColor: color,
                      fillOpacity: browserSuggestion ? 0.28 : isScanned ? 0.44 : 0.18,
                      dashArray: browserSuggestion ? activeTileDashArray : tone === "dropped" ? droppedTileDashArray : undefined,
                      opacity: isScanned ? 1 : 0.92,
                    }}
                    eventHandlers={{ click: () => onSelectTile(tile) }}
                  />
                );
              })
            : null}

          {showGrid && activeTileBounds ? (
            <Rectangle
              bounds={activeTileBounds}
              pathOptions={{
                color: "#fff2c2",
                weight: 5,
                opacity: 1,
                fillColor: "#f5c05d",
                fillOpacity: 0.18,
                dashArray: activeTileDashArray,
                className: "scan-active-rect",
              }}
            />
          ) : null}

          {showGrid && activeTileCenter ? (
            <CircleMarker
              center={activeTileCenter}
              radius={8}
              pathOptions={{
                color: "#fff8dd",
                weight: 2,
                fillColor: "#f5c05d",
                fillOpacity: 1,
                className: "scan-active-dot",
              }}
            />
          ) : null}
        </MapContainer>
      </div>
      {mapDebug ? <MapValidationPanel snapshot={validationSnapshot} /> : null}
    </section>
  );
}
