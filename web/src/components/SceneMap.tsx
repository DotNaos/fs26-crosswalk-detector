import type { Dispatch, ReactNode, SetStateAction } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card } from "@heroui/react";
import type { Map as LeafletMap } from "leaflet";
import { MapContainer } from "react-leaflet";
import type { AutopilotBvhCell, AutopilotPlan } from "../autopilot-planner";
import type { BrowserCrosswalkLabelerHandle } from "../hooks/useBrowserCrosswalkLabeler";
import { useSceneFootprint } from "../hooks/useSceneFootprint";
import { buildScanBatchJob, exportScanBatchJob, readScanBatchResultFile } from "../scan-batch";
import { type ValidationInteractionPhase, type ValidationPoint } from "../map-validation";
import { normalizeSceneReviewState } from "../review-state";
import type { BrowserLabelSuggestion, DatasetSummary, DatasetTile, MapBasemap, ReviewState, SceneReviewState } from "../types";
import { useMapValidationRuntime } from "../useMapValidationRuntime";
import {
  formatProbability,
  sceneLabel,
  sceneLatLngBounds,
  sortScenes,
} from "../utils";
import { MapScanControls } from "./MapScanControls";
import { MapValidationPanel } from "./MapValidationPanel";
import { SceneBasemapSwitch } from "./SceneBasemapSwitch";
import { SceneMapLayers } from "./SceneMapLayers";
import { DEFAULT_MAP_CENTER, GRID_ZOOM_THRESHOLD, REMOTE_SCAN_THRESHOLD, clamp, dashArrayForZoom, snapBboxToLeafletTileGrid } from "./scene-map-geometry";

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
  autopilotPlan?: unknown;
  isMobileLayout?: boolean;
  onMobileScanPanelChange?: (panel: ReactNode | null) => void;
  labeler: BrowserCrosswalkLabelerHandle;
  browserSuggestions: Record<string, BrowserLabelSuggestion>;
  remoteActiveTileId?: string;
  remoteScanActive?: boolean;
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
  autopilotPlan,
  isMobileLayout = false,
  onMobileScanPanelChange,
  labeler,
  browserSuggestions,
  remoteActiveTileId,
  remoteScanActive = false,
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
  const visibleAutopilotPlan = useMemo(() => {
    const candidate = autopilotPlan as Partial<AutopilotPlan> | undefined;
    return candidate?.mode === "swiss-lowres-urban-grid" && Array.isArray(candidate.cells) ? (candidate as AutopilotPlan) : null;
  }, [autopilotPlan]);
  const focusedSceneId = focusSceneId ?? (visibleAutopilotPlan ? undefined : selectedSceneId);
  const focusedScene = scenes.find((scene) => scene.scene_id === focusedSceneId);
  const selectedBounds = useMemo(() => {
    if (focusedScene) return sceneLatLngBounds(focusedScene);
    if (visibleAutopilotPlan && !focusSceneId) return bboxToLatLngBounds(visibleAutopilotPlan.coarseGrid.bboxMercator);
    return null;
  }, [focusSceneId, focusedScene?.scene_id, focusedScene?.latitude, focusedScene?.longitude, focusedScene?.size_m, visibleAutopilotPlan]);
  const mapCenter = useMemo<[number, number]>(() => (selectedScene?.latitude && selectedScene?.longitude ? [Number(selectedScene.latitude), Number(selectedScene.longitude)] : DEFAULT_MAP_CENTER), [selectedScene?.latitude, selectedScene?.longitude]);
  const effectiveMapZoom = mapInstance?.getZoom() ?? mapZoom;
  const showAutopilotDetailGrid = effectiveMapZoom >= 10;
  const cityFineGridCells = useMemo(
    () => visibleAutopilotPlan?.coarseCells.filter((cell) => cell.status !== "background") ?? [],
    [visibleAutopilotPlan],
  );
  const bvhOverlayCells = useMemo(() => {
    const cells = visibleAutopilotPlan?.bvhCells ?? [];
    if (!cells.length) return [];
    return cells.map((cell) => ({
      ...cell,
      bboxMercator: snapBboxToLeafletTileGrid(cell.bboxMercator, cell.sizeM ?? visibleAutopilotPlan.sceneSizeM),
    }));
  }, [visibleAutopilotPlan]);
  const showGrid = Boolean(selectedScene && sceneTiles.length && effectiveMapZoom >= GRID_ZOOM_THRESHOLD);
  const sceneReviewState = useMemo(() => normalizeSceneReviewState(reviewState), [reviewState]);
  const autopilotSceneMode = Boolean(selectedScene?.autopilot_cell_id);
  const [liveSuggestions, setLiveSuggestions] = useState<Record<string, BrowserLabelSuggestion>>({});
  const displaySuggestions = useMemo(() => ({ ...browserSuggestions, ...liveSuggestions }), [browserSuggestions, liveSuggestions]);
  const { derivedTileSizeM, footprintCenter, footprintCenterMercator, footprintRadiusM, footprintTileIds, footprintTiles, orderedScanTiles } =
    useSceneFootprint({
      scene: selectedScene,
      sceneTiles,
      sceneReviewState,
      mode: autopilotSceneMode ? "scene" : "radius",
    });
  const persistedScannedTileIds = useMemo(() => sceneReviewState.scanned_tile_ids.filter((tileId) => footprintTileIds.has(tileId)), [footprintTileIds, sceneReviewState.scanned_tile_ids]);
  const scannedTileIds = useMemo(() => {
    if (runtimeScanCount == null) {
      return persistedScannedTileIds;
    }
    return orderedScanTiles.slice(0, runtimeScanCount).map((tile) => tile.tile_id);
  }, [orderedScanTiles, persistedScannedTileIds, runtimeScanCount]);
  const scannedTileIdSet = useMemo(() => new Set(scannedTileIds), [scannedTileIds]);
  const scanFocusBounds = useMemo(() => (footprintCenterMercator ? bboxToLatLngBounds([footprintCenterMercator.x - footprintRadiusM, footprintCenterMercator.y - footprintRadiusM, footprintCenterMercator.x + footprintRadiusM, footprintCenterMercator.y + footprintRadiusM]) : selectedBounds), [footprintCenterMercator, footprintRadiusM, selectedBounds]);
  const scanCircleDashArray = useMemo(() => dashArrayForZoom(effectiveMapZoom, 10, 8), [effectiveMapZoom]);
  const droppedTileDashArray = useMemo(() => dashArrayForZoom(effectiveMapZoom, 6, 8), [effectiveMapZoom]);
  const activeTileDashArray = useMemo(() => dashArrayForZoom(effectiveMapZoom, 6, 6), [effectiveMapZoom]);

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
    if (scanRunning && labeler.progress.currentTileId) {
      return orderedScanTiles.find((tile) => tile.tile_id === labeler.progress.currentTileId);
    }
    if (scanRunning) {
      const activeIndex = Math.min(runtimeScanCount ?? 0, Math.max(orderedScanTiles.length - 1, 0));
      return orderedScanTiles[activeIndex];
    }
    if (remoteActiveTileId) {
      return orderedScanTiles.find((tile) => tile.tile_id === remoteActiveTileId) ?? sceneTiles.find((tile) => tile.tile_id === remoteActiveTileId);
    }
    return undefined;
  }, [labeler.progress.currentTileId, orderedScanTiles, remoteActiveTileId, remoteScanActive, runtimeScanCount, scanRunning, sceneTiles]);
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
  const mobileScanPanel = useMemo(
    () => (
      <MapScanControls
        activeSummary={activeSummary}
        autopilotMode={autopilotSceneMode}
        crosswalkCount={scanCounts.crosswalk}
        isMobileLayout
        liveScanStep={liveScanStep}
        mapZoom={effectiveMapZoom}
        noCrosswalkCount={scanCounts.noCrosswalk}
        onPauseScan={() => {
          scanAbortRef.current = true;
          setScanRunning(false);
          setScanPreparing(false);
          setInteractionPhase("idle");
        }}
        scanPreparing={scanPreparing}
        sceneTilesReady={sceneTilesReady}
        sceneImagesReady={sceneImagesReady}
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
    ),
    [
      activeSummary,
      autopilotSceneMode,
      effectiveMapZoom,
      handleExportBatchJob,
      handleImportBatchResult,
      handleStartScan,
      isMobileLayout,
      liveScanStep,
      scanCounts.crosswalk,
      scanCounts.noCrosswalk,
      scanCounts.scanned,
      scanCounts.total,
      scanPreparing,
      scanQueued,
      scanRunning,
      sceneImagesReady,
      sceneReviewState.scan_delay_ms,
      sceneReviewState.scan_radius,
      sceneTilesReady,
      selectedScene,
      setScanDelay,
      setScanRadius,
      showGrid,
    ],
  );

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
    if (!onMobileScanPanelChange) return;
    if (!isMobileLayout) {
      onMobileScanPanelChange(null);
      return;
    }
    onMobileScanPanelChange(mobileScanPanel);
    return () => onMobileScanPanelChange(null);
  }, [
    activeSummary,
    autopilotSceneMode,
    effectiveMapZoom,
    isMobileLayout,
    liveScanStep,
    onMobileScanPanelChange,
    scanCounts.scanned,
    scanCounts.total,
    scanPreparing,
    scanQueued,
    scanRunning,
    sceneImagesReady,
    sceneReviewState.scan_radius,
    sceneTilesReady,
    selectedScene?.scene_id,
    showGrid,
  ]);

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
      threshold: REMOTE_SCAN_THRESHOLD,
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
        onReviewStateChange({
          ...sceneReviewState,
          scanned_tile_ids: imported.tiles.map((tile) => tile.tile_id),
        });
        setRuntimeScanCount(imported.summary.total);
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
    <section className="relative h-full w-full overflow-hidden" ref={mapShellRef}>
      <div className="h-full w-full">
        <MapContainer
          className="h-full w-full"
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
          <SceneMapLayers
            activeTile={activeTile}
            activeTileDashArray={activeTileDashArray}
            basemap={basemap}
            bvhOverlayCells={bvhOverlayCells}
            cityFineGridCells={cityFineGridCells}
            displaySuggestions={displaySuggestions}
            droppedTileDashArray={droppedTileDashArray}
            effectiveMapZoom={effectiveMapZoom}
            focusSceneId={focusSceneId}
            focusedSceneId={selectedSceneId}
            footprintCenter={footprintCenter}
            footprintRadiusM={footprintRadiusM}
            footprintTiles={footprintTiles}
            onCameraUpdate={handleCameraUpdate}
            onInteractionPhase={handleInteractionPhase}
            onMapReady={handleMapReady}
            onSelectScene={onSelectScene}
            onSelectTile={onSelectTile}
            onZoomChange={onZoomChange}
            scanCircleDashArray={scanCircleDashArray}
            scannedTileIdSet={scannedTileIdSet}
            scenes={scenes}
            selectedBounds={selectedBounds}
            selectedSceneId={selectedSceneId}
            selectedTileId={selectedTileId}
            showAutopilotDetailGrid={showAutopilotDetailGrid}
            showGrid={showGrid}
            visibleAutopilotPlan={visibleAutopilotPlan}
          />
        </MapContainer>
      </div>
      {!isMobileLayout ? (
        <div className="pointer-events-none absolute bottom-4 left-4 z-[1000]">
          <MapScanControls
            activeSummary={activeSummary}
            autopilotMode={autopilotSceneMode}
            crosswalkCount={scanCounts.crosswalk}
            liveScanStep={liveScanStep}
            mapZoom={effectiveMapZoom}
            noCrosswalkCount={scanCounts.noCrosswalk}
            onPauseScan={() => {
              scanAbortRef.current = true;
              setScanRunning(false);
              setScanPreparing(false);
              setInteractionPhase("idle");
            }}
            scanPreparing={scanPreparing}
            sceneTilesReady={sceneTilesReady}
            sceneImagesReady={sceneImagesReady}
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
        </div>
      ) : null}
      {!isMobileLayout ? (
        <div className="pointer-events-none absolute bottom-4 left-1/2 z-[1000] -translate-x-1/2">
          <SceneBasemapSwitch basemap={basemap} onBasemapChange={onBasemapChange} />
        </div>
      ) : null}
      {mapDebug ? (
        <div className="pointer-events-none absolute left-4 top-4 z-[1000]">
          <MapValidationPanel snapshot={validationSnapshot} />
        </div>
      ) : null}
    </section>
  );
}
