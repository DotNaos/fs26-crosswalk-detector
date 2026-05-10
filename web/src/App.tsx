import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { createAutopilotDataset, createDataset, ensureSceneImages, listDatasets, loadAutopilotPlan, loadConfig, loadDatasetMeta, loadReviewState, loadScene, saveConfig, saveReviewState, updateTile } from "./api";
import type { AutopilotPlan } from "./autopilot-planner";
import { asAutopilotPlan, FALLBACK_EXPORT, FALLBACK_RUN, needsRealAutopilotPlanRefresh } from "./app-autopilot";
import { DEFAULT_BROWSER_CONFIG } from "./default-config";
import { checksum, sameSceneReviewState } from "./map-validation";
import { normalizeReviewState, sceneReviewStateFor } from "./review-state";
import { buildScanBatchJob, type ScanBatchResult } from "./scan-batch";
import type { BrowserLabelSuggestion, DatasetListEntry, DatasetSummary, MapBasemap, RealDatasetConfig, ReviewState, ScenePayload } from "./types";
import { sortScenes } from "./utils";
import { AppDrawers } from "./components/AppDrawers";
import { AppOverlayPanels } from "./components/AppOverlayPanels";
import { SceneMap } from "./components/SceneMap";
import { summarizeErrorMessage } from "./error-summary";
import { useBrowserCrosswalkLabeler } from "./hooks/useBrowserCrosswalkLabeler";
import { useMobileLayout } from "./hooks/useMobileLayout";
import { useMobileRemoteJob } from "./hooks/useMobileRemoteJob";
import { useSceneFootprint } from "./hooks/useSceneFootprint";

const REMOTE_SCAN_THRESHOLD = 0.5, ACTIVE_REMOTE_STATUSES = new Set(["bootstrapping", "syncing", "submitting", "queued", "running"]);

export default function App() {
  const [datasets, setDatasets] = useState<DatasetListEntry[]>([]);
  const [runName, setRunName] = useState(FALLBACK_RUN);
  const [exportName, setExportName] = useState(FALLBACK_EXPORT);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [sceneCache, setSceneCache] = useState<Record<string, ScenePayload>>({});
  const [focusSceneId, setFocusSceneId] = useState<string | undefined>();
  const [reviewState, setReviewState] = useState<ReviewState>(normalizeReviewState());
  const [reviewStateReady, setReviewStateReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const [config, setConfig] = useState<RealDatasetConfig | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [isErrorDrawerOpen, setIsErrorDrawerOpen] = useState(false);
  const [basemap, setBasemap] = useState<MapBasemap>(() => {
    const stored = window.localStorage.getItem("crosswalk-review:basemap");
    return stored === "swisstopo" || stored === "roads" ? stored : "osm";
  });
  const [browserSuggestions, setBrowserSuggestions] = useState<Record<string, BrowserLabelSuggestion>>({});
  const [sceneImagesReady, setSceneImagesReady] = useState<Record<string, boolean>>({});
  const labeler = useBrowserCrosswalkLabeler({
    runName,
    exportName,
    scenes: config?.scenes ?? [],
    tileSizeM: config?.tile_size_m ?? 25,
  });
  const [isTerminalDrawerOpen, setIsTerminalDrawerOpen] = useState(false);
  const [isCreateDatasetDrawerOpen, setIsCreateDatasetDrawerOpen] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState("");
  const [newDatasetSceneId, setNewDatasetSceneId] = useState(DEFAULT_BROWSER_CONFIG.scenes[0]?.scene_id ?? "");
  const [creatingDataset, setCreatingDataset] = useState(false);
  const [creatingAutopilot, setCreatingAutopilot] = useState(false);
  const [autopilotPreviewPlan, setAutopilotPreviewPlan] = useState<AutopilotPlan | null>(null);
  const isMobileLayout = useMobileLayout();
  const { activeRemoteJobId, setActiveRemoteJobId, activeRemoteJob, setActiveRemoteJob, remoteSnapshot } = useMobileRemoteJob(isMobileLayout);
  const [mobileScanPanel, setMobileScanPanel] = useState<ReactNode>(null);
  const urlParams = useMemo(() => new URLSearchParams(window.location.search), []);
  const mapDebug = urlParams.get("mapDebug") === "1";
  const validationCase = urlParams.get("validationCase");
  const validationRunId = urlParams.get("validationRunId") ?? "manual";

  useEffect(() => {
    listDatasets()
      .then((entries) => {
        setDatasets(entries);
        const preferred =
          entries.find((entry) => entry.run_name === FALLBACK_RUN && entry.export_name === FALLBACK_EXPORT) ??
          entries[0];
        if (preferred) {
          setRunName(preferred.run_name);
          setExportName(preferred.export_name);
        }
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    let canceled = false;
    setSceneCache({});
    setBrowserSuggestions({});
    setSceneImagesReady({});
    setReviewStateReady(false);
    Promise.all([loadDatasetMeta(runName, exportName), loadReviewState(runName, exportName)])
      .then(([nextSummary, loadedReviewState]) => {
        if (canceled) return;
        setSummary(nextSummary);
        const normalizedReviewState = normalizeReviewState(loadedReviewState);
        const selectedSceneId =
          normalizedReviewState.selected_scene_id &&
          nextSummary.scenes.some((scene) => scene.scene_id === normalizedReviewState.selected_scene_id)
            ? normalizedReviewState.selected_scene_id
            : nextSummary.scenes[0]?.scene_id;
        setReviewState({
          ...normalizedReviewState,
          selected_scene_id: selectedSceneId,
          selected_tile_id: selectedSceneId ? normalizedReviewState.selected_tile_id : undefined,
        });
        setReviewStateReady(true);
        setError(null);
        setFocusSceneId(undefined);
      })
      .catch((reason) => {
        if (canceled) return;
        if (String(reason).includes("Unknown scene")) {
          setError(null);
          return;
        }
        setError(String(reason));
      });
    return () => {
      canceled = true;
    };
  }, [runName, exportName]);

  useEffect(() => {
    let canceled = false;
    loadConfig(runName, exportName)
      .then((loadedConfig) => {
        if (canceled) return;
        setConfig(loadedConfig);
      })
      .catch((reason) => {
        if (canceled) return;
        if (String(reason).includes("Unknown scene")) {
          setError(null);
          return;
        }
        setError(String(reason));
      });
    return () => {
      canceled = true;
    };
  }, [runName, exportName]);

  useEffect(() => {
    if (!config || !needsRealAutopilotPlanRefresh(config.autopilot)) return;
    let canceled = false;
    const stalePlan = config.autopilot;
    loadAutopilotPlan(config.target_per_class, stalePlan.maxPanels, stalePlan.sceneBudget)
      .then((nextPlan) => {
        if (canceled) return null;
        return saveConfig(runName, exportName, {
          ...config,
          target_per_class: nextPlan.targetPositiveCount,
          tile_size_m: nextPlan.tileSizeM,
          scenes: nextPlan.scenes,
          autopilot: nextPlan,
        });
      })
      .then((nextConfig) => {
        if (canceled || !nextConfig) return;
        setConfig(nextConfig);
        setSceneCache({});
        setReviewState((current) => ({
          ...current,
          selected_scene_id: nextConfig.scenes[0]?.scene_id,
          selected_tile_id: undefined,
        }));
        setFocusSceneId(undefined);
        void loadDatasetMeta(runName, exportName).then((nextSummary) => {
          if (!canceled) setSummary(nextSummary);
        });
      })
      .catch((reason) => {
        if (!canceled) setError(String(reason));
      });
    return () => {
      canceled = true;
    };
  }, [config, exportName, runName]);

  useEffect(() => {
    window.localStorage.setItem("crosswalk-review:basemap", basemap);
  }, [basemap]);

  useEffect(() => {
    if (!reviewStateReady || !summary) return;
    const selectedSceneId = reviewState.selected_scene_id;
    if (!selectedSceneId || sceneCache[selectedSceneId]) return;
    if (summary && !summary.scenes.some((scene) => scene.scene_id === selectedSceneId)) {
      const fallbackSceneId = summary.scenes[0]?.scene_id;
      setReviewState((current) => ({
        ...current,
        selected_scene_id: fallbackSceneId,
        selected_tile_id: undefined,
      }));
      setError(null);
      return;
    }
    let canceled = false;
    loadScene(runName, exportName, selectedSceneId)
      .then((payload) => {
        if (canceled) return;
        setSceneCache((current) => ({ ...current, [payload.scene.scene_id]: payload }));
        setSceneImagesReady((current) => ({ ...current, [payload.scene.scene_id]: true }));
        setSummary(payload.summary);
        setReviewState((current) => {
          const tileStillExists = current.selected_tile_id && payload.tiles.some((tile) => tile.tile_id === current.selected_tile_id);
          return {
            ...current,
            selected_tile_id: tileStillExists ? current.selected_tile_id : payload.tiles[0]?.tile_id,
          };
        });
      })
      .catch((reason) => {
        if (!canceled) {
          if (String(reason).includes("Unknown scene")) {
            setError(null);
            return;
          }
          setError(String(reason));
        }
      });
    ensureSceneImages(runName, exportName, selectedSceneId)
      .then((imageUrls) => {
        if (canceled) return;
        setSceneCache((current) => {
          const existing = current[selectedSceneId];
          if (!existing) return current;
          return {
            ...current,
            [selectedSceneId]: {
              ...existing,
              tiles: existing.tiles.map((tile) => ({
                ...tile,
                image_path: imageUrls.get(tile.tile_id) ?? tile.image_path,
              })),
            },
          };
        });
        setSceneImagesReady((current) => ({ ...current, [selectedSceneId]: true }));
      })
      .catch((reason) => {
        if (!canceled) {
          if (String(reason).includes("Unknown scene")) {
            setError(null);
            return;
          }
          setError(String(reason));
        }
      });
    return () => {
      canceled = true;
    };
  }, [exportName, reviewState.selected_scene_id, reviewStateReady, runName, sceneCache, summary]);

  useEffect(() => {
    if (!reviewStateReady) return;
    const timer = window.setTimeout(() => {
      void saveReviewState(runName, exportName, reviewState).catch((reason) => setError(String(reason)));
    }, 180);
    return () => window.clearTimeout(timer);
  }, [exportName, reviewState, reviewStateReady, runName]);

  const scenes = useMemo(() => summary?.scenes.slice().sort(sortScenes) ?? [], [summary]);
  const selectedSceneId = reviewState.selected_scene_id;
  const selectedTileId = reviewState.selected_tile_id;
  const mapZoom = reviewState.map_zoom ?? 8;
  const selectedScene = scenes.find((scene) => scene.scene_id === selectedSceneId);
  const scenePayload = selectedScene ? sceneCache[selectedScene.scene_id] : undefined;
  const sceneTiles = scenePayload?.tiles ?? [];
  const selectedTile = sceneTiles.find((tile) => tile.tile_id === selectedTileId) ?? sceneTiles[0];
  const selectedSuggestion = selectedTile ? browserSuggestions[selectedTile.tile_id] : undefined;
  const visibleActiveRemoteJob = activeRemoteJob && scenes.some((scene) => scene.scene_id === activeRemoteJob.scene_id) ? activeRemoteJob : null;
  const remoteScanActive = visibleActiveRemoteJob ? ACTIVE_REMOTE_STATUSES.has(visibleActiveRemoteJob.status) : false;
  const remoteActiveTileId =
    visibleActiveRemoteJob && selectedScene && visibleActiveRemoteJob.scene_id === selectedScene.scene_id
      ? visibleActiveRemoteJob.live_scanned_tile_ids?.at(-1)
      : undefined;
  const sceneSuggestionTiles = useMemo(
    () =>
      sceneTiles
        .filter((tile) => browserSuggestions[tile.tile_id])
        .sort((left, right) => {
          const leftSuggestion = browserSuggestions[left.tile_id];
          const rightSuggestion = browserSuggestions[right.tile_id];
          if (!leftSuggestion || !rightSuggestion) return 0;
          if (leftSuggestion.label !== rightSuggestion.label) {
            return leftSuggestion.label === "crosswalk" ? -1 : 1;
          }
          return rightSuggestion.score - leftSuggestion.score;
        }),
    [browserSuggestions, sceneTiles],
  );
  const scenePositiveSuggestionCount = useMemo(
    () => sceneSuggestionTiles.filter((tile) => browserSuggestions[tile.tile_id]?.label === "crosswalk").length,
    [browserSuggestions, sceneSuggestionTiles],
  );
  const selectedSceneReviewState = sceneReviewStateFor(reviewState, selectedScene?.scene_id);
  const selectedSceneAutopilotMode = Boolean(selectedScene?.autopilot_cell_id);
  const { derivedTileSizeM, orderedScanTiles } = useSceneFootprint({
    scene: selectedScene,
    sceneTiles,
    sceneReviewState: selectedSceneReviewState,
    mode: selectedSceneAutopilotMode ? "scene" : "radius",
  });
  const currentScanJob = useMemo(() => {
    if (!summary || !selectedScene || orderedScanTiles.length === 0) return null;
    return buildScanBatchJob({
      summary,
      scene: selectedScene,
      tileSizeM: derivedTileSizeM,
      scanRadiusTiles: selectedSceneReviewState.scan_radius,
      threshold: REMOTE_SCAN_THRESHOLD,
      promptsText: labeler.defaultPromptText,
      tiles: orderedScanTiles,
    });
  }, [derivedTileSizeM, labeler.defaultPromptText, orderedScanTiles, selectedScene, selectedSceneReviewState.scan_radius, summary]);
  const mapAutopilotPlan = asAutopilotPlan(config?.autopilot) ?? autopilotPreviewPlan;

  useEffect(() => {
    if (labeler.status !== "idle") return;
    const timer = window.setTimeout(() => {
      void labeler.ensureReady().catch(() => {
        // Surface the real error only when the user actively starts a scan.
      });
    }, 180);
    return () => window.clearTimeout(timer);
  }, [labeler]);

  const persistedReviewChecksum = useMemo(() => checksum(reviewState), [reviewState]);

  async function commit(label: string, selected: boolean) {
    if (!selectedTile) return;
    setSaving(true);
    try {
      const payload = await updateTile(runName, exportName, selectedTile.tile_id, { label, selected });
      setSummary(payload.summary);
      setSceneCache((current) => ({ ...current, [payload.scene.scene_id]: payload }));
      const refreshedTile = payload.tiles.find((tile) => tile.tile_id === selectedTile.tile_id) ?? payload.tiles[0];
      setReviewState((current) => ({ ...current, selected_tile_id: refreshedTile?.tile_id }));
      setError(null);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function handleSaveConfig() {
    if (!config) return;
    setSaving(true);
    try {
      const nextConfig = await saveConfig(runName, exportName, config);
      setConfig(nextConfig);
      setDatasets(await listDatasets());
      setError(null);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setSaving(false);
    }
  }

  async function handleCreateDataset() {
    setCreatingDataset(true);
    try {
      const created = await createDataset(newDatasetName, newDatasetSceneId);
      const entries = await listDatasets();
      setDatasets(entries);
      setRunName(created.run_name);
      setExportName(created.export_name);
      setNewDatasetName("");
      setNewDatasetSceneId(DEFAULT_BROWSER_CONFIG.scenes[0]?.scene_id ?? "");
      setIsCreateDatasetDrawerOpen(false);
      setActiveRemoteJobId(null);
      setActiveRemoteJob(null);
      setError(null);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setCreatingDataset(false);
    }
  }

  async function handleCreateAutopilotDataset(input: { targetPositiveCount: number; maxPanels?: number; perimeterBudget?: number }) {
    setCreatingAutopilot(true);
    try {
      const created = await createAutopilotDataset(input.targetPositiveCount, undefined, input.maxPanels, input.perimeterBudget);
      const entries = await listDatasets();
      setDatasets(entries);
      setRunName(created.run_name);
      setExportName(created.export_name);
      setSceneCache({});
      setBrowserSuggestions({});
      setFocusSceneId(undefined);
      setActiveRemoteJobId(null);
      setActiveRemoteJob(null);
      setError(null);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setCreatingAutopilot(false);
    }
  }

  const handleZoomChange = useCallback((zoom: number) => {
    setReviewState((current) => (current.map_zoom === zoom ? current : { ...current, map_zoom: zoom }));
  }, []);

  const importScanResult = useCallback(
    (result: ScanBatchResult) => {
      const sceneId = result.job.scene.scene_id;
      const firstCrosswalkTileId = result.tiles.find((tile) => tile.label === "crosswalk")?.tile_id;
      const firstTileId = result.tiles[0]?.tile_id;

      setBrowserSuggestions((current) => ({
        ...current,
        ...result.results,
      }));
      setReviewState((current) => ({
        ...current,
        selected_scene_id: sceneId,
        selected_tile_id:
          current.selected_scene_id === sceneId && current.selected_tile_id
            ? current.selected_tile_id
            : firstCrosswalkTileId ?? firstTileId ?? current.selected_tile_id,
        scenes: {
          ...current.scenes,
          [sceneId]: {
            ...sceneReviewStateFor(current, sceneId),
            scanned_tile_ids: result.tiles.map((tile) => tile.tile_id),
          },
        },
      }));
      setFocusSceneId(sceneId);
    },
    [setBrowserSuggestions, setReviewState],
  );

  useEffect(() => {
    const job = activeRemoteJob;
    if (!job?.scene_id) return;
    if (!scenes.some((scene) => scene.scene_id === job.scene_id)) return;
    const liveScannedTileIds = job.live_scanned_tile_ids ?? [];
    const liveResults = job.live_results ?? {};
    if (!liveScannedTileIds.length && !Object.keys(liveResults).length) {
      return;
    }
    const sceneId = job.scene_id;
    setBrowserSuggestions((current) => ({
      ...current,
      ...liveResults,
    }));
    setReviewState((current) => ({
      ...current,
      scenes: {
        ...current.scenes,
        [sceneId]: {
          ...sceneReviewStateFor(current, sceneId),
          scanned_tile_ids:
            liveScannedTileIds.length > 0
              ? liveScannedTileIds
              : sceneReviewStateFor(current, sceneId).scanned_tile_ids,
        },
      },
    }));
    setFocusSceneId((current) => current ?? sceneId);
  }, [activeRemoteJob, scenes]);

  const jumpToSceneSuggestion = useCallback(
    (mode: "next" | "positive") => {
      if (!sceneSuggestionTiles.length) return;
      const pool =
        mode === "positive"
          ? sceneSuggestionTiles.filter((tile) => browserSuggestions[tile.tile_id]?.label === "crosswalk")
          : sceneSuggestionTiles;
      if (!pool.length) return;
      const currentIndex = pool.findIndex((tile) => tile.tile_id === selectedTile?.tile_id);
      const nextTile = pool[(currentIndex + 1 + pool.length) % pool.length] ?? pool[0];
      if (!nextTile) return;
      setReviewState((current) => ({
        ...current,
        selected_scene_id: nextTile.scene_id,
        selected_tile_id: nextTile.tile_id,
      }));
    },
    [browserSuggestions, sceneSuggestionTiles, selectedTile?.tile_id],
  );

  const handleSceneReviewStateChange = useCallback(
    (nextSceneState: ReturnType<typeof sceneReviewStateFor>) => {
      if (!selectedScene?.scene_id) return;
      setReviewState((current) => {
        const previous = sceneReviewStateFor(current, selectedScene.scene_id);
        if (sameSceneReviewState(previous, nextSceneState)) {
          return current;
        }
        return {
          ...current,
          scenes: {
            ...current.scenes,
            [selectedScene.scene_id]: nextSceneState,
          },
        };
      });
    },
    [selectedScene?.scene_id],
  );

  const handleSelectScene = useCallback(
    (sceneId: string) => {
      setFocusSceneId(sceneId);
      const payload = sceneCache[sceneId];
      setReviewState((current) => {
        const nextTileId = payload?.tiles[0]?.tile_id;
        if (current.selected_scene_id === sceneId && current.selected_tile_id === nextTileId) {
          return current;
        }
        return {
          ...current,
          selected_scene_id: sceneId,
          selected_tile_id: nextTileId,
        };
      });
    },
    [sceneCache],
  );

  const handleSelectTile = useCallback((tile: ScenePayload["tiles"][number]) => {
    setReviewState((current) =>
      current.selected_scene_id === tile.scene_id && current.selected_tile_id === tile.tile_id
        ? current
        : {
            ...current,
            selected_scene_id: tile.scene_id,
            selected_tile_id: tile.tile_id,
          },
    );
  }, []);

  const handleEnsureSceneImages = useCallback(async () => {
    if (!selectedScene) {
      return new Map<string, string>();
    }
    const imageUrls = await ensureSceneImages(runName, exportName, selectedScene.scene_id);
    setSceneCache((current) => {
      const existing = current[selectedScene.scene_id];
      if (!existing) return current;
      return {
        ...current,
        [selectedScene.scene_id]: {
          ...existing,
          tiles: existing.tiles.map((tile) => ({
            ...tile,
            image_path: imageUrls.get(tile.tile_id) ?? tile.image_path,
          })),
        },
      };
    });
    setSceneImagesReady((current) => ({ ...current, [selectedScene.scene_id]: true }));
    return imageUrls;
  }, [exportName, runName, selectedScene]);

  const handleEnsureSceneTiles = useCallback(async () => {
    if (!selectedScene) return;
    if (sceneCache[selectedScene.scene_id]) return;
    const payload = await loadScene(runName, exportName, selectedScene.scene_id);
    setSceneCache((current) => ({ ...current, [payload.scene.scene_id]: payload }));
    setSummary(payload.summary);
    setSceneImagesReady((current) => ({ ...current, [payload.scene.scene_id]: true }));
    setReviewState((current) => {
      const tileStillExists = current.selected_tile_id && payload.tiles.some((tile) => tile.tile_id === current.selected_tile_id);
      return {
        ...current,
        selected_tile_id: tileStillExists ? current.selected_tile_id : payload.tiles[0]?.tile_id,
      };
    });
  }, [exportName, runName, sceneCache, selectedScene]);

  useEffect(() => {
    if (error?.includes("Unknown scene")) {
      setError(null);
      return;
    }
    if (!error) {
      setIsErrorDrawerOpen(false);
    }
  }, [error]);

  const errorSummary = useMemo(() => summarizeErrorMessage(error, "Something went wrong."), [error]);
  const createDatasetSceneOptions = useMemo(() => DEFAULT_BROWSER_CONFIG.scenes.map((scene) => ({ scene_id: scene.scene_id, city: scene.city, split: scene.split })), []);

  return (
    <div className="h-dvh overflow-hidden bg-content2 text-foreground">
      <div className="relative h-full w-full">
        <SceneMap
          summary={summary}
          selectedSceneId={selectedScene?.scene_id}
          focusSceneId={focusSceneId}
          sceneTiles={sceneTiles}
          selectedTileId={selectedTile?.tile_id}
          mapZoom={mapZoom}
          reviewState={selectedSceneReviewState}
          mapDebug={mapDebug}
          validationCase={validationCase}
          validationRunId={validationRunId}
          persistedReviewChecksum={persistedReviewChecksum}
          persistedReviewState={reviewState}
          basemap={basemap}
          autopilotPlan={mapAutopilotPlan}
          isMobileLayout={isMobileLayout}
          onMobileScanPanelChange={setMobileScanPanel}
          labeler={labeler}
          browserSuggestions={browserSuggestions}
          remoteActiveTileId={remoteActiveTileId}
          remoteScanActive={remoteScanActive}
          sceneTilesReady={sceneTiles.length > 0}
          sceneImagesReady={selectedScene ? sceneImagesReady[selectedScene.scene_id] !== false : false}
          onSuggestionsChange={setBrowserSuggestions}
          onError={setError}
          onEnsureSceneTiles={handleEnsureSceneTiles}
          onEnsureSceneImages={handleEnsureSceneImages}
          onZoomChange={handleZoomChange}
          onReviewStateChange={handleSceneReviewStateChange}
          onBasemapChange={setBasemap}
          onSelectScene={handleSelectScene}
          onSelectTile={handleSelectTile}
        />

        <div className="pointer-events-none absolute inset-0 z-[1000]">
          <AppOverlayPanels
            activeRemoteJobId={activeRemoteJobId}
            basemap={basemap}
            browserSuggestion={selectedSuggestion}
            config={config}
            creatingAutopilot={creatingAutopilot}
            datasets={datasets}
            errorSummary={error ? errorSummary : null}
            exportName={exportName}
            isMobileLayout={isMobileLayout}
            mobileScanPanel={mobileScanPanel}
            remoteSnapshot={remoteSnapshot}
            runName={runName}
            scanJob={currentScanJob}
            scene={selectedScene}
            sceneSuggestionCount={sceneSuggestionTiles.length}
            selectedTile={selectedTile}
            serverJob={visibleActiveRemoteJob}
            summary={summary}
            saving={saving}
            onActiveRemoteJobChange={setActiveRemoteJobId}
            onActiveRemoteJobResolved={setActiveRemoteJob}
            onApplySuggestion={
              selectedSuggestion
                ? () => {
                    void commit(selectedSuggestion.label, selectedSuggestion.selected);
                  }
                : undefined
            }
            onBasemapChange={setBasemap}
            onCommit={commit}
            onCreateAutopilot={(input) => {
              void handleCreateAutopilotDataset(input);
            }}
            onCreateDataset={() => {
              setNewDatasetSceneId(selectedSceneId ?? DEFAULT_BROWSER_CONFIG.scenes[0]?.scene_id ?? "");
              setIsCreateDatasetDrawerOpen(true);
            }}
            onDatasetSelect={(value) => {
              const [nextRun, nextExport] = value.split("::");
              setRunName(nextRun);
              setExportName(nextExport);
              setActiveRemoteJobId(null);
              setActiveRemoteJob(null);
            }}
            onError={setError}
            onJumpToNextSuggestion={() => jumpToSceneSuggestion("next")}
            onJumpToNextPositive={() => jumpToSceneSuggestion("positive")}
            onOpenErrorDetails={() => setIsErrorDrawerOpen(true)}
            onOpenTerminal={() => setIsTerminalDrawerOpen(true)}
            onPreviewPlanChange={setAutopilotPreviewPlan}
            onResultImported={importScanResult}
          />
        </div>
      </div>

      <AppDrawers
        activeRemoteJob={activeRemoteJob}
        creatingDataset={creatingDataset}
        error={error}
        isCreateDatasetDrawerOpen={isCreateDatasetDrawerOpen}
        isErrorDrawerOpen={isErrorDrawerOpen}
        isMobileLayout={isMobileLayout}
        isTerminalDrawerOpen={isTerminalDrawerOpen}
        newDatasetName={newDatasetName}
        newDatasetSceneId={newDatasetSceneId}
        sceneOptions={createDatasetSceneOptions}
        onCreateDataset={() => {
          void handleCreateDataset();
        }}
        onCreateDatasetOpenChange={setIsCreateDatasetDrawerOpen}
        onErrorOpenChange={setIsErrorDrawerOpen}
        onNameChange={setNewDatasetName}
        onSceneChange={setNewDatasetSceneId}
        onTerminalOpenChange={setIsTerminalDrawerOpen}
      />
    </div>
  );
}
