import { useCallback, useEffect, useMemo, useState } from "react";
import { ensureSceneImages, listDatasets, loadConfig, loadDatasetMeta, loadReviewState, loadScene, saveConfig, saveReviewState, updateTile } from "./api";
import { exportBrowserDatasetZip } from "./browser-export";
import { checksum, sameSceneReviewState } from "./map-validation";
import { normalizeReviewState, sceneReviewStateFor } from "./review-state";
import type { BrowserLabelSuggestion, DatasetListEntry, DatasetSummary, MapBasemap, RealDatasetConfig, ReviewState, ScenePayload } from "./types";
import { sortScenes } from "./utils";
import { BrowserDatasetBuilder } from "./components/BrowserDatasetBuilder";
import { ConfigEditor } from "./components/ConfigEditor";
import { Inspector } from "./components/Inspector";
import { SceneMap } from "./components/SceneMap";
import { useBrowserCrosswalkLabeler } from "./hooks/useBrowserCrosswalkLabeler";

const FALLBACK_RUN = "real-v1";
const FALLBACK_EXPORT = "real-balanced-256";
const DEFAULT_SCAN_BACKEND_URL = "http://127.0.0.1:8000";
export default function App() {
  const [datasets, setDatasets] = useState<DatasetListEntry[]>([]);
  const [runName, setRunName] = useState(FALLBACK_RUN);
  const [exportName, setExportName] = useState(FALLBACK_EXPORT);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [sceneCache, setSceneCache] = useState<Record<string, ScenePayload>>({});
  const [focusSceneId, setFocusSceneId] = useState<string | undefined>();
  const [reviewState, setReviewState] = useState<ReviewState>(normalizeReviewState());
  const [reviewStateReady, setReviewStateReady] = useState(false);
  const [draftLabel, setDraftLabel] = useState("crosswalk");
  const [draftSelected, setDraftSelected] = useState(true);
  const [saving, setSaving] = useState(false);
  const [config, setConfig] = useState<RealDatasetConfig | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [scanBackendUrl, setScanBackendUrl] = useState(() => window.localStorage.getItem("crosswalk.scan-backend-url") ?? DEFAULT_SCAN_BACKEND_URL);
  const [basemap, setBasemap] = useState<MapBasemap>(() => {
    const stored = window.localStorage.getItem("crosswalk-review:basemap");
    return stored === "swisstopo" ? "swisstopo" : "osm";
  });
  const [browserSuggestions, setBrowserSuggestions] = useState<Record<string, BrowserLabelSuggestion>>({});
  const [sceneImagesReady, setSceneImagesReady] = useState<Record<string, boolean>>({});
  const labeler = useBrowserCrosswalkLabeler({
    backendUrl: scanBackendUrl,
    scenes: config?.scenes ?? [],
    tileSizeM: config?.tile_size_m ?? 25,
  });
  const [sideTab, setSideTab] = useState<"review" | "builder" | "config">("review");
  const [exporting, setExporting] = useState(false);
  const [lastExportLabel, setLastExportLabel] = useState<string | null>(null);
  const urlParams = useMemo(() => new URLSearchParams(window.location.search), []);
  const mapDebug = urlParams.get("mapDebug") === "1";
  const validationCase = urlParams.get("validationCase");
  const validationRunId = urlParams.get("validationRunId") ?? "manual";

  useEffect(() => {
    Promise.all([listDatasets(), loadConfig()])
      .then(([entries, loadedConfig]) => {
        setDatasets(entries);
        setConfig(loadedConfig);
        const preferred =
          entries.find(
            (entry) => entry.run_name === loadedConfig.run_name && entry.export_name === loadedConfig.export_name,
          ) ??
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
        if (!canceled) setError(String(reason));
      });
    return () => {
      canceled = true;
    };
  }, [runName, exportName]);

  useEffect(() => {
    window.localStorage.setItem("crosswalk-review:basemap", basemap);
  }, [basemap]);

  useEffect(() => {
    window.localStorage.setItem("crosswalk.scan-backend-url", scanBackendUrl);
  }, [scanBackendUrl]);

  useEffect(() => {
    const selectedSceneId = reviewState.selected_scene_id;
    if (!selectedSceneId || sceneCache[selectedSceneId]) return;
    let canceled = false;
    loadScene(runName, exportName, selectedSceneId)
      .then((payload) => {
        if (canceled) return;
        setSceneCache((current) => ({ ...current, [payload.scene.scene_id]: payload }));
        setSceneImagesReady((current) => ({ ...current, [payload.scene.scene_id]: false }));
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
        if (!canceled) setError(String(reason));
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
        if (!canceled) setError(String(reason));
      });
    return () => {
      canceled = true;
    };
  }, [exportName, reviewState.selected_scene_id, runName, sceneCache]);

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

  useEffect(() => {
    if (!selectedTile) return;
    setDraftLabel(selectedTile.label);
    setDraftSelected(selectedTile.selected);
  }, [selectedTile?.tile_id]);

  useEffect(() => {
    if (labeler.status !== "idle") return;
    const timer = window.setTimeout(() => {
      void labeler.ensureReady().catch(() => {
        // Surface the real error only when the user actively starts a scan.
      });
    }, 180);
    return () => window.clearTimeout(timer);
  }, [labeler]);

  const stats = summary ?? {
    selected_crosswalk: 0,
    selected_no_crosswalk: 0,
    dropped_tiles: 0,
    selected_tiles: 0,
  };
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
      setDraftLabel(refreshedTile?.label ?? label);
      setDraftSelected(refreshedTile?.selected ?? selected);
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
      const nextConfig = await saveConfig(config);
      setConfig(nextConfig);
      setError(null);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setSaving(false);
    }
  }

  const handleZoomChange = useCallback((zoom: number) => {
    setReviewState((current) => (current.map_zoom === zoom ? current : { ...current, map_zoom: zoom }));
  }, []);

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
    setSceneImagesReady((current) => ({ ...current, [payload.scene.scene_id]: false }));
    setReviewState((current) => {
      const tileStillExists = current.selected_tile_id && payload.tiles.some((tile) => tile.tile_id === current.selected_tile_id);
      return {
        ...current,
        selected_tile_id: tileStillExists ? current.selected_tile_id : payload.tiles[0]?.tile_id,
      };
    });
  }, [exportName, runName, sceneCache, selectedScene]);

  return (
    <div className="shell">
      <header className="topbar topbar-compact">
        <div className="topbar-main">
          <p className="eyebrow">Crosswalk review</p>
          <h1 className="title-compact">Review Desk</h1>
        </div>
        <div className="topbar-controls">
          <label className="dataset-control">
            <span className="eyebrow">Dataset</span>
            <div className="topbar-dataset-row">
              <select
                value={`${runName}::${exportName}`}
                onChange={(event) => {
                  const [nextRun, nextExport] = event.target.value.split("::");
                  setRunName(nextRun);
                  setExportName(nextExport);
                }}
              >
                {datasets.map((entry) => (
                  <option key={`${entry.run_name}::${entry.export_name}`} value={`${entry.run_name}::${entry.export_name}`}>
                    {entry.run_name}/{entry.export_name} ({entry.tile_count})
                  </option>
                ))}
              </select>
              <button
                className="primary"
                disabled={exporting}
                onClick={async () => {
                  setExporting(true);
                  try {
                    const result = await exportBrowserDatasetZip();
                    setLastExportLabel(`${result.selectedCount} tiles exported`);
                    setError(null);
                  } catch (reason) {
                    setError(String(reason));
                  } finally {
                    setExporting(false);
                  }
                }}
                type="button"
              >
                {exporting ? "Exporting" : "Export ZIP"}
              </button>
            </div>
          </label>
          {lastExportLabel ? <p className="eyebrow topbar-export-note">{lastExportLabel}</p> : null}
          <div className="metric-grid">
            <div className="metric-card">
              <span className="metric-label">Crosswalk</span>
              <strong>{stats.selected_crosswalk}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">No Crosswalk</span>
              <strong>{stats.selected_no_crosswalk}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">Dropped</span>
              <strong>{stats.dropped_tiles}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">Selected</span>
              <strong>{stats.selected_tiles}</strong>
            </div>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="canvas-stack">
          {error ? <div className="error">{error}</div> : null}

          <SceneMap
            summary={summary}
            selectedSceneId={selectedScene?.scene_id}
            focusSceneId={focusSceneId}
            sceneTiles={sceneTiles}
            selectedTileId={selectedTile?.tile_id}
            mapZoom={mapZoom}
            reviewState={sceneReviewStateFor(reviewState, selectedScene?.scene_id)}
            mapDebug={mapDebug}
            validationCase={validationCase}
            validationRunId={validationRunId}
            persistedReviewChecksum={persistedReviewChecksum}
            persistedReviewState={reviewState}
            basemap={basemap}
            labeler={labeler}
            browserSuggestions={browserSuggestions}
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
        </section>

        <section className="side-stack side-panel panel">
          <div className="side-panel-header">
            <div>
              <p className="eyebrow">Controls</p>
              <h2>{sideTab === "review" ? "Tile review" : sideTab === "builder" ? "Browser builder" : "Dataset config"}</h2>
            </div>
            <div className="pill-row">
              <button className={`pill pill-button ${sideTab === "review" ? "active" : ""}`} onClick={() => setSideTab("review")} type="button">
                Review
              </button>
              <button className={`pill pill-button ${sideTab === "builder" ? "active" : ""}`} onClick={() => setSideTab("builder")} type="button">
                Builder
              </button>
              <button className={`pill pill-button ${sideTab === "config" ? "active" : ""}`} onClick={() => setSideTab("config")} type="button">
                Config
              </button>
            </div>
          </div>

          {sideTab === "review" ? (
            <Inspector
              scene={selectedScene}
              tile={selectedTile}
              browserSuggestion={selectedSuggestion}
              draftLabel={draftLabel}
              draftSelected={draftSelected}
              onDraftLabel={setDraftLabel}
              onDraftSelected={setDraftSelected}
              onCommit={commit}
              onReset={() => {
                if (!selectedTile) return;
                setDraftLabel(selectedTile.label);
                setDraftSelected(selectedTile.selected);
              }}
              saving={saving}
            />
          ) : null}
          {sideTab === "builder" ? (
            <BrowserDatasetBuilder
              exportName={exportName}
              labeler={labeler}
              backendUrl={scanBackendUrl}
              onBackendUrlChange={setScanBackendUrl}
              onApplied={(payload) => {
                if (!selectedScene?.scene_id) return;
                setSummary(payload.summary);
                setSceneCache((current) => ({
                  ...current,
                  [selectedScene.scene_id]: {
                    ...payload,
                  },
                }));
              }}
              onError={setError}
              onSuggestionsChange={setBrowserSuggestions}
              runName={runName}
              scene={selectedScene}
              sceneTiles={sceneTiles}
              suggestions={browserSuggestions}
            />
          ) : null}
          {sideTab === "config" ? (
            <ConfigEditor
              config={config}
              backendUrl={scanBackendUrl}
              backendHealth={labeler.health}
              onBackendUrlChange={setScanBackendUrl}
              onCheckBackend={() => void labeler.ensureReady()}
              onChange={setConfig}
              onSave={handleSaveConfig}
              saving={saving}
            />
          ) : null}
        </section>
      </main>

      <footer className="statusbar">
        {summary
          ? `${summary.run_name}/${summary.export_name} · ${summary.total_tiles} total tiles · ${
              selectedScene ? `${sceneTiles.length} loaded in ${selectedScene.city}` : "pick a city on the map"
            }`
          : "Loading dataset…"}
      </footer>
    </div>
  );
}
