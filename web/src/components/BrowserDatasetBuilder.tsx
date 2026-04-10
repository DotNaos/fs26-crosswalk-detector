import type { Dispatch, SetStateAction } from "react";
import { useMemo, useState } from "react";
import { updateTiles } from "../api";
import type { BrowserCrosswalkLabelerHandle } from "../hooks/useBrowserCrosswalkLabeler";
import type { BrowserLabelSuggestion, DatasetScene, DatasetTile, ScenePayload } from "../types";
import { formatProbability, sceneLabel } from "../utils";

type BrowserDatasetBuilderProps = {
  runName: string;
  exportName: string;
  backendUrl: string;
  scene?: DatasetScene;
  sceneTiles: DatasetTile[];
  labeler: BrowserCrosswalkLabelerHandle;
  suggestions: Record<string, BrowserLabelSuggestion>;
  onSuggestionsChange: Dispatch<SetStateAction<Record<string, BrowserLabelSuggestion>>>;
  onApplied: (payload: ScenePayload) => void;
  onBackendUrlChange: (next: string) => void;
  onError: (message: string | null) => void;
};

export function BrowserDatasetBuilder({
  runName,
  exportName,
  backendUrl,
  scene,
  sceneTiles,
  labeler,
  suggestions,
  onSuggestionsChange,
  onApplied,
  onBackendUrlChange,
  onError,
}: BrowserDatasetBuilderProps) {
  const [promptText, setPromptText] = useState(labeler.defaultPromptText);
  const [threshold, setThreshold] = useState(0.32);
  const [saving, setSaving] = useState(false);

  const sceneSuggestions = useMemo(
    () => sceneTiles.map((tile) => suggestions[tile.tile_id]).filter((entry): entry is BrowserLabelSuggestion => Boolean(entry)),
    [sceneTiles, suggestions],
  );
  const suggestionStats = useMemo(
    () => ({
      total: sceneSuggestions.length,
      crosswalk: sceneSuggestions.filter((entry) => entry.label === "crosswalk").length,
      noCrosswalk: sceneSuggestions.filter((entry) => entry.label === "no_crosswalk").length,
      strongest: sceneSuggestions.slice().sort((a, b) => b.score - a.score)[0],
    }),
    [sceneSuggestions],
  );

  async function handleRunScene() {
    if (!sceneTiles.length) return;
    onError(null);
    try {
      const next = await labeler.runSceneLabeling(sceneTiles, promptText, threshold);
      onSuggestionsChange({
        ...suggestions,
        ...next,
      });
    } catch (reason) {
      onError(String(reason));
    }
  }

  async function handleApplyScene() {
    if (!scene?.scene_id || sceneSuggestions.length === 0) return;
    setSaving(true);
    onError(null);
    try {
      const sceneUpdates = sceneSuggestions.map((suggestion) => ({
        tile_id: suggestion.tile_id,
        label: suggestion.label,
        selected: suggestion.selected,
        combined_probability: suggestion.score,
        predicted_label: suggestion.label,
        review_source: suggestion.review_source,
      }));
      const payload = await updateTiles(runName, exportName, scene.scene_id, sceneUpdates);
      onApplied(payload);
      const nextSuggestions = { ...suggestions };
      for (const tile of payload.tiles) {
        delete nextSuggestions[tile.tile_id];
      }
      onSuggestionsChange(nextSuggestions);
    } catch (reason) {
      onError(String(reason));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel builder-panel">
      <div className="builder-header">
        <div>
          <p className="eyebrow">Server dataset builder</p>
          <h2>{scene ? sceneLabel(scene) : "Select a scene"}</h2>
        </div>
        <span className={`pill ${labeler.status === "error" ? "red" : labeler.status === "running" ? "amber" : "green"}`}>
          {labeler.status}
        </span>
      </div>

      <div className="builder-grid">
        <label className="config-span-2">
          Backend URL
          <input value={backendUrl} onChange={(event) => onBackendUrlChange(event.target.value)} placeholder="http://127.0.0.1:8000" />
        </label>
      </div>

      <p className="builder-copy">
        Runs the scan on a Python service on your laptop or in Colab, then writes the accepted labels back into the local dataset state.
      </p>

      <div className="builder-grid">
        <label>
          Prompt list
          <input value={promptText} onChange={(event) => setPromptText(event.target.value)} placeholder="crosswalk, zebra crossing" />
        </label>
        <label>
          Crosswalk threshold
          <input
            type="number"
            min={0.2}
            max={0.98}
            step={0.01}
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value) || threshold)}
          />
        </label>
      </div>

      <div className="builder-actions">
        <button className="ghost" disabled={labeler.status === "connecting" || labeler.status === "running"} onClick={() => void labeler.ensureReady()} type="button">
          Check Backend
        </button>
        <button className="primary positive" disabled={!sceneTiles.length || labeler.status === "running"} onClick={() => void handleRunScene()} type="button">
          Scan Loaded Scene
        </button>
        <button className="primary" disabled={sceneSuggestions.length === 0 || saving} onClick={() => void handleApplyScene()} type="button">
          Apply Scene Labels
        </button>
        <button
          className="ghost"
          disabled={sceneSuggestions.length === 0}
          onClick={() => {
            const nextSuggestions = { ...suggestions };
            for (const tile of sceneTiles) {
              delete nextSuggestions[tile.tile_id];
            }
            onSuggestionsChange(nextSuggestions);
          }}
          type="button"
        >
          Clear Scene Suggestions
        </button>
      </div>

      <div className="pill-row tight">
        <span className="pill muted">
          {sceneTiles.length} loaded tile{sceneTiles.length === 1 ? "" : "s"}
        </span>
        <span className="pill green">Suggested crosswalk {suggestionStats.crosswalk}</span>
        <span className="pill red">Suggested no_crosswalk {suggestionStats.noCrosswalk}</span>
        {labeler.progress.total > 0 ? (
          <span className="pill amber">
            Running {labeler.progress.done}/{labeler.progress.total}
          </span>
        ) : null}
      </div>

      <dl className="builder-meta">
        <div>
          <dt>Latest batch</dt>
          <dd>{labeler.lastBatchCount}</dd>
        </div>
        <div>
          <dt>Current tile</dt>
          <dd>{labeler.progress.currentTileId ?? "n/a"}</dd>
        </div>
        <div>
          <dt>Top suggestion</dt>
          <dd>
            {suggestionStats.strongest
              ? `${suggestionStats.strongest.label} · ${formatProbability(suggestionStats.strongest.score)}`
              : "n/a"}
          </dd>
        </div>
        <div>
          <dt>Top prompt</dt>
          <dd>{suggestionStats.strongest?.prompt ?? "n/a"}</dd>
        </div>
      </dl>

      {labeler.error ? <p className="builder-error">{labeler.error}</p> : null}
    </section>
  );
}
