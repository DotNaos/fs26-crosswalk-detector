import type { BrowserLabelSuggestion, DatasetScene, DatasetTile } from "../types";
import { formatProbability, sceneLabel } from "../utils";

type InspectorProps = {
  scene?: DatasetScene;
  tile?: DatasetTile;
  browserSuggestion?: BrowserLabelSuggestion;
  draftLabel: string;
  draftSelected: boolean;
  onDraftLabel: (label: string) => void;
  onDraftSelected: (selected: boolean) => void;
  onCommit: (label: string, selected: boolean) => void;
  onReset: () => void;
  saving: boolean;
};

export function Inspector({
  scene,
  tile,
  browserSuggestion,
  draftLabel,
  draftSelected,
  onDraftLabel,
  onDraftSelected,
  onCommit,
  onReset,
  saving,
}: InspectorProps) {
  if (!tile) {
    return (
      <aside className="inspector panel">
        <p className="eyebrow">Inspector</p>
        <h2>Awaiting selection</h2>
        <p className="empty-copy">Select a tile on the map to inspect and correct its label.</p>
      </aside>
    );
  }

  return (
    <aside className="inspector panel">
      <div className="inspector-header">
        <div>
          <p className="inspector-kicker">{scene ? sceneLabel(scene) : tile.scene_id}</p>
          <h2>{tile.relative_path}</h2>
        </div>
        <button className="ghost" onClick={onReset} type="button">
          Reset
        </button>
      </div>

      <div className="inspector-preview">
        {tile.image_path ? (
          <img src={tile.image_path} alt={tile.relative_path} loading="lazy" decoding="async" />
        ) : (
          <div className="inspector-preview-empty">
            <p className="eyebrow">Tile image pending</p>
            <p>The scene tiles are still loading for this selection.</p>
          </div>
        )}
      </div>

      <div className="pill-row tight">
        <span className="pill">Label {tile.label}</span>
        <span className="pill">Predicted {tile.predicted_label}</span>
        <span className="pill muted">{tile.selected ? "Selected" : "Dropped"}</span>
        {browserSuggestion ? <span className="pill amber">Browser {browserSuggestion.label}</span> : null}
      </div>

      <dl className="inspector-meta">
        <div>
          <dt>Tile</dt>
          <dd>{tile.tile_id}</dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>{tile.status}</dd>
        </div>
        <div>
          <dt>Review source</dt>
          <dd>{tile.review_source}</dd>
        </div>
        <div>
          <dt>Combined probability</dt>
          <dd>{formatProbability(tile.combined_probability)}</dd>
        </div>
        <div>
          <dt>Browser score</dt>
          <dd>{browserSuggestion ? formatProbability(browserSuggestion.score) : "n/a"}</dd>
        </div>
      </dl>

      <div className="review-form">
        <label>
          Label
          <select value={draftLabel} onChange={(event) => onDraftLabel(event.target.value)}>
            <option value="unknown">unknown</option>
            <option value="crosswalk">crosswalk</option>
            <option value="no_crosswalk">no_crosswalk</option>
          </select>
        </label>
        <label className="checkbox-row">
          <input checked={draftSelected} onChange={(event) => onDraftSelected(event.target.checked)} type="checkbox" />
          Included in export
        </label>
      </div>

      <div className="inspector-actions">
        <button className="primary positive" disabled={saving} onClick={() => onCommit("crosswalk", true)} type="button">
          Mark Crosswalk
        </button>
        <button className="primary negative" disabled={saving} onClick={() => onCommit("no_crosswalk", true)} type="button">
          Mark No Crosswalk
        </button>
        <button className="ghost" disabled={saving} onClick={() => onCommit(draftLabel, false)} type="button">
          Drop
        </button>
        <button className="primary" disabled={saving} onClick={() => onCommit(draftLabel, draftSelected)} type="button">
          Save Changes
        </button>
      </div>
    </aside>
  );
}
