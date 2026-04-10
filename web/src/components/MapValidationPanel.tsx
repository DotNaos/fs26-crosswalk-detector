import type { MapValidationSnapshot } from "../map-validation";

type MapValidationPanelProps = {
  snapshot: MapValidationSnapshot | null;
};

function formatPoint(value: { x: number; y: number } | null) {
  if (!value) return "n/a";
  return `${value.x.toFixed(1)}, ${value.y.toFixed(1)}`;
}

function formatBounds(
  value:
    | {
        minX: number;
        minY: number;
        maxX: number;
        maxY: number;
      }
    | null,
) {
  if (!value) return "n/a";
  return `${value.minX.toFixed(1)},${value.minY.toFixed(1)} → ${value.maxX.toFixed(1)},${value.maxY.toFixed(1)}`;
}

export function MapValidationPanel({ snapshot }: MapValidationPanelProps) {
  if (!snapshot) {
    return (
      <aside className="map-validation-panel">
        <p className="eyebrow">Map validation</p>
        <strong>Diagnostics booting…</strong>
      </aside>
    );
  }

  return (
    <aside className="map-validation-panel">
      <div className="map-validation-header">
        <div>
          <p className="eyebrow">Map validation</p>
          <strong>
            {snapshot.validationCase ?? "manual"} · {snapshot.status}
          </strong>
        </div>
        <span className={`pill ${snapshot.verdicts.every((verdict) => verdict.pass) ? "green" : "red"}`}>
          {snapshot.verdicts.filter((verdict) => verdict.pass).length}/{snapshot.verdicts.length || 1} pass
        </span>
      </div>

      <dl className="map-validation-grid">
        <div>
          <dt>Scene</dt>
          <dd>{snapshot.sceneId ?? "n/a"}</dd>
        </div>
        <div>
          <dt>Tile</dt>
          <dd>{snapshot.selectedTileId ?? "n/a"}</dd>
        </div>
        <div>
          <dt>Zoom</dt>
          <dd>{snapshot.camera.zoom.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Phase</dt>
          <dd>{snapshot.interactionPhase}</dd>
        </div>
        <div>
          <dt>Anchor</dt>
          <dd>{formatPoint(snapshot.zoomAnchor)}</dd>
        </div>
        <div>
          <dt>Scan</dt>
          <dd>
            {snapshot.scanIndex}/{snapshot.orderedScanTileIds.length}
          </dd>
        </div>
        <div>
          <dt>Viewport</dt>
          <dd>
            {snapshot.camera.viewport.width}×{snapshot.camera.viewport.height}
          </dd>
        </div>
        <div>
          <dt>Page scale</dt>
          <dd>{snapshot.camera.pageScale.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Selected bounds</dt>
          <dd>{formatBounds(snapshot.selectedTileBounds)}</dd>
        </div>
        <div>
          <dt>Active bounds</dt>
          <dd>{formatBounds(snapshot.activeTileBounds)}</dd>
        </div>
      </dl>

      <div className="map-validation-list">
        <p className="eyebrow">Invariants</p>
        {snapshot.invariants.map((invariant) => (
          <div className={`map-validation-item ${invariant.pass ? "pass" : "fail"}`} key={invariant.id}>
            <strong>{invariant.id}</strong>
            <span>{invariant.detail}</span>
          </div>
        ))}
      </div>

      <div className="map-validation-list">
        <p className="eyebrow">Verdicts</p>
        {snapshot.verdicts.map((verdict) => (
          <div className={`map-validation-item ${verdict.pass ? "pass" : "fail"}`} key={verdict.name}>
            <strong>{verdict.name}</strong>
            <span>{verdict.detail}</span>
          </div>
        ))}
      </div>

      <div className="map-validation-list">
        <p className="eyebrow">Counters</p>
        <div className="map-validation-item">
          <strong>Renders</strong>
          <span>{snapshot.counters.componentRenders}</span>
        </div>
        <div className="map-validation-item">
          <strong>Overlay</strong>
          <span>{snapshot.counters.overlayRecomputes}</span>
        </div>
        <div className="map-validation-item">
          <strong>Camera</strong>
          <span>{snapshot.counters.cameraUpdates}</span>
        </div>
        <div className="map-validation-item">
          <strong>Console</strong>
          <span>{snapshot.counters.consoleErrors}</span>
        </div>
      </div>
    </aside>
  );
}
