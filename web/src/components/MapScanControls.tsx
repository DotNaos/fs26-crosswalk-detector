import type { MapBasemap } from "../types";

type MapScanControlsProps = {
  sceneLabel: string;
  mapZoom: number;
  showGrid: boolean;
  sceneTilesReady: boolean;
  totalTilesInCircle: number;
  basemap: MapBasemap;
  scanRadius: number;
  scanDelay: number;
  scanRunning: boolean;
  scanQueued: boolean;
  scanPreparing: boolean;
  sceneImagesReady: boolean;
  scannedCount: number;
  crosswalkCount: number;
  noCrosswalkCount: number;
  liveScanStep: number;
  activeSummary: string | null;
  onBasemapChange: (next: MapBasemap) => void;
  onScanRadiusChange: (next: number) => void;
  onScanDelayChange: (next: number) => void;
  onStartScan: () => void;
  onPauseScan: () => void;
  onResetScan: () => void;
  onExportBatchJob: () => void;
  onImportBatchResult: () => void;
};

export function MapScanControls({
  sceneLabel,
  mapZoom,
  showGrid,
  sceneTilesReady,
  totalTilesInCircle,
  basemap,
  scanRadius,
  scanDelay,
  scanRunning,
  scanQueued,
  scanPreparing,
  sceneImagesReady,
  scannedCount,
  crosswalkCount,
  noCrosswalkCount,
  liveScanStep,
  activeSummary,
  onBasemapChange,
  onScanRadiusChange,
  onScanDelayChange,
  onStartScan,
  onPauseScan,
  onResetScan,
  onExportBatchJob,
  onImportBatchResult,
}: MapScanControlsProps) {
  return (
    <>
      <div className="map-header">
        <div>
          <p className="eyebrow">Map canvas</p>
          <h2>{sceneLabel}</h2>
        </div>
        <div className="pill-row">
          <span className="pill muted">Zoom {mapZoom.toFixed(1)}</span>
          <span className="pill muted">
            {!sceneTilesReady ? "Loading tile field" : showGrid ? `${totalTilesInCircle} tiles in circle` : "Zoom in to open the tile field"}
          </span>
          <button className={`pill pill-button ${basemap === "osm" ? "active" : ""}`} onClick={() => onBasemapChange("osm")} type="button">
            OSM
          </button>
          <button className={`pill pill-button ${basemap === "swisstopo" ? "active" : ""}`} onClick={() => onBasemapChange("swisstopo")} type="button">
            SWISSIMAGE
          </button>
        </div>
      </div>

      <div className="scan-toolbar map-toolbar">
        <label>
          Radius (tiles)
          <input min={2} max={12} step={1} type="number" value={scanRadius} onChange={(event) => onScanRadiusChange(Number(event.target.value) || scanRadius)} />
        </label>
        <label>
          Sweep speed (ms)
          <input min={8} max={400} step={8} type="number" value={scanDelay} onChange={(event) => onScanDelayChange(Number(event.target.value) || scanDelay)} />
        </label>
        <div className="scan-buttons">
          {scanRunning ? (
            <button className="ghost" onClick={onPauseScan} type="button">
              Pause
            </button>
          ) : (
            <button className="primary positive" disabled={scanQueued || scanPreparing} onClick={onStartScan} type="button">
              {scanQueued || scanPreparing ? "Preparing Scan" : "Start Scan"}
            </button>
          )}
          <button className="ghost" onClick={onResetScan} type="button">
            Reset Results
          </button>
          <button className="ghost" disabled={!showGrid || totalTilesInCircle === 0} onClick={onExportBatchJob} type="button">
            Export Batch Job
          </button>
          <button className="ghost" disabled={!showGrid} onClick={onImportBatchResult} type="button">
            Import Batch Result
          </button>
        </div>
      </div>

      <div className="scan-status">
        <span className="pill muted">Scanned {scannedCount}/{totalTilesInCircle}</span>
        <span className="pill green">Crosswalk {crosswalkCount}</span>
        <span className="pill red">No crosswalk {noCrosswalkCount}</span>
        {scanQueued || scanPreparing ? <span className="pill amber">Preparing scan</span> : null}
        {scanRunning ? <span className="pill amber">{scannedCount === 0 && liveScanStep <= 1 ? "Classifying first tiles" : `Running ${liveScanStep}/${totalTilesInCircle}`}</span> : null}
        {!sceneImagesReady ? <span className="pill amber">Loading scene tiles</span> : null}
        <span className="pill muted">{activeSummary ?? "Results are saved automatically"}</span>
      </div>
    </>
  );
}
