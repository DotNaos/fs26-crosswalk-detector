import type { ScanBatchJob, ScanBatchResult, ScanBatchTileGeometry } from "./scan-batch";
import type { BrowserLabelSuggestion } from "./types";

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function round(value: number, digits = 6) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function normalizedGridPosition(job: ScanBatchJob, tile: ScanBatchTileGeometry) {
  const rowSpan = Math.max(1, job.scene.grid.max_row - job.scene.grid.min_row);
  const colSpan = Math.max(1, job.scene.grid.max_col - job.scene.grid.min_col);
  return {
    row: (tile.row - job.scene.grid.min_row) / rowSpan,
    col: (tile.col - job.scene.grid.min_col) / colSpan,
  };
}

export function fakeRemoteSuggestion(job: ScanBatchJob, tile: ScanBatchTileGeometry, index: number): BrowserLabelSuggestion {
  const normalized = normalizedGridPosition(job, tile);
  const seed = hashString(`${job.scene.scene_id}:${tile.tile_id}:${index}`);
  const noise = (seed % 1000) / 1000;
  const centerBand = Math.abs(normalized.row - 0.48) < 0.18 && Math.abs(normalized.col - 0.58) < 0.22;
  const diagonalBand = Math.abs(normalized.row - normalized.col) < 0.09 && normalized.row > 0.2 && normalized.row < 0.85;
  const sparseCrossing = seed % 17 === 0;
  const label = centerBand || diagonalBand || sparseCrossing ? "crosswalk" : "no_crosswalk";
  const score = label === "crosswalk" ? 0.56 + noise * 0.36 : 0.03 + noise * 0.2;
  const coverage = label === "crosswalk" ? 0.04 + noise * 0.24 : noise * 0.02;

  return {
    tile_id: tile.tile_id,
    label,
    score: round(score),
    peak: round(Math.min(0.98, score + 0.08)),
    coverage: round(coverage),
    prompt: "fake-sam3.1-yellow-zebra-crossing",
    selected: true,
    review_source: "fake-sam31-scan",
  };
}

export function buildFakeScanBatchResult(job: ScanBatchJob, createdAt = new Date().toISOString()): ScanBatchResult {
  const results: Record<string, BrowserLabelSuggestion> = {};
  const tiles = job.tiles.map((tile, index) => {
    const suggestion = fakeRemoteSuggestion(job, tile, index);
    results[tile.tile_id] = suggestion;
    return {
      ...tile,
      ...suggestion,
    };
  });
  const crosswalk = tiles.filter((tile) => tile.label === "crosswalk").length;
  const noCrosswalk = tiles.length - crosswalk;

  return {
    version: 1,
    created_at: createdAt,
    completed_at: new Date().toISOString(),
    job,
    scanner: {
      detector_model: "fake-sam3.1",
      clip_model: "fake-audit-fixture",
      device: "fake-gpu",
    },
    summary: {
      total: tiles.length,
      crosswalk,
      no_crosswalk: noCrosswalk,
    },
    tiles,
    results,
  };
}
