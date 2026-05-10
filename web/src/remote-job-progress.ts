import type { ScanBatchResult } from "./scan-batch";
import type { BrowserLabelSuggestion } from "./types";

const TILE_PROGRESS_PATTERN =
  /^\[\s*\d+\/\d+\]\s+([^\s]+)\s+->\s+(crosswalk|no_crosswalk)\s+\(([-+]?\d*\.?\d+)\)$/;

type RemoteJobLiveProgress = {
  results: Record<string, BrowserLabelSuggestion>;
  scannedTileIds: string[];
  summary: {
    total: number;
    crosswalk: number;
    no_crosswalk: number;
  };
};

function emptyProgress(): RemoteJobLiveProgress {
  return {
    results: {},
    scannedTileIds: [],
    summary: {
      total: 0,
      crosswalk: 0,
      no_crosswalk: 0,
    },
  };
}

export function parseRemoteJobProgressFromLog(logContents: string, promptText: string): RemoteJobLiveProgress {
  const progress = emptyProgress();

  for (const rawLine of logContents.split(/\r?\n/)) {
    const line = rawLine.replace(/\u001b\[[0-9;]*m/g, "").trim();
    const match = TILE_PROGRESS_PATTERN.exec(line);
    if (!match) continue;

    const [, tileId, label, scoreText] = match;
    const score = Number(scoreText);
    if (!Number.isFinite(score)) continue;

    progress.scannedTileIds.push(tileId);
    progress.results[tileId] = {
      tile_id: tileId,
      label: label as "crosswalk" | "no_crosswalk",
      score,
      peak: score,
      coverage: 1,
      prompt: promptText,
      selected: true,
      review_source: "remote-live",
    };
    if (label === "crosswalk") {
      progress.summary.crosswalk += 1;
    } else {
      progress.summary.no_crosswalk += 1;
    }
  }

  progress.summary.total = progress.scannedTileIds.length;
  return progress;
}

export function progressFromScanBatchResult(result: ScanBatchResult): RemoteJobLiveProgress {
  return {
    results: result.results,
    scannedTileIds: result.tiles.map((tile) => tile.tile_id),
    summary: { ...result.summary },
  };
}
