import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { appendMetadataLabelVotes, isMetadataDataset } from "./metadata-dataset";
import type { DatasetTile } from "./types";

type CrossMaskRunRequest = {
  export_name?: string;
  max_tiles?: number;
  run_name?: string;
  threshold?: number;
  tiles?: DatasetTile[];
};

type CrossMaskPrediction = {
  confidence: number;
  decision: "crosswalk" | "no_crosswalk";
  mask_coverage: number;
  mask_score: number;
  tile_id: string;
};

const DEFAULT_MODEL_ROOT = "models/crossmask/sam3-500k-road-channel-v4";
const DEFAULT_MAX_TILES = 64;
const CROSSMASK_SOURCE = {
  display_name: "CrossMaskNet v4",
  kind: "model" as const,
  priority: 120,
  source_id: "crossmasknet-v4",
};

function pythonExecutable(projectRoot: string) {
  const venvPython = join(projectRoot, ".venv", "bin", "python");
  return existsSync(venvPython) ? venvPython : "python3";
}

export function runCrossMaskBatch(projectRoot: string, body: CrossMaskRunRequest) {
  const runName = String(body.run_name ?? "");
  const exportName = String(body.export_name ?? "");
  if (!runName || !exportName) {
    throw new Error("run_name and export_name are required.");
  }
  if (!isMetadataDataset(runName, exportName)) {
    throw new Error("CrossMask runs are currently enabled for metadata datasets only.");
  }
  const tiles = (body.tiles ?? []).slice(0, Math.max(1, Math.min(Number(body.max_tiles ?? DEFAULT_MAX_TILES) || DEFAULT_MAX_TILES, 256)));
  if (!tiles.length) {
    throw new Error("No visible tiles were sent for the CrossMask run.");
  }

  const runId = `crossmask-${Date.now()}`;
  const runRoot = join(projectRoot, ".local", "crossmask-runs", runId);
  mkdirSync(runRoot, { recursive: true });
  const requestPath = join(runRoot, "request.json");
  const outputPath = join(runRoot, "result.json");
  const requestPayload = {
    dataset_root: join(projectRoot, "datasets", runName),
    model_root: join(projectRoot, process.env.CROSSWALK_CROSSMASK_MODEL_ROOT ?? DEFAULT_MODEL_ROOT),
    run_id: runId,
    threshold: Number(body.threshold ?? 0.005),
    tiles: tiles.map((tile) => ({
      bbox_mercator: tile.bbox_mercator,
      city: tile.city,
      col: tile.col,
      relative_path: tile.relative_path,
      row: tile.row,
      scene_id: tile.scene_id,
      split: tile.split,
      tile_id: tile.tile_id,
    })),
  };
  writeFileSync(requestPath, JSON.stringify(requestPayload, null, 2), "utf8");

  execFileSync(pythonExecutable(projectRoot), ["-m", "crosswalk_detector.cli", "run-crossmask-tiles", "--request", requestPath, "--output", outputPath], {
    cwd: projectRoot,
    encoding: "utf8",
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    maxBuffer: 20 * 1024 * 1024,
    stdio: ["ignore", "pipe", "pipe"],
  });

  const result = JSON.parse(readFileSync(outputPath, "utf8")) as {
    model: string;
    predictions: CrossMaskPrediction[];
    run_id: string;
    summary: { crosswalk: number; no_crosswalk: number; total: number };
  };
  const updatedTiles = appendMetadataLabelVotes(
    runName,
    exportName,
    result.predictions.map((prediction) => ({
      confidence: prediction.confidence,
      decision: prediction.decision,
      metadata: {
        mask_coverage: prediction.mask_coverage,
        mask_score: prediction.mask_score,
        model: result.model,
        run_id: result.run_id,
      },
      tile_id: prediction.tile_id,
    })),
    CROSSMASK_SOURCE,
  );

  return { ...result, updated_tiles: updatedTiles };
}
