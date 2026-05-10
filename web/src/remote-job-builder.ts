import { buildScanBatchJob } from "./scan-batch";
import { computeSceneFootprint } from "./scene-footprint";
import { sortScenes } from "./utils";
import type { DatasetContract, DatasetScene, DatasetTile, SceneReviewState } from "./types";

type BuildDatasetScanJobArgs = {
  dataset: DatasetContract;
  sceneId: string;
  scanRadiusTiles: number;
  threshold: number;
  promptsText: string;
};

function defaultSceneReviewState(scanRadiusTiles: number): SceneReviewState {
  return {
    scan_radius: scanRadiusTiles,
    scan_delay_ms: 0,
    scanned_tile_ids: [],
  };
}

export function listDatasetScenes(dataset: DatasetContract) {
  return dataset.scenes
    .map((scene) => ({
      ...scene,
      tile_count: scene.tile_count ?? dataset.tiles.filter((tile) => tile.scene_id === scene.scene_id).length,
    }))
    .sort(sortScenes);
}

export function sceneTilesForDataset(dataset: DatasetContract, sceneId: string) {
  return dataset.tiles
    .filter((tile) => tile.scene_id === sceneId)
    .sort((left, right) => left.row - right.row || left.col - right.col || left.tile_id.localeCompare(right.tile_id));
}

export function findDatasetScene(dataset: DatasetContract, sceneId: string) {
  const scene =
    dataset.scenes.find((entry) => entry.scene_id === sceneId) ??
    dataset.tiles.find((tile) => tile.scene_id === sceneId)?.scene;
  if (!scene) {
    throw new Error(`Scene not found in dataset: ${sceneId}`);
  }
  return scene as DatasetScene;
}

export function buildDatasetScanJob(args: BuildDatasetScanJobArgs) {
  const scene = findDatasetScene(args.dataset, args.sceneId);
  const sceneTiles = sceneTilesForDataset(args.dataset, args.sceneId);
  if (!sceneTiles.length) {
    throw new Error(`Scene ${args.sceneId} has no tiles in this dataset.`);
  }

  const footprint = computeSceneFootprint({
    scene,
    sceneTiles,
    sceneReviewState: defaultSceneReviewState(args.scanRadiusTiles),
  });

  if (!footprint.orderedScanTiles.length) {
    throw new Error(`Scene ${args.sceneId} produced an empty scan footprint.`);
  }

  const job = buildScanBatchJob({
    summary: {
      run_name: args.dataset.run_name,
      export_name: args.dataset.export_name,
    } as never,
    scene,
    tileSizeM: footprint.derivedTileSizeM,
    scanRadiusTiles: args.scanRadiusTiles,
    threshold: args.threshold,
    promptsText: args.promptsText,
    tiles: footprint.orderedScanTiles as DatasetTile[],
  });

  return {
    job,
    scene,
    sceneTiles,
    orderedTiles: footprint.orderedScanTiles,
    derivedTileSizeM: footprint.derivedTileSizeM,
    footprint,
  };
}
