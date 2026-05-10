import type { DatasetScene, DatasetTile, SceneReviewState } from "./types";
import {
  bboxAverageSizeM,
  bboxCenterLatLng,
  bboxToMercatorRect,
  circleIntersectsMercatorRect,
  mercatorToLatLng,
  sceneMercatorCenter,
} from "./utils";

type SceneFootprintArgs = {
  scene?: DatasetScene;
  sceneTiles: DatasetTile[];
  sceneReviewState: SceneReviewState;
  mode?: "radius" | "scene";
};

export function computeSceneFootprint({ scene, sceneTiles, sceneReviewState, mode = "radius" }: SceneFootprintArgs) {
  const gridStats = (() => {
    if (!sceneTiles.length) return null;
    const rows = sceneTiles.map((tile) => tile.row);
    const cols = sceneTiles.map((tile) => tile.col);
    const minRow = Math.min(...rows);
    const maxRow = Math.max(...rows);
    const minCol = Math.min(...cols);
    const maxCol = Math.max(...cols);
    return {
      minRow,
      maxRow,
      minCol,
      maxCol,
      centerRow: (minRow + maxRow) / 2,
      centerCol: (minCol + maxCol) / 2,
    };
  })();

  const averageTileSizeM =
    sceneTiles.reduce((sum, tile) => sum + bboxAverageSizeM(tile.bbox_mercator), 0) / Math.max(1, sceneTiles.length);

  const footprintCenterMercator = (() => {
    if (scene) {
      return sceneMercatorCenter(scene);
    }
    if (!gridStats || !sceneTiles.length) return null;
    const centerCandidates = sceneTiles.filter(
      (tile) => Math.abs(tile.row - gridStats.centerRow) <= 0.5 && Math.abs(tile.col - gridStats.centerCol) <= 0.5,
    );
    if (!centerCandidates.length) return null;
    const rects = centerCandidates
      .map((tile) => bboxToMercatorRect(tile.bbox_mercator))
      .filter((rect): rect is NonNullable<ReturnType<typeof bboxToMercatorRect>> => Boolean(rect));
    if (!rects.length) return null;
    return {
      x: (Math.min(...rects.map((rect) => rect.minX)) + Math.max(...rects.map((rect) => rect.maxX))) / 2,
      y: (Math.min(...rects.map((rect) => rect.minY)) + Math.max(...rects.map((rect) => rect.maxY))) / 2,
    };
  })();

  const sceneGridSize = gridStats ? Math.max(gridStats.maxRow - gridStats.minRow + 1, gridStats.maxCol - gridStats.minCol + 1) : 1;
  const footprintRadiusM =
    mode === "scene"
      ? Math.max(averageTileSizeM * sceneGridSize * 0.72, averageTileSizeM * 1.5)
      : Math.max(averageTileSizeM * sceneReviewState.scan_radius, averageTileSizeM * 1.5);

  const footprintTiles = mode === "scene"
    ? sceneTiles
    : !footprintCenterMercator
    ? []
    : sceneTiles.filter((tile) => {
        const rect = bboxToMercatorRect(tile.bbox_mercator);
        return rect ? circleIntersectsMercatorRect(rect, footprintCenterMercator.x, footprintCenterMercator.y, footprintRadiusM) : false;
      });

  const orderedScanTiles = footprintTiles
    .slice()
    .sort((a, b) => a.row - b.row || a.col - b.col || a.tile_id.localeCompare(b.tile_id));

  const footprintCenter = footprintCenterMercator ? mercatorToLatLng(footprintCenterMercator.x, footprintCenterMercator.y) : null;
  const derivedTileSizeM = Math.max(1, Math.round(averageTileSizeM || 25));
  const footprintCenterLatLng = !footprintCenterMercator
    ? null
    : (() => {
        const [latitude, longitude] = mercatorToLatLng(footprintCenterMercator.x, footprintCenterMercator.y);
        return { latitude, longitude };
      })();

  const footprintBBoxLatLng = (() => {
    if (!footprintTiles.length) return null;
    const centers = footprintTiles
      .map((tile) => bboxCenterLatLng(tile.bbox_mercator))
      .filter((center): center is [number, number] => Array.isArray(center));
    if (!centers.length) return null;
    return {
      south: Math.min(...centers.map(([latitude]) => latitude)),
      north: Math.max(...centers.map(([latitude]) => latitude)),
      west: Math.min(...centers.map(([, longitude]) => longitude)),
      east: Math.max(...centers.map(([, longitude]) => longitude)),
    };
  })();

  return {
    averageTileSizeM,
    derivedTileSizeM,
    footprintBBoxLatLng,
    footprintCenter,
    footprintCenterLatLng,
    footprintCenterMercator,
    footprintRadiusM,
    footprintTileIds: new Set(footprintTiles.map((tile) => tile.tile_id)),
    footprintTiles,
    gridStats,
    orderedScanTiles,
  };
}
