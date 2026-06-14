import type { RoadGridLine } from "./types";
import { mercatorToLatLng } from "./utils";

export const TARGET_ROAD_GRID_CELL_SIZE_M = 25;

const WEB_MERCATOR_WORLD_MIN = -20_037_508.342789244;

function gridBounds(bbox: [number, number, number, number], cellSizeM: number) {
  const [minX, minY, maxX, maxY] = bbox;
  const col0 = Math.floor((minX - WEB_MERCATOR_WORLD_MIN) / cellSizeM);
  const col1 = Math.ceil((maxX - WEB_MERCATOR_WORLD_MIN) / cellSizeM);
  const row0 = Math.floor((minY - WEB_MERCATOR_WORLD_MIN) / cellSizeM);
  const row1 = Math.ceil((maxY - WEB_MERCATOR_WORLD_MIN) / cellSizeM);
  return { col0, col1, row0, row1 };
}

export function generateViewportRoadGridLines(
  bbox: [number, number, number, number],
  cellSizeM = TARGET_ROAD_GRID_CELL_SIZE_M,
) {
  const [minX, minY, maxX, maxY] = bbox;
  const { col0, col1, row0, row1 } = gridBounds(bbox, cellSizeM);
  const firstX = WEB_MERCATOR_WORLD_MIN + col0 * cellSizeM;
  const firstY = WEB_MERCATOR_WORLD_MIN + row0 * cellSizeM;
  const lines: RoadGridLine[] = [];

  for (let x = firstX; x <= maxX; x += cellSizeM) {
    lines.push({
      id: `grid-x-${Math.round(x)}`,
      positions: [mercatorToLatLng(x, minY), mercatorToLatLng(x, maxY)],
    });
  }

  for (let y = firstY; y <= maxY; y += cellSizeM) {
    lines.push({
      id: `grid-y-${Math.round(y)}`,
      positions: [mercatorToLatLng(minX, y), mercatorToLatLng(maxX, y)],
    });
  }

  return {
    cellCount: Math.max(0, (col1 - col0) * (row1 - row0)),
    lines,
  };
}
