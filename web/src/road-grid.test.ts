import { describe, expect, test } from "bun:test";
import { generateViewportRoadGridLines, TARGET_ROAD_GRID_CELL_SIZE_M } from "./road-grid";
import { latLngToMercator } from "./utils";

function verticalLineXPositions(bbox: [number, number, number, number]) {
  return generateViewportRoadGridLines(bbox).lines
    .filter((line) => line.id.startsWith("grid-x-"))
    .map((line) => latLngToMercator(line.positions[0][0], line.positions[0][1]).x);
}

describe("road grid", () => {
  test("keeps the rendered grid at 25 m world spacing even for large viewports", () => {
    const xPositions = verticalLineXPositions([1_000_000, 5_900_000, 1_050_000, 5_950_000]);

    expect(xPositions.length).toBeGreaterThan(1_800);
    for (let index = 1; index < xPositions.length; index += 1) {
      expect(xPositions[index] - xPositions[index - 1]).toBeCloseTo(TARGET_ROAD_GRID_CELL_SIZE_M, 6);
    }
  });
});
