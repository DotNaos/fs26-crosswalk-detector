import { describe, expect, test } from "bun:test";
import { buildFakeScanBatchResult } from "./fake-remote-scan";
import type { ScanBatchJob } from "./scan-batch";

function testJob(): ScanBatchJob {
  const tiles = Array.from({ length: 25 }, (_, index) => {
    const row = Math.floor(index / 5);
    const col = index % 5;
    return {
      tile_id: `fake:r${row}:c${col}`,
      row,
      col,
      bbox_mercator: [col * 25, row * 25, col * 25 + 25, row * 25 + 25] as [number, number, number, number],
      center_mercator: { x: col * 25 + 12.5, y: row * 25 + 12.5 },
      center_latlon: { latitude: 47 + row / 1000, longitude: 8 + col / 1000 },
      relative_path: `fake/r${row}-c${col}.jpg`,
    };
  });

  return {
    version: 1,
    created_at: "2026-05-02T00:00:00.000Z",
    dataset: { run_name: "test", export_name: "fake" },
    scene: {
      scene_id: "fake-scene",
      city: "Fake City",
      split: "train",
      latitude: 47,
      longitude: 8,
      size_m: 125,
      image_px: 512,
      tile_size_m: 25,
      geometry: {
        crs: "EPSG:3857",
        center_mercator: { x: 0, y: 0 },
        center_latlon: { latitude: 47, longitude: 8 },
        bbox_mercator: [0, 0, 125, 125],
        bbox_latlon: {
          south_west: { latitude: 47, longitude: 8 },
          north_east: { latitude: 47.1, longitude: 8.1 },
        },
      },
      grid: {
        tile_size_m: 25,
        tile_count: tiles.length,
        rows: 5,
        cols: 5,
        min_row: 0,
        max_row: 4,
        min_col: 0,
        max_col: 4,
      },
    },
    selection: {
      mode: "circle",
      scan_radius_tiles: 2,
      scan_radius_m: 50,
      tile_count: tiles.length,
      center_mercator: { x: 0, y: 0 },
      center_latlon: { latitude: 47, longitude: 8 },
    },
    threshold: 0.5,
    prompts_text: "fake prompts",
    tiles,
  };
}

describe("buildFakeScanBatchResult", () => {
  test("creates deterministic complete scan results", () => {
    const job = testJob();
    const first = buildFakeScanBatchResult(job, "2026-05-02T00:00:00.000Z");
    const second = buildFakeScanBatchResult(job, "2026-05-02T00:00:00.000Z");

    expect(first.summary.total).toBe(25);
    expect(first.summary.crosswalk).toBeGreaterThan(0);
    expect(first.summary.no_crosswalk).toBeGreaterThan(0);
    expect(Object.keys(first.results)).toHaveLength(25);
    expect(first.tiles.map((tile) => [tile.tile_id, tile.label, tile.score])).toEqual(
      second.tiles.map((tile) => [tile.tile_id, tile.label, tile.score]),
    );
  });
});
