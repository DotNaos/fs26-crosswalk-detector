import { describe, expect, test } from "bun:test";
import { loadDatasetViewport } from "./dataset";

describe("dataset viewport loading", () => {
  test("returns only tiles intersecting the requested mercator viewport", () => {
    const payload = loadDatasetViewport(
      "real-v1",
      "real-balanced-256",
      [844409, 6034223, 844460, 6034250],
      18,
      { limit: 100 },
    );

    expect(payload.mode).toBe("tiles");
    expect(payload.total_matching).toBeGreaterThan(0);
    expect(payload.tiles.every((tile) => tile.city === "Basel")).toBe(true);
  });

  test("clusters dense low-zoom requests instead of returning every tile", () => {
    const payload = loadDatasetViewport(
      "real-v1",
      "real-balanced-256",
      [820000, 5800000, 940000, 6060000],
      9,
      { limit: 2 },
    );

    expect(payload.mode).toBe("clusters");
    expect(payload.returned_tiles).toBe(0);
    expect(payload.returned_clusters).toBeGreaterThan(0);
  });
});
