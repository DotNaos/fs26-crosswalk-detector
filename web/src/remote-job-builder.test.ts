import { describe, expect, test } from "bun:test";
import { loadDataset } from "./dataset";
import { buildDatasetScanJob } from "./remote-job-builder";

describe("buildDatasetScanJob", () => {
  test("builds the Zurich center radius-4 scan job from the exported dataset", () => {
    const dataset = loadDataset("real-v1", "real-balanced-256");

    const result = buildDatasetScanJob({
      dataset,
      sceneId: "zurich-center",
      scanRadiusTiles: 4,
      threshold: 0.32,
      promptsText: "server-side hybrid scan",
    });

    expect(result.scene.scene_id).toBe("zurich-center");
    expect(result.orderedTiles.length).toBe(66);
    expect(result.job.tiles.length).toBe(66);
    expect(result.job.selection.tile_count).toBe(66);
    expect(result.job.selection.scan_radius_tiles).toBe(4);
    expect(result.job.threshold).toBe(0.32);
    expect(result.job.prompts_text).toBe("server-side hybrid scan");
  });
});
