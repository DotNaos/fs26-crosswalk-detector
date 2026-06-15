import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { appendMetadataLabelVotes, loadMetadataTilePage } from "./metadata-dataset";
import type { MetadataDatasetImage, MetadataDatasetIndex } from "./types";

function image(id: string, overrides: Partial<MetadataDatasetImage> = {}): MetadataDatasetImage {
  return {
    image_id: id,
    tile_id: id,
    source_scene_id: "scene-a",
    perimeter_id: "p0000",
    scene_id: "scene-a",
    city: "Zurich",
    split: "train",
    row: 0,
    col: 0,
    bbox_mercator: [0, 0, 25, 25],
    swisstopo: {
      provider: "swisstopo",
      product: "SWISSIMAGE",
      access: "stac-cog",
      crs: "EPSG:2056",
      asset_id: "swissimage-demo",
      resolution_m: 0.1,
    },
    reconstruction: {
      source_scene_id: "scene-a",
      row: 0,
      col: 0,
      tile_size_m: 25,
      tile_bbox_mercator: [0, 0, 25, 25],
      crop_px: { left: 0, top: 0, width: 64, height: 64 },
      relative_path: `${id}.jpg`,
    },
    labels: [],
    resolved_label: {
      decision: "crosswalk",
      source_id: "sam3.1",
      source_kind: "model",
      resolved_by: "priority",
      confidence: 0.9,
      updated_at: "2026-05-15T00:00:00.000Z",
    },
    review_state: "unreviewed",
    selected_for_training: true,
    ...overrides,
  };
}

describe("metadata dataset JSONL pagination", () => {
  test("loads pages across shards without loading one large file", () => {
    const root = mkdtempSync(join(tmpdir(), "crosswalk-metadata-dataset-pagination-"));
    const datasetId = "sam3-100k";
    const datasetRoot = join(root, datasetId);
    mkdirSync(join(datasetRoot, "scenes", "scene-a", "perimeters", "p0000"), { recursive: true });
    mkdirSync(join(datasetRoot, "scenes", "scene-b", "perimeters", "p0000"), { recursive: true });

    const index: MetadataDatasetIndex = {
      format: "crosswalk-jsonl-v1",
      dataset_id: datasetId,
      run_name: "real-v2-sam3",
      export_name: "sam3-balanced-100k-v1",
      tile_count: 3,
      selected_count: 3,
      shard_target_count: 2_000,
      shards: [
        { shard_id: "scene-a-p0000", path: "scenes/scene-a/perimeters/p0000/tiles.jsonl", tile_count: 2 },
        { shard_id: "scene-b-p0000", path: "scenes/scene-b/perimeters/p0000/tiles.jsonl", tile_count: 1 },
      ],
    };
    writeFileSync(join(datasetRoot, "dataset.json"), JSON.stringify(index, null, 2), "utf8");
    writeFileSync(
      join(datasetRoot, "scenes", "scene-a", "perimeters", "p0000", "tiles.jsonl"),
      `${JSON.stringify(image("a"))}\n${JSON.stringify(image("b", { city: "Basel" }))}\n`,
      "utf8",
    );
    writeFileSync(
      join(datasetRoot, "scenes", "scene-b", "perimeters", "p0000", "tiles.jsonl"),
      `${JSON.stringify(image("c"))}\n`,
      "utf8",
    );

    const first = loadMetadataTilePage(datasetId, null, { limit: 2 }, root);
    expect(first.rows.map((row) => row.image_id)).toEqual(["a", "b"]);
    expect(first.next_cursor).toBe("0:2");

    const second = loadMetadataTilePage(datasetId, first.next_cursor, { limit: 2 }, root);
    expect(second.rows.map((row) => row.image_id)).toEqual(["c"]);
    expect(second.next_cursor).toBeNull();
  });

  test("filters by city and resolved label", () => {
    const root = mkdtempSync(join(tmpdir(), "crosswalk-metadata-dataset-filter-"));
    const datasetId = "sam3-filter";
    const datasetRoot = join(root, datasetId);
    mkdirSync(join(datasetRoot, "scenes", "scene-a", "perimeters", "p0000"), { recursive: true });
    writeFileSync(
      join(datasetRoot, "dataset.json"),
      JSON.stringify({
        format: "crosswalk-jsonl-v1",
        dataset_id: datasetId,
        run_name: "real-v2-sam3",
        export_name: "sam3-balanced-100k-v1",
        tile_count: 2,
        selected_count: 2,
        shard_target_count: 2_000,
        shards: [{ shard_id: "scene-a-p0000", path: "scenes/scene-a/perimeters/p0000/tiles.jsonl", tile_count: 2 }],
      } satisfies MetadataDatasetIndex),
      "utf8",
    );
    writeFileSync(
      join(datasetRoot, "scenes", "scene-a", "perimeters", "p0000", "tiles.jsonl"),
      `${JSON.stringify(image("a", { city: "Zurich" }))}\n${JSON.stringify(
        image("b", {
          city: "Basel",
          resolved_label: {
            decision: "no_crosswalk",
            source_id: "sam3.1",
            source_kind: "model",
            resolved_by: "priority",
            updated_at: "2026-05-15T00:00:00.000Z",
          },
        }),
      )}\n`,
      "utf8",
    );

    const page = loadMetadataTilePage(datasetId, null, { city: "Basel", label: "no_crosswalk" }, root);
    expect(page.rows.map((row) => row.image_id)).toEqual(["b"]);
  });

  test("appends model and human votes without deleting earlier model labels", () => {
    const root = mkdtempSync(join(tmpdir(), "crosswalk-metadata-dataset-votes-"));
    const datasetId = "sam3-votes";
    const datasetRoot = join(root, datasetId);
    mkdirSync(join(datasetRoot, "scenes", "scene-a", "perimeters", "p0000"), { recursive: true });
    writeFileSync(
      join(datasetRoot, "dataset.json"),
      JSON.stringify({
        format: "crosswalk-jsonl-v1",
        dataset_id: datasetId,
        run_name: "sam3-votes",
        export_name: "metadata-votes",
        tile_count: 1,
        selected_count: 1,
        shard_target_count: 2_000,
        shards: [{ shard_id: "scene-a-p0000", path: "scenes/scene-a/perimeters/p0000/tiles.jsonl", tile_count: 1 }],
      } satisfies MetadataDatasetIndex),
      "utf8",
    );
    writeFileSync(
      join(datasetRoot, "scenes", "scene-a", "perimeters", "p0000", "tiles.jsonl"),
      `${JSON.stringify(
        image("a", {
          labels: [
            {
              confidence: 0.6,
              created_at: "2026-05-15T00:00:00.000Z",
              decision: "crosswalk",
              source: { display_name: "SAM3.1", kind: "model", priority: 100, source_id: "sam3.1" },
              vote_id: "sam3.1:a",
            },
          ],
        }),
      )}\n`,
      "utf8",
    );

    appendMetadataLabelVotes(
      "sam3-votes",
      "metadata-votes",
      [{ confidence: 0.9, decision: "no_crosswalk", tile_id: "a" }],
      { display_name: "CrossMaskNet v4", kind: "model", priority: 120, source_id: "crossmasknet-v4" },
      root,
    );
    appendMetadataLabelVotes(
      "sam3-votes",
      "metadata-votes",
      [{ confidence: 1.0, decision: "crosswalk", tile_id: "a" }],
      { display_name: "Human reviewer", kind: "human", priority: 1000, source_id: "human:reviewer" },
      root,
    );

    const [row] = loadMetadataTilePage(datasetId, null, { limit: 1 }, root).rows;
    expect(row.labels.map((vote) => vote.source.source_id)).toEqual(["sam3.1", "crossmasknet-v4", "human:reviewer"]);
    expect(row.resolved_label).toMatchObject({ decision: "crosswalk", resolved_by: "human_override", source_id: "human:reviewer" });
  });
});
