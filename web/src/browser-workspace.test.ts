import { beforeEach, describe, expect, test } from "bun:test";
import {
  createBrowserDataset,
  listBrowserDatasets,
  loadBrowserConfig,
  loadBrowserDatasetMeta,
  loadBrowserReviewState,
} from "./browser-workspace";

class MemoryStorage {
  private store = new Map<string, string>();

  clear() {
    this.store.clear();
  }

  getItem(key: string) {
    return this.store.get(key) ?? null;
  }

  key(index: number) {
    return Array.from(this.store.keys())[index] ?? null;
  }

  removeItem(key: string) {
    this.store.delete(key);
  }

  setItem(key: string, value: string) {
    this.store.set(key, value);
  }

  get length() {
    return this.store.size;
  }
}

beforeEach(() => {
  (globalThis as { window?: unknown }).window = globalThis;
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: new MemoryStorage(),
  });
});

describe("browser workspace datasets", () => {
  test("creates a new dataset with its own scene and readable name", () => {
    const created = createBrowserDataset({
      name: "HB Review Set",
      sceneId: "zurich-center",
    });

    expect(created.display_name).toBe("HB Review Set");
    expect(created.export_name).toBe("hb-review-set");

    const datasets = listBrowserDatasets();
    expect(datasets.some((entry) => entry.export_name === "hb-review-set")).toBe(true);

    const config = loadBrowserConfig(created.run_name, created.export_name);
    expect(config.display_name).toBe("HB Review Set");
    expect(config.scenes).toHaveLength(1);
    expect(config.scenes[0]?.scene_id).toBe("zurich-center");

    const summary = loadBrowserDatasetMeta(created.run_name, created.export_name);
    expect(summary.display_name).toBe("HB Review Set");
    expect(summary.scenes).toHaveLength(1);
    expect(summary.scenes[0]?.scene_id).toBe("zurich-center");

    const reviewState = loadBrowserReviewState(created.run_name, created.export_name);
    expect(reviewState.selected_scene_id).toBe("zurich-center");
  });
});
