import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parse as parseToml, stringify as stringifyToml } from "smol-toml";
import {
  listAvailableDatasets,
  loadDataset,
  loadReviewStateFromDataset,
  rawAssetPath,
  saveReviewStateForDataset,
  updateTileInDataset,
  updateTilesInDataset,
  writeDatasetArtifacts,
} from "./dataset";
import { normalizeReviewState } from "./review-state";
import type { DatasetScene, DatasetTile, ReviewState } from "./types";

const WEB_ROOT = resolve(import.meta.dir, "..");
const PROJECT_ROOT = resolve(WEB_ROOT, "..");
const DEFAULT_RUN = "real-v1";
const DEFAULT_EXPORT = "real-balanced-256";
const PORT = Number(process.env.PORT ?? 8787);
const REAL_CONFIG_PATH = join(PROJECT_ROOT, "configs", "real-dataset.toml");
const VALIDATION_OUTPUT_ROOT = join(PROJECT_ROOT, "validation-output", "map-canvas");
const validationStateStore = new Map<string, unknown>();

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
    ...init,
  });
}

function textResponse(text: string, init?: ResponseInit) {
  return new Response(text, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
    },
    ...init,
  });
}

function datasetFromRequest(request: Request) {
  const url = new URL(request.url);
  const run = url.searchParams.get("run") ?? DEFAULT_RUN;
  const exportName = url.searchParams.get("export") ?? DEFAULT_EXPORT;
  return { run, exportName };
}

function datasetSummary(dataset: ReturnType<typeof loadDataset>) {
  const selected = dataset.tiles.filter((tile) => tile.selected);
  const counts = selected.reduce(
    (acc, tile) => {
      acc[tile.label] = (acc[tile.label] ?? 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );
  const scenes = dataset.scenes.map((scene) => ({
    ...scene,
    selected_count: selected.filter((tile) => tile.scene_id === scene.scene_id).length,
  }));
  return {
    run_name: dataset.run_name,
    export_name: dataset.export_name,
    target_per_class: dataset.target_per_class ?? null,
    split_targets: dataset.split_targets ?? {},
    total_tiles: dataset.tiles.length,
    selected_tiles: selected.length,
    selected_crosswalk: counts.crosswalk ?? 0,
    selected_no_crosswalk: counts.no_crosswalk ?? 0,
    dropped_tiles: dataset.tiles.length - selected.length,
    scenes,
  };
}

function sortedSceneTiles(dataset: ReturnType<typeof loadDataset>, sceneId: string) {
  return dataset.tiles
    .filter((tile) => tile.scene_id === sceneId)
    .sort((a, b) => a.row - b.row || a.col - b.col || a.tile_id.localeCompare(b.tile_id));
}

function scenePayload(dataset: ReturnType<typeof loadDataset>, sceneId: string) {
  const scene = dataset.scenes.find((entry) => entry.scene_id === sceneId);
  if (!scene) {
    throw new Error(`Scene not found: ${sceneId}`);
  }
  return {
    summary: datasetSummary(dataset),
    scene,
    tiles: sortedSceneTiles(dataset, sceneId),
  };
}

function readRealConfig() {
  return parseToml(readFileSync(REAL_CONFIG_PATH, "utf8")) as Record<string, unknown>;
}

function writeRealConfig(config: Record<string, unknown>) {
  writeFileSync(REAL_CONFIG_PATH, stringifyToml(config), "utf8");
}

function serveAsset(urlPath: string) {
  const rawPrefix = "/assets/raw/";
  if (urlPath.startsWith(rawPrefix)) {
    const relativePath = urlPath.slice(rawPrefix.length);
    const fullPath = rawAssetPath(relativePath);
    const file = Bun.file(fullPath);
    if (!file.size) return textResponse("not found", { status: 404 });
    return new Response(file, { headers: { "Cache-Control": "public, max-age=3600" } });
  }

  if (urlPath.startsWith("/assets/processed/")) {
    const relativePath = urlPath.replace(/^\/assets\/processed\//, "data/processed/");
    const fullPath = join(PROJECT_ROOT, relativePath);
    const file = Bun.file(fullPath);
    if (!file.size) return textResponse("not found", { status: 404 });
    return new Response(file, { headers: { "Cache-Control": "public, max-age=3600" } });
  }

  return textResponse("not found", { status: 404 });
}

function serveIndex() {
  const dist = join(WEB_ROOT, "dist", "index.html");
  if (statSync(dist, { throwIfNoEntry: false })) {
    return new Response(Bun.file(dist));
  }
  return textResponse("Run `bun run dev` inside web/ to start the UI.");
}

async function handleDataset(request: Request) {
  const { run, exportName } = datasetFromRequest(request);
  const dataset = loadDataset(run, exportName);
  return jsonResponse(dataset);
}

async function handleDatasetMeta(request: Request) {
  const { run, exportName } = datasetFromRequest(request);
  const dataset = loadDataset(run, exportName);
  return jsonResponse(datasetSummary(dataset));
}

async function handleScene(request: Request) {
  const { run, exportName } = datasetFromRequest(request);
  const url = new URL(request.url);
  const sceneId = url.searchParams.get("scene");
  if (!sceneId) {
    return jsonResponse({ error: "scene is required" }, { status: 400 });
  }
  const dataset = loadDataset(run, exportName);
  try {
    return jsonResponse(scenePayload(dataset, sceneId));
  } catch (error) {
    return jsonResponse({ error: String(error) }, { status: 404 });
  }
}

async function handleTileUpdate(request: Request, tileId: string) {
  const { run, exportName } = datasetFromRequest(request);
  const body = (await request.json()) as { label?: string; selected?: boolean };
  if (typeof body.label !== "string" || typeof body.selected !== "boolean") {
    return jsonResponse({ error: "label and selected are required" }, { status: 400 });
  }
  const dataset = updateTileInDataset(run, exportName, tileId, { label: body.label, selected: body.selected });
  const tile = dataset.tiles.find((entry) => entry.tile_id === tileId);
  if (!tile) {
    return jsonResponse({ error: "tile not found after update" }, { status: 404 });
  }
  return jsonResponse(scenePayload(dataset, tile.scene_id));
}

async function handleBatchTileUpdate(request: Request) {
  const { run, exportName } = datasetFromRequest(request);
  const body = (await request.json()) as {
    scene_id?: string;
    updates?: Array<{ tile_id: string; label?: string; selected?: boolean; combined_probability?: number; predicted_label?: string; review_source?: string }>;
  };
  const sceneId = body.scene_id;
  const updates = (body.updates ?? []).filter(
    (update): update is { tile_id: string; label: string; selected: boolean; combined_probability?: number; predicted_label?: string; review_source?: string } =>
      typeof update.tile_id === "string" && typeof update.label === "string" && typeof update.selected === "boolean",
  );
  if (!sceneId || updates.length === 0) {
    return jsonResponse({ error: "scene_id and updates are required" }, { status: 400 });
  }
  const dataset = updateTilesInDataset(run, exportName, updates);
  return jsonResponse(scenePayload(dataset, sceneId));
}

async function handleLegacyPatch(request: Request) {
  const payload = (await request.json()) as {
    exportName?: string;
    updates?: Array<{ relative_path: string; label?: string; review_status?: string; note?: string; confidence?: number; label_source?: string }>;
  };
  const exportName = payload.exportName ?? DEFAULT_EXPORT;
  const dataset = loadDataset(DEFAULT_RUN, exportName);
  for (const update of payload.updates ?? []) {
    const tile = dataset.tiles.find((entry) => entry.relative_path === update.relative_path);
    if (!tile) continue;
    if (typeof update.label === "string") tile.label = update.label;
    if (typeof update.confidence === "number") {
      tile.combined_probability = update.confidence;
      tile.clip_probability = update.confidence;
      tile.heuristic_probability = update.confidence;
    }
    if (typeof update.review_status === "string") {
      tile.status = update.review_status;
      tile.selected = !update.review_status.toLowerCase().includes("drop");
    }
    if (typeof update.label_source === "string") {
      tile.review_source = update.label_source;
    }
    if (typeof update.note === "string") {
      tile.note = update.note;
    }
  }
  writeDatasetArtifacts(dataset);
  return jsonResponse(datasetSummary(dataset));
}

async function handleConfig(request: Request) {
  if (request.method === "GET") {
    return jsonResponse(readRealConfig());
  }
  const body = (await request.json()) as Record<string, unknown>;
  writeRealConfig(body);
  return jsonResponse(readRealConfig());
}

async function handleReviewState(request: Request) {
  const { run, exportName } = datasetFromRequest(request);
  if (request.method === "GET") {
    return jsonResponse(loadReviewStateFromDataset(run, exportName));
  }

  const body = normalizeReviewState((await request.json()) as ReviewState);
  return jsonResponse(saveReviewStateForDataset(run, exportName, body));
}

async function handleMapValidationState(request: Request) {
  const url = new URL(request.url);
  const key = `${url.searchParams.get("validationRunId") ?? "default"}::${url.searchParams.get("validationCase") ?? "default"}`;
  if (request.method === "GET") {
    return jsonResponse(validationStateStore.get(key) ?? null);
  }

  const body = (await request.json()) as Record<string, unknown>;
  validationStateStore.set(key, body);
  return jsonResponse({ ok: true, key });
}

async function handleMapValidationArtifact(request: Request) {
  const body = (await request.json()) as {
    validationRunId?: string;
    validationCase?: string;
    name?: string;
    artifact?: unknown;
  };
  const validationRunId = body.validationRunId ?? "default";
  const validationCase = body.validationCase ?? "default";
  const name = body.name ?? "artifact";
  const dir = join(VALIDATION_OUTPUT_ROOT, validationRunId, validationCase);
  mkdirSync(dir, { recursive: true });
  const path = join(dir, `${name}.json`);
  writeFileSync(path, JSON.stringify(body.artifact ?? body, null, 2), "utf8");
  return jsonResponse({ ok: true, path });
}

const server = Bun.serve({
  port: PORT,
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/api/exports" && request.method === "GET") {
      return jsonResponse(listAvailableDatasets());
    }
    if (url.pathname === "/api/dataset" && request.method === "GET") {
      return handleDataset(request);
    }
    if (url.pathname === "/api/dataset-meta" && request.method === "GET") {
      return handleDatasetMeta(request);
    }
    if (url.pathname === "/api/scene" && request.method === "GET") {
      return handleScene(request);
    }
    if (url.pathname === "/api/tiles/batch" && request.method === "POST") {
      return handleBatchTileUpdate(request);
    }
    if (url.pathname.startsWith("/api/tiles/") && request.method === "POST") {
      const tileId = decodeURIComponent(url.pathname.replace(/^\/api\/tiles\//, ""));
      return handleTileUpdate(request, tileId);
    }
    if (url.pathname === "/api/config" && (request.method === "GET" || request.method === "POST")) {
      return handleConfig(request);
    }
    if (url.pathname === "/api/review-state" && (request.method === "GET" || request.method === "POST")) {
      return handleReviewState(request);
    }
    if (url.pathname === "/api/map-validation-state" && (request.method === "GET" || request.method === "POST")) {
      return handleMapValidationState(request);
    }
    if (url.pathname === "/api/map-validation-artifact" && request.method === "POST") {
      return handleMapValidationArtifact(request);
    }
    if (url.pathname === "/api/state" && request.method === "GET") {
      return handleDataset(request);
    }
    if (url.pathname === "/api/labels" && request.method === "PATCH") {
      return handleLegacyPatch(request);
    }
    if (url.pathname.startsWith("/assets/")) {
      return serveAsset(url.pathname);
    }
    if (url.pathname === "/healthz") {
      return textResponse("ok");
    }
    return serveIndex();
  },
});

console.log(`Crosswalk review server on http://localhost:${server.port}`);
