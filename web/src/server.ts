import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parse as parseToml, stringify as stringifyToml } from "smol-toml";
import { loadLocalEnv } from "./local-env";
import { runCrossMaskBatch } from "./crossmask-runner";
import {
  listAvailableDatasets,
  loadDataset,
  loadDatasetViewport,
  loadReviewStateFromDataset,
  rawAssetPath,
  saveReviewStateForDataset,
  updateTileInDataset,
  updateTilesInDataset,
  writeDatasetArtifacts,
} from "./dataset";
import {
  appendMetadataLabelVotes,
  isMetadataDataset,
  listMetadataDatasetEntries,
  listMetadataDatasets,
  loadMetadataDatasetSummary,
  loadMetadataDatasetViewport,
  loadMetadataTilePage,
} from "./metadata-dataset";
import { createRemoteController } from "./remote-controller";
import { normalizeReviewState } from "./review-state";
import type { ScanBatchJob } from "./scan-batch";
import type { DatasetScene, DatasetTile, ReviewState } from "./types";
import {
  type RemoteServeInstance,
  type RemoteTerminalSocketData,
  type RemoteUpgradeServer,
  createRemoteWebsocketHandlers,
} from "./server-websocket";

const WEB_ROOT = resolve(import.meta.dir, "..");
const PROJECT_ROOT = resolve(WEB_ROOT, "..");
loadLocalEnv(PROJECT_ROOT, WEB_ROOT);
const DEFAULT_RUN = "real-v1";
const DEFAULT_EXPORT = "real-balanced-256";
const PORT = Number(process.env.PORT ?? 8787);
const REAL_CONFIG_PATH = join(PROJECT_ROOT, "configs", "real-dataset.toml");
const VALIDATION_OUTPUT_ROOT = join(PROJECT_ROOT, "validation-output", "map-canvas");
const AUTOPILOT_CACHE_ROOT = join(PROJECT_ROOT, "data", "cache", "autopilot");
const ROAD_CLUSTER_CACHE_ROOT = join(PROJECT_ROOT, "data", "cache", "road-cluster-grid");
const AUTOPILOT_SCHEMA_VERSION = 6;
const validationStateStore = new Map<string, unknown>();
const autopilotPlanCache = new Map<string, unknown>();
const roadClusterGridCache = new Map<string, unknown>();
const remoteController = createRemoteController(PROJECT_ROOT);
const TMUX_SCROLLBACK_LINES = "-2000";

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

function captureTmuxPane(tmuxSession: string) {
  return execFileSync("tmux", ["capture-pane", "-e", "-p", "-t", tmuxSession, "-S", TMUX_SCROLLBACK_LINES], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function tmuxSessionExists(tmuxSession: string) {
  try {
    execFileSync("tmux", ["has-session", "-t", tmuxSession], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function readLogSnapshot(path: string) {
  return existsSync(path) ? readFileSync(path, "utf8") : "";
}

function readTerminalSnapshot(socketData: RemoteTerminalSocketData) {
  if (tmuxSessionExists(socketData.tmuxSession)) {
    return captureTmuxPane(socketData.tmuxSession);
  }
  return readLogSnapshot(socketData.localLogPath);
}

function sendTmuxInput(tmuxSession: string, input: string) {
  if (!input.length) return;
  const normalized = input.replace(/\r/g, "\n");
  const parts = normalized.split(/(\n)/);
  for (const part of parts) {
    if (!part) continue;
    if (part === "\n") {
      execFileSync("tmux", ["send-keys", "-t", tmuxSession, "Enter"], { stdio: "ignore" });
      continue;
    }
    execFileSync("tmux", ["send-keys", "-t", tmuxSession, "-l", "--", part], { stdio: "ignore" });
  }
}

function buildTerminalWebSocketResponse(
  server: RemoteUpgradeServer,
  request: Request,
  jobId: string,
) {
  const terminalSource = remoteController.getTerminalSource(jobId);
  if (!terminalSource.tmux_session) {
    return jsonResponse({ error: "No tmux session is available for this run." }, { status: 404 });
  }
  const upgraded = server.upgrade(request, {
    data: {
      jobId,
      tmuxSession: terminalSource.tmux_session,
      localLogPath: terminalSource.local_log_path,
      lastSnapshot: "",
    },
  });
  if (!upgraded) {
    return jsonResponse({ error: "Terminal upgrade failed." }, { status: 500 });
  }
  return new Response(null);
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
  const distAssetPath = join(WEB_ROOT, "dist", urlPath.replace(/^\//, ""));
  if (statSync(distAssetPath, { throwIfNoEntry: false })) {
    return new Response(Bun.file(distAssetPath), { headers: { "Cache-Control": "public, max-age=3600" } });
  }

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

  if (urlPath.startsWith("/assets/metadata/")) {
    const match = /^\/assets\/metadata\/([^/]+)\/(.+)$/.exec(urlPath);
    const datasetId = match?.[1];
    const relativePath = match?.[2];
    if (!datasetId || !relativePath || relativePath.includes("..")) return textResponse("not found", { status: 404 });
    const fullPath = join(PROJECT_ROOT, "datasets", datasetId, relativePath);
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
  if (isMetadataDataset(run, exportName)) {
    return jsonResponse(loadMetadataDatasetSummary(run, exportName));
  }
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

async function handleDatasetViewport(request: Request) {
  const { run, exportName } = datasetFromRequest(request);
  const url = new URL(request.url);
  const bbox = (url.searchParams.get("bbox") ?? "").split(",").map((value) => Number(value));
  const zoom = Number(url.searchParams.get("zoom") ?? 8);
  const limit = Number(url.searchParams.get("limit") ?? 1400);
  const city = url.searchParams.get("city");
  const label = url.searchParams.get("label");
  const split = url.searchParams.get("split");
  if (bbox.length !== 4 || bbox.some((value) => !Number.isFinite(value)) || !Number.isFinite(zoom)) {
    return jsonResponse({ error: "bbox=minX,minY,maxX,maxY and zoom are required." }, { status: 400 });
  }
  if (isMetadataDataset(run, exportName)) {
    return jsonResponse(
      loadMetadataDatasetViewport(run, exportName, bbox as [number, number, number, number], zoom, {
        city: city && city !== "all" ? city : null,
        label: label && label !== "all" ? label : null,
        limit,
        split: split && split !== "all" ? split : null,
      }),
    );
  }
  return jsonResponse(
    loadDatasetViewport(run, exportName, bbox as [number, number, number, number], zoom, {
      city: city && city !== "all" ? city : null,
      label: label && label !== "all" ? label : null,
      limit,
      split: split && split !== "all" ? split : null,
    }),
  );
}

async function handleMetadataDatasets() {
  return jsonResponse(listMetadataDatasets());
}

async function handleMetadataDatasetTiles(request: Request) {
  const url = new URL(request.url);
  const dataset = url.searchParams.get("dataset");
  if (!dataset) {
    return jsonResponse({ error: "dataset is required" }, { status: 400 });
  }
  const selectedRaw = url.searchParams.get("selected");
  return jsonResponse(
    loadMetadataTilePage(dataset, url.searchParams.get("cursor"), {
      city: nullableFilter(url.searchParams.get("city")),
      label: nullableFilter(url.searchParams.get("label")),
      limit: Number(url.searchParams.get("limit") ?? 200),
      review_state: nullableFilter(url.searchParams.get("review_state")),
      selected: selectedRaw == null || selectedRaw === "all" ? null : selectedRaw === "true",
      split: nullableFilter(url.searchParams.get("split")),
    }),
  );
}

async function handleCrossMaskRun(request: Request) {
  return jsonResponse(runCrossMaskBatch(PROJECT_ROOT, await request.json()));
}

function nullableFilter(value: string | null) {
  return value && value !== "all" ? value : null;
}

async function handleTileUpdate(request: Request, tileId: string) {
  const { run, exportName } = datasetFromRequest(request);
  const body = (await request.json()) as { label?: string; selected?: boolean };
  if (typeof body.label !== "string" || typeof body.selected !== "boolean") {
    return jsonResponse({ error: "label and selected are required" }, { status: 400 });
  }
  if (isMetadataDataset(run, exportName)) {
    const [tile] = appendMetadataLabelVotes(
      run,
      exportName,
      [{ confidence: 1.0, decision: body.selected ? (body.label as "crosswalk" | "no_crosswalk") : "drop", tile_id: tileId }],
      { display_name: "Oli", kind: "human", priority: 1000, source_id: "human:oli" },
    );
    return jsonResponse({ tile });
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

async function handleRemoteConfig(request: Request) {
  if (request.method === "GET") {
    return jsonResponse(remoteController.getSnapshot());
  }

  const body = (await request.json()) as Parameters<typeof remoteController.saveConfig>[0];
  return jsonResponse(remoteController.saveConfig(body));
}

async function handleRemoteConnect() {
  return jsonResponse(remoteController.connect());
}

function pythonExecutable() {
  const venvPython = join(PROJECT_ROOT, ".venv", "bin", "python");
  return existsSync(venvPython) ? venvPython : "python3";
}

async function handleAutopilotPlan(request: Request) {
  const url = new URL(request.url);
  const targetPositiveCount = Number(url.searchParams.get("targetPositiveCount") ?? 500) || 500;
  const maxPanels = Number(url.searchParams.get("maxPanels") ?? 8) || 8;
  const perimeterBudget = Number(url.searchParams.get("perimeterBudget") ?? 72) || 72;
  const cacheKey = `v${AUTOPILOT_SCHEMA_VERSION}:${targetPositiveCount}:${maxPanels}:${perimeterBudget}`;
  if (url.searchParams.get("refresh") !== "1") {
    const cached = autopilotPlanCache.get(cacheKey);
    if (cached) return jsonResponse(cached);
  }
  mkdirSync(AUTOPILOT_CACHE_ROOT, { recursive: true });
  const output = execFileSync(
    pythonExecutable(),
    [
      "-m",
      "crosswalk_detector.autopilot",
      "--web-plan",
      "--target-positive-count",
      String(targetPositiveCount),
      "--max-panels",
      String(maxPanels),
      "--perimeter-budget",
      String(perimeterBudget),
      "--cache-dir",
      AUTOPILOT_CACHE_ROOT,
    ],
    {
      cwd: PROJECT_ROOT,
      encoding: "utf8",
      maxBuffer: 20 * 1024 * 1024,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  const plan = JSON.parse(output);
  autopilotPlanCache.set(cacheKey, plan);
  return jsonResponse(plan);
}

async function handleRoadClusterGrid(request: Request) {
  const startedAt = Date.now();
  const url = new URL(request.url);
  const bbox = (url.searchParams.get("bbox") ?? "").split(",").map((value) => Number(value));
  const zoom = Number(url.searchParams.get("zoom") ?? 0);
  if (bbox.length !== 4 || bbox.some((value) => !Number.isFinite(value)) || !Number.isFinite(zoom)) {
    return jsonResponse({ error: "bbox=minX,minY,maxX,maxY and zoom are required." }, { status: 400 });
  }
  const roundedBbox = bbox.map((value) => Math.round(value));
  const roundedZoom = Math.round(zoom * 4) / 4;
  const cacheKey = `${roundedBbox.join(",")}:${roundedZoom}`;
  const cached = roadClusterGridCache.get(cacheKey);
  if (cached) return jsonResponse(cached);

  mkdirSync(ROAD_CLUSTER_CACHE_ROOT, { recursive: true });
  const output = execFileSync(
    pythonExecutable(),
    [
      "-m",
      "crosswalk_detector.road_cluster_grid",
      "--bbox",
      roundedBbox.join(","),
      "--zoom",
      String(roundedZoom),
      "--cache-dir",
      ROAD_CLUSTER_CACHE_ROOT,
    ],
    {
      cwd: PROJECT_ROOT,
      encoding: "utf8",
      maxBuffer: 20 * 1024 * 1024,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  const grid = JSON.parse(output);
  roadClusterGridCache.set(cacheKey, grid);
  console.log(
    `Road cluster grid zoom=${roundedZoom} bbox=${roundedBbox.join(",")} cells=${grid.cells?.length ?? 0} cellSize=${grid.cellSizeM ?? 0} durationMs=${Date.now() - startedAt}`,
  );
  return jsonResponse(grid);
}

async function handleRemoteJobs(request: Request) {
  if (request.method === "GET") {
    return jsonResponse(remoteController.listJobs());
  }

  const body = (await request.json()) as { job?: ScanBatchJob };
  if (!body.job) {
    return jsonResponse({ error: "job is required" }, { status: 400 });
  }
  return jsonResponse(remoteController.startJob(body.job));
}

async function handleRemoteJobDetail(jobId: string) {
  return jsonResponse(remoteController.getJob(jobId));
}

async function handleRemoteJobResult(jobId: string) {
  return jsonResponse(remoteController.loadResult(jobId));
}

async function handleRemoteJobCancel(jobId: string) {
  return jsonResponse(remoteController.cancelJob(jobId));
}

let server: RemoteServeInstance;

server = Bun.serve({
  port: PORT,
  async fetch(request: Request) {
    try {
      return await (async () => {
        const url = new URL(request.url);
        if (url.pathname.startsWith("/api/remote/jobs/") && url.pathname.endsWith("/terminal")) {
          const jobId = decodeURIComponent(url.pathname.replace(/^\/api\/remote\/jobs\//, "").replace(/\/terminal$/, ""));
          return buildTerminalWebSocketResponse(server, request, jobId);
        }
        if (url.pathname === "/api/exports" && request.method === "GET") {
          return jsonResponse([...listAvailableDatasets(), ...listMetadataDatasetEntries()]);
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
        if (url.pathname === "/api/dataset/viewport" && request.method === "GET") {
          return handleDatasetViewport(request);
        }
        if (url.pathname === "/api/metadata-datasets" && request.method === "GET") {
          return handleMetadataDatasets();
        }
        if (url.pathname === "/api/metadata-dataset/tiles" && request.method === "GET") {
          return handleMetadataDatasetTiles(request);
        }
        if (url.pathname === "/api/crossmask/run" && request.method === "POST") {
          return handleCrossMaskRun(request);
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
        if (url.pathname === "/api/autopilot/plan" && request.method === "GET") {
          return handleAutopilotPlan(request);
        }
        if (url.pathname === "/api/autopilot/road-clusters" && request.method === "GET") {
          return handleRoadClusterGrid(request);
        }
        if (url.pathname === "/api/remote/config" && (request.method === "GET" || request.method === "PUT")) {
          return handleRemoteConfig(request);
        }
        if (url.pathname === "/api/remote/connect" && request.method === "POST") {
          return handleRemoteConnect();
        }
        if (url.pathname === "/api/remote/jobs" && (request.method === "GET" || request.method === "POST")) {
          return handleRemoteJobs(request);
        }
        if (url.pathname.startsWith("/api/remote/jobs/")) {
          const suffix = url.pathname.replace(/^\/api\/remote\/jobs\//, "");
          if (suffix.endsWith("/cancel") && request.method === "POST") {
            const jobId = decodeURIComponent(suffix.replace(/\/cancel$/, ""));
            return handleRemoteJobCancel(jobId);
          }
          if (suffix.endsWith("/result") && request.method === "GET") {
            const jobId = decodeURIComponent(suffix.replace(/\/result$/, ""));
            return handleRemoteJobResult(jobId);
          }
          if (request.method === "GET") {
            const jobId = decodeURIComponent(suffix);
            return handleRemoteJobDetail(jobId);
          }
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
      })();
    } catch (error) {
      return jsonResponse({ error: String(error) }, { status: 500 });
    }
  },
  websocket: createRemoteWebsocketHandlers({ readTerminalSnapshot, tmuxSessionExists, sendTmuxInput }),
} as never) as RemoteServeInstance;

console.log(`Crosswalk review server on http://localhost:${server.port}`);
const remoteSnapshot = remoteController.getSnapshot();
if (!remoteSnapshot.password_configured) {
  console.warn(
    "[remote-controller] CROSSWALK_REMOTE_PASSWORD is not set. Put it in your local .env file and restart `bun run dev` before connecting.",
  );
}
