import { parse, unparse } from "papaparse";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, renameSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { normalizeReviewState } from "./review-state";
import type { DatasetContract, DatasetListEntry, DatasetScene, DatasetTile, DatasetViewportCluster, DatasetViewportPayload, ReviewState } from "./types";

type LegacyRow = Record<string, string>;

const WEB_ROOT = resolve(import.meta.dir, "..");
const PROJECT_ROOT = resolve(WEB_ROOT, "..");
const PROCESSED_ROOT = join(PROJECT_ROOT, "data", "processed");
const RAW_ROOT = join(PROJECT_ROOT, "data", "raw");

export function datasetDir(runName: string, exportName: string) {
  return join(PROCESSED_ROOT, runName, "exports", exportName);
}

export function tilesJsonPath(runName: string, exportName: string) {
  return join(datasetDir(runName, exportName), "tiles.json");
}

export function labelsCsvPath(runName: string, exportName: string) {
  return join(datasetDir(runName, exportName), "labels.csv");
}

export function summaryJsonPath(runName: string, exportName: string) {
  return join(datasetDir(runName, exportName), "summary.json");
}

export function reviewStateJsonPath(runName: string, exportName: string) {
  return join(datasetDir(runName, exportName), "review-state.json");
}

export function listAvailableDatasets(): DatasetListEntry[] {
  const entries: DatasetListEntry[] = [];
  if (!existsSync(PROCESSED_ROOT)) {
    return entries;
  }
  for (const runDir of readdirSync(PROCESSED_ROOT, { withFileTypes: true })) {
    if (!runDir.isDirectory()) continue;
    const exportsRoot = join(PROCESSED_ROOT, runDir.name, "exports");
    if (!existsSync(exportsRoot)) continue;
    for (const exportDir of readdirSync(exportsRoot, { withFileTypes: true })) {
      if (!exportDir.isDirectory()) continue;
      const runName = runDir.name;
      const exportName = exportDir.name;
      const dataset = loadDataset(runName, exportName);
      entries.push({
        run_name: runName,
        export_name: exportName,
        tile_count: dataset.tiles.length,
        selected_count: dataset.tiles.filter((tile) => tile.selected).length,
        path: datasetDir(runName, exportName),
      });
    }
  }
  return entries.sort((a, b) => {
    if (a.run_name !== b.run_name) return a.run_name.localeCompare(b.run_name);
    return a.export_name.localeCompare(b.export_name);
  });
}

export function loadDataset(runName: string, exportName: string): DatasetContract {
  const path = tilesJsonPath(runName, exportName);
  if (existsSync(path)) {
    return normalizeDataset(JSON.parse(readFileSync(path, "utf8")) as DatasetContract);
  }

  const legacy = loadLegacyDataset(runName, exportName);
  writeDatasetArtifacts(legacy);
  return legacy;
}

export function loadDatasetViewport(
  runName: string,
  exportName: string,
  bbox: [number, number, number, number],
  zoom: number,
  options: {
    city?: string | null;
    label?: string | null;
    limit?: number;
    split?: string | null;
  } = {},
): DatasetViewportPayload {
  const dataset = loadDataset(runName, exportName);
  const limit = Math.max(100, Math.min(Number(options.limit ?? 1400) || 1400, 5000));
  const matching = dataset.tiles.filter((tile) => {
    const rect = bboxToRect(tile.bbox_mercator);
    if (!rect) return false;
    if (!rectIntersects(rect, bbox)) return false;
    if (options.city && tile.city !== options.city) return false;
    if (options.label && tile.label !== options.label) return false;
    if (options.split && tile.split !== options.split) return false;
    return true;
  });

  const summary = datasetSummary(dataset);
  if (matching.length <= Math.min(250, limit) || zoom >= 14.5) {
    return {
      summary,
      bbox_mercator: bbox,
      zoom,
      mode: "tiles",
      total_matching: matching.length,
      returned_tiles: matching.length,
      returned_clusters: 0,
      tiles: matching.sort((a, b) => a.tile_id.localeCompare(b.tile_id)),
      clusters: [],
    };
  }

  const clusters = clusterTiles(matching, bbox, zoom);
  return {
    summary,
    bbox_mercator: bbox,
    zoom,
    mode: "clusters",
    total_matching: matching.length,
    returned_tiles: 0,
    returned_clusters: clusters.length,
    tiles: [],
    clusters,
  };
}

export function updateTileInDataset(
  runName: string,
  exportName: string,
  tileId: string,
  update: { label: string; selected: boolean; combined_probability?: number; predicted_label?: string; review_source?: string },
): DatasetContract {
  const dataset = loadDataset(runName, exportName);
  const tile = dataset.tiles.find((entry) => entry.tile_id === tileId);
  if (!tile) {
    throw new Error(`Tile not found: ${tileId}`);
  }
  applyTileUpdate(tile, update);

  writeDatasetArtifacts(dataset);
  return dataset;
}

export function updateTilesInDataset(
  runName: string,
  exportName: string,
  updates: Array<{ tile_id: string; label: string; selected: boolean; combined_probability?: number; predicted_label?: string; review_source?: string }>,
): DatasetContract {
  const dataset = loadDataset(runName, exportName);
  for (const update of updates) {
    const tile = dataset.tiles.find((entry) => entry.tile_id === update.tile_id);
    if (!tile) {
      continue;
    }
    applyTileUpdate(tile, update);
  }
  writeDatasetArtifacts(dataset);
  return dataset;
}

export function normalizeDataset(dataset: DatasetContract): DatasetContract {
  const tiles = dataset.tiles.map((tile) => ({
    ...tile,
    image_path: toWebImagePath(toStoredImagePath(dataset.run_name, tile)),
    selected: Boolean(tile.selected),
    label: tile.label ?? tile.predicted_label ?? "unknown",
    status: tile.status ?? (tile.selected ? "selected" : "dropped"),
    review_source: tile.review_source ?? "unknown",
    clip_probability: Number(tile.clip_probability ?? 0),
    heuristic_probability: Number(tile.heuristic_probability ?? 0),
    combined_probability: Number(tile.combined_probability ?? 0),
    row: Number(tile.row ?? 0),
    col: Number(tile.col ?? 0),
  }));

  const scenes = normalizeScenes(dataset.scenes, tiles);
  return {
    run_name: dataset.run_name,
    export_name: dataset.export_name,
    target_per_class: dataset.target_per_class,
    split_targets: dataset.split_targets ?? {},
    scenes,
    tiles,
  };
}

function datasetSummary(dataset: DatasetContract) {
  const selected = dataset.tiles.filter((tile) => tile.selected);
  const counts = countLabels(dataset.tiles);
  const selectedCounts = countLabels(selected);
  return {
    run_name: dataset.run_name,
    export_name: dataset.export_name,
    target_per_class: dataset.target_per_class ?? null,
    split_targets: dataset.split_targets ?? {},
    total_tiles: dataset.tiles.length,
    selected_tiles: selected.length,
    selected_crosswalk: selectedCounts.crosswalk ?? 0,
    selected_no_crosswalk: selectedCounts.no_crosswalk ?? 0,
    dropped_tiles: dataset.tiles.length - selected.length,
    label_counts: counts,
    scenes: dataset.scenes,
  };
}

function normalizeScenes(scenes: DatasetScene[] | undefined, tiles: DatasetTile[]) {
  if (scenes?.length) {
    return scenes;
  }
  const groups = new Map<string, DatasetScene>();
  for (const tile of tiles) {
    const existing = groups.get(tile.scene_id);
    if (!existing) {
      groups.set(tile.scene_id, {
        scene_id: tile.scene_id,
        city: tile.city,
        split: tile.split,
        tile_count: 1,
        min_row: tile.row,
        max_row: tile.row,
        min_col: tile.col,
        max_col: tile.col,
      });
      continue;
    }
    existing.tile_count = Number(existing.tile_count ?? 0) + 1;
    existing.min_row = Math.min(Number(existing.min_row ?? tile.row), tile.row);
    existing.max_row = Math.max(Number(existing.max_row ?? tile.row), tile.row);
    existing.min_col = Math.min(Number(existing.min_col ?? tile.col), tile.col);
    existing.max_col = Math.max(Number(existing.max_col ?? tile.col), tile.col);
  }
  return [...groups.values()].sort((a, b) => a.city.localeCompare(b.city) || a.scene_id.localeCompare(b.scene_id));
}

function loadLegacyDataset(runName: string, exportName: string): DatasetContract {
  const csvFile = labelsCsvPath(runName, exportName);
  if (!existsSync(csvFile)) {
    throw new Error(`Missing tiles.json and legacy labels.csv for ${runName}/${exportName}`);
  }
  const csv = readFileSync(csvFile, "utf8");
  const parsed = parse<LegacyRow>(csv, { header: true, skipEmptyLines: true });
  const summary = readSummary(runName, exportName);
  const rows = (parsed.data as LegacyRow[]).map((row) => legacyRowToTile(runName, row));
  const scenes = buildScenesFromTiles(rows);
  return {
    run_name: runName,
    export_name: exportName,
    target_per_class: Number(summary.target_per_class ?? 0) || undefined,
    split_targets: typeof summary.split_targets === "object" && summary.split_targets ? (summary.split_targets as Record<string, number>) : {},
    scenes,
    tiles: rows,
  };
}

function applyTileUpdate(
  tile: DatasetTile,
  update: { label: string; selected: boolean; combined_probability?: number; predicted_label?: string; review_source?: string },
) {
  tile.label = update.label;
  tile.selected = update.selected;
  tile.review_source = update.review_source ?? "manual-review-ui";
  tile.predicted_label = update.predicted_label ?? tile.predicted_label;
  if (typeof update.combined_probability === "number") {
    tile.combined_probability = update.combined_probability;
    tile.clip_probability = update.combined_probability;
    tile.heuristic_probability = update.combined_probability;
  }
  if (!update.selected) {
    tile.status = "dropped";
  } else if ((update.review_source ?? "").startsWith("browser-")) {
    tile.status = "browser-selected";
  } else {
    tile.status = tile.predicted_label === update.label ? "manual-selected" : "manual-corrected";
  }
}

function legacyRowToTile(runName: string, row: LegacyRow): DatasetTile {
  const relativePath = row.relative_path || row.image_path || row.image_url || "";
  const city = row.city || inferCity(relativePath);
  const split = row.split || "train";
  const rowIndex = Number(row.tile_row ?? row.row ?? 0);
  const colIndex = Number(row.tile_col ?? row.col ?? 0);
  const sceneId = row.scene_id || `${city}:${split}`;
  const tileId = row.tile_id || `${sceneId}:${rowIndex}:${colIndex}`;
  const selected = !(row.review_status ?? "").toLowerCase().includes("drop");
  const label = row.label || row.predicted_label || "unknown";
  const probability = Number(row.probability_crosswalk ?? row.combined_probability ?? row.clip_probability ?? row.heuristic_probability ?? 0.5);
  const reviewSource = row.label_source || row.review_source || "legacy-csv";
  const status = row.review_status || (selected ? "selected" : "dropped");
  const imagePath = toWebImagePath(row.image_url || row.image_path || `/assets/raw/${relativePath}`);

  return {
    tile_id: tileId,
    scene_id: sceneId,
    city,
    split,
    row: rowIndex,
    col: colIndex,
    relative_path: relativePath,
    image_path: imagePath,
    bbox_mercator: parseBBox(row.bbox_mercator) ?? parseBBoxColumns(row),
    clip_probability: Number(row.clip_probability ?? probability),
    heuristic_probability: Number(row.heuristic_probability ?? probability),
    combined_probability: Number(row.combined_probability ?? probability),
    predicted_label: row.predicted_label || label,
    label,
    selected,
    status,
    review_source: reviewSource,
  };
}

function parseBBoxColumns(row: LegacyRow) {
  const minX = Number(row.min_x);
  const minY = Number(row.min_y);
  const maxX = Number(row.max_x);
  const maxY = Number(row.max_y);
  if ([minX, minY, maxX, maxY].every(Number.isFinite)) {
    return [minX, minY, maxX, maxY] as [number, number, number, number];
  }
  return null;
}

function toWebImagePath(value: string) {
  if (!value) return value;
  if (value.startsWith("/assets/")) return value;

  const normalized = value.replace(/\\/g, "/");
  const processedMarker = "/data/processed/";
  const rawMarker = "/data/raw/";

  const processedIndex = normalized.indexOf(processedMarker);
  if (processedIndex >= 0) {
    return `/assets/processed/${normalized.slice(processedIndex + processedMarker.length)}`;
  }

  const rawIndex = normalized.indexOf(rawMarker);
  if (rawIndex >= 0) {
    return `/assets/raw/${normalized.slice(rawIndex + rawMarker.length)}`;
  }

  return value;
}

function toStoredImagePath(runName: string, tile: Pick<DatasetTile, "relative_path" | "image_path">) {
  const current = String(tile.image_path ?? "");
  if (current && !current.startsWith("/assets/")) {
    return current;
  }

  const processedCandidate = join(PROCESSED_ROOT, runName, "tiles", tile.relative_path);
  if (existsSync(processedCandidate)) {
    return processedCandidate;
  }

  const localDebugCandidate = join(RAW_ROOT, runName, "wmts-3857-z20", tile.relative_path);
  if (existsSync(localDebugCandidate)) {
    return localDebugCandidate;
  }

  if (current.startsWith("/assets/processed/")) {
    return join(PROJECT_ROOT, current.replace(/^\/assets\/processed\//, "data/processed/"));
  }

  if (current.startsWith("/assets/raw/")) {
    const rawRelativePath = current.replace(/^\/assets\/raw\//, "");
    const directRawCandidate = join(RAW_ROOT, rawRelativePath);
    if (existsSync(directRawCandidate)) {
      return directRawCandidate;
    }
  }

  return current;
}

function inferCity(relativePath: string) {
  const match = /^([^/]+)/.exec(relativePath);
  return match?.[1] ?? "unknown";
}

function parseBBox(value?: string) {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function bboxToRect(value: DatasetTile["bbox_mercator"]) {
  if (!value) return null;
  if (Array.isArray(value)) {
    const [minX, minY, maxX, maxY] = value.map(Number);
    if ([minX, minY, maxX, maxY].every(Number.isFinite)) {
      return [minX, minY, maxX, maxY] as [number, number, number, number];
    }
    return null;
  }
  const minX = Number(value.min_x ?? value.left);
  const minY = Number(value.min_y ?? value.bottom);
  const maxX = Number(value.max_x ?? value.right);
  const maxY = Number(value.max_y ?? value.top);
  if ([minX, minY, maxX, maxY].every(Number.isFinite)) {
    return [minX, minY, maxX, maxY] as [number, number, number, number];
  }
  return null;
}

function rectIntersects(rect: [number, number, number, number], bbox: [number, number, number, number]) {
  return rect[0] <= bbox[2] && rect[2] >= bbox[0] && rect[1] <= bbox[3] && rect[3] >= bbox[1];
}

function countLabels(tiles: DatasetTile[]) {
  const counts: Record<string, number> = {};
  for (const tile of tiles) {
    const label = tile.label || "unknown";
    counts[label] = (counts[label] ?? 0) + 1;
  }
  return counts;
}

function increment(map: Record<string, number>, key: string) {
  map[key] = (map[key] ?? 0) + 1;
}

function clusterTiles(tiles: DatasetTile[], bbox: [number, number, number, number], zoom: number): DatasetViewportCluster[] {
  const width = Math.max(1, bbox[2] - bbox[0]);
  const height = Math.max(1, bbox[3] - bbox[1]);
  const grid = Math.max(8, Math.min(48, Math.round(10 + zoom * 1.6)));
  const cols = grid;
  const rows = Math.max(4, Math.round(grid * (height / width)));
  const clusters = new Map<string, DatasetViewportCluster>();

  for (const tile of tiles) {
    const rect = bboxToRect(tile.bbox_mercator);
    if (!rect) continue;
    const centerX = (rect[0] + rect[2]) / 2;
    const centerY = (rect[1] + rect[3]) / 2;
    const col = Math.max(0, Math.min(cols - 1, Math.floor(((centerX - bbox[0]) / width) * cols)));
    const row = Math.max(0, Math.min(rows - 1, Math.floor(((bbox[3] - centerY) / height) * rows)));
    const id = `${row}:${col}`;
    const cellMinX = bbox[0] + (col / cols) * width;
    const cellMaxX = bbox[0] + ((col + 1) / cols) * width;
    const cellMaxY = bbox[3] - (row / rows) * height;
    const cellMinY = bbox[3] - ((row + 1) / rows) * height;
    let cluster = clusters.get(id);
    if (!cluster) {
      cluster = {
        id,
        bbox_mercator: [cellMinX, cellMinY, cellMaxX, cellMaxY],
        center_mercator: [(cellMinX + cellMaxX) / 2, (cellMinY + cellMaxY) / 2],
        total: 0,
        crosswalk: 0,
        no_crosswalk: 0,
        labels: {},
        splits: {},
        cities: {},
      };
      clusters.set(id, cluster);
    }
    cluster.total += 1;
    if (tile.label === "crosswalk") cluster.crosswalk += 1;
    if (tile.label === "no_crosswalk") cluster.no_crosswalk += 1;
    increment(cluster.labels, tile.label || "unknown");
    increment(cluster.splits, tile.split || "unknown");
    increment(cluster.cities, tile.city || "unknown");
  }

  return [...clusters.values()].sort((a, b) => b.total - a.total);
}

function buildScenesFromTiles(tiles: DatasetTile[]): DatasetScene[] {
  const scenes = new Map<string, DatasetScene>();
  for (const tile of tiles) {
    const scene = scenes.get(tile.scene_id);
    if (!scene) {
      scenes.set(tile.scene_id, {
        scene_id: tile.scene_id,
        city: tile.city,
        split: tile.split,
        tile_count: 1,
        min_row: tile.row,
        max_row: tile.row,
        min_col: tile.col,
        max_col: tile.col,
      });
      continue;
    }
    scene.tile_count = Number(scene.tile_count ?? 0) + 1;
    scene.min_row = Math.min(Number(scene.min_row ?? tile.row), tile.row);
    scene.max_row = Math.max(Number(scene.max_row ?? tile.row), tile.row);
    scene.min_col = Math.min(Number(scene.min_col ?? tile.col), tile.col);
    scene.max_col = Math.max(Number(scene.max_col ?? tile.col), tile.col);
  }
  return [...scenes.values()].sort((a, b) => a.city.localeCompare(b.city) || a.scene_id.localeCompare(b.scene_id));
}

function readSummary(runName: string, exportName: string) {
  const path = summaryJsonPath(runName, exportName);
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  } catch {
    return {};
  }
}

export function writeDatasetArtifacts(dataset: DatasetContract) {
  const dir = datasetDir(dataset.run_name, dataset.export_name);
  mkdirSync(dir, { recursive: true });
  const normalized = normalizeDataset(dataset);
  const storageDataset: DatasetContract = {
    ...normalized,
    tiles: normalized.tiles.map((tile) => ({
      ...tile,
      image_path: toStoredImagePath(dataset.run_name, tile),
    })),
  };
  writeJsonAtomic(tilesJsonPath(dataset.run_name, dataset.export_name), storageDataset);
  writeJsonAtomic(summaryJsonPath(dataset.run_name, dataset.export_name), buildSummary(normalized));
  writeLabelsCsv(storageDataset);
}

export function loadReviewStateFromDataset(runName: string, exportName: string): ReviewState {
  const path = reviewStateJsonPath(runName, exportName);
  if (!existsSync(path)) {
    return normalizeReviewState();
  }
  try {
    return normalizeReviewState(JSON.parse(readFileSync(path, "utf8")) as ReviewState);
  } catch {
    return normalizeReviewState();
  }
}

export function saveReviewStateForDataset(runName: string, exportName: string, state: ReviewState) {
  const normalized = normalizeReviewState(state);
  writeJsonAtomic(reviewStateJsonPath(runName, exportName), normalized);
  return normalized;
}

function buildSummary(dataset: DatasetContract) {
  const selectedTiles = dataset.tiles.filter((tile) => tile.selected);
  const counts = countLabels(selectedTiles);
  return {
    run_name: dataset.run_name,
    export_name: dataset.export_name,
    target_per_class: dataset.target_per_class ?? null,
    split_targets: dataset.split_targets ?? {},
    total_tiles: dataset.tiles.length,
    selected_tiles: selectedTiles.length,
    selected_crosswalk: counts.crosswalk ?? 0,
    selected_no_crosswalk: counts.no_crosswalk ?? 0,
    dropped_tiles: dataset.tiles.length - selectedTiles.length,
    label_counts: counts,
    scene_count: dataset.scenes.length,
    last_review_update_at: new Date().toISOString(),
  };
}

function writeLabelsCsv(dataset: DatasetContract) {
  const selectedTiles = dataset.tiles
    .filter((tile) => tile.selected)
    .sort((a, b) => a.scene_id.localeCompare(b.scene_id) || a.row - b.row || a.col - b.col || a.tile_id.localeCompare(b.tile_id));
  const headers = [
    "tile_id",
    "scene_id",
    "city",
    "split",
    "row",
    "col",
    "relative_path",
    "image_path",
    "bbox_mercator",
    "clip_probability",
    "heuristic_probability",
    "combined_probability",
    "predicted_label",
    "label",
    "selected",
    "status",
    "review_source",
  ];
  const rows = selectedTiles.map((tile) => ({
    ...tile,
    bbox_mercator: tile.bbox_mercator ? JSON.stringify(tile.bbox_mercator) : "",
    selected: String(tile.selected),
  }));
  const csv = unparse(rows, { columns: headers, quotes: true });
  writeTextAtomic(labelsCsvPath(dataset.run_name, dataset.export_name), csv);
}

function writeJsonAtomic(path: string, data: unknown) {
  const tmp = `${path}.${process.pid}.tmp`;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(tmp, JSON.stringify(data, null, 2), "utf8");
  renameSync(tmp, path);
}

function writeTextAtomic(path: string, text: string) {
  const tmp = `${path}.${process.pid}.tmp`;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(tmp, text, "utf8");
  renameSync(tmp, path);
}

export function ensureLegacyTilesJson(runName: string, exportName: string) {
  const path = tilesJsonPath(runName, exportName);
  if (existsSync(path)) return loadDataset(runName, exportName);
  return loadDataset(runName, exportName);
}

export function listDatasetDirectories() {
  return readdirSync(PROCESSED_ROOT, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .flatMap((runDir) => {
      const exportsRoot = join(PROCESSED_ROOT, runDir.name, "exports");
      if (!existsSync(exportsRoot)) return [];
      return readdirSync(exportsRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((exportDir) => ({
          run_name: runDir.name,
          export_name: exportDir.name,
          has_tiles_json: statSync(join(exportsRoot, exportDir.name, "tiles.json"), { throwIfNoEntry: false }) != null,
          has_labels_csv: statSync(join(exportsRoot, exportDir.name, "labels.csv"), { throwIfNoEntry: false }) != null,
        }));
    });
}

export function rawAssetPath(relativePath: string) {
  return join(RAW_ROOT, relativePath);
}
