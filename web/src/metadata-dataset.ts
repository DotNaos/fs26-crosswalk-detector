import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, normalize, resolve } from "node:path";
import type {
  DatasetListEntry,
  DatasetScene,
  DatasetSummary,
  DatasetTile,
  DatasetViewportCluster,
  DatasetViewportPayload,
  MetadataDatasetImage,
  MetadataDatasetIndex,
  MetadataTilePage,
  ImageLabelVote,
  ResolvedLabel,
} from "./types";

const WEB_ROOT = resolve(import.meta.dir, "..");
const PROJECT_ROOT = resolve(WEB_ROOT, "..");
export const METADATA_DATASETS_ROOT = join(PROJECT_ROOT, "datasets");
const DEFAULT_LIMIT = 200;
const MAX_LIMIT = 1_000;
const VIEWPORT_TILE_LIMIT = 5_000;

const tileCache = new Map<string, DatasetTile[]>();

type MetadataTileFilters = {
  city?: string | null;
  label?: string | null;
  limit?: number;
  review_state?: string | null;
  selected?: boolean | null;
  split?: string | null;
};

type Cursor = {
  shardIndex: number;
  lineIndex: number;
};

export type MetadataLabelPrediction = {
  tile_id: string;
  decision: "crosswalk" | "no_crosswalk" | "drop";
  confidence?: number;
  metadata?: Record<string, unknown>;
};

type MetadataLabelSource = {
  source_id: string;
  kind: "model" | "human";
  priority: number;
  display_name: string;
};

export function listMetadataDatasets(root = METADATA_DATASETS_ROOT): MetadataDatasetIndex[] {
  if (!existsSync(root)) return [];
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      try {
        return loadMetadataDatasetIndex(entry.name, root);
      } catch {
        return null;
      }
    })
    .filter((entry): entry is MetadataDatasetIndex => Boolean(entry))
    .sort((left, right) => left.dataset_id.localeCompare(right.dataset_id));
}

export function listMetadataDatasetEntries(root = METADATA_DATASETS_ROOT): DatasetListEntry[] {
  return listMetadataDatasets(root).map((dataset) => ({
    display_name: dataset.display_name,
    export_name: dataset.export_name,
    path: join(root, dataset.dataset_id),
    run_name: dataset.run_name,
    selected_count: dataset.selected_count,
    tile_count: dataset.tile_count,
  }));
}

export function isMetadataDataset(runName: string, exportName: string, root = METADATA_DATASETS_ROOT) {
  return listMetadataDatasets(root).some((dataset) => dataset.run_name === runName && dataset.export_name === exportName);
}

export function loadMetadataDatasetSummary(runName: string, exportName: string, root = METADATA_DATASETS_ROOT): DatasetSummary {
  const dataset = findMetadataDataset(runName, exportName, root);
  return {
    display_name: dataset.display_name,
    dropped_tiles: dataset.tile_count - dataset.selected_count,
    export_name: dataset.export_name,
    run_name: dataset.run_name,
    scenes: metadataScenes(dataset),
    selected_crosswalk: 0,
    selected_no_crosswalk: 0,
    selected_tiles: dataset.selected_count,
    total_tiles: dataset.tile_count,
  };
}

export function loadMetadataDatasetViewport(
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
  root = METADATA_DATASETS_ROOT,
): DatasetViewportPayload {
  const dataset = findMetadataDataset(runName, exportName, root);
  const limit = Math.max(100, Math.min(Number(options.limit ?? 1400) || 1400, VIEWPORT_TILE_LIMIT));
  const matching = loadMetadataDatasetTiles(dataset, root).filter((tile) => {
    const rect = bboxToRect(tile.bbox_mercator);
    if (!rect) return false;
    if (!rectIntersects(rect, bbox)) return false;
    if (options.city && tile.city !== options.city) return false;
    if (options.label && tile.label !== options.label) return false;
    if (options.split && tile.split !== options.split) return false;
    return true;
  });

  const summary = loadMetadataDatasetSummary(runName, exportName, root);
  if (matching.length <= Math.min(250, limit) || zoom >= 14.5) {
    const tiles = matching.slice(0, limit).sort((a, b) => a.tile_id.localeCompare(b.tile_id));
    return {
      bbox_mercator: bbox,
      clusters: [],
      mode: "tiles",
      returned_clusters: 0,
      returned_tiles: tiles.length,
      summary,
      tiles,
      total_matching: matching.length,
      zoom,
    };
  }

  const clusters = clusterTiles(matching, bbox, zoom);
  return {
    bbox_mercator: bbox,
    clusters,
    mode: "clusters",
    returned_clusters: clusters.length,
    returned_tiles: 0,
    summary,
    tiles: [],
    total_matching: matching.length,
    zoom,
  };
}

export function loadMetadataDatasetIndex(datasetId: string, root = METADATA_DATASETS_ROOT): MetadataDatasetIndex {
  assertSafeSegment(datasetId, "dataset id");
  const path = join(root, datasetId, "dataset.json");
  if (!existsSync(path)) {
    throw new Error(`Missing metadata dataset index: ${path}`);
  }
  const parsed = JSON.parse(readFileSync(path, "utf8")) as MetadataDatasetIndex;
  if (parsed.format !== "crosswalk-jsonl-v1") {
    throw new Error(`Unsupported metadata dataset format: ${parsed.format}`);
  }
  if (!Array.isArray(parsed.shards)) {
    throw new Error(`Metadata dataset ${datasetId} has no shard list.`);
  }
  return parsed;
}

function findMetadataDataset(runName: string, exportName: string, root = METADATA_DATASETS_ROOT) {
  const dataset = listMetadataDatasets(root).find((entry) => entry.run_name === runName && entry.export_name === exportName);
  if (!dataset) {
    throw new Error(`Metadata dataset not found: ${runName}/${exportName}`);
  }
  return dataset;
}

function loadMetadataDatasetTiles(dataset: MetadataDatasetIndex, root = METADATA_DATASETS_ROOT) {
  const cached = tileCache.get(dataset.dataset_id);
  if (cached) return cached;

  const tiles: DatasetTile[] = [];
  for (const shard of dataset.shards) {
    const shardPath = resolveShardPath(root, dataset.dataset_id, shard.path);
    if (!existsSync(shardPath)) continue;
    for (const line of readFileSync(shardPath, "utf8").split(/\r?\n/)) {
      if (!line.trim()) continue;
      tiles.push(metadataImageToDatasetTile(dataset, JSON.parse(line) as MetadataDatasetImage));
    }
  }

  tileCache.set(dataset.dataset_id, tiles);
  return tiles;
}

function metadataImageToDatasetTile(dataset: MetadataDatasetIndex, image: MetadataDatasetImage): DatasetTile {
  const label = image.resolved_label.decision;
  const confidence = Number(image.resolved_label.confidence ?? 0);
  const relativePath = image.reconstruction.relative_path;
  return {
    bbox_mercator: image.bbox_mercator,
    city: image.city,
    clip_probability: confidence,
    col: image.col,
    combined_probability: confidence,
    heuristic_probability: confidence,
    has_image_asset: false,
    image_path: `/assets/metadata/${dataset.dataset_id}/${relativePath}`,
    image_id: image.image_id,
    label,
    labels: image.labels,
    predicted_label: label,
    relative_path: relativePath,
    resolved_label: image.resolved_label,
    review_source: image.resolved_label.source_id,
    row: image.row,
    scene_id: image.scene_id,
    selected: image.selected_for_training,
    split: image.split,
    status: image.review_state,
    tile_id: image.tile_id,
  };
}

export function appendMetadataLabelVotes(
  runName: string,
  exportName: string,
  predictions: MetadataLabelPrediction[],
  source: MetadataLabelSource,
  root = METADATA_DATASETS_ROOT,
) {
  const dataset = findMetadataDataset(runName, exportName, root);
  const byTileId = new Map(predictions.map((prediction) => [prediction.tile_id, prediction]));
  const updatedTiles: DatasetTile[] = [];
  const createdAt = new Date().toISOString();

  for (const shard of dataset.shards) {
    const path = resolveShardPath(root, dataset.dataset_id, shard.path);
    if (!existsSync(path)) continue;
    let changed = false;
    const rows = readFileSync(path, "utf8")
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
      .map((line) => {
        const image = JSON.parse(line) as MetadataDatasetImage;
        const prediction = byTileId.get(image.tile_id);
        if (!prediction) return image;
        const vote: ImageLabelVote = {
          confidence: prediction.confidence,
          created_at: createdAt,
          decision: prediction.decision,
          metadata: prediction.metadata,
          source,
          vote_id: `${source.source_id}:${createdAt}:${image.tile_id}`,
        };
        image.labels = [...(image.labels ?? []), vote];
        image.resolved_label = resolveLabelVotes(image.labels, createdAt);
        image.selected_for_training = prediction.decision !== "drop";
        if (source.kind === "human") image.review_state = prediction.decision === "drop" ? "dropped" : "reviewed";
        changed = true;
        updatedTiles.push(metadataImageToDatasetTile(dataset, image));
        return image;
      });
    if (changed) {
      writeFileSync(path, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
    }
  }

  if (updatedTiles.length > 0) {
    tileCache.delete(dataset.dataset_id);
  }
  return updatedTiles;
}

function metadataScenes(dataset: MetadataDatasetIndex): DatasetScene[] {
  const scenes = new Map<string, DatasetScene>();
  for (const shard of dataset.shards) {
    const sceneId = shard.scene_id ?? shard.shard_id;
    const existing = scenes.get(sceneId);
    if (existing) {
      existing.tile_count = Number(existing.tile_count ?? 0) + shard.tile_count;
      continue;
    }
    scenes.set(sceneId, {
      city: inferCity(sceneId),
      scene_id: sceneId,
      split: "all",
      tile_count: shard.tile_count,
    });
  }
  return [...scenes.values()].sort((a, b) => a.city.localeCompare(b.city) || a.scene_id.localeCompare(b.scene_id));
}

export function loadMetadataTilePage(
  datasetId: string,
  cursor: string | null,
  filters: MetadataTileFilters = {},
  root = METADATA_DATASETS_ROOT,
): MetadataTilePage {
  const dataset = loadMetadataDatasetIndex(datasetId, root);
  const limit = Math.max(1, Math.min(Number(filters.limit ?? DEFAULT_LIMIT) || DEFAULT_LIMIT, MAX_LIMIT));
  const start = parseCursor(cursor);
  const rows: MetadataDatasetImage[] = [];
  let nextCursor: string | null = null;

  for (let shardIndex = start.shardIndex; shardIndex < dataset.shards.length; shardIndex += 1) {
    const shard = dataset.shards[shardIndex];
    const shardPath = resolveShardPath(root, datasetId, shard.path);
    const lines = existsSync(shardPath) ? readFileSync(shardPath, "utf8").split(/\r?\n/) : [];
    const firstLine = shardIndex === start.shardIndex ? start.lineIndex : 0;
    for (let lineIndex = firstLine; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex]?.trim();
      if (!line) continue;
      const image = JSON.parse(line) as MetadataDatasetImage;
      if (!matchesFilters(image, filters)) continue;
      rows.push(image);
      if (rows.length >= limit) {
        nextCursor = encodeCursor({ shardIndex, lineIndex: lineIndex + 1 });
        return { dataset, rows, next_cursor: nextCursor, returned_count: rows.length };
      }
    }
  }

  return { dataset, rows, next_cursor: null, returned_count: rows.length };
}

function matchesFilters(image: MetadataDatasetImage, filters: MetadataTileFilters) {
  if (filters.city && image.city !== filters.city) return false;
  if (filters.split && image.split !== filters.split) return false;
  if (filters.review_state && image.review_state !== filters.review_state) return false;
  if (filters.label && image.resolved_label.decision !== filters.label) return false;
  if (typeof filters.selected === "boolean" && image.selected_for_training !== filters.selected) return false;
  return true;
}

function resolveLabelVotes(labels: ImageLabelVote[], updatedAt: string): ResolvedLabel {
  const valid = labels.filter((label) => label.decision && label.source);
  const humanVotes = valid.filter((label) => label.source.kind === "human");
  if (humanVotes.length) {
    const winning = newestVote(humanVotes);
    return resolvedFromVote(winning, "human_override", updatedAt);
  }
  const winning = [...valid].sort(compareVotes).at(-1);
  if (!winning) {
    return {
      decision: "drop",
      resolved_by: "priority",
      source_id: "none",
      source_kind: "model",
      updated_at: updatedAt,
    };
  }
  return resolvedFromVote(winning, "priority", updatedAt);
}

function compareVotes(left: ImageLabelVote, right: ImageLabelVote) {
  return (
    Number(left.source.priority ?? 0) - Number(right.source.priority ?? 0) ||
    Number(left.confidence ?? 0) - Number(right.confidence ?? 0) ||
    left.created_at.localeCompare(right.created_at)
  );
}

function newestVote(votes: ImageLabelVote[]) {
  return [...votes].sort((left, right) => left.created_at.localeCompare(right.created_at)).at(-1) ?? votes[0];
}

function resolvedFromVote(vote: ImageLabelVote, resolvedBy: ResolvedLabel["resolved_by"], updatedAt: string): ResolvedLabel {
  return {
    confidence: vote.confidence,
    decision: vote.decision,
    resolved_by: resolvedBy,
    source_id: vote.source.source_id,
    source_kind: vote.source.kind,
    updated_at: updatedAt,
  };
}

function bboxToRect(value: DatasetTile["bbox_mercator"]) {
  if (!value) return null;
  if (Array.isArray(value)) {
    const [minX, minY, maxX, maxY] = value.map(Number);
    if ([minX, minY, maxX, maxY].every(Number.isFinite)) {
      return [minX, minY, maxX, maxY] as [number, number, number, number];
    }
  }
  return null;
}

function rectIntersects(rect: [number, number, number, number], bbox: [number, number, number, number]) {
  return rect[0] <= bbox[2] && rect[2] >= bbox[0] && rect[1] <= bbox[3] && rect[3] >= bbox[1];
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
        bbox_mercator: [cellMinX, cellMinY, cellMaxX, cellMaxY],
        center_mercator: [(cellMinX + cellMaxX) / 2, (cellMinY + cellMaxY) / 2],
        cities: {},
        crosswalk: 0,
        id,
        labels: {},
        no_crosswalk: 0,
        splits: {},
        total: 0,
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

function inferCity(sceneId: string) {
  const prefix = sceneId.split("-r")[0] || sceneId;
  return prefix
    .split("-")
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function parseCursor(cursor: string | null): Cursor {
  if (!cursor) return { shardIndex: 0, lineIndex: 0 };
  const [shardIndex, lineIndex] = cursor.split(":").map((value) => Number(value));
  if (!Number.isInteger(shardIndex) || shardIndex < 0 || !Number.isInteger(lineIndex) || lineIndex < 0) {
    throw new Error(`Invalid metadata dataset cursor: ${cursor}`);
  }
  return { shardIndex, lineIndex };
}

function encodeCursor(cursor: Cursor) {
  return `${cursor.shardIndex}:${cursor.lineIndex}`;
}

function resolveShardPath(root: string, datasetId: string, relativePath: string) {
  if (relativePath.startsWith("/") || relativePath.includes("..")) {
    throw new Error(`Unsafe metadata shard path: ${relativePath}`);
  }
  const datasetRoot = resolve(root, datasetId);
  const path = resolve(datasetRoot, normalize(relativePath));
  if (!path.startsWith(datasetRoot)) {
    throw new Error(`Metadata shard path escapes dataset root: ${relativePath}`);
  }
  return path;
}

function assertSafeSegment(value: string, label: string) {
  if (!/^[a-zA-Z0-9._-]+$/.test(value)) {
    throw new Error(`Invalid ${label}: ${value}`);
  }
}
