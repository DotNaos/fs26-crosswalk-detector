import type { DatasetListEntry, DatasetScene, DatasetSummary, DatasetTile, DatasetViewportCluster, DatasetViewportPayload, MetadataDatasetImage, MetadataDatasetShard } from "./types";

const STATIC_DATASET_INDEX = "/static-datasets/index.json";
const TILE_LIMIT = 5_000;

type StaticDatasetManifest = {
  datasets: Array<{
    dataset_id: string;
    display_name?: string;
    export_name: string;
    path: string;
    run_name: string;
    selected_count: number;
    tile_count: number;
  }>;
};

type StaticDatasetShard = MetadataDatasetShard & {
  bbox_mercator: [number, number, number, number];
  center_mercator: [number, number];
  cities: Record<string, number>;
  labels: Record<string, number>;
  selected_count: number;
  splits: Record<string, number>;
};

type StaticDatasetIndex = {
  city_counts?: Record<string, number>;
  dataset_id: string;
  display_name?: string;
  export_name: string;
  format: "crosswalk-static-jsonl-v1";
  label_counts?: Record<string, number>;
  run_name: string;
  scenes?: DatasetScene[];
  selected_count: number;
  shards: StaticDatasetShard[];
  split_counts?: Record<string, number>;
  tile_count: number;
};

const manifestCache = new Map<string, StaticDatasetManifest | null>();
const datasetCache = new Map<string, StaticDatasetIndex>();
const shardCache = new Map<string, DatasetTile[]>();

type ViewportInput = {
  bboxMercator: [number, number, number, number];
  city?: string;
  label?: string;
  limit?: number;
  split?: string;
  zoom: number;
};

export async function listStaticDatasetEntries(): Promise<DatasetListEntry[]> {
  const manifest = await loadStaticManifest();
  if (!manifest) return [];
  return manifest.datasets.map((dataset) => ({
    display_name: dataset.display_name,
    export_name: dataset.export_name,
    path: dataset.path,
    run_name: dataset.run_name,
    selected_count: dataset.selected_count,
    tile_count: dataset.tile_count,
  }));
}

export async function loadStaticDatasetSummary(runName: string, exportName: string): Promise<DatasetSummary> {
  const { index } = await findStaticDataset(runName, exportName);
  const labelCounts = index.label_counts ?? {};
  return {
    display_name: index.display_name,
    dropped_tiles: index.tile_count - index.selected_count,
    export_name: index.export_name,
    label_counts: labelCounts,
    run_name: index.run_name,
    scenes: index.scenes ?? scenesFromShards(index.shards),
    selected_crosswalk: labelCounts.crosswalk ?? 0,
    selected_no_crosswalk: labelCounts.no_crosswalk ?? 0,
    selected_tiles: index.selected_count,
    total_tiles: index.tile_count,
  };
}

export async function loadStaticDatasetViewport(runName: string, exportName: string, input: ViewportInput): Promise<DatasetViewportPayload> {
  const { basePath, index } = await findStaticDataset(runName, exportName);
  const limit = Math.max(100, Math.min(Number(input.limit ?? 1400) || 1400, TILE_LIMIT));
  const summary = await loadStaticDatasetSummary(runName, exportName);
  const matchingShards = index.shards.filter((shard) => shardMatchesViewport(shard, input));
  const estimatedMatching = matchingShards.reduce((sum, shard) => sum + countForFilters(shard, input), 0);

  if (estimatedMatching > Math.min(250, limit) && input.zoom < 14.5) {
    const clusters = clusterShards(matchingShards, input.bboxMercator, input.zoom, input);
    return {
      bbox_mercator: input.bboxMercator,
      clusters,
      mode: "clusters",
      returned_clusters: clusters.length,
      returned_tiles: 0,
      summary,
      tiles: [],
      total_matching: estimatedMatching,
      zoom: input.zoom,
    };
  }

  const tiles: DatasetTile[] = [];
  let totalMatching = 0;
  for (const shard of matchingShards) {
    const shardTiles = await loadStaticShardTiles(basePath, index, shard);
    for (const tile of shardTiles) {
      if (!tileMatchesViewport(tile, input)) continue;
      totalMatching += 1;
      if (tiles.length < limit) tiles.push(tile);
    }
  }
  tiles.sort((left, right) => left.tile_id.localeCompare(right.tile_id));
  return {
    bbox_mercator: input.bboxMercator,
    clusters: [],
    mode: "tiles",
    returned_clusters: 0,
    returned_tiles: tiles.length,
    summary,
    tiles,
    total_matching: totalMatching,
    zoom: input.zoom,
  };
}

export function isStaticDatasetEntry(entry: DatasetListEntry | undefined) {
  return Boolean(entry?.path?.startsWith("/static-datasets/"));
}

async function loadStaticManifest() {
  if (manifestCache.has(STATIC_DATASET_INDEX)) return manifestCache.get(STATIC_DATASET_INDEX) ?? null;
  try {
    const response = await fetch(STATIC_DATASET_INDEX);
    if (!response.ok) {
      manifestCache.set(STATIC_DATASET_INDEX, null);
      return null;
    }
    const manifest = (await response.json()) as StaticDatasetManifest;
    manifestCache.set(STATIC_DATASET_INDEX, manifest);
    return manifest;
  } catch {
    manifestCache.set(STATIC_DATASET_INDEX, null);
    return null;
  }
}

async function findStaticDataset(runName: string, exportName: string) {
  const manifest = await loadStaticManifest();
  const entry = manifest?.datasets.find((dataset) => dataset.run_name === runName && dataset.export_name === exportName);
  if (!entry) throw new Error(`Static dataset not found: ${runName}/${exportName}`);
  const index = await loadStaticDatasetIndex(entry.path);
  return { basePath: entry.path.replace(/\/+$/, ""), index };
}

async function loadStaticDatasetIndex(basePath: string) {
  const normalizedBasePath = basePath.replace(/\/+$/, "");
  const cached = datasetCache.get(normalizedBasePath);
  if (cached) return cached;
  const response = await fetch(`${normalizedBasePath}/dataset.json`);
  if (!response.ok) throw new Error(await response.text());
  const index = (await response.json()) as StaticDatasetIndex;
  datasetCache.set(normalizedBasePath, index);
  return index;
}

async function loadStaticShardTiles(basePath: string, index: StaticDatasetIndex, shard: StaticDatasetShard) {
  const cacheKey = `${index.dataset_id}:${shard.shard_id}`;
  const cached = shardCache.get(cacheKey);
  if (cached) return cached;
  const text = await fetchTextMaybeGzip(`${basePath.replace(/\/+$/, "")}/${shard.path}`);
  const tiles = text
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => staticImageToDatasetTile(index, JSON.parse(line) as MetadataDatasetImage));
  shardCache.set(cacheKey, tiles);
  return tiles;
}

async function fetchTextMaybeGzip(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  if (!url.endsWith(".gz") || response.headers.get("content-encoding") === "gzip") {
    return response.text();
  }
  const stream = response.body;
  const DecompressionStreamCtor = globalThis.DecompressionStream;
  if (!stream || !DecompressionStreamCtor) {
    throw new Error("This browser cannot read compressed static dataset shards.");
  }
  return new Response(stream.pipeThrough(new DecompressionStreamCtor("gzip"))).text();
}

function staticImageToDatasetTile(dataset: StaticDatasetIndex, image: MetadataDatasetImage): DatasetTile {
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
    image_id: image.image_id,
    image_path: `/assets/metadata/${dataset.dataset_id}/${relativePath}`,
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

function shardMatchesViewport(shard: StaticDatasetShard, input: ViewportInput) {
  if (!rectIntersects(shard.bbox_mercator, input.bboxMercator)) return false;
  if (input.city && input.city !== "all" && !shard.cities[input.city]) return false;
  if (input.split && input.split !== "all" && !shard.splits[input.split]) return false;
  if (input.label && input.label !== "all" && !shard.labels[input.label]) return false;
  return true;
}

function tileMatchesViewport(tile: DatasetTile, input: ViewportInput) {
  const rect = bboxToRect(tile.bbox_mercator);
  if (!rect || !rectIntersects(rect, input.bboxMercator)) return false;
  if (input.city && input.city !== "all" && tile.city !== input.city) return false;
  if (input.split && input.split !== "all" && tile.split !== input.split) return false;
  if (input.label && input.label !== "all" && tile.label !== input.label) return false;
  return true;
}

function countForFilters(shard: StaticDatasetShard, input: ViewportInput) {
  if (input.label && input.label !== "all") return shard.labels[input.label] ?? 0;
  if (input.city && input.city !== "all") return shard.cities[input.city] ?? 0;
  if (input.split && input.split !== "all") return shard.splits[input.split] ?? 0;
  return shard.tile_count;
}

function clusterShards(shards: StaticDatasetShard[], bbox: [number, number, number, number], zoom: number, input: ViewportInput): DatasetViewportCluster[] {
  const width = Math.max(1, bbox[2] - bbox[0]);
  const height = Math.max(1, bbox[3] - bbox[1]);
  const grid = Math.max(8, Math.min(48, Math.round(10 + zoom * 1.6)));
  const cols = grid;
  const rows = Math.max(4, Math.round(grid * (height / width)));
  const clusters = new Map<string, DatasetViewportCluster>();

  for (const shard of shards) {
    const [centerX, centerY] = shard.center_mercator;
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
    const count = countForFilters(shard, input);
    cluster.total += count;
    cluster.crosswalk += input.label && input.label !== "all" && input.label !== "crosswalk" ? 0 : (shard.labels.crosswalk ?? 0);
    cluster.no_crosswalk += input.label && input.label !== "all" && input.label !== "no_crosswalk" ? 0 : (shard.labels.no_crosswalk ?? 0);
    incrementCounts(cluster.labels, shard.labels);
    incrementCounts(cluster.splits, shard.splits);
    incrementCounts(cluster.cities, shard.cities);
  }

  return [...clusters.values()].sort((left, right) => right.total - left.total);
}

function scenesFromShards(shards: StaticDatasetShard[]): DatasetScene[] {
  const scenes = new Map<string, DatasetScene>();
  for (const shard of shards) {
    const sceneId = shard.scene_id ?? shard.shard_id;
    const existing = scenes.get(sceneId);
    if (existing) {
      existing.tile_count = Number(existing.tile_count ?? 0) + shard.tile_count;
      continue;
    }
    scenes.set(sceneId, {
      city: Object.entries(shard.cities).sort((left, right) => right[1] - left[1])[0]?.[0] ?? "Unknown",
      scene_id: sceneId,
      split: Object.entries(shard.splits).sort((left, right) => right[1] - left[1])[0]?.[0] ?? "all",
      tile_count: shard.tile_count,
    });
  }
  return [...scenes.values()].sort((left, right) => left.city.localeCompare(right.city) || left.scene_id.localeCompare(right.scene_id));
}

function incrementCounts(target: Record<string, number>, source: Record<string, number>) {
  for (const [key, value] of Object.entries(source)) {
    target[key] = (target[key] ?? 0) + value;
  }
}

function bboxToRect(value: DatasetTile["bbox_mercator"]) {
  if (!Array.isArray(value)) return null;
  const [minX, minY, maxX, maxY] = value.map(Number);
  if ([minX, minY, maxX, maxY].every(Number.isFinite)) return [minX, minY, maxX, maxY] as [number, number, number, number];
  return null;
}

function rectIntersects(rect: [number, number, number, number], bbox: [number, number, number, number]) {
  return rect[0] <= bbox[2] && rect[2] >= bbox[0] && rect[1] <= bbox[3] && rect[3] >= bbox[1];
}
