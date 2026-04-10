import { MAP_BASEMAPS } from "./map-basemaps";
import { latLngToMercator } from "./utils";
import type { DatasetTile, RealDatasetConfig } from "./types";

const TILE_SIZE_PX = 256;
const EARTH_RADIUS_M = 6_378_137;
const sceneRasterCache = new Map<string, Promise<Map<string, string>>>();

function mercatorToWorldPixel(x: number, y: number, zoom: number) {
  const worldSize = TILE_SIZE_PX * 2 ** zoom;
  return {
    x: ((x + Math.PI * EARTH_RADIUS_M) / (2 * Math.PI * EARTH_RADIUS_M)) * worldSize,
    y: ((Math.PI * EARTH_RADIUS_M - y) / (2 * Math.PI * EARTH_RADIUS_M)) * worldSize,
  };
}

function resolveSceneZoom(scene: RealDatasetConfig["scenes"][number]) {
  const metersPerPixel = scene.size_m / scene.image_px;
  const latitudeScale = Math.cos((scene.latitude * Math.PI) / 180);
  const numerator = latitudeScale * 2 * Math.PI * EARTH_RADIUS_M;
  const zoom = Math.log2(numerator / (TILE_SIZE_PX * metersPerPixel));
  return Math.max(16, Math.min(19, Math.round(zoom)));
}

async function loadTileImage(url: string) {
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.decoding = "async";
  image.referrerPolicy = "no-referrer";

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error(`Failed to load map tile: ${url}`));
    image.src = url;
  });

  return image;
}

async function buildSceneCanvas(scene: RealDatasetConfig["scenes"][number]) {
  const zoom = resolveSceneZoom(scene);
  const center = latLngToMercator(scene.latitude, scene.longitude);
  const half = scene.size_m / 2;
  const min = mercatorToWorldPixel(center.x - half, center.y + half, zoom);
  const max = mercatorToWorldPixel(center.x + half, center.y - half, zoom);
  const minTileX = Math.floor(min.x / TILE_SIZE_PX);
  const maxTileX = Math.floor(max.x / TILE_SIZE_PX);
  const minTileY = Math.floor(min.y / TILE_SIZE_PX);
  const maxTileY = Math.floor(max.y / TILE_SIZE_PX);

  const stitchCanvas = document.createElement("canvas");
  stitchCanvas.width = (maxTileX - minTileX + 1) * TILE_SIZE_PX;
  stitchCanvas.height = (maxTileY - minTileY + 1) * TILE_SIZE_PX;
  const stitchContext = stitchCanvas.getContext("2d");
  if (!stitchContext) {
    throw new Error("Failed to get 2D context for scene stitching.");
  }

  const imagery = MAP_BASEMAPS.swisstopo;
  const tileRequests: Array<Promise<{ image: HTMLImageElement; tileX: number; tileY: number }>> = [];
  for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
    for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
      const url = imagery.url
        .replace("{z}", String(zoom))
        .replace("{x}", String(tileX))
        .replace("{y}", String(tileY))
        .replace("{s}", "a");
      tileRequests.push(loadTileImage(url).then((image) => ({ image, tileX, tileY })));
    }
  }

  const resolvedTiles = await Promise.all(tileRequests);
  for (const { image, tileX, tileY } of resolvedTiles) {
    stitchContext.drawImage(image, (tileX - minTileX) * TILE_SIZE_PX, (tileY - minTileY) * TILE_SIZE_PX, TILE_SIZE_PX, TILE_SIZE_PX);
  }

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = scene.image_px;
  cropCanvas.height = scene.image_px;
  const cropContext = cropCanvas.getContext("2d");
  if (!cropContext) {
    throw new Error("Failed to get 2D context for scene crop.");
  }

  cropContext.drawImage(
    stitchCanvas,
    min.x - minTileX * TILE_SIZE_PX,
    min.y - minTileY * TILE_SIZE_PX,
    max.x - min.x,
    max.y - min.y,
    0,
    0,
    scene.image_px,
    scene.image_px,
  );

  return cropCanvas;
}

async function canvasToObjectUrl(canvas: HTMLCanvasElement) {
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((value) => (value ? resolve(value) : reject(new Error("Failed to encode tile image."))), "image/jpeg", 0.92);
  });
  return URL.createObjectURL(blob);
}

async function sliceSceneTiles(scene: RealDatasetConfig["scenes"][number], tiles: DatasetTile[]) {
  const sceneCanvas = await buildSceneCanvas(scene);
  const tileCanvas = document.createElement("canvas");
  const gridSize = Math.max(1, Math.max(...tiles.map((tile) => Math.max(tile.row, tile.col))) + 1);
  const tilePx = Math.round(scene.image_px / gridSize);
  tileCanvas.width = tilePx;
  tileCanvas.height = tilePx;
  const tileContext = tileCanvas.getContext("2d");
  if (!tileContext) {
    throw new Error("Failed to get 2D context for tile extraction.");
  }

  const objectUrls = new Map<string, string>();
  for (const tile of tiles) {
    tileContext.clearRect(0, 0, tilePx, tilePx);
    tileContext.drawImage(sceneCanvas, tile.col * tilePx, tile.row * tilePx, tilePx, tilePx, 0, 0, tilePx, tilePx);
    objectUrls.set(tile.tile_id, await canvasToObjectUrl(tileCanvas));
  }
  return objectUrls;
}

export function clearSceneRasterCache() {
  sceneRasterCache.clear();
}

export async function ensureSceneTileImages(scene: RealDatasetConfig["scenes"][number], tiles: DatasetTile[]) {
  const cacheKey = `${scene.scene_id}:${scene.size_m}:${scene.image_px}`;
  const cached = sceneRasterCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const next = sliceSceneTiles(scene, tiles);
  sceneRasterCache.set(cacheKey, next);
  return next;
}
