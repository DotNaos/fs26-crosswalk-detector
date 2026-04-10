import type { DatasetScene, DatasetTile, TileBBoxMercator } from "./types";

const EARTH_RADIUS_M = 6_378_137;

export type MercatorRect = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

export function sceneLabel(scene: DatasetScene) {
  return `${scene.city} · ${scene.split}`;
}

export function tileTone(tile: DatasetTile) {
  const status = tile.status.toLowerCase();
  if (!tile.selected || status.includes("drop") || status.includes("bucket")) {
    return "dropped";
  }
  if (tile.label === "crosswalk") return "crosswalk";
  if (tile.label === "no_crosswalk") return "no_crosswalk";
  return "unknown";
}

export function tileActionLabel(tile: DatasetTile) {
  if (!tile.selected) return "Dropped";
  if (tile.label === "crosswalk") return "Crosswalk";
  if (tile.label === "no_crosswalk") return "No crosswalk";
  return tile.label || "Unknown";
}

export function formatProbability(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return value.toFixed(3);
}

export function sortScenes(a: DatasetScene, b: DatasetScene) {
  return a.city.localeCompare(b.city) || a.split.localeCompare(b.split) || a.scene_id.localeCompare(b.scene_id);
}

export function mercatorToLatLng(x: number, y: number): [number, number] {
  const lng = (x / EARTH_RADIUS_M) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / EARTH_RADIUS_M)) - Math.PI / 2) * (180 / Math.PI);
  return [lat, lng];
}

export function latLngToMercator(latitude: number, longitude: number) {
  return {
    x: EARTH_RADIUS_M * (longitude * Math.PI / 180),
    y: EARTH_RADIUS_M * Math.log(Math.tan(Math.PI / 4 + (latitude * Math.PI / 180) / 2)),
  };
}

export function bboxToMercatorRect(bbox: TileBBoxMercator): MercatorRect | null {
  if (!bbox) return null;

  if (Array.isArray(bbox)) {
    const [minX, minY, maxX, maxY] = bbox;
    return { minX, minY, maxX, maxY };
  }

  return {
    minX: Number(bbox.min_x ?? bbox.left ?? 0),
    minY: Number(bbox.min_y ?? bbox.bottom ?? 0),
    maxX: Number(bbox.max_x ?? bbox.right ?? 0),
    maxY: Number(bbox.max_y ?? bbox.top ?? 0),
  };
}

export function bboxToLatLngBounds(bbox: TileBBoxMercator) {
  const rect = bboxToMercatorRect(bbox);
  if (!rect) return null;

  const { minX, minY, maxX, maxY } = rect;
  return [mercatorToLatLng(minX, minY), mercatorToLatLng(maxX, maxY)] as [[number, number], [number, number]];
}

export function bboxCenterLatLng(bbox: TileBBoxMercator) {
  const bounds = bboxToLatLngBounds(bbox);
  if (!bounds) return null;
  return [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2] as [number, number];
}

export function bboxAverageSizeM(bbox: TileBBoxMercator) {
  const rect = bboxToMercatorRect(bbox);
  if (!rect) return 0;

  const { minX, minY, maxX, maxY } = rect;

  return (Math.abs(maxX - minX) + Math.abs(maxY - minY)) / 2;
}

export function sceneMercatorCenter(scene: DatasetScene) {
  const latitude = Number(scene.latitude ?? 0);
  const longitude = Number(scene.longitude ?? 0);
  if (!latitude || !longitude) return null;
  return latLngToMercator(latitude, longitude);
}

export function circleIntersectsMercatorRect(rect: MercatorRect, centerX: number, centerY: number, radiusM: number) {
  const nearestX = Math.max(rect.minX, Math.min(centerX, rect.maxX));
  const nearestY = Math.max(rect.minY, Math.min(centerY, rect.maxY));
  return Math.hypot(nearestX - centerX, nearestY - centerY) <= radiusM;
}

export function sceneLatLngBounds(scene: DatasetScene) {
  const latitude = Number(scene.latitude ?? 0);
  const longitude = Number(scene.longitude ?? 0);
  const sizeM = Number(scene.size_m ?? 0);
  if (!latitude || !longitude || !sizeM) return null;

  const centerX = EARTH_RADIUS_M * (longitude * Math.PI / 180);
  const centerY = EARTH_RADIUS_M * Math.log(Math.tan(Math.PI / 4 + (latitude * Math.PI / 180) / 2));
  const half = sizeM / 2;
  return [
    mercatorToLatLng(centerX - half, centerY - half),
    mercatorToLatLng(centerX + half, centerY + half),
  ] as [[number, number], [number, number]];
}
