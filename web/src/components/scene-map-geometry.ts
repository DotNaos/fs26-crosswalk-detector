import type { AutopilotBvhCell } from "../autopilot-planner";

export const DEFAULT_MAP_CENTER: [number, number] = [46.8, 8.25];
export const GRID_ZOOM_THRESHOLD = 16;
export const ROAD_CLUSTER_MIN_ZOOM = 14;
export const ROAD_CLUSTER_VIEWPORT_ZOOM = 14;
export const ROAD_CLUSTER_REQUEST_ZOOM = 19;
export const REMOTE_SCAN_THRESHOLD = 0.5;

const WEB_MERCATOR_WORLD_MIN = -20037508.342789244;
const WEB_MERCATOR_WORLD_MAX = 20037508.342789244;
const WEB_MERCATOR_WORLD_SIZE = WEB_MERCATOR_WORLD_MAX - WEB_MERCATOR_WORLD_MIN;

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function dashArrayForZoom(zoom: number, on: number, off: number, minimum = 2) {
  const scale = clamp(0.34 + (zoom - GRID_ZOOM_THRESHOLD) * 0.32, 0.22, 1.8);
  const dashOn = Math.max(minimum, Math.round(on * scale));
  const dashOff = Math.max(minimum, Math.round(off * scale));
  return `${dashOn} ${dashOff}`;
}

type MapLayerVisibilityInput = {
  hasAutopilotPlan?: boolean;
  hasSceneTiles?: boolean;
  hasSelectedScene?: boolean;
  zoom: number;
};

export function shouldShowFineGrid({ hasSceneTiles, hasSelectedScene, zoom }: MapLayerVisibilityInput) {
  return Boolean(hasSelectedScene && hasSceneTiles && zoom >= GRID_ZOOM_THRESHOLD);
}

export function shouldShowAutopilotSegmentation(input: MapLayerVisibilityInput) {
  return input.zoom >= ROAD_CLUSTER_MIN_ZOOM;
}

export function shouldUseViewportRoadGrid(zoom: number) {
  return zoom >= ROAD_CLUSTER_VIEWPORT_ZOOM;
}

export function roadClusterRequestZoomForMapZoom(_zoom: number) {
  return ROAD_CLUSTER_REQUEST_ZOOM;
}

export function bvhLineWeight(depth: number) {
  if (depth <= 0) return 4.8;
  if (depth <= 1) return 4;
  if (depth <= 2) return 3.2;
  if (depth <= 3) return 2.2;
  return 1.4;
}

export function bvhLineColor(cell: AutopilotBvhCell) {
  if (cell.depth <= 1) return "#fde047";
  if (cell.status === "urban") return "#facc15";
  if (cell.status === "candidate") return "#fbbf24";
  return "#f59e0b";
}

function webMercatorTileSizeM(zoom: number) {
  const tileZoom = Math.round(clamp(zoom, 0, 22));
  return WEB_MERCATOR_WORLD_SIZE / 2 ** tileZoom;
}

function webMercatorZoomForCellSize(sizeM: number) {
  return Math.round(clamp(Math.log2(WEB_MERCATOR_WORLD_SIZE / Math.max(1, sizeM)), 0, 22));
}

export function snapBboxToWebMercatorTileGrid(
  bbox: [number, number, number, number],
  sizeM: number,
): [number, number, number, number] {
  const tileSizeM = webMercatorTileSizeM(webMercatorZoomForCellSize(sizeM));
  const snapDown = (value: number) => WEB_MERCATOR_WORLD_MIN + Math.floor((value - WEB_MERCATOR_WORLD_MIN) / tileSizeM) * tileSizeM;
  const snapUp = (value: number) => WEB_MERCATOR_WORLD_MIN + Math.ceil((value - WEB_MERCATOR_WORLD_MIN) / tileSizeM) * tileSizeM;
  return [snapDown(bbox[0]), snapDown(bbox[1]), snapUp(bbox[2]), snapUp(bbox[3])];
}
