import type { BrowserLabelSuggestion, DatasetScene, DatasetSummary, DatasetTile } from "./types";
import { bboxCenterLatLng, bboxToMercatorRect, mercatorToLatLng, sceneMercatorCenter } from "./utils";

export type ScanBatchSceneGeometry = {
  crs: "EPSG:3857";
  center_mercator: { x: number; y: number };
  center_latlon: { latitude: number; longitude: number };
  bbox_mercator: [number, number, number, number];
  bbox_latlon: {
    south_west: { latitude: number; longitude: number };
    north_east: { latitude: number; longitude: number };
  };
};

export type ScanBatchGrid = {
  tile_size_m: number;
  tile_count: number;
  rows: number;
  cols: number;
  min_row: number;
  max_row: number;
  min_col: number;
  max_col: number;
};

export type ScanBatchSelection = {
  mode: "circle";
  scan_radius_tiles: number;
  scan_radius_m: number;
  tile_count: number;
  center_mercator: { x: number; y: number };
  center_latlon: { latitude: number; longitude: number };
};

export type ScanBatchTileGeometry = {
  tile_id: string;
  row: number;
  col: number;
  bbox_mercator: [number, number, number, number];
  center_mercator: { x: number; y: number };
  center_latlon: { latitude: number; longitude: number };
  relative_path: string;
};

export type ScanBatchJob = {
  version: 1;
  created_at: string;
  dataset: {
    run_name: string;
    export_name: string;
  };
  scene: {
    scene_id: string;
    city: string;
    split: string;
    latitude: number;
    longitude: number;
    size_m: number;
    image_px: number;
    tile_size_m: number;
    geometry: ScanBatchSceneGeometry;
    grid: ScanBatchGrid;
  };
  selection: ScanBatchSelection;
  threshold: number;
  prompts_text: string;
  tiles: ScanBatchTileGeometry[];
};

export type ScanBatchResult = {
  version: 1;
  created_at: string;
  completed_at: string;
  job: ScanBatchJob;
  scanner: {
    detector_model: string;
    clip_model: string;
    device: string;
  };
  summary: {
    total: number;
    crosswalk: number;
    no_crosswalk: number;
  };
  tiles: Array<ScanBatchTileGeometry & BrowserLabelSuggestion>;
  results: Record<string, BrowserLabelSuggestion>;
};

function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function buildScanBatchJob(args: {
  summary: DatasetSummary;
  scene: DatasetScene;
  tileSizeM: number;
  scanRadiusTiles: number;
  threshold: number;
  promptsText: string;
  tiles: DatasetTile[];
}): ScanBatchJob {
  const mercatorCenter = sceneMercatorCenter(args.scene);
  if (!mercatorCenter) {
    throw new Error(`Scene ${args.scene.scene_id} is missing valid center coordinates.`);
  }

  const sizeM = Number(args.scene.size_m ?? 0);
  const imagePx = Number(args.scene.image_px ?? 0);
  const sceneBBoxMercator: [number, number, number, number] = [
    mercatorCenter.x - sizeM / 2,
    mercatorCenter.y - sizeM / 2,
    mercatorCenter.x + sizeM / 2,
    mercatorCenter.y + sizeM / 2,
  ];
  const southWest = mercatorToLatLng(sceneBBoxMercator[0], sceneBBoxMercator[1]);
  const northEast = mercatorToLatLng(sceneBBoxMercator[2], sceneBBoxMercator[3]);

  const rows = args.tiles.map((tile) => tile.row);
  const cols = args.tiles.map((tile) => tile.col);
  const minRow = Math.min(...rows);
  const maxRow = Math.max(...rows);
  const minCol = Math.min(...cols);
  const maxCol = Math.max(...cols);

  const tileGeometries: ScanBatchTileGeometry[] = args.tiles.map((tile) => {
    const rect = bboxToMercatorRect(tile.bbox_mercator);
    if (!rect) {
      throw new Error(`Tile ${tile.tile_id} is missing a valid bounding box.`);
    }
    const centerLatLng = bboxCenterLatLng(tile.bbox_mercator);
    if (!centerLatLng) {
      throw new Error(`Tile ${tile.tile_id} is missing a valid center coordinate.`);
    }
    return {
      tile_id: tile.tile_id,
      row: tile.row,
      col: tile.col,
      bbox_mercator: [rect.minX, rect.minY, rect.maxX, rect.maxY],
      center_mercator: {
        x: (rect.minX + rect.maxX) / 2,
        y: (rect.minY + rect.maxY) / 2,
      },
      center_latlon: {
        latitude: centerLatLng[0],
        longitude: centerLatLng[1],
      },
      relative_path: tile.relative_path,
    };
  });

  return {
    version: 1,
    created_at: new Date().toISOString(),
    dataset: {
      run_name: args.summary.run_name,
      export_name: args.summary.export_name,
    },
    scene: {
      scene_id: args.scene.scene_id,
      city: args.scene.city,
      split: args.scene.split,
      latitude: Number(args.scene.latitude ?? 0),
      longitude: Number(args.scene.longitude ?? 0),
      size_m: sizeM,
      image_px: imagePx,
      tile_size_m: args.tileSizeM,
      geometry: {
        crs: "EPSG:3857",
        center_mercator: mercatorCenter,
        center_latlon: {
          latitude: Number(args.scene.latitude ?? 0),
          longitude: Number(args.scene.longitude ?? 0),
        },
        bbox_mercator: sceneBBoxMercator,
        bbox_latlon: {
          south_west: { latitude: southWest[0], longitude: southWest[1] },
          north_east: { latitude: northEast[0], longitude: northEast[1] },
        },
      },
      grid: {
        tile_size_m: args.tileSizeM,
        tile_count: args.tiles.length,
        rows: maxRow - minRow + 1,
        cols: maxCol - minCol + 1,
        min_row: minRow,
        max_row: maxRow,
        min_col: minCol,
        max_col: maxCol,
      },
    },
    selection: {
      mode: "circle",
      scan_radius_tiles: args.scanRadiusTiles,
      scan_radius_m: args.scanRadiusTiles * args.tileSizeM,
      tile_count: args.tiles.length,
      center_mercator: mercatorCenter,
      center_latlon: {
        latitude: Number(args.scene.latitude ?? 0),
        longitude: Number(args.scene.longitude ?? 0),
      },
    },
    threshold: args.threshold,
    prompts_text: args.promptsText,
    tiles: tileGeometries,
  };
}

export function exportScanBatchJob(job: ScanBatchJob) {
  downloadJson(`${job.scene.scene_id}-scan-job.json`, job);
}

export async function readScanBatchResultFile(file: File): Promise<ScanBatchResult> {
  const text = await file.text();
  const parsed = JSON.parse(text) as ScanBatchResult;
  if (parsed.version !== 1 || !parsed.job?.scene?.scene_id || !parsed.results || !Array.isArray(parsed.tiles)) {
    throw new Error("This file is not a valid scan result export.");
  }
  return parsed;
}
