import type { BrowserLabelSuggestion, DatasetTile } from "./types";

export type ScanSceneRequest = {
  scene_id: string;
  latitude: number;
  longitude: number;
  size_m: number;
  image_px: number;
  tile_size_m: number;
};

export type ScanHealth = {
  ok: boolean;
  ready: boolean;
  warming: boolean;
  model: string;
  device: string;
  busy: boolean;
};

export type ScanJobStatus = {
  job_id: string;
  scene_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  stage: string;
  total: number;
  done: number;
  current_tile_id: string | null;
  results: Record<string, BrowserLabelSuggestion>;
  error: string | null;
  created_at: string;
  updated_at: string;
};

type StartScanResponse = {
  job_id: string;
  status: string;
};

function withPath(baseUrl: string, path: string) {
  return new URL(path, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`).toString();
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchScanHealth(baseUrl: string) {
  const response = await fetch(withPath(baseUrl, "/health"));
  return readJson<ScanHealth>(response);
}

export async function warmupScanBackend(baseUrl: string) {
  const response = await fetch(withPath(baseUrl, "/warmup"), {
    method: "POST",
  });
  return readJson<ScanHealth>(response);
}

export async function startSceneScan(
  baseUrl: string,
  scene: ScanSceneRequest,
  tiles: DatasetTile[],
  threshold: number,
) {
  const response = await fetch(withPath(baseUrl, "/scan/start"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      scene,
      threshold,
      tiles: tiles.map((tile) => ({
        tile_id: tile.tile_id,
        row: tile.row,
        col: tile.col,
        bbox_mercator: Array.isArray(tile.bbox_mercator) ? tile.bbox_mercator : [],
        relative_path: tile.relative_path,
      })),
    }),
  });
  return readJson<StartScanResponse>(response);
}

export async function fetchSceneScan(baseUrl: string, jobId: string) {
  const response = await fetch(withPath(baseUrl, `/scan/${jobId}`));
  return readJson<ScanJobStatus>(response);
}

export async function cancelSceneScan(baseUrl: string, jobId: string) {
  const response = await fetch(withPath(baseUrl, `/scan/${jobId}/cancel`), {
    method: "POST",
  });
  return readJson<{ ok: boolean }>(response);
}
