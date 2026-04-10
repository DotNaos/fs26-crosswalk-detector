import { useCallback, useMemo, useRef, useState } from "react";
import { cancelSceneScan, fetchScanHealth, fetchSceneScan, startSceneScan, type ScanHealth, type ScanSceneRequest, warmupScanBackend } from "../scan-api";
import type { BrowserLabelSuggestion, DatasetTile, RealDatasetConfig } from "../types";

type LabelerStatus = "idle" | "connecting" | "ready" | "running" | "error";

type ProgressState = {
  done: number;
  total: number;
  currentTileId?: string;
};

type RunSceneLabelingOptions = {
  onSuggestion?: (suggestion: BrowserLabelSuggestion, index: number, total: number) => void;
  shouldAbort?: () => boolean;
};

type RemoteCrosswalkLabelerArgs = {
  backendUrl: string;
  scenes: RealDatasetConfig["scenes"];
  tileSizeM: number;
};

export type RemoteCrosswalkLabelerHandle = {
  defaultPromptText: string;
  status: LabelerStatus;
  error: string | null;
  progress: ProgressState;
  lastBatchCount: number;
  health: ScanHealth | null;
  ensureReady: () => Promise<void>;
  runSceneLabeling: (
    tiles: DatasetTile[],
    promptsText: string,
    threshold: number,
    options?: RunSceneLabelingOptions,
  ) => Promise<Record<string, BrowserLabelSuggestion>>;
};

const DEFAULT_PROMPT_TEXT = "server-side hybrid scan";

export function useRemoteCrosswalkLabeler({ backendUrl, scenes, tileSizeM }: RemoteCrosswalkLabelerArgs) {
  const [status, setStatus] = useState<LabelerStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressState>({ done: 0, total: 0 });
  const [lastBatchCount, setLastBatchCount] = useState(0);
  const [health, setHealth] = useState<ScanHealth | null>(null);
  const activeJobIdRef = useRef<string | null>(null);

  const ensureReady = useCallback(async () => {
    setStatus("connecting");
    setError(null);
    try {
      let nextHealth = await fetchScanHealth(backendUrl);
      if (!nextHealth.ready) {
        nextHealth = await warmupScanBackend(backendUrl);
      }
      setHealth(nextHealth);
      if (nextHealth.ready) {
        setStatus("ready");
        return;
      }

      const timeoutAt = window.performance.now() + 180_000;
      for (;;) {
        await new Promise((resolve) => window.setTimeout(resolve, 1_000));
        nextHealth = await fetchScanHealth(backendUrl);
        setHealth(nextHealth);
        if (nextHealth.ready) {
          setStatus("ready");
          return;
        }
        if (window.performance.now() > timeoutAt) {
          throw new Error("Scan backend did not finish loading its models in time.");
        }
      }
    } catch (reason) {
      setStatus("error");
      setError(String(reason));
      throw reason;
    }
  }, [backendUrl]);

  const runSceneLabeling = useCallback(
    async (tiles: DatasetTile[], _promptsText: string, threshold: number, options?: RunSceneLabelingOptions) => {
      if (!tiles.length) return {};
      const sceneId = tiles[0]?.scene_id;
      const scene = scenes.find((entry) => entry.scene_id === sceneId);
      if (!scene) {
        throw new Error(`Unknown scene for scan: ${sceneId}`);
      }

      await ensureReady();
      setStatus("running");
      setError(null);
      setLastBatchCount(0);
      setProgress({ done: 0, total: tiles.length });

      const startScene: ScanSceneRequest = {
        scene_id: scene.scene_id,
        latitude: scene.latitude,
        longitude: scene.longitude,
        size_m: scene.size_m,
        image_px: scene.image_px,
        tile_size_m: tileSizeM,
      };
      const { job_id } = await startSceneScan(backendUrl, startScene, tiles, threshold);
      activeJobIdRef.current = job_id;

      const seenTileIds = new Set<string>();
      let lastDone = 0;
      for (;;) {
        if (options?.shouldAbort?.()) {
          if (activeJobIdRef.current) {
            await cancelSceneScan(backendUrl, activeJobIdRef.current).catch(() => undefined);
          }
          break;
        }

        const job = await fetchSceneScan(backendUrl, job_id);
        setProgress({
          done: job.done,
          total: job.total,
          currentTileId: job.current_tile_id ?? undefined,
        });
        const sortedSuggestions = Object.values(job.results);
        for (const suggestion of sortedSuggestions) {
          if (seenTileIds.has(suggestion.tile_id)) continue;
          seenTileIds.add(suggestion.tile_id);
          options?.onSuggestion?.(suggestion, seenTileIds.size - 1, job.total);
        }
        if (job.done > lastDone) {
          lastDone = job.done;
          setLastBatchCount(lastDone);
        }

        if (job.status === "completed") {
          setStatus("ready");
          activeJobIdRef.current = null;
          return job.results;
        }
        if (job.status === "failed") {
          setStatus("error");
          activeJobIdRef.current = null;
          setError(job.error ?? "Remote scan failed.");
          throw new Error(job.error ?? "Remote scan failed.");
        }
        if (job.status === "cancelled") {
          setStatus("ready");
          activeJobIdRef.current = null;
          return job.results;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 450));
      }

      setStatus("ready");
      activeJobIdRef.current = null;
      return {};
    },
    [backendUrl, ensureReady, scenes, tileSizeM],
  );

  return useMemo(
    () => ({
      defaultPromptText: DEFAULT_PROMPT_TEXT,
      status,
      error,
      progress,
      lastBatchCount,
      health,
      ensureReady,
      runSceneLabeling,
    }),
    [ensureReady, error, health, lastBatchCount, progress, runSceneLabeling, status],
  );
}
