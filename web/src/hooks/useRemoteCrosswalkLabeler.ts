import { useCallback, useMemo, useRef, useState } from "react";
import { buildScanBatchJob } from "../scan-batch";
import {
  cancelRemoteScanJob,
  connectRemoteController,
  getRemoteScanJob,
  getRemoteScanResult,
  loadRemoteControllerSnapshot,
  startRemoteScanJob,
} from "../remote-api";
import type { BrowserLabelSuggestion, DatasetTile, RealDatasetConfig, RemoteControllerSnapshot } from "../types";

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
  runName: string;
  exportName: string;
  scenes: RealDatasetConfig["scenes"];
  tileSizeM: number;
};

export type RemoteCrosswalkLabelerHandle = {
  defaultPromptText: string;
  status: LabelerStatus;
  error: string | null;
  progress: ProgressState;
  lastBatchCount: number;
  controller: RemoteControllerSnapshot | null;
  ensureReady: () => Promise<void>;
  runSceneLabeling: (
    tiles: DatasetTile[],
    promptsText: string,
    threshold: number,
    options?: RunSceneLabelingOptions,
  ) => Promise<Record<string, BrowserLabelSuggestion>>;
};

const DEFAULT_PROMPT_TEXT = "server-side hybrid scan";

export function useRemoteCrosswalkLabeler({ runName, exportName, scenes, tileSizeM }: RemoteCrosswalkLabelerArgs) {
  const [status, setStatus] = useState<LabelerStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressState>({ done: 0, total: 0 });
  const [lastBatchCount, setLastBatchCount] = useState(0);
  const [controller, setController] = useState<RemoteControllerSnapshot | null>(null);
  const activeJobIdRef = useRef<string | null>(null);

  const ensureReady = useCallback(async () => {
    setStatus("connecting");
    setError(null);
    try {
      let snapshot = await loadRemoteControllerSnapshot();
      if (!snapshot.connected) {
        snapshot = await connectRemoteController();
      }
      setController(snapshot);
      setStatus("ready");
    } catch (reason) {
      setStatus("error");
      setError(String(reason));
      throw reason;
    }
  }, []);

  const runSceneLabeling = useCallback(
    async (tiles: DatasetTile[], promptsText: string, threshold: number, options?: RunSceneLabelingOptions) => {
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

      const scanRadiusTiles = Math.max(1, Math.round(Math.sqrt(tiles.length / Math.PI)));
      const job = buildScanBatchJob({
        summary: {
          run_name: runName,
          export_name: exportName,
        } as never,
        scene: scene as never,
        tileSizeM,
        scanRadiusTiles,
        threshold,
        promptsText,
        tiles,
      });

      const record = await startRemoteScanJob(job);
      activeJobIdRef.current = record.id;

      for (;;) {
        if (options?.shouldAbort?.() && activeJobIdRef.current) {
          await cancelRemoteScanJob(activeJobIdRef.current).catch(() => undefined);
          activeJobIdRef.current = null;
          setStatus("ready");
          setProgress({ done: 0, total: tiles.length });
          return {};
        }

        const next = await getRemoteScanJob(record.id);
        setProgress({
          done: next.status === "completed" && next.summary ? next.summary.total : 0,
          total: tiles.length,
          currentTileId: next.slurm_job_id ?? next.remote_state ?? undefined,
        });

        if (next.status === "completed") {
          const result = await getRemoteScanResult(record.id);
          const orderedSuggestions = result.tiles;
          orderedSuggestions.forEach((suggestion, index) => {
            options?.onSuggestion?.(suggestion, index, orderedSuggestions.length);
          });
          setLastBatchCount(result.summary.total);
          setProgress({ done: result.summary.total, total: result.summary.total });
          setStatus("ready");
          activeJobIdRef.current = null;
          return result.results;
        }

        if (next.status === "failed") {
          setStatus("error");
          setError(next.error ?? "Remote scan failed.");
          activeJobIdRef.current = null;
          throw new Error(next.error ?? "Remote scan failed.");
        }

        if (next.status === "cancelled") {
          setStatus("ready");
          activeJobIdRef.current = null;
          return {};
        }

        await new Promise((resolve) => window.setTimeout(resolve, 1_250));
      }
    },
    [ensureReady, exportName, runName, scenes, tileSizeM],
  );

  return useMemo(
    () => ({
      defaultPromptText: DEFAULT_PROMPT_TEXT,
      status,
      error,
      progress,
      lastBatchCount,
      controller,
      ensureReady,
      runSceneLabeling,
    }),
    [controller, ensureReady, error, lastBatchCount, progress, runSceneLabeling, status],
  );
}
