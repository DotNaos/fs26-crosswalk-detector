import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { Button, ButtonGroup } from "@heroui/react";
import { AlertTriangle, FolderOpen, Globe, Inspect, LoaderCircle, Play, Route, Satellite, Server } from "lucide-react";
import type { MapBasemap, RemoteScanJobRecord } from "../types";

type MobileOverlayPanelsProps = {
  scanPanel: ReactNode;
  datasetPanel: ReactNode;
  reviewPanel: ReactNode;
  serverPanel: ReactNode;
  errorSummary?: string | null;
  onOpenErrorDetails?: () => void;
  activeRemoteJob?: RemoteScanJobRecord | null;
  remoteConnected: boolean;
  remoteLastError?: string | null;
  basemap: MapBasemap;
  onBasemapChange: (next: MapBasemap) => void;
};

function formatErrorChip(summary?: string | null) {
  if (!summary) return "Error";
  const compact = summary.replace(/\s+/g, " ").trim();
  if (compact.length <= 42) return `Error: ${compact}`;
  return `Error: ${compact.slice(0, 39)}...`;
}

function isRemoteBusy(job?: RemoteScanJobRecord | null) {
  return job ? ["bootstrapping", "syncing", "submitting", "queued", "running"].includes(job.status) : false;
}

export function MobileOverlayPanels({
  scanPanel,
  datasetPanel,
  reviewPanel,
  serverPanel,
  errorSummary,
  onOpenErrorDetails,
  activeRemoteJob,
  remoteConnected,
  remoteLastError,
  basemap,
  onBasemapChange,
}: MobileOverlayPanelsProps) {
  const [openPanel, setOpenPanel] = useState<"scan" | "dataset" | "review" | "server" | "layers" | null>(null);
  const [visualViewportHeight, setVisualViewportHeight] = useState(() =>
    Math.round(window.visualViewport?.height ?? window.innerHeight),
  );

  useEffect(() => {
    const viewport = window.visualViewport;
    const syncHeight = () => setVisualViewportHeight(Math.round(viewport?.height ?? window.innerHeight));
    syncHeight();
    viewport?.addEventListener("resize", syncHeight);
    window.addEventListener("resize", syncHeight);
    return () => {
      viewport?.removeEventListener("resize", syncHeight);
      window.removeEventListener("resize", syncHeight);
    };
  }, []);

  const runBusy = isRemoteBusy(activeRemoteJob);
  const serverStatusClass = remoteLastError
    ? "bg-danger text-danger-foreground shadow-lg shadow-danger/30"
    : remoteConnected
      ? "bg-success text-success-foreground shadow-lg shadow-success/30"
      : "bg-warning text-warning-foreground shadow-lg shadow-warning/30";
  const liveCount = activeRemoteJob?.live_scanned_tile_ids.length ?? 0;
  const activeChunkBusy = runBusy;
  const remoteStatusText = activeRemoteJob
    ? `${activeRemoteJob.status} · ${liveCount}/${activeRemoteJob.tile_count} tiles${
        activeRemoteJob.summary ? ` · ${activeRemoteJob.summary.crosswalk}+/${activeRemoteJob.summary.no_crosswalk}-` : ""
      }`
    : remoteLastError
      ? "Server error"
      : remoteConnected
        ? "Server connected"
        : "Server not connected";

  const panelBody = useMemo(() => {
    if (openPanel === "scan") return scanPanel;
    if (openPanel === "dataset") return datasetPanel;
    if (openPanel === "review") return reviewPanel;
    if (openPanel === "server") return serverPanel;
    if (openPanel === "layers") {
      return (
        <div className="flex flex-col gap-3">
          <ButtonGroup className="w-full">
            <Button className="flex-1" onPress={() => onBasemapChange("osm")} variant={basemap === "osm" ? "primary" : "secondary"}>
              <Globe className="size-4" />
              Map
            </Button>
            <Button className="flex-1" onPress={() => onBasemapChange("swisstopo")} variant={basemap === "swisstopo" ? "primary" : "secondary"}>
              <Satellite className="size-4" />
              Satellite
            </Button>
            <Button className="flex-1" onPress={() => onBasemapChange("roads")} variant={basemap === "roads" ? "primary" : "secondary"}>
              <Route className="size-4" />
              Roads
            </Button>
          </ButtonGroup>
        </div>
      );
    }
    return null;
  }, [basemap, datasetPanel, onBasemapChange, openPanel, reviewPanel, scanPanel, serverPanel]);

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-[1000]" style={{ height: visualViewportHeight }}>
      {activeRemoteJob ? (
        <div className="pointer-events-none absolute inset-x-3 top-[calc(env(safe-area-inset-top)+0.75rem)] flex justify-center">
          <Button
            aria-label={`Open server status for ${activeRemoteJob.scene_label ?? "scan"}`}
            className="pointer-events-auto h-8 max-w-[calc(100vw-1.5rem)] rounded-full border border-white/15 bg-black/72 px-3 text-xs font-semibold text-white shadow-xl backdrop-blur-xl"
            size="sm"
            variant="ghost"
            onPress={() => setOpenPanel((current) => (current === "server" ? null : "server"))}
          >
            {activeChunkBusy ? <LoaderCircle className="size-4 animate-spin" /> : null}
            <span className="truncate">{remoteStatusText}</span>
          </Button>
        </div>
      ) : null}

      <div className="pointer-events-none absolute inset-x-0 bottom-20 flex flex-col items-center px-2">
        {errorSummary ? (
          <Button
            className="pointer-events-auto mb-2 h-9 max-w-[calc(100vw-3rem)] rounded-full px-3 text-xs shadow-xl"
            onPress={onOpenErrorDetails}
            size="sm"
            variant="danger"
          >
            <AlertTriangle className="size-4 shrink-0" />
            <span className="truncate">{formatErrorChip(errorSummary)}</span>
          </Button>
        ) : null}

        {openPanel ? (
          <div
            className="pointer-events-auto max-h-[min(44dvh,24rem)] w-[min(calc(100vw-1rem),34rem)] overflow-y-auto overscroll-contain rounded-[24px] border border-white/15 bg-black/86 p-3 text-white shadow-2xl backdrop-blur-xl [corner-shape:squircle]"
            data-scroll-guard="mobile-overlay-panel"
          >
            <div className="flex w-full min-w-0 flex-col gap-3 text-sm">{panelBody}</div>
          </div>
        ) : null}
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-0">
        <div className="pointer-events-auto w-full border-t border-white/15 bg-black/84 px-3 pb-[calc(env(safe-area-inset-bottom)+0.55rem)] pt-2 shadow-[0_-14px_35px_rgba(0,0,0,0.32)] backdrop-blur-xl">
          <div className="mx-auto flex w-fit max-w-full flex-row items-center justify-center gap-2">
            <Button
              aria-label="Open scan controls"
              className={`size-9 rounded-[16px] [corner-shape:squircle] ${
                runBusy
                  ? "bg-blue-500 text-white shadow-lg shadow-blue-500/30"
                  : openPanel === "scan"
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30"
                  : "bg-white/8 text-white"
              }`}
              isIconOnly
              size="sm"
              variant="ghost"
              onPress={() => setOpenPanel((current) => (current === "scan" ? null : "scan"))}
            >
              {runBusy ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
            </Button>

            <Button
              aria-label="Open dataset picker"
              className={`size-9 rounded-[16px] [corner-shape:squircle] ${
                openPanel === "dataset"
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30"
                  : "bg-white/8 text-white"
              }`}
              isIconOnly
              size="sm"
              variant="ghost"
              onPress={() => setOpenPanel((current) => (current === "dataset" ? null : "dataset"))}
            >
              <FolderOpen className="size-4" />
            </Button>

            <Button
              aria-label="Open review panel"
              className={`size-9 rounded-[16px] [corner-shape:squircle] ${
                openPanel === "review"
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30"
                  : "bg-white/8 text-white"
              }`}
              isIconOnly
              size="sm"
              variant="ghost"
              onPress={() => setOpenPanel((current) => (current === "review" ? null : "review"))}
            >
              <Inspect className="size-4" />
            </Button>

            <Button
              aria-label="Open server panel"
              className={`size-9 rounded-[16px] [corner-shape:squircle] ${
                openPanel === "server"
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30"
                  : serverStatusClass
              }`}
              isIconOnly
              size="sm"
              variant="ghost"
              onPress={() => setOpenPanel((current) => (current === "server" ? null : "server"))}
            >
              <Server className="size-4" />
            </Button>

            <Button
              aria-label="Open map style picker"
              className={`size-9 rounded-[16px] [corner-shape:squircle] ${
                openPanel === "layers"
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30"
                  : "bg-white/8 text-white"
              }`}
              isIconOnly
              size="sm"
              variant="ghost"
              onPress={() => setOpenPanel((current) => (current === "layers" ? null : "layers"))}
            >
              {basemap === "roads" ? <Route className="size-4" /> : basemap === "swisstopo" ? <Satellite className="size-4" /> : <Globe className="size-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
