import { Button, Card, Input, Label, TextField } from "@heroui/react";
import { LoaderCircle, Pause, Play } from "lucide-react";

type MapScanControlsProps = {
  sceneLabel: string;
  isMobileLayout?: boolean;
  mapZoom: number;
  showGrid: boolean;
  sceneTilesReady: boolean;
  totalTilesInCircle: number;
  scanRadius: number;
  scanDelay: number;
  scanRunning: boolean;
  scanQueued: boolean;
  scanPreparing: boolean;
  sceneImagesReady: boolean;
  scannedCount: number;
  crosswalkCount: number;
  noCrosswalkCount: number;
  liveScanStep: number;
  activeSummary: string | null;
  autopilotMode?: boolean;
  onScanRadiusChange: (next: number) => void;
  onScanDelayChange: (next: number) => void;
  onStartScan: () => void;
  onPauseScan: () => void;
  onExportBatchJob: () => void;
  onImportBatchResult: () => void;
};

export function MapScanControls({
  sceneLabel,
  isMobileLayout = false,
  mapZoom,
  showGrid,
  sceneTilesReady,
  totalTilesInCircle,
  scanRadius,
  scanRunning,
  scanQueued,
  scanPreparing,
  sceneImagesReady,
  scannedCount,
  liveScanStep,
  activeSummary,
  autopilotMode = false,
  onScanRadiusChange,
  onStartScan,
  onPauseScan,
}: MapScanControlsProps) {
  const statusLine = !sceneTilesReady
    ? "Loading area"
    : !showGrid
      ? `Zoom ${mapZoom.toFixed(1)} to open the grid`
      : scanQueued || scanPreparing
        ? "Preparing scan"
        : scanRunning
          ? `Running ${liveScanStep}/${totalTilesInCircle}`
          : !sceneImagesReady
            ? "Loading imagery"
            : `${scannedCount}/${totalTilesInCircle} scanned`;

  return (
    <>
      {isMobileLayout ? (
        <div className="flex w-full min-w-0 flex-col gap-3">
          <div className="flex min-w-0 items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-base font-semibold">{sceneLabel}</div>
              <div className="line-clamp-2 text-xs leading-snug text-white/65">{activeSummary ?? statusLine}</div>
            </div>
            <div className="shrink-0 rounded-[16px] bg-white/10 px-3 py-2 text-sm font-semibold [corner-shape:squircle]">
              {scannedCount}/{totalTilesInCircle}
            </div>
          </div>

          <div className="grid grid-cols-[1fr,auto] gap-2">
            {autopilotMode ? (
              <div className="rounded-[16px] bg-white/10 px-3 py-2 [corner-shape:squircle]">
                <div className="text-xs font-semibold">Auto urban panel</div>
                <div className="text-[11px] text-white/60">planned perimeter</div>
              </div>
            ) : (
              <TextField className="min-w-0" variant="secondary">
                <Label>Radius</Label>
                <Input
                  className="h-10 text-center"
                  max={12}
                  min={2}
                  type="number"
                  value={String(scanRadius)}
                  onChange={(event) => onScanRadiusChange(Number(event.target.value) || scanRadius)}
                />
              </TextField>
            )}

            {scanRunning ? (
              <Button aria-label="Pause scan" className="mt-6 size-10 shrink-0 rounded-[16px] [corner-shape:squircle]" isIconOnly onPress={onPauseScan} size="sm" variant="secondary">
                <Pause className="size-4" />
              </Button>
            ) : (
              <Button
                aria-label="Start scan"
                className="mt-6 size-10 shrink-0 rounded-[16px] [corner-shape:squircle]"
                isDisabled={scanQueued || scanPreparing}
                isIconOnly
                onPress={onStartScan}
                size="sm"
                variant="primary"
              >
                {scanQueued || scanPreparing ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
              </Button>
            )}
          </div>
        </div>
      ) : (
        <Card className="pointer-events-auto max-h-[calc(100dvh-2rem)] w-[min(22rem,calc(100vw-2rem))] overflow-hidden shadow-2xl" variant="secondary">
          <Card.Header>
            <div className="flex min-w-0 flex-col gap-1">
              <Card.Title>{sceneLabel}</Card.Title>
              <Card.Description>{activeSummary ?? statusLine}</Card.Description>
            </div>
          </Card.Header>
          <Card.Content className="flex max-h-full min-h-0 flex-col gap-3 overflow-y-auto" data-scroll-guard="map-scan-controls">
            {autopilotMode ? (
              <div className="rounded-[16px] bg-content1/70 px-3 py-2 [corner-shape:squircle]">
                <div className="text-sm font-semibold">Auto urban panel</div>
                <div className="text-xs text-foreground/60">Scanning the planned perimeter cell.</div>
              </div>
            ) : (
              <TextField variant="secondary">
                <Label>Radius</Label>
                <Input
                  max={12}
                  min={2}
                  type="number"
                  value={String(scanRadius)}
                  onChange={(event) => onScanRadiusChange(Number(event.target.value) || scanRadius)}
                />
              </TextField>
            )}
            <div className="flex gap-2">
              {scanRunning ? (
                <Button fullWidth onPress={onPauseScan} variant="secondary">
                  Pause
                </Button>
              ) : (
                <Button fullWidth isDisabled={scanQueued || scanPreparing} onPress={onStartScan} variant="primary">
                  {scanQueued || scanPreparing ? "Preparing" : "Start scan"}
                </Button>
              )}
            </div>
          </Card.Content>
        </Card>
      )}
    </>
  );
}
