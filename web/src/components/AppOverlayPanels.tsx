import type { ReactNode } from "react";
import { Alert, Card } from "@heroui/react";
import { AlertTriangle } from "lucide-react";
import type { AutopilotPlan } from "../autopilot-planner";
import type { ScanBatchJob, ScanBatchResult } from "../scan-batch";
import type {
  BrowserLabelSuggestion,
  DatasetListEntry,
  DatasetSummary,
  MapBasemap,
  RealDatasetConfig,
  RemoteControllerSnapshot,
  RemoteScanJobRecord,
  ScenePayload,
} from "../types";
import { AutopilotPanel } from "./AutopilotPanel";
import { DatasetHud } from "./DatasetHud";
import { Inspector } from "./Inspector";
import { MobileOverlayPanels } from "./MobileOverlayPanels";
import { RemoteScanPanel } from "./RemoteScanPanel";

type AppOverlayPanelsProps = {
  activeRemoteJobId: string | null;
  basemap: MapBasemap;
  browserSuggestion?: BrowserLabelSuggestion;
  config?: RealDatasetConfig;
  creatingAutopilot: boolean;
  datasets: DatasetListEntry[];
  errorSummary: string | null;
  exportName: string;
  isMobileLayout: boolean;
  mobileScanPanel: ReactNode;
  remoteSnapshot: RemoteControllerSnapshot | null;
  runName: string;
  scanJob: ScanBatchJob | null;
  scene?: DatasetSummary["scenes"][number];
  sceneSuggestionCount: number;
  selectedTile?: ScenePayload["tiles"][number];
  serverJob: RemoteScanJobRecord | null;
  summary: DatasetSummary | null;
  saving: boolean;
  onActiveRemoteJobChange: (jobId: string | null) => void;
  onActiveRemoteJobResolved: (job: RemoteScanJobRecord | null) => void;
  onApplySuggestion?: () => void;
  onBasemapChange: (next: MapBasemap) => void;
  onCommit: (label: string, selected: boolean) => void;
  onCreateAutopilot: (input: { targetPositiveCount: number; maxPanels?: number; perimeterBudget?: number }) => void;
  onCreateDataset: () => void;
  onDatasetSelect: (value: string) => void;
  onError: (message: string | null) => void;
  onJumpToNextPositive: () => void;
  onJumpToNextSuggestion: () => void;
  onOpenErrorDetails: () => void;
  onOpenTerminal: () => void;
  onPreviewPlanChange: (plan: AutopilotPlan | null) => void;
  onResultImported: (result: ScanBatchResult) => void;
};

export function AppOverlayPanels({
  activeRemoteJobId,
  basemap,
  browserSuggestion,
  config,
  creatingAutopilot,
  datasets,
  errorSummary,
  exportName,
  isMobileLayout,
  mobileScanPanel,
  remoteSnapshot,
  runName,
  scanJob,
  scene,
  sceneSuggestionCount,
  selectedTile,
  serverJob,
  summary,
  saving,
  onActiveRemoteJobChange,
  onActiveRemoteJobResolved,
  onApplySuggestion,
  onBasemapChange,
  onCommit,
  onCreateAutopilot,
  onCreateDataset,
  onDatasetSelect,
  onError,
  onJumpToNextPositive,
  onJumpToNextSuggestion,
  onOpenErrorDetails,
  onOpenTerminal,
  onPreviewPlanChange,
  onResultImported,
}: AppOverlayPanelsProps) {
  if (isMobileLayout) {
    return (
      <MobileOverlayPanels
        scanPanel={mobileScanPanel ?? <Card variant="secondary"><Card.Content>Scan controls are loading.</Card.Content></Card>}
        datasetPanel={
          <div className="flex w-full min-w-0 flex-col gap-4">
            <DatasetHud compact currentValue={`${runName}::${exportName}`} datasets={datasets} onCreate={onCreateDataset} onSelect={onDatasetSelect} />
            <AutopilotPanel
              compact
              config={config}
              creating={creatingAutopilot}
              summary={summary}
              onPreviewPlanChange={onPreviewPlanChange}
              onCreate={onCreateAutopilot}
            />
          </div>
        }
        errorSummary={errorSummary}
        onOpenErrorDetails={onOpenErrorDetails}
        activeRemoteJob={serverJob}
        remoteConnected={remoteSnapshot?.connected ?? false}
        remoteLastError={remoteSnapshot?.last_error ?? null}
        basemap={basemap}
        onBasemapChange={onBasemapChange}
        reviewPanel={
          <Inspector
            compact
            embedded
            scene={scene}
            tile={selectedTile}
            browserSuggestion={browserSuggestion}
            suggestionCount={sceneSuggestionCount}
            onCommit={onCommit}
            onApplySuggestion={onApplySuggestion}
            onJumpToNextSuggestion={onJumpToNextSuggestion}
            onJumpToNextPositive={onJumpToNextPositive}
            saving={saving}
          />
        }
        serverPanel={
          <RemoteScanPanel
            compact
            scanJob={scanJob}
            activeRemoteJobId={activeRemoteJobId}
            onActiveRemoteJobChange={onActiveRemoteJobChange}
            onActiveRemoteJobResolved={onActiveRemoteJobResolved}
            onOpenTerminal={onOpenTerminal}
            onError={onError}
            onResultImported={onResultImported}
          />
        }
      />
    );
  }

  return (
    <>
      <div className="pointer-events-none absolute left-4 top-4 flex max-w-[calc(100vw-2rem)] flex-col gap-4">
        <DatasetHud currentValue={`${runName}::${exportName}`} datasets={datasets} onCreate={onCreateDataset} onSelect={onDatasetSelect} />
        {errorSummary ? (
          <div
            className="pointer-events-auto"
            role="button"
            tabIndex={0}
            onClick={onOpenErrorDetails}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onOpenErrorDetails();
              }
            }}
          >
            <Alert className="max-w-md cursor-pointer shadow-xl" status="danger">
              <Alert.Content>
                <Alert.Title className="flex items-center gap-2">
                  <AlertTriangle className="size-4" />
                  <span>Error</span>
                </Alert.Title>
                <Alert.Description>{errorSummary}</Alert.Description>
              </Alert.Content>
            </Alert>
          </div>
        ) : null}
        <AutopilotPanel config={config} creating={creatingAutopilot} summary={summary} onPreviewPlanChange={onPreviewPlanChange} onCreate={onCreateAutopilot} />
      </div>

      <aside
        className="pointer-events-auto absolute bottom-4 right-4 top-4 flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-4 overflow-y-auto overscroll-contain pr-1"
        data-scroll-guard="desktop-right-rail"
      >
        <Inspector
          scene={scene}
          tile={selectedTile}
          browserSuggestion={browserSuggestion}
          suggestionCount={sceneSuggestionCount}
          onCommit={onCommit}
          onApplySuggestion={onApplySuggestion}
          onJumpToNextSuggestion={onJumpToNextSuggestion}
          onJumpToNextPositive={onJumpToNextPositive}
          saving={saving}
        />
        <RemoteScanPanel
          scanJob={scanJob}
          activeRemoteJobId={activeRemoteJobId}
          onActiveRemoteJobChange={onActiveRemoteJobChange}
          onActiveRemoteJobResolved={onActiveRemoteJobResolved}
          onOpenTerminal={onOpenTerminal}
          onError={onError}
          onResultImported={onResultImported}
        />
      </aside>
    </>
  );
}
