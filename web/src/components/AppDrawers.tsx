import { Button, Card, Drawer } from "@heroui/react";
import type { RemoteScanJobRecord } from "../types";
import { CreateDatasetDrawer } from "./CreateDatasetDrawer";
import { RemoteTmuxTerminal } from "./RemoteTmuxTerminal";

type SceneOption = {
  scene_id: string;
  city: string;
  split: string;
};

type AppDrawersProps = {
  activeRemoteJob: RemoteScanJobRecord | null;
  creatingDataset: boolean;
  error: string | null;
  isCreateDatasetDrawerOpen: boolean;
  isErrorDrawerOpen: boolean;
  isMobileLayout: boolean;
  isTerminalDrawerOpen: boolean;
  newDatasetName: string;
  newDatasetSceneId: string;
  sceneOptions: SceneOption[];
  onCreateDataset: () => void;
  onCreateDatasetOpenChange: (isOpen: boolean) => void;
  onErrorOpenChange: (isOpen: boolean) => void;
  onNameChange: (name: string) => void;
  onSceneChange: (sceneId: string) => void;
  onTerminalOpenChange: (isOpen: boolean) => void;
};

export function AppDrawers({
  activeRemoteJob,
  creatingDataset,
  error,
  isCreateDatasetDrawerOpen,
  isErrorDrawerOpen,
  isMobileLayout,
  isTerminalDrawerOpen,
  newDatasetName,
  newDatasetSceneId,
  sceneOptions,
  onCreateDataset,
  onCreateDatasetOpenChange,
  onErrorOpenChange,
  onNameChange,
  onSceneChange,
  onTerminalOpenChange,
}: AppDrawersProps) {
  return (
    <>
      <CreateDatasetDrawer
        creating={creatingDataset}
        isMobileLayout={isMobileLayout}
        isOpen={isCreateDatasetDrawerOpen}
        name={newDatasetName}
        sceneId={newDatasetSceneId}
        sceneOptions={sceneOptions}
        onCreate={onCreateDataset}
        onNameChange={onNameChange}
        onOpenChange={onCreateDatasetOpenChange}
        onSceneChange={onSceneChange}
      />

      <Drawer.Backdrop className="z-[1400]" isOpen={isErrorDrawerOpen} onOpenChange={onErrorOpenChange} variant="blur">
        <Drawer.Content placement={isMobileLayout ? "bottom" : "right"}>
          <Drawer.Dialog aria-label="Error details">
            <Drawer.CloseTrigger />
            {isMobileLayout ? <Drawer.Handle /> : null}
            <Drawer.Header>
              <Drawer.Heading>Error details</Drawer.Heading>
            </Drawer.Header>
            <Drawer.Body className={isMobileLayout ? "max-h-[70dvh] overflow-y-auto" : "overflow-y-auto"}>
              <pre className="whitespace-pre-wrap break-words text-sm">{error ?? "No error details available."}</pre>
            </Drawer.Body>
            <Drawer.Footer>
              <Button slot="close" variant="secondary">Close</Button>
            </Drawer.Footer>
          </Drawer.Dialog>
        </Drawer.Content>
      </Drawer.Backdrop>

      <Drawer.Backdrop className="z-[1400]" isOpen={isTerminalDrawerOpen} onOpenChange={onTerminalOpenChange} variant="blur">
        <Drawer.Content placement={isMobileLayout ? "bottom" : "right"}>
          <Drawer.Dialog aria-label="Live terminal">
            <Drawer.CloseTrigger />
            {isMobileLayout ? <Drawer.Handle /> : null}
            <Drawer.Header>
              <Drawer.Heading>Live terminal</Drawer.Heading>
            </Drawer.Header>
            <Drawer.Body className={isMobileLayout ? "max-h-[72dvh] overflow-y-auto" : "overflow-y-auto"}>
              {activeRemoteJob ? (
                <RemoteTmuxTerminal jobId={activeRemoteJob.id} tmuxSession={activeRemoteJob.tmux_session} />
              ) : (
                <Card variant="secondary"><Card.Content>No active remote job.</Card.Content></Card>
              )}
            </Drawer.Body>
            <Drawer.Footer>
              <Button slot="close" variant="secondary">Close</Button>
            </Drawer.Footer>
          </Drawer.Dialog>
        </Drawer.Content>
      </Drawer.Backdrop>
    </>
  );
}
