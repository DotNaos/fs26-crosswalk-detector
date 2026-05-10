import { Card } from "@heroui/react";
import type { MapValidationSnapshot } from "../map-validation";

type MapValidationPanelProps = {
  snapshot: MapValidationSnapshot | null;
};

export function MapValidationPanel({ snapshot }: MapValidationPanelProps) {
  if (!snapshot) {
    return (
      <Card className="pointer-events-auto w-80 shadow-xl" variant="secondary">
        <Card.Header>
          <div className="flex flex-col gap-1">
            <Card.Title>Map validation</Card.Title>
            <Card.Description>Diagnostics booting</Card.Description>
          </div>
        </Card.Header>
      </Card>
    );
  }

  return (
    <Card className="pointer-events-auto w-80 shadow-xl" variant="secondary">
      <Card.Header>
        <div className="flex min-w-0 flex-col gap-1">
          <Card.Title>{snapshot.validationCase ?? "manual"}</Card.Title>
          <Card.Description>{snapshot.status}</Card.Description>
        </div>
      </Card.Header>
      <Card.Content className="space-y-3 text-sm">
        <div className="grid grid-cols-2 gap-2">
          <span>Scene</span>
          <strong className="truncate text-right">{snapshot.sceneId ?? "n/a"}</strong>
          <span>Tile</span>
          <strong className="truncate text-right">{snapshot.selectedTileId ?? "n/a"}</strong>
          <span>Zoom</span>
          <strong className="text-right">{snapshot.camera.zoom.toFixed(2)}</strong>
          <span>Scan</span>
          <strong className="text-right">
            {snapshot.scanIndex}/{snapshot.orderedScanTileIds.length}
          </strong>
        </div>
        <div className="space-y-2">
          {snapshot.verdicts.map((verdict) => (
            <div className="flex items-center justify-between gap-3" key={verdict.name}>
              <span>{verdict.name}</span>
              <strong>{verdict.pass ? "pass" : "fail"}</strong>
            </div>
          ))}
        </div>
      </Card.Content>
    </Card>
  );
}
