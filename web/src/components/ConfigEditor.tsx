import { Button, Card, Input, Label, Skeleton, TextField } from "@heroui/react";
import type { RealDatasetConfig } from "../types";

type ConfigEditorProps = {
  config?: RealDatasetConfig;
  onChange: (next: RealDatasetConfig) => void;
  onSave: () => void;
  saving: boolean;
};

export function ConfigEditor({ config, onChange, onSave, saving }: ConfigEditorProps) {
  if (!config) {
    return (
      <Card variant="secondary" className="pointer-events-auto shadow-xl">
        <Card.Header>
          <div className="flex flex-col gap-1">
            <Card.Title>Pipeline config</Card.Title>
            <Card.Description>Loading configuration…</Card.Description>
          </div>
        </Card.Header>
        <Card.Content>
          <div className="flex flex-col gap-3">
            <Skeleton className="h-10 rounded-lg" />
            <Skeleton className="h-10 rounded-lg" />
            <Skeleton className="h-10 rounded-lg" />
          </div>
        </Card.Content>
      </Card>
    );
  }

  const updateNumber = (path: string, value: string) => {
    const numericValue = Number(value);
    const next = structuredClone(config);
    const [section, key] = path.split(".");
    if (key) {
      (next[section as "split_ratios" | "selection"] as Record<string, number>)[key] = numericValue;
    } else {
      (next as unknown as Record<string, number>)[section] = numericValue;
    }
    onChange(next);
  };

  return (
    <Card variant="secondary" className="pointer-events-auto shadow-xl">
      <Card.Header>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <Card.Title>Pipeline config</Card.Title>
          <Card.Description>Dataset settings</Card.Description>
        </div>
        <Button isDisabled={saving} onPress={onSave} variant="primary" size="sm">
          Save
        </Button>
      </Card.Header>
      <Card.Content className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3">
          <TextField variant="secondary">
            <Label>Bucket size</Label>
            <Input
              type="number"
              value={String(config.target_per_class)}
              onChange={(e) => updateNumber("target_per_class", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Tile size (m)</Label>
            <Input
              type="number"
              value={String(config.tile_size_m)}
              onChange={(e) => updateNumber("tile_size_m", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Train ratio</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.split_ratios.train)}
              onChange={(e) => updateNumber("split_ratios.train", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Val ratio</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.split_ratios.val)}
              onChange={(e) => updateNumber("split_ratios.val", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Test ratio</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.split_ratios.test)}
              onChange={(e) => updateNumber("split_ratios.test", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Min combined</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.selection.positive_min_combined)}
              onChange={(e) => updateNumber("selection.positive_min_combined", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Min road</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.selection.positive_min_road_surface)}
              onChange={(e) => updateNumber("selection.positive_min_road_surface", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Min heuristic</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.selection.positive_min_heuristic)}
              onChange={(e) => updateNumber("selection.positive_min_heuristic", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Max neg combined</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.selection.negative_max_combined)}
              onChange={(e) => updateNumber("selection.negative_max_combined", e.target.value)}
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Neg penalty</Label>
            <Input
              type="number"
              step="0.01"
              value={String(config.selection.negative_positive_penalty)}
              onChange={(e) => updateNumber("selection.negative_positive_penalty", e.target.value)}
            />
          </TextField>
        </div>
        <p className="text-xs text-foreground-muted">
          Values are written to the TOML file and apply on the next dataset rebuild.
        </p>
      </Card.Content>
    </Card>
  );
}
