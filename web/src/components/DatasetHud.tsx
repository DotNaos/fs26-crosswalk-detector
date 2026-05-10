import { Button, Card, ListBox, ListBoxItem, Select } from "@heroui/react";
import { FolderOpen, Plus } from "lucide-react";
import type { DatasetListEntry } from "../types";

type DatasetHudProps = {
  datasets: DatasetListEntry[];
  currentValue: string;
  compact?: boolean;
  onSelect: (value: string) => void;
  onCreate: () => void;
};

export function DatasetHud({ datasets, currentValue, compact = false, onSelect, onCreate }: DatasetHudProps) {
  const selectedEntry = datasets.find((entry) => `${entry.run_name}::${entry.export_name}` === currentValue);
  const selector = (
    <Select
      className={compact ? "w-52 max-w-[56vw]" : undefined}
      fullWidth={!compact}
      placeholder="Choose dataset"
      selectedKey={currentValue || null}
      variant="secondary"
      onSelectionChange={(key) => {
        if (typeof key === "string") {
          onSelect(key);
        }
      }}
    >
      <Select.Trigger>
        <Select.Value>{selectedEntry ? selectedEntry.display_name ?? selectedEntry.export_name : "Choose dataset"}</Select.Value>
        <Select.Indicator />
      </Select.Trigger>
      <Select.Popover>
        <ListBox aria-label="Dataset selection">
          {datasets.map((entry) => {
            const value = `${entry.run_name}::${entry.export_name}`;
            return (
              <ListBoxItem id={value} key={value} textValue={`${entry.display_name ?? entry.export_name} ${entry.export_name}`}>
                {entry.display_name ?? entry.export_name}
              </ListBoxItem>
            );
          })}
        </ListBox>
      </Select.Popover>
    </Select>
  );

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        {selector}
        <Button aria-label="Create dataset" className="size-9 shrink-0 rounded-full" isIconOnly onPress={onCreate} size="sm" variant="secondary">
          <Plus className="size-4" />
        </Button>
      </div>
    );
  }

  return (
    <Card className="pointer-events-auto w-[320px] max-w-[calc(100vw-2rem)] shadow-xl" variant="secondary">
      <Card.Header>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <Card.Title className="flex items-center gap-2">
            <FolderOpen className="size-4" />
            <span>Review Desk</span>
          </Card.Title>
          <Card.Description>Select the dataset you want to review.</Card.Description>
        </div>
        <Button isIconOnly onPress={onCreate} size="sm" variant="secondary">
          <Plus className="size-4" />
        </Button>
      </Card.Header>
      <Card.Content>
        {selector}
      </Card.Content>
    </Card>
  );
}
