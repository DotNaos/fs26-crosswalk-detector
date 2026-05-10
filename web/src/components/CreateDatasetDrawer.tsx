import { Button, Drawer, Input, Label, ListBox, ListBoxItem, Select, TextField } from "@heroui/react";
import { Plus } from "lucide-react";

type SceneOption = {
  scene_id: string;
  city: string;
  split: string;
};

type CreateDatasetDrawerProps = {
  isMobileLayout: boolean;
  isOpen: boolean;
  name: string;
  sceneId: string;
  sceneOptions: SceneOption[];
  creating: boolean;
  onNameChange: (value: string) => void;
  onSceneChange: (value: string) => void;
  onOpenChange: (isOpen: boolean) => void;
  onCreate: () => void;
};

export function CreateDatasetDrawer({
  isMobileLayout,
  isOpen,
  name,
  sceneId,
  sceneOptions,
  creating,
  onNameChange,
  onSceneChange,
  onOpenChange,
  onCreate,
}: CreateDatasetDrawerProps) {
  return (
    <Drawer.Backdrop className="z-[1450]" isOpen={isOpen} onOpenChange={onOpenChange} variant="blur">
      <Drawer.Content placement={isMobileLayout ? "bottom" : "right"}>
        <Drawer.Dialog aria-label="Create dataset">
          <Drawer.CloseTrigger />
          {isMobileLayout ? <Drawer.Handle /> : null}
          <Drawer.Header>
            <Drawer.Heading>Create dataset</Drawer.Heading>
          </Drawer.Header>
          <Drawer.Body className="flex flex-col gap-4">
            <p className="text-sm text-foreground/70">
              Create a new working dataset, pick the area, then scan and review inside that dataset.
            </p>

            <TextField variant="secondary">
              <Label>Name</Label>
              <Input placeholder="Zurich HB review set" value={name} onChange={(event) => onNameChange(event.target.value)} />
            </TextField>

            <Select fullWidth selectedKey={sceneId || null} variant="secondary" onSelectionChange={(key) => typeof key === "string" && onSceneChange(key)}>
              <Label>Area</Label>
              <Select.Trigger>
                <Select.Value>
                  {sceneOptions.find((entry) => entry.scene_id === sceneId)
                    ? `${sceneOptions.find((entry) => entry.scene_id === sceneId)?.city} · ${sceneOptions.find((entry) => entry.scene_id === sceneId)?.split}`
                    : "Choose area"}
                </Select.Value>
                <Select.Indicator />
              </Select.Trigger>
              <Select.Popover>
                <ListBox aria-label="Area selection">
                  {sceneOptions.map((scene) => (
                    <ListBoxItem id={scene.scene_id} key={scene.scene_id} textValue={`${scene.city} ${scene.split}`}>
                      {scene.city} · {scene.split}
                    </ListBoxItem>
                  ))}
                </ListBox>
              </Select.Popover>
            </Select>
          </Drawer.Body>
          <Drawer.Footer>
            <Button slot="close" variant="secondary">
              Cancel
            </Button>
            <Button isDisabled={!name.trim() || !sceneId || creating} onPress={onCreate} variant="primary">
              <Plus className="size-4" />
              Create
            </Button>
          </Drawer.Footer>
        </Drawer.Dialog>
      </Drawer.Content>
    </Drawer.Backdrop>
  );
}
