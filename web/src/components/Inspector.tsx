import { Alert, Button, Card, Skeleton } from "@heroui/react";
import { Check, CheckCheck, CircleOff, MinusCircle, SkipForward, Tags } from "lucide-react";
import type { BrowserLabelSuggestion, DatasetScene, DatasetTile } from "../types";
import { formatProbability, sceneLabel } from "../utils";

type InspectorProps = {
  scene?: DatasetScene;
  tile?: DatasetTile;
  browserSuggestion?: BrowserLabelSuggestion;
  suggestionCount: number;
  embedded?: boolean;
  compact?: boolean;
  onCommit: (label: string, selected: boolean) => void;
  onApplySuggestion?: () => void;
  onJumpToNextSuggestion: () => void;
  onJumpToNextPositive: () => void;
  saving: boolean;
};

export function Inspector({
  scene,
  tile,
  browserSuggestion,
  suggestionCount,
  embedded = false,
  compact = false,
  onCommit,
  onApplySuggestion,
  onJumpToNextSuggestion,
  onJumpToNextPositive,
  saving,
}: InspectorProps) {
  if (compact) {
    return (
      <div className="flex w-full min-w-0 flex-col gap-3">
        <div className="grid min-w-0 grid-cols-[5rem,1fr] gap-3">
          {tile?.image_path ? (
            <img
              alt={tile.relative_path}
              className="aspect-square w-20 rounded-[18px] object-cover [corner-shape:squircle]"
              decoding="async"
              loading="lazy"
              src={tile.image_path}
            />
          ) : (
            <Skeleton className="aspect-square w-20 rounded-[18px] [corner-shape:squircle]" />
          )}
          <div className="flex min-w-0 flex-col justify-center gap-1">
            <div className="truncate text-base font-semibold">{tile ? tile.relative_path : "No tile selected"}</div>
            <div className="truncate text-xs text-white/65">{tile ? (scene ? sceneLabel(scene) : tile.scene_id) : "Tap a tile on the map"}</div>
            <div className="flex min-w-0 flex-wrap gap-1.5 pt-1 text-[11px] font-semibold">
              {tile ? <span className="rounded-full bg-white/10 px-2 py-1">{tile.selected ? tile.label : "unreviewed"}</span> : null}
              {browserSuggestion ? (
                <span className="rounded-full bg-primary/25 px-2 py-1 text-primary-foreground">
                  machine {browserSuggestion.label} · {formatProbability(browserSuggestion.score)}
                </span>
              ) : null}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button
            aria-label="Mark as crosswalk"
            className="h-10 rounded-[16px] [corner-shape:squircle]"
            isDisabled={!tile || saving}
            onPress={() => tile && onCommit("crosswalk", true)}
            size="sm"
            variant="primary"
          >
            <Check className="size-4" />
            Crosswalk
          </Button>
          <Button
            aria-label="Mark as no crosswalk"
            className="h-10 rounded-[16px] [corner-shape:squircle]"
            isDisabled={!tile || saving}
            onPress={() => tile && onCommit("no_crosswalk", true)}
            size="sm"
            variant="danger"
          >
            <CircleOff className="size-4" />
            No
          </Button>
          <Button
            aria-label="Drop label"
            className="h-10 rounded-[16px] [corner-shape:squircle]"
            isDisabled={!tile || saving}
            onPress={() => tile && onCommit(tile.label, false)}
            size="sm"
            variant="secondary"
          >
            <MinusCircle className="size-4" />
            Drop
          </Button>
          <Button
            aria-label="Next server result"
            className="h-10 rounded-[16px] [corner-shape:squircle]"
            isDisabled={!suggestionCount}
            onPress={onJumpToNextSuggestion}
            size="sm"
            variant={browserSuggestion ? "primary" : "secondary"}
          >
            <SkipForward className="size-4" />
            Next
          </Button>
        </div>
        {browserSuggestion && onApplySuggestion ? (
          <Button className="h-10 rounded-[16px] [corner-shape:squircle]" onPress={onApplySuggestion} size="sm" variant="secondary">
            <CheckCheck className="size-4" />
            Accept machine label
          </Button>
        ) : null}
      </div>
    );
  }

  const content = (
    <>
      <div className="flex min-w-0 flex-col gap-1">
        <h3 className="truncate text-lg font-semibold text-foreground">{tile ? tile.relative_path : "Awaiting selection"}</h3>
        <p className="text-sm text-foreground/70">
          {tile ? (scene ? sceneLabel(scene) : tile.scene_id) : "Select a tile on the map to review it."}
        </p>
      </div>

      {tile ? (
        tile.image_path ? (
          <img
            alt={tile.relative_path}
            className="aspect-square w-full rounded-xl object-cover"
            decoding="async"
            loading="lazy"
            src={tile.image_path}
          />
        ) : (
          <Skeleton className="aspect-square w-full rounded-xl" />
        )
      ) : (
        <Skeleton className="aspect-square w-full rounded-xl" />
      )}

      {tile && browserSuggestion ? (
        <Alert status="accent">
          <Alert.Content>
            <Alert.Title className="flex items-center gap-2">
              <Tags className="size-4" />
              <span>Server guess</span>
            </Alert.Title>
            <Alert.Description>{browserSuggestion.label}</Alert.Description>
          </Alert.Content>
        </Alert>
      ) : null}

      {tile && (browserSuggestion || suggestionCount > 0) ? (
        <div className="flex flex-wrap gap-2">
          {browserSuggestion && onApplySuggestion ? (
            <Button onPress={onApplySuggestion} size="sm" variant="secondary">
              <CheckCheck className="size-4" />
              Accept result
            </Button>
          ) : null}
          <Button onPress={onJumpToNextPositive} size="sm" variant="ghost">
            <SkipForward className="size-4" />
            Next positive
          </Button>
          <Button onPress={onJumpToNextSuggestion} size="sm" variant="ghost">
            <SkipForward className="size-4" />
            Next
          </Button>
        </div>
      ) : null}

      {tile ? (
        <div className="flex flex-wrap gap-2">
          <Button isDisabled={saving} onPress={() => onCommit("crosswalk", true)} variant="primary">
            <Check className="size-4" />
            Crosswalk
          </Button>
          <Button isDisabled={saving} onPress={() => onCommit("no_crosswalk", true)} variant="danger">
            <CircleOff className="size-4" />
            No crosswalk
          </Button>
          <Button isDisabled={saving} onPress={() => onCommit(tile.label, false)} variant="tertiary">
            <MinusCircle className="size-4" />
            Drop
          </Button>
        </div>
      ) : null}
    </>
  );

  if (embedded) {
    return <div className="flex flex-col gap-4">{content}</div>;
  }

  return (
    <Card className="pointer-events-auto max-h-[calc(100dvh-2rem)] overflow-hidden shadow-xl" variant="secondary">
      <Card.Content className="flex max-h-full min-h-0 flex-col gap-4 overflow-y-auto" data-scroll-guard="inspector-card">
        {content}
      </Card.Content>
    </Card>
  );
}
