import type { Dispatch, SetStateAction } from "react";
import { useMemo, useState } from "react";
import { Alert, Button, Card, Chip, Input, Label, Spinner, TextField } from "@heroui/react";
import { updateTiles } from "../api";
import type { BrowserCrosswalkLabelerHandle } from "../hooks/useBrowserCrosswalkLabeler";
import type { BrowserLabelSuggestion, DatasetScene, DatasetTile, ScenePayload } from "../types";
import { formatProbability, sceneLabel } from "../utils";

type BrowserDatasetBuilderProps = {
  runName: string;
  exportName: string;
  backendUrl: string;
  scene?: DatasetScene;
  sceneTiles: DatasetTile[];
  labeler: BrowserCrosswalkLabelerHandle;
  suggestions: Record<string, BrowserLabelSuggestion>;
  onSuggestionsChange: Dispatch<SetStateAction<Record<string, BrowserLabelSuggestion>>>;
  onApplied: (payload: ScenePayload) => void;
  onBackendUrlChange: (next: string) => void;
  onError: (message: string | null) => void;
};

export function BrowserDatasetBuilder({
  runName,
  exportName,
  backendUrl,
  scene,
  sceneTiles,
  labeler,
  suggestions,
  onSuggestionsChange,
  onApplied,
  onBackendUrlChange,
  onError,
}: BrowserDatasetBuilderProps) {
  const [promptText, setPromptText] = useState(labeler.defaultPromptText);
  const [threshold, setThreshold] = useState(0.32);
  const [saving, setSaving] = useState(false);

  const sceneSuggestions = useMemo(
    () => sceneTiles.map((tile) => suggestions[tile.tile_id]).filter((entry): entry is BrowserLabelSuggestion => Boolean(entry)),
    [sceneTiles, suggestions],
  );
  const suggestionStats = useMemo(
    () => ({
      total: sceneSuggestions.length,
      crosswalk: sceneSuggestions.filter((entry) => entry.label === "crosswalk").length,
      noCrosswalk: sceneSuggestions.filter((entry) => entry.label === "no_crosswalk").length,
      strongest: sceneSuggestions.slice().sort((a, b) => b.score - a.score)[0],
    }),
    [sceneSuggestions],
  );

  async function handleRunScene() {
    if (!sceneTiles.length) return;
    onError(null);
    try {
      const next = await labeler.runSceneLabeling(sceneTiles, promptText, threshold);
      onSuggestionsChange({
        ...suggestions,
        ...next,
      });
    } catch (reason) {
      onError(String(reason));
    }
  }

  async function handleApplyScene() {
    if (!scene?.scene_id || sceneSuggestions.length === 0) return;
    setSaving(true);
    onError(null);
    try {
      const sceneUpdates = sceneSuggestions.map((suggestion) => ({
        tile_id: suggestion.tile_id,
        label: suggestion.label,
        selected: suggestion.selected,
        combined_probability: suggestion.score,
        predicted_label: suggestion.label,
        review_source: suggestion.review_source,
      }));
      const payload = await updateTiles(runName, exportName, scene.scene_id, sceneUpdates);
      onApplied(payload);
      const nextSuggestions = { ...suggestions };
      for (const tile of payload.tiles) {
        delete nextSuggestions[tile.tile_id];
      }
      onSuggestionsChange(nextSuggestions);
    } catch (reason) {
      onError(String(reason));
    } finally {
      setSaving(false);
    }
  }

  const statusColor =
    labeler.status === "error" ? "danger" : labeler.status === "running" ? "warning" : "success";

  return (
    <Card variant="secondary" className="pointer-events-auto shadow-xl">
      <Card.Header>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <Card.Title>{scene ? sceneLabel(scene) : "Select a scene"}</Card.Title>
          <Card.Description>Browser dataset builder</Card.Description>
        </div>
        <Chip color={statusColor} size="sm">{labeler.status}</Chip>
      </Card.Header>

      <Card.Content className="flex flex-col gap-4">
        <TextField variant="secondary">
          <Label>Backend URL</Label>
          <Input
            value={backendUrl}
            onChange={(e) => onBackendUrlChange(e.target.value)}
            placeholder="http://127.0.0.1:8000"
          />
        </TextField>

        <p className="text-sm text-foreground-secondary">
          Runs the scan on a Python service on your laptop or in Colab, then writes the accepted labels back into
          the local dataset state.
        </p>

        <div className="flex flex-col gap-3">
          <TextField variant="secondary">
            <Label>Prompt list</Label>
            <Input
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder="crosswalk, zebra crossing"
            />
          </TextField>
          <TextField variant="secondary">
            <Label>Crosswalk threshold</Label>
            <Input
              type="number"
              min={0.2}
              max={0.98}
              step={0.01}
              value={String(threshold)}
              onChange={(e) => setThreshold(Number(e.target.value) || threshold)}
            />
          </TextField>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            isDisabled={labeler.status === "connecting" || labeler.status === "running"}
            onPress={() => void labeler.ensureReady()}
            variant="ghost"
            size="sm"
          >
            Check Backend
          </Button>
          <Button
            isDisabled={!sceneTiles.length || labeler.status === "running"}
            onPress={() => void handleRunScene()}
            variant="primary"
            size="sm"
          >
            {labeler.status === "running" ? <Spinner size="sm" /> : null}
            Scan Loaded Scene
          </Button>
          <Button
            isDisabled={sceneSuggestions.length === 0 || saving}
            onPress={() => void handleApplyScene()}
            variant="secondary"
            size="sm"
          >
            {saving ? <Spinner size="sm" /> : null}
            Apply Scene Labels
          </Button>
          <Button
            isDisabled={sceneSuggestions.length === 0}
            onPress={() => {
              const nextSuggestions = { ...suggestions };
              for (const tile of sceneTiles) {
                delete nextSuggestions[tile.tile_id];
              }
              onSuggestionsChange(nextSuggestions);
            }}
            variant="ghost"
            size="sm"
          >
            Clear Suggestions
          </Button>
        </div>

        <div className="flex flex-wrap gap-2">
          <Chip color="default" size="sm">{sceneTiles.length} tiles loaded</Chip>
          <Chip color="success" size="sm">✓ {suggestionStats.crosswalk} crosswalk</Chip>
          <Chip color="danger" size="sm">✕ {suggestionStats.noCrosswalk} no crosswalk</Chip>
          {labeler.progress.total > 0 ? (
            <Chip color="warning" size="sm">
              Running {labeler.progress.done}/{labeler.progress.total}
            </Chip>
          ) : null}
        </div>

        <div className="grid grid-cols-2 gap-2 text-sm">
          {[
            ["Latest batch", String(labeler.lastBatchCount)],
            ["Current tile", labeler.progress.currentTileId ?? "n/a"],
            [
              "Top suggestion",
              suggestionStats.strongest
                ? `${suggestionStats.strongest.label} · ${formatProbability(suggestionStats.strongest.score)}`
                : "n/a",
            ],
            ["Top prompt", suggestionStats.strongest?.prompt ?? "n/a"],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg border border-divider bg-content1 p-3">
              <p className="text-xs uppercase tracking-wide text-foreground-muted">{label}</p>
              <p className="mt-1 break-all text-foreground">{value}</p>
            </div>
          ))}
        </div>

        {labeler.error ? (
          <Alert status="danger">
            <Alert.Content>
              <Alert.Description>{labeler.error}</Alert.Description>
            </Alert.Content>
          </Alert>
        ) : null}
      </Card.Content>
    </Card>
  );
}
