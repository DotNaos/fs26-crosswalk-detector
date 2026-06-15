import { useEffect, useMemo, useState } from "react";
import { Button, Chip, Input, Label, ListBox, ListBoxItem, Select } from "@heroui/react";
import { Bot, CheckCircle2, Database, LoaderCircle, LocateFixed, MapIcon, Play, Route, XCircle } from "lucide-react";
import { listDatasets, loadDatasetMeta, runCrossMaskOnTiles, updateTile } from "./api";
import { FALLBACK_EXPORT, FALLBACK_RUN } from "./app-autopilot";
import { DatasetMap } from "./components/DatasetMap";
import { SelectedTilePreview } from "./components/SelectedTilePreview";
import { isStaticDatasetEntry } from "./static-dataset";
import type { DatasetListEntry, DatasetSummary, DatasetTile, DatasetViewportPayload, ImageLabelVote, TileUpdate } from "./types";
import { mercatorToLatLng } from "./utils";

type SelectOption = {
  id: string;
  label: string;
};

function datasetValue(runName: string, exportName: string) {
  return `${runName}::${exportName}`;
}

function splitDatasetValue(value: string) {
  const [runName, exportName] = value.split("::");
  return { runName, exportName };
}

function HeroSelect({
  label,
  options,
  selectedKey,
  onChange,
}: {
  label: string;
  options: SelectOption[];
  selectedKey: string | null;
  onChange: (value: string) => void;
}) {
  const selected = options.find((option) => option.id === selectedKey);
  return (
    <Select aria-label={label} selectedKey={selectedKey} variant="secondary" onSelectionChange={(key) => typeof key === "string" && onChange(key)}>
      <Label>{label}</Label>
      <Select.Trigger>
        <Select.Value>{selected?.label ?? "All"}</Select.Value>
        <Select.Indicator />
      </Select.Trigger>
      <Select.Popover>
        <ListBox aria-label={`${label} selection`}>
          {options.map((option) => (
            <ListBoxItem id={option.id} key={option.id} textValue={option.label}>
              {option.label}
            </ListBoxItem>
          ))}
        </ListBox>
      </Select.Popover>
    </Select>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="border-t border-white/10 pt-2">
      <span className="block text-xs text-slate-300/60">{label}</span>
      <strong className="mt-1 block text-lg text-slate-50">{value}</strong>
    </div>
  );
}

function tileNumber(tile: DatasetTile, key: string) {
  const value = tile[key];
  return typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
}

function labelVotes(tile: DatasetTile) {
  return Array.isArray(tile.labels) ? (tile.labels as ImageLabelVote[]) : [];
}

function formatScore(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "n/a";
}

function tileCoordinates(tile: DatasetTile | null) {
  if (!tile) return null;
  const latitude = tileNumber(tile, "latitude");
  const longitude = tileNumber(tile, "longitude");
  if (Number.isFinite(latitude) && Number.isFinite(longitude)) return { latitude, longitude };
  if (Array.isArray(tile.bbox_mercator) && tile.bbox_mercator.length === 4) {
    const [minX, minY, maxX, maxY] = tile.bbox_mercator.map(Number);
    if ([minX, minY, maxX, maxY].every(Number.isFinite)) {
      const center = mercatorToLatLng((minX + maxX) / 2, (minY + maxY) / 2);
      return { latitude: center[0], longitude: center[1] };
    }
  }
  return null;
}

function LabelHistory({ tile }: { tile: DatasetTile }) {
  const votes = labelVotes(tile).slice().reverse();
  return (
    <div className="grid gap-2 border-t border-white/10 pt-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-bold">
          <Bot className="size-4 text-primary" />
          Label votes
        </div>
        <Chip variant="secondary">{votes.length}</Chip>
      </div>
      <div className="grid gap-2">
        {votes.map((vote, index) => {
          const source = vote.source ?? {};
          const sourceId = String(source.source_id ?? "unknown");
          const kind = String(source.kind ?? "model");
          return (
            <div className="border border-white/10 bg-white/[0.03] p-2 text-xs" key={`${String(vote.vote_id ?? sourceId)}:${index}`}>
              <div className="flex items-center justify-between gap-2">
                <strong className="truncate text-slate-100">{String(source.display_name ?? sourceId)}</strong>
                <Chip variant={kind === "human" ? "primary" : "secondary"}>{kind}</Chip>
              </div>
              <div className="mt-1 grid grid-cols-2 gap-2 text-slate-300/70">
                <span>{String(vote.decision ?? "unknown")}</span>
                <span className="text-right">conf {formatScore(vote.confidence)}</span>
              </div>
              {vote.metadata && typeof vote.metadata === "object" ? (
                <div className="mt-1 text-slate-400">
                  {Object.entries(vote.metadata)
                    .filter(([, value]) => typeof value === "number" || typeof value === "string")
                    .slice(0, 3)
                    .map(([key, value]) => `${key}: ${typeof value === "number" ? formatScore(value) : value}`)
                    .join(" · ")}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function App() {
  const [datasets, setDatasets] = useState<DatasetListEntry[]>([]);
  const [runName, setRunName] = useState(FALLBACK_RUN);
  const [exportName, setExportName] = useState(FALLBACK_EXPORT);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [viewport, setViewport] = useState<DatasetViewportPayload | null>(null);
  const [selectedTile, setSelectedTile] = useState<DatasetTile | null>(null);
  const [labelFilter, setLabelFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("all");
  const [cityFilter, setCityFilter] = useState("all");
  const [resetToken, setResetToken] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [crossmaskRunning, setCrossmaskRunning] = useState(false);
  const [crossmaskResult, setCrossmaskResult] = useState<string | null>(null);

  useEffect(() => {
    listDatasets()
      .then((entries) => {
        setDatasets(entries);
        const preferred =
          entries.find((entry) => entry.run_name === "sam3-500k-masks-v1" && entry.export_name === "metadata-500k-masks-v1") ??
          entries.find((entry) => entry.run_name === FALLBACK_RUN && entry.export_name === FALLBACK_EXPORT) ??
          entries.find((entry) => entry.run_name === "osm-v2-50k") ??
          entries.find((entry) => entry.run_name === "real-v1" && entry.export_name === "real-balanced-256") ??
          entries[0];
        if (preferred) {
          setRunName(preferred.run_name);
          setExportName(preferred.export_name);
        }
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    let canceled = false;
    setLoading(true);
    setSelectedTile(null);
    setViewport(null);
    loadDatasetMeta(runName, exportName)
      .then((nextSummary) => {
        if (canceled) return;
        setSummary(nextSummary);
        setError(null);
      })
      .catch((reason) => {
        if (!canceled) setError(String(reason));
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });
    return () => {
      canceled = true;
    };
  }, [exportName, runName]);

  const datasetOptions = datasets.map((dataset) => ({
    id: datasetValue(dataset.run_name, dataset.export_name),
    label: dataset.display_name ?? `${dataset.run_name}/${dataset.export_name}`,
  }));
  const selectedDatasetEntry = datasets.find((dataset) => dataset.run_name === runName && dataset.export_name === exportName);
  const readOnlyDataset = isStaticDatasetEntry(selectedDatasetEntry);

  const cityOptions = useMemo<SelectOption[]>(() => {
    const cities = [...new Set(summary?.scenes.map((scene) => scene.city).filter(Boolean) ?? [])].sort();
    return [{ id: "all", label: "All cities" }, ...cities.map((city) => ({ id: city, label: city }))];
  }, [summary]);

  const splitOptions = [
    { id: "all", label: "All splits" },
    { id: "train", label: "Train" },
    { id: "val", label: "Validation" },
    { id: "test", label: "Test" },
  ];

  const labelOptions = [
    { id: "all", label: "All labels" },
    { id: "crosswalk", label: "Crosswalk" },
    { id: "no_crosswalk", label: "No crosswalk" },
  ];

  const selectDataset = (value: string) => {
    const next = splitDatasetValue(value);
    if (!next.runName || !next.exportName) return;
    setRunName(next.runName);
    setExportName(next.exportName);
  };

  const patchTile = async (tile: DatasetTile, update: TileUpdate) => {
    const response = await updateTile(runName, exportName, tile.tile_id, update);
    const updatedTile = (response as unknown as { tile?: DatasetTile }).tile;
    setSelectedTile(updatedTile ?? { ...tile, ...update, status: update.selected ? "manual-selected" : "dropped" });
    setSummary(await loadDatasetMeta(runName, exportName));
  };

  const handleTileAction = (tile: DatasetTile, action: "crosswalk" | "no_crosswalk" | "drop") => {
    const update =
      action === "drop"
        ? { label: tile.label, selected: false }
        : {
            label: action,
            selected: true,
            predicted_label: action,
            review_source: "manual-dataset-map",
          };
    patchTile(tile, update).catch((reason) => setError(String(reason)));
  };

  const visibleTiles = viewport?.mode === "tiles" ? viewport.tiles : [];
  const runCrossMask = async () => {
    if (readOnlyDataset) return;
    const selectedVisibleTile = selectedTile ? visibleTiles.find((tile) => tile.tile_id === selectedTile.tile_id) : undefined;
    const runCandidates = selectedVisibleTile ? [selectedVisibleTile, ...visibleTiles.filter((tile) => tile.tile_id !== selectedVisibleTile.tile_id)] : visibleTiles;
    const tiles = runCandidates.slice(0, 64);
    if (!tiles.length) return;
    setCrossmaskRunning(true);
    setCrossmaskResult(null);
    try {
      const result = await runCrossMaskOnTiles({ exportName, maxTiles: 64, runName, tiles });
      const updatedById = new Map(result.updated_tiles.map((tile) => [tile.tile_id, tile]));
      setSelectedTile((current) => (current ? (updatedById.get(current.tile_id) ?? current) : (result.updated_tiles[0] ?? null)));
      setCrossmaskResult(`${result.summary.crosswalk} crosswalk / ${result.summary.no_crosswalk} no crosswalk`);
      setSummary(await loadDatasetMeta(runName, exportName));
    } catch (reason) {
      setError(String(reason));
    } finally {
      setCrossmaskRunning(false);
    }
  };

  const visibleCrosswalk = viewport?.mode === "clusters"
    ? viewport.clusters.reduce((sum, cluster) => sum + cluster.crosswalk, 0)
    : (viewport?.tiles.filter((tile) => tile.label === "crosswalk").length ?? 0);
  const visibleNoCrosswalk = viewport?.mode === "clusters"
    ? viewport.clusters.reduce((sum, cluster) => sum + cluster.no_crosswalk, 0)
    : (viewport?.tiles.filter((tile) => tile.label === "no_crosswalk").length ?? 0);
  const coordinates = tileCoordinates(selectedTile);

  return (
    <div className="flex h-dvh min-w-0 flex-col bg-[#10151b] text-slate-200">
      <header className="flex min-h-16 flex-wrap items-center gap-3 border-b border-white/10 bg-[#10151b]/95 px-4 py-3 max-[640px]:flex-col max-[640px]:items-stretch">
        <div className="flex flex-none items-center gap-2 text-sm font-bold tracking-normal max-[640px]:w-full">
          <Route className="size-4 text-primary" />
          <span>Crosswalk Dataset Explorer</span>
        </div>
        <div className="grid flex-1 items-end gap-2 [grid-template-columns:minmax(14rem,1.4fr)_repeat(3,minmax(8rem,0.8fr))_auto] max-[980px]:grid-cols-2 max-[640px]:w-full">
          <HeroSelect label="Dataset" options={datasetOptions} selectedKey={datasetValue(runName, exportName)} onChange={selectDataset} />
          <HeroSelect label="City" options={cityOptions} selectedKey={cityFilter} onChange={setCityFilter} />
          <HeroSelect label="Split" options={splitOptions} selectedKey={splitFilter} onChange={setSplitFilter} />
          <HeroSelect label="Label" options={labelOptions} selectedKey={labelFilter} onChange={setLabelFilter} />
          <Button size="sm" variant="secondary" onPress={() => setResetToken((current) => current + 1)}>
            <LocateFixed className="size-4" />
            Switzerland
          </Button>
        </div>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(20rem,26rem)] max-[980px]:grid-cols-1 max-[980px]:grid-rows-[minmax(22rem,1fr)_minmax(18rem,42dvh)]">
        <section className="min-h-0 min-w-0">
          <DatasetMap
            city={cityFilter}
            exportName={exportName}
            label={labelFilter}
            resetToken={resetToken}
            runName={runName}
            selectedTileId={selectedTile?.tile_id}
            split={splitFilter}
            onError={setError}
            onPayload={setViewport}
            onSelectTile={setSelectedTile}
          />
        </section>

        <aside className="flex min-h-0 flex-col gap-4 overflow-y-auto border-l border-white/10 bg-[#151b22] p-4 max-[980px]:border-l-0 max-[980px]:border-t">
          <div className="flex items-center gap-2">
            <MapIcon className="size-4 text-primary" />
            <h1 className="m-0 text-base font-bold">Dataset inspection</h1>
          </div>
          {error ? <div className="border border-red-400/30 bg-red-400/10 p-3 text-sm text-red-100">{error}</div> : null}
          {loading ? <Chip variant="secondary">Loading dataset</Chip> : null}

          <div className="grid grid-cols-2 gap-3">
            <Metric label="Dataset tiles" value={summary?.total_tiles ?? 0} />
            <Metric label="Visible" value={viewport?.total_matching ?? 0} />
            <Metric label="Visible crosswalk" value={visibleCrosswalk} />
            <Metric label="Visible no crosswalk" value={visibleNoCrosswalk} />
          </div>

          <div className="text-sm leading-relaxed text-slate-300/65">
            {readOnlyDataset
              ? "Static read-only inspection mode is using compressed dataset metadata bundled with this deployment."
              : "The main view is the Leaflet road map. Visible dataset satellite tiles are loaded as overlays on top."}
          </div>

          <div className="grid gap-2 border border-white/10 bg-white/[0.03] p-3">
            <div className="flex items-center justify-between gap-2 text-sm">
              <span className="font-bold text-slate-100">CrossMaskNet v4</span>
              <Chip variant="secondary">{readOnlyDataset ? "static preview" : visibleTiles.length ? `${Math.min(64, visibleTiles.length)} ready` : "zoom in"}</Chip>
            </div>
            <Button isDisabled={readOnlyDataset || !visibleTiles.length || crossmaskRunning} size="sm" variant="primary" onPress={runCrossMask}>
              {crossmaskRunning ? <LoaderCircle className="size-4 animate-spin" /> : <Play className="size-4" />}
              Run on visible tiles
            </Button>
            {readOnlyDataset ? <div className="text-xs text-slate-300/70">Live model runs need a local writable backend. This deployment is for dataset inspection only.</div> : null}
            {crossmaskResult ? <div className="text-xs text-slate-300/70">{crossmaskResult}</div> : null}
          </div>

          {selectedTile ? (
            <div className="flex flex-col gap-3">
              {selectedTile.has_image_asset === false ? (
                <SelectedTilePreview tile={selectedTile} />
              ) : (
                <img className="aspect-square w-full border border-white/10 bg-slate-900 object-cover" alt={`Tile ${selectedTile.tile_id}`} src={selectedTile.image_path} />
              )}
              <div className="grid gap-2">
                <Input readOnly value={selectedTile.tile_id} aria-label="Tile id" />
                <Input readOnly value={`${selectedTile.city} / ${selectedTile.split}`} aria-label="City and split" />
                <Input readOnly value={selectedTile.label} aria-label="Label" />
                <Input
                  readOnly
                  value={coordinates ? `${coordinates.latitude.toFixed(6)}, ${coordinates.longitude.toFixed(6)}` : "No coordinates"}
                  aria-label="Coordinates"
                />
              </div>
              <div className="flex flex-wrap gap-2">
                <Button isDisabled={readOnlyDataset} size="sm" variant="primary" onPress={() => handleTileAction(selectedTile, "crosswalk")}>
                  <CheckCircle2 className="size-4" />
                  Crosswalk
                </Button>
                <Button isDisabled={readOnlyDataset} size="sm" variant="secondary" onPress={() => handleTileAction(selectedTile, "no_crosswalk")}>
                  <XCircle className="size-4" />
                  No crosswalk
                </Button>
                <Button isDisabled={readOnlyDataset} size="sm" variant="secondary" onPress={() => handleTileAction(selectedTile, "drop")}>
                  Drop
                </Button>
              </div>
              <LabelHistory tile={selectedTile} />
            </div>
          ) : (
            <div className="flex items-center gap-3 border border-dashed border-white/20 p-4 text-slate-300/65">
              <Database className="size-5" />
              <span>Select a visible tile to inspect the source image and label.</span>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
