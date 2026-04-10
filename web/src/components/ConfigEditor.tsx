import type { RealDatasetConfig } from "../types";
import type { ScanHealth } from "../scan-api";

type ConfigEditorProps = {
  config?: RealDatasetConfig;
  backendUrl: string;
  backendHealth: ScanHealth | null;
  onBackendUrlChange: (next: string) => void;
  onCheckBackend: () => void;
  onChange: (next: RealDatasetConfig) => void;
  onSave: () => void;
  saving: boolean;
};

export function ConfigEditor({
  config,
  backendUrl,
  backendHealth,
  onBackendUrlChange,
  onCheckBackend,
  onChange,
  onSave,
  saving,
}: ConfigEditorProps) {
  if (!config) {
    return (
      <section className="panel config-panel">
        <p className="eyebrow">Pipeline config</p>
        <h2>Loading configuration</h2>
        <p className="empty-copy">Reading the current rebuild settings from disk.</p>
      </section>
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
    <section className="panel config-panel">
      <div className="config-header">
        <div>
          <p className="eyebrow">Scan backend</p>
          <h2>Connection</h2>
        </div>
        <button className="ghost" onClick={onCheckBackend} type="button">
          Check Backend
        </button>
      </div>

      <div className="config-grid">
        <label className="config-span-2">
          Backend URL
          <input value={backendUrl} onChange={(event) => onBackendUrlChange(event.target.value)} placeholder="http://127.0.0.1:8000" />
        </label>
      </div>

      <p className="config-note">
        {backendHealth
          ? backendHealth.ready
            ? `Connected to ${backendHealth.model} on ${backendHealth.device}.`
            : backendHealth.warming
              ? `Loading ${backendHealth.model}. This can take a minute on the first run.`
              : "Backend reachable. Click Check Backend to warm up the models."
          : "Point the UI at your local laptop server or a Colab tunnel URL."}
      </p>

      <div className="config-header">
        <div>
          <p className="eyebrow">Pipeline config</p>
          <h2>Dataset settings</h2>
        </div>
        <button className="primary" disabled={saving} onClick={onSave} type="button">
          Save Config
        </button>
      </div>

      <div className="config-grid">
        <label>
          Bucket size
          <input
            type="number"
            value={config.target_per_class}
            onChange={(event) => updateNumber("target_per_class", event.target.value)}
          />
        </label>
        <label>
          Tile size (m)
          <input type="number" value={config.tile_size_m} onChange={(event) => updateNumber("tile_size_m", event.target.value)} />
        </label>
        <label>
          Train ratio
          <input
            type="number"
            step="0.01"
            value={config.split_ratios.train}
            onChange={(event) => updateNumber("split_ratios.train", event.target.value)}
          />
        </label>
        <label>
          Val ratio
          <input
            type="number"
            step="0.01"
            value={config.split_ratios.val}
            onChange={(event) => updateNumber("split_ratios.val", event.target.value)}
          />
        </label>
        <label>
          Test ratio
          <input
            type="number"
            step="0.01"
            value={config.split_ratios.test}
            onChange={(event) => updateNumber("split_ratios.test", event.target.value)}
          />
        </label>
        <label>
          Positive min combined
          <input
            type="number"
            step="0.01"
            value={config.selection.positive_min_combined}
            onChange={(event) => updateNumber("selection.positive_min_combined", event.target.value)}
          />
        </label>
        <label>
          Positive min road
          <input
            type="number"
            step="0.01"
            value={config.selection.positive_min_road_surface}
            onChange={(event) => updateNumber("selection.positive_min_road_surface", event.target.value)}
          />
        </label>
        <label>
          Positive min heuristic
          <input
            type="number"
            step="0.01"
            value={config.selection.positive_min_heuristic}
            onChange={(event) => updateNumber("selection.positive_min_heuristic", event.target.value)}
          />
        </label>
        <label>
          Negative max combined
          <input
            type="number"
            step="0.01"
            value={config.selection.negative_max_combined}
            onChange={(event) => updateNumber("selection.negative_max_combined", event.target.value)}
          />
        </label>
        <label>
          Negative penalty
          <input
            type="number"
            step="0.01"
            value={config.selection.negative_positive_penalty}
            onChange={(event) => updateNumber("selection.negative_positive_penalty", event.target.value)}
          />
        </label>
      </div>

      <p className="config-note">These values are written back to the TOML file and apply to the next dataset rebuild.</p>
    </section>
  );
}
