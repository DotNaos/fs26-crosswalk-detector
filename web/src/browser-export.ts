import JSZip from "jszip";
import Papa from "papaparse";
import { stringify as stringifyToml } from "smol-toml";
import { loadBrowserConfig, loadBrowserDataset, loadBrowserScene } from "./browser-workspace";
import type { DatasetTile } from "./types";

async function blobFromUrl(url: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch tile image: ${response.status}`);
  }
  return response.blob();
}

function selectedTiles(tiles: DatasetTile[]) {
  return tiles.filter((tile) => tile.selected && (tile.label === "crosswalk" || tile.label === "no_crosswalk"));
}

export async function exportBrowserDatasetZip() {
  const config = loadBrowserConfig();
  const dataset = loadBrowserDataset();
  const zip = new JSZip();
  const exportTiles: DatasetTile[] = [];

  for (const scene of dataset.scenes) {
    const scenePayload = await loadBrowserScene(scene.scene_id);
    exportTiles.push(...selectedTiles(scenePayload.tiles));
  }

  const rows = exportTiles.map((tile) => ({
    tile_id: tile.tile_id,
    scene_id: tile.scene_id,
    city: tile.city,
    split: tile.split,
    row: tile.row,
    col: tile.col,
    relative_path: tile.relative_path,
    label: tile.label,
    predicted_label: tile.predicted_label,
    selected: tile.selected,
    status: tile.status,
    review_source: tile.review_source,
    combined_probability: tile.combined_probability,
    bbox_mercator: JSON.stringify(tile.bbox_mercator),
  }));

  zip.file("labels.csv", Papa.unparse(rows));
  zip.file("tiles.json", JSON.stringify(exportTiles, null, 2));
  zip.file("config.toml", stringifyToml(config as unknown as Record<string, unknown>));

  const imagesFolder = zip.folder("images");
  if (!imagesFolder) {
    throw new Error("Failed to create images folder in ZIP.");
  }

  for (const tile of exportTiles) {
    const blob = await blobFromUrl(tile.image_path);
    imagesFolder.file(tile.relative_path, blob);
  }

  const archive = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(archive);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${config.export_name}.zip`;
  anchor.click();
  URL.revokeObjectURL(url);

  return {
    exportName: config.export_name,
    selectedCount: exportTiles.length,
  };
}
