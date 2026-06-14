import { useEffect } from "react";
import type { LatLngBoundsExpression, Map as LeafletMap, PathOptions } from "leaflet";
import { ImageOverlay, MapContainer, Rectangle, TileLayer, useMap } from "react-leaflet";
import type { DatasetTile } from "../types";
import { bboxToLatLngBounds, bboxToMercatorRect, mercatorToLatLng } from "../utils";

const SWISSIMAGE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/current/3857/{z}/{x}/{y}.png";
const SWISSIMAGE_ATTRIBUTION = '&copy; <a href="https://www.swisstopo.admin.ch/">swisstopo</a>';

type MaskOverlay = {
  sourceId: string;
  url: string;
};

function tileMaskUrls(tile: DatasetTile): MaskOverlay[] {
  const labels = Array.isArray(tile.labels) ? tile.labels : [];
  const urls: MaskOverlay[] = [];
  for (const vote of labels) {
    if (!vote || typeof vote !== "object") continue;
    const source = "source" in vote ? vote.source : null;
    const sourceId =
      source && typeof source === "object" && "source_id" in source && typeof source.source_id === "string" ? source.source_id : "unknown";
    const metadata = "metadata" in vote ? vote.metadata : null;
    if (!metadata || typeof metadata !== "object") continue;
    const artifact = "mask_artifact" in metadata ? metadata.mask_artifact : null;
    if (!artifact || typeof artifact !== "object") continue;
    const url = "url" in artifact ? artifact.url : null;
    if (typeof url === "string" && url.length > 0) urls.push({ sourceId, url });
  }
  return urls;
}

function FitTileBounds({ bounds }: { bounds: LatLngBoundsExpression }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(bounds, { animate: false, padding: [0, 0] });
  }, [bounds, map]);
  return null;
}

function previewBounds(tile: DatasetTile): LatLngBoundsExpression | null {
  const rect = bboxToMercatorRect(tile.bbox_mercator);
  if (!rect) return null;
  const width = Math.abs(rect.maxX - rect.minX);
  const height = Math.abs(rect.maxY - rect.minY);
  const padX = width * 0.25;
  const padY = height * 0.25;
  return [
    mercatorToLatLng(rect.minX - padX, rect.minY - padY),
    mercatorToLatLng(rect.maxX + padX, rect.maxY + padY),
  ];
}

function tileOutlineOptions(): PathOptions {
  return {
    color: "#ffffff",
    className: "selected-tile-preview-outline",
    fillOpacity: 0,
    opacity: 0.95,
    weight: 2,
  };
}

export function SelectedTilePreview({ tile }: { tile: DatasetTile }) {
  const bounds = bboxToLatLngBounds(tile.bbox_mercator);
  const fitBounds = previewBounds(tile);
  if (!bounds || !fitBounds) {
    return (
      <div className="flex aspect-square w-full items-center justify-center border border-white/10 bg-slate-900 p-4 text-center text-sm text-slate-300/70">
        No map bounds are available for this tile.
      </div>
    );
  }

  const masks = tileMaskUrls(tile);
  return (
    <div className="relative aspect-square w-full overflow-hidden border border-white/10 bg-slate-900" data-testid="selected-tile-preview">
      <MapContainer
        attributionControl={false}
        className="h-full w-full bg-slate-900"
        dragging={false}
        doubleClickZoom={false}
        scrollWheelZoom={false}
        zoomControl={false}
        boxZoom={false}
        keyboard={false}
        center={bounds[0]}
        zoom={18}
        maxZoom={22}
        zoomSnap={0}
      >
        <TileLayer attribution={SWISSIMAGE_ATTRIBUTION} maxNativeZoom={18} maxZoom={22} url={SWISSIMAGE_URL} />
        <FitTileBounds bounds={fitBounds} />
        {masks.map(({ sourceId, url }, index) => (
          <ImageOverlay
            bounds={bounds}
            key={`${sourceId}:${index}`}
            opacity={sourceId.startsWith("crossmask") ? 0.66 : 0.58}
            url={url}
            zIndex={sourceId.startsWith("crossmask") ? 465 : 455}
          />
        ))}
        <Rectangle bounds={bounds} pathOptions={tileOutlineOptions()} />
      </MapContainer>
      <div className="pointer-events-none absolute left-2 top-2 bg-[#151b22]/85 px-2 py-1 text-xs font-bold text-slate-100">SwissTopo preview</div>
    </div>
  );
}
