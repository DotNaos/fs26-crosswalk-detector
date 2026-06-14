import { useCallback, useEffect, useMemo, useState } from "react";
import { Button, Chip } from "@heroui/react";
import type { LatLngBoundsExpression, Map as LeafletMap, PathOptions } from "leaflet";
import { Maximize2 } from "lucide-react";
import { CircleMarker, ImageOverlay, MapContainer, Rectangle, TileLayer, useMap, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { loadDatasetViewport } from "../api";
import type { DatasetTile, DatasetViewportCluster, DatasetViewportPayload } from "../types";
import { bboxToLatLngBounds, latLngToMercator, mercatorToLatLng } from "../utils";

type BaseLayerId = "osm" | "swissimage";
type MaskLayerId = "all" | "crossmask" | "sam3" | "none";

const SWITZERLAND_BOUNDS = [
  [45.75, 5.75],
  [47.95, 10.65],
] as LatLngBoundsExpression;

const SWITZERLAND_CENTER = [46.8, 8.25] as [number, number];
const DEFAULT_ZOOM = 8;
const TILE_LIMIT = 1800;
const BASE_LAYERS: Record<BaseLayerId, { attribution: string; label: string; maxNativeZoom?: number; url: string }> = {
  osm: {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    label: "Map",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  },
  swissimage: {
    attribution: '&copy; <a href="https://www.swisstopo.admin.ch/">swisstopo</a>',
    label: "SwissTopo",
    maxNativeZoom: 18,
    url: "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/current/3857/{z}/{x}/{y}.png",
  },
};

const MASK_LAYERS: Record<MaskLayerId, string> = {
  all: "All masks",
  crossmask: "CrossMask",
  sam3: "SAM3",
  none: "Masks off",
};

type DatasetMapProps = {
  city: string;
  exportName: string;
  label: string;
  resetToken: number;
  runName: string;
  selectedTileId?: string;
  split: string;
  onError: (message: string | null) => void;
  onPayload: (payload: DatasetViewportPayload | null) => void;
  onSelectTile: (tile: DatasetTile) => void;
};

type ViewportSnapshot = {
  bboxMercator: [number, number, number, number];
  zoom: number;
};

function mapViewport(map: LeafletMap): ViewportSnapshot {
  const bounds = map.getBounds();
  const southWest = bounds.getSouthWest();
  const northEast = bounds.getNorthEast();
  const min = latLngToMercator(southWest.lat, southWest.lng);
  const max = latLngToMercator(northEast.lat, northEast.lng);
  return {
    bboxMercator: [min.x, min.y, max.x, max.y],
    zoom: map.getZoom(),
  };
}

function clusterColor(cluster: DatasetViewportCluster) {
  const ratio = cluster.total > 0 ? cluster.crosswalk / cluster.total : 0;
  if (ratio >= 0.68) return "#eab308";
  if (ratio <= 0.32) return "#2563eb";
  return "#65a30d";
}

function selectedTilePathOptions(tile: DatasetTile, selectedTileId?: string): PathOptions {
  const selected = tile.tile_id === selectedTileId;
  return {
    color: "#ffffff",
    fillOpacity: 0,
    opacity: selected ? 1 : 0,
    weight: selected ? 3 : 1,
  };
}

function tileMaskUrls(tile: DatasetTile, maskLayer: MaskLayerId) {
  if (maskLayer === "none") return [];
  const labels = Array.isArray(tile.labels) ? tile.labels : [];
  const urls: Array<{ sourceId: string; url: string }> = [];
  for (const vote of labels) {
    if (!vote || typeof vote !== "object") continue;
    const source = "source" in vote ? vote.source : null;
    const sourceId =
      source && typeof source === "object" && "source_id" in source && typeof source.source_id === "string" ? source.source_id : "unknown";
    if (maskLayer === "crossmask" && !sourceId.startsWith("crossmask")) continue;
    if (maskLayer === "sam3" && !sourceId.startsWith("sam")) continue;
    const metadata = "metadata" in vote ? vote.metadata : null;
    if (!metadata || typeof metadata !== "object") continue;
    const artifact = "mask_artifact" in metadata ? metadata.mask_artifact : null;
    if (!artifact || typeof artifact !== "object") continue;
    const url = "url" in artifact ? artifact.url : null;
    if (typeof url === "string" && url.length > 0) urls.push({ sourceId, url });
  }
  return urls;
}

function tileBoundsPathOptions(tile: DatasetTile, selectedTileId: string | undefined, baseLayer: BaseLayerId): PathOptions {
  const selected = tile.tile_id === selectedTileId;
  const color = tile.label === "crosswalk" ? "#eab308" : tile.label === "no_crosswalk" ? "#2563eb" : "#64748b";
  const imagery = baseLayer === "swissimage";
  return {
    color: selected ? "#ffffff" : color,
    fillColor: color,
    fillOpacity: selected ? (imagery ? 0.06 : 0.32) : imagery ? 0 : 0.16,
    opacity: selected ? 1 : imagery ? 0.56 : 0.52,
    weight: selected ? 3 : imagery ? 1.25 : 1,
  };
}

function clusterPathOptions(cluster: DatasetViewportCluster): PathOptions {
  const color = clusterColor(cluster);
  return {
    color: "#ffffff",
    fillColor: color,
    fillOpacity: 0.72,
    opacity: 0.78,
    weight: 1,
  };
}

function FitSwitzerland({ resetToken }: { resetToken: string }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(SWITZERLAND_BOUNDS, { animate: true, padding: [24, 24] });
  }, [map, resetToken]);
  return null;
}

function DatasetViewportLayer({
  baseLayer,
  city,
  exportName,
  label,
  resetSignal,
  runName,
  selectedTileId,
  split,
  maskLayer,
  onError,
  onLoading,
  onPayload,
  onSelectTile,
}: Omit<DatasetMapProps, "resetToken"> & {
  baseLayer: BaseLayerId;
  maskLayer: MaskLayerId;
  onLoading: (loading: boolean) => void;
  resetSignal: string;
}) {
  const map = useMap();
  const [viewport, setViewport] = useState<ViewportSnapshot>(() => mapViewport(map));
  const [payload, setPayload] = useState<DatasetViewportPayload | null>(null);

  const updateViewport = useCallback(() => setViewport(mapViewport(map)), [map]);
  useMapEvents({
    moveend: updateViewport,
    zoomend: updateViewport,
  });

  useEffect(() => {
    updateViewport();
  }, [city, exportName, label, resetSignal, runName, split, updateViewport]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      onLoading(true);
      void loadDatasetViewport(
        runName,
        exportName,
        {
          bboxMercator: viewport.bboxMercator,
          city,
          label,
          limit: TILE_LIMIT,
          split,
          zoom: viewport.zoom,
        },
        { signal: controller.signal },
      )
        .then((next) => {
          setPayload(next);
          onPayload(next);
          onError(null);
        })
        .catch((reason) => {
          if (controller.signal.aborted || reason?.name === "AbortError") return;
          setPayload(null);
          onPayload(null);
          onError(String(reason));
        })
        .finally(() => {
          if (!controller.signal.aborted) onLoading(false);
        });
    }, 160);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [city, exportName, label, onError, onLoading, onPayload, runName, split, viewport]);

  const clusters = payload?.mode === "clusters" ? payload.clusters : [];
  const tiles = payload?.mode === "tiles" ? payload.tiles : [];

  return (
    <>
      {clusters.map((cluster) => {
        const [x, y] = cluster.center_mercator;
        const center = mercatorToLatLng(x, y);
        const radius = Math.min(34, Math.max(9, 7 + Math.sqrt(cluster.total) * 0.92));
        return (
          <CircleMarker
            center={center}
            eventHandlers={{
              click: () => {
                const [minX, minY, maxX, maxY] = cluster.bbox_mercator;
                map.fitBounds([mercatorToLatLng(minX, minY), mercatorToLatLng(maxX, maxY)], { padding: [36, 36] });
              },
            }}
            key={cluster.id}
            pathOptions={clusterPathOptions(cluster)}
            radius={radius}
          />
        );
      })}

      {tiles.map((tile) => {
        const bounds = bboxToLatLngBounds(tile.bbox_mercator);
        const maskUrls = tileMaskUrls(tile, maskLayer);
        if (!bounds || maskUrls.length === 0) return null;
        return maskUrls.map(({ sourceId, url }, index) => (
          <ImageOverlay
            bounds={bounds}
            eventHandlers={{ click: () => onSelectTile(tile) }}
            key={`${tile.tile_id}:mask:${sourceId}:${index}`}
            opacity={tile.tile_id === selectedTileId ? 0.95 : 0.72}
            url={url}
            zIndex={(sourceId.startsWith("crossmask") ? 465 : 455) + (tile.tile_id === selectedTileId ? 20 : 0) + index}
          />
        ));
      })}

      {tiles.map((tile) => {
        const bounds = bboxToLatLngBounds(tile.bbox_mercator);
        if (!bounds) return null;
        if (tile.has_image_asset === false) return null;
        return (
          <ImageOverlay
            bounds={bounds}
            eventHandlers={{ click: () => onSelectTile(tile) }}
            key={tile.tile_id}
            opacity={tile.tile_id === selectedTileId ? 0.9 : 0.66}
            url={tile.image_path}
            zIndex={tile.tile_id === selectedTileId ? 430 : 420}
          />
        );
      })}

      {tiles.map((tile) => {
        const bounds = bboxToLatLngBounds(tile.bbox_mercator);
        if (!bounds) return null;
        if (tile.has_image_asset === false) {
          return (
            <Rectangle
              bounds={bounds}
              eventHandlers={{ click: () => onSelectTile(tile) }}
              key={`${tile.tile_id}:bounds`}
              pathOptions={tileBoundsPathOptions(tile, selectedTileId, baseLayer)}
            />
          );
        }
        return (
          <Rectangle
            bounds={bounds}
            eventHandlers={{ click: () => onSelectTile(tile) }}
            key={`${tile.tile_id}:selection`}
            pathOptions={selectedTilePathOptions(tile, selectedTileId)}
          />
        );
      })}
    </>
  );
}

export function DatasetMap({
  city,
  exportName,
  label,
  resetToken,
  runName,
  selectedTileId,
  split,
  onError,
  onPayload,
  onSelectTile,
}: DatasetMapProps) {
  const [loading, setLoading] = useState(false);
  const [payload, setPayload] = useState<DatasetViewportPayload | null>(null);
  const [fitToken, setFitToken] = useState(0);
  const [baseLayer, setBaseLayer] = useState<BaseLayerId>("osm");
  const [maskLayer, setMaskLayer] = useState<MaskLayerId>("all");
  const fitSignal = `${resetToken}:${fitToken}`;
  const activeBaseLayer = BASE_LAYERS[baseLayer];
  const statusText = useMemo(() => {
    if (payload?.mode === "clusters") return `${payload.returned_clusters} clusters`;
    return `${payload?.returned_tiles ?? 0} tiles`;
  }, [payload]);

  const handlePayload = useCallback(
    (next: DatasetViewportPayload | null) => {
      setPayload(next);
      onPayload(next);
    },
    [onPayload],
  );

  return (
    <div className="relative h-full w-full overflow-hidden bg-slate-900">
      <MapContainer
        attributionControl
        center={SWITZERLAND_CENTER}
        className="h-full w-full bg-slate-900 font-sans"
        maxBounds={[
          [45.3, 5.2],
          [48.3, 11.2],
        ]}
        maxZoom={19}
        minZoom={7}
        scrollWheelZoom
        zoom={DEFAULT_ZOOM}
        zoomControl={false}
      >
        <TileLayer
          attribution={activeBaseLayer.attribution}
          key={baseLayer}
          maxNativeZoom={activeBaseLayer.maxNativeZoom}
          maxZoom={19}
          url={activeBaseLayer.url}
        />
        <FitSwitzerland resetToken={fitSignal} />
        <DatasetViewportLayer
          baseLayer={baseLayer}
          city={city}
          exportName={exportName}
          label={label}
          maskLayer={maskLayer}
          resetSignal={fitSignal}
          runName={runName}
          selectedTileId={selectedTileId}
          split={split}
          onError={onError}
          onLoading={setLoading}
          onPayload={handlePayload}
          onSelectTile={onSelectTile}
        />
      </MapContainer>

      <div className="pointer-events-none absolute left-4 top-4 z-[700] flex flex-wrap gap-2">
        <Chip variant="secondary">{statusText}</Chip>
        <Chip variant="secondary">{payload?.total_matching ?? 0} visible</Chip>
        <Chip variant="secondary">zoom {payload?.zoom.toFixed(1) ?? DEFAULT_ZOOM.toFixed(1)}</Chip>
        {loading ? <Chip variant="secondary">loading</Chip> : null}
      </div>
      <div className="absolute right-4 top-4 z-[700] flex max-w-[calc(100%-2rem)] flex-wrap items-center justify-end gap-2">
        <div className="flex overflow-hidden rounded-full border border-white/15 bg-[#151b22]/90 px-1 py-1 shadow-sm">
          {(Object.keys(BASE_LAYERS) as BaseLayerId[]).map((layerId) => (
            <Button
              aria-label={`Use ${BASE_LAYERS[layerId].label} base map`}
              className={baseLayer === layerId ? "bg-primary text-primary-foreground" : "bg-transparent text-slate-100"}
              key={layerId}
              size="sm"
              variant={baseLayer === layerId ? "primary" : "secondary"}
              onPress={() => setBaseLayer(layerId)}
            >
              {BASE_LAYERS[layerId].label}
            </Button>
          ))}
        </div>
        <div className="flex overflow-hidden rounded-full border border-white/15 bg-[#151b22]/90 px-1 py-1 shadow-sm">
          {(Object.keys(MASK_LAYERS) as MaskLayerId[]).map((layerId) => (
            <Button
              aria-label={`${MASK_LAYERS[layerId]} layer`}
              className={maskLayer === layerId ? "bg-primary text-primary-foreground" : "bg-transparent text-slate-100"}
              key={layerId}
              size="sm"
              variant={maskLayer === layerId ? "primary" : "secondary"}
              onPress={() => setMaskLayer(layerId)}
            >
              {MASK_LAYERS[layerId]}
            </Button>
          ))}
        </div>
        <Button isIconOnly size="sm" variant="secondary" onPress={() => setFitToken((current) => current + 1)} aria-label="Fit Switzerland">
          <Maximize2 className="size-4" />
        </Button>
      </div>
    </div>
  );
}
