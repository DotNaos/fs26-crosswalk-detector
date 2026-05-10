import type { Dispatch, SetStateAction } from "react";
import { Circle, CircleMarker, Rectangle, TileLayer, WMSTileLayer } from "react-leaflet";
import type { AutopilotBvhCell, AutopilotPlan } from "../autopilot-planner";
import type { BrowserLabelSuggestion, DatasetScene, DatasetTile, MapBasemap } from "../types";
import { validationClassSuffix, type ValidationInteractionPhase } from "../map-validation";
import { bboxCenterLatLng, bboxToLatLngBounds, tileTone } from "../utils";
import { MAP_BASEMAPS } from "../map-basemaps";
import { MapCamera } from "./MapCamera";
import { MapEventBridge } from "./MapEventBridge";
import { bvhLineColor, bvhLineWeight, dashArrayForZoom } from "./scene-map-geometry";

type SceneMapLayersProps = {
  activeTile?: DatasetTile;
  activeTileDashArray: string;
  basemap: MapBasemap;
  bvhOverlayCells: AutopilotBvhCell[];
  cityFineGridCells: AutopilotPlan["coarseCells"];
  displaySuggestions: Record<string, BrowserLabelSuggestion>;
  droppedTileDashArray: string;
  effectiveMapZoom: number;
  focusSceneId?: string;
  focusedSceneId?: string;
  footprintCenter: [number, number] | null;
  footprintRadiusM: number;
  footprintTiles: DatasetTile[];
  onCameraUpdate: () => void;
  onInteractionPhase: Dispatch<SetStateAction<ValidationInteractionPhase>> | ((phase: ValidationInteractionPhase) => void);
  onMapReady: Parameters<typeof MapEventBridge>[0]["onMapReady"];
  onSelectScene: (sceneId: string) => void;
  onSelectTile: (tile: DatasetTile) => void;
  onZoomChange: (zoom: number) => void;
  scanCircleDashArray: string;
  scannedTileIdSet: Set<string>;
  scenes: DatasetScene[];
  selectedBounds: ReturnType<typeof bboxToLatLngBounds>;
  selectedSceneId?: string;
  selectedTileId?: string;
  showAutopilotDetailGrid: boolean;
  showGrid: boolean;
  visibleAutopilotPlan: AutopilotPlan | null;
};

export function SceneMapLayers({
  activeTile,
  activeTileDashArray,
  basemap,
  bvhOverlayCells,
  cityFineGridCells,
  displaySuggestions,
  droppedTileDashArray,
  effectiveMapZoom,
  focusSceneId,
  focusedSceneId,
  footprintCenter,
  footprintRadiusM,
  footprintTiles,
  onCameraUpdate,
  onInteractionPhase,
  onMapReady,
  onSelectScene,
  onSelectTile,
  onZoomChange,
  scanCircleDashArray,
  scannedTileIdSet,
  scenes,
  selectedBounds,
  selectedSceneId,
  selectedTileId,
  showAutopilotDetailGrid,
  showGrid,
  visibleAutopilotPlan,
}: SceneMapLayersProps) {
  const basemapConfig = MAP_BASEMAPS[basemap];
  const activeTileBounds = activeTile ? bboxToLatLngBounds(activeTile.bbox_mercator) : null;
  const activeTileCenter = activeTile ? bboxCenterLatLng(activeTile.bbox_mercator) : null;

  return (
    <>
      <TileLayer
        attribution={basemapConfig.attribution}
        crossOrigin="anonymous"
        key={basemap}
        maxZoom={basemapConfig.maxZoom}
        keepBuffer={6}
        updateWhenIdle
        updateWhenZooming={false}
        url={basemapConfig.url}
      />
      {basemapConfig.roadsOverlay ? (
        <WMSTileLayer
          attribution={basemapConfig.attribution}
          format="image/png"
          layers="ch.swisstopo.swisstlm3d-strassen"
          opacity={0.95}
          transparent
          url="https://wms.geo.admin.ch/"
          version="1.3.0"
        />
      ) : null}
      <MapEventBridge onMapReady={onMapReady} onZoomChange={onZoomChange} onCameraUpdate={onCameraUpdate} onInteractionPhase={onInteractionPhase} />
      <MapCamera
        bounds={selectedBounds}
        focusKey={focusSceneId ?? (visibleAutopilotPlan ? `autopilot-overview-${visibleAutopilotPlan.createdAt}` : focusedSceneId)}
        onFitPhase={onInteractionPhase}
      />

      {!visibleAutopilotPlan
        ? scenes.map((scene) => {
            const latitude = Number(scene.latitude ?? 0);
            const longitude = Number(scene.longitude ?? 0);
            const isSelected = scene.scene_id === selectedSceneId;
            return (
              <CircleMarker
                key={scene.scene_id}
                center={[latitude, longitude]}
                radius={isSelected ? 10 : 7}
                pathOptions={{
                  color: isSelected ? "#f5c05d" : "#29c37d",
                  weight: isSelected ? 3 : 2,
                  fillOpacity: isSelected ? 0.74 : 0.45,
                }}
                eventHandlers={{ click: () => onSelectScene(scene.scene_id) }}
              />
            );
          })
        : null}

      {cityFineGridCells.map((cell) => {
        const bounds = bboxToLatLngBounds(cell.bboxMercator);
        if (!bounds) return null;
        const isUrban = cell.status === "urban";
        return (
          <Rectangle
            key={`autopilot-city-grid-${cell.id}`}
            bounds={bounds}
            interactive={false}
            pathOptions={{
              color: isUrban ? "#34d399" : "#67e8f9",
              weight: isUrban ? 1.15 : 0.85,
              opacity: isUrban ? 0.7 : 0.42,
              fillColor: isUrban ? "#22c55e" : "#06b6d4",
              fillOpacity: isUrban ? 0.045 : 0.018,
              lineCap: "square",
              lineJoin: "round",
              dashArray: isUrban ? undefined : "4 6",
              className: `autopilot-city-grid autopilot-city-grid-${cell.status}`,
            }}
          />
        );
      })}

      {showAutopilotDetailGrid
        ? visibleAutopilotPlan?.panels.map((panel) => {
            const bounds = bboxToLatLngBounds(panel.bboxMercator);
            if (!bounds) return null;
            return (
              <Rectangle
                key={`autopilot-panel-${panel.id}`}
                bounds={bounds}
                pathOptions={{
                  color: panel.plannedScenes > 0 ? "#38bdf8" : "#64748b",
                  weight: panel.plannedScenes > 0 ? 1.2 : 0.8,
                  opacity: panel.plannedScenes > 0 ? 0.34 : 0.18,
                  fillColor: "#38bdf8",
                  fillOpacity: 0,
                  dashArray: dashArrayForZoom(effectiveMapZoom, 10, 10),
                }}
              />
            );
          })
        : null}

      {showAutopilotDetailGrid
        ? visibleAutopilotPlan?.cells
            .filter((cell) => cell.status !== "panel")
            .map((cell) => {
              const bounds = bboxToLatLngBounds(cell.bboxMercator);
              if (!bounds) return null;
              const isSelectedCell = cell.sceneId === selectedSceneId;
              const isPlannedScene = Boolean(cell.sceneId);
              return (
                <Rectangle
                  key={`autopilot-cell-${cell.id}`}
                  bounds={bounds}
                  pathOptions={{
                    color: isSelectedCell ? "#fff2c2" : isPlannedScene ? "#22c55e" : "#f59e0b",
                    weight: isSelectedCell ? 4 : isPlannedScene ? 1.5 : 1,
                    opacity: isSelectedCell ? 1 : 0.58,
                    fillColor: isPlannedScene ? "#22c55e" : "#f59e0b",
                    fillOpacity: isSelectedCell ? 0.08 : 0,
                    dashArray: isPlannedScene ? undefined : dashArrayForZoom(effectiveMapZoom, 6, 8),
                    className: `autopilot-cell autopilot-cell-${cell.status}`,
                  }}
                  eventHandlers={cell.sceneId ? { click: () => onSelectScene(cell.sceneId!) } : undefined}
                />
              );
            })
        : null}

      {bvhOverlayCells.map((cell) => {
        const bounds = bboxToLatLngBounds(cell.bboxMercator);
        if (!bounds) return null;
        return (
          <Rectangle
            key={`autopilot-bvh-halo-${cell.id}`}
            bounds={bounds}
            interactive={false}
            pathOptions={{
              color: "#020617",
              weight: bvhLineWeight(cell.depth) + 2,
              opacity: cell.depth <= 2 ? 0.5 : 0.34,
              fillOpacity: 0,
              lineCap: "square",
              lineJoin: "round",
              dashArray: "18 10",
              className: `autopilot-bvh autopilot-bvh-halo autopilot-bvh-depth-${cell.depth}`,
            }}
          />
        );
      })}

      {bvhOverlayCells.map((cell) => {
        const bounds = bboxToLatLngBounds(cell.bboxMercator);
        if (!bounds) return null;
        return (
          <Rectangle
            key={`autopilot-bvh-line-${cell.id}`}
            bounds={bounds}
            interactive={false}
            pathOptions={{
              color: bvhLineColor(cell),
              weight: bvhLineWeight(cell.depth),
              opacity: cell.depth <= 2 ? 0.96 : 0.86,
              fillOpacity: 0,
              lineCap: "square",
              lineJoin: "round",
              dashArray: "18 10",
              className: `autopilot-bvh autopilot-bvh-depth-${cell.depth}`,
            }}
          />
        );
      })}

      {showGrid && footprintCenter ? (
        <>
          <Circle
            center={footprintCenter}
            radius={footprintRadiusM * 1.02}
            pathOptions={{
              color: "rgba(255, 246, 214, 0.9)",
              weight: 8,
              opacity: 0.34,
              fillColor: "rgba(245, 192, 93, 0.1)",
              fillOpacity: 0.1,
              className: "scan-circle-glow",
            }}
          />
          <Circle
            center={footprintCenter}
            radius={footprintRadiusM}
            pathOptions={{
              color: "rgba(245, 192, 93, 0.98)",
              weight: 3,
              fillColor: "rgba(245, 192, 93, 0.08)",
              fillOpacity: 0.08,
              dashArray: scanCircleDashArray,
              className: "scan-circle-ring validation-scan-circle",
            }}
          />
          <CircleMarker
            center={footprintCenter}
            radius={7}
            pathOptions={{
              color: "#fff2c2",
              weight: 2,
              fillColor: "#f5c05d",
              fillOpacity: 0.95,
              className: "scan-circle-center",
            }}
          />
        </>
      ) : null}

      {showGrid
        ? footprintTiles.map((tile) => {
            const bounds = bboxToLatLngBounds(tile.bbox_mercator);
            if (!bounds) return null;
            const isScanned = scannedTileIdSet.has(tile.tile_id);
            const tone = tileTone(tile);
            const isSelectedTile = tile.tile_id === selectedTileId;
            const isActive = tile.tile_id === activeTile?.tile_id;
            const browserSuggestion = displaySuggestions[tile.tile_id];
            const displayTone = browserSuggestion?.label === "crosswalk" ? "crosswalk" : browserSuggestion?.label === "no_crosswalk" ? "no_crosswalk" : tone;
            const color = displayTone === "crosswalk" ? "#29c37d" : displayTone === "no_crosswalk" ? "#ff6b6b" : "rgba(141, 161, 184, 0.8)";

            return (
              <Rectangle
                key={tile.tile_id}
                bounds={bounds}
                pathOptions={{
                  className: `validation-tile validation-tile-${validationClassSuffix(tile.tile_id)}${isSelectedTile ? " validation-selected-tile" : ""}${isActive ? " validation-active-tile" : ""}`,
                  color: isSelectedTile ? "#f5c05d" : isActive ? "#f5c05d" : color,
                  weight: isSelectedTile || isActive ? 4 : browserSuggestion ? 3 : isScanned ? 3 : 2,
                  fillColor: color,
                  fillOpacity: browserSuggestion ? 0.28 : isScanned ? 0.44 : 0.18,
                  dashArray: browserSuggestion ? activeTileDashArray : tone === "dropped" ? droppedTileDashArray : undefined,
                  opacity: isScanned ? 1 : 0.92,
                }}
                eventHandlers={{ click: () => onSelectTile(tile) }}
              />
            );
          })
        : null}

      {showGrid && activeTileBounds ? (
        <Rectangle
          bounds={activeTileBounds}
          pathOptions={{
            color: "#fff2c2",
            weight: 5,
            opacity: 1,
            fillColor: "#f5c05d",
            fillOpacity: 0.18,
            dashArray: activeTileDashArray,
            className: "scan-active-rect",
          }}
        />
      ) : null}

      {showGrid && activeTileCenter ? (
        <CircleMarker
          center={activeTileCenter}
          radius={8}
          pathOptions={{
            color: "#fff8dd",
            weight: 2,
            fillColor: "#f5c05d",
            fillOpacity: 1,
            className: "scan-active-dot",
          }}
        />
      ) : null}
    </>
  );
}
