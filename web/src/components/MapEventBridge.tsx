import { useEffect } from "react";
import { useMap, useMapEvents } from "react-leaflet";
import type { Map as LeafletMap } from "leaflet";
import type { ValidationInteractionPhase } from "../map-validation";

type MapEventBridgeProps = {
  onMapReady: (map: LeafletMap) => void;
  onZoomChange: (zoom: number) => void;
  onCameraUpdate: () => void;
  onInteractionPhase: (phase: ValidationInteractionPhase) => void;
};

export function MapEventBridge({ onMapReady, onZoomChange, onCameraUpdate, onInteractionPhase }: MapEventBridgeProps) {
  const map = useMapEvents({
    zoomstart() {
      onInteractionPhase("wheel-zoom");
    },
    zoom() {
      onCameraUpdate();
    },
    zoomend() {
      onZoomChange(map.getZoom());
      onCameraUpdate();
      onInteractionPhase("idle");
    },
    movestart(event) {
      onInteractionPhase("originalEvent" in event ? "pan" : "idle");
    },
    move() {
      onCameraUpdate();
    },
    moveend() {
      onZoomChange(map.getZoom());
      onCameraUpdate();
      onInteractionPhase("idle");
    },
    resize() {
      onCameraUpdate();
    },
  });

  useEffect(() => {
    onMapReady(map);
    onZoomChange(map.getZoom());
    onCameraUpdate();
  }, [map, onCameraUpdate, onMapReady, onZoomChange]);

  return null;
}
