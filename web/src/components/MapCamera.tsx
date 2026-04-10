import { useEffect, useRef } from "react";
import { useMap } from "react-leaflet";
import type { ValidationInteractionPhase } from "../map-validation";

type MapCameraProps = {
  bounds: [[number, number], [number, number]] | null;
  focusKey?: string;
  onFitPhase: (phase: ValidationInteractionPhase) => void;
};

export function MapCamera({ bounds, focusKey, onFitPhase }: MapCameraProps) {
  const map = useMap();
  const lastFocusKeyRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!bounds || !focusKey || lastFocusKeyRef.current === focusKey) return;
    lastFocusKeyRef.current = focusKey;
    onFitPhase("fit");
    map.fitBounds(bounds, { padding: [24, 24] });
    window.setTimeout(() => onFitPhase("idle"), 120);
  }, [bounds, focusKey, map, onFitPhase]);

  return null;
}
