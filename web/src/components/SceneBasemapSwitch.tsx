import { Button } from "@heroui/react";
import { Globe, Route, Satellite } from "lucide-react";
import type { MapBasemap } from "../types";

type SceneBasemapSwitchProps = {
  basemap: MapBasemap;
  onBasemapChange: (next: MapBasemap) => void;
};

export function SceneBasemapSwitch({ basemap, onBasemapChange }: SceneBasemapSwitchProps) {
  return (
    <div className="pointer-events-auto flex items-center gap-2 rounded-full bg-black/72 p-2 shadow-xl backdrop-blur">
      <Button onClick={() => onBasemapChange("osm")} variant={basemap === "osm" ? "primary" : "ghost"}>
        <Globe className="size-4" />
        Map
      </Button>
      <Button onClick={() => onBasemapChange("swisstopo")} variant={basemap === "swisstopo" ? "primary" : "ghost"}>
        <Satellite className="size-4" />
        Satellite
      </Button>
      <Button onClick={() => onBasemapChange("roads")} variant={basemap === "roads" ? "primary" : "ghost"}>
        <Route className="size-4" />
        Roads
      </Button>
    </div>
  );
}
