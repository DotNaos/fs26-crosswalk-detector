import type { MapBasemap } from "./types";

export const MAP_BASEMAPS: Record<
  MapBasemap,
  {
    label: string;
    url: string;
    attribution: string;
    maxZoom?: number;
    roadsOverlay?: boolean;
  }
> = {
  osm: {
    label: "OSM",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 19,
  },
  swisstopo: {
    label: "SWISSIMAGE",
    url: "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/current/3857/{z}/{x}/{y}.jpeg",
    attribution: '&copy; <a href="https://www.swisstopo.admin.ch/">swisstopo</a>',
    maxZoom: 19,
  },
  roads: {
    label: "ROADS",
    url: "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/current/3857/{z}/{x}/{y}.jpeg",
    attribution: '&copy; <a href="https://www.swisstopo.admin.ch/">swisstopo</a>',
    maxZoom: 19,
    roadsOverlay: true,
  },
};
