import { describe, expect, test } from "bun:test";
import {
  GRID_ZOOM_THRESHOLD,
  ROAD_CLUSTER_MIN_ZOOM,
  ROAD_CLUSTER_REQUEST_ZOOM,
  ROAD_CLUSTER_VIEWPORT_ZOOM,
  roadClusterRequestZoomForMapZoom,
  shouldShowAutopilotSegmentation,
  shouldShowFineGrid,
  shouldUseViewportRoadGrid,
} from "./scene-map-geometry";

describe("map layer visibility", () => {
  test("does not request road-clump segmentation while zoomed too far out", () => {
    expect(
      shouldShowAutopilotSegmentation({
        zoom: ROAD_CLUSTER_MIN_ZOOM - 0.1,
      }),
    ).toBe(false);
  });

  test("shows road-clump segmentation once the map is at labeling detail", () => {
    expect(
      shouldShowAutopilotSegmentation({
        hasSceneTiles: true,
        hasSelectedScene: true,
        zoom: ROAD_CLUSTER_MIN_ZOOM,
      }),
    ).toBe(true);
  });

  test("requests road segmentation by city zoom", () => {
    const input = {
      hasSceneTiles: true,
      hasSelectedScene: true,
      zoom: ROAD_CLUSTER_MIN_ZOOM,
    };

    expect(shouldShowFineGrid({ ...input, zoom: GRID_ZOOM_THRESHOLD })).toBe(true);
    expect(shouldShowAutopilotSegmentation(input)).toBe(true);
  });

  test("uses viewport road grid once the city view is visible", () => {
    expect(shouldUseViewportRoadGrid(ROAD_CLUSTER_VIEWPORT_ZOOM - 0.1)).toBe(false);
    expect(shouldUseViewportRoadGrid(ROAD_CLUSTER_VIEWPORT_ZOOM)).toBe(true);
  });

  test("uses a fixed detail request level so grid cells keep constant world size", () => {
    expect(roadClusterRequestZoomForMapZoom(12)).toBe(ROAD_CLUSTER_REQUEST_ZOOM);
    expect(roadClusterRequestZoomForMapZoom(15)).toBe(ROAD_CLUSTER_REQUEST_ZOOM);
    expect(roadClusterRequestZoomForMapZoom(22)).toBe(ROAD_CLUSTER_REQUEST_ZOOM);
  });
});
