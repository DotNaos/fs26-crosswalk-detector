import { useMemo } from "react";
import type { DatasetScene, DatasetTile, SceneReviewState } from "../types";
import { computeSceneFootprint } from "../scene-footprint";

type SceneFootprintArgs = {
  scene?: DatasetScene;
  sceneTiles: DatasetTile[];
  sceneReviewState: SceneReviewState;
  mode?: "radius" | "scene";
};

export function useSceneFootprint({ scene, sceneTiles, sceneReviewState, mode = "radius" }: SceneFootprintArgs) {
  return useMemo(
    () =>
      computeSceneFootprint({
        scene,
        sceneTiles,
        sceneReviewState,
        mode,
      }),
    [mode, scene, sceneReviewState, sceneTiles],
  );
}
