import { describe, expect, test } from "bun:test";
import { buildAutopilotPlan } from "./autopilot-planner";

describe("buildAutopilotPlan", () => {
  test("creates a full coarse Switzerland grid and selected urban panel perimeters", () => {
    const plan = buildAutopilotPlan({ targetPositiveCount: 500, maxPanels: 6, perimeterBudget: 42 });

    expect(plan.targetPositiveCount).toBe(500);
    expect(plan.mode).toBe("swiss-lowres-urban-grid");
    expect(plan.coarseCells).toHaveLength(plan.coarseGrid.rows * plan.coarseGrid.cols);
    expect(plan.coarseCells.length).toBeGreaterThan(2000);
    expect(plan.coarseCells.every((cell) => cell.bboxMercator.length === 4)).toBe(true);
    expect(plan.panels.length).toBeGreaterThan(10);
    expect(plan.panels.filter((panel) => panel.plannedScenes > 0)).toHaveLength(6);
    expect(plan.scenes.length).toBeGreaterThan(20);
    expect(plan.scenes.length).toBeLessThanOrEqual(42);
    expect(plan.cells.some((cell) => cell.status === "panel")).toBe(true);
    expect(plan.cells.some((cell) => cell.status === "selected" && cell.sceneId)).toBe(true);
    expect(plan.scenes[0].scene_id).toContain("auto-panel-");
  });
});
