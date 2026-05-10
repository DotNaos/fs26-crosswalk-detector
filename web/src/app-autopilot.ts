import type { AutopilotPlan } from "./autopilot-planner";

export const FALLBACK_RUN = "real-v1";
export const FALLBACK_EXPORT = "real-balanced-256";

export function needsRealAutopilotPlanRefresh(autopilot: unknown): autopilot is Partial<AutopilotPlan> {
  const plan = autopilot as Partial<AutopilotPlan> | undefined;
  if (plan?.mode !== "swiss-lowres-urban-grid") return false;
  return plan.version !== 6 || plan.source !== "swisstopo-official-road-density" || !Array.isArray(plan.bvhCells) || plan.bvhCells.length === 0;
}

export function asAutopilotPlan(value: unknown): AutopilotPlan | null {
  const plan = value as Partial<AutopilotPlan> | undefined;
  return plan?.mode === "swiss-lowres-urban-grid" && Array.isArray(plan.scenes) ? (plan as AutopilotPlan) : null;
}
