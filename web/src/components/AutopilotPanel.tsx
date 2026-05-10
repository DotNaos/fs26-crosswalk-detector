import { useEffect, useMemo, useState } from "react";
import { Button, Card, Input, Label, TextField } from "@heroui/react";
import { Bot, Layers3, LoaderCircle, Network, Sparkles } from "lucide-react";
import { buildAutopilotPlan, type AutopilotPlan } from "../autopilot-planner";
import { loadAutopilotPlan } from "../api";
import type { DatasetSummary, RealDatasetConfig } from "../types";

export type AutopilotCreateInput = {
  targetPositiveCount: number;
  maxPanels: number;
  perimeterBudget: number;
};

type AutopilotPanelProps = {
  config?: RealDatasetConfig;
  summary?: DatasetSummary | null;
  compact?: boolean;
  creating?: boolean;
  onCreate: (input: AutopilotCreateInput) => void;
  onPreviewPlanChange?: (plan: AutopilotPlan | null) => void;
};

function asAutopilotPlan(value: unknown): AutopilotPlan | null {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<AutopilotPlan>;
  return candidate.mode === "swiss-lowres-urban-grid" && Array.isArray(candidate.scenes) ? (candidate as AutopilotPlan) : null;
}

function MiniPanelMap({ intensity }: { intensity: number }) {
  return (
    <div className="grid h-12 grid-cols-6 gap-0.5 rounded-[12px] bg-black/20 p-1 [corner-shape:squircle]">
      {Array.from({ length: 18 }, (_, index) => {
        const active = (index * 17 + Math.round(intensity * 100)) % 9 < 4;
        return (
          <span
            aria-hidden
            className={`rounded-[4px] ${active ? "bg-sky-400/65" : "bg-white/8"}`}
            key={index}
          />
        );
      })}
    </div>
  );
}

export function AutopilotPanel({ config, summary, compact = false, creating = false, onCreate, onPreviewPlanChange }: AutopilotPanelProps) {
  const activePlan = asAutopilotPlan(config?.autopilot);
  const [target, setTarget] = useState(() => activePlan?.targetPositiveCount ?? 500);
  const [maxPanels, setMaxPanels] = useState(() => activePlan?.maxPanels ?? 8);
  const [perimeterBudget, setPerimeterBudget] = useState(() => activePlan?.sceneBudget ?? 72);
  const fallbackPreviewPlan = useMemo(
    () => buildAutopilotPlan({ targetPositiveCount: target, maxPanels, perimeterBudget }),
    [maxPanels, perimeterBudget, target],
  );
  const [realPreviewPlan, setRealPreviewPlan] = useState<AutopilotPlan | null>(activePlan);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const plan = activePlan ?? realPreviewPlan ?? fallbackPreviewPlan;
  const positiveCount = summary?.selected_crosswalk ?? 0;
  const progress = Math.min(100, Math.round((positiveCount / Math.max(1, plan.targetPositiveCount)) * 100));
  const visiblePanels = plan.panels.slice(0, compact ? 5 : 8);

  useEffect(() => {
    onPreviewPlanChange?.(activePlan ?? realPreviewPlan ?? null);
  }, [activePlan, onPreviewPlanChange, realPreviewPlan]);

  useEffect(() => {
    if (activePlan) {
      setRealPreviewPlan(activePlan);
      setPreviewError(null);
      setPreviewLoading(false);
      return;
    }
    let canceled = false;
    setPreviewLoading(true);
    const timer = window.setTimeout(() => {
      loadAutopilotPlan(target, maxPanels, perimeterBudget)
        .then((nextPlan) => {
          if (canceled) return;
          setRealPreviewPlan(nextPlan);
          setPreviewError(null);
        })
        .catch((reason) => {
          if (canceled) return;
          setPreviewError(String(reason));
          setRealPreviewPlan(null);
        })
        .finally(() => {
          if (!canceled) setPreviewLoading(false);
        });
    }, 240);
    return () => {
      canceled = true;
      window.clearTimeout(timer);
    };
  }, [activePlan, maxPanels, perimeterBudget, target]);

  const content = (
    <>
      <div className="flex min-w-0 items-start gap-3">
        <div className="grid size-10 shrink-0 place-items-center rounded-[16px] bg-primary/15 text-primary [corner-shape:squircle]">
          <Bot className="size-5" />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-base font-semibold">Dataset autopilot</h3>
            {activePlan ? <span className="rounded-full bg-success/15 px-2 py-0.5 text-[11px] font-semibold text-success">active</span> : null}
            {plan.source === "swisstopo-official-road-density" ? <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[11px] font-semibold text-primary">roads</span> : null}
            {plan.source === "swisstopo-swissimage" ? <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[11px] font-semibold text-primary">swissimage</span> : null}
          </div>
          <p className="text-sm text-foreground/65">
            Builds a SwissTopo road-density surface, drills into the BVH grid, then opens the best verified cells as scan panels.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-[16px] bg-content1/70 p-2 [corner-shape:squircle]">
          <div className="text-lg font-semibold">{plan.scenes.length}</div>
          <div className="text-foreground/60">perimeters</div>
        </div>
        <div className="rounded-[16px] bg-content1/70 p-2 [corner-shape:squircle]">
          <div className="text-lg font-semibold">{plan.panels.length}</div>
          <div className="text-foreground/60">panels</div>
        </div>
        <div className="rounded-[16px] bg-content1/70 p-2 [corner-shape:squircle]">
          <div className="text-lg font-semibold">{progress}%</div>
          <div className="text-foreground/60">target</div>
        </div>
      </div>

      <TextField variant="secondary">
        <Label>Target positives</Label>
        <Input
          min={20}
          step={10}
          type="number"
          value={String(target)}
          onChange={(event) => setTarget(Math.max(20, Number(event.target.value) || 500))}
        />
      </TextField>

      <div className="grid grid-cols-2 gap-2">
        <TextField variant="secondary">
          <Label>Active panels</Label>
          <Input
            max={24}
            min={1}
            type="number"
            value={String(maxPanels)}
            onChange={(event) => setMaxPanels(Math.max(1, Math.min(24, Number(event.target.value) || 8)))}
          />
        </TextField>
        <TextField variant="secondary">
          <Label>Perimeters</Label>
          <Input
            max={120}
            min={4}
            type="number"
            value={String(perimeterBudget)}
            onChange={(event) => setPerimeterBudget(Math.max(4, Math.min(120, Number(event.target.value) || 72)))}
          />
        </TextField>
      </div>

      <div className="grid max-h-48 min-h-0 grid-cols-1 gap-2 overflow-y-auto rounded-[18px] bg-content1/45 p-2 [corner-shape:squircle]" data-scroll-guard="autopilot-panel-list">
        {visiblePanels.map((panel) => (
          <div key={panel.id} className="grid grid-cols-[4.5rem,1fr,auto] items-center gap-2 rounded-[14px] bg-black/10 p-2 text-xs [corner-shape:squircle]">
            <MiniPanelMap intensity={panel.urbanScore} />
            <span className="min-w-0">
              <span className="block truncate font-semibold">{panel.name}</span>
              <span className="block truncate text-foreground/60">
                {Math.round(panel.urbanScore * 100)}% urban · {panel.coarseCellCount} cells
              </span>
            </span>
            <span className="inline-flex items-center gap-1 rounded-[12px] bg-primary/15 px-2 py-1 font-semibold text-primary [corner-shape:squircle]">
              <Layers3 className="size-3.5" />
              {panel.plannedScenes}
            </span>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2 text-xs text-foreground/65">
        <span className="inline-flex items-center gap-1 rounded-full bg-content1/70 px-2 py-1">
          <Network className="size-3.5" />
          {plan.bvhCells?.length ?? plan.coarseCells.length} BVH cells
        </span>
        <span className="inline-flex items-center gap-1 rounded-full bg-content1/70 px-2 py-1">
          {previewLoading ? <LoaderCircle className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}
          {previewLoading ? "loading real data" : `est. ${plan.estimatedPositiveCount} positives`}
        </span>
      </div>

      {previewError ? (
        <div className="rounded-[14px] bg-danger/12 px-3 py-2 text-xs text-danger [corner-shape:squircle]">
          Real Swissimage plan failed: {previewError}
        </div>
      ) : null}

      <Button
        className="rounded-[16px] [corner-shape:squircle]"
        isDisabled={creating || (!activePlan && previewLoading)}
        onPress={() => onCreate({ targetPositiveCount: target, maxPanels, perimeterBudget })}
        variant={activePlan ? "secondary" : "primary"}
      >
        {creating ? <LoaderCircle className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
        {activePlan ? "Create new autopilot dataset" : "Generate autopilot dataset"}
      </Button>
    </>
  );

  if (compact) {
    return <div className="flex w-full min-w-0 flex-col gap-3">{content}</div>;
  }

  return (
    <Card className="pointer-events-auto max-w-md shadow-xl" variant="secondary">
      <Card.Content className="flex flex-col gap-4">{content}</Card.Content>
    </Card>
  );
}
