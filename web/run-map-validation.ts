import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

type ValidationCaseName =
  | "idle-stability"
  | "zoom-sequence"
  | "circle-coverage"
  | "pan-sequence"
  | "fit-sequence"
  | "scan-sequence"
  | "persistence-sequence";

type ValidationSnapshot = {
  status: "idle" | "running" | "completed" | "failed";
  verdicts: Array<{ name: string; pass: boolean; detail: string; metrics?: Record<string, number> }>;
  counters: { consoleErrors: number };
};

const validationCases: ValidationCaseName[] = [
  "idle-stability",
  "zoom-sequence",
  "circle-coverage",
  "pan-sequence",
  "fit-sequence",
  "scan-sequence",
  "persistence-sequence",
];

const WEB_ROOT = resolve(import.meta.dir);
const PROJECT_ROOT = resolve(WEB_ROOT, "..");
const BASE_URL = process.env.MAP_VALIDATION_URL ?? "http://crosswalk-review.localhost:1355";
const API_URL = process.env.MAP_VALIDATION_API_URL ?? "http://127.0.0.1:8787";
const DATASET_RUN = process.env.MAP_VALIDATION_DATASET_RUN ?? "real-v1";
const DATASET_EXPORT = process.env.MAP_VALIDATION_DATASET_EXPORT ?? "real-balanced-256";
const SESSION = "map-validation";
const RUN_ID = process.env.MAP_VALIDATION_RUN_ID ?? `run-${new Date().toISOString().replace(/[:.]/g, "-")}`;
const OUTPUT_ROOT = join(PROJECT_ROOT, "validation-output", "map-canvas", RUN_ID);

function runCommand(command: string, args: string[]) {
  const result = Bun.spawnSync([command, ...args], {
    stdout: "pipe",
    stderr: "pipe",
    cwd: WEB_ROOT,
    env: process.env,
  });
  if (result.exitCode !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed:\n${result.stderr.toString() || result.stdout.toString()}`);
  }
  return result.stdout.toString().trim();
}

async function fetchValidationState(validationCase: ValidationCaseName) {
  const response = await fetch(
    `${API_URL}/api/map-validation-state?validationRunId=${encodeURIComponent(RUN_ID)}&validationCase=${encodeURIComponent(validationCase)}`,
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch validation state for ${validationCase}: ${response.status}`);
  }
  return (await response.json()) as ValidationSnapshot | null;
}

async function prepareBaseReviewState() {
  const metaResponse = await fetch(
    `${API_URL}/api/dataset-meta?run=${encodeURIComponent(DATASET_RUN)}&export=${encodeURIComponent(DATASET_EXPORT)}`,
  );
  if (!metaResponse.ok) {
    throw new Error(`Failed to fetch dataset meta: ${metaResponse.status}`);
  }
  const meta = (await metaResponse.json()) as { scenes: Array<{ scene_id: string }> };
  const preferredSceneId = meta.scenes.find((scene) => scene.scene_id === "bern-center")?.scene_id ?? meta.scenes[0]?.scene_id;
  if (!preferredSceneId) {
    throw new Error("No scene available for validation preparation.");
  }

  const sceneResponse = await fetch(
    `${API_URL}/api/scene?run=${encodeURIComponent(DATASET_RUN)}&export=${encodeURIComponent(DATASET_EXPORT)}&scene=${encodeURIComponent(preferredSceneId)}`,
  );
  if (!sceneResponse.ok) {
    throw new Error(`Failed to fetch base scene: ${sceneResponse.status}`);
  }
  const scene = (await sceneResponse.json()) as { tiles: Array<{ tile_id: string }> };
  const selectedTileId = scene.tiles[0]?.tile_id;
  const reviewState = {
    selected_scene_id: preferredSceneId,
    selected_tile_id: selectedTileId,
    map_zoom: 16.65,
    scenes: {
      [preferredSceneId]: {
        scan_radius: 6,
        scan_delay_ms: 24,
        scanned_tile_ids: [],
      },
    },
  };

  const saveResponse = await fetch(
    `${API_URL}/api/review-state?run=${encodeURIComponent(DATASET_RUN)}&export=${encodeURIComponent(DATASET_EXPORT)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reviewState),
    },
  );
  if (!saveResponse.ok) {
    throw new Error(`Failed to prime review state: ${saveResponse.status}`);
  }
}

async function waitForValidationCase(validationCase: ValidationCaseName, timeoutMs: number) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const snapshot = await fetchValidationState(validationCase);
    if (snapshot && (snapshot.status === "completed" || snapshot.status === "failed")) {
      return snapshot;
    }
    await Bun.sleep(500);
  }
  throw new Error(`Timed out waiting for ${validationCase}`);
}

async function main() {
  mkdirSync(OUTPUT_ROOT, { recursive: true });
  runCommand("agent-browser", ["--session", SESSION, "set", "viewport", "1440", "960"]);

  const reportLines = [`# Map validation run`, ``, `- Run ID: \`${RUN_ID}\``, `- Base URL: ${BASE_URL}`, ``];

  for (const validationCase of validationCases) {
    const caseDir = join(OUTPUT_ROOT, validationCase);
    mkdirSync(caseDir, { recursive: true });
    await prepareBaseReviewState();
    const url = `${BASE_URL}/?mapDebug=1&validationRunId=${encodeURIComponent(RUN_ID)}&validationCase=${encodeURIComponent(validationCase)}`;
    runCommand("agent-browser", ["--session", SESSION, "open", url]);
    runCommand("agent-browser", ["--session", SESSION, "wait", "1200"]);

    const timeoutMs = validationCase === "idle-stability" ? 16_000 : validationCase === "persistence-sequence" ? 20_000 : 8_000;
    const snapshot = await waitForValidationCase(validationCase, timeoutMs);

    runCommand("agent-browser", ["--session", SESSION, "screenshot", join(caseDir, "final.png")]);
    writeFileSync(join(caseDir, "state.json"), JSON.stringify(snapshot, null, 2), "utf8");

    const passed = snapshot.verdicts.every((verdict) => verdict.pass);
    reportLines.push(`## ${validationCase}`);
    reportLines.push(`- Status: ${snapshot.status}`);
    reportLines.push(`- Verdict: ${passed ? "PASS" : "FAIL"}`);
    reportLines.push(`- Console errors: ${snapshot.counters.consoleErrors}`);
    for (const verdict of snapshot.verdicts) {
      reportLines.push(`- ${verdict.name}: ${verdict.pass ? "pass" : "fail"}${verdict.detail ? ` — ${verdict.detail}` : ""}`);
    }
    reportLines.push(`- Screenshot: \`${join("validation-output", "map-canvas", RUN_ID, validationCase, "final.png")}\``);
    reportLines.push("");
  }

  runCommand("agent-browser", ["--session", SESSION, "close"]);
  writeFileSync(join(OUTPUT_ROOT, "report.md"), reportLines.join("\n"), "utf8");
  console.log(`Validation artifacts written to ${OUTPUT_ROOT}`);
}

await main();
