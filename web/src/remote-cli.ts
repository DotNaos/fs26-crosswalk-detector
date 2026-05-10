import { copyFileSync, existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadDataset } from "./dataset";
import { loadLocalEnv } from "./local-env";
import { buildDatasetScanJob, listDatasetScenes } from "./remote-job-builder";
import { createRemoteController } from "./remote-controller";
import type { RemoteControllerSnapshot, RemoteScanJobRecord } from "./types";

const WEB_ROOT = resolve(import.meta.dir, "..");
const PROJECT_ROOT = resolve(WEB_ROOT, "..");
const JOBS_ROOT = join(PROJECT_ROOT, ".local", "remote-controller", "jobs");
const DEFAULT_RUN = "real-v1";
const DEFAULT_EXPORT = "real-balanced-256";
const DEFAULT_THRESHOLD = 0.32;
const DEFAULT_PROMPT = "server-side hybrid scan";

loadLocalEnv(PROJECT_ROOT, WEB_ROOT);

const controller = createRemoteController(PROJECT_ROOT);

function usage() {
  console.log(`crosswalk remote cli

Commands
  snapshot [--json]
  servers
  scenes [--run <name>] [--export <name>]
  connect [--server <id>]
  run --scene <id> [--run <name>] [--export <name>] [--radius <n>] [--server <id>] [--threshold <n>] [--prompt <text>] [--detach] [--output <path>]
  jobs [--json]
  log --job <id> [--follow]
  result --job <id> [--json] [--output <path>]
  cancel --job <id>
`);
}

function requireCommand<T>(value: T | undefined, message: string): T {
  if (value == null || value === "") {
    throw new Error(message);
  }
  return value;
}

function jobLogPath(jobId: string) {
  return join(JOBS_ROOT, jobId, "run.log");
}

function jobResultPath(jobId: string) {
  return join(JOBS_ROOT, jobId, "result.json");
}

function sleep(ms: number) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

function isDone(status: RemoteScanJobRecord["status"]) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function printSnapshot(snapshot: RemoteControllerSnapshot) {
  const selectedServer = snapshot.server_options.find((option) => option.id === snapshot.selected_server_id);
  console.log(`Selected server: ${selectedServer?.label ?? snapshot.config.server_name ?? "none"}`);
  console.log(`Host: ${snapshot.config.host || "n/a"}`);
  console.log(`Mode: ${snapshot.config.execution_mode}`);
  console.log(`Connected: ${snapshot.connected ? "yes" : "no"}`);
  console.log(`Password configured: ${snapshot.password_configured ? "yes" : "no"}`);
  if (snapshot.remote_hostname) {
    console.log(`Remote host: ${snapshot.remote_hostname}`);
  }
  if (snapshot.last_error) {
    console.log(`Last error: ${snapshot.last_error}`);
  }
}

function printJob(job: RemoteScanJobRecord) {
  const summary = job.summary
    ? ` · ${job.summary.total} tiles · ${job.summary.crosswalk} positive · ${job.summary.no_crosswalk} negative`
    : "";
  const remoteState = job.remote_state ? ` · ${job.remote_state}` : "";
  const error = job.error ? ` · ${job.error}` : "";
  console.log(`${job.id} · ${job.scene_label} · ${job.status}${remoteState}${summary}${error}`);
}

function printResultSummary(jobId: string) {
  const result = controller.loadResult(jobId);
  console.log(`Job: ${jobId}`);
  console.log(`Scene: ${result.job.scene.city} · ${result.job.scene.split}`);
  console.log(`Tiles: ${result.summary.total}`);
  console.log(`Crosswalk: ${result.summary.crosswalk}`);
  console.log(`No crosswalk: ${result.summary.no_crosswalk}`);
  console.log(`Result file: ${jobResultPath(jobId)}`);
}

function ensureServer(serverId?: string) {
  let snapshot = controller.getSnapshot();
  if (!serverId) {
    return snapshot;
  }
  const option = snapshot.server_options.find((entry) => entry.id === serverId);
  if (!option) {
    throw new Error(`Unknown server: ${serverId}`);
  }
  snapshot = controller.saveConfig({
    ...snapshot.config,
    server_id: option.id,
  });
  return snapshot;
}

async function followJob(jobId: string, printFullLog: boolean) {
  let offset = 0;
  let lastStatus = "";
  for (;;) {
    const logPath = jobLogPath(jobId);
    if (existsSync(logPath)) {
      const contents = readFileSync(logPath, "utf8");
      if (printFullLog && contents.length > offset) {
        process.stdout.write(contents.slice(offset));
        offset = contents.length;
      }
    }

    const job = controller.getJob(jobId);
    const stateLabel = [job.status, job.remote_state].filter(Boolean).join(" · ");
    if (stateLabel !== lastStatus) {
      console.error(`[remote] ${stateLabel}`);
      lastStatus = stateLabel;
    }
    if (isDone(job.status)) {
      return job;
    }
    await sleep(1000);
  }
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  if (!command) {
    usage();
    return 0;
  }

  switch (command) {
    case "snapshot": {
      const options = parseArgs({
        args: rest,
        options: {
          json: { type: "boolean" },
        },
      });
      const snapshot = controller.getSnapshot();
      if (options.values.json) {
        console.log(JSON.stringify(snapshot, null, 2));
      } else {
        printSnapshot(snapshot);
      }
      return 0;
    }

    case "servers": {
      const snapshot = controller.getSnapshot();
      for (const option of snapshot.server_options) {
        const selected = option.id === snapshot.selected_server_id ? "*" : " ";
        console.log(`${selected} ${option.id} · ${option.label} · ${option.host} · ${option.execution_mode ?? "slurm"}`);
      }
      return 0;
    }

    case "scenes": {
      const options = parseArgs({
        args: rest,
        options: {
          run: { type: "string", default: DEFAULT_RUN },
          export: { type: "string", default: DEFAULT_EXPORT },
        },
      });
      const dataset = loadDataset(String(options.values.run), String(options.values.export));
      for (const scene of listDatasetScenes(dataset)) {
        console.log(`${scene.scene_id} · ${scene.city} · ${scene.split} · ${scene.tile_count} tiles`);
      }
      return 0;
    }

    case "connect": {
      const options = parseArgs({
        args: rest,
        options: {
          server: { type: "string" },
          json: { type: "boolean" },
        },
      });
      ensureServer(options.values.server);
      const snapshot = controller.connect();
      if (options.values.json) {
        console.log(JSON.stringify(snapshot, null, 2));
      } else {
        printSnapshot(snapshot);
      }
      return 0;
    }

    case "run": {
      const options = parseArgs({
        args: rest,
        options: {
          run: { type: "string", default: DEFAULT_RUN },
          export: { type: "string", default: DEFAULT_EXPORT },
          scene: { type: "string" },
          radius: { type: "string", default: "4" },
          threshold: { type: "string", default: String(DEFAULT_THRESHOLD) },
          prompt: { type: "string", default: DEFAULT_PROMPT },
          server: { type: "string" },
          detach: { type: "boolean", default: false },
          output: { type: "string" },
          json: { type: "boolean", default: false },
        },
      });
      const sceneId = requireCommand(options.values.scene, "--scene is required");
      ensureServer(options.values.server);
      const dataset = loadDataset(String(options.values.run), String(options.values.export));
      const built = buildDatasetScanJob({
        dataset,
        sceneId,
        scanRadiusTiles: Number(options.values.radius),
        threshold: Number(options.values.threshold),
        promptsText: String(options.values.prompt),
      });
      const record = controller.startJob(built.job);
      if (options.values.json) {
        console.log(JSON.stringify(record, null, 2));
      } else {
        console.log(`Started ${record.id} for ${record.scene_label} with ${record.tile_count} tiles.`);
      }
      if (options.values.detach) {
        console.log(`Log file: ${jobLogPath(record.id)}`);
        return 0;
      }
      const finished = await followJob(record.id, true);
      printJob(finished);
      if (finished.status !== "completed") {
        return 1;
      }
      if (options.values.output) {
        copyFileSync(jobResultPath(record.id), String(options.values.output));
        console.log(`Copied result to ${String(options.values.output)}`);
      }
      printResultSummary(record.id);
      return 0;
    }

    case "jobs": {
      const options = parseArgs({
        args: rest,
        options: {
          json: { type: "boolean" },
        },
      });
      const jobs = controller.listJobs();
      if (options.values.json) {
        console.log(JSON.stringify(jobs, null, 2));
      } else {
        for (const job of jobs) {
          printJob(job);
        }
      }
      return 0;
    }

    case "log": {
      const options = parseArgs({
        args: rest,
        options: {
          job: { type: "string" },
          follow: { type: "boolean", default: false },
        },
      });
      const jobId = requireCommand(options.values.job, "--job is required");
      if (!options.values.follow) {
        const logPath = jobLogPath(jobId);
        if (!existsSync(logPath)) {
          throw new Error(`Log file not found for ${jobId}`);
        }
        process.stdout.write(readFileSync(logPath, "utf8"));
        return 0;
      }
      const finished = await followJob(jobId, true);
      return finished.status === "completed" ? 0 : 1;
    }

    case "result": {
      const options = parseArgs({
        args: rest,
        options: {
          job: { type: "string" },
          json: { type: "boolean", default: false },
          output: { type: "string" },
        },
      });
      const jobId = requireCommand(options.values.job, "--job is required");
      const result = controller.loadResult(jobId);
      if (options.values.output) {
        copyFileSync(jobResultPath(jobId), String(options.values.output));
        console.log(`Copied result to ${String(options.values.output)}`);
      }
      if (options.values.json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        printResultSummary(jobId);
      }
      return 0;
    }

    case "cancel": {
      const options = parseArgs({
        args: rest,
        options: {
          job: { type: "string" },
        },
      });
      const jobId = requireCommand(options.values.job, "--job is required");
      const job = controller.cancelJob(jobId);
      printJob(job);
      return 0;
    }

    default:
      usage();
      return 1;
  }
}

main().then(
  (code) => process.exit(code),
  (error) => {
    console.error(String(error));
    process.exit(1);
  },
);
