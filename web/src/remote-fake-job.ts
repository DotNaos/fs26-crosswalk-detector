import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { buildFakeScanBatchResult, fakeRemoteSuggestion } from "./fake-remote-scan";
import type { ScanBatchJob } from "./scan-batch";

type FakeJobMetadata = {
  local_error_path: string;
  local_log_path: string;
  local_remote_state_path: string;
  local_result_path: string;
  local_status_path: string;
  remote_result_path: string;
};

function sleep(ms: number) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function writeJsonFile(path: string, data: unknown) {
  writeFileSync(path, JSON.stringify(data, null, 2), "utf8");
}

export async function runFakeRemoteJob(metadata: FakeJobMetadata, job: ScanBatchJob) {
  const delayMs = Math.max(40, Number(process.env.CROSSWALK_FAKE_REMOTE_DELAY_MS ?? 420));
  const createdAt = new Date().toISOString();
  const appendLog = (line: string) => appendFileSync(metadata.local_log_path, `${line}\n`, "utf8");

  try {
    writeFileSync(metadata.local_remote_state_path, "FAKE_GPU_BOOTING", "utf8");
    writeFileSync(metadata.local_status_path, "bootstrapping", "utf8");
    appendLog("Preparing fake GPU scan job");
    await sleep(delayMs);
    writeFileSync(metadata.local_status_path, "syncing", "utf8");
    writeFileSync(metadata.local_remote_state_path, "FAKE_SYNCING", "utf8");
    appendLog("Syncing fake repository and cached SAM3.1 fixtures");
    await sleep(delayMs);
    writeFileSync(metadata.local_status_path, "running", "utf8");
    writeFileSync(metadata.local_remote_state_path, "FAKE_GPU_RUNNING", "utf8");
    appendLog("Running simulated SAM3.1 scan on fake GPU");

    let crosswalk = 0;
    let noCrosswalk = 0;
    for (const [index, tile] of job.tiles.entries()) {
      const currentStatus = existsSync(metadata.local_status_path) ? readFileSync(metadata.local_status_path, "utf8").trim() : "";
      if (currentStatus === "cancelled") {
        appendLog("Fake scan cancelled");
        return;
      }
      await sleep(delayMs);
      const suggestion = fakeRemoteSuggestion(job, tile, index);
      if (suggestion.label === "crosswalk") crosswalk += 1;
      else noCrosswalk += 1;
      appendLog(`[${String(index + 1).padStart(4, " ")}/${job.tiles.length}] ${tile.tile_id} -> ${suggestion.label} (${suggestion.score.toFixed(3)})`);
    }

    writeJsonFile(metadata.local_result_path, buildFakeScanBatchResult(job, createdAt));
    writeFileSync(metadata.local_remote_state_path, "FAKE_GPU_COMPLETED", "utf8");
    writeFileSync(metadata.local_status_path, "completed", "utf8");
    appendLog(`Scene ${job.scene.scene_id}: ${job.tiles.length} tiles, ${crosswalk} crosswalk, ${noCrosswalk} no_crosswalk`);
    appendLog(`Results: ${metadata.remote_result_path}`);
    appendLog(`Downloaded result to ${metadata.local_result_path}`);
  } catch (error) {
    writeFileSync(metadata.local_status_path, "failed", "utf8");
    writeFileSync(metadata.local_remote_state_path, "FAKE_GPU_FAILED", "utf8");
    writeFileSync(metadata.local_error_path, String(error), "utf8");
    appendLog(`Fake scan failed: ${String(error)}`);
  }
}
