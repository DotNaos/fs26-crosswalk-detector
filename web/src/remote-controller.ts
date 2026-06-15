import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { join, resolve } from "node:path";
import { runFakeRemoteJob } from "./remote-fake-job";
import { parseRemoteJobProgressFromLog, progressFromScanBatchResult } from "./remote-job-progress";
import { DEFAULT_REMOTE_CONFIG, DEFAULT_REMOTE_SERVER_OPTIONS } from "./remote-controller-defaults";
import { buildRemoteRunScript, shellQuote } from "./remote-run-script";
import type { ScanBatchJob, ScanBatchResult } from "./scan-batch";
import type { RemoteControllerConfig, RemoteControllerSnapshot, RemoteJobStatus, RemoteScanJobRecord, RemoteServerOption } from "./types";

type RemoteJobMetadata = {
  id: string;
  scene_id: string;
  scene_label: string;
  tile_count: number;
  created_at: string;
  updated_at: string;
  execution_mode: "slurm" | "direct";
  tmux_session: string;
  local_job_path: string;
  local_result_path: string;
  local_log_path: string;
  local_script_path: string;
  local_status_path: string;
  local_remote_state_path: string;
  local_slurm_job_id_path: string;
  local_error_path: string;
  remote_job_path: string;
  remote_result_path: string;
};

function nowIso() {
  return new Date().toISOString();
}

function trimLines(input: string, maxLines = 80) {
  return input
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .slice(-maxLines);
}

function readJsonFile<T>(path: string, fallback: T): T {
  if (!existsSync(path)) return fallback;
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function writeJsonFile(path: string, data: unknown) {
  writeFileSync(path, JSON.stringify(data, null, 2), "utf8");
}

function homePath(path: string) {
  if (path === "~") return process.env.HOME ?? path;
  if (path.startsWith("~/")) return join(process.env.HOME ?? "", path.slice(2));
  return path;
}

function runCommand(command: string, args: string[]) {
  return execFileSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      SSHPASS: process.env.SSHPASS ?? process.env.CROSSWALK_REMOTE_PASSWORD,
    },
  }).trim();
}

function formatRemoteError(error: unknown, host: string) {
  const message = String(error);
  if (message.includes("Could not resolve hostname")) {
    return `Could not reach ${host}. The hostname cannot be resolved right now. You are probably not connected to the school VPN/network.`;
  }
  if (message.includes("Permission denied")) {
    return `Could not log in to ${host}. Check the username and password or SSH key setup.`;
  }
  if (message.includes("executable file not found") && message.includes("sshpass")) {
    return "sshpass is required for password-based SSH. Install sshpass or use SSH keys.";
  }
  if (message.includes("Operation timed out") || message.includes("Connection timed out")) {
    return `Could not reach ${host}. The connection timed out. Check the VPN/network and the server address.`;
  }
  if (message.includes("No route to host")) {
    return `Could not reach ${host}. There is no network route to that server from this machine right now.`;
  }
  return message;
}

function missingRemotePasswordMessage() {
  return "CROSSWALK_REMOTE_PASSWORD is not set. Put it in your local .env file and restart the dev server before connecting.";
}

function missingRemoteToolsMessage(path: string) {
  return `Remote helper tools were not found at ${path}. Set CROSSWALK_REMOTE_TOOLS_DIR or disable CROSSWALK_USE_REMOTE_TOOLS.`;
}

export function createRemoteController(projectRoot: string) {
  const root = join(projectRoot, ".local", "remote-controller");
  const jobsRoot = join(root, "jobs");
  const configPath = join(root, "config.json");
  const statePath = join(root, "state.json");
  const serverOptionsPath = join(root, "server-options.json");
  const expectScriptPath = join(root, "ssh-login.expect");
  mkdirSync(jobsRoot, { recursive: true });

  function remoteToolsDir() {
    return resolve(homePath(process.env.CROSSWALK_REMOTE_TOOLS_DIR ?? ".local/remote-tools"));
  }

  function remoteTool(name: "remote-vpn-check" | "remote-ssh" | "remote-scp") {
    return join(remoteToolsDir(), "bin", name);
  }

  function hasRemoteTools() {
    return existsSync(remoteTool("remote-vpn-check")) && existsSync(remoteTool("remote-ssh")) && existsSync(remoteTool("remote-scp"));
  }

  function useRemoteTools() {
    return process.env.CROSSWALK_USE_REMOTE_TOOLS === "1" && hasRemoteTools();
  }

  function selectedRemoteToolHost(config: RemoteControllerConfig) {
    const option = loadServerOptions().find((entry) => entry.id === config.server_id);
    return option?.hostname ?? config.host;
  }

  function isFakeConfig(config = loadConfig()) {
    return config.server_id === "fake-gpu" || loadServerOptions().some((option) => option.id === config.server_id && option.kind === "fake");
  }

  function requireVpn() {
    if (!useRemoteTools()) return;
    try {
      runCommand(remoteTool("remote-vpn-check"), []);
    } catch (error) {
      throw new Error(`Remote VPN check failed. Connect to the required VPN manually, then retry. ${formatRemoteError(error, "remote VPN")}`);
    }
  }

  function passwordConfiguredOrRemoteTools() {
    if (isFakeConfig()) return true;
    return useRemoteTools() || hasPasswordConfigured();
  }

  function ensureExpectScript() {
    const script = String.raw`set timeout -1
if {![info exists env(CROSSWALK_REMOTE_PASSWORD)] || $env(CROSSWALK_REMOTE_PASSWORD) eq ""} {
  puts stderr "CROSSWALK_REMOTE_PASSWORD is required"
  exit 2
}
set cmd [lindex $argv 0]
set argv [lrange $argv 1 end]
spawn -noecho $cmd {*}$argv
expect {
  -re "(?i)(yes/no|are you sure you want to continue connecting)" {
    send "yes\r"
    exp_continue
  }
  -re "(?i)password:" {
    send "$env(CROSSWALK_REMOTE_PASSWORD)\r"
    exp_continue
  }
  eof
}
catch wait result
set exit_status [lindex $result 3]
if {$exit_status eq ""} {
  set exit_status 0
}
exit $exit_status
`;
    if (!existsSync(expectScriptPath) || readFileSync(expectScriptPath, "utf8") !== script) {
      writeFileSync(expectScriptPath, script, { encoding: "utf8", mode: 0o700 });
    }
  }

  function loadServerOptions() {
    if (!existsSync(serverOptionsPath)) {
      writeJsonFile(serverOptionsPath, DEFAULT_REMOTE_SERVER_OPTIONS);
      return DEFAULT_REMOTE_SERVER_OPTIONS;
    }
    const storedOptions = readJsonFile<RemoteServerOption[]>(serverOptionsPath, DEFAULT_REMOTE_SERVER_OPTIONS);
    const mergedOptions = [
      ...DEFAULT_REMOTE_SERVER_OPTIONS.filter((defaultOption) => !storedOptions.some((storedOption) => storedOption.id === defaultOption.id)),
      ...storedOptions,
    ];
    if (mergedOptions.length !== storedOptions.length) {
      writeJsonFile(serverOptionsPath, mergedOptions);
    }
    return mergedOptions;
  }

  function applyServerOption(config: RemoteControllerConfig, serverOptions = loadServerOptions()): RemoteControllerConfig {
    const option = serverOptions.find((entry) => entry.id === config.server_id);
    if (!option) return config;
    const nextConfig: RemoteControllerConfig = {
      ...config,
      server_id: option.id,
      server_name: option.label,
      host: option.host,
      username: option.username,
      port: option.port,
      repo_path: option.repo_path,
      execution_mode: option.execution_mode === "direct" ? "direct" : "slurm",
      sbatch_script_path: option.sbatch_script_path ?? config.sbatch_script_path,
      direct_run_command: option.direct_run_command ?? config.direct_run_command,
      partition: option.partition ?? config.partition,
      time_limit: option.time_limit ?? config.time_limit,
    };
    return nextConfig;
  }

  function resolveSelectedServerId(config: RemoteControllerConfig, serverOptions = loadServerOptions()) {
    if (config.server_id && serverOptions.some((entry) => entry.id === config.server_id)) {
      return config.server_id;
    }
    const matchingOption = serverOptions.find(
      (entry) =>
        entry.host === config.host ||
        entry.hostname === config.host ||
        entry.label === config.server_name,
    );
    return matchingOption?.id ?? null;
  }

  function loadConfig(): RemoteControllerConfig {
    const storedConfig = readJsonFile<Partial<RemoteControllerConfig>>(configPath, {});
    const mergedConfig: RemoteControllerConfig = {
      ...DEFAULT_REMOTE_CONFIG,
      ...storedConfig,
      execution_mode: storedConfig.execution_mode === "direct" ? "direct" : "slurm",
    };
    const selectedServerId = resolveSelectedServerId(mergedConfig);
    return applyServerOption(
      {
        ...mergedConfig,
        server_id: selectedServerId ?? mergedConfig.server_id,
      },
      loadServerOptions(),
    );
  }

  function saveConfig(config: RemoteControllerConfig) {
    mkdirSync(root, { recursive: true });
    writeJsonFile(configPath, applyServerOption(config, loadServerOptions()));
    return getSnapshot();
  }

  function loadState() {
    return readJsonFile<{ remote_home: string | null; remote_hostname: string | null; last_error: string | null }>(statePath, {
      remote_home: null,
      remote_hostname: null,
      last_error: null,
    });
  }

  function saveState(next: { remote_home: string | null; remote_hostname: string | null; last_error: string | null }) {
    writeJsonFile(statePath, next);
  }

  function hasLocalTool(name: string) {
    try {
      runCommand("which", [name]);
      return true;
    } catch {
      return false;
    }
  }

  function hasPasswordConfigured() {
    return Boolean(process.env.CROSSWALK_REMOTE_PASSWORD?.length);
  }

  function hasSshpass() {
    return hasLocalTool("sshpass");
  }

  function hasExpect() {
    return hasLocalTool("expect");
  }

  function hasPasswordTransport() {
    return hasSshpass() || hasExpect();
  }

  function buildSshArgs(config: RemoteControllerConfig) {
    const args = [
      "-p",
      String(config.port),
      "-o",
      "StrictHostKeyChecking=accept-new",
      "-o",
      "ServerAliveInterval=15",
      "-o",
      "ServerAliveCountMax=3",
    ];
    if (!hasPasswordConfigured()) {
      args.push("-o", "BatchMode=yes");
    }
    return args;
  }

  function buildScpArgs(config: RemoteControllerConfig) {
    const args = [
      "-P",
      String(config.port),
      "-o",
      "StrictHostKeyChecking=accept-new",
      "-o",
      "ServerAliveInterval=15",
      "-o",
      "ServerAliveCountMax=3",
    ];
    if (!hasPasswordConfigured()) {
      args.push("-o", "BatchMode=yes");
    }
    return args;
  }

  function buildRemoteLogin(config: RemoteControllerConfig) {
    return `${config.username}@${config.host}`;
  }

  function buildExpectCommand(command: string, args: string[]): [string, string[]] {
    ensureExpectScript();
    return ["expect", [expectScriptPath, command, ...args]];
  }

  function buildRemoteCommand(config: RemoteControllerConfig, remoteCommand: string): [string, string[]] {
    if (useRemoteTools()) {
      return [remoteTool("remote-ssh"), [selectedRemoteToolHost(config), remoteCommand]];
    }
    const args = buildSshArgs(config);
    if (hasPasswordConfigured() && hasSshpass()) {
      return ["sshpass", ["-e", "ssh", ...args, buildRemoteLogin(config), remoteCommand]];
    }
    if (hasPasswordConfigured() && hasExpect()) {
      return buildExpectCommand("ssh", [...args, buildRemoteLogin(config), remoteCommand]);
    }
    return ["ssh", [...args, buildRemoteLogin(config), remoteCommand]];
  }

  function getSnapshot(): RemoteControllerSnapshot {
    const config = loadConfig();
    const state = loadState();
    const fake = isFakeConfig(config);
    const passwordConfigured = fake || passwordConfiguredOrRemoteTools();
    const serverOptions = loadServerOptions();
    const selectedServerId = resolveSelectedServerId(config, serverOptions);
    const lastError = passwordConfigured ? state.last_error : missingRemotePasswordMessage();
    const connected = fake ? !lastError : Boolean(state.remote_home && state.remote_hostname && !lastError);
    return {
      config,
      server_options: serverOptions,
      selected_server_id: selectedServerId,
      connected,
      password_configured: passwordConfigured,
      sshpass_available: fake || useRemoteTools() || hasSshpass(),
      expect_available: hasExpect(),
      password_transport_available: fake || useRemoteTools() || hasPasswordTransport(),
      tmux_available: fake || hasLocalTool("tmux"),
      remote_home: fake ? join(root, "fake-home") : state.remote_home,
      remote_hostname: fake ? "fake-gpu-simulator" : state.remote_hostname,
      last_error: lastError,
    };
  }

  function connect() {
    const config = loadConfig();
    if (isFakeConfig(config)) {
      saveState({
        remote_home: join(root, "fake-home"),
        remote_hostname: "fake-gpu-simulator",
        last_error: null,
      });
      return getSnapshot();
    }
    if (!hasRemoteTools() && process.env.CROSSWALK_USE_REMOTE_TOOLS === "1") {
      const message = missingRemoteToolsMessage(remoteToolsDir());
      saveState({
        remote_home: null,
        remote_hostname: null,
        last_error: message,
      });
      throw new Error(message);
    }
    if (!passwordConfiguredOrRemoteTools()) {
      const message = missingRemotePasswordMessage();
      console.warn(`[remote-controller] ${message}`);
      saveState({
        remote_home: null,
        remote_hostname: null,
        last_error: message,
      });
      throw new Error(message);
    }
    if (!config.host || !config.username || !config.repo_path) {
      throw new Error("Remote config is incomplete. Fill host, username, and repo path first.");
    }
    requireVpn();
    const [command, args] = buildRemoteCommand(
      config,
      "printf 'HOME=%s\\nHOST=%s\\n' \"$HOME\" \"$(hostname)\"",
    );
    try {
      const output = runCommand(command, args);
      const remoteHome = output.match(/^HOME=(.*)$/m)?.[1] ?? null;
      const remoteHostname = output.match(/^HOST=(.*)$/m)?.[1] ?? null;
      saveState({
        remote_home: remoteHome,
        remote_hostname: remoteHostname,
        last_error: null,
      });
      return getSnapshot();
    } catch (error) {
      const formattedError = formatRemoteError(error, config.host);
      saveState({
        remote_home: null,
        remote_hostname: null,
        last_error: formattedError,
      });
      throw new Error(formattedError);
    }
  }

  function buildMetadata(job: ScanBatchJob, state: ReturnType<typeof loadState>): RemoteJobMetadata {
    const id = `${job.scene.scene_id}-${Date.now()}-${randomUUID().slice(0, 8)}`;
    const jobDir = join(jobsRoot, id);
    mkdirSync(jobDir, { recursive: true });
    const remoteRepo = configRemoteRepo(loadConfig(), state.remote_home);
    const remoteJobPath = `${remoteRepo}/jobs/${id}.json`;
    const remoteResultPath = `${remoteRepo}/results/${id}.result.json`;
    return {
      id,
      scene_id: job.scene.scene_id,
      scene_label: `${job.scene.city} · ${job.scene.split}`,
      tile_count: job.tiles.length,
      created_at: nowIso(),
      updated_at: nowIso(),
      execution_mode: loadConfig().execution_mode,
      tmux_session: `crosswalk-remote-${id.slice(0, 24)}`,
      local_job_path: join(jobDir, "job.json"),
      local_result_path: join(jobDir, "result.json"),
      local_log_path: join(jobDir, "run.log"),
      local_script_path: join(jobDir, "run.zsh"),
      local_status_path: join(jobDir, "status.txt"),
      local_remote_state_path: join(jobDir, "remote-state.txt"),
      local_slurm_job_id_path: join(jobDir, "slurm-job-id.txt"),
      local_error_path: join(jobDir, "error.txt"),
      remote_job_path: remoteJobPath,
      remote_result_path: remoteResultPath,
    };
  }

  function configRemoteRepo(config: RemoteControllerConfig, remoteHome: string | null) {
    if (isFakeConfig(config)) {
      return join(root, "fake-home", "repo");
    }
    if (config.repo_path.startsWith("~/")) {
      if (!remoteHome) {
        throw new Error("Remote repo path starts with ~/ but remote home is unknown. Connect first.");
      }
      return join(remoteHome, config.repo_path.slice(2));
    }
    return config.repo_path;
  }

  function writeMetadata(path: string, metadata: RemoteJobMetadata) {
    writeJsonFile(path, metadata);
  }

  function readMetadata(path: string) {
    return readJsonFile<RemoteJobMetadata | null>(path, null);
  }

  function metadataPath(jobId: string) {
    return join(jobsRoot, jobId, "metadata.json");
  }

  function writeRunScript(config: RemoteControllerConfig, state: ReturnType<typeof loadState>, metadata: RemoteJobMetadata) {
    const usePassword = hasPasswordConfigured();
    const script = buildRemoteRunScript({
      config,
      metadata,
      jobsRoot,
      repoPath: configRemoteRepo(config, state.remote_home),
      sshArgs: buildSshArgs(config),
      scpArgs: buildScpArgs(config),
      useRemoteTools: useRemoteTools(),
      remoteToolHost: selectedRemoteToolHost(config),
      remoteSshTool: remoteTool("remote-ssh"),
      remoteScpTool: remoteTool("remote-scp"),
      usePassword,
      useSshPass: usePassword && hasSshpass(),
      useExpect: usePassword && !hasSshpass() && hasExpect(),
      login: buildRemoteLogin(config),
      hfTokenOpRef: process.env.CROSSWALK_HF_TOKEN_OP_REF ?? "op://dev/Hugging Face Read Token/credential",
    });
    writeFileSync(metadata.local_script_path, script, { encoding: "utf8", mode: 0o755 });
  }
  function tmuxSessionExists(session: string) {
    try {
      execFileSync("tmux", ["has-session", "-t", session], { stdio: "ignore" });
      return true;
    } catch {
      return false;
    }
  }

  function startJob(job: ScanBatchJob) {
    if (isFakeConfig()) {
      connect();
      const state = loadState();
      const metadata = buildMetadata(job, state);
      writeJsonFile(metadata.local_job_path, job);
      writeMetadata(metadataPath(metadata.id), metadata);
      writeFileSync(metadata.local_status_path, "bootstrapping", "utf8");
      void runFakeRemoteJob(metadata, job);
      return getJob(metadata.id);
    }
    if (!passwordConfiguredOrRemoteTools()) {
      const message = missingRemotePasswordMessage();
      console.warn(`[remote-controller] ${message}`);
      throw new Error(message);
    }
    const snapshot = connect();
    const config = snapshot.config;
    const state = loadState();
    const metadata = buildMetadata(job, state);
    writeJsonFile(metadata.local_job_path, job);
    writeMetadata(metadataPath(metadata.id), metadata);
    writeFileSync(metadata.local_status_path, "bootstrapping", "utf8");
    writeRunScript(config, state, metadata);
    const runnerCommand = hasPasswordConfigured()
      ? `env CROSSWALK_REMOTE_PASSWORD=${shellQuote(process.env.CROSSWALK_REMOTE_PASSWORD ?? "")} zsh ${shellQuote(metadata.local_script_path)}`
      : `zsh ${shellQuote(metadata.local_script_path)}`;
    execFileSync("tmux", ["new-session", "-d", "-s", metadata.tmux_session, runnerCommand], {
      stdio: "ignore",
      env: process.env,
    });
    return getJob(metadata.id);
  }

  function resolveStatus(metadata: RemoteJobMetadata): RemoteJobStatus {
    if (existsSync(metadata.local_result_path)) return "completed";
    const rawStatus = existsSync(metadata.local_status_path) ? readFileSync(metadata.local_status_path, "utf8").trim() : "";
    if (rawStatus === "cancelled") return "cancelled";
    if (rawStatus === "failed") return "failed";
    if (rawStatus === "completed") return "completed";
    if (rawStatus === "running") return "running";
    if (rawStatus === "queued") return "queued";
    if (rawStatus === "submitting") return "submitting";
    if (rawStatus === "syncing") return "syncing";
    if (rawStatus === "bootstrapping") return "bootstrapping";
    if (tmuxSessionExists(metadata.tmux_session)) return "bootstrapping";
    return "idle";
  }

  function cancelJob(jobId: string) {
    const metadata = readMetadata(metadataPath(jobId));
    if (!metadata) {
      throw new Error(`Unknown remote job: ${jobId}`);
    }
    const config = loadConfig();
    const slurmJobId = existsSync(metadata.local_slurm_job_id_path)
      ? readFileSync(metadata.local_slurm_job_id_path, "utf8").trim()
      : "";
    if (slurmJobId) {
      const [command, args] = buildRemoteCommand(config, `scancel ${shellQuote(slurmJobId)} || true`);
      try {
        runCommand(command, args);
      } catch {
        // Best effort only.
      }
    }
    if (tmuxSessionExists(metadata.tmux_session)) {
      execFileSync("tmux", ["kill-session", "-t", metadata.tmux_session], { stdio: "ignore" });
    }
    writeFileSync(metadata.local_status_path, "cancelled", "utf8");
    writeFileSync(metadata.local_error_path, "Cancelled by user.", "utf8");
    return getJob(jobId);
  }

  function getJob(jobId: string): RemoteScanJobRecord {
    const metadata = readMetadata(metadataPath(jobId));
    if (!metadata) {
      throw new Error(`Unknown remote job: ${jobId}`);
    }
    const status = resolveStatus(metadata);
    const remoteState = existsSync(metadata.local_remote_state_path)
      ? readFileSync(metadata.local_remote_state_path, "utf8").trim() || null
      : null;
    const slurmJobId = existsSync(metadata.local_slurm_job_id_path)
      ? readFileSync(metadata.local_slurm_job_id_path, "utf8").trim() || null
      : null;
    const error = existsSync(metadata.local_error_path)
      ? readFileSync(metadata.local_error_path, "utf8").trim() || null
      : null;
    const logContents = existsSync(metadata.local_log_path) ? readFileSync(metadata.local_log_path, "utf8") : "";
    const logTail = logContents ? trimLines(logContents, 24) : [];
    const jobPayload = readJsonFile<ScanBatchJob | null>(metadata.local_job_path, null);
    const liveProgress = existsSync(metadata.local_result_path)
      ? progressFromScanBatchResult(readJsonFile<ScanBatchResult>(metadata.local_result_path, null as never))
      : parseRemoteJobProgressFromLog(logContents, jobPayload?.prompts_text ?? "remote-live");
    const summary = liveProgress.summary.total > 0 ? liveProgress.summary : null;
    return {
      id: metadata.id,
      scene_id: metadata.scene_id,
      scene_label: metadata.scene_label,
      tile_count: metadata.tile_count,
      created_at: metadata.created_at,
      updated_at: nowIso(),
      tmux_session: metadata.tmux_session,
      log_tmux_session: null,
      execution_mode: metadata.execution_mode,
      status,
      remote_state: remoteState,
      slurm_job_id: slurmJobId,
      error,
      result_available: existsSync(metadata.local_result_path),
      log_tail: logTail,
      summary,
      live_results: liveProgress.results,
      live_scanned_tile_ids: liveProgress.scannedTileIds,
    };
  }

  function listJobs() {
    if (!existsSync(jobsRoot)) return [];
    return readJsonFile<string[]>(
      join(root, "job-index.json"),
      [],
    );
  }

  function updateJobIndex(jobId: string) {
    const current = listJobs();
    if (!current.includes(jobId)) {
      writeJsonFile(join(root, "job-index.json"), [jobId, ...current]);
    }
  }

  function listJobRecords() {
    return listJobs()
      .map((jobId) => {
        try {
          return getJob(jobId);
        } catch {
          return null;
        }
      })
      .filter((job): job is RemoteScanJobRecord => Boolean(job));
  }

  function loadResult(jobId: string) {
    const metadata = readMetadata(metadataPath(jobId));
    if (!metadata || !existsSync(metadata.local_result_path)) {
      throw new Error(`No result available for ${jobId}`);
    }
    return readJsonFile<ScanBatchResult>(metadata.local_result_path, null as never);
  }

  function getTerminalSource(jobId: string) {
    const metadata = readMetadata(metadataPath(jobId));
    if (!metadata) {
      throw new Error(`Unknown remote job: ${jobId}`);
    }
    return {
      tmux_session: metadata.tmux_session,
      local_log_path: metadata.local_log_path,
      status: resolveStatus(metadata),
    };
  }

  return {
    getSnapshot,
    saveConfig,
    connect,
    startJob(job: ScanBatchJob) {
      const record = startJob(job);
      updateJobIndex(record.id);
      return record;
    },
    listJobs: listJobRecords,
    getJob,
    getTerminalSource,
    cancelJob,
    loadResult,
  };
}
