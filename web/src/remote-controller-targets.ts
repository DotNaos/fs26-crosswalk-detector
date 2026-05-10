import { join } from "node:path";
import type { RemoteControllerConfig, RemoteExecutionMode, RemoteServerOption } from "./types";

export const DEFAULT_REMOTE_CONFIG: RemoteControllerConfig = {
  server_id: "",
  server_name: "",
  host: "",
  username: "",
  port: 22,
  repo_path: "",
  execution_mode: "slurm",
  sbatch_script_path: "cluster/scan-job.slurm",
  direct_run_command: "python -m crosswalk_detector.cli run-scan-job",
  partition: "debug",
  time_limit: "00:05:00",
  poll_interval_seconds: 2,
};

export const DEFAULT_REMOTE_SERVER_OPTIONS: RemoteServerOption[] = [
  {
    id: "fake-gpu",
    label: "Fake GPU",
    kind: "fake",
    hostname: "fake-gpu.local",
    host: "fake-gpu.local",
    username: "local",
    port: 22,
    repo_path: ".local/fake-gpu",
    execution_mode: "direct",
    direct_run_command: "fake-sam31-scan",
  },
  {
    id: "mercury",
    label: "Mercury",
    hostname: "mercury.fhgr.ch",
    host: "10.0.140.30",
    username: "schuetoliver",
    port: 22,
    repo_path: "~/projects/fs26-crosswalk-detector",
    partition: "debug",
    time_limit: "00:05:00",
  },
  {
    id: "iridium",
    label: "Iridium",
    hostname: "iridium.fhgr.ch",
    host: "10.0.106.140",
    username: "schuetoliver",
    port: 22,
    repo_path: "~/projects/fs26-crosswalk-detector",
    execution_mode: "direct",
    direct_run_command: ".venv/bin/python -m crosswalk_detector.cli run-scan-job",
  },
];

function executionModeFor(optionOrConfig: Pick<RemoteServerOption, "execution_mode"> | Pick<RemoteControllerConfig, "execution_mode">): RemoteExecutionMode {
  return optionOrConfig.execution_mode === "direct" ? "direct" : "slurm";
}

export function configRemoteRepo(config: RemoteControllerConfig, remoteHome: string | null) {
  if (config.server_id === "fake-gpu") {
    return config.repo_path;
  }
  if (config.repo_path.startsWith("~/")) {
    if (!remoteHome) {
      throw new Error("Remote repo path starts with ~/ but remote home is unknown. Connect first.");
    }
    return join(remoteHome, config.repo_path.slice(2));
  }
  return config.repo_path;
}

export function mergeRemoteConfigFromOption(config: RemoteControllerConfig, option: RemoteServerOption): RemoteControllerConfig {
  return {
    ...config,
    server_id: option.id,
    server_name: option.label,
    host: option.host,
    username: option.username,
    port: option.port,
    repo_path: option.repo_path,
    execution_mode: executionModeFor(option),
    sbatch_script_path: option.sbatch_script_path ?? config.sbatch_script_path,
    direct_run_command: option.direct_run_command ?? config.direct_run_command,
    partition: option.partition ?? config.partition,
    time_limit: option.time_limit ?? config.time_limit,
  };
}

export function applyServerOption(config: RemoteControllerConfig, serverOptions: RemoteServerOption[]) {
  const option = serverOptions.find((entry) => entry.id === config.server_id);
  if (!option) {
    return {
      ...config,
      execution_mode: executionModeFor(config),
    };
  }
  return mergeRemoteConfigFromOption(config, option);
}

export function resolveSelectedServerId(config: RemoteControllerConfig, serverOptions: RemoteServerOption[]) {
  if (config.server_id && serverOptions.some((entry) => entry.id === config.server_id)) {
    return config.server_id;
  }
  const matchingOption = serverOptions.find(
    (entry) => entry.host === config.host || entry.hostname === config.host || entry.label === config.server_name,
  );
  return matchingOption?.id ?? null;
}
