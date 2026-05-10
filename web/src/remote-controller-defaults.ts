import type { RemoteControllerConfig, RemoteServerOption } from "./types";

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
