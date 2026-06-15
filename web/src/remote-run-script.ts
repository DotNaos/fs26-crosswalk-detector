import { join } from "node:path";
import type { RemoteControllerConfig } from "./types";

type RemoteRunMetadata = {
  id: string;
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

type RemoteRunScriptInput = {
  config: RemoteControllerConfig;
  metadata: RemoteRunMetadata;
  jobsRoot: string;
  repoPath: string;
  sshArgs: string[];
  scpArgs: string[];
  useRemoteTools: boolean;
  remoteToolHost: string;
  remoteSshTool: string;
  remoteScpTool: string;
  usePassword: boolean;
  useSshPass: boolean;
  useExpect: boolean;
  login: string;
  hfTokenOpRef: string;
};

export function shellQuote(value: string) {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}

export function buildRemoteRunScript(input: RemoteRunScriptInput) {
  const { config, metadata, jobsRoot, repoPath, sshArgs, scpArgs, useRemoteTools, remoteToolHost, remoteSshTool, remoteScpTool, usePassword, useSshPass, useExpect, login, hfTokenOpRef } = input;
  return `#!/bin/zsh
set -euo pipefail

JOB_FILE=${shellQuote(metadata.local_job_path)}
RESULT_FILE=${shellQuote(metadata.local_result_path)}
LOG_FILE=${shellQuote(metadata.local_log_path)}
STATUS_FILE=${shellQuote(metadata.local_status_path)}
REMOTE_STATE_FILE=${shellQuote(metadata.local_remote_state_path)}
SLURM_FILE=${shellQuote(metadata.local_slurm_job_id_path)}
ERROR_FILE=${shellQuote(metadata.local_error_path)}
EXPECT_HELPER=${shellQuote(join(jobsRoot, metadata.id, "expect-helper.tcl"))}
REMOTE_LOGIN=${shellQuote(login)}
REMOTE_TOOL_HOST=${shellQuote(remoteToolHost)}
REMOTE_SSH_TOOL=${shellQuote(remoteSshTool)}
REMOTE_SCP_TOOL=${shellQuote(remoteScpTool)}
REMOTE_REPO=${shellQuote(repoPath)}
REMOTE_JOB=${shellQuote(metadata.remote_job_path)}
REMOTE_RESULT=${shellQuote(metadata.remote_result_path)}
POLL_SECONDS=${shellQuote(String(Math.max(1, config.poll_interval_seconds)))}
PARTITION=${shellQuote(config.partition)}
TIME_LIMIT=${shellQuote(config.time_limit)}
HF_TOKEN_OP_REF=${shellQuote(hfTokenOpRef)}

exec > >(tee -a "$LOG_FILE") 2>&1

write_status() {
  printf '%s' "$1" > "$STATUS_FILE"
}

write_status bootstrapping
echo "Preparing remote scan job"

${usePassword ? 'export CROSSWALK_REMOTE_PASSWORD="${CROSSWALK_REMOTE_PASSWORD:?CROSSWALK_REMOTE_PASSWORD is required}"' : ""}
${useSshPass ? 'export SSHPASS="$CROSSWALK_REMOTE_PASSWORD"' : ""}
${useExpect ? `cat > "$EXPECT_HELPER" <<'EOF'
set timeout -1
if {![info exists env(CROSSWALK_REMOTE_PASSWORD)] || $env(CROSSWALK_REMOTE_PASSWORD) eq ""} {
  puts stderr "CROSSWALK_REMOTE_PASSWORD is required"
  exit 2
}
set cmd [lindex $argv 0]
set argv [lrange $argv 1 end]
spawn -noecho $cmd {*}$argv
expect {
  -re "(?i)(yes/no|are you sure you want to continue connecting)" {
    send "yes\\r"
    exp_continue
  }
  -re "(?i)password:" {
    send "$env(CROSSWALK_REMOTE_PASSWORD)\\r"
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
EOF
chmod 700 "$EXPECT_HELPER"

run_with_password() {
  expect "$EXPECT_HELPER" "$@"
}` : ""}

${useRemoteTools ? `run_ssh() {
  "$REMOTE_SSH_TOOL" "$REMOTE_TOOL_HOST" "$@"
}

run_scp() {
  local source_path="$1"
  local dest_path="$2"
  case "$source_path" in
    "$REMOTE_LOGIN":*) source_path=":\${source_path#"$REMOTE_LOGIN":"}" ;;
  esac
  case "$dest_path" in
    "$REMOTE_LOGIN":*) dest_path=":\${dest_path#"$REMOTE_LOGIN":"}" ;;
  esac
  "$REMOTE_SCP_TOOL" "$REMOTE_TOOL_HOST" "$source_path" "$dest_path"
}` : `run_ssh() {
${useSshPass ? "  sshpass -e ssh " : useExpect ? "  run_with_password ssh " : "  ssh "}${sshArgs.map(shellQuote).join(" ")} "$REMOTE_LOGIN" "$@"
}

run_scp() {
${useSshPass ? "  sshpass -e scp " : useExpect ? "  run_with_password scp " : "  scp "}${scpArgs.map(shellQuote).join(" ")} "$@"
}`}

write_status syncing
run_ssh "mkdir -p '$REMOTE_REPO/jobs' '$REMOTE_REPO/results' '$REMOTE_REPO/logs' && cd '$REMOTE_REPO' && git pull --ff-only origin main && UV_BIN=\\\"\\$(command -v uv || true)\\\" && if [[ -z \\\"\\$UV_BIN\\\" && -x \\\"\\$HOME/.local/bin/uv\\\" ]]; then UV_BIN=\\\"\\$HOME/.local/bin/uv\\\"; fi && if [[ -n \\\"\\$UV_BIN\\\" ]]; then \\\"\\$UV_BIN\\\" sync; elif [[ -x .venv/bin/python ]]; then echo 'uv missing on PATH, reusing existing virtualenv'; else echo 'uv missing and .venv not found' >&2; exit 1; fi"

write_status submitting
run_scp "$JOB_FILE" "$REMOTE_LOGIN:$REMOTE_JOB"

if [[ ${shellQuote(config.execution_mode)} == 'direct' ]]; then
  write_status running
  printf '%s' 'RUNNING' > "$REMOTE_STATE_FILE"
  echo "Running direct scan on remote host"
  HF_TOKEN_VALUE=""
  if command -v op >/dev/null 2>&1; then
    HF_TOKEN_VALUE="$(op read "$HF_TOKEN_OP_REF" 2>/dev/null || true)"
  fi
  REMOTE_SCAN_COMMAND="read -r HF_TOKEN; export HF_TOKEN CROSSWALK_SCAN_BACKEND=sam31 PYTHONUNBUFFERED=1; cd '$REMOTE_REPO' && ${config.direct_run_command} --job-file '$REMOTE_JOB' --output '$REMOTE_RESULT' --progress"
  if [[ -n "$HF_TOKEN_VALUE" ]]; then
    printf '%s\n' "$HF_TOKEN_VALUE" | run_ssh "$REMOTE_SCAN_COMMAND"
  else
    printf '\n' | run_ssh "$REMOTE_SCAN_COMMAND"
  fi
  run_scp "$REMOTE_LOGIN:$REMOTE_RESULT" "$RESULT_FILE"
  printf '%s' 'COMPLETED' > "$REMOTE_STATE_FILE"
  write_status completed
  echo "Downloaded result to $RESULT_FILE"
  exit 0
fi

SLURM_JOB_ID="$(run_ssh "cd '$REMOTE_REPO' && sbatch --parsable -p '$PARTITION' --time '$TIME_LIMIT' ${config.sbatch_script_path} '$REMOTE_JOB' '$REMOTE_RESULT'" | tail -n 1 | tr -d '\\r')"
printf '%s' "$SLURM_JOB_ID" > "$SLURM_FILE"
echo "Submitted Slurm job $SLURM_JOB_ID"

while true; do
  REMOTE_STATE="$(run_ssh "squeue -h -j '$SLURM_JOB_ID' -o '%T' || true" | tr -d '\\r')"
  if [[ -z "$REMOTE_STATE" ]]; then
    REMOTE_STATE="$(run_ssh "sacct -n -j '$SLURM_JOB_ID' --format=State | awk 'NF { print; exit }' || true" | tr -d '\\r')"
  fi
  printf '%s' "$REMOTE_STATE" > "$REMOTE_STATE_FILE"
  case "$REMOTE_STATE" in
    PENDING*)
      write_status queued
      ;;
    RUNNING*|COMPLETING*|CONFIGURING*)
      write_status running
      ;;
  esac

  if run_ssh "test -f '$REMOTE_RESULT'"; then
    write_status running
    run_scp "$REMOTE_LOGIN:$REMOTE_RESULT" "$RESULT_FILE"
    write_status completed
    echo "Downloaded result to $RESULT_FILE"
    exit 0
  fi

  case "$REMOTE_STATE" in
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*)
      write_status failed
      printf '%s' "$REMOTE_STATE" > "$ERROR_FILE"
      echo "Remote job failed with state $REMOTE_STATE"
      exit 1
      ;;
  esac

  sleep "$POLL_SECONDS"
done
`;
}
