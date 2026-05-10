import { useEffect, useMemo, useState } from "react";
import { Alert, Button, Card, ListBox, ListBoxItem, Select, Spinner } from "@heroui/react";
import { CheckCircle2, CloudDownload, Play, PlugZap, Server, TerminalSquare } from "lucide-react";
import { summarizeErrorMessage } from "../error-summary";
import { connectRemoteController, getRemoteScanResult, listRemoteScanJobs, loadRemoteControllerSnapshot, saveRemoteControllerConfig, startRemoteScanJob } from "../remote-api";
import type { ScanBatchJob, ScanBatchResult } from "../scan-batch";
import type { RemoteControllerSnapshot, RemoteScanJobRecord } from "../types";

type RemoteScanPanelProps = {
  scanJob: ScanBatchJob | null;
  activeRemoteJobId?: string | null;
  onActiveRemoteJobChange: (jobId: string | null) => void;
  onActiveRemoteJobResolved?: (job: RemoteScanJobRecord | null) => void;
  onOpenTerminal?: () => void;
  onResultImported: (result: ScanBatchResult) => void;
  onError: (message: string | null) => void;
  compact?: boolean;
};

export function RemoteScanPanel({
  scanJob,
  activeRemoteJobId,
  onActiveRemoteJobChange,
  onActiveRemoteJobResolved,
  onOpenTerminal,
  onResultImported,
  onError,
  compact = false,
}: RemoteScanPanelProps) {
  const [snapshot, setSnapshot] = useState<RemoteControllerSnapshot | null>(null);
  const [jobs, setJobs] = useState<RemoteScanJobRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [importedJobIds, setImportedJobIds] = useState<Record<string, true>>({});
  const serverOptions = snapshot?.server_options ?? [];

  useEffect(() => {
    let cancelled = false;
    Promise.all([loadRemoteControllerSnapshot(), listRemoteScanJobs()])
      .then(([nextSnapshot, nextJobs]) => {
        if (cancelled) return;
        setSnapshot(nextSnapshot);
        setJobs(nextJobs);
      })
      .catch((reason) => {
        if (!cancelled) onError(String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [onError]);

  useEffect(() => {
    const hasActive = jobs.some((job) => ["bootstrapping", "syncing", "submitting", "queued", "running"].includes(job.status));
    if (!hasActive) return;
    const timer = window.setInterval(() => {
      void listRemoteScanJobs()
        .then(setJobs)
        .catch((reason) => onError(String(reason)));
    }, 200);
    return () => window.clearInterval(timer);
  }, [jobs, onError]);

  useEffect(() => {
    const activeJob = jobs.find((job) => job.id === activeRemoteJobId);
    if (!activeJob || activeJob.status !== "completed" || importedJobIds[activeJob.id]) {
      return;
    }
    void getRemoteScanResult(activeJob.id)
      .then((result) => {
        onResultImported(result);
        setImportedJobIds((current) => ({ ...current, [activeJob.id]: true }));
      })
      .catch((reason) => onError(String(reason)));
  }, [activeRemoteJobId, importedJobIds, jobs, onError, onResultImported]);

  const activeJob = useMemo(() => jobs.find((job) => job.id === activeRemoteJobId) ?? jobs[0] ?? null, [activeRemoteJobId, jobs]);
  const selectedServer = useMemo(
    () => serverOptions.find((entry) => entry.id === snapshot?.selected_server_id) ?? null,
    [serverOptions, snapshot?.selected_server_id],
  );
  const connectionSummary = useMemo(
    () =>
      summarizeErrorMessage(
        snapshot?.last_error,
        "The local controller is running, but the remote server is not connected yet.",
      ),
    [snapshot?.last_error],
  );
  const canConnect = Boolean(snapshot?.password_configured) && !busy;
  const canStart = Boolean(snapshot?.password_configured) && Boolean(scanJob) && !busy;
  const activeJobLiveCount = activeJob?.live_scanned_tile_ids.length ?? 0;
  const activeJobProgressLabel = activeJob ? `${activeJobLiveCount}/${activeJob.tile_count}` : "0/0";
  const activeJobLastTile = activeJob?.live_scanned_tile_ids.at(-1);

  useEffect(() => {
    onActiveRemoteJobResolved?.(activeJob);
  }, [activeJob, onActiveRemoteJobResolved]);

  async function handleSelectServer(serverId: string) {
    if (!snapshot) return;
    const option = serverOptions.find((entry) => entry.id === serverId);
    if (!option) return;
    setBusy(true);
    onError(null);
    try {
      const nextSnapshot = await saveRemoteControllerConfig({
        ...snapshot.config,
        server_id: option.id,
        server_name: option.label,
        host: option.host,
        username: option.username,
        port: option.port,
        repo_path: option.repo_path,
        partition: option.partition ?? snapshot.config.partition,
        time_limit: option.time_limit ?? snapshot.config.time_limit,
      });
      setSnapshot(nextSnapshot);
    } catch (reason) {
      onError(String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function handleConnect() {
    setBusy(true);
    onError(null);
    try {
      const nextSnapshot = await connectRemoteController();
      setSnapshot(nextSnapshot);
    } catch (reason) {
      onError(String(reason));
      void loadRemoteControllerSnapshot().then(setSnapshot).catch(() => undefined);
    } finally {
      setBusy(false);
    }
  }

  async function handleStart() {
    if (!scanJob) return;
    setBusy(true);
    onError(null);
    try {
      const job = await startRemoteScanJob(scanJob);
      onActiveRemoteJobChange(job.id);
      setJobs((current) => [job, ...current.filter((entry) => entry.id !== job.id)]);
    } catch (reason) {
      onError(String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function handleImportLatest() {
    if (!activeJob?.result_available) return;
    setBusy(true);
    onError(null);
    try {
      const result = await getRemoteScanResult(activeJob.id);
      onActiveRemoteJobChange(activeJob.id);
      onResultImported(result);
      setImportedJobIds((current) => ({ ...current, [activeJob.id]: true }));
    } catch (reason) {
      onError(String(reason));
    } finally {
      setBusy(false);
    }
  }

  if (compact) {
    const connectionLabel = snapshot?.connected ? "Connected" : snapshot?.last_error ? "Error" : "Offline";
    const activeBusy = activeJob ? ["bootstrapping", "syncing", "submitting", "queued", "running"].includes(activeJob.status) : false;

    return (
      <div className="flex w-full min-w-0 flex-col gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <Button
            aria-label={snapshot?.connected ? "Server connected" : "Connect server"}
            className={`h-10 shrink-0 rounded-[16px] px-3 [corner-shape:squircle] ${
              snapshot?.connected
                ? "bg-success text-success-foreground"
                : snapshot?.last_error
                  ? "bg-danger text-danger-foreground"
                  : "bg-warning text-warning-foreground"
            }`}
            isDisabled={!canConnect}
            onPress={handleConnect}
            size="sm"
            variant="primary"
          >
            {busy && !activeJob ? <Spinner size="sm" /> : snapshot?.connected ? <CheckCircle2 className="size-4" /> : <PlugZap className="size-4" />}
            {connectionLabel}
          </Button>
          <div className="min-w-0 flex-1 rounded-[16px] bg-white/10 px-3 py-2 text-xs font-semibold [corner-shape:squircle]">
            <div className="truncate">{activeJob?.status ?? selectedServer?.label ?? "Server"}</div>
            <div className="truncate text-white/60">{activeJobLastTile ?? `${activeJobProgressLabel} tiles`}</div>
          </div>
        </div>

        <div className="grid w-full grid-cols-[1fr,auto,auto] gap-2">
          <Button
            aria-label="Start server scan"
            className="h-10 rounded-[16px] [corner-shape:squircle]"
            isDisabled={!canStart}
            onPress={handleStart}
            size="sm"
            variant={activeBusy ? "primary" : "secondary"}
          >
            {busy && !!scanJob ? <Spinner size="sm" /> : activeBusy ? <Spinner size="sm" /> : <Play className="size-4" />}
            Start scan
          </Button>
          <Button
            aria-label="Import latest server result"
            className="size-10 shrink-0 rounded-[16px] [corner-shape:squircle]"
            isDisabled={busy || !activeJob?.result_available}
            isIconOnly
            onPress={handleImportLatest}
            size="sm"
            variant="secondary"
          >
            <CloudDownload className="size-4" />
          </Button>
          <Button
            aria-label="Open terminal"
            className="size-10 shrink-0 rounded-[16px] [corner-shape:squircle]"
            isDisabled={busy || !activeJob}
            isIconOnly
            onPress={onOpenTerminal}
            size="sm"
            variant="secondary"
          >
            <TerminalSquare className="size-4" />
          </Button>
        </div>

        {activeJob?.log_tail.length ? (
          <pre className="max-h-28 overflow-y-auto whitespace-pre-wrap rounded-[16px] bg-black/35 p-3 text-[11px] leading-relaxed text-white/75 [corner-shape:squircle]">
            {activeJob.log_tail.slice(-6).join("\n")}
          </pre>
        ) : (
          <div className="rounded-[16px] bg-white/8 p-3 text-xs text-white/65 [corner-shape:squircle]">{connectionSummary}</div>
        )}
      </div>
    );
  }

  return (
    <Card className="pointer-events-auto max-h-[calc(100dvh-2rem)] overflow-hidden shadow-xl" variant="secondary">
      <Card.Header>
        <div className="flex min-w-0 flex-col gap-1">
          <Card.Title>Server scan</Card.Title>
          <Card.Description>
            {selectedServer
              ? snapshot?.connected
                ? `${selectedServer.label} · ${selectedServer.host} · connected`
                : `${selectedServer.label} · ${selectedServer.host}`
              : "Choose a server, then connect."}
          </Card.Description>
        </div>
      </Card.Header>
      <Card.Content className="flex max-h-full min-h-0 flex-col gap-4 overflow-y-auto" data-scroll-guard="remote-scan-panel">
        {!snapshot?.connected ? (
          <Alert status="warning">
            <Alert.Content>
              <Alert.Title>Server not connected</Alert.Title>
              <Alert.Description>{connectionSummary}</Alert.Description>
            </Alert.Content>
          </Alert>
        ) : null}

        <div className="flex flex-wrap gap-2">
          {serverOptions.length ? (
            <Select
              aria-label="Server"
              className="min-w-56"
              isDisabled={busy}
              selectedKey={selectedServer?.id ?? null}
              variant="secondary"
              onSelectionChange={(key) => {
                if (typeof key === "string") {
                  void handleSelectServer(key);
                }
              }}
            >
              <Select.Trigger>
                <Select.Value>{selectedServer ? `${selectedServer.label} · ${selectedServer.host}` : "Choose server"}</Select.Value>
                <Select.Indicator />
              </Select.Trigger>
              <Select.Popover>
                <ListBox aria-label="Server selection">
                  {serverOptions.map((option) => (
                    <ListBoxItem id={option.id} key={option.id} textValue={`${option.label} ${option.host}`}>
                      {option.label} · {option.host}
                    </ListBoxItem>
                  ))}
                </ListBox>
              </Select.Popover>
            </Select>
          ) : null}
          <Button
            className={snapshot?.connected ? "bg-success text-success-foreground" : undefined}
            isDisabled={!canConnect}
            onPress={handleConnect}
            variant={snapshot?.connected ? "primary" : "secondary"}
          >
            {busy && !activeJob ? <Spinner size="sm" /> : null}
            {!busy ? snapshot?.connected ? <CheckCircle2 className="size-4" /> : <PlugZap className="size-4" /> : null}
            {snapshot?.connected ? "Connected" : "Connect"}
          </Button>
          <Button isDisabled={!canStart} onPress={handleStart} variant="primary">
            {busy && !!scanJob ? <Spinner size="sm" /> : null}
            {!busy ? <Play className="size-4" /> : null}
            Start scan
          </Button>
          {activeJob?.result_available ? (
            <Button isDisabled={busy} onPress={handleImportLatest} variant="outline">
              <CloudDownload className="size-4" />
              Import latest
            </Button>
          ) : null}
          {activeJob ? (
            <Button isDisabled={busy} onPress={onOpenTerminal} variant="ghost">
              <TerminalSquare className="size-4" />
              Terminal
            </Button>
          ) : null}
        </div>

        {activeJob ? (
          <Alert status={activeJob.status === "completed" ? "success" : activeJob.status === "failed" ? "danger" : "default"}>
            <Alert.Content>
              <Alert.Title className="flex items-center gap-2">
                {activeJob.status === "completed" ? <CheckCircle2 className="size-4" /> : <Server className="size-4" />}
                <span>{activeJob.scene_label}</span>
              </Alert.Title>
              <Alert.Description>
                {activeJob.status} · {activeJobProgressLabel} tiles
                {activeJob.summary ? ` · ${activeJob.summary.crosswalk} positive · ${activeJob.summary.no_crosswalk} negative` : ""}
              </Alert.Description>
            </Alert.Content>
          </Alert>
        ) : null}

        {activeJob ? (
          <Card className="bg-content1/70" variant="secondary">
            <Card.Content className="flex flex-col gap-2 text-sm">
              <div className="grid grid-cols-[6rem,1fr] gap-x-3 gap-y-1">
                <span className="text-foreground/60">State</span>
                <span>{activeJob.remote_state ?? activeJob.status}</span>
                <span className="text-foreground/60">Progress</span>
                <span>{activeJobProgressLabel} tiles</span>
                <span className="text-foreground/60">Current</span>
                <span className="break-all">{activeJobLastTile ?? "Waiting for first tile"}</span>
              </div>
              {activeJob.log_tail.length ? (
                <pre className="max-h-28 overflow-y-auto whitespace-pre-wrap rounded-lg bg-black/30 p-2 text-xs text-foreground/80">
                  {activeJob.log_tail.slice(-5).join("\n")}
                </pre>
              ) : null}
            </Card.Content>
          </Card>
        ) : null}

        {!snapshot?.password_configured ? (
          <Alert status="warning">
            <Alert.Content>
              <Alert.Title>Password missing</Alert.Title>
              <Alert.Description>
                Add <code>CROSSWALK_REMOTE_PASSWORD</code> to your local <code>.env</code> file and restart <code>bun run dev</code>.
              </Alert.Description>
            </Alert.Content>
          </Alert>
        ) : null}
      </Card.Content>
    </Card>
  );
}
