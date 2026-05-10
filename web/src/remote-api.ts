import type { ScanBatchJob, ScanBatchResult } from "./scan-batch";
import type { RemoteControllerConfig, RemoteControllerSnapshot, RemoteScanJobRecord } from "./types";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    try {
      const parsed = JSON.parse(text) as { error?: string };
      throw new Error(parsed.error || text || `Request failed with status ${response.status}`);
    } catch {
      throw new Error(text || `Request failed with status ${response.status}`);
    }
  }
  return (await response.json()) as T;
}

export async function loadRemoteControllerSnapshot(): Promise<RemoteControllerSnapshot> {
  return parseJson<RemoteControllerSnapshot>(await fetch("/api/remote/config"));
}

export async function saveRemoteControllerConfig(config: RemoteControllerConfig): Promise<RemoteControllerSnapshot> {
  return parseJson<RemoteControllerSnapshot>(
    await fetch("/api/remote/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  );
}

export async function connectRemoteController(): Promise<RemoteControllerSnapshot> {
  return parseJson<RemoteControllerSnapshot>(
    await fetch("/api/remote/connect", {
      method: "POST",
    }),
  );
}

export async function startRemoteScanJob(job: ScanBatchJob): Promise<RemoteScanJobRecord> {
  return parseJson<RemoteScanJobRecord>(
    await fetch("/api/remote/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job }),
    }),
  );
}

export async function listRemoteScanJobs(): Promise<RemoteScanJobRecord[]> {
  return parseJson<RemoteScanJobRecord[]>(await fetch("/api/remote/jobs"));
}

export async function getRemoteScanJob(jobId: string): Promise<RemoteScanJobRecord> {
  return parseJson<RemoteScanJobRecord>(await fetch(`/api/remote/jobs/${encodeURIComponent(jobId)}`));
}

export async function getRemoteScanResult(jobId: string): Promise<ScanBatchResult> {
  return parseJson<ScanBatchResult>(await fetch(`/api/remote/jobs/${encodeURIComponent(jobId)}/result`));
}

export async function cancelRemoteScanJob(jobId: string): Promise<RemoteScanJobRecord> {
  return parseJson<RemoteScanJobRecord>(
    await fetch(`/api/remote/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    }),
  );
}
