import { useEffect, useState } from "react";
import { listRemoteScanJobs, loadRemoteControllerSnapshot } from "../remote-api";
import type { RemoteControllerSnapshot, RemoteScanJobRecord } from "../types";

export function useMobileRemoteJob(isMobileLayout: boolean) {
  const [activeRemoteJobId, setActiveRemoteJobId] = useState<string | null>(null);
  const [activeRemoteJob, setActiveRemoteJob] = useState<RemoteScanJobRecord | null>(null);
  const [remoteSnapshot, setRemoteSnapshot] = useState<RemoteControllerSnapshot | null>(null);

  useEffect(() => {
    if (!isMobileLayout) return;
    let cancelled = false;

    const syncRemoteJob = async () => {
      try {
        const [jobs, snapshot] = await Promise.all([listRemoteScanJobs(), loadRemoteControllerSnapshot()]);
        if (cancelled) return;
        setRemoteSnapshot(snapshot);
        const nextActiveJob = jobs.find((job) => job.id === activeRemoteJobId) ?? jobs[0] ?? null;
        setActiveRemoteJob(nextActiveJob);
        if (!activeRemoteJobId && nextActiveJob) {
          setActiveRemoteJobId(nextActiveJob.id);
        }
      } catch {
        // Keep the mobile HUD usable even if the remote controller is temporarily unavailable.
      }
    };

    void syncRemoteJob();
    const timer = window.setInterval(syncRemoteJob, 200);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeRemoteJobId, isMobileLayout]);

  return {
    activeRemoteJobId,
    setActiveRemoteJobId,
    activeRemoteJob,
    setActiveRemoteJob,
    remoteSnapshot,
  };
}
