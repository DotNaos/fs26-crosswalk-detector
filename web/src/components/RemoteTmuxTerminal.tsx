import { useEffect, useMemo, useRef, useState } from "react";
import { Card, Spinner } from "@heroui/react";
import { Terminal, useTerminal } from "@wterm/react";
import "@wterm/react/css";

type RemoteTmuxTerminalProps = {
  jobId: string;
  tmuxSession: string;
};

export function RemoteTmuxTerminal({ jobId, tmuxSession }: RemoteTmuxTerminalProps) {
  const { ref, write } = useTerminal();
  const socketRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const socketUrl = useMemo(() => {
    const url = new URL(`/api/remote/jobs/${encodeURIComponent(jobId)}/terminal`, window.location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
  }, [jobId]);

  useEffect(() => {
    setConnectionError(null);
    setConnected(false);
    const socket = new WebSocket(socketUrl);
    socketRef.current = socket;

    socket.addEventListener("open", () => setConnected(true));
    socket.addEventListener("message", (event) => {
      if (typeof event.data === "string") {
        write(event.data);
        return;
      }
      if (event.data instanceof ArrayBuffer) {
        write(new Uint8Array(event.data));
      }
    });
    socket.addEventListener("close", () => setConnected(false));
    socket.addEventListener("error", () => setConnectionError("Terminal connection failed."));

    return () => {
      socket.close();
      if (socketRef.current === socket) {
        socketRef.current = null;
      }
    };
  }, [socketUrl, write]);

  return (
    <Card className="min-h-0 shadow-xl" variant="tertiary">
      <Card.Header>
        <div className="flex min-w-0 items-center justify-between gap-3">
          <div className="flex min-w-0 flex-col gap-1">
            <Card.Title>Live terminal</Card.Title>
            <Card.Description className="truncate">{connected ? tmuxSession : connectionError ?? "Connecting…"}</Card.Description>
          </div>
          {!connected && !connectionError ? <Spinner size="sm" /> : null}
        </div>
      </Card.Header>
      <Card.Content className="min-h-0">
        <div className="h-[min(56dvh,28rem)] min-h-64 overflow-hidden rounded-xl border border-divider">
          <Terminal
            ref={ref}
            autoResize
            className="h-full w-full"
            cursorBlink
            onData={(data) => socketRef.current?.send(data)}
            theme="monokai"
          />
        </div>
      </Card.Content>
    </Card>
  );
}
