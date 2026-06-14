const SOCKET_POLL_INTERVAL_MS = 250;

type SocketMessage = string | Buffer<ArrayBuffer>;

export type RemoteTerminalSocketData = {
  jobId: string;
  tmuxSession: string;
  localLogPath: string;
  lastSnapshot: string;
  pollTimer?: Timer;
};

export type RemoteUpgradeServer = {
  upgrade(request: Request, options: { data: RemoteTerminalSocketData }): boolean;
};

export type RemoteServeInstance = RemoteUpgradeServer & {
  port: number;
};

export type RemoteTerminalSocket = {
  data: RemoteTerminalSocketData;
  send(data: string | BufferSource): unknown;
  close(): void;
};

type RemoteWebsocketDeps = {
  tmuxSessionExists: (tmuxSession: string) => boolean;
  readTerminalSnapshot: (socketData: RemoteTerminalSocketData) => string;
  sendTmuxInput: (tmuxSession: string, input: string) => void;
};

export function createRemoteWebsocketHandlers({
  readTerminalSnapshot,
  tmuxSessionExists,
  sendTmuxInput,
}: RemoteWebsocketDeps) {
  return {
    open(ws: RemoteTerminalSocket) {
      try {
        const snapshot = readTerminalSnapshot(ws.data);
        ws.data.lastSnapshot = snapshot;
        ws.send(`\x1bc${snapshot}`);
        ws.data.pollTimer = setInterval(() => {
          try {
            const nextSnapshot = readTerminalSnapshot(ws.data);
            if (nextSnapshot === ws.data.lastSnapshot) return;
            if (nextSnapshot.startsWith(ws.data.lastSnapshot)) {
              ws.send(nextSnapshot.slice(ws.data.lastSnapshot.length));
            } else {
              ws.send(`\x1bc${nextSnapshot}`);
            }
            ws.data.lastSnapshot = nextSnapshot;
          } catch (error) {
            ws.send(`\r\n[terminal disconnected] ${String(error)}\r\n`);
            if (ws.data.pollTimer) {
              clearInterval(ws.data.pollTimer);
            }
            ws.close();
          }
        }, SOCKET_POLL_INTERVAL_MS);
      } catch (error) {
        ws.send(`\r\n[terminal unavailable] ${String(error)}\r\n`);
        ws.close();
      }
    },
    message(ws: RemoteTerminalSocket, message: SocketMessage) {
      try {
        if (!tmuxSessionExists(ws.data.tmuxSession)) {
          ws.send("\r\n[read-only] This run has finished. Showing the saved log.\r\n");
          return;
        }
        const input = typeof message === "string" ? message : Buffer.from(message).toString("utf8");
        sendTmuxInput(ws.data.tmuxSession, input);
      } catch (error) {
        ws.send(`\r\n[input failed] ${String(error)}\r\n`);
      }
    },
    close(ws: RemoteTerminalSocket) {
      if (ws.data.pollTimer) {
        clearInterval(ws.data.pollTimer);
      }
    },
  };
}
