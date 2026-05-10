import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { loadLocalEnv } from "./src/local-env";

const PORTLESS_URL = "http://crosswalk-review.localhost:1355";
const PORTLESS_APP_NAME = "crosswalk-review";
const WEB_ROOT = import.meta.dir;
const PROJECT_ROOT = resolve(WEB_ROOT, "..");

type Subprocess = {
  kill(): void;
  exited: Promise<number>;
};

let client: Subprocess | null = null;
let apiServer: Subprocess | null = null;

loadLocalEnv(PROJECT_ROOT, WEB_ROOT);

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function commandExists(name: string) {
  return (
    spawnSync("/bin/zsh", ["-lc", `command -v ${name}`], {
      stdio: "ignore",
    }).status ?? 1
  ) === 0;
}

function startClient(): Subprocess {
  return Bun.spawn(["bun", "run", "dev:client"], {
    cwd: WEB_ROOT,
    stdout: "inherit",
    stderr: "inherit",
    stdin: "inherit",
  });
}

function startApiServer(): Subprocess {
  return Bun.spawn(["env", "PORT=8787", "bun", "run", "src/server.ts"], {
    cwd: WEB_ROOT,
    stdout: "inherit",
    stderr: "inherit",
    stdin: "inherit",
  });
}

function activePortlessRoute(name: string) {
  if (!commandExists("portless")) return null;
  const listed = spawnSync("portless", ["list"], {
    cwd: WEB_ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "ignore"],
  });
  const output = listed.stdout ?? "";
  const pattern = new RegExp(`http://${name}\\.localhost:1355\\s+->\\s+localhost:(\\d+)\\s+\\(pid\\s+(\\d+)\\)`);
  const match = output.match(pattern);
  return match
    ? {
        port: Number(match[1]),
        pid: Number(match[2]),
      }
    : null;
}

async function ensureClientOwnership() {
  const route = activePortlessRoute(PORTLESS_APP_NAME);
  const pid = route?.pid;
  if (!pid || pid === process.pid) return;

  try {
    process.kill(pid, "SIGTERM");
  } catch {
    return;
  }

  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      process.kill(pid, 0);
      await sleep(150);
    } catch {
      return;
    }
  }

  try {
    process.kill(pid, "SIGKILL");
  } catch {
    // Ignore if the process already exited.
  }
}

async function ensureTailnetShare() {
  if (process.env.TAILSCALE_SERVE === "0") {
    console.log("Skipping Tailscale Serve because TAILSCALE_SERVE=0");
    return;
  }
  if (!commandExists("tailscale")) {
    console.log("Skipping Tailscale Serve because tailscale is not installed");
    return;
  }

  let route = null as ReturnType<typeof activePortlessRoute>;
  for (let attempt = 0; attempt < 30; attempt += 1) {
    route = activePortlessRoute(PORTLESS_APP_NAME);
    if (route?.port) break;
    await sleep(150);
  }

  if (!route?.port) {
    console.warn("Skipping Tailscale Serve because the portless route is not ready yet");
    return;
  }

  const tailscaleTarget = `http://127.0.0.1:${route.port}`;

  const serve = spawnSync("tailscale", ["serve", "--bg", tailscaleTarget], {
    cwd: WEB_ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const stdout = serve.stdout?.trim() ?? "";
  const stderr = serve.stderr?.trim() ?? "";
  if ((serve.status ?? 1) !== 0) {
    const combined = [stdout, stderr].filter(Boolean).join("\n");
    const enableUrl = combined.match(/https:\/\/\S+/)?.[0];
    console.warn("Tailscale Serve could not be enabled");
    if (enableUrl) {
      console.warn(`Enable it once here: ${enableUrl}`);
    }
    if (combined) console.warn(combined);
    return;
  }

  const status = spawnSync("tailscale", ["status", "--json"], {
    cwd: WEB_ROOT,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const statusStdout = status.stdout?.trim() ?? "";
  let tailnetUrl: string | null = null;
  if ((status.status ?? 1) === 0) {
    try {
      const parsed = JSON.parse(statusStdout) as {
        Self?: { DNSName?: string };
      };
      const dnsName = parsed.Self?.DNSName?.replace(/\.$/, "");
      if (dnsName) {
        tailnetUrl = `https://${dnsName}`;
      }
    } catch {
      // ignore parse failures and fall back to stdout
    }
  }

  console.log("Tailscale Serve is enabled for the portless URL");
  console.log(`Tailscale backend target: ${tailscaleTarget}`);
  if (tailnetUrl) {
    console.log(`Tailnet URL: ${tailnetUrl}`);
  }
  if (stdout) console.log(stdout);
  if (stderr) console.log(stderr);
}

const shutdown = () => {
  client?.kill();
  apiServer?.kill();
};

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

await ensureClientOwnership();
apiServer = startApiServer();
await sleep(600);
client = startClient();
await ensureTailnetShare();
await Promise.race([client?.exited, apiServer?.exited].filter(Boolean));
shutdown();
