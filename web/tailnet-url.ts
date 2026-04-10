const status = Bun.spawnSync(["tailscale", "status", "--json"], {
  cwd: import.meta.dir,
  stdout: "pipe",
  stderr: "pipe",
});

if (status.exitCode !== 0) {
  const stderr = new TextDecoder().decode(status.stderr).trim();
  console.error(stderr || "tailscale status failed");
  process.exit(status.exitCode);
}

const parsed = JSON.parse(new TextDecoder().decode(status.stdout)) as {
  Self?: { DNSName?: string };
};
const dnsName = parsed.Self?.DNSName?.replace(/\.$/, "");

if (!dnsName) {
  console.error("Could not determine the Tailscale DNS name for this machine.");
  process.exit(1);
}

console.log(`Expected Tailnet URL: https://${dnsName}`);
