import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

function normalizeEnvValue(rawValue: string) {
  const value = rawValue.trim();
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function parseEnvFile(path: string) {
  const entries = new Map<string, string>();
  const contents = readFileSync(path, "utf8");
  for (const line of contents.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const normalized = trimmed.startsWith("export ") ? trimmed.slice(7).trim() : trimmed;
    const separator = normalized.indexOf("=");
    if (separator <= 0) continue;
    const key = normalized.slice(0, separator).trim();
    const value = normalizeEnvValue(normalized.slice(separator + 1));
    if (!key) continue;
    entries.set(key, value);
  }
  return entries;
}

export function loadLocalEnv(projectRoot: string, webRoot: string) {
  const envFiles = [
    join(projectRoot, ".env"),
    join(projectRoot, ".env.local"),
    join(webRoot, ".env"),
    join(webRoot, ".env.local"),
  ];

  for (const path of envFiles) {
    if (!existsSync(path)) continue;
    const entries = parseEnvFile(path);
    for (const [key, value] of entries) {
      if (process.env[key] == null || process.env[key] === "") {
        process.env[key] = value;
      }
    }
  }
}
