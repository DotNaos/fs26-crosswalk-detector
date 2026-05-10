import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const root = join(process.cwd(), "src");
const guardedTagPattern = /<[^>]*\bdata-scroll-guard=(?:"[^"]*"|'[^']*')[^>]*>/gs;
const classNamePattern = /\bclassName=(?:"([^"]*)"|'([^']*)'|\{`([^`]*)`\})/s;
const overflowPattern = /\boverflow-(?:y-)?(?:auto|scroll)\b/;
const heightLimitPattern = /\b(?:max-h-|h-\[|bottom-|inset-)/;
const failures = [];

function walk(directory) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    const stat = statSync(path);
    if (stat.isDirectory()) {
      walk(path);
      continue;
    }
    if (!path.endsWith(".tsx")) continue;
    checkFile(path);
  }
}

function checkFile(path) {
  const source = readFileSync(path, "utf8");
  for (const match of source.matchAll(guardedTagPattern)) {
    const tag = match[0];
    const classMatch = tag.match(classNamePattern);
    const className = classMatch?.[1] ?? classMatch?.[2] ?? classMatch?.[3] ?? "";
    if (!overflowPattern.test(className) || !heightLimitPattern.test(className)) {
      const line = source.slice(0, match.index).split("\n").length;
      failures.push(`${path}:${line} needs both overflow-y-auto/overflow-auto and a height limit`);
    }
  }
}

walk(root);

if (failures.length > 0) {
  console.error("Scroll guard check failed:");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log("Scroll guard check passed.");
