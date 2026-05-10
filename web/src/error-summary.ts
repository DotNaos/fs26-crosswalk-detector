export function summarizeErrorMessage(message: string | null | undefined, fallback: string) {
  if (!message) return fallback;
  const normalized = message.replace(/^Error:\s*/g, "").trim();
  const firstLine = normalized.split(/\r?\n/).find((line) => line.trim().length > 0)?.trim() ?? fallback;
  return firstLine.length > 140 ? `${firstLine.slice(0, 137)}...` : firstLine;
}
