import { ModelTimeoutError } from "./model-errors";

const PYTHON_STDERR_TAGS = [
  { marker: "[VALIDATION]", servicePrefix: "[PYTHON_VALIDATION]" },
  { marker: "[DATA]", servicePrefix: "[PYTHON_DATA]" },
  { marker: "[TIMEOUT]", servicePrefix: "[PYTHON_TIMEOUT]" },
] as const;

const PYTHON_SERVICE_ERROR_PATTERN = /^\[PYTHON_(VALIDATION|DATA)\]\s*([\s\S]*)$/;

export const extractPythonTaggedError = (
  stderr: string,
  timeoutMs: number,
): Error | null => {
  const lines = stderr
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index]!;
    for (const { marker, servicePrefix } of PYTHON_STDERR_TAGS) {
      const markerIndex = line.indexOf(marker);
      if (markerIndex === -1) continue;
      const detail = line.slice(markerIndex + marker.length).trim();
      if (marker === "[TIMEOUT]") {
        return new ModelTimeoutError(
          detail || "Python 子进程执行超时",
          timeoutMs,
          "python_subprocess",
        );
      }
      return new Error(`${servicePrefix}${detail ? ` ${detail}` : ""}`);
    }
  }
  return null;
};

export const parsePythonServiceError = (error: unknown): string | null => {
  const message = error instanceof Error ? error.message : "";
  const match = message.match(PYTHON_SERVICE_ERROR_PATTERN);
  return match ? (match[2] ?? "").trim() : null;
};
