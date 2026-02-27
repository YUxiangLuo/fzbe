import type { Context } from "hono";
import logger from "./logger";

export const handleUnexpectedError = (
  c: Context,
  error: unknown,
  message = "服务器内部错误",
) => {
  logger.error(
    {
      err: error instanceof Error ? { message: error.message, stack: error.stack } : error,
      path: c.req.path,
      method: c.req.method,
    },
    message,
  );

  return c.json({ error: message }, 500);
};

export const logWarning = (message: string, details?: Record<string, unknown>) => {
  logger.warn(details ?? {}, message);
};
