import type { Context } from "hono";
import logger from "./logger";

export const handleUnexpectedError = (
  c: Context,
  error: unknown,
  message = "服务器内部错误",
) => {
  const databaseError = error as { code?: string; message?: string } | null;
  if (
    databaseError?.code === "ER_CON_COUNT_ERROR"
    || /^Queue limit reached\.?$/.test(databaseError?.message ?? "")
  ) {
    logger.warn({ path: c.req.path, method: c.req.method }, "Database capacity exhausted");
    c.header("Retry-After", "2");
    return c.json({
      error: "数据库服务繁忙，请稍后重试",
      code: "DATABASE_BUSY",
      retryable: true,
    }, 503);
  }
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
