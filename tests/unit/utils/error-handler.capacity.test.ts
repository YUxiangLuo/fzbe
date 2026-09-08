import { expect, test } from "bun:test";
import { Hono } from "hono";
import { handleUnexpectedError } from "../../../src/utils/error-handler";

test.each([
  new Error("Queue limit reached."),
  Object.assign(new Error("Too many connections"), { code: "ER_CON_COUNT_ERROR" }),
])("maps exhausted database capacity to recoverable 503: %s", async (error) => {
  const app = new Hono().get("/", (c) => handleUnexpectedError(c, error));
  const response = await app.request("/");
  expect(response.status).toBe(503);
  expect(response.headers.get("Retry-After")).toBe("2");
  expect(await response.json()).toMatchObject({ code: "DATABASE_BUSY", retryable: true });
});

test("keeps unrelated failures as internal errors", async () => {
  const app = new Hono().get("/", (c) => handleUnexpectedError(c, new Error("unexpected SQL failure")));
  const response = await app.request("/");
  expect(response.status).toBe(500);
  expect(response.headers.has("Retry-After")).toBe(false);
});
