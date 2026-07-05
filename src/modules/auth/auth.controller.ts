import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { loginSchema } from "./auth.schema";
import { login } from "./auth.service";
import { enforceLoginRateLimit } from "./auth.rate-limit";
import { handleUnexpectedError } from "../../utils/error-handler";

const authController = new Hono();

authController.post(
  "/api/v1/sessions",
  zValidator("json", loginSchema),
  async (c) => {
    try {
      const input = c.req.valid("json");
      const rateLimitResponse = enforceLoginRateLimit(c, input);
      if (rateLimitResponse) return rateLimitResponse;

      const result = await login(input);

      if (!result.success) {
        const { status, ...body } = result;
        return c.json(body, status);
      }

      return c.json({ token: result.token, user: result.user });
    } catch (error) {
      return handleUnexpectedError(c, error, "登录失败");
    }
  },
);

export default authController;
