import { Hono } from "hono";
import { sign } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import { handleUnexpectedError } from "../utils/error-handler";

// Zod Schema for Validation
const loginSchema = z.object({
  username: z.string().min(1, { message: "Username is required" }),
  password: z.string().min(1, { message: "Password is required" }),
  role: z.enum(["Student", "Teacher", "Assistant", "Admin"]).optional(),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const authApi = new Hono();

  // POST /api/v1/sessions - create an authenticated session (login)
  authApi.post("/api/v1/sessions", zValidator("json", loginSchema), async (c) => {
    try {
      const { username, password, role } = c.req.valid("json");

      const [rows]: any[] = await pool.query(
        "SELECT * FROM users WHERE username = ?",
        [username],
      );
      const user = rows[0];

      if (!user) {
        return c.json({ error: "Invalid credentials" }, 401);
      }

      const passwordMatch = await Bun.password.verify(
        password,
        user.password_hash,
      );

      if (!passwordMatch) {
        return c.json({ error: "Invalid credentials" }, 401);
      }

      // Validate role if provided
      if (role && user.role !== role) {
        const roleNames: Record<string, string> = {
          Student: "学生",
          Teacher: "教师",
          Assistant: "助教",
          Admin: "管理员"
        };
        return c.json({
          error: "角色不匹配",
          message: `该账号的角色是${roleNames[user.role] || user.role}，但您正在尝试以${roleNames[role] || role}身份登录`
        }, 403);
      }

      const payload = {
        sub: user.user_id,
        username: user.username,
        role: user.role,
        exp: Math.floor(Date.now() / 1000) + 60 * 60 * 24, // 24 hours
      };

      const token = await sign(payload, JWT_SECRET);
      return c.json({ token, user: { role: user.role } });
    } catch (error) {
      return handleUnexpectedError(c, error, "Login failed");
    }
  });

  return authApi;
};
