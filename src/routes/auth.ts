import { Hono } from "hono";
import { sign } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";

// Zod Schema for Validation
const loginSchema = z.object({
  username: z.string().min(1, { message: "Username is required" }),
  password: z.string().min(1, { message: "Password is required" }),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const authApi = new Hono();

  // POST /api/v1/sessions - create an authenticated session (login)
  authApi.post("/api/v1/sessions", zValidator("json", loginSchema), async (c) => {
    try {
      const { username, password } = c.req.valid("json");

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

      const payload = {
        sub: user.user_id,
        username: user.username,
        role: user.role,
        exp: Math.floor(Date.now() / 1000) + 60 * 60 * 24, // 24 hours
      };

      const token = await sign(payload, JWT_SECRET);
      return c.json({ token });
    } catch (error) {
      console.error("Login error:", error);
      throw error;
    }
  });

  return authApi;
};
