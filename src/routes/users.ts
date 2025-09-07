import { Hono } from "hono";
import { jwt, sign } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";

// Zod Schemas for Validation
const loginSchema = z.object({
  username: z.string().min(1, { message: "Username is required" }),
  password: z.string().min(1, { message: "Password is required" }),
});

const registerSchema = z.object({
  username: z.string().min(3),
  password: z
    .string()
    .min(6, { message: "Password must be at least 6 characters long" }),
  full_name: z.string().min(1),
  email: z.string().email(),
  role: z.enum(["Student", "Teacher", "Assistant", "Admin"]).default("Student"),
});

const updateUserSchema = z
  .object({
    full_name: z.string().min(1).optional(),
    phone_number: z.string().min(1).optional(),
  })
  .refine((data) => data.full_name || data.phone_number, {
    message: "Either full_name or phone_number must be provided",
  });

const paramsSchema = z.object({
  id: z.string().regex(/^\d+$/, { message: "ID must be a number" }),
});

export default (pool: Pool, JWT_SECRET: string) => {
  const usersApi = new Hono();

  // --- Auth Routes ---
  usersApi.post("/auth/login", zValidator("json", loginSchema), async (c) => {
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

  // --- User Management Routes ---

  // Get all users (Admin only)
  usersApi.get("/users", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin") {
        return c.json({ error: "Forbidden" }, 403);
      }
      const [rows] = await pool.query(
        "SELECT user_id, username, full_name, email, role, created_at FROM users",
      );
      return c.json(rows);
    } catch (error) {
      console.error("Error fetching users:", error);
      throw error;
    }
  });

  // Register a new user
  usersApi.post(
    "/users/register",
    zValidator("json", registerSchema),
    async (c) => {
      try {
        const { username, password, full_name, email, role } =
          c.req.valid("json");
        const password_hash = await Bun.password.hash(password);

        const [result]: any = await pool.query(
          "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, ?)",
          [username, password_hash, full_name, email, role],
        );

        const newUser = {
          user_id: result.insertId,
          username,
          full_name,
          email,
          role,
        };
        return c.json(newUser, 201);
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Username or email already exists" }, 409);
        }
        console.error("Registration error:", error);
        return c.json({ error: "Internal Server Error" }, 500);
      }
    },
  );

  // Get current user's info
  usersApi.get("/users/me", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      const userId = payload.sub;

      const [rows]: any[] = await pool.query(
        "SELECT user_id, username, full_name, email, role, created_at FROM users WHERE user_id = ?",
        [userId],
      );
      const user = rows[0];

      if (!user) {
        return c.json({ error: "User not found" }, 404);
      }
      return c.json(user);
    } catch (error) {
      console.error("Error fetching current user:", error);
      throw error;
    }
  });

  // Update current user's info
  usersApi.put(
    "/users/me",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", updateUserSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const userId = payload.sub;
        const { full_name, phone_number } = c.req.valid("json");

        const fields = [];
        const values = [];
        if (full_name) {
          fields.push("full_name = ?");
          values.push(full_name);
        }
        if (phone_number) {
          fields.push("phone_number = ?");
          values.push(phone_number);
        }
        values.push(userId);

        const sql = `UPDATE users SET ${fields.join(", ")} WHERE user_id = ?`;
        await pool.query(sql, values);

        const [rows]: any[] = await pool.query(
          "SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?",
          [userId],
        );
        return c.json(rows[0]);
      } catch (error) {
        console.error("Error updating user:", error);
        throw error;
      }
    },
  );

  // Get user by ID
  usersApi.get(
    "/users/:id",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    async (c) => {
      try {
        const requesterPayload = c.get("jwtPayload");
        const { id } = c.req.valid("param");
        const targetUserId = parseInt(id, 10);

        if (
          requesterPayload.role !== "Admin" &&
          requesterPayload.sub !== targetUserId
        ) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const [rows]: any[] = await pool.query(
          "SELECT user_id, username, full_name, email, role, created_at FROM users WHERE user_id = ?",
          [targetUserId],
        );
        const user = rows[0];

        if (!user) {
          return c.json({ error: "User not found" }, 404);
        }
        return c.json(user);
      } catch (error) {
        console.error("Error fetching user by ID:", error);
        throw error;
      }
    },
  );

  return usersApi;
}
