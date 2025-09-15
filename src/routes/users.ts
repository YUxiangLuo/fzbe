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

const changePasswordSchema = z.object({
  currentPassword: z.string().min(1, { message: "Current password is required" }),
  newPassword: z.string().min(6, { message: "New password must be at least 6 characters long" }),
});

const adminChangePasswordSchema = z.object({
  newPassword: z.string().min(6, { message: "New password must be at least 6 characters long" }),
});

const paramsSchema = z.object({
  id: z.string().regex(/^\d+$/, { message: "ID must be a number" }),
});

const paginationQuerySchema = z.object({
  page: z.coerce.number().int().min(1).optional().default(1),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
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
  usersApi.get(
    "/users",
    jwt({ secret: JWT_SECRET }),
    zValidator("query", paginationQuerySchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { page, limit } = c.req.valid("query");
        const offset = (page - 1) * limit;

        // Perform two queries in parallel: one for the total count, one for the paginated data
        const [totalResult, rows] = await Promise.all([
          pool.query("SELECT COUNT(*) as total FROM users"),
          pool.query(
            "SELECT user_id, username, full_name, email, role, created_at FROM users ORDER BY user_id ASC LIMIT ? OFFSET ?",
            [limit, offset],
          ),
        ]);

        const totalItems = (totalResult[0] as any)[0].total;
        const totalPages = Math.ceil(totalItems / limit);

        return c.json({
          data: rows[0],
          pagination: {
            currentPage: page,
            pageSize: limit,
            totalItems,
            totalPages,
          },
        });
      } catch (error) {
        console.error("Error fetching users:", error);
        throw error;
      }
    },
  );

  // Register a new user
  usersApi.post(
    "/users/register",
    zValidator("json", registerSchema),
    async (c) => {
      try {
        const { username, password, full_name, email, role } =
          c.req.valid("json");
        const password_hash = await Bun.password.hash(password, {
          algorithm: "bcrypt",
          cost: 10,
        });

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

  // Change current user's password
  usersApi.put(
    "/users/me/password",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", changePasswordSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const userId = payload.sub;
        const { currentPassword, newPassword } = c.req.valid("json");

        // 1. Fetch user's current hash
        const [rows]: any[] = await pool.query(
          "SELECT password_hash FROM users WHERE user_id = ?",
          [userId],
        );
        const user = rows[0];

        if (!user) {
          // This case should ideally not be reached if JWT is valid
          return c.json({ error: "User not found" }, 404);
        }

        // 2. Verify current password
        const passwordMatch = await Bun.password.verify(
          currentPassword,
          user.password_hash,
        );

        if (!passwordMatch) {
          return c.json({ error: "Invalid current password" }, 401);
        }

        // 3. Hash new password and update database
        const newPasswordHash = await Bun.password.hash(newPassword, {
          algorithm: "bcrypt",
          cost: 10,
        });
        await pool.query(
          "UPDATE users SET password_hash = ? WHERE user_id = ?",
          [newPasswordHash, userId],
        );

        return c.json({ message: "Password updated successfully" });
      } catch (error) {
        console.error("Error changing password:", error);
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

  // Admin: Force change user's password
  usersApi.put(
    "/users/:id/password",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    zValidator("json", adminChangePasswordSchema),
    async (c) => {
      try {
        const requesterPayload = c.get("jwtPayload");
        if (requesterPayload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { id } = c.req.valid("param");
        const targetUserId = parseInt(id, 10);
        const { newPassword } = c.req.valid("json");

        if (requesterPayload.sub === targetUserId) {
          return c.json(
            {
              error:
                "Admins should use the /users/me/password route to change their own password.",
            },
            400,
          );
        }

        const newPasswordHash = await Bun.password.hash(newPassword, {
          algorithm: "bcrypt",
          cost: 10,
        });

        const [result]: any = await pool.query(
          "UPDATE users SET password_hash = ? WHERE user_id = ?",
          [newPasswordHash, targetUserId],
        );

        if (result.affectedRows === 0) {
          return c.json({ error: "User not found" }, 404);
        }

        return c.json({ message: "Password updated successfully for user" });
      } catch (error) {
        console.error("Error force changing password:", error);
        throw error;
      }
    },
  );

  // Delete user by ID (Admin only)
  usersApi.delete(
    "/users/:id",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", paramsSchema),
    async (c) => {
      try {
        const requesterPayload = c.get("jwtPayload");
        const { id } = c.req.valid("param");
        const targetUserId = parseInt(id, 10);

        if (requesterPayload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        if (requesterPayload.sub === targetUserId) {
          return c.json({ error: "Admin cannot delete themselves" }, 400);
        }

        const [result]: any = await pool.query(
          "DELETE FROM users WHERE user_id = ?",
          [targetUserId],
        );

        if (result.affectedRows === 0) {
          return c.json({ error: "User not found" }, 404);
        }

        return c.json({ message: "User deleted successfully" });
      } catch (error) {
        console.error("Error deleting user:", error);
        throw error;
      }
    },
  );

  return usersApi;
}
