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
  phone_number: z.string().optional(),
});

const createTeacherSchema = z.object({
  username: z.string().min(3),
  full_name: z.string().min(1),
  email: z.string().email(),
});

const updateUserSchema = z
  .object({
    full_name: z.string().min(1).optional(),
    phone_number: z.string().min(1).optional(),
    email: z.string().email("Invalid email format").optional(),
  })
  .refine((data) => data.full_name || data.phone_number || data.email, {
    message:
      "At least one field (full_name, phone_number, or email) must be provided",
  });

const changePasswordSchema = z.object({
  currentPassword: z
    .string()
    .min(1, { message: "Current password is required" }),
  newPassword: z
    .string()
    .min(6, { message: "New password must be at least 6 characters long" }),
});

const adminChangePasswordSchema = z.object({
  newPassword: z
    .string()
    .min(6, { message: "New password must be at least 6 characters long" }),
});

const userIdParamSchema = z.object({
  userId: z.string().regex(/^\d+$/, { message: "userId must be a number" }),
});

const teacherIdParamSchema = z.object({
  teacherId: z.string().regex(/^\d+$/, { message: "teacherId must be a number" }),
});

const paginationQuerySchema = z.object({
  page: z.coerce.number().int().min(1).optional().default(1),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
});

const createAssistantSchema = registerSchema
  .extend({
    class_ids: z
      .array(z.number().int().min(1))
      .min(1, "At least one class ID is required"),
  })
  .omit({ role: true }); // Role is fixed to 'Assistant', so we don't need it from the input.

// Teacher-centric grade weights have been removed. All grade weights are now class-centric
// via the `grade_weights` table keyed by `class_id`. Any legacy teacher-based initialization
// logic has been deleted.

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

  // Create an assistant and assign them to classes
  usersApi.post(
    "/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", createAssistantSchema),
    async (c) => {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin" && payload.role !== "Teacher") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const { username, password, full_name, email, phone_number, class_ids } =
        c.req.valid("json");

      const connection = await pool.getConnection();
      try {
        await connection.beginTransaction();

        // 1. Validate class IDs
        const placeholders = class_ids.map(() => "?").join(",");
        const [classRows]: any[] = await connection.query(
          `SELECT class_id, teacher_id FROM classes WHERE class_id IN (${placeholders})`,
          class_ids,
        );

        if (classRows.length !== class_ids.length) {
          await connection.rollback();
          return c.json({ error: "One or more class IDs are invalid" }, 400);
        }

        // 2. If user is a teacher, verify they own all the classes
        if (payload.role === "Teacher") {
          const teacherId = payload.sub;
          for (const row of classRows) {
            if (row.teacher_id !== teacherId) {
              await connection.rollback();
              return c.json(
                {
                  error: `Forbidden to assign assistant to class ID ${row.class_id} which you do not own.`,
                },
                403,
              );
            }
          }
        }

        // 3. Create the new user with 'Assistant' role
        const password_hash = await Bun.password.hash(password, {
          algorithm: "bcrypt",
          cost: 10,
        });
        const [userResult]: any = await connection.query(
          "INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES (?, ?, ?, ?, 'Assistant', ?)",
          [username, password_hash, full_name, email, phone_number || null],
        );
        const newAssistantId = userResult.insertId;

        // 4. Assign the new assistant to the classes
        const assignmentValues = class_ids.map((classId) => [
          classId,
          newAssistantId,
        ]);
        await connection.query(
          "INSERT INTO class_assistants (class_id, assistant_id) VALUES ?",
          [assignmentValues],
        );

        await connection.commit();

        const newUser = {
          user_id: newAssistantId,
          username,
          full_name,
          email,
          phone_number: phone_number || null,
          role: "Assistant",
          assigned_classes: class_ids,
        };
        return c.json(newUser, 201);
      } catch (error: any) {
        await connection.rollback();
        if (error.code === "ER_DUP_ENTRY") {
          if (error.message.includes("users.username")) {
            return c.json({ error: "Username already exists" }, 409);
          }
          if (error.message.includes("users.email")) {
            return c.json({ error: "Email already exists" }, 409);
          }
        }
        console.error("Error creating and assigning assistant:", error);
        throw error;
      } finally {
        connection.release();
      }
    },
  );

  // Admin: Create a new teacher user
  usersApi.post(
    "/users/teachers",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", createTeacherSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { username, full_name, email } = c.req.valid("json");

        // Use a default password for admin-created teachers
        const defaultPasswordHash =
          "$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG";

        const [result]: any = await pool.query(
          "INSERT INTO users (username, password_hash, full_name, email, role) VALUES (?, ?, ?, ?, 'Teacher')",
          [username, defaultPasswordHash, full_name, email],
        );

        const newTeacherId = result.insertId;

        const newUser = {
          user_id: newTeacherId,
          username,
          full_name,
          email,
          role: "Teacher",
        };
        return c.json(newUser, 201);
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          if (error.message.includes("users.username")) {
            return c.json({ error: "Username already exists" }, 409);
          }
          if (error.message.includes("users.email")) {
            return c.json({ error: "Email already exists" }, 409);
          }
        }
        console.error("Error creating teacher:", error);
        throw error;
      }
    },
  );

  // Get all assistants associated with a specific teacher through their classes
  usersApi.get(
    "/teachers/:teacherId/assistants",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", teacherIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { teacherId } = c.req.valid("param");
        const teacherIdNumber = parseInt(teacherId, 10);

        // Authorization: Admin can see anyone's, teachers can only see their own.
        if (payload.role !== "Admin" && payload.sub !== teacherIdNumber) {
          return c.json({ error: "Forbidden" }, 403);
        }

        const sql = `
          SELECT DISTINCT
            u.user_id,
            u.username,
            u.full_name,
            u.email,
            u.phone_number
          FROM users u
          JOIN class_assistants ca ON u.user_id = ca.assistant_id
          JOIN classes c ON ca.class_id = c.class_id
          WHERE c.teacher_id = ? AND u.role = 'Assistant'
        `;

        const [assistantRows] = await pool.query(sql, [teacherIdNumber]);

        return c.json(assistantRows);
      } catch (error) {
        console.error("Error fetching assistants for teacher:", error);
        throw error;
      }
    },
  );

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

  // Get all assistants (Admin and Teacher only)
  usersApi.get("/assistants", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Admin" && payload.role !== "Teacher") {
        return c.json({ error: "Forbidden" }, 403);
      }
      const [rows] = await pool.query(
        "SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE role = 'Assistant'",
      );
      return c.json(rows);
    } catch (error) {
      console.error("Error fetching assistants:", error);
      throw error;
    }
  });

  // Register a new user
  usersApi.post(
    "/users",
    zValidator("json", registerSchema),
    async (c) => {
      try {
        const { username, password, full_name, email, phone_number } =
          c.req.valid("json");
        const password_hash = await Bun.password.hash(password, {
          algorithm: "bcrypt",
          cost: 10,
        });

        const [result]: any = await pool.query(
          "INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES (?, ?, ?, ?, 'Student', ?)",
          [
            username,
            password_hash,
            full_name,
            email,
            phone_number || null,
          ],
        );

        const newUser = {
          user_id: result.insertId,
          username,
          full_name,
          email,
          role: "Student",
          phone_number: phone_number || null,
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
        "SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?",
        [userId],
      );
      const user = rows[0];

      if (!user) {
        return c.json({ error: "User not found" }, 404);
      }

      if (user.role === "Student") {
        const [classRows]: any[] = await pool.query(
          `SELECT c.class_id,
                  c.class_name,
                  c.class_code,
                  c.teacher_id,
                  teacher.full_name AS teacher_name
             FROM enrollments e
             JOIN classes c ON c.class_id = e.class_id
        LEFT JOIN users teacher ON teacher.user_id = c.teacher_id
            WHERE e.student_id = ?
            LIMIT 1`,
          [userId],
        );
        user.class = classRows[0] ?? null;
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
        const { full_name, phone_number, email } = c.req.valid("json");

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
        if (email) {
          fields.push("email = ?");
          values.push(email);
        }
        values.push(userId);

        const sql = `UPDATE users SET ${fields.join(", ")} WHERE user_id = ?`;
        await pool.query(sql, values);

        const [rows]: any[] = await pool.query(
          "SELECT user_id, username, full_name, email, phone_number, role, created_at FROM users WHERE user_id = ?",
          [userId],
        );
        return c.json(rows[0]);
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Email already exists" }, 409);
        }
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
    "/users/:userId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", userIdParamSchema),
    async (c) => {
      try {
        const requesterPayload = c.get("jwtPayload");
        const { userId } = c.req.valid("param");
        const targetUserId = parseInt(userId, 10);

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
    "/users/:userId/password",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", userIdParamSchema),
    zValidator("json", adminChangePasswordSchema),
    async (c) => {
      try {
        const requesterPayload = c.get("jwtPayload");
        if (requesterPayload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { userId } = c.req.valid("param");
        const targetUserId = parseInt(userId, 10);
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

  // Teacher/Admin: Reset a student's password to default
  usersApi.post(
    "/users/:userId/reset-password",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", userIdParamSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { userId } = c.req.valid("param");
        const targetUserId = parseInt(userId, 10);

        if (payload.role !== "Admin" && payload.role !== "Teacher") {
          return c.json({ error: "Forbidden" }, 403);
        }

        // 1. Fetch target user's role
        const [userRows]: any[] = await pool.query(
          "SELECT role FROM users WHERE user_id = ?",
          [targetUserId],
        );
        const targetUser = userRows[0];

        if (!targetUser) {
          return c.json({ error: "User not found" }, 404);
        }

        if (targetUser.role !== "Student") {
          return c.json({ error: "Can only reset passwords for students" }, 400);
        }

        // 2. Authorization check for teachers
        if (payload.role === "Teacher") {
          const teacherId = payload.sub;
          const [enrollmentRows]: any[] = await pool.query(
            `SELECT 1
               FROM enrollments e
               JOIN classes c ON e.class_id = c.class_id
              WHERE e.student_id = ? AND c.teacher_id = ?
              LIMIT 1`,
            [targetUserId, teacherId],
          );

          if (enrollmentRows.length === 0) {
            return c.json({ error: "Forbidden: Student is not in one of your classes" }, 403);
          }
        }
        // Admins are allowed to proceed

        // 3. Update password to the default hash
        const defaultPasswordHash = "$2b$10$DVZtxD9Efhm/j8.LSWMFx.hfdpk81iHO4Tv0fUaDr1LlW/mFrgXrG";
        const [result]: any = await pool.query(
          "UPDATE users SET password_hash = ? WHERE user_id = ?",
          [defaultPasswordHash, targetUserId],
        );

        if (result.affectedRows === 0) {
          // This case should be rare given the check above, but good for safety
          return c.json({ error: "User not found" }, 404);
        }

        return c.json({ message: "Password reset to default successfully" });
      } catch (error) {
        console.error("Error resetting password:", error);
        throw error;
      }
    },
  );

  // Delete user by ID (Admin only)
  usersApi.delete(
    "/users/:userId",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", userIdParamSchema),
    async (c) => {
      try {
        const requesterPayload = c.get("jwtPayload");
        const { userId } = c.req.valid("param");
        const targetUserId = parseInt(userId, 10);

        if (requesterPayload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        if (requesterPayload.sub === targetUserId) {
          return c.json({ error: "Admin cannot delete themselves" }, 400);
        }

        const connection = await pool.getConnection();
        try {
          await connection.beginTransaction();

          const [userRows]: any[] = await connection.query(
            "SELECT role FROM users WHERE user_id = ?",
            [targetUserId],
          );

          const userRecord = userRows[0];
          if (!userRecord) {
            await connection.rollback();
            return c.json({ error: "User not found" }, 404);
          }

          const cleanupSummary: { deletedClasses?: number; removedAssistantAssignments?: number; removedEnrollments?: number } = {};

          switch (userRecord.role) {
            case "Teacher": {
              const [classRows]: any[] = await connection.query(
                "SELECT class_id FROM classes WHERE teacher_id = ?",
                [targetUserId],
              );
              if (classRows.length > 0) {
                await connection.query("DELETE FROM classes WHERE teacher_id = ?", [
                  targetUserId,
                ]);
                cleanupSummary.deletedClasses = classRows.length;
              }
              break;
            }
            case "Assistant": {
              const [assignmentRows]: any[] = await connection.query(
                "SELECT class_id FROM class_assistants WHERE assistant_id = ?",
                [targetUserId],
              );
              if (assignmentRows.length > 0) {
                await connection.query(
                  "DELETE FROM class_assistants WHERE assistant_id = ?",
                  [targetUserId],
                );
                cleanupSummary.removedAssistantAssignments = assignmentRows.length;
              }
              break;
            }
            case "Student": {
              const [enrollmentRows]: any[] = await connection.query(
                "SELECT class_id FROM enrollments WHERE student_id = ?",
                [targetUserId],
              );
              if (enrollmentRows.length > 0) {
                await connection.query(
                  "DELETE FROM enrollments WHERE student_id = ?",
                  [targetUserId],
                );
                cleanupSummary.removedEnrollments = enrollmentRows.length;
              }
              break;
            }
            default:
              break;
          }

          await connection.query("DELETE FROM users WHERE user_id = ?", [targetUserId]);

          await connection.commit();

          return c.json({
            message: "User deleted successfully",
            cleanup: cleanupSummary,
          });
        } catch (error) {
          await connection.rollback();
          console.error("Error deleting user with cleanup:", error);
          throw error;
        } finally {
          connection.release();
        }
      } catch (error) {
        console.error("Error deleting user:", error);
        throw error;
      }
    },
  );

  return usersApi;
};
