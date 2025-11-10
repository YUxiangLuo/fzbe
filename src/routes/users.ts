import { Hono } from "hono";
import { jwt } from "hono/jwt";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import type { Pool } from "mysql2/promise";
import { getDefaultPasswordHash } from "../utils";
import csv from "csv-parser";
import path from "path";
import { mkdir, unlink } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { handleUnexpectedError, logWarning } from "../utils/error-handler";

// Zod Schemas for Validation
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

const PHONE_REGEX = /^\d{6,15}$/;
const MIN_TEACHER_NAME_LENGTH = 2;
const TEACHER_EMAIL_DOMAIN = "teacher.edu";

const DEFAULT_PASSWORD_HASH = getDefaultPasswordHash();

const createTeacherSchema = z.object({
  username: z.string().min(3),
  full_name: z.string().min(1),
  email: z.string().email().optional(),
  phone: z
    .string()
    .length(11, "手机号必须为11位")
    .regex(/^1\d{10}$/, "手机号必须以1开头的11位数字")
    .optional(),
  password: z
    .string()
    .min(6, "密码必须至少6位")
    .optional(),
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

const resetPasswordSchema = z.object({
  newPassword: z
    .string()
    .min(6, { message: "New password must be at least 6 characters long" })
    .optional(),
});

const userSearchQuerySchema = z.object({
  q: z.string().trim().min(1, { message: "Search term is required" }),
  page: z.coerce.number().int().min(1).optional().default(1),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
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

// Self-registration schemas for different roles
const roleParamSchema = z.object({
  role: z.enum(["Student", "Teacher", "Assistant"], "Invalid role. Must be Student, Teacher, or Assistant"),
});

const studentSelfRegisterSchema = z.object({
  username: z
    .string()
    .min(8, "Student username must be at least 8 digits")
    .regex(/^\d+$/, "Student username must contain only digits"),
  password: z
    .string()
    .min(6, "Password must be at least 6 characters long"),
  name: z
    .string()
    .min(1, "Name is required")
    .regex(/^[\u4e00-\u9fa5a-zA-Z\s]+$/, "Name can only contain Chinese characters, English letters, and spaces"),
  email: z.string().regex(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "Invalid email format"),
});

const teacherAssistantSelfRegisterSchema = z.object({
  username: z
    .string()
    .min(3, "Username must be at least 3 characters long")
    .regex(/^[a-zA-Z0-9]+$/, "Username can only contain letters and numbers"),
  password: z
    .string()
    .min(6, "Password must be at least 6 characters long"),
  name: z
    .string()
    .min(1, "Name is required")
    .regex(/^[\u4e00-\u9fa5a-zA-Z\s]+$/, "Name can only contain Chinese characters, English letters, and spaces"),
  email: z.string().regex(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, "Invalid email format"),
  phone: z
    .string()
    .length(11, "Phone number must be exactly 11 digits")
    .regex(/^\d{11}$/, "Phone number must contain only digits"),
});

// Teacher-centric grade weights have been removed. All grade weights are now class-centric
// via the `grade_weights` table keyed by `class_id`. Any legacy teacher-based initialization
// logic has been deleted.

export default (pool: Pool, JWT_SECRET: string) => {
  const usersApi = new Hono();

  // Self-registration endpoint (no authentication required)
  usersApi.post(
    "/api/v1/users/register/:role",
    zValidator("param", roleParamSchema),
    async (c) => {
      try {
        const { role } = c.req.valid("param");
        const body = await c.req.json();

        // Validate request body based on role
        let validatedData: any;
        if (role === "Student") {
          const result = studentSelfRegisterSchema.safeParse(body);
          if (!result.success) {
            return c.json(
              { error: "Validation failed", details: result.error.issues },
              400
            );
          }
          validatedData = result.data;
        } else if (role === "Teacher" || role === "Assistant") {
          const result = teacherAssistantSelfRegisterSchema.safeParse(body);
          if (!result.success) {
            return c.json(
              { error: "Validation failed", details: result.error.issues },
              400
            );
          }
          validatedData = result.data;
        } else {
          return c.json({ error: "Invalid role" }, 400);
        }

        const { username, password, name, email } = validatedData;
        const phone = validatedData.phone || null;

        // Hash the password
        const password_hash = await Bun.password.hash(password, {
          algorithm: "bcrypt",
          cost: 10,
        });

        try {
          // Insert the new user
          const [result]: any = await pool.query(
            "INSERT INTO users (username, password_hash, full_name, email, phone_number, role) VALUES (?, ?, ?, ?, ?, ?)",
            [username, password_hash, name, email, phone, role]
          );

          const newUserId = result.insertId;

          // Return the created user (without password hash)
          return c.json(
            {
              user_id: newUserId,
              username,
              full_name: name,
              email,
              phone_number: phone,
              role,
              message: `${role} account created successfully`,
            },
            201
          );
        } catch (error: any) {
          if (error.code === "ER_DUP_ENTRY") {
            if (error.message.includes("users.username")) {
              return c.json({ error: "Username already exists" }, 409);
            }
            if (error.message.includes("users.email")) {
              return c.json({ error: "Email already exists" }, 409);
            }
            if (error.message.includes("users.phone_number")) {
              return c.json({ error: "Phone number already exists" }, 409);
            }
          }
          return handleUnexpectedError(c, error, "Registration failed");
        }
      } catch (error) {
        return handleUnexpectedError(c, error, "Registration failed");
      }
    }
  );

  // Admin: Fuzzy search users by name/email/phone
  usersApi.get(
    "/api/v1/users/search",
    jwt({ secret: JWT_SECRET }),
    zValidator("query", userSearchQuerySchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { q, page, limit } = c.req.valid("query");
        const term = q.trim();
        if (!term) {
          return c.json({ error: "Search term is required" }, 400);
        }

        const offset = (page - 1) * limit;

        // Use FULLTEXT search only for longer terms (> 4 chars) to avoid min token size issues
        // For short terms (1-4 chars) or potential pinyin searches, use LIKE for better fuzzy matching
        // Fallback to LIKE for special characters
        const useFullText = term.length > 4 && /^[\w\s@.-]+$/.test(term);

        let countQuery: string;
        let dataQuery: string;
        let queryParams: any[];

        if (useFullText) {
          // Use FULLTEXT search with BOOLEAN MODE for better performance
          // Prepare search term for boolean mode (add wildcard for partial matching)
          const searchTerm = term.split(/\s+/).map(word => `+${word}*`).join(' ');

          countQuery = `
            SELECT COUNT(*) AS total
              FROM users
             WHERE MATCH(full_name, email, phone_number) AGAINST(? IN BOOLEAN MODE)
          `;

          dataQuery = `
            SELECT
              user_id,
              username,
              full_name,
              email,
              role,
              created_at,
              MATCH(full_name, email, phone_number) AGAINST(? IN BOOLEAN MODE) AS relevance
            FROM users
            WHERE MATCH(full_name, email, phone_number) AGAINST(? IN BOOLEAN MODE)
            ORDER BY relevance DESC, user_id ASC
            LIMIT ? OFFSET ?
          `;

          queryParams = [searchTerm];
        } else {
          // Fallback to LIKE search for special cases
          const likeTerm = `%${term}%`;

          countQuery = `
            SELECT COUNT(*) AS total
              FROM users
             WHERE full_name LIKE ?
                OR email LIKE ?
                OR phone_number LIKE ?
          `;

          dataQuery = `
            SELECT
              user_id,
              username,
              full_name,
              email,
              role,
              created_at
            FROM users
            WHERE full_name LIKE ?
               OR email LIKE ?
               OR phone_number LIKE ?
            ORDER BY user_id ASC
            LIMIT ? OFFSET ?
          `;

          queryParams = [likeTerm, likeTerm, likeTerm];
        }

        // Execute count query
        const [countRows]: any[] = await pool.query(
          countQuery,
          queryParams
        );
        const totalItems = Number(countRows[0]?.total ?? 0);
        const totalPages = Math.ceil(totalItems / limit);

        // Execute data query
        const [dataRows]: any[] = await pool.query(
          dataQuery,
          useFullText ? [queryParams[0], queryParams[0], limit, offset] : [...queryParams, limit, offset]
        );

        // Remove relevance field from results if present
        const results = dataRows.map((row: any) => {
          const { relevance, ...rest } = row;
          return rest;
        });

        return c.json({
          data: results,
          pagination: {
            currentPage: page,
            pageSize: limit,
            totalItems,
            totalPages,
          },
          searchMethod: useFullText ? 'fulltext' : 'like', // For debugging
        });
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to search users");
      }
    },
  );

  // Create an assistant and assign them to classes
  usersApi.post(
    "/api/v1/assistants",
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
        return handleUnexpectedError(c, error, "Failed to create assistant");
      } finally {
        connection.release();
      }
    },
  );

  // Admin: Create a new teacher user
  usersApi.post(
    "/api/v1/users/teachers",
    jwt({ secret: JWT_SECRET }),
    zValidator("json", createTeacherSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const { username, full_name, email, phone, password } = c.req.valid("json");
        const teacherEmail = email ?? `${username}@${TEACHER_EMAIL_DOMAIN}`;

        // Use provided password or default
        let password_hash: string;
        if (password) {
          password_hash = await Bun.password.hash(password, {
            algorithm: "bcrypt",
            cost: 10,
          });
        } else {
          password_hash = DEFAULT_PASSWORD_HASH;
        }

        const [result]: any = await pool.query(
          "INSERT INTO users (username, password_hash, full_name, email, phone_number, role) VALUES (?, ?, ?, ?, ?, 'Teacher')",
          [username, password_hash, full_name, teacherEmail, phone || null],
        );

        const newTeacherId = result.insertId;

        const newUser = {
          user_id: newTeacherId,
          username,
          full_name,
          email: teacherEmail,
          phone_number: phone || null,
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
          if (error.message.includes("users.phone_number")) {
            return c.json({ error: "Phone number already exists" }, 409);
          }
        }
        return handleUnexpectedError(c, error, "Failed to create teacher");
      }
    },
  );

  // Admin: Batch create teachers via CSV
  usersApi.post(
    "/api/v1/users/teachers/batch",
    jwt({ secret: JWT_SECRET }),
    async (c) => {
      let tempCsvPath: string | null = null;
      try {
        const payload = c.get("jwtPayload");
        if (payload.role !== "Admin") {
          return c.json({ error: "Forbidden" }, 403);
        }

        const body = await c.req.parseBody();
        const file = body["file"];
        const password = body["password"];
        let normalizedPassword: string | null = null;

        if (!(file instanceof File)) {
          return c.json({ error: "CSV 文件缺失或类型错误" }, 400);
        }

        // Validate password if provided
        if (password !== undefined && password !== null && password !== "") {
          if (typeof password !== "string") {
            return c.json({ error: "密码必须是字符串类型" }, 400);
          }
          const trimmedPassword = password.trim();
          if (trimmedPassword.length < 6) {
            return c.json({ error: "统一密码必须至少6位" }, 400);
          }
          normalizedPassword = trimmedPassword;
        }

        if (
          file.type &&
          file.type !== "text/csv" &&
          !file.type.includes("csv")
        ) {
          return c.json({ error: "仅支持上传 CSV 文件" }, 400);
        }

        const maxSizeBytes = 5 * 1024 * 1024;
        if (file.size > maxSizeBytes) {
          return c.json({ error: "CSV 文件大小不能超过 5MB" }, 413);
        }

        const tempDir = path.resolve(import.meta.dir, "../../uploads", "temp");
        await mkdir(tempDir, { recursive: true });
        tempCsvPath = path.join(
          tempDir,
          `teachers-${Date.now()}-${file.name.replace(/[\/\\?%*:|"<>]/g, "-")}`,
        );
        await Bun.write(tempCsvPath, file);

        const teacherRows: Array<{
          row: number;
          fullName: string;
          phone: string;
          username: string;
          email: string;
        }> = [];
        const invalidRows: Array<{ row: number; reason: string }> = [];
        const seenPhones = new Set<string>();
        let rowNumber = 0;

        await new Promise<void>((resolve, reject) => {
          createReadStream(tempCsvPath!)
            .pipe(
              csv({
                mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
              }),
            )
            .on("data", (row: Record<string, string>) => {
              rowNumber += 1;
              const fullName = row["姓名"]?.trim() ?? "";
              const phone = row["手机号"]?.trim() ?? "";

              if (!fullName || !phone) {
                invalidRows.push({ row: rowNumber, reason: "姓名或手机号缺失" });
                return;
              }

              if (fullName.length < MIN_TEACHER_NAME_LENGTH) {
                invalidRows.push({
                  row: rowNumber,
                  reason: `姓名长度需至少 ${MIN_TEACHER_NAME_LENGTH} 个字符`,
                });
                return;
              }

              if (!PHONE_REGEX.test(phone)) {
                invalidRows.push({
                  row: rowNumber,
                  reason: `手机号 ${phone} 格式不正确，应为 6-15 位数字`,
                });
                return;
              }

              if (seenPhones.has(phone)) {
                invalidRows.push({
                  row: rowNumber,
                  reason: `CSV 内存在重复手机号 ${phone}`,
                });
                return;
              }

              seenPhones.add(phone);
              const username = `prof_${phone}`;
              const email = `prof_${phone}@${TEACHER_EMAIL_DOMAIN}`;

              teacherRows.push({ row: rowNumber, fullName, phone, username, email });
            })
            .on("end", resolve)
            .on("error", reject);
        });

        if (invalidRows.length > 0) {
          return c.json(
            {
              error: "CSV 文件包含无效记录",
              invalid_rows: invalidRows.slice(0, 10),
            },
            400,
          );
        }

        if (teacherRows.length === 0) {
          return c.json({ error: "CSV 文件未包含有效的教师记录" }, 400);
        }

        const connection = await pool.getConnection();
        try {
          await connection.beginTransaction();

          const usernames = Array.from(
            new Set(teacherRows.map((row) => row.username)),
          );
          const emails = Array.from(
            new Set(teacherRows.map((row) => row.email)),
          );
          const phones = Array.from(
            new Set(teacherRows.map((row) => row.phone)),
          );

          // Optimized conflict detection using UNION ALL for better index usage
          const queries: string[] = [];
          const allParams: any[] = [];

          if (usernames.length > 0) {
            queries.push(
              `SELECT DISTINCT username, phone_number, email FROM users WHERE username IN (${usernames.map(() => "?").join(", ")})`
            );
            allParams.push(...usernames);
          }

          if (phones.length > 0) {
            queries.push(
              `SELECT DISTINCT username, phone_number, email FROM users WHERE phone_number IN (${phones.map(() => "?").join(", ")})`
            );
            allParams.push(...phones);
          }

          if (emails.length > 0) {
            queries.push(
              `SELECT DISTINCT username, phone_number, email FROM users WHERE email IN (${emails.map(() => "?").join(", ")})`
            );
            allParams.push(...emails);
          }

          if (queries.length > 0) {
            const unionQuery = queries.join(" UNION ");
            const [existing]: any[] = await connection.query(unionQuery, allParams);

            if (existing.length > 0) {
              await connection.rollback();
              return c.json(
                {
                  error: "部分教师信息已存在，批量创建已取消",
                  conflicts: existing.slice(0, 10),
                },
                409,
              );
            }
          }

          // Hash passwords: use provided custom password for all, otherwise default to each teacher's phone number
          let hashedPasswords: string[];
          if (normalizedPassword) {
            const hashed = await Bun.password.hash(normalizedPassword, {
              algorithm: "bcrypt",
              cost: 10,
            });
            hashedPasswords = teacherRows.map(() => hashed);
          } else {
            hashedPasswords = await Promise.all(
              teacherRows.map((row) =>
                Bun.password.hash(row.phone, {
                  algorithm: "bcrypt",
                  cost: 10,
                }),
              ),
            );
          }

          // Batch insert all teachers at once for better performance
          const teacherValues = teacherRows.map((row, index) => [
            row.username,
            hashedPasswords[index],
            row.fullName,
            row.email,
            "Teacher",
            row.phone,
          ]);

          const [result]: any = await connection.query(
            "INSERT INTO users (username, password_hash, full_name, email, role, phone_number) VALUES ?",
            [teacherValues],
          );

          // Calculate user IDs: MySQL returns the first insertId, subsequent IDs are consecutive
          const firstInsertId = result.insertId;
          const createdTeachers = teacherRows.map((row, index) => ({
            user_id: firstInsertId + index,
            username: row.username,
            full_name: row.fullName,
            phone_number: row.phone,
            email: row.email,
          }));

          await connection.commit();

          return c.json(
            {
              created: createdTeachers.length,
              teachers: createdTeachers,
            },
            201,
          );
        } catch (error) {
          await connection.rollback();
          return handleUnexpectedError(c, error, "Failed to batch create teachers");
        } finally {
          connection.release();
        }
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to batch create teachers");
      } finally {
        if (tempCsvPath) {
          await unlink(tempCsvPath).catch((err) =>
            logWarning("Failed to cleanup temp teacher CSV", {
              error: err instanceof Error ? err.message : String(err),
            }),
          );
        }
      }
    },
  );

  // Get all assistants associated with a specific teacher through their classes
  usersApi.get(
    "/api/v1/teachers/:teacherId/assistants",
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
        return handleUnexpectedError(c, error, "Failed to fetch assistants for teacher");
      }
    },
  );

  // Get all users (Admin only)
  usersApi.get(
    "/api/v1/users",
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
        return handleUnexpectedError(c, error, "Failed to fetch users");
      }
    },
  );

  // Get all assistants (Admin and Teacher only)
  usersApi.get("/api/v1/assistants", jwt({ secret: JWT_SECRET }), async (c) => {
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
      return handleUnexpectedError(c, error, "Failed to fetch assistants");
    }
  });

  // Get students not enrolled in any class (for Teacher/Assistant)
  usersApi.get("/api/v1/students/unenrolled", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");

      if (
        payload.role !== "Admin" &&
        payload.role !== "Teacher" &&
        payload.role !== "Assistant"
      ) {
        return c.json({ error: "Forbidden" }, 403);
      }

      const [rows] = await pool.query(
        `SELECT u.user_id, u.username, u.full_name, u.email, u.phone_number, u.created_at
           FROM users u
          WHERE u.role = 'Student'
            AND NOT EXISTS (
              SELECT 1 FROM enrollments e WHERE e.student_id = u.user_id
            )
          ORDER BY u.created_at DESC`
      );

      return c.json(rows);
    } catch (error) {
      return handleUnexpectedError(c, error, "Failed to fetch unenrolled students");
    }
  });

  // Get current user's info
  usersApi.get("/api/v1/users/me", jwt({ secret: JWT_SECRET }), async (c) => {
    try {
      const payload = c.get("jwtPayload");
      const userId = payload.sub;

      const [rows]: any[] = await pool.query(
        `SELECT
           user_id,
           username,
           full_name,
           email,
           phone_number,
           role,
           created_at,
           password_hash
         FROM users
        WHERE user_id = ?`,
        [userId],
      );
      const user = rows[0];

      if (!user) {
        return c.json({ error: "User not found" }, 404);
      }

      const passwordHash: string | null = user.password_hash ?? null;
      delete user.password_hash;

      const shouldAuditPassword =
        user.role === "Student" || user.role === "Teacher";
      if (shouldAuditPassword) {
        let mustChangePassword = false;
        if (passwordHash) {
          const matchesDefault = passwordHash === DEFAULT_PASSWORD_HASH;
          let matchesIdentifier = false;

          if (user.role === "Student") {
            matchesIdentifier = await Bun.password.verify(
              user.username,
              passwordHash,
            ).catch(() => false);
          } else if (user.role === "Teacher" && user.phone_number) {
            matchesIdentifier = await Bun.password.verify(
              user.phone_number,
              passwordHash,
            ).catch(() => false);
          }

          mustChangePassword = Boolean(matchesDefault || matchesIdentifier);
        }
        user.must_change_password = mustChangePassword;
      }

      if (user.role === "Student") {
        // Fetch all classes the student is enrolled in
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
         ORDER BY e.enrollment_id ASC`,
          [userId],
        );
        // Return all classes for full transparency
        user.classes = classRows;
        // For backward compatibility, also return the first class as user.class
        user.class = classRows[0] ?? null;
      } else if (!shouldAuditPassword) {
        delete user.must_change_password;
      }

      return c.json(user);
    } catch (error) {
      return handleUnexpectedError(c, error, "Failed to fetch current user");
    }
  });

  // Update current user's info
  usersApi.put(
    "/api/v1/users/me",
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
          `SELECT
             user_id,
             username,
             full_name,
             email,
             phone_number,
             role,
             created_at
           FROM users
          WHERE user_id = ?`,
          [userId],
        );
        return c.json(rows[0]);
      } catch (error: any) {
        if (error.code === "ER_DUP_ENTRY") {
          return c.json({ error: "Email already exists" }, 409);
        }
        return handleUnexpectedError(c, error, "Failed to update user");
      }
    },
  );

  // Change current user's password
  usersApi.put(
    "/api/v1/users/me/password",
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
        return handleUnexpectedError(c, error, "Failed to change password");
      }
    },
  );

  // Get user by ID
  usersApi.get(
    "/api/v1/users/:userId",
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
          `SELECT
             user_id,
             username,
             full_name,
             email,
             role,
             created_at
           FROM users
          WHERE user_id = ?`,
          [targetUserId],
        );
        const user = rows[0];

      if (!user) {
        return c.json({ error: "User not found" }, 404);
      }
      return c.json(user);
    } catch (error) {
      return handleUnexpectedError(c, error, "Failed to fetch user");
    }
  },
);

  // Admin: Force change user's password
  usersApi.put(
    "/api/v1/users/:userId/password",
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
        return handleUnexpectedError(c, error, "Failed to force change password");
      }
    },
  );

  // Teacher/Admin/Assistant: Reset a student's password
  usersApi.post(
    "/api/v1/users/:userId/reset-password",
    jwt({ secret: JWT_SECRET }),
    zValidator("param", userIdParamSchema),
    zValidator("json", resetPasswordSchema),
    async (c) => {
      try {
        const payload = c.get("jwtPayload");
        const { userId } = c.req.valid("param");
        const targetUserId = parseInt(userId, 10);
        const { newPassword } = c.req.valid("json");

        if (
          payload.role !== "Admin" &&
          payload.role !== "Teacher" &&
          payload.role !== "Assistant"
        ) {
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
          const teacherId = Number(payload.sub);
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
        } else if (payload.role === "Assistant") {
          const assistantId = Number(payload.sub);
          const [assignmentRows]: any[] = await pool.query(
            `SELECT 1
               FROM enrollments e
               JOIN class_assistants ca ON e.class_id = ca.class_id
              WHERE e.student_id = ? AND ca.assistant_id = ?
              LIMIT 1`,
            [targetUserId, assistantId],
          );

          if (assignmentRows.length === 0) {
            return c.json({
              error: "Forbidden: Student is not assigned to one of your classes",
            }, 403);
          }
        }
        // Admins are allowed to proceed

        // 3. Determine password hash: custom or default
        let password_hash: string;
        if (newPassword) {
          // Use custom password
          password_hash = await Bun.password.hash(newPassword, {
            algorithm: "bcrypt",
            cost: 10,
          });
        } else {
          // Use default password
          password_hash = DEFAULT_PASSWORD_HASH;
        }

        // 4. Update password
        const [result]: any = await pool.query(
          "UPDATE users SET password_hash = ? WHERE user_id = ?",
          [password_hash, targetUserId],
        );

        if (result.affectedRows === 0) {
          // This case should be rare given the check above, but good for safety
          return c.json({ error: "User not found" }, 404);
        }

        const message = newPassword
          ? "Password updated successfully"
          : "Password reset to default successfully";

        return c.json({ message });
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to reset password");
      }
    },
  );

  // Delete user by ID (Admin only)
  usersApi.delete(
    "/api/v1/users/:userId",
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
          return handleUnexpectedError(c, error, "Failed to delete user");
        } finally {
          connection.release();
        }
      } catch (error) {
        return handleUnexpectedError(c, error, "Failed to delete user");
      }
    },
  );

  return usersApi;
};
