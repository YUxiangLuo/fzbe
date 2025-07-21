import { Hono } from "hono";
import type { Context, Next } from "hono";
import { HTTPException } from "hono/http-exception";
import { jwt } from "hono/jwt";
import { sign } from "hono/jwt";
import mysql from "mysql2/promise";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";

// --- 自定义日志中间件 ---
const timestampedLogger = async (c: Context, next: Next) => {
  const start = Date.now();
  await next();
  const duration = Date.now() - start;
  const timestamp = new Date().toISOString();
  // 增强日志输出，使其更具信息量
  console.log(
    `[${timestamp}] ${c.req.method} ${c.req.path} | ${c.res.status} | ${duration}ms`,
  );
};

// --- 配置与数据库连接 ---
const {
  DB_HOST,
  DB_USER,
  DB_PASSWORD,
  DB_DATABASE,
  JWT_SECRET, // 用于 JWT 签名的密钥
  PORT,
} = process.env;

if (!JWT_SECRET) {
  throw new Error("JWT_SECRET environment variable not set!");
}

const pool = mysql.createPool({
  host: DB_HOST,
  user: DB_USER,
  password: DB_PASSWORD,
  database: DB_DATABASE,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

console.log("Database connection pool created.");

// --- Zod Schemas for Validation ---

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

const createClassSchema = z.object({
  class_name: z.string().min(1),
  class_code: z.string().optional(),
});

const enrollSchema = z.object({
  class_code: z.string().min(1, { message: "Class code is required" }),
});

// --- Hono 应用 ---
const app = new Hono();

// --- 中间件 ---
app.use("*", timestampedLogger);

// --- 路由 ---

// 公开路由：登录
app.post("/api/auth/login", zValidator("json", loginSchema), async (c) => {
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
      exp: Math.floor(Date.now() / 1000) + 60 * 60 * 24, // 24 小时
    };

    const token = await sign(payload, JWT_SECRET);
    return c.json({ token });
  } catch (error) {
    console.error("Login error:", error);
    throw error;
  }
});

// --- 用户管理 API ---

// 受保护的路由示例：获取所有用户
app.get("/api/users", jwt({ secret: JWT_SECRET }), async (c) => {
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

// 公开路由: 注册新用户
app.post(
  "/api/users/register",
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

// 受保护的路由: 获取当前登录用户的信息
app.get("/api/users/me", jwt({ secret: JWT_SECRET }), async (c) => {
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

// 受保护的路由: 更新当前用户信息
app.put(
  "/api/users/me",
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

// 受保护的路由: 获取指定 ID 的用户信息
app.get(
  "/api/users/:id",
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

// --- 班级与学生管理 API ---

// 受保护的路由: 创建新班级
app.post(
  "/api/classes",
  jwt({ secret: JWT_SECRET }),
  zValidator("json", createClassSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Teacher" && payload.role !== "Admin") {
        return c.json({ error: "Forbidden" }, 403);
      }

      const { class_name, class_code } = c.req.valid("json");
      const teacher_id = payload.sub;

      const [result]: any = await pool.query(
        "INSERT INTO classes (class_name, class_code, teacher_id) VALUES (?, ?, ?)",
        [class_name, class_code || null, teacher_id],
      );

      const newClass = {
        class_id: result.insertId,
        class_name,
        class_code,
        teacher_id,
      };
      return c.json(newClass, 201);
    } catch (error: any) {
      if (error.code === "ER_DUP_ENTRY") {
        return c.json({ error: "Class code already exists" }, 409);
      }
      console.error("Error creating class:", error);
      throw error;
    }
  },
);

// 受保护的路由: 学生使用班级代码加入班级
app.post(
  "/api/classes/enroll",
  jwt({ secret: JWT_SECRET }),
  zValidator("json", enrollSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload");
      if (payload.role !== "Student") {
        return c.json({ error: "Only students can enroll in classes" }, 403);
      }

      const { class_code } = c.req.valid("json");
      const [classRows]: any[] = await pool.query(
        "SELECT class_id FROM classes WHERE class_code = ?",
        [class_code],
      );
      const targetClass = classRows[0];

      if (!targetClass) {
        return c.json({ error: "Class not found" }, 404);
      }

      const student_id = payload.sub;
      await pool.query(
        "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
        [student_id, targetClass.class_id],
      );

      return c.json({ message: "Enrolled successfully" }, 200);
    } catch (error: any) {
      if (error.code === "ER_DUP_ENTRY") {
        return c.json({ error: "Already enrolled in this class" }, 409);
      }
      console.error("Error enrolling in class:", error);
      throw error;
    }
  },
);

// 受保护的路由: 获取班级列表
app.get("/api/classes", jwt({ secret: JWT_SECRET }), async (c) => {
  try {
    const payload = c.get("jwtPayload");
    let query = "";
    const params: (string | number)[] = [payload.sub];

    switch (payload.role) {
      case "Admin":
        query =
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id";
        params.pop();
        break;
      case "Teacher":
        query =
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.teacher_id = ?";
        break;
      case "Student":
        query =
          "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN enrollments e ON c.class_id = e.class_id JOIN users u ON c.teacher_id = u.user_id WHERE e.student_id = ?";
        break;
      default:
        return c.json({ error: "Forbidden" }, 403);
    }

    const [rows] = await pool.query(query, params);
    return c.json(rows);
  } catch (error) {
    console.error("Error fetching classes:", error);
    throw error;
  }
});

// 受保护的路由: 获取单个班级详情 (包括学生列表)
app.get(
  "/api/classes/:id",
  jwt({ secret: JWT_SECRET }),
  zValidator("param", paramsSchema),
  async (c) => {
    try {
      const payload = c.get("jwtPayload");
      const { id } = c.req.valid("param");
      const classId = parseInt(id, 10);

      const [classRows]: any[] = await pool.query(
        "SELECT c.*, u.full_name as teacher_name FROM classes c JOIN users u ON c.teacher_id = u.user_id WHERE c.class_id = ?",
        [classId],
      );
      const classInfo = classRows[0];

      if (!classInfo) {
        return c.json({ error: "Class not found" }, 404);
      }

      let isAllowed = false;
      if (payload.role === "Admin" || payload.sub === classInfo.teacher_id) {
        isAllowed = true;
      } else if (payload.role === "Student") {
        const [enrollmentRows]: any[] = await pool.query(
          "SELECT enrollment_id FROM enrollments WHERE student_id = ? AND class_id = ?",
          [payload.sub, classId],
        );
        if (enrollmentRows.length > 0) {
          isAllowed = true;
        }
      }

      if (!isAllowed) {
        return c.json({ error: "Forbidden" }, 403);
      }

      const [studentRows] = await pool.query(
        "SELECT u.user_id, u.username, u.full_name, u.email FROM users u JOIN enrollments e ON u.user_id = e.student_id WHERE e.class_id = ?",
        [classId],
      );

      const response = { ...classInfo, students: studentRows };
      return c.json(response);
    } catch (error) {
      console.error("Error fetching class details:", error);
      throw error;
    }
  },
);

// --- 404 与错误处理 ---
app.notFound((c) => c.json({ message: "Not Found" }, 404));

// The zValidator middleware handles validation errors automatically,
// so they won't reach the generic onError handler.
app.onError((err, c) => {
  console.error("Global error handler:", err);

  if (err instanceof HTTPException) {
    // Re-throw HTTPException to let Hono's default handler manage it
    return err.getResponse();
  }

  // Fallback for other unexpected errors
  return c.json({ error: "Internal Server Error" }, 500);
});

// --- 服务器启动 ---
const port = PORT ? parseInt(PORT, 10) : 3000;
console.log(`Server listening on http://localhost:${port}`);

export default {
  port,
  fetch: app.fetch,
};
