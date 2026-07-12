import { jwt } from "hono/jwt";
import { createMiddleware } from "hono/factory";
import type { Context, Next } from "hono";
import { env } from "../config";
import type { JwtPayload, UserRole } from "../types";
import { AUTH_CONSTANTS } from "../utils/constants";

export const jwtAuth = jwt({
  secret: env.JWT_SECRET,
  alg: AUTH_CONSTANTS.JWT_ALGORITHM,
});

export const requireRoles = (...allowedRoles: UserRole[]) => {
  return createMiddleware(async (c: Context, next: Next) => {
    const payload = c.get("jwtPayload") as JwtPayload | undefined;
    if (!payload) {
      return c.json({ error: "未授权" }, 401);
    }
    if (!allowedRoles.includes(payload.role)) {
      return c.json({ error: "权限不足" }, 403);
    }
    await next();
  });
};

export const requireAdmin = requireRoles("Admin");
export const requireTeacher = requireRoles("Teacher", "Admin");
export const requireTeacherOrAssistant = requireRoles("Teacher", "Assistant", "Admin");
export const requireStudent = requireRoles("Student");
