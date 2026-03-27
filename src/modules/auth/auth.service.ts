import { sign } from "hono/jwt";
import type { RowDataPacket } from "mysql2";
import { pool } from "../../config";
import { env } from "../../config";
import { verifyPassword } from "../../utils/password";
import { AUTH_CONSTANTS } from "../../utils/constants";
import type { User, UserRole } from "../../types";
import type { LoginInput } from "./auth.schema";

type UserRow = User & RowDataPacket;

export interface LoginResult {
  success: true;
  token: string;
  user: { role: UserRole };
}

export interface LoginError {
  success: false;
  error: string;
  message?: string;
  status: 401 | 403;
}

const ROLE_NAMES: Record<string, string> = {
  Student: "学生",
  Teacher: "教师",
  Assistant: "助教",
  Admin: "管理员",
};

export async function login(input: LoginInput): Promise<LoginResult | LoginError> {
  const { username, password, role } = input;

  const [rows] = await pool.query<UserRow[]>(
    "SELECT * FROM users WHERE username = ?",
    [username],
  );
  const user = rows[0];

  if (!user) {
    return { success: false, error: "用户名或密码错误", status: 401 };
  }

  const passwordMatch = await verifyPassword(password, user.password_hash);
  if (!passwordMatch) {
    return { success: false, error: "用户名或密码错误", status: 401 };
  }

  if (role && user.role !== role) {
    return {
      success: false,
      error: "角色不匹配",
      message: `该账号的角色是${ROLE_NAMES[user.role] || user.role}，但您正在尝试以${ROLE_NAMES[role] || role}身份登录`,
      status: 403,
    };
  }

  const payload = {
    sub: user.user_id,
    username: user.username,
    role: user.role,
    exp: Math.floor(Date.now() / 1000) + AUTH_CONSTANTS.JWT_EXPIRATION_SECONDS,
  };

  const token = await sign(payload, env.JWT_SECRET);
  return { success: true, token, user: { role: user.role } };
}
