/**
 * 密码相关工具函数
 * 统一密码哈希逻辑，避免代码重复
 */

import { USER_CONSTANTS } from "./constants";

const { BCRYPT_COST } = USER_CONSTANTS;

/**
 * 对密码进行哈希
 * @param password 明文密码
 * @returns 哈希后的密码
 */
export async function hashPassword(password: string): Promise<string> {
  return Bun.password.hash(password, {
    algorithm: "bcrypt",
    cost: BCRYPT_COST,
  });
}

/**
 * 验证密码是否匹配
 * @param password 明文密码
 * @param hash 哈希值
 * @returns 是否匹配
 */
export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return Bun.password.verify(password, hash);
}

/**
 * 检查密码是否与标识符相同
 * 用于判断用户是否需要修改密码
 * @param passwordHash 当前密码哈希
 * @param identifier 用户标识符 (学生的学号，教师/助教的手机号)
 * @returns 是否需要修改密码
 */
export async function shouldChangePassword(
  passwordHash: string,
  identifier: string | null,
): Promise<boolean> {
  // 检查密码是否与标识符相同
  if (identifier) {
    try {
      const matchesIdentifier = await verifyPassword(identifier, passwordHash);
      if (matchesIdentifier) {
        return true;
      }
    } catch {
      // 验证失败时忽略
    }
  }

  return false;
}
