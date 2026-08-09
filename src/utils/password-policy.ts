import { USER_CONSTANTS } from "./constants";

export const PASSWORD_POLICY_MESSAGE =
  `密码至少${USER_CONSTANTS.MIN_PASSWORD_LENGTH}位，且必须同时包含字母和数字`;

export function isPasswordPolicyCompliant(password: string): boolean {
  return password.length >= USER_CONSTANTS.MIN_PASSWORD_LENGTH
    && /[A-Za-z]/.test(password)
    && /\d/.test(password);
}
