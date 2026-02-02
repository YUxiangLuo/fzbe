import type { UserRole } from "./entities";

export interface JwtPayload {
  sub: number;
  username: string;
  role: UserRole;
  exp: number;
}
