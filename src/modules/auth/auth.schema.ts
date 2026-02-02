import { z } from "zod";

export const loginSchema = z.object({
  username: z.string().min(1, { message: "用户名不能为空" }),
  password: z.string().min(1, { message: "密码不能为空" }),
  role: z.enum(["Student", "Teacher", "Assistant", "Admin"]).optional(),
});

export type LoginInput = z.infer<typeof loginSchema>;
