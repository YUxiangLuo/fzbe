import { describe, test, expect } from "bun:test";
import { loginSchema } from "../../../src/modules/auth/auth.schema";

describe("Auth Schemas", () => {
  describe("loginSchema", () => {
    test("should accept valid login credentials", () => {
      const result = loginSchema.safeParse({
        username: "testuser",
        password: "password123",
      });
      expect(result.success).toBe(true);
    });

    test("should reject missing username", () => {
      const result = loginSchema.safeParse({
        password: "password123",
      });
      expect(result.success).toBe(false);
    });

    test("should reject missing password", () => {
      const result = loginSchema.safeParse({
        username: "testuser",
      });
      expect(result.success).toBe(false);
    });

    test("should reject empty username", () => {
      const result = loginSchema.safeParse({
        username: "",
        password: "password123",
      });
      expect(result.success).toBe(false);
    });

    test("should reject empty password", () => {
      const result = loginSchema.safeParse({
        username: "testuser",
        password: "",
      });
      expect(result.success).toBe(false);
    });

    test("should accept username with special characters", () => {
      const result = loginSchema.safeParse({
        username: "user@email.com",
        password: "password123",
      });
      expect(result.success).toBe(true);
    });

    test("should accept numeric username (student ID)", () => {
      const result = loginSchema.safeParse({
        username: "20240001",
        password: "password123",
      });
      expect(result.success).toBe(true);
    });
  });
});
