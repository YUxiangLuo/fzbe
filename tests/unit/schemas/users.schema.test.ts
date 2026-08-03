import { describe, test, expect } from "bun:test";
import {
  registerSchema,
  createTeacherSchema,
  createAssistantSingleSchema,
  createAssistantSchema,
  updateUserSchema,
  changePasswordSchema,
  studentSelfRegisterSchema,
  teacherAssistantSelfRegisterSchema,
  userSearchQuerySchema,
  bulkDeleteUsersSchema,
  paginationQuerySchema,
  roleParamSchema,
} from "../../../src/modules/users/users.schema";

describe("User Schemas", () => {
  describe("studentSelfRegisterSchema", () => {
    test("should accept valid student data", () => {
      const validData = {
        username: "12345678",
        password: "password123",
        name: "张三",
        email: "test@example.com",
      };
      const result = studentSelfRegisterSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    test("should reject username with less than 8 digits", () => {
      const result = studentSelfRegisterSchema.safeParse({
        username: "1234567",
        password: "password123",
        name: "Test",
        email: "test@example.com",
      });
      expect(result.success).toBe(false);
    });

    test("should reject non-numeric username", () => {
      const result = studentSelfRegisterSchema.safeParse({
        username: "student01",
        password: "password123",
        name: "Test",
        email: "test@example.com",
      });
      expect(result.success).toBe(false);
    });

    test("should reject password shorter than 6 characters", () => {
      const result = studentSelfRegisterSchema.safeParse({
        username: "12345678",
        password: "12345",
        name: "Test",
        email: "test@example.com",
      });
      expect(result.success).toBe(false);
    });

    test("should accept Chinese name", () => {
      const result = studentSelfRegisterSchema.safeParse({
        username: "12345678",
        password: "password123",
        name: "李四",
        email: "test@example.com",
      });
      expect(result.success).toBe(true);
    });

    test("should reject name with special characters", () => {
      const result = studentSelfRegisterSchema.safeParse({
        username: "12345678",
        password: "password123",
        name: "Test@User",
        email: "test@example.com",
      });
      expect(result.success).toBe(false);
    });

    test("should reject invalid email format", () => {
      const result = studentSelfRegisterSchema.safeParse({
        username: "12345678",
        password: "password123",
        name: "Test",
        email: "invalid-email",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("teacherAssistantSelfRegisterSchema", () => {
    test("should accept valid teacher/assistant data", () => {
      const validData = {
        username: "teacher01",
        password: "password123",
        name: "王老师",
        email: "teacher@example.com",
        phone: "13800138000",
      };
      const result = teacherAssistantSelfRegisterSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    test("should reject username shorter than 3 characters", () => {
      const result = teacherAssistantSelfRegisterSchema.safeParse({
        username: "ab",
        password: "password123",
        name: "Test",
        email: "test@example.com",
        phone: "13800138000",
      });
      expect(result.success).toBe(false);
    });

    test("should reject username with special characters", () => {
      const result = teacherAssistantSelfRegisterSchema.safeParse({
        username: "user@name",
        password: "password123",
        name: "Test",
        email: "test@example.com",
        phone: "13800138000",
      });
      expect(result.success).toBe(false);
    });

    test("should require exactly 11 digit phone", () => {
      const result = teacherAssistantSelfRegisterSchema.safeParse({
        username: "teacher01",
        password: "password123",
        name: "Test",
        email: "test@example.com",
        phone: "1380013800", // 10 digits
      });
      expect(result.success).toBe(false);
    });

    test("should reject phone with non-digits", () => {
      const result = teacherAssistantSelfRegisterSchema.safeParse({
        username: "teacher01",
        password: "password123",
        name: "Test",
        email: "test@example.com",
        phone: "138-0013-800",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("createTeacherSchema", () => {
    test("should accept minimal valid data", () => {
      const result = createTeacherSchema.safeParse({
        username: "prof_zhang",
        full_name: "Professor Zhang",
      });
      expect(result.success).toBe(true);
    });

    test("should accept full data with optional fields", () => {
      const result = createTeacherSchema.safeParse({
        username: "prof_zhang",
        full_name: "Professor Zhang",
        email: "zhang@university.edu",
        phone: "13800138000",
        password: "secret123",
      });
      expect(result.success).toBe(true);
    });

    test("should validate phone format when provided", () => {
      const result = createTeacherSchema.safeParse({
        username: "prof_zhang",
        full_name: "Professor Zhang",
        phone: "1234567890", // invalid: 10 digits, should be 11 starting with 1
      });
      expect(result.success).toBe(false);
    });

    test("should normalize an optional blank phone instead of persisting whitespace", () => {
      const result = createTeacherSchema.safeParse({
        username: "prof_blank",
        full_name: "Professor Blank",
        phone: "   ",
      });

      expect(result.success).toBe(true);
      if (result.success) expect(result.data.phone).toBeUndefined();
    });
  });

  describe("createAssistantSchema", () => {
    test("should trim valid phone numbers and normalize blanks to undefined", () => {
      const base = {
        username: "assistant01",
        password: "password123",
        full_name: "Assistant",
        email: "assistant01@example.com",
        class_ids: [1],
      };

      const trimmed = createAssistantSchema.safeParse({ ...base, phone_number: " 13800138000 " });
      const blank = createAssistantSchema.safeParse({ ...base, phone_number: "   " });

      expect(trimmed.success).toBe(true);
      if (trimmed.success) expect(trimmed.data.phone_number).toBe("13800138000");
      expect(blank.success).toBe(true);
      if (blank.success) expect(blank.data.phone_number).toBeUndefined();
    });

    test("should reject non-numeric phone numbers", () => {
      const result = createAssistantSchema.safeParse({
        username: "assistant01",
        password: "password123",
        full_name: "Assistant",
        email: "assistant01@example.com",
        phone_number: "not-a-phone",
        class_ids: [1],
      });

      expect(result.success).toBe(false);
    });
  });

  describe("updateUserSchema", () => {
    test("should accept at least one field", () => {
      const result = updateUserSchema.safeParse({ full_name: "New Name" });
      expect(result.success).toBe(true);
    });

    test("should reject empty object", () => {
      const result = updateUserSchema.safeParse({});
      expect(result.success).toBe(false);
    });

    test("should validate email format when provided", () => {
      const result = updateUserSchema.safeParse({ email: "invalid" });
      expect(result.success).toBe(false);
    });

    test("should accept multiple fields", () => {
      const result = updateUserSchema.safeParse({
        full_name: "New Name",
        email: "new@example.com",
        phone_number: "13800138000",
      });
      expect(result.success).toBe(true);
    });

    test("should turn a blank phone into an explicit clear operation", () => {
      const result = updateUserSchema.safeParse({ phone_number: "   " });

      expect(result.success).toBe(true);
      if (result.success) expect(result.data.phone_number).toBeNull();
    });

    test("should accept null to clear a phone number", () => {
      const result = updateUserSchema.safeParse({ phone_number: null });

      expect(result.success).toBe(true);
      if (result.success) expect(result.data.phone_number).toBeNull();
    });
  });

  describe("changePasswordSchema", () => {
    test("should accept valid password change", () => {
      const result = changePasswordSchema.safeParse({
        currentPassword: "oldpass123",
        newPassword: "newpass123",
      });
      expect(result.success).toBe(true);
    });

    test("should reject empty current password", () => {
      const result = changePasswordSchema.safeParse({
        currentPassword: "",
        newPassword: "newpass123",
      });
      expect(result.success).toBe(false);
    });

    test("should reject new password shorter than 6 characters", () => {
      const result = changePasswordSchema.safeParse({
        currentPassword: "oldpass123",
        newPassword: "12345",
      });
      expect(result.success).toBe(false);
    });
  });

  describe("userSearchQuerySchema", () => {
    test("should provide default values for page and limit", () => {
      const result = userSearchQuerySchema.safeParse({ q: "test" });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.page).toBe(1);
        expect(result.data.limit).toBe(10);
      }
    });

    test("should coerce string numbers", () => {
      const result = userSearchQuerySchema.safeParse({
        q: "test",
        page: "2",
        limit: "20",
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.page).toBe(2);
        expect(result.data.limit).toBe(20);
      }
    });

    test("should reject empty search query", () => {
      const result = userSearchQuerySchema.safeParse({ q: "" });
      expect(result.success).toBe(false);
    });

    test("should reject limit over 100", () => {
      const result = userSearchQuerySchema.safeParse({
        q: "test",
        limit: 101,
      });
      expect(result.success).toBe(false);
    });
  });

  describe("bulkDeleteUsersSchema", () => {
    test("should accept distinct positive user IDs", () => {
      const result = bulkDeleteUsersSchema.safeParse({ userIds: [1, 2, 3] });
      expect(result.success).toBe(true);
    });

    test("should reject duplicate user IDs", () => {
      const result = bulkDeleteUsersSchema.safeParse({ userIds: [1, 2, 2] });
      expect(result.success).toBe(false);
    });

    test("should reject empty selections", () => {
      const result = bulkDeleteUsersSchema.safeParse({ userIds: [] });
      expect(result.success).toBe(false);
    });
  });

  describe("paginationQuerySchema", () => {
    test("should provide default values", () => {
      const result = paginationQuerySchema.safeParse({});
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.page).toBe(1);
        expect(result.data.limit).toBe(10);
      }
    });

    test("should reject page less than 1", () => {
      const result = paginationQuerySchema.safeParse({ page: 0 });
      expect(result.success).toBe(false);
    });
  });

  describe("roleParamSchema", () => {
    test("should allow Student self-registration role", () => {
      const result = roleParamSchema.safeParse({ role: "Student" });
      expect(result.success).toBe(true);
    });

    test("should reject Teacher self-registration role", () => {
      const result = roleParamSchema.safeParse({ role: "Teacher" });
      expect(result.success).toBe(false);
    });

    test("should reject Assistant self-registration role", () => {
      const result = roleParamSchema.safeParse({ role: "Assistant" });
      expect(result.success).toBe(false);
    });
  });
});
