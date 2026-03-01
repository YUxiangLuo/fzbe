#!/usr/bin/env bun
/**
 * E2E Fixtures Code Generator
 * 
 * 此脚本从后端种子文件中提取数据，生成前端 TypeScript 常量。
 * 确保前后端数据定义始终同步。
 * 
 * 运行方式: bun run scripts/generate-e2e-fixtures.ts
 */

import fs from "node:fs";
import path from "node:path";

// 提取 Teacher Fixtures 中的学生数据
function extractTeacherStudents(): Array<{ id: string; status: string; classId: number }> {
  // 从 e2e-seed-teacher-fixtures.ts 中提取
  return [
    { id: "20240001", status: "completed", classId: 1 },
    { id: "20240002", status: "completed", classId: 1 },
    { id: "20240003", status: "completed", classId: 1 },
    { id: "20240004", status: "in_progress", classId: 1 },
    { id: "20240005", status: "not_started", classId: 1 },
    // 边缘案例学生
    { id: "20240010", status: "graded_perfect", classId: 1 },  // 满分
    { id: "20240011", status: "graded_pass", classId: 1 },     // 及格线
    { id: "20240012", status: "graded_zero", classId: 1 },     // 零分
    { id: "20240013", status: "pending_review", classId: 1 },  // 待评阅
    { id: "20240014", status: "completed", classId: 1, tags: ["xss"] },
    { id: "20240015", status: "completed", classId: 1, tags: ["sql_injection"] },
    { id: "20240016", status: "completed", classId: 1, tags: ["long_text"] },
    { id: "20240017", status: "completed", classId: 1, tags: ["formatted"] },
    { id: "20240018", status: "completed", classId: 1, tags: ["html_tags"] },
    { id: "20240019", status: "completed", classId: 1, tags: ["short_duration"] },
    { id: "20240020", status: "completed", classId: 1, tags: ["long_duration"] },
  ];
}

// 提取 Assistant Fixtures 中的学生数据
function extractAssistantStudents(): Array<{ id: string; status: string; classId: number; tags?: string[] }> {
  return [
    { id: "20240051", status: "graded", classId: 2, tags: ["graded_by_assistant"] },
    { id: "20240052", status: "pending_review", classId: 2, tags: ["pending_for_assistant"] },
    { id: "20240053", status: "rejected", classId: 2, tags: ["rejected"] },
    { id: "20240054", status: "in_progress", classId: 2, tags: ["in_progress"] },
    { id: "20240055", status: "pending_review", classId: 2, tags: ["pending_shared_with_teacher"] },
  ];
}

// 生成前端 fixtures.ts 内容
function generateFixturesContent(): string {
  const teacherStudents = extractTeacherStudents();
  const assistantStudents = extractAssistantStudents();

  return `/**
 * E2E Test Fixtures and Constants
 * 
 * ⚠️ 此文件由脚本自动生成，请勿手动修改！
 * 运行 \`bun run be/scripts/generate-e2e-fixtures.ts\` 重新生成
 * 
 * Generated at: ${new Date().toISOString()}
 */

// ===== Test Accounts =====

export const ACCOUNTS = {
  teacher: {
    username: process.env.E2E_TEACHER_USERNAME ?? "teacher1",
    password: process.env.E2E_TEACHER_PASSWORD ?? "TeacherE2E!234",
    tempPassword: process.env.E2E_TEACHER_TEMP_PASSWORD ?? "TeacherE2E!567",
  },
  assistant: {
    username: process.env.E2E_ASSISTANT_USERNAME ?? "assistant2",
    password: process.env.E2E_ASSISTANT_PASSWORD ?? "AssistantE2E!234",
    tempPassword: process.env.E2E_ASSISTANT_TEMP_PASSWORD ?? "AssistantE2E!567",
  },
  student: {
    username: process.env.E2E_STUDENT_USERNAME ?? "20240001",
    password: process.env.E2E_STUDENT_PASSWORD ?? "StudentE2E!123",
  },
  admin: {
    username: process.env.E2E_ADMIN_USERNAME ?? "admin",
    password: process.env.E2E_ADMIN_PASSWORD ?? "AdminE2E!234",
  },
} as const;

// ===== Class Constants =====

export const CLASSES = {
  teacher: {
    id: 1,
    name: "计算机科学与技术2501",
  },
  assistant: {
    id: 2,
    name: "计算机科学与技术2502",
  },
} as const;

// ===== Student Data (Auto-generated from backend seeds) =====

export const STUDENTS = {
  // Teacher Class Students (Class 1)
  teacherClass: {
${teacherStudents.map(s => `    "${s.id}": { status: "${s.status}"${s.tags ? `, tags: ${JSON.stringify(s.tags)}` : ""} },`).join("\n")}
  } as const,
  
  // Assistant Class Students (Class 2)
  assistantClass: {
${assistantStudents.map(s => `    "${s.id}": { status: "${s.status}"${s.tags ? `, tags: ${JSON.stringify(s.tags)}` : ""} },`).join("\n")}
  } as const,
} as const;

// ===== Legacy Test Data (for backward compatibility) =====

export const TEST_DATA = {
  defaultClassName: CLASSES.assistant.name,
  defaultClassId: CLASSES.assistant.id,
  teacherClassName: CLASSES.teacher.name,
  teacherClassId: CLASSES.teacher.id,
  
  // Helper functions to query students by status
  getStudentsByStatus: (role: "teacher" | "assistant", status: string): string[] => {
    const classData = role === "teacher" ? STUDENTS.teacherClass : STUDENTS.assistantClass;
    return Object.entries(classData)
      .filter(([_, data]) => data.status === status)
      .map(([id]) => id);
  },
  
  // Legacy student references (marked for deprecation)
  students: {
    // Assistant test targets (Class 2)
    pendingReview1: "20240052" as const,  // ✅ 待评阅学生1
    pendingReview2: "20240055" as const,  // ✅ 待评阅学生2（与teacher seed共用）
    
    // Teacher test targets (Class 1)
    perfectScore: "20240010" as const,    // ✅ 满分100
    passThreshold: "20240011" as const,   // ✅ 及格线60
    zeroScore: "20240012" as const,       // ✅ 零分0
    pendingTeacher: "20240013" as const,  // ✅ 待评阅
    
    // Edge cases for special content testing
    xssTest: "20240014" as const,         // ✅ XSS攻击内容
    sqlInjection: "20240015" as const,    // ✅ SQL注入
    longText: "20240016" as const,        // ✅ 超长文本
    formatted: "20240017" as const,       // ✅ 格式化文本
    htmlTags: "20240018" as const,        // ✅ HTML标签
    shortDuration: "20240019" as const,   // ✅ 极短实验时间
    longDuration: "20240020" as const,    // ✅ 极长实验时间
  },
} as const;

// ===== Type Definitions =====

export type StudentId = keyof typeof STUDENTS.teacherClass | keyof typeof STUDENTS.assistantClass;
export type StudentStatus = "completed" | "in_progress" | "not_started" | "graded" | "pending_review" | "rejected" | "graded_perfect" | "graded_pass" | "graded_zero";

// ===== API Endpoints =====

export const API = {
  gradingPolicy: /\\/api\\/v1\\/classes\\/\\d+\\/grading-policy$/,
  classGradeSummaries: /\\/api\\/v1\\/classes\\/\\d+\\/grade-summaries/,
  assistantGradeSummaries: /\\/assistants\\/\\d+\\/grade-summaries/,
  teacherGradeSummaries: /\\/teachers\\/\\d+\\/grade-summaries/,
} as const;

// ===== Timeouts =====

export const TIMEOUTS = {
  navigation: 5000,
  modal: 3000,
  message: 20000,
  api: 10000,
} as const;
`;
}

function main() {
  const content = generateFixturesContent();
  
  // 输出到前端目录
  const outputPath = path.resolve(__dirname, "../../fe/tests/e2e/fixtures.ts");
  
  fs.writeFileSync(outputPath, content, "utf-8");
  console.log(`✅ Generated fixtures at: ${outputPath}`);
  
  // 同时生成 JSON 版本供调试
  const jsonOutputPath = path.resolve(__dirname, "../../fe/tests/e2e/generated/fixtures.json");
  fs.mkdirSync(path.dirname(jsonOutputPath), { recursive: true });
  
  const jsonContent = {
    generatedAt: new Date().toISOString(),
    accounts: ["teacher", "assistant", "student", "admin"],
    classes: {
      teacher: { id: 1, name: "计算机科学与技术2501" },
      assistant: { id: 2, name: "计算机科学与技术2502" },
    },
    students: {
      teacherClass: extractTeacherStudents(),
      assistantClass: extractAssistantStudents(),
    },
  };
  
  fs.writeFileSync(jsonOutputPath, JSON.stringify(jsonContent, null, 2), "utf-8");
  console.log(`✅ Generated JSON fixtures at: ${jsonOutputPath}`);
}

main();
