import mysql from "mysql2/promise";
import type { Connection, RowDataPacket } from "mysql2/promise";

// Edge Case Seed Script
// Provides specialized test data for boundary conditions and unusual scenarios

type ExperimentStatusValue = "Completed" | "In Progress";
type ReportStatusValue = "submitted" | "graded" | "rejected";

interface GradeFixture {
  exp_flow_demand_data_preparation: number;
  exp_flow_demand_descriptive_stats: number;
  exp_flow_demand_model_selection: number;
  exp_flow_demand_generate_results: number;
  exp_flow_production_inventory_calc: number;
  exp_flow_production_service_level: number;
  exp_flow_production_variable_calc: number;
  exp_flow_production_plan_creation: number;
  exp_flow_report_submission: number;
  knowledge_test: number;
  model_quality: number;
  report_quality: number;
}

interface ReportFixture {
  status: ReportStatusValue;
  submittedAt: string;
  grade: number | null;
  feedback: string | null;
  gradedBy: number | null;
  useReportPdf: boolean;
  content: string;
}

interface ExperimentFixture {
  username: string;
  classId: number;
  experimentId: number;
  status: ExperimentStatusValue;
  highestCompletedStep: number;
  currentStep: number;
  totalActiveDurationSeconds: number;
  selectedIndustry: string;
  selectedCompany: string;
  selectedProduct: string;
  selectedBestModel: "arima" | "exp" | "ma" | "lstm";
  startTime: string;
  lastActivityAt: string;
  completionTime: string | null;
  grade?: GradeFixture;
  report?: ReportFixture;
}

interface StudentAssignmentRow extends RowDataPacket {
  user_id: number;
  username: string;
  class_id: number;
}

interface StudentIdRow extends RowDataPacket {
  student_id: number;
}

// Special content for edge case testing
const SPECIAL_CONTENT = {
  // XSS attempt (should be properly escaped by frontend)
  xssAttempt: `<script>alert('xss')</script><img src="x" onerror="alert('xss')">`,
  
  // SQL injection attempt
  sqlInjection: `'; DROP TABLE users; --'; DELETE FROM experiment_reports; --`,
  
  // Very long text (10,000 characters)
  veryLongText: "A".repeat(10000),
  
  // Unicode and special characters
  unicodeAndSpecial: `中文测试 🎉 Emoji 测试 \n\t\r Special chars: <>&"'`,
  
  // Newlines and formatting
  formattedText: `第一行内容\n\n第二行内容\n\t- 列表项1\n\t- 列表项2\n\n第三行内容`,
  
  // HTML tags (should be escaped or sanitized)
  htmlTags: `<h1>标题</h1><p>段落<strong>粗体</strong><em>斜体</em></p><table><tr><td>表格</td></tr></table>`,
};

// Edge case fixtures for Class 1 (special content testing)
const EDGE_CASE_FIXTURES: ExperimentFixture[] = [
  // XSS and special characters test
  {
    username: "20240014",
    classId: 1,
    experimentId: 900014,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4200,
    selectedIndustry: "网络安全",
    selectedCompany: "安全卫士科技",
    selectedProduct: "防火墙系统",
    selectedBestModel: "lstm",
    startTime: "2026-01-27 08:00:00",
    lastActivityAt: "2026-01-27 09:45:00",
    completionTime: "2026-01-27 09:45:00",
    grade: {
      exp_flow_demand_data_preparation: 88,
      exp_flow_demand_descriptive_stats: 87,
      exp_flow_demand_model_selection: 89,
      exp_flow_demand_generate_results: 88,
      exp_flow_production_inventory_calc: 86,
      exp_flow_production_service_level: 85,
      exp_flow_production_variable_calc: 87,
      exp_flow_production_plan_creation: 88,
      exp_flow_report_submission: 89,
      knowledge_test: 86,
      model_quality: 88,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-27 09:50:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: `<h1>XSS Test Report</h1><p>${SPECIAL_CONTENT.xssAttempt}</p><p>${SPECIAL_CONTENT.unicodeAndSpecial}</p>`,
    },
  },
  // SQL injection attempt test
  {
    username: "20240015",
    classId: 1,
    experimentId: 900015,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4500,
    selectedIndustry: "数据库",
    selectedCompany: "数据大师",
    selectedProduct: "分布式存储",
    selectedBestModel: "arima",
    startTime: "2026-01-28 08:00:00",
    lastActivityAt: "2026-01-28 09:50:00",
    completionTime: "2026-01-28 09:50:00",
    grade: {
      exp_flow_demand_data_preparation: 85,
      exp_flow_demand_descriptive_stats: 84,
      exp_flow_demand_model_selection: 86,
      exp_flow_demand_generate_results: 85,
      exp_flow_production_inventory_calc: 83,
      exp_flow_production_service_level: 82,
      exp_flow_production_variable_calc: 84,
      exp_flow_production_plan_creation: 85,
      exp_flow_report_submission: 86,
      knowledge_test: 83,
      model_quality: 85,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-28 09:55:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: `<h1>SQL Injection Test</h1><p>${SPECIAL_CONTENT.sqlInjection}</p><p>正常内容在此之后</p>`,
    },
  },
  // Very long text test
  {
    username: "20240016",
    classId: 1,
    experimentId: 900016,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 6000,
    selectedIndustry: "出版印刷",
    selectedCompany: "万卷出版",
    selectedProduct: "电子书平台",
    selectedBestModel: "exp",
    startTime: "2026-01-29 08:00:00",
    lastActivityAt: "2026-01-29 10:00:00",
    completionTime: "2026-01-29 10:00:00",
    grade: {
      exp_flow_demand_data_preparation: 90,
      exp_flow_demand_descriptive_stats: 89,
      exp_flow_demand_model_selection: 91,
      exp_flow_demand_generate_results: 90,
      exp_flow_production_inventory_calc: 88,
      exp_flow_production_service_level: 87,
      exp_flow_production_variable_calc: 89,
      exp_flow_production_plan_creation: 90,
      exp_flow_report_submission: 91,
      knowledge_test: 88,
      model_quality: 90,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-29 10:05:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: `<h1>Long Content Test</h1><p>${SPECIAL_CONTENT.veryLongText.substring(0, 5000)}</p>`,
    },
  },
  // Formatted text with newlines
  {
    username: "20240017",
    classId: 1,
    experimentId: 900017,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4800,
    selectedIndustry: "文案策划",
    selectedCompany: "创意无限",
    selectedProduct: "内容管理系统",
    selectedBestModel: "ma",
    startTime: "2026-01-30 08:00:00",
    lastActivityAt: "2026-01-30 09:50:00",
    completionTime: "2026-01-30 09:50:00",
    grade: {
      exp_flow_demand_data_preparation: 87,
      exp_flow_demand_descriptive_stats: 86,
      exp_flow_demand_model_selection: 88,
      exp_flow_demand_generate_results: 87,
      exp_flow_production_inventory_calc: 85,
      exp_flow_production_service_level: 84,
      exp_flow_production_variable_calc: 86,
      exp_flow_production_plan_creation: 87,
      exp_flow_report_submission: 88,
      knowledge_test: 85,
      model_quality: 87,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-30 09:55:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: `<h1>Formatted Text Test</h1><pre>${SPECIAL_CONTENT.formattedText}</pre>`,
    },
  },
  // HTML tags test
  {
    username: "20240018",
    classId: 1,
    experimentId: 900018,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4700,
    selectedIndustry: "网页设计",
    selectedCompany: "前端工坊",
    selectedProduct: "UI组件库",
    selectedBestModel: "lstm",
    startTime: "2026-01-31 08:00:00",
    lastActivityAt: "2026-01-31 09:45:00",
    completionTime: "2026-01-31 09:45:00",
    grade: {
      exp_flow_demand_data_preparation: 86,
      exp_flow_demand_descriptive_stats: 85,
      exp_flow_demand_model_selection: 87,
      exp_flow_demand_generate_results: 86,
      exp_flow_production_inventory_calc: 84,
      exp_flow_production_service_level: 83,
      exp_flow_production_variable_calc: 85,
      exp_flow_production_plan_creation: 86,
      exp_flow_report_submission: 87,
      knowledge_test: 84,
      model_quality: 86,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-31 09:50:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: `<h1>HTML Tags Test</h1>${SPECIAL_CONTENT.htmlTags}`,
    },
  },
];

// Edge case: Student with extremely short experiment time
const SHORT_EXPERIMENT_FIXTURE: ExperimentFixture = {
  username: "20240019",
  classId: 1,
  experimentId: 900019,
  status: "Completed",
  highestCompletedStep: 7,
  currentStep: 7,
  totalActiveDurationSeconds: 60, // Only 1 minute
  selectedIndustry: "快餐",
  selectedCompany: "速食帝国",
  selectedProduct: "预制菜品",
  selectedBestModel: "arima",
  startTime: "2026-02-01 08:00:00",
  lastActivityAt: "2026-02-01 08:01:00",
  completionTime: "2026-02-01 08:01:00",
  grade: {
    exp_flow_demand_data_preparation: 50,
    exp_flow_demand_descriptive_stats: 50,
    exp_flow_demand_model_selection: 50,
    exp_flow_demand_generate_results: 50,
    exp_flow_production_inventory_calc: 50,
    exp_flow_production_service_level: 50,
    exp_flow_production_variable_calc: 50,
    exp_flow_production_plan_creation: 50,
    exp_flow_report_submission: 50,
    knowledge_test: 50,
    model_quality: 50,
    report_quality: 50,
  },
  report: {
    status: "graded",
    submittedAt: "2026-02-01 08:02:00",
    grade: 50,
    feedback: "实验时间过短，可能存在异常。",
    gradedBy: 101,
    useReportPdf: true,
    content: "<h1>Short Experiment Test</h1><p>Completed in 1 minute.</p>",
  },
};

// Edge case: Student with extremely long experiment time
const LONG_EXPERIMENT_FIXTURE: ExperimentFixture = {
  username: "20240020",
  classId: 1,
  experimentId: 900020,
  status: "Completed",
  highestCompletedStep: 7,
  currentStep: 7,
  totalActiveDurationSeconds: 86400, // 24 hours
  selectedIndustry: "航天",
  selectedCompany: "深空探索",
  selectedProduct: "卫星控制系统",
  selectedBestModel: "lstm",
  startTime: "2026-02-02 08:00:00",
  lastActivityAt: "2026-02-03 08:00:00",
  completionTime: "2026-02-03 08:00:00",
  grade: {
    exp_flow_demand_data_preparation: 95,
    exp_flow_demand_descriptive_stats: 94,
    exp_flow_demand_model_selection: 96,
    exp_flow_demand_generate_results: 95,
    exp_flow_production_inventory_calc: 93,
    exp_flow_production_service_level: 92,
    exp_flow_production_variable_calc: 94,
    exp_flow_production_plan_creation: 95,
    exp_flow_report_submission: 96,
    knowledge_test: 93,
    model_quality: 95,
    report_quality: 94,
  },
  report: {
    status: "graded",
    submittedAt: "2026-02-03 08:05:00",
    grade: 94,
    feedback: "详细且完整的实验报告，耗时较长但质量优秀。",
    gradedBy: 101,
    useReportPdf: true,
    content: "<h1>Long Experiment Test</h1><p>Completed over 24 hours.</p>",
  },
};

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required env var: ${name}`);
  return value;
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function toSqlTimestamp(date: Date): string {
  const year = date.getUTCFullYear();
  const month = pad2(date.getUTCMonth() + 1);
  const day = pad2(date.getUTCDate());
  const hour = pad2(date.getUTCHours());
  const minute = pad2(date.getUTCMinutes());
  const second = pad2(date.getUTCSeconds());
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

function addMinutes(timestamp: string, minutes: number): string {
  const iso = timestamp.replace(" ", "T") + "Z";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    throw new Error(`Invalid timestamp: ${timestamp}`);
  }
  date.setUTCMinutes(date.getUTCMinutes() + minutes);
  return toSqlTimestamp(date);
}

async function getStudentAssignments(conn: Connection): Promise<Map<string, { userId: number; classId: number }>> {
  const allFixtures = [...EDGE_CASE_FIXTURES, SHORT_EXPERIMENT_FIXTURE, LONG_EXPERIMENT_FIXTURE];
  const usernames = Array.from(new Set(allFixtures.map((fixture) => fixture.username)));
  const placeholders = usernames.map(() => "?").join(", ");
  const [rows] = await conn.query<StudentAssignmentRow[]>(
    `SELECT u.user_id, u.username, e.class_id
       FROM users u
       JOIN enrollments e ON e.student_id = u.user_id
      WHERE u.username IN (${placeholders})`,
    usernames,
  );

  const mapping = new Map<string, { userId: number; classId: number }>();
  for (const row of rows) {
    mapping.set(row.username, { userId: row.user_id, classId: row.class_id });
  }
  return mapping;
}

async function clearEdgeCaseData(conn: Connection, classIds: number[]) {
  const placeholders = classIds.map(() => "?").join(", ");
  const [studentRows] = await conn.query<StudentIdRow[]>(
    `SELECT e.student_id
       FROM enrollments e
      WHERE e.class_id IN (${placeholders})`,
    classIds,
  );
  const studentIds = studentRows.map((row) => row.student_id);
  if (studentIds.length === 0) return;

  // Only delete data for specific edge case experiment IDs
  const edgeCaseExperimentIds = [
    900014, 900015, 900016, 900017, 900018, 900019, 900020
  ];
  const expPlaceholders = edgeCaseExperimentIds.map(() => "?").join(", ");
  
  await conn.query(`DELETE FROM quiz_answers WHERE experiment_id IN (${expPlaceholders})`, edgeCaseExperimentIds);
  await conn.query(`DELETE FROM experiment_reports WHERE experiment_id IN (${expPlaceholders})`, edgeCaseExperimentIds);
  await conn.query(`DELETE FROM experiment_grade WHERE experiment_id IN (${expPlaceholders})`, edgeCaseExperimentIds);
  await conn.query(`DELETE FROM experiment_step_events WHERE experiment_id IN (${expPlaceholders})`, edgeCaseExperimentIds);
  await conn.query(`DELETE FROM experiment_status WHERE experiment_id IN (${expPlaceholders})`, edgeCaseExperimentIds);
}

async function insertStepEvents(conn: Connection, fixture: ExperimentFixture, studentId: number) {
  const completedSteps = Math.max(0, Math.min(7, fixture.highestCompletedStep));
  const currentStep = Math.max(1, Math.min(7, fixture.currentStep));
  let offsetMinutes = 5;

  for (let step = 1; step <= completedSteps; step += 1) {
    const startedAt = addMinutes(fixture.startTime, offsetMinutes);
    offsetMinutes += Math.max(1, Math.floor(fixture.totalActiveDurationSeconds / 60 / completedSteps));
    const completedAt = addMinutes(fixture.startTime, offsetMinutes);
    offsetMinutes += 5;

    await conn.query(
      "INSERT INTO experiment_step_events (experiment_id, student_id, step_order, event_type, event_timestamp) VALUES (?, ?, ?, 'STARTED', ?)",
      [fixture.experimentId, studentId, step, startedAt],
    );
    await conn.query(
      "INSERT INTO experiment_step_events (experiment_id, student_id, step_order, event_type, event_timestamp) VALUES (?, ?, ?, 'COMPLETED', ?)",
      [fixture.experimentId, studentId, step, completedAt],
    );
  }

  if (fixture.status === "In Progress" && currentStep > completedSteps) {
    const startedAt = addMinutes(fixture.startTime, offsetMinutes);
    await conn.query(
      "INSERT INTO experiment_step_events (experiment_id, student_id, step_order, event_type, event_timestamp) VALUES (?, ?, ?, 'STARTED', ?)",
      [fixture.experimentId, studentId, currentStep, startedAt],
    );
  }
}

async function insertExperimentFixture(
  conn: Connection,
  fixture: ExperimentFixture,
  studentId: number,
  reportPath: string,
) {
  await conn.query(
    `INSERT INTO experiment_status (
      experiment_id,
      student_id,
      status,
      highest_completed_step,
      current_step,
      total_active_duration_seconds,
      selected_industry,
      selected_company,
      selected_product,
      selected_best_model,
      start_time,
      last_activity_at,
      completion_time
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      fixture.experimentId,
      studentId,
      fixture.status,
      fixture.highestCompletedStep,
      fixture.currentStep,
      fixture.totalActiveDurationSeconds,
      fixture.selectedIndustry,
      fixture.selectedCompany,
      fixture.selectedProduct,
      fixture.selectedBestModel,
      fixture.startTime,
      fixture.lastActivityAt,
      fixture.completionTime,
    ],
  );

  await insertStepEvents(conn, fixture, studentId);

  if (fixture.grade) {
    const g = fixture.grade;
    await conn.query(
      `INSERT INTO experiment_grade (
        experiment_id,
        student_id,
        exp_flow_demand_data_preparation,
        exp_flow_demand_descriptive_stats,
        exp_flow_demand_model_selection,
        exp_flow_demand_generate_results,
        exp_flow_production_inventory_calc,
        exp_flow_production_service_level,
        exp_flow_production_variable_calc,
        exp_flow_production_plan_creation,
        exp_flow_report_submission,
        knowledge_test,
        model_quality,
        report_quality
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        fixture.experimentId,
        studentId,
        g.exp_flow_demand_data_preparation,
        g.exp_flow_demand_descriptive_stats,
        g.exp_flow_demand_model_selection,
        g.exp_flow_demand_generate_results,
        g.exp_flow_production_inventory_calc,
        g.exp_flow_production_service_level,
        g.exp_flow_production_variable_calc,
        g.exp_flow_production_plan_creation,
        g.exp_flow_report_submission,
        g.knowledge_test,
        g.model_quality,
        g.report_quality,
      ],
    );
  }

  if (fixture.report) {
    const r = fixture.report;
    await conn.query(
      `INSERT INTO experiment_reports (
        experiment_id,
        student_id,
        report_content,
        pdf_file_path,
        status,
        submitted_at,
        grade,
        feedback,
        graded_by
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        fixture.experimentId,
        studentId,
        r.content,
        r.useReportPdf ? reportPath : null,
        r.status,
        r.submittedAt,
        r.grade,
        r.feedback,
        r.gradedBy,
      ],
    );
  }
}

async function main() {
  const dbHost = requireEnv("DB_HOST");
  const dbUser = requireEnv("DB_USER");
  const dbDatabase = requireEnv("DB_DATABASE");
  const dbPassword = process.env.DB_PASSWORD ?? "";
  const reportPath = requireEnv("E2E_FIXTURE_REPORT_PATH");

  const conn = await mysql.createConnection({
    host: dbHost,
    user: dbUser,
    password: dbPassword,
    database: dbDatabase,
    charset: "utf8mb4",
  });

  try {
    await conn.beginTransaction();

    const targetClassIds = Array.from(new Set(EDGE_CASE_FIXTURES.map((f) => f.classId)));
    await clearEdgeCaseData(conn, targetClassIds);

    const assignments = await getStudentAssignments(conn);
    
    // Insert all edge case fixtures
    const allFixtures = [...EDGE_CASE_FIXTURES, SHORT_EXPERIMENT_FIXTURE, LONG_EXPERIMENT_FIXTURE];
    for (const fixture of allFixtures) {
      const assignment = assignments.get(fixture.username);
      if (!assignment) {
        console.warn(`Skipping ${fixture.username}: not found in database`);
        continue;
      }
      if (assignment.classId !== fixture.classId) {
        console.warn(`Skipping ${fixture.username}: class mismatch`);
        continue;
      }

      await insertExperimentFixture(conn, fixture, assignment.userId, reportPath);
    }

    await conn.commit();
    console.log(`Seeded ${allFixtures.length} edge case fixtures.`);
  } catch (error) {
    await conn.rollback();
    throw error;
  } finally {
    await conn.end();
  }
}

main().catch((error) => {
  console.error("Failed to seed edge case fixtures:", error);
  process.exit(1);
});
