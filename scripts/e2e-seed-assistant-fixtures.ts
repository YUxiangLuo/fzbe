import mysql from "mysql2/promise";
import type { Connection, RowDataPacket } from "mysql2/promise";

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

const FIXTURES: ExperimentFixture[] = [
  {
    username: "20240051",
    classId: 2,
    experimentId: 902051,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4880,
    selectedIndustry: "新能源",
    selectedCompany: "远航电池",
    selectedProduct: "动力电芯",
    selectedBestModel: "arima",
    startTime: "2026-01-20 08:30:00",
    lastActivityAt: "2026-01-20 10:00:00",
    completionTime: "2026-01-20 10:00:00",
    grade: {
      exp_flow_demand_data_preparation: 88,
      exp_flow_demand_descriptive_stats: 86,
      exp_flow_demand_model_selection: 90,
      exp_flow_demand_generate_results: 87,
      exp_flow_production_inventory_calc: 85,
      exp_flow_production_service_level: 84,
      exp_flow_production_variable_calc: 83,
      exp_flow_production_plan_creation: 86,
      exp_flow_report_submission: 89,
      knowledge_test: 82,
      model_quality: 88,
      report_quality: 86,
    },
    report: {
      status: "graded",
      submittedAt: "2026-01-20 10:05:00",
      grade: 86,
      feedback: "整体表现较好，建议补充参数敏感性分析。",
      gradedBy: 202,
      useReportPdf: true,
      content: "<h1>E2E Assistant Fixture Report - Graded</h1>",
    },
  },
  {
    username: "20240052",
    classId: 2,
    experimentId: 902052,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4520,
    selectedIndustry: "医疗器械",
    selectedCompany: "华康医疗",
    selectedProduct: "监护仪模块",
    selectedBestModel: "exp",
    startTime: "2026-01-21 08:20:00",
    lastActivityAt: "2026-01-21 09:50:00",
    completionTime: "2026-01-21 09:50:00",
    grade: {
      exp_flow_demand_data_preparation: 80,
      exp_flow_demand_descriptive_stats: 79,
      exp_flow_demand_model_selection: 81,
      exp_flow_demand_generate_results: 80,
      exp_flow_production_inventory_calc: 78,
      exp_flow_production_service_level: 77,
      exp_flow_production_variable_calc: 79,
      exp_flow_production_plan_creation: 80,
      exp_flow_report_submission: 82,
      knowledge_test: 76,
      model_quality: 81,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-21 09:55:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: "<h1>E2E Assistant Fixture Report - Pending</h1>",
    },
  },
  {
    username: "20240053",
    classId: 2,
    experimentId: 902053,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4260,
    selectedIndustry: "生物医药",
    selectedCompany: "同济生物",
    selectedProduct: "检测试剂盒",
    selectedBestModel: "ma",
    startTime: "2026-01-22 08:10:00",
    lastActivityAt: "2026-01-22 09:25:00",
    completionTime: "2026-01-22 09:25:00",
    grade: {
      exp_flow_demand_data_preparation: 72,
      exp_flow_demand_descriptive_stats: 71,
      exp_flow_demand_model_selection: 73,
      exp_flow_demand_generate_results: 70,
      exp_flow_production_inventory_calc: 69,
      exp_flow_production_service_level: 68,
      exp_flow_production_variable_calc: 67,
      exp_flow_production_plan_creation: 69,
      exp_flow_report_submission: 66,
      knowledge_test: 70,
      model_quality: 71,
      report_quality: 0,
    },
    report: {
      status: "rejected",
      submittedAt: "2026-01-22 09:30:00",
      grade: null,
      feedback: "报告关键结论与实验过程不一致，请修订后重提。",
      gradedBy: 202,
      useReportPdf: true,
      content: "<h1>E2E Assistant Fixture Report - Rejected</h1>",
    },
  },
  {
    username: "20240054",
    classId: 2,
    experimentId: 902054,
    status: "In Progress",
    highestCompletedStep: 3,
    currentStep: 4,
    totalActiveDurationSeconds: 2180,
    selectedIndustry: "汽车零部件",
    selectedCompany: "拓越部件",
    selectedProduct: "制动总成",
    selectedBestModel: "lstm",
    startTime: "2026-01-23 08:00:00",
    lastActivityAt: "2026-01-23 08:55:00",
    completionTime: null,
  },
];

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
  const usernames = Array.from(new Set(FIXTURES.map((fixture) => fixture.username)));
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

async function clearAssistantFixtureData(conn: Connection, classIds: number[]) {
  const placeholders = classIds.map(() => "?").join(", ");
  const [studentRows] = await conn.query<StudentIdRow[]>(
    `SELECT e.student_id
       FROM enrollments e
      WHERE e.class_id IN (${placeholders})`,
    classIds,
  );
  const studentIds = studentRows.map((row) => row.student_id);
  if (studentIds.length === 0) return;

  const studentPlaceholders = studentIds.map(() => "?").join(", ");
  await conn.query(`DELETE FROM quiz_answers WHERE student_id IN (${studentPlaceholders})`, studentIds);
  await conn.query(`DELETE FROM experiment_reports WHERE student_id IN (${studentPlaceholders})`, studentIds);
  await conn.query(`DELETE FROM experiment_grade WHERE student_id IN (${studentPlaceholders})`, studentIds);
  await conn.query(`DELETE FROM experiment_step_events WHERE student_id IN (${studentPlaceholders})`, studentIds);
  await conn.query(`DELETE FROM experiment_status WHERE student_id IN (${studentPlaceholders})`, studentIds);
}

async function insertStepEvents(conn: Connection, fixture: ExperimentFixture, studentId: number) {
  const completedSteps = Math.max(0, Math.min(7, fixture.highestCompletedStep));
  const currentStep = Math.max(1, Math.min(7, fixture.currentStep));
  let offsetMinutes = 5;

  for (let step = 1; step <= completedSteps; step += 1) {
    const startedAt = addMinutes(fixture.startTime, offsetMinutes);
    offsetMinutes += 5;
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

  const assistantId = Number(process.env.E2E_FIXTURE_ASSISTANT_ID ?? "202");
  const assistantClassId = Number(process.env.E2E_FIXTURE_ASSISTANT_CLASS_ID ?? "2");
  const reportPath = requireEnv("E2E_FIXTURE_REPORT_PATH");
  if (!Number.isInteger(assistantId)) throw new Error("E2E_FIXTURE_ASSISTANT_ID is invalid");
  if (!Number.isInteger(assistantClassId)) throw new Error("E2E_FIXTURE_ASSISTANT_CLASS_ID is invalid");

  const conn = await mysql.createConnection({
    host: dbHost,
    user: dbUser,
    password: dbPassword,
    database: dbDatabase,
    charset: "utf8mb4",
  });

  try {
    await conn.beginTransaction();

    await conn.query("DELETE FROM class_assistants WHERE assistant_id = ?", [assistantId]);
    await conn.query(
      "INSERT INTO class_assistants (assistant_id, class_id) VALUES (?, ?)",
      [assistantId, assistantClassId],
    );

    const targetClassIds = Array.from(new Set(FIXTURES.map((fixture) => fixture.classId)));
    await clearAssistantFixtureData(conn, targetClassIds);

    const assignments = await getStudentAssignments(conn);
    for (const fixture of FIXTURES) {
      const assignment = assignments.get(fixture.username);
      if (!assignment) {
        throw new Error(`Cannot find enrolled student for username=${fixture.username}`);
      }
      if (assignment.classId !== fixture.classId) {
        throw new Error(
          `Student ${fixture.username} is in class ${assignment.classId}, expected class ${fixture.classId}`,
        );
      }

      await insertExperimentFixture(conn, fixture, assignment.userId, reportPath);
    }

    await conn.commit();
  } catch (error) {
    await conn.rollback();
    throw error;
  } finally {
    await conn.end();
  }

  console.log("Seeded assistant e2e fixture data.");
}

main().catch((error) => {
  console.error("Failed to seed assistant e2e fixture data:", error);
  process.exit(1);
});
