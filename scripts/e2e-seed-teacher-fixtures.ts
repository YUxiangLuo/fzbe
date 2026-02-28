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
    username: "20240001",
    classId: 1,
    experimentId: 900001,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 5400,
    selectedIndustry: "纺织业",
    selectedCompany: "鲁泰纺织股份有限公司",
    selectedProduct: "高支纱",
    selectedBestModel: "arima",
    startTime: "2026-01-10 08:00:00",
    lastActivityAt: "2026-01-10 10:00:00",
    completionTime: "2026-01-10 10:00:00",
    grade: {
      exp_flow_demand_data_preparation: 92,
      exp_flow_demand_descriptive_stats: 90,
      exp_flow_demand_model_selection: 88,
      exp_flow_demand_generate_results: 91,
      exp_flow_production_inventory_calc: 89,
      exp_flow_production_service_level: 87,
      exp_flow_production_variable_calc: 86,
      exp_flow_production_plan_creation: 90,
      exp_flow_report_submission: 95,
      knowledge_test: 84,
      model_quality: 93,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-10 10:15:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Pending Review</h1><p>seeded for teacher/assistant review workflow.</p>",
    },
  },
  {
    username: "20240003",
    classId: 1,
    experimentId: 900003,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 5000,
    selectedIndustry: "电子制造",
    selectedCompany: "智联制造集团",
    selectedProduct: "工业传感器",
    selectedBestModel: "arima",
    startTime: "2026-01-11 09:00:00",
    lastActivityAt: "2026-01-11 10:35:00",
    completionTime: "2026-01-11 10:35:00",
    grade: {
      exp_flow_demand_data_preparation: 96,
      exp_flow_demand_descriptive_stats: 94,
      exp_flow_demand_model_selection: 95,
      exp_flow_demand_generate_results: 94,
      exp_flow_production_inventory_calc: 93,
      exp_flow_production_service_level: 92,
      exp_flow_production_variable_calc: 94,
      exp_flow_production_plan_creation: 95,
      exp_flow_report_submission: 97,
      knowledge_test: 91,
      model_quality: 95,
      report_quality: 94,
    },
    report: {
      status: "graded",
      submittedAt: "2026-01-11 10:40:00",
      grade: 94,
      feedback: "报告结构清晰，模型选择与结论一致，结果解释充分。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Excellent</h1>",
    },
  },
  {
    username: "20240004",
    classId: 1,
    experimentId: 900004,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4700,
    selectedIndustry: "机械制造",
    selectedCompany: "宏远机电",
    selectedProduct: "精密轴承",
    selectedBestModel: "exp",
    startTime: "2026-01-12 08:30:00",
    lastActivityAt: "2026-01-12 09:50:00",
    completionTime: "2026-01-12 09:50:00",
    grade: {
      exp_flow_demand_data_preparation: 86,
      exp_flow_demand_descriptive_stats: 85,
      exp_flow_demand_model_selection: 87,
      exp_flow_demand_generate_results: 84,
      exp_flow_production_inventory_calc: 83,
      exp_flow_production_service_level: 82,
      exp_flow_production_variable_calc: 84,
      exp_flow_production_plan_creation: 85,
      exp_flow_report_submission: 86,
      knowledge_test: 81,
      model_quality: 85,
      report_quality: 84,
    },
    report: {
      status: "graded",
      submittedAt: "2026-01-12 10:00:00",
      grade: 84,
      feedback: "关键步骤完整，逻辑较清晰，个别结论可再细化。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Good</h1>",
    },
  },
  {
    username: "20240005",
    classId: 1,
    experimentId: 900005,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4300,
    selectedIndustry: "食品加工",
    selectedCompany: "新味食品",
    selectedProduct: "速食汤包",
    selectedBestModel: "ma",
    startTime: "2026-01-13 08:10:00",
    lastActivityAt: "2026-01-13 09:25:00",
    completionTime: "2026-01-13 09:25:00",
    grade: {
      exp_flow_demand_data_preparation: 76,
      exp_flow_demand_descriptive_stats: 75,
      exp_flow_demand_model_selection: 78,
      exp_flow_demand_generate_results: 74,
      exp_flow_production_inventory_calc: 73,
      exp_flow_production_service_level: 72,
      exp_flow_production_variable_calc: 71,
      exp_flow_production_plan_creation: 74,
      exp_flow_report_submission: 76,
      knowledge_test: 72,
      model_quality: 75,
      report_quality: 74,
    },
    report: {
      status: "graded",
      submittedAt: "2026-01-13 09:30:00",
      grade: 74,
      feedback: "流程完成度尚可，建议加强误差分析与结论解释。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Average</h1>",
    },
  },
  {
    username: "20240006",
    classId: 1,
    experimentId: 900006,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 3950,
    selectedIndustry: "家电",
    selectedCompany: "云帆电器",
    selectedProduct: "智能电饭煲",
    selectedBestModel: "exp",
    startTime: "2026-01-14 08:20:00",
    lastActivityAt: "2026-01-14 09:35:00",
    completionTime: "2026-01-14 09:35:00",
    grade: {
      exp_flow_demand_data_preparation: 67,
      exp_flow_demand_descriptive_stats: 66,
      exp_flow_demand_model_selection: 68,
      exp_flow_demand_generate_results: 65,
      exp_flow_production_inventory_calc: 64,
      exp_flow_production_service_level: 63,
      exp_flow_production_variable_calc: 62,
      exp_flow_production_plan_creation: 64,
      exp_flow_report_submission: 66,
      knowledge_test: 63,
      model_quality: 65,
      report_quality: 64,
    },
    report: {
      status: "graded",
      submittedAt: "2026-01-14 09:40:00",
      grade: 64,
      feedback: "勉强达标，建议重新检查关键参数设置。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Pass</h1>",
    },
  },
  {
    username: "20240007",
    classId: 1,
    experimentId: 900007,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 3600,
    selectedIndustry: "文旅服务",
    selectedCompany: "星河文旅",
    selectedProduct: "景区联票",
    selectedBestModel: "ma",
    startTime: "2026-01-15 08:05:00",
    lastActivityAt: "2026-01-15 09:10:00",
    completionTime: "2026-01-15 09:10:00",
    grade: {
      exp_flow_demand_data_preparation: 58,
      exp_flow_demand_descriptive_stats: 57,
      exp_flow_demand_model_selection: 56,
      exp_flow_demand_generate_results: 55,
      exp_flow_production_inventory_calc: 54,
      exp_flow_production_service_level: 53,
      exp_flow_production_variable_calc: 52,
      exp_flow_production_plan_creation: 54,
      exp_flow_report_submission: 53,
      knowledge_test: 56,
      model_quality: 55,
      report_quality: 54,
    },
    report: {
      status: "graded",
      submittedAt: "2026-01-15 09:15:00",
      grade: 54,
      feedback: "核心分析不完整，建议重新完成实验并补交报告。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Fail</h1>",
    },
  },
  {
    username: "20240008",
    classId: 1,
    experimentId: 900008,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4100,
    selectedIndustry: "物流",
    selectedCompany: "速联物流",
    selectedProduct: "干线运输",
    selectedBestModel: "lstm",
    startTime: "2026-01-16 08:15:00",
    lastActivityAt: "2026-01-16 09:30:00",
    completionTime: "2026-01-16 09:30:00",
    grade: {
      exp_flow_demand_data_preparation: 69,
      exp_flow_demand_descriptive_stats: 68,
      exp_flow_demand_model_selection: 70,
      exp_flow_demand_generate_results: 67,
      exp_flow_production_inventory_calc: 66,
      exp_flow_production_service_level: 65,
      exp_flow_production_variable_calc: 64,
      exp_flow_production_plan_creation: 66,
      exp_flow_report_submission: 62,
      knowledge_test: 69,
      model_quality: 68,
      report_quality: 0,
    },
    report: {
      status: "rejected",
      submittedAt: "2026-01-16 09:35:00",
      grade: null,
      feedback: "报告中关键图表缺失，请补全后重新提交。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Rejected</h1>",
    },
  },
  {
    username: "20240009",
    classId: 1,
    experimentId: 900009,
    status: "In Progress",
    highestCompletedStep: 4,
    currentStep: 5,
    totalActiveDurationSeconds: 2600,
    selectedIndustry: "化工",
    selectedCompany: "海川化工",
    selectedProduct: "工业溶剂",
    selectedBestModel: "arima",
    startTime: "2026-01-17 08:20:00",
    lastActivityAt: "2026-01-17 09:05:00",
    completionTime: null,
  },
  {
    username: "20240051",
    classId: 2,
    experimentId: 901051,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4900,
    selectedIndustry: "新能源",
    selectedCompany: "远航电池",
    selectedProduct: "动力电芯",
    selectedBestModel: "arima",
    startTime: "2026-01-15 08:40:00",
    lastActivityAt: "2026-01-15 10:05:00",
    completionTime: "2026-01-15 10:05:00",
    grade: {
      exp_flow_demand_data_preparation: 88,
      exp_flow_demand_descriptive_stats: 87,
      exp_flow_demand_model_selection: 90,
      exp_flow_demand_generate_results: 88,
      exp_flow_production_inventory_calc: 86,
      exp_flow_production_service_level: 84,
      exp_flow_production_variable_calc: 85,
      exp_flow_production_plan_creation: 87,
      exp_flow_report_submission: 89,
      knowledge_test: 83,
      model_quality: 88,
      report_quality: 86,
    },
    report: {
      status: "graded",
      submittedAt: "2026-01-15 10:10:00",
      grade: 86,
      feedback: "整体较好，参数解释可进一步细化。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Class2 Graded</h1>",
    },
  },
  {
    username: "20240052",
    classId: 2,
    experimentId: 901052,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4500,
    selectedIndustry: "医疗器械",
    selectedCompany: "华康医疗",
    selectedProduct: "监护仪模块",
    selectedBestModel: "exp",
    startTime: "2026-01-16 08:35:00",
    lastActivityAt: "2026-01-16 09:55:00",
    completionTime: "2026-01-16 09:55:00",
    grade: {
      exp_flow_demand_data_preparation: 81,
      exp_flow_demand_descriptive_stats: 80,
      exp_flow_demand_model_selection: 82,
      exp_flow_demand_generate_results: 81,
      exp_flow_production_inventory_calc: 79,
      exp_flow_production_service_level: 78,
      exp_flow_production_variable_calc: 80,
      exp_flow_production_plan_creation: 81,
      exp_flow_report_submission: 83,
      knowledge_test: 77,
      model_quality: 82,
      report_quality: 0,
    },
    report: {
      status: "submitted",
      submittedAt: "2026-01-16 10:00:00",
      grade: null,
      feedback: null,
      gradedBy: null,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Class2 Pending</h1>",
    },
  },
  {
    username: "20240053",
    classId: 2,
    experimentId: 901053,
    status: "Completed",
    highestCompletedStep: 7,
    currentStep: 7,
    totalActiveDurationSeconds: 4200,
    selectedIndustry: "生物医药",
    selectedCompany: "同济生物",
    selectedProduct: "检测试剂盒",
    selectedBestModel: "ma",
    startTime: "2026-01-17 08:10:00",
    lastActivityAt: "2026-01-17 09:20:00",
    completionTime: "2026-01-17 09:20:00",
    grade: {
      exp_flow_demand_data_preparation: 74,
      exp_flow_demand_descriptive_stats: 72,
      exp_flow_demand_model_selection: 73,
      exp_flow_demand_generate_results: 71,
      exp_flow_production_inventory_calc: 70,
      exp_flow_production_service_level: 69,
      exp_flow_production_variable_calc: 68,
      exp_flow_production_plan_creation: 70,
      exp_flow_report_submission: 65,
      knowledge_test: 71,
      model_quality: 72,
      report_quality: 0,
    },
    report: {
      status: "rejected",
      submittedAt: "2026-01-17 09:25:00",
      grade: null,
      feedback: "报告结论与实验过程不一致，请修订后重提。",
      gradedBy: 101,
      useReportPdf: true,
      content: "<h1>E2E Fixture Report - Class2 Rejected</h1>",
    },
  },
  {
    username: "20240054",
    classId: 2,
    experimentId: 901054,
    status: "In Progress",
    highestCompletedStep: 3,
    currentStep: 4,
    totalActiveDurationSeconds: 2100,
    selectedIndustry: "汽车零部件",
    selectedCompany: "拓越部件",
    selectedProduct: "制动总成",
    selectedBestModel: "exp",
    startTime: "2026-01-18 08:00:00",
    lastActivityAt: "2026-01-18 08:55:00",
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

async function clearTeacherFixtureData(conn: Connection, classIds: number[]) {
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

  const teacherId = Number(process.env.E2E_FIXTURE_TEACHER_ID ?? "101");
  const assistantId = Number(process.env.E2E_FIXTURE_ASSISTANT_ID ?? "202");
  const assistantClassId = Number(process.env.E2E_FIXTURE_ASSISTANT_CLASS_ID ?? "1");
  const reportPath = requireEnv("E2E_FIXTURE_REPORT_PATH");
  if (!Number.isInteger(teacherId)) throw new Error("E2E_FIXTURE_TEACHER_ID is invalid");
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
    await clearTeacherFixtureData(conn, targetClassIds);

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

  console.log("Seeded teacher e2e fixture data.");
}

main().catch((error) => {
  console.error("Failed to seed teacher e2e fixture data:", error);
  process.exit(1);
});
