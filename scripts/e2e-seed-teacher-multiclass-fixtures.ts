import mysql from "mysql2/promise";
import type { Connection } from "mysql2/promise";

type ExperimentStatusValue = "Completed" | "In Progress";
type ReportStatusValue = "submitted" | "graded" | "rejected";
type BestModelValue = "arima" | "exp" | "ma" | "lstm" | "ensemble_weighted";

interface UserSeed {
  userId: number;
  username: string;
  password: string;
  fullName: string;
  email: string;
  role: "Teacher" | "Assistant" | "Student";
  phoneNumber: string | null;
}

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
  experimentId: number;
  status: ExperimentStatusValue;
  highestCompletedStep: number;
  currentStep: number;
  totalActiveDurationSeconds: number;
  selectedIndustry: string;
  selectedCompany: string;
  selectedProduct: string;
  selectedBestModel: BestModelValue;
  startTime: string;
  lastActivityAt: string;
  completionTime: string | null;
  grade?: GradeFixture;
  report?: ReportFixture;
}

interface StudentSeed extends UserSeed {
  classId: number;
  fixture?: ExperimentFixture;
}

interface ClassSeed {
  classId: number;
  className: string;
  classCode: string;
  assistantIds: number[];
}

const USER_IDS = {
  teacher: 910101,
  assistant: 910201,
  liveStudent: 20247001,
} as const;

const CLASS_IDS = {
  highPerforming: 8611,
  mixedState: 8612,
  linkedFlow: 8613,
} as const;

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
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
  const date = new Date(`${timestamp.replace(" ", "T")}Z`);
  if (Number.isNaN(date.getTime())) {
    throw new Error(`Invalid timestamp: ${timestamp}`);
  }
  date.setUTCMinutes(date.getUTCMinutes() + minutes);
  return toSqlTimestamp(date);
}

function buildGrade(base: number, reportQuality = base): GradeFixture {
  return {
    exp_flow_demand_data_preparation: base + 1,
    exp_flow_demand_descriptive_stats: base,
    exp_flow_demand_model_selection: base + 2,
    exp_flow_demand_generate_results: base + 1,
    exp_flow_production_inventory_calc: base,
    exp_flow_production_service_level: Math.max(0, base - 1),
    exp_flow_production_variable_calc: base,
    exp_flow_production_plan_creation: base + 1,
    exp_flow_report_submission: base + 2,
    knowledge_test: Math.max(0, base - 2),
    model_quality: base + 1,
    report_quality: reportQuality,
  };
}

const TEACHER: UserSeed = {
  userId: USER_IDS.teacher,
  username: process.env.E2E_TEACHER_USERNAME ?? "teacher_multiclass",
  password: process.env.E2E_TEACHER_PASSWORD ?? "TeacherMulti!234",
  fullName: "多班级教师",
  email: `${
    process.env.E2E_TEACHER_USERNAME ?? "teacher_multiclass"
  }@teacher-multiclass.test.local`,
  phoneNumber: "13800009101",
  role: "Teacher",
};

const ASSISTANT: UserSeed = {
  userId: USER_IDS.assistant,
  username: process.env.E2E_ASSISTANT_USERNAME ?? "assistant_multiclass",
  password: process.env.E2E_ASSISTANT_PASSWORD ?? "AssistantMulti!234",
  fullName: "多班级助教",
  email: `${
    process.env.E2E_ASSISTANT_USERNAME ?? "assistant_multiclass"
  }@teacher-multiclass.test.local`,
  phoneNumber: "13900009101",
  role: "Assistant",
};

function buildStudents(): StudentSeed[] {
  const liveStudentUsername =
    process.env.E2E_STUDENT_USERNAME ?? "20247001";
  const liveStudentPassword =
    process.env.E2E_STUDENT_PASSWORD ?? "StudentMulti!234";

  return [
    {
      userId: 20247002,
      username: "20247002",
      password: liveStudentPassword,
      fullName: "高分班学生甲",
      email: "20247002@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.highPerforming,
      fixture: {
        experimentId: 970002,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 5200,
        selectedIndustry: "新能源汽车",
        selectedCompany: "瀚星动力",
        selectedProduct: "动力电池包",
        selectedBestModel: "arima",
        startTime: "2026-03-05 08:00:00",
        lastActivityAt: "2026-03-05 09:28:00",
        completionTime: "2026-03-05 09:28:00",
        grade: buildGrade(95, 96),
        report: {
          status: "graded",
          submittedAt: "2026-03-05 09:35:00",
          grade: 96,
          feedback: "报告结构完整，分析结论清晰。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>多班级高分班报告 A</h1>",
        },
      },
    },
    {
      userId: 20247003,
      username: "20247003",
      password: liveStudentPassword,
      fullName: "高分班学生乙",
      email: "20247003@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.highPerforming,
      fixture: {
        experimentId: 970003,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4980,
        selectedIndustry: "新能源汽车",
        selectedCompany: "瀚星动力",
        selectedProduct: "电驱控制器",
        selectedBestModel: "exp",
        startTime: "2026-03-06 08:10:00",
        lastActivityAt: "2026-03-06 09:22:00",
        completionTime: "2026-03-06 09:22:00",
        grade: buildGrade(91, 92),
        report: {
          status: "graded",
          submittedAt: "2026-03-06 09:30:00",
          grade: 92,
          feedback: "论证充分，图表表达准确。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>多班级高分班报告 B</h1>",
        },
      },
    },
    {
      userId: 20247004,
      username: "20247004",
      password: liveStudentPassword,
      fullName: "高分班学生丙",
      email: "20247004@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.highPerforming,
      fixture: {
        experimentId: 970004,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4720,
        selectedIndustry: "智能装备",
        selectedCompany: "远海装备",
        selectedProduct: "工业机器人臂",
        selectedBestModel: "ma",
        startTime: "2026-03-07 08:15:00",
        lastActivityAt: "2026-03-07 09:14:00",
        completionTime: "2026-03-07 09:14:00",
        grade: buildGrade(87, 88),
        report: {
          status: "graded",
          submittedAt: "2026-03-07 09:20:00",
          grade: 88,
          feedback: "步骤完整，细节处理较好。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>多班级高分班报告 C</h1>",
        },
      },
    },
    {
      userId: 20247005,
      username: "20247005",
      password: liveStudentPassword,
      fullName: "高分班学生丁",
      email: "20247005@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.highPerforming,
      fixture: {
        experimentId: 970005,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4540,
        selectedIndustry: "智能装备",
        selectedCompany: "远海装备",
        selectedProduct: "智能夹具",
        selectedBestModel: "lstm",
        startTime: "2026-03-08 08:20:00",
        lastActivityAt: "2026-03-08 09:12:00",
        completionTime: "2026-03-08 09:12:00",
        grade: buildGrade(83, 0),
        report: {
          status: "submitted",
          submittedAt: "2026-03-08 09:18:00",
          grade: null,
          feedback: null,
          gradedBy: null,
          useReportPdf: true,
          content: "<h1>多班级高分班待评阅报告</h1>",
        },
      },
    },
    {
      userId: 20247006,
      username: "20247006",
      password: liveStudentPassword,
      fullName: "高分班学生戊",
      email: "20247006@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.highPerforming,
      fixture: {
        experimentId: 970006,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4410,
        selectedIndustry: "高端制造",
        selectedCompany: "星跃制造",
        selectedProduct: "精密减速机",
        selectedBestModel: "arima",
        startTime: "2026-03-09 08:00:00",
        lastActivityAt: "2026-03-09 09:05:00",
        completionTime: "2026-03-09 09:05:00",
        grade: buildGrade(78, 80),
        report: {
          status: "graded",
          submittedAt: "2026-03-09 09:10:00",
          grade: 80,
          feedback: "整体稳定，仍可补强结果解释。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>多班级高分班报告 D</h1>",
        },
      },
    },
    {
      userId: 20247007,
      username: "20247007",
      password: liveStudentPassword,
      fullName: "混合班学生甲",
      email: "20247007@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.mixedState,
      fixture: {
        experimentId: 970007,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4300,
        selectedIndustry: "医疗器械",
        selectedCompany: "宁康医疗",
        selectedProduct: "监护仪模块",
        selectedBestModel: "exp",
        startTime: "2026-03-10 08:10:00",
        lastActivityAt: "2026-03-10 09:03:00",
        completionTime: "2026-03-10 09:03:00",
        grade: buildGrade(82, 84),
        report: {
          status: "graded",
          submittedAt: "2026-03-10 09:08:00",
          grade: 84,
          feedback: "结果解释较完整。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>混合班已评分报告 A</h1>",
        },
      },
    },
    {
      userId: 20247008,
      username: "20247008",
      password: liveStudentPassword,
      fullName: "混合班学生乙",
      email: "20247008@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.mixedState,
      fixture: {
        experimentId: 970008,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4010,
        selectedIndustry: "物流装备",
        selectedCompany: "畅联物流",
        selectedProduct: "分拣设备",
        selectedBestModel: "ma",
        startTime: "2026-03-11 08:30:00",
        lastActivityAt: "2026-03-11 09:10:00",
        completionTime: "2026-03-11 09:10:00",
        grade: buildGrade(67, 69),
        report: {
          status: "graded",
          submittedAt: "2026-03-11 09:15:00",
          grade: 69,
          feedback: "达到要求，但细节仍可提升。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>混合班已评分报告 B</h1>",
        },
      },
    },
    {
      userId: 20247009,
      username: "20247009",
      password: liveStudentPassword,
      fullName: "混合班学生丙",
      email: "20247009@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.mixedState,
      fixture: {
        experimentId: 970009,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 3880,
        selectedIndustry: "新材料",
        selectedCompany: "蓝峰材料",
        selectedProduct: "特种薄膜",
        selectedBestModel: "lstm",
        startTime: "2026-03-12 08:05:00",
        lastActivityAt: "2026-03-12 08:56:00",
        completionTime: "2026-03-12 08:56:00",
        grade: buildGrade(74, 0),
        report: {
          status: "rejected",
          submittedAt: "2026-03-12 09:01:00",
          grade: null,
          feedback: "报告缺少关键图表，请补充后重提。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>混合班已驳回报告</h1>",
        },
      },
    },
    {
      userId: 20247010,
      username: "20247010",
      password: liveStudentPassword,
      fullName: "混合班学生丁",
      email: "20247010@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.mixedState,
      fixture: {
        experimentId: 970010,
        status: "In Progress",
        highestCompletedStep: 4,
        currentStep: 5,
        totalActiveDurationSeconds: 2400,
        selectedIndustry: "绿色能源",
        selectedCompany: "辰光储能",
        selectedProduct: "储能变流器",
        selectedBestModel: "exp",
        startTime: "2026-03-13 08:15:00",
        lastActivityAt: "2026-03-13 09:00:00",
        completionTime: null,
      },
    },
    {
      userId: 20247011,
      username: "20247011",
      password: liveStudentPassword,
      fullName: "混合班学生戊",
      email: "20247011@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.mixedState,
    },
    {
      userId: 20247017,
      username: "20247017",
      password: liveStudentPassword,
      fullName: "混合班学生己",
      email: "20247017@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.mixedState,
      fixture: {
        experimentId: 970017,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 3950,
        selectedIndustry: "新材料",
        selectedCompany: "蓝峰材料",
        selectedProduct: "高分子薄膜",
        selectedBestModel: "exp",
        startTime: "2026-03-12 08:20:00",
        lastActivityAt: "2026-03-12 09:08:00",
        completionTime: "2026-03-12 09:08:00",
        grade: buildGrade(72, 0),
        report: {
          status: "rejected",
          submittedAt: "2026-03-12 09:14:00",
          grade: null,
          feedback: "结论部分缺少对异常波动的解释，请补充后重提。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>混合班额外已驳回报告</h1>",
        },
      },
    },
    {
      userId: USER_IDS.liveStudent,
      username: liveStudentUsername,
      password: liveStudentPassword,
      fullName: "联动班学生甲",
      email: `${liveStudentUsername}@teacher-multiclass.test.local`,
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.linkedFlow,
    },
    {
      userId: 20247012,
      username: "20247012",
      password: liveStudentPassword,
      fullName: "联动班学生乙",
      email: "20247012@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.linkedFlow,
      fixture: {
        experimentId: 970012,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4630,
        selectedIndustry: "智慧家电",
        selectedCompany: "启衡电器",
        selectedProduct: "智能净化器",
        selectedBestModel: "arima",
        startTime: "2026-03-14 08:10:00",
        lastActivityAt: "2026-03-14 09:16:00",
        completionTime: "2026-03-14 09:16:00",
        grade: buildGrade(88, 89),
        report: {
          status: "graded",
          submittedAt: "2026-03-14 09:21:00",
          grade: 89,
          feedback: "结果说明清楚，表达规范。",
          gradedBy: USER_IDS.teacher,
          useReportPdf: true,
          content: "<h1>联动班已评分报告</h1>",
        },
      },
    },
    {
      userId: 20247013,
      username: "20247013",
      password: liveStudentPassword,
      fullName: "联动班学生丙",
      email: "20247013@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.linkedFlow,
      fixture: {
        experimentId: 970013,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4480,
        selectedIndustry: "智慧家电",
        selectedCompany: "启衡电器",
        selectedProduct: "空气炸锅",
        selectedBestModel: "ma",
        startTime: "2026-03-15 08:00:00",
        lastActivityAt: "2026-03-15 09:01:00",
        completionTime: "2026-03-15 09:01:00",
        grade: buildGrade(84, 0),
        report: {
          status: "submitted",
          submittedAt: "2026-03-15 09:06:00",
          grade: null,
          feedback: null,
          gradedBy: null,
          useReportPdf: true,
          content: "<h1>联动班待评阅报告</h1>",
        },
      },
    },
    {
      userId: 20247014,
      username: "20247014",
      password: liveStudentPassword,
      fullName: "联动班学生丁",
      email: "20247014@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.linkedFlow,
      fixture: {
        experimentId: 970014,
        status: "In Progress",
        highestCompletedStep: 4,
        currentStep: 5,
        totalActiveDurationSeconds: 2220,
        selectedIndustry: "消费电子",
        selectedCompany: "明远电子",
        selectedProduct: "蓝牙耳机",
        selectedBestModel: "exp",
        startTime: "2026-03-16 08:20:00",
        lastActivityAt: "2026-03-16 08:58:00",
        completionTime: null,
      },
    },
    {
      userId: 20247015,
      username: "20247015",
      password: liveStudentPassword,
      fullName: "联动班学生戊",
      email: "20247015@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.linkedFlow,
      fixture: {
        experimentId: 970015,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4390,
        selectedIndustry: "消费电子",
        selectedCompany: "明远电子",
        selectedProduct: "智能音箱",
        selectedBestModel: "exp",
        startTime: "2026-03-16 08:05:00",
        lastActivityAt: "2026-03-16 09:02:00",
        completionTime: "2026-03-16 09:02:00",
        grade: buildGrade(79, 0),
        report: {
          status: "submitted",
          submittedAt: "2026-03-16 09:08:00",
          grade: null,
          feedback: null,
          gradedBy: null,
          useReportPdf: true,
          content: "<h1>联动班助教待评阅报告</h1>",
        },
      },
    },
    {
      userId: 20247016,
      username: "20247016",
      password: liveStudentPassword,
      fullName: "联动班学生己",
      email: "20247016@teacher-multiclass.test.local",
      phoneNumber: null,
      role: "Student",
      classId: CLASS_IDS.linkedFlow,
      fixture: {
        experimentId: 970016,
        status: "Completed",
        highestCompletedStep: 7,
        currentStep: 7,
        totalActiveDurationSeconds: 4280,
        selectedIndustry: "消费电子",
        selectedCompany: "明远电子",
        selectedProduct: "智能手表",
        selectedBestModel: "ma",
        startTime: "2026-03-16 08:12:00",
        lastActivityAt: "2026-03-16 09:06:00",
        completionTime: "2026-03-16 09:06:00",
        grade: buildGrade(76, 0),
        report: {
          status: "submitted",
          submittedAt: "2026-03-16 09:12:00",
          grade: null,
          feedback: null,
          gradedBy: null,
          useReportPdf: true,
          content: "<h1>联动班助教待驳回报告</h1>",
        },
      },
    },
  ];
}

const CLASSES: ClassSeed[] = [
  {
    classId: CLASS_IDS.highPerforming,
    className: "E2E教师多班级-高分班",
    classCode: "E2E-TM-8601",
    assistantIds: [],
  },
  {
    classId: CLASS_IDS.mixedState,
    className: "E2E教师多班级-混合班",
    classCode: "E2E-TM-8602",
    assistantIds: [USER_IDS.assistant],
  },
  {
    classId: CLASS_IDS.linkedFlow,
    className: "E2E教师多班级-联动班",
    classCode: "E2E-TM-8603",
    assistantIds: [USER_IDS.assistant],
  },
];

const STUDENTS = buildStudents();

async function cleanupTargetData(conn: Connection) {
  const studentIds = STUDENTS.map((student) => student.userId);
  const classIds = CLASSES.map((classSeed) => classSeed.classId);
  const userIds = [TEACHER.userId, ASSISTANT.userId, ...studentIds];

  if (studentIds.length > 0) {
    const studentPlaceholders = studentIds.map(() => "?").join(", ");
    await conn.query(
      `DELETE FROM quiz_answers WHERE student_id IN (${studentPlaceholders})`,
      studentIds,
    );
    await conn.query(
      `DELETE FROM experiment_reports WHERE student_id IN (${studentPlaceholders})`,
      studentIds,
    );
    await conn.query(
      `DELETE FROM experiment_grade WHERE student_id IN (${studentPlaceholders})`,
      studentIds,
    );
    await conn.query(
      `DELETE FROM experiment_step_events WHERE student_id IN (${studentPlaceholders})`,
      studentIds,
    );
    await conn.query(
      `DELETE FROM experiment_status WHERE student_id IN (${studentPlaceholders})`,
      studentIds,
    );
  }

  if (classIds.length > 0) {
    const classPlaceholders = classIds.map(() => "?").join(", ");
    await conn.query(
      `DELETE FROM class_assistants WHERE class_id IN (${classPlaceholders})`,
      classIds,
    );
    await conn.query(
      `DELETE FROM enrollments WHERE class_id IN (${classPlaceholders})`,
      classIds,
    );
    await conn.query(
      `DELETE FROM grade_weights WHERE class_id IN (${classPlaceholders})`,
      classIds,
    );
    await conn.query(
      `DELETE FROM classes WHERE class_id IN (${classPlaceholders})`,
      classIds,
    );
  }

  if (userIds.length > 0) {
    const userPlaceholders = userIds.map(() => "?").join(", ");
    await conn.query(
      `DELETE FROM users WHERE user_id IN (${userPlaceholders})`,
      userIds,
    );
  }
}

async function insertUsers(conn: Connection) {
  const users = [TEACHER, ASSISTANT, ...STUDENTS];
  const hashes = await Promise.all(
    users.map((user) => Bun.password.hash(user.password, { algorithm: "bcrypt", cost: 10 })),
  );

  for (const [index, user] of users.entries()) {
    await conn.query(
      `INSERT INTO users (
        user_id,
        username,
        password_hash,
        full_name,
        email,
        phone_number,
        role
      ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        user.userId,
        user.username,
        hashes[index],
        user.fullName,
        user.email,
        user.phoneNumber,
        user.role,
      ],
    );
  }
}

async function insertClasses(conn: Connection) {
  for (const classSeed of CLASSES) {
    await conn.query(
      "INSERT INTO classes (class_id, class_name, class_code, teacher_id) VALUES (?, ?, ?, ?)",
      [classSeed.classId, classSeed.className, classSeed.classCode, TEACHER.userId],
    );
    await conn.query(
      `INSERT INTO grade_weights (
        class_id,
        exp_flow_demand_data_preparation,
        exp_flow_demand_descriptive_stats,
        exp_flow_demand_model_selection,
        exp_flow_demand_generate_results,
        exp_flow_production_inventory_calc,
        exp_flow_production_service_level,
        exp_flow_production_variable_calc,
        exp_flow_production_plan_creation,
        exp_flow_report_submission,
        exp_flow_weight,
        knowledge_test_weight,
        model_quality_weight,
        report_quality_weight
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [classSeed.classId, 5, 5, 10, 10, 10, 5, 5, 15, 35, 40, 20, 20, 20],
    );

    for (const assistantId of classSeed.assistantIds) {
      await conn.query(
        "INSERT INTO class_assistants (assistant_id, class_id) VALUES (?, ?)",
        [assistantId, classSeed.classId],
      );
    }
  }
}

async function insertEnrollments(conn: Connection) {
  for (const student of STUDENTS) {
    await conn.query(
      "INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)",
      [student.userId, student.classId],
    );
  }
}

async function insertStepEvents(
  conn: Connection,
  fixture: ExperimentFixture,
  studentId: number,
) {
  const completedSteps = Math.max(0, Math.min(7, fixture.highestCompletedStep));
  const currentStep = Math.max(1, Math.min(7, fixture.currentStep));
  let offsetMinutes = 3;

  for (let step = 1; step <= completedSteps; step += 1) {
    const startedAt = addMinutes(fixture.startTime, offsetMinutes);
    offsetMinutes += 4;
    const completedAt = addMinutes(fixture.startTime, offsetMinutes);
    offsetMinutes += 3;

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
  student: StudentSeed,
  reportPath: string,
) {
  if (!student.fixture) {
    return;
  }

  const fixture = student.fixture;
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
      student.userId,
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

  await insertStepEvents(conn, fixture, student.userId);

  if (fixture.grade) {
    const grade = fixture.grade;
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
        student.userId,
        grade.exp_flow_demand_data_preparation,
        grade.exp_flow_demand_descriptive_stats,
        grade.exp_flow_demand_model_selection,
        grade.exp_flow_demand_generate_results,
        grade.exp_flow_production_inventory_calc,
        grade.exp_flow_production_service_level,
        grade.exp_flow_production_variable_calc,
        grade.exp_flow_production_plan_creation,
        grade.exp_flow_report_submission,
        grade.knowledge_test,
        grade.model_quality,
        grade.report_quality,
      ],
    );
  }

  if (fixture.report) {
    const report = fixture.report;
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
        student.userId,
        report.content,
        report.useReportPdf ? reportPath : null,
        report.status,
        report.submittedAt,
        report.grade,
        report.feedback,
        report.gradedBy,
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
    await cleanupTargetData(conn);
    await insertUsers(conn);
    await insertClasses(conn);
    await insertEnrollments(conn);

    for (const student of STUDENTS) {
      await insertExperimentFixture(conn, student, reportPath);
    }

    await conn.commit();
  } catch (error) {
    await conn.rollback();
    throw error;
  } finally {
    await conn.end();
  }

  console.log("Seeded teacher multiclass e2e fixture data.");
}

main().catch((error) => {
  console.error("Failed to seed teacher multiclass e2e fixture data:", error);
  process.exit(1);
});
