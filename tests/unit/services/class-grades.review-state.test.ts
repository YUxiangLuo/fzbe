import path from "node:path";
import { beforeEach, describe, expect, mock, test } from "bun:test";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const serviceModulePath = path.join(
  srcRoot,
  "modules",
  "class-grades",
  "class-grades.service.ts",
);

const beginTransaction = mock(async () => {});
const commit = mock(async () => {});
const rollback = mock(async () => {});
const release = mock(() => {});
const connectionQuery = mock(async (_query: string, _values?: unknown[]) => [[]]);
const connection = {
  beginTransaction,
  commit,
  rollback,
  release,
  query: connectionQuery,
};
const getConnection = mock(async () => connection);

mock.module(configModulePath, () => ({
  env: {
    DB_DATABASE: "class_grades_review_test",
    DB_HOST: "127.0.0.1",
    E2E_ALLOW_DB_RESET: false,
  },
  pool: {
    getConnection,
    query: mock(async () => [[]]),
  },
}));

const {
  buildReportReviewRevision,
  gradeReport,
} = await import(`${serviceModulePath}?class-grade-review-state-test`);

type ReportStatus = "submitted" | "graded" | "rejected";
const REVIEW_REVISION_GRADE_FIELDS = [
  "exp_flow_demand_data_preparation",
  "exp_flow_demand_descriptive_stats",
  "exp_flow_demand_model_selection",
  "exp_flow_demand_generate_results",
  "exp_flow_production_inventory_calc",
  "exp_flow_production_service_level",
  "exp_flow_production_variable_calc",
  "exp_flow_production_plan_creation",
  "exp_flow_report_submission",
  "model_quality",
  "report_quality",
] as const;

const installSuccessfulQueryPlan = (
  initialStatus: ReportStatus,
  options: {
    latestExperimentId?: number;
    hasExperimentGrade?: boolean;
    reviewerId?: number;
    reviewerRole?: "Admin" | "Teacher" | "Assistant";
  } = {},
) => {
  const latestExperimentId = options.latestExperimentId ?? 41;
  const hasExperimentGrade = options.hasExperimentGrade ?? true;
  const reviewerId = options.reviewerId ?? 99;
  const reviewerRole = options.reviewerRole ?? "Admin";
  let currentStatus = initialStatus;
  let reportGrade: number | null = initialStatus === "graded" ? 80 : null;
  let feedback: string | null = null;
  let gradedBy: number | null = initialStatus === "graded" ? 55 : null;
  const experimentGrade: Record<string, number | null> = {
    exp_flow_demand_data_preparation: 73,
    exp_flow_demand_descriptive_stats: 74,
    exp_flow_demand_model_selection: 75,
    exp_flow_demand_generate_results: 76,
    exp_flow_production_inventory_calc: 77,
    exp_flow_production_service_level: 78,
    exp_flow_production_variable_calc: 79,
    exp_flow_production_plan_creation: 80,
    exp_flow_report_submission: 81,
    model_quality: 82,
    report_quality: reportGrade,
  };

  connectionQuery.mockImplementation(async (query: string, values?: unknown[]) => {
    if (query.startsWith("SELECT student_id FROM experiment_reports")) {
      return [[{ student_id: 7 }]];
    }
    if (query.includes("FROM users") && query.includes("FOR UPDATE")) {
      return [[
        { user_id: 7, role: "Student" },
        { user_id: reviewerId, role: reviewerRole },
      ]];
    }
    if (query.includes("FROM experiment_reports WHERE report_id = ? FOR UPDATE")) {
      return [[{
        experiment_id: 41,
        student_id: 7,
        status: currentStatus,
        grade: reportGrade,
        feedback,
        graded_by: gradedBy,
      }]];
    }
    if (query.includes("FROM experiment_status") && query.includes("ORDER BY experiment_id DESC")) {
      return [[{ experiment_id: latestExperimentId }]];
    }
    if (query.includes("FROM experiment_grade") && query.includes("FOR UPDATE")) {
      return hasExperimentGrade ? [[{ grade_id: 3, ...experimentGrade }]] : [[]];
    }
    if (query.startsWith("UPDATE experiment_grade SET report_quality = NULL")) {
      experimentGrade.report_quality = null;
      return [{ affectedRows: 1 }];
    }
    if (query.startsWith("UPDATE experiment_grade SET")) {
      const assignments = query
        .slice("UPDATE experiment_grade SET ".length, query.indexOf(" WHERE "))
        .split(", ");
      assignments.forEach((assignment, index) => {
        const field = assignment.split(" = ?")[0];
        if (field) experimentGrade[field] = values?.[index] as number | null;
      });
      return [{ affectedRows: 1 }];
    }
    if (query.startsWith("UPDATE experiment_reports SET grade = NULL")) {
      reportGrade = null;
      feedback = values?.[0] as string | null;
      currentStatus = "rejected";
      gradedBy = Number(values?.[1]);
      return [{ affectedRows: 1 }];
    }
    if (query.startsWith("UPDATE experiment_reports SET")) {
      const assignments = query
        .slice("UPDATE experiment_reports SET ".length, query.indexOf(" WHERE "))
        .split(", ");
      assignments.forEach((assignment, index) => {
        if (assignment === "grade = ?") reportGrade = Number(values?.[index]);
        if (assignment === "feedback = ?") feedback = values?.[index] as string | null;
        if (assignment === "status = ?") currentStatus = values?.[index] as ReportStatus;
        if (assignment === "graded_by = ?") gradedBy = Number(values?.[index]);
      });
      return [{ affectedRows: 1 }];
    }
    if (query.includes("FROM experiment_reports er JOIN users u")) {
      return [[{
        user_id: 7,
        username: "S007",
        full_name: "测试学生",
        report_id: 9,
        experiment_id: 41,
        status: currentStatus,
        submitted_at: null,
        pdf_file_path: null,
        grade: reportGrade,
        feedback,
        review_graded_by: gradedBy,
        grader_name: "测试教师",
      }]];
    }
    if (query.includes("FROM experiment_grade")) return [[experimentGrade]];
    return [[]];
  });

  const getReviewRevision = () => buildReportReviewRevision({
    status: currentStatus,
    grade: reportGrade,
    feedback,
    graded_by: gradedBy,
  }, hasExperimentGrade ? experimentGrade : null);

  const setReviewState = (next: {
    status?: ReportStatus;
    grade?: number | null;
    feedback?: string | null;
    gradedBy?: number | null;
    experimentGrade?: Partial<Record<string, number | null>>;
  }) => {
    if (next.status !== undefined) currentStatus = next.status;
    if (Object.hasOwn(next, "grade")) reportGrade = next.grade ?? null;
    if (Object.hasOwn(next, "feedback")) feedback = next.feedback ?? null;
    if (Object.hasOwn(next, "gradedBy")) gradedBy = next.gradedBy ?? null;
    Object.assign(experimentGrade, next.experimentGrade);
  };

  return { getReviewRevision, setReviewState };
};

describe("gradeReport review state transitions", () => {
  beforeEach(() => {
    beginTransaction.mockClear();
    commit.mockClear();
    rollback.mockClear();
    release.mockClear();
    getConnection.mockClear();
    connectionQuery.mockReset();
  });

  test("builds a deterministic revision that covers every overwritable review field", () => {
    const report: Record<string, unknown> = {
      status: "graded",
      grade: 80,
      feedback: "初次评阅",
      graded_by: 55,
    };
    const experimentGrade = Object.fromEntries(
      REVIEW_REVISION_GRADE_FIELDS.map((field, index) => [field, 70 + index]),
    );
    const revision = buildReportReviewRevision(report, experimentGrade);

    expect(revision).toMatch(/^[a-f0-9]{64}$/);
    expect(buildReportReviewRevision({
      ...report,
      grade: "80.00",
      graded_by: "55",
    }, Object.fromEntries(
      Object.entries(experimentGrade).map(([field, value]) => [field, String(value)]),
    ))).toBe(revision);

    for (const field of ["status", "grade", "feedback", "graded_by"] as const) {
      expect(buildReportReviewRevision({ ...report, [field]: `${String(report[field])}-changed` }, experimentGrade))
        .not.toBe(revision);
    }
    for (const field of REVIEW_REVISION_GRADE_FIELDS) {
      expect(buildReportReviewRevision(report, { ...experimentGrade, [field]: 99 }))
        .not.toBe(revision);
    }
  });

  test("rejects a submitted report without overwriting preserved experiment scores", async () => {
    const plan = installSuccessfulQueryPlan("submitted");

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "submitted",
      expected_review_revision: plan.getReviewRevision(),
      status: "rejected",
      feedback: "请完整重做实验",
    });

    expect("data" in result).toBeTrue();
    expect(beginTransaction).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();

    const queries = connectionQuery.mock.calls.map(([query]) => String(query));
    expect(queries[0]).toStartWith("SELECT student_id FROM experiment_reports");
    expect(queries[1]).toContain("FROM users");
    expect(queries[1]).toContain("ORDER BY user_id ASC");
    expect(connectionQuery.mock.calls[1]?.[1]).toEqual([7, 99]);
    expect(queries[2]).toContain("FROM experiment_reports WHERE report_id = ? FOR UPDATE");
    expect(queries[3]).toContain("FROM experiment_status");
    expect(queries[3]).not.toContain("FOR UPDATE");

    const lockCall = connectionQuery.mock.calls.find(([query]) =>
      String(query).includes("FROM experiment_reports WHERE report_id = ? FOR UPDATE")
    );
    expect(lockCall).toBeDefined();

    const experimentUpdate = connectionQuery.mock.calls.find(([query]) =>
      String(query).startsWith("UPDATE experiment_grade SET")
    );
    expect(experimentUpdate).toEqual([
      "UPDATE experiment_grade SET report_quality = NULL WHERE experiment_id = ? AND student_id = ?",
      [41, 7],
    ]);
    expect(String(experimentUpdate?.[0])).not.toContain("model_quality");
    expect(String(experimentUpdate?.[0])).not.toContain("exp_flow_");
    expect(String(experimentUpdate?.[0])).not.toContain("knowledge_test");

    const reportUpdate = connectionQuery.mock.calls.find(([query]) =>
      String(query).startsWith("UPDATE experiment_reports SET")
    );
    expect(reportUpdate).toEqual([
      "UPDATE experiment_reports SET grade = NULL, feedback = ?, status = 'rejected', graded_by = ? WHERE report_id = ? AND status = 'submitted'",
      ["请完整重做实验", 99, 9],
    ]);
  });

  test("treats rejected reports as immutable", async () => {
    const plan = installSuccessfulQueryPlan("rejected");

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "submitted",
      expected_review_revision: plan.getReviewRevision(),
      status: "graded",
      grade: 90,
      feedback: "迟到的评分",
    });

    expect(result).toEqual({ error: "status_conflict", current_status: "rejected" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([query]) => String(query).startsWith("UPDATE"))).toBeFalse();
  });

  test("locks the reviewer and student in primary-key order", async () => {
    const plan = installSuccessfulQueryPlan("submitted", {
      reviewerId: 3,
      reviewerRole: "Admin",
    });

    const result = await gradeReport(9, 3, "Admin", {
      expected_status: "submitted",
      expected_review_revision: plan.getReviewRevision(),
      status: "rejected",
      feedback: "请重做",
    });

    expect("data" in result).toBeTrue();
    const userLock = connectionQuery.mock.calls.find(([query]) =>
      String(query).includes("FROM users") && String(query).includes("FOR UPDATE")
    );
    expect(userLock?.[1]).toEqual([3, 7]);
  });

  test("rejects a stale token when the reviewer database role has changed", async () => {
    installSuccessfulQueryPlan("submitted", {
      reviewerRole: "Assistant",
    });

    const result = await gradeReport(9, 99, "Teacher", {
      expected_status: "submitted",
      expected_review_revision: "stale-token-does-not-reach-report",
      status: "graded",
      grade: 90,
    });

    expect(result).toEqual({ error: "forbidden" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(connectionQuery.mock.calls.some(([query]) =>
      String(query).includes("FROM experiment_reports WHERE report_id = ? FOR UPDATE")
    )).toBeFalse();
  });

  test("forbids converting a graded report into a rejection", async () => {
    const plan = installSuccessfulQueryPlan("graded");

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "graded",
      expected_review_revision: plan.getReviewRevision(),
      status: "rejected",
      feedback: "不能用普通驳回撤销成绩",
    });

    expect(result).toEqual({ error: "status_conflict", current_status: "graded" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([query]) => String(query).startsWith("UPDATE"))).toBeFalse();
  });

  test("refuses to review a report from a superseded experiment", async () => {
    const plan = installSuccessfulQueryPlan("submitted", { latestExperimentId: 42 });

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "submitted",
      expected_review_revision: plan.getReviewRevision(),
      status: "rejected",
      feedback: "旧尝试不能再变更",
    });

    expect(result).toEqual({ error: "status_conflict", current_status: "submitted" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([query]) => String(query).startsWith("UPDATE"))).toBeFalse();
  });

  test("allows an existing graded report to be revised without changing its state", async () => {
    const plan = installSuccessfulQueryPlan("graded");
    const openedRevision = plan.getReviewRevision();

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "graded",
      expected_review_revision: openedRevision,
      status: "graded",
      grade: 91,
      feedback: "复核后修订",
      experiment_grade: { model_quality: 88 },
    });

    expect("data" in result).toBeTrue();
    expect(commit).toHaveBeenCalledTimes(1);
    expect(rollback).not.toHaveBeenCalled();

    const experimentUpdate = connectionQuery.mock.calls.find(([query]) =>
      String(query).startsWith("UPDATE experiment_grade SET")
    );
    expect(experimentUpdate).toEqual([
      "UPDATE experiment_grade SET model_quality = ?, report_quality = ? WHERE experiment_id = ? AND student_id = ?",
      [88, 91, 41, 7],
    ]);

    const reportUpdate = connectionQuery.mock.calls.find(([query]) =>
      String(query).startsWith("UPDATE experiment_reports SET")
    );
    expect(reportUpdate?.[0]).toContain("WHERE report_id = ? AND status = ?");
    expect(reportUpdate?.[1]).toEqual([91, "复核后修订", "graded", 99, 9, "graded"]);
    if ("data" in result) {
      expect(result.data.review_revision).toBe(plan.getReviewRevision());
      expect(result.data.review_revision).not.toBe(openedRevision);
    }
  });

  test("rejects a stale submitted review after another reviewer changes its review fields", async () => {
    const plan = installSuccessfulQueryPlan("submitted");
    const staleRevision = plan.getReviewRevision();
    plan.setReviewState({
      feedback: "助教已经保存的新评语",
      gradedBy: 56,
      experimentGrade: { model_quality: 89 },
    });

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "submitted",
      expected_review_revision: staleRevision,
      status: "graded",
      grade: 92,
      feedback: "来自过期待评分页面的覆盖",
    });

    expect(result).toEqual({ error: "status_conflict", current_status: "submitted" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([query]) => String(query).startsWith("UPDATE"))).toBeFalse();
  });

  test("rejects a stale graded revision without issuing any update", async () => {
    const plan = installSuccessfulQueryPlan("graded");
    const staleRevision = plan.getReviewRevision();
    plan.setReviewState({
      grade: 82,
      feedback: "教师已经完成的新复核",
      gradedBy: 57,
      experimentGrade: { model_quality: 87, report_quality: 82 },
    });

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "graded",
      expected_review_revision: staleRevision,
      status: "graded",
      grade: 72,
      feedback: "来自旧页面的覆盖",
      experiment_grade: { model_quality: 70 },
    });

    expect(result).toEqual({ error: "status_conflict", current_status: "graded" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([query]) => String(query).startsWith("UPDATE"))).toBeFalse();
  });

  test("writes null when explicitly clearing an optional experiment grade", async () => {
    const plan = installSuccessfulQueryPlan("graded");

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "graded",
      expected_review_revision: plan.getReviewRevision(),
      grade: 91,
      status: "graded",
      experiment_grade: { model_quality: null },
    });

    expect("data" in result).toBeTrue();
    const experimentUpdate = connectionQuery.mock.calls.find(([query]) =>
      String(query).startsWith("UPDATE experiment_grade SET")
    );
    expect(experimentUpdate).toEqual([
      "UPDATE experiment_grade SET model_quality = ?, report_quality = ? WHERE experiment_id = ? AND student_id = ?",
      [null, 91, 41, 7],
    ]);
    if ("data" in result) {
      expect(result.data.experiment_grade?.model_quality).toBeNull();
    }
  });

  test("rolls back rejection when the experiment grade row is missing", async () => {
    const plan = installSuccessfulQueryPlan("submitted", { hasExperimentGrade: false });

    const result = await gradeReport(9, 99, "Admin", {
      expected_status: "submitted",
      expected_review_revision: plan.getReviewRevision(),
      status: "rejected",
      feedback: "请重做",
    });

    expect(result).toEqual({ error: "grade_not_found" });
    expect(rollback).toHaveBeenCalledTimes(1);
    expect(commit).not.toHaveBeenCalled();
    expect(connectionQuery.mock.calls.some(([query]) =>
      String(query).startsWith("UPDATE experiment_reports SET")
    )).toBeFalse();
  });
});
