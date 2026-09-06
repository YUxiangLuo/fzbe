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
const poolQuery = mock(async (_query: string, _values?: unknown[]) => [[]]);

mock.module(configModulePath, () => ({
  env: {
    DB_DATABASE: "class_grades_access_test",
    DB_HOST: "127.0.0.1",
    DB_TIMEZONE: "Z",
    E2E_ALLOW_DB_RESET: false,
  },
  pool: { query: poolQuery },
}));

const {
  buildTeacherClassSummary,
  checkClassAccess,
  getClassReports,
  getAssistantClassGradeSummaries,
  getClassExperimentRuns,
  getExperimentEvents,
  getGradeSummaries,
  getTeacherClassGradeSummaries,
} = await import(`${serviceModulePath}?class-grade-access-test`);

const weights = {
  exp_flow_demand_data_preparation: 5,
  exp_flow_demand_descriptive_stats: 5,
  exp_flow_demand_model_selection: 10,
  exp_flow_demand_generate_results: 10,
  exp_flow_production_inventory_calc: 10,
  exp_flow_production_service_level: 5,
  exp_flow_production_variable_calc: 5,
  exp_flow_production_plan_creation: 15,
  exp_flow_report_submission: 35,
  exp_flow_weight: 40,
  knowledge_test_weight: 20,
  model_quality_weight: 20,
  report_quality_weight: 20,
};

const completeGradeComponents = (score: number) => ({
  knowledge_test: score,
  model_quality: score,
  report_quality: score,
  exp_flow_demand_data_preparation: score,
  exp_flow_demand_descriptive_stats: score,
  exp_flow_demand_model_selection: score,
  exp_flow_demand_generate_results: score,
  exp_flow_production_inventory_calc: score,
  exp_flow_production_service_level: score,
  exp_flow_production_variable_calc: score,
  exp_flow_production_plan_creation: score,
  exp_flow_report_submission: score,
});

describe("class grade access and summary services", () => {
  beforeEach(() => {
    poolQuery.mockReset();
    poolQuery.mockResolvedValue([[]]);
  });

  test("distinguishes missing classes, owners, admins, assistants, and enrolled students", async () => {
    poolQuery.mockResolvedValueOnce([[]]);
    await expect(checkClassAccess(1, 9, "Teacher")).resolves.toEqual({ error: "not_found" });

    poolQuery.mockResolvedValueOnce([[{ teacher_id: 9 }]]);
    await expect(checkClassAccess(1, 9, "Teacher")).resolves.toEqual({
      allowed: true,
      teacherId: 9,
    });

    poolQuery.mockResolvedValueOnce([[{ teacher_id: 9 }]]);
    await expect(checkClassAccess(1, 77, "Admin")).resolves.toEqual({
      allowed: true,
      teacherId: 9,
    });

    poolQuery.mockResolvedValueOnce([[{ teacher_id: 9 }]]);
    poolQuery.mockResolvedValueOnce([[{ assigned: 1 }]]);
    await expect(checkClassAccess(1, 20, "Assistant")).resolves.toEqual({
      allowed: true,
      teacherId: 9,
    });

    poolQuery.mockResolvedValueOnce([[{ teacher_id: 9 }]]);
    poolQuery.mockResolvedValueOnce([[{ enrollment_id: 3 }]]);
    await expect(checkClassAccess(1, 30, "Student", true)).resolves.toEqual({
      allowed: true,
      teacherId: 9,
    });

    poolQuery.mockResolvedValueOnce([[{ teacher_id: 9 }]]);
    await expect(checkClassAccess(1, 30, "Student", false)).resolves.toEqual({
      error: "forbidden",
    });
  });

  test("selects the newest retry for progress even before it has activity timestamps", async () => {
    poolQuery.mockImplementationOnce(async (query: string, values?: unknown[]) => {
      expect(query).toContain(
        "ROW_NUMBER() OVER (PARTITION BY es.student_id ORDER BY es.experiment_id DESC)",
      );
      expect(query).not.toContain("COALESCE(es.last_activity_at");
      expect(values).toEqual([9, 9]);
      return [[{
        student_id: 3,
        username: "S003",
        full_name: "二次重做学生",
        experiment_id: 970020,
        status: "Not Started",
        current_step: 1,
        highest_completed_step: 0,
        start_time: null,
        last_activity_at: null,
        completion_time: null,
      }]];
    });
    poolQuery.mockImplementationOnce(async (query: string, values?: unknown[]) => {
      expect(query).toContain("WHERE se.experiment_id IN (?)");
      expect(values).toEqual([970020]);
      return [[]];
    });

    await expect(getExperimentEvents(9)).resolves.toEqual([{
      student_id: 3,
      username: "S003",
      full_name: "二次重做学生",
      experiment_id: 970020,
      status: "Not Started",
      current_step: 1,
      highest_completed_step: 0,
      start_time: null,
      last_activity_at: null,
      completion_time: null,
      timeline: [],
      steps: Array.from({ length: 7 }, (_, index) => ({
        step_order: index + 1,
        started_at: null,
        completed_at: null,
        latest_event_type: null,
      })),
    }]);
  });

  test("normalizes progress and event timestamps using the database timezone", async () => {
    poolQuery
      .mockResolvedValueOnce([[
        { user_id: 3, username: "S003", full_name: "时区学生" },
      ]])
      .mockResolvedValueOnce([[
        {
          experiment_id: 970030,
          student_id: 3,
          status: "In Progress",
          current_step: 2,
          highest_completed_step: 1,
          total_active_duration_seconds: 600,
          selected_industry: "制造业",
          selected_company: "企业A",
          selected_product: "产品A",
          start_time: "2026-08-04 09:00:00",
          last_activity_at: "2026-08-04 09:10:00",
          completion_time: null,
        },
      ]]);

    await expect(getClassExperimentRuns(9)).resolves.toMatchObject([{
      experiments: [{
        start_time: "2026-08-04T09:00:00.000Z",
        last_activity_at: "2026-08-04T09:10:00.000Z",
        completion_time: null,
      }],
    }]);

    poolQuery
      .mockResolvedValueOnce([[
        {
          student_id: 3,
          username: "S003",
          full_name: "时区学生",
          experiment_id: 970030,
          status: "In Progress",
          current_step: 2,
          highest_completed_step: 1,
          start_time: "2026-08-04 09:00:00",
          last_activity_at: "2026-08-04 09:10:00",
          completion_time: null,
        },
      ]])
      .mockResolvedValueOnce([[
        {
          event_id: 1,
          experiment_id: 970030,
          student_id: 3,
          step_order: 1,
          event_type: "STARTED",
          event_timestamp: "2026-08-04 09:01:00",
        },
      ]]);

    const events = await getExperimentEvents(9);
    expect(events[0]).toMatchObject({
      start_time: "2026-08-04T09:00:00.000Z",
      last_activity_at: "2026-08-04T09:10:00.000Z",
    });
    expect(events[0]?.timeline[0]).toMatchObject({
      event_timestamp: "2026-08-04T09:01:00.000Z",
    });
    expect(events[0]?.steps[0]).toMatchObject({
      started_at: "2026-08-04T09:01:00.000Z",
    });
  });

  test("masks stale report scores until the latest report is graded", async () => {
    poolQuery.mockResolvedValueOnce([[weights]]);
    poolQuery.mockResolvedValueOnce([[
      {
        user_id: 1,
        username: "S001",
        full_name: "已评分学生",
        experiment_id: 11,
        report_status: "graded",
        report_feedback: "很好",
        ...completeGradeComponents(100),
        model_quality: 80,
        report_quality: 90,
      },
      {
        user_id: 2,
        username: "S002",
        full_name: "已驳回学生",
        experiment_id: 12,
        report_status: "rejected",
        report_feedback: "请修改",
        knowledge_test: 100,
        model_quality: 100,
        report_quality: 100,
      },
      {
        user_id: 3,
        username: "S003",
        full_name: "重做学生",
        experiment_id: 13,
        retry_of_experiment_id: 12,
        report_status: "redo_in_progress",
        report_feedback: "必须重做",
        ...completeGradeComponents(100),
      },
      {
        user_id: 4,
        username: "S003",
        full_name: "未开始学生",
        experiment_id: null,
        report_status: null,
        report_feedback: null,
      },
    ]]);

    const result = await getGradeSummaries(9);
    expect("data" in result).toBeTrue();
    if (!("data" in result)) return;

    expect(result.data[0]).toMatchObject({
      report_status: "graded",
      report_quality: 90,
      final_score: 94,
    });
    expect(result.data[1]).toMatchObject({
      report_status: "rejected",
      report_quality: null,
      final_score: null,
      report_feedback: "请修改",
      model_quality: null,
      knowledge_test: null,
    });
    expect(result.data[1]?.exp_flow_breakdown.every((entry) =>
      entry.score === null && entry.weighted_score === null
    )).toBeTrue();
    expect(Object.values(result.data[1]?.final_score_breakdown ?? {}).every((entry) =>
      entry.score === null && entry.weighted_score === null
    )).toBeTrue();
    expect(result.data[2]).toMatchObject({
      experiment_id: 13,
      retry_of_experiment_id: 12,
      report_status: "redo_in_progress",
      report_feedback: "必须重做",
      exp_flow_score: null,
      model_quality: null,
      knowledge_test: null,
      report_quality: null,
      final_score: null,
    });
    expect(result.data[2]).not.toHaveProperty("rejected_report_id");
    expect(result.data[3]).toMatchObject({
      experiment_id: null,
      final_score: null,
    });

    const gradeSelectionSql = String(poolQuery.mock.calls[1]?.[0]);
    expect(gradeSelectionSql).toContain("latest.retry_of_experiment_id");
    expect(gradeSelectionSql).toContain("LEFT JOIN experiment_reports parent_er");
    expect(gradeSelectionSql).toContain("THEN 'redo_in_progress'");
  });

  test("returns retry experiments without reports as redoing with parent rejection context", async () => {
    poolQuery.mockImplementationOnce(async (query: string, values?: unknown[]) => {
      expect(query).toContain(
        "LEFT JOIN experiment_status parent ON parent.experiment_id = latest.retry_of_experiment_id AND parent.student_id = latest.student_id",
      );
      expect(query).toContain("LEFT JOIN experiment_reports parent_er");
      expect(query).toContain("LEFT JOIN users parent_grader");
      expect(query).toContain("THEN 'redo_in_progress'");
      expect(query).toContain("er.graded_by AS review_graded_by");
      expect(query).toContain("eg.report_quality AS grade_report_quality");
      expect(query).not.toContain("AS rejected_report_id");
      expect(values).toEqual([9, 9]);
      return [[{
        user_id: 3,
        username: "S003",
        full_name: "重做学生",
        report_id: null,
        experiment_id: 13,
        retry_of_experiment_id: 12,
        status: "redo_in_progress",
        submitted_at: null,
        pdf_file_path: null,
        grade: null,
        feedback: "必须完整重做",
        grader_name: "李老师",
        rejected_report_parent_report_id: 22,
        rejected_report_experiment_id: 12,
        rejected_report_status: "rejected",
        rejected_report_feedback: "必须完整重做",
        rejected_report_submitted_at: "2026-08-01 09:30:00",
        rejected_report_pdf_file_path: "/missing/archive/archived-22.pdf",
        rejected_report_grader_name: "李老师",
        grade_model_quality: 0,
        grade_exp_flow_demand_data_preparation: 0,
        grade_exp_flow_demand_descriptive_stats: 0,
        grade_exp_flow_demand_model_selection: 0,
        grade_exp_flow_demand_generate_results: 0,
        grade_exp_flow_production_inventory_calc: 0,
        grade_exp_flow_production_service_level: 0,
        grade_exp_flow_production_variable_calc: 0,
        grade_exp_flow_production_plan_creation: 0,
        grade_exp_flow_report_submission: 0,
      }]];
    });

    const reports = await getClassReports(9);

    expect(reports).toEqual([expect.objectContaining({
      experiment_id: 13,
      retry_of_experiment_id: 12,
      report_id: null,
      status: "redo_in_progress",
      feedback: "必须完整重做",
      grader_name: "李老师",
      grade: null,
      review_revision: null,
      experiment_grade: null,
      rejected_report: {
        report_id: 22,
        experiment_id: 12,
        status: "rejected",
        feedback: "必须完整重做",
        grader_name: "李老师",
        submitted_at: "2026-08-01T09:30:00.000Z",
        pdf_file_path: "reports/archived-22.pdf",
        pdf_file_available: false,
      },
    })]);
    expect(reports[0]).not.toHaveProperty("rejected_report_id");
    expect(reports[0]).not.toHaveProperty("rejected_report_pdf_file_path");
  });

  test("keeps automatic experiment evidence once the retry report is submitted", async () => {
    poolQuery.mockResolvedValueOnce([[
      {
        user_id: 3,
        username: "S003",
        full_name: "重做学生",
        report_id: 33,
        experiment_id: 13,
        retry_of_experiment_id: 12,
        status: "submitted",
        submitted_at: "2026-08-02 10:00:00",
        pdf_file_path: null,
        grade: null,
        feedback: null,
        review_graded_by: null,
        grader_name: null,
        rejected_report_parent_report_id: 22,
        rejected_report_experiment_id: 12,
        rejected_report_status: "rejected",
        rejected_report_feedback: "必须完整重做",
        rejected_report_submitted_at: "2026-08-01 09:30:00",
        rejected_report_pdf_file_path: "/missing/archive/archived-22.pdf",
        rejected_report_grader_name: "李老师",
        grade_model_quality: 82,
        grade_exp_flow_demand_data_preparation: 100,
        grade_report_quality: null,
      },
    ]]);

    const reports = await getClassReports(9);

    expect(reports).toEqual([expect.objectContaining({
      report_id: 33,
      status: "submitted",
      submitted_at: "2026-08-02T10:00:00.000Z",
      rejected_report: {
        report_id: 22,
        experiment_id: 12,
        status: "rejected",
        feedback: "必须完整重做",
        grader_name: "李老师",
        submitted_at: "2026-08-01T09:30:00.000Z",
        pdf_file_path: "reports/archived-22.pdf",
        pdf_file_available: false,
      },
      experiment_grade: expect.objectContaining({
        model_quality: 82,
        exp_flow_demand_data_preparation: 100,
      }),
    })]);
    expect(reports[0]?.review_revision).toMatch(/^[a-f0-9]{64}$/);
  });

  test("keeps the archived rejection distinct from a graded retry report", async () => {
    poolQuery.mockResolvedValueOnce([[
      {
        user_id: 3,
        username: "S003",
        full_name: "重做学生",
        report_id: 34,
        experiment_id: 14,
        retry_of_experiment_id: 13,
        status: "graded",
        submitted_at: "2026-08-03 10:00:00",
        pdf_file_path: "/missing/current/child-34.pdf",
        grade: 91,
        feedback: "本轮已满足要求",
        review_graded_by: 91,
        grader_name: "王老师",
        rejected_report_parent_report_id: 23,
        rejected_report_experiment_id: 13,
        rejected_report_status: "rejected",
        rejected_report_feedback: "上一轮缺少完整生产计划",
        rejected_report_submitted_at: "2026-08-02 08:30:00",
        rejected_report_pdf_file_path: null,
        rejected_report_grader_name: "李老师",
        grade_model_quality: 90,
        grade_exp_flow_demand_data_preparation: 100,
        grade_report_quality: 91,
      },
    ]]);

    const reports = await getClassReports(9);

    expect(reports).toEqual([expect.objectContaining({
      report_id: 34,
      experiment_id: 14,
      retry_of_experiment_id: 13,
      status: "graded",
      grade: 91,
      feedback: "本轮已满足要求",
      grader_name: "王老师",
      submitted_at: "2026-08-03T10:00:00.000Z",
      pdf_file_path: "reports/child-34.pdf",
      rejected_report: {
        report_id: 23,
        experiment_id: 13,
        status: "rejected",
        feedback: "上一轮缺少完整生产计划",
        grader_name: "李老师",
        submitted_at: "2026-08-02T08:30:00.000Z",
        pdf_file_path: null,
        pdf_file_available: false,
      },
      experiment_grade: expect.objectContaining({ model_quality: 90 }),
    })]);
    expect(reports[0]?.review_revision).toMatch(/^[a-f0-9]{64}$/);
    expect(reports[0]).not.toHaveProperty("rejected_report_parent_report_id");
  });

  test("builds teacher class summaries from all report states and applies the term filter", async () => {
    poolQuery.mockImplementation(async (query: string, values?: unknown[]) => {
      if (query.includes("role = 'Teacher'")) return [[{ role: "Teacher" }]];
      if (query.includes("FROM classes c")) {
        expect(values).toEqual([9, 4]);
        return [[{
          class_id: 17,
          class_name: "测试班",
          term_id: 4,
          academic_year: "2026-2027",
          semester: 1,
          term_label: "2026-2027 第1学期",
          is_current_term: 1,
        }]];
      }
      if (query.includes("SELECT * FROM grade_weights")) return [[weights]];
      if (query.includes("FROM enrollments e")) {
        return [[
          { user_id: 1, report_status: "graded", ...completeGradeComponents(80) },
          { user_id: 2, report_status: "graded", ...completeGradeComponents(100) },
          { user_id: 3, report_status: "submitted", final_score: null },
          { user_id: 4, report_status: "rejected", final_score: null },
          { user_id: 5, report_status: "redo_in_progress", final_score: null },
          { user_id: 6, report_status: null, final_score: null },
        ]];
      }
      return [[]];
    });

    const result = await getTeacherClassGradeSummaries(9, 9, "Teacher", 4);
    expect(result).toEqual({
      data: [{
        class_id: 17,
        class_name: "测试班",
        term_id: 4,
        academic_year: "2026-2027",
        semester: 1,
        term_label: "2026-2027 第1学期",
        is_current_term: true,
        load_status: "ready",
        load_error: null,
        total_students: 6,
        graded_count: 2,
        submitted_count: 1,
        rejected_count: 2,
        not_submitted_count: 1,
        average_score: null,
      }],
    });

    expect(buildTeacherClassSummary({
      class_id: 17,
      class_name: "测试班",
      term_id: 4,
      academic_year: "2026-2027",
      semester: 1,
      term_label: "2026-2027 第1学期",
      is_current_term: 1,
    }, [
      { report_status: "graded", final_score: 80 },
      { report_status: "graded", final_score: 100 },
      { report_status: "submitted", final_score: null },
      { report_status: "rejected", final_score: 99 },
      { report_status: "redo_in_progress", final_score: 99 },
      { report_status: null, final_score: null },
    ])).toMatchObject({
      total_students: 6,
      graded_count: 2,
      submitted_count: 1,
      rejected_count: 2,
      not_submitted_count: 1,
      average_score: 90,
    });
  });

  for (const role of ["Teacher", "Assistant"] as const) {
    test(`${role} summaries distinguish missing policy, query failure and an empty class`, async () => {
      poolQuery.mockImplementation(async (query: string, values?: unknown[]) => {
        if (query.includes("SELECT role FROM users")) return [[{ role }]];
        if (query.includes("FROM classes c")) return [[1, 2, 3].map(class_id => ({
          class_id, class_name: `Class ${class_id}`, term_id: 4,
          academic_year: "2026-2027", semester: 1, term_label: "2026-2027 第1学期", is_current_term: 1,
        }))];
        if (query.includes("SELECT * FROM grade_weights")) return values?.[0] === 1 ? [[]] : [[weights]];
        if (query.includes("FROM enrollments e") && values?.[0] === 2) throw new Error("query unavailable");
        return [[]];
      });
      const result = role === "Teacher"
        ? await getTeacherClassGradeSummaries(9, 9, role, 4)
        : await getAssistantClassGradeSummaries(9, 9, role, 4);
      expect(result.data).toEqual([
        expect.objectContaining({ class_id: 1, load_status: "error", load_error: "missing_grading_policy", total_students: null }),
        expect.objectContaining({ class_id: 2, load_status: "error", load_error: "query_failed", total_students: null }),
        expect.objectContaining({ class_id: 3, load_status: "ready", load_error: null, total_students: 0 }),
      ]);
      for (const summary of result.data!.slice(0, 2)) {
        expect(summary).toMatchObject({ graded_count: null, submitted_count: null,
          rejected_count: null, not_submitted_count: null, average_score: null });
      }
    });
  }

  test("fails closed for cross-account summary access and invalid role targets", async () => {
    await expect(getTeacherClassGradeSummaries(9, 10, "Teacher"))
      .resolves.toEqual({ error: "forbidden" });
    await expect(getAssistantClassGradeSummaries(20, 21, "Assistant"))
      .resolves.toEqual({ error: "forbidden" });
    expect(poolQuery).not.toHaveBeenCalled();

    poolQuery.mockResolvedValueOnce([[]]);
    await expect(getTeacherClassGradeSummaries(9, 1, "Admin"))
      .resolves.toEqual({ error: "not_teacher" });
    poolQuery.mockResolvedValueOnce([[]]);
    await expect(getAssistantClassGradeSummaries(20, 1, "Admin"))
      .resolves.toEqual({ error: "not_assistant" });
  });
});
