import { describe, expect, test } from "bun:test";
import {
  MANAGED_EXPERIMENT_UPDATE_FIELDS,
  prepareStudentExperimentUpdate,
} from "../../../src/modules/experiments/experiment-update-policy";

describe("experiment update policy", () => {
  test("removes backend-managed fields from student payloads", () => {
    const now = new Date("2026-03-08T10:00:00.000Z");
    const result = prepareStudentExperimentUpdate(
      {
        status: "In Progress",
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      },
      {
        selected_company: "企业A",
        retry_of_experiment_id: 990001,
        quiz_about_plan_completed: true,
        completion_time: "2026-03-08T10:00:00.000Z",
      },
      now,
    );

    expect("error" in result).toBeFalse();
    if ("data" in result) {
      expect(result.data.selected_company).toBe("企业A");
      expect("retry_of_experiment_id" in result.data).toBeFalse();
      expect("quiz_about_plan_completed" in result.data).toBeFalse();
      expect("completion_time" in result.data).toBeFalse();
      expect(result.data.last_activity_at).toBe(now);
      expect(result.data.total_active_duration_seconds).toBe(1920);
    }
  });

  test("promotes the first meaningful update to an in-progress experiment", () => {
    const now = new Date("2026-03-08T10:00:00.000Z");
    const result = prepareStudentExperimentUpdate(
      {
        status: "Not Started",
        start_time: null,
        last_activity_at: null,
        total_active_duration_seconds: null,
      },
      {
        selected_industry: "制造业",
      },
      now,
    );

    expect("error" in result).toBeFalse();
    if ("data" in result) {
      expect(result.data.status).toBe("In Progress");
      expect(result.data.start_time).toBe(now);
      expect(result.data.last_activity_at).toBe(now);
    }
  });

  test("accumulates activity from timezone-free database strings in the configured timezone", () => {
    const result = prepareStudentExperimentUpdate(
      {
        status: "In Progress",
        start_time: "2026-08-04 09:00:00",
        last_activity_at: "2026-08-04 09:30:00",
        total_active_duration_seconds: 120,
      },
      { selected_company: "企业B" },
      new Date("2026-08-04T10:00:00.000Z"),
      "Z",
    );

    expect(result).toMatchObject({
      data: {
        selected_company: "企业B",
        total_active_duration_seconds: 1920,
      },
    });
  });

  test("rejects updates to completed experiments", () => {
    const result = prepareStudentExperimentUpdate(
      {
        status: "Completed",
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      },
      {
        selected_product: "产品A",
      },
    );

    expect(result).toEqual({ error: "completed" });
  });

  test("documents the current set of backend-managed fields", () => {
    expect(MANAGED_EXPERIMENT_UPDATE_FIELDS.has("status")).toBeTrue();
    expect(MANAGED_EXPERIMENT_UPDATE_FIELDS.has("retry_of_experiment_id")).toBeTrue();
    expect(MANAGED_EXPERIMENT_UPDATE_FIELDS.has("quiz_about_model_completed")).toBeTrue();
    expect(MANAGED_EXPERIMENT_UPDATE_FIELDS.has("completion_time")).toBeTrue();
  });

  test("treats a lineage-only payload as having no student-writable fields", () => {
    const result = prepareStudentExperimentUpdate(
      {
        status: "In Progress",
        start_time: "2026-03-08T09:00:00.000Z",
        last_activity_at: "2026-03-08T09:30:00.000Z",
        total_active_duration_seconds: 120,
      },
      {
        retry_of_experiment_id: null,
      },
    );

    expect(result).toEqual({ error: "no_fields" });
  });
});
