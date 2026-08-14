import { describe, expect, test } from "bun:test";
import path from "node:path";
import { parse } from "yaml";

type SchemaRecord = Record<string, any>;

const openApiPath = path.resolve(import.meta.dir, "../../../docs/openapi.yaml");
const document = parse(await Bun.file(openApiPath).text()) as SchemaRecord;
const schemas = document.components.schemas as SchemaRecord;

describe("OpenAPI model contracts", () => {
  test("documents honest durable manual deletion outcomes", () => {
    const responses = document.paths["/api/v1/manuals/{manualId}"].delete.responses;
    expect(responses["200"].content["application/json"].schema.$ref)
      .toBe("#/components/schemas/ManagedFileDeletionResponse");
    expect(responses["500"].description).toContain("数据库删除未执行");
    expect(schemas.ManagedFileDeletionResponse.required).toEqual(["data"]);
    expect(schemas.ManagedFileDeletionResponse.properties.data.required)
      .toEqual(["message", "cleanup_pending"]);
    expect(schemas.ManagedFileDeletionResponse.properties.data.properties.cleanup_pending.type)
      .toBe("boolean");
  });

  test("documents explicit target-only LSTM features and conditional ensemble requirements", () => {
    const standaloneArray = schemas.LstmTrainRequest.allOf[1].properties.lstmFeatures.oneOf[1];
    expect(standaloneArray.minItems).toBeUndefined();
    expect(standaloneArray.maxItems).toBe(64);

    const withLstm = schemas.EnsembleWithLstmTrainRequest.allOf[1];
    expect(withLstm.required).toEqual([
      "models",
      "lstmFeatures",
      "lstmTargetFeature",
      "lstmNormalization",
    ]);
    expect(withLstm.properties.lstmFeatures.oneOf[1].minItems).toBeUndefined();
    expect(withLstm.properties.lstmFeatures.oneOf[1].maxItems).toBe(64);
    expect(withLstm.properties.lstmFeatures.description).toContain("最多 64 个");

    const containsLstm = new RegExp(withLstm.properties.models.pattern);
    const withoutLstm = schemas.EnsembleWithoutLstmTrainRequest.allOf[1];
    const excludesLstm = new RegExp(withoutLstm.properties.models.not.pattern);
    expect(containsLstm.test("ma, lstm")).toBe(true);
    expect(containsLstm.test("ma,es")).toBe(false);
    expect(excludesLstm.test("ma,lstm")).toBe(true);
  });

  test("documents the wrapped production prediction uncertainty contract", () => {
    const point = schemas.ProductionPredictionPoint;
    expect(point.required).toEqual(expect.arrayContaining([
      "prediction",
      "std_dev",
      "upper_error_p99",
      "uncertainty_source",
      "calibration_mean_error",
      "calibration_count",
      "interval_lower",
      "interval_upper",
      "interval_level",
      "interval_kind",
      "coverage_claim",
    ]));
    expect(point.properties.uncertainty_source.enum).toEqual(["model", "empirical", "fallback"]);
    expect(point.properties.interval_level.enum).toEqual([0.95]);
    expect(point.properties.coverage_claim.enum).toEqual(["model_nominal", "none"]);
    expect(point.oneOf).toEqual([
      expect.objectContaining({
        properties: expect.objectContaining({
          interval_kind: expect.objectContaining({ enum: ["model_prediction_interval"] }),
          coverage_claim: expect.objectContaining({ enum: ["model_nominal"] }),
        }),
      }),
      expect.objectContaining({
        properties: expect.objectContaining({
          interval_kind: expect.objectContaining({
            enum: [
              "empirical_residual_quantile",
              "normal_approximation",
              "fallback_normal_approximation",
            ],
          }),
          coverage_claim: expect.objectContaining({ enum: ["none"] }),
        }),
      }),
    ]);
    expect(point.properties.upper_error_p99.description).toContain("不读取该字段");
    expect(schemas.ProductionPredictionResults.properties.implementation_notes)
      .toEqual({ type: "array", items: { type: "string" } });
    expect(schemas.ProductionPredictionResults.required)
      .toEqual(["predictions", "forecast_evidence"]);
    expect(schemas.ProductionPredictionResults.properties.forecast_evidence)
      .toMatchObject({ type: "string", minLength: 1, maxLength: 4096 });
    expect(
      document.paths["/api/v1/models/{model_type}/predict"].post.responses["200"]
        .content["application/json"].schema.$ref,
    ).toBe("#/components/schemas/ModelPredictionResponse");
    expect(schemas.ModelPredictionResponse.required).toEqual(["data"]);
  });

  test("uses the persistable 2-12 period horizon across production model APIs", () => {
    expect(schemas.ModelPrepareProductionRequest.properties.forecast_steps)
      .toMatchObject({ type: "integer", minimum: 2, maximum: 12 });
    expect(schemas.ModelPredictRequest.properties.forecast_steps)
      .toMatchObject({ type: "integer", minimum: 2, maximum: 12 });
    expect(schemas.ProductionPreparationSessionResponsePayload.properties.results.properties
      .prepared_forecast_steps)
      .toMatchObject({ type: "integer", minimum: 2, maximum: 12 });
    expect(schemas.ProductionPredictionResults.properties.predictions)
      .toMatchObject({ type: "array", minItems: 2, maxItems: 12 });
  });

  test("documents the production retraining recovery contract", () => {
    expect(schemas.ErrorResponse.properties.required_action.enum)
      .toContain("retrain");
    expect(schemas.ErrorResponse.properties.model_type.enum).toEqual([
      "ma",
      "es",
      "arima",
      "lstm",
      "weighted_avg",
      "boosting",
      "stacking",
    ]);
    expect(
      document.paths["/api/v1/models/{model_type}/production-preparation/sessions"]
        .post.responses["400"].description,
    ).toContain("MODEL_TRAINING_RUN_REQUIRED");
    expect(
      document.paths["/api/v1/models/{model_type}/production-preparation/sessions"]
        .post.responses["400"].description,
    ).toContain("MODEL_RETRAINING_REQUIRED");
    expect(
      document.paths[
        "/api/v1/models/{model_type}/production-preparation/sessions/{session_id}/steps/{step_id}/run"
      ].post.responses["400"].description,
    ).toContain("MODEL_ARTIFACT_INVALID");
    expect(
      document.paths["/api/v1/models/{model_type}/predict"]
        .post.responses["400"].description,
    ).toContain("MODEL_RETRAINING_REQUIRED");
  });

  test("documents the application-level envelope for model observability", () => {
    expect(
      document.paths["/api/v1/models/experiments/{experimentId}/observability"].get
        .responses["200"].content["application/json"].schema.$ref,
    ).toBe("#/components/schemas/ModelObservabilityResponse");
    expect(schemas.ModelObservabilityResponse.required).toEqual(["data"]);
    expect(schemas.ModelObservabilityResponse.properties.data.$ref)
      .toBe("#/components/schemas/ModelObservabilityResponsePayload");
    expect(schemas.ModelObservabilityResponsePayload.required).toEqual(["models"]);
  });

  test("documents the application-level envelopes for guided and production sessions", () => {
    const guidedResponseRef = "#/components/schemas/GuidedTrainingSessionResponse";
    const guidedOperations = [
      document.paths["/api/v1/models/{model_type}/guided-training/sessions"].post,
      document.paths["/api/v1/models/{model_type}/guided-training/sessions/{session_id}"].get,
      document.paths[
        "/api/v1/models/{model_type}/guided-training/sessions/{session_id}/steps/{step_id}/run"
      ].post,
    ];
    for (const operation of guidedOperations) {
      expect(operation.responses["200"].content["application/json"].schema.$ref)
        .toBe(guidedResponseRef);
    }
    expect(schemas.GuidedTrainingSessionResponse.required).toEqual(["data"]);
    expect(schemas.GuidedTrainingSessionResponse.properties.data.$ref)
      .toBe("#/components/schemas/GuidedTrainingSession");
    expect(schemas.GuidedTrainingSession.required).toEqual([
      "session_id",
      "experiment_id",
      "model_type",
      "status",
      "current_step_id",
      "next_step_id",
      "steps",
      "step_outputs",
      "result",
      "error_message",
      "error_code",
      "experiment_state_version",
    ]);

    const preparationResponseRef =
      "#/components/schemas/ProductionPreparationSessionResponse";
    const preparationOperations = [
      document.paths["/api/v1/models/{model_type}/production-preparation/sessions"].post,
      document.paths[
        "/api/v1/models/{model_type}/production-preparation/sessions/{session_id}"
      ].get,
      document.paths[
        "/api/v1/models/{model_type}/production-preparation/sessions/{session_id}/steps/{step_id}/run"
      ].post,
    ];
    for (const operation of preparationOperations) {
      expect(operation.responses["200"].content["application/json"].schema.$ref)
        .toBe(preparationResponseRef);
    }
    expect(schemas.ProductionPreparationSessionResponse.required).toEqual(["data"]);
    expect(schemas.ProductionPreparationSessionResponse.properties.data.$ref)
      .toBe("#/components/schemas/ProductionPreparationSessionResponsePayload");
    expect(schemas.ProductionPreparationSessionResponsePayload.required)
      .toEqual(["status", "results"]);
  });

  test("documents read-only legacy forecasts separately from strict v2 writes", () => {
    const legacyPoint = schemas.LegacyProductionPredictionPoint;
    expect(legacyPoint.required).toEqual(["prediction", "std_dev"]);
    expect(legacyPoint.not.anyOf).toEqual([
      { properties: { interval_lower: {} }, required: ["interval_lower"] },
      { properties: { interval_upper: {} }, required: ["interval_upper"] },
      { properties: { interval_level: {} }, required: ["interval_level"] },
      { properties: { interval_kind: {} }, required: ["interval_kind"] },
      { properties: { coverage_claim: {} }, required: ["coverage_claim"] },
    ]);

    const persisted = schemas.CompatiblePersistedProductionForecastResults.oneOf;
    expect(persisted[0].nullable).toBe(true);
    expect(persisted[0].minItems).toBe(1);
    expect(persisted[0].items.$ref)
      .toBe("#/components/schemas/LegacyProductionPredictionPoint");
    expect(persisted[1].items.$ref)
      .toBe("#/components/schemas/ProductionPredictionPoint");
    expect(
      schemas.ExperimentStateRecord.properties.production_forecast_results.$ref,
    ).toBe("#/components/schemas/CompatiblePersistedProductionForecastResults");

    const writable = schemas.ExperimentStudentWritableFields.properties;
    expect(writable.production_forecast_periods.minimum).toBe(2);
    expect(writable.production_forecast_periods.maximum).toBe(12);
    expect(writable.production_forecast_results.type).toBe("array");
    expect(writable.production_forecast_results.nullable).toBe(true);
    expect(writable.production_forecast_results.maxItems).toBe(12);
    expect(writable.production_forecast_results.items.$ref)
      .toBe("#/components/schemas/ProductionPredictionPoint");
    expect(writable.production_forecast_evidence)
      .toMatchObject({ type: "string", nullable: true, maxLength: 4096 });
    expect(schemas.ExperimentStateRecord.properties.production_forecast_evidence)
      .toMatchObject({ type: "string", nullable: true, maxLength: 4096 });
    expect(schemas.ExperimentStateRecord.properties.retry_of_experiment_id)
      .toMatchObject({ type: "integer", minimum: 1, nullable: true });
    expect(schemas.ExperimentRetryRequest).toMatchObject({
      required: ["rejected_report_id"],
      additionalProperties: false,
    });
    expect(schemas.ExperimentRetryRequest.properties.rejected_report_id)
      .toMatchObject({ type: "integer", minimum: 1 });
    expect(schemas.ExperimentCreateRequest.properties.production_forecast_results).toBeUndefined();
  });

  test("documents optimistic report review writes and nullable experiment components", () => {
    const request = schemas.GradeReportRequest;
    expect(request.required).toEqual(["expected_status", "expected_review_revision"]);
    expect(request.properties.expected_status).toMatchObject({
      type: "string",
      enum: ["submitted", "graded"],
    });
    expect(request.properties.expected_review_revision).toMatchObject({
      type: "string",
      pattern: "^[a-f0-9]{64}$",
    });
    expect(request.description).toContain("两个从 graded 打开的页面");
    expect(request.description).toContain("status=rejected 时 expected_status 必须为");
    expect(request.properties.feedback.description).toContain("空白字符串会清除已有评语");
    expect(request.properties.grade.nullable).toBeUndefined();
    expect(request.properties.experiment_grade.minProperties).toBe(1);
    expect(request.properties.experiment_grade.description).toContain("传 null 表示清除");

    for (const component of Object.values(
      request.properties.experiment_grade.properties,
    ) as SchemaRecord[]) {
      expect(component).toMatchObject({
        type: "number",
        nullable: true,
        minimum: 0,
        maximum: 100,
      });
    }

    const conflictResponse = document.paths["/api/v1/reports/{reportId}"].put.responses["409"];
    expect(conflictResponse.description).toContain("expected_status");
    expect(conflictResponse.description).toContain("expected_review_revision");
    expect(conflictResponse.content["application/json"].schema.$ref)
      .toBe("#/components/schemas/GradeReportConflictResponse");
    expect(schemas.GradeReportConflictResponse).toMatchObject({
      required: ["error", "current_status"],
      additionalProperties: false,
      properties: {
        current_status: {
          type: "string",
          enum: ["submitted", "graded", "rejected"],
        },
      },
    });

    const classReport = schemas.ClassReportRecord;
    expect(classReport.required).toContain("review_revision");
    expect(classReport.properties.review_revision).toMatchObject({
      type: "string",
      pattern: "^[a-f0-9]{64}$",
      nullable: true,
      readOnly: true,
    });

    const successResponse = document.paths["/api/v1/reports/{reportId}"].put.responses["200"];
    expect(successResponse.content["application/json"].schema.$ref)
      .toBe("#/components/schemas/GradeReportSuccessResponse");
    expect(schemas.GradeReportSuccessResponse.properties.data).toMatchObject({
      required: ["review_revision"],
      properties: {
        review_revision: {
          type: "string",
          pattern: "^[a-f0-9]{64}$",
          readOnly: true,
        },
      },
    });
  });

  test("documents strict MPS rows and strongly typed experiment responses", () => {
    const mpsRow = schemas.ProductionMpsRow;
    expect(mpsRow.required).toEqual([
      "period",
      "period_label",
      "demand_forecast",
      "safety_stock",
      "planned_production",
      "beginning_inventory",
      "production_output",
      "ending_inventory",
      "stockout",
      "service_level",
    ]);
    expect(mpsRow.additionalProperties).toBe(false);
    expect(schemas.ExperimentStudentWritableFields.properties.production_mps_table.items.$ref)
      .toBe("#/components/schemas/ProductionMpsRow");

    const responseRefs = [
      document.paths["/api/v1/students/me/experiment-runs/active"].get
        .responses["200"].content["application/json"].schema.$ref,
      document.paths["/api/v1/students/me/experiment-runs"].post
        .responses["201"].content["application/json"].schema.$ref,
      document.paths["/api/v1/experiment-runs/{experimentId}"].patch
        .responses["200"].content["application/json"].schema.$ref,
      document.paths["/api/v1/students/me/experiment-runs/retry"].post
        .responses["200"].content["application/json"].schema.$ref,
      document.paths["/api/v1/students/me/experiment-runs/retry"].post
        .responses["201"].content["application/json"].schema.$ref,
    ];
    expect(responseRefs).toEqual(Array(5).fill(
      "#/components/schemas/ExperimentStateResponse",
    ));
    const retryOperation = document.paths["/api/v1/students/me/experiment-runs/retry"].post;
    expect(retryOperation.description).toContain("尚未生成报告");
    expect(retryOperation.description).toContain("Completed 但尚无报告");
    expect(retryOperation.description).toContain("submitted、graded 或 rejected");
    expect(retryOperation.responses["200"].description).toContain("Completed 无报告恢复态");
    expect(retryOperation.responses["409"].description).toContain("child 已有报告");
    expect(retryOperation.responses["409"].content["application/json"].schema.$ref)
      .toBe("#/components/schemas/ExperimentRetryConflictResponse");
    expect(schemas.ExperimentRetryConflictCurrent).toMatchObject({
      required: [
        "experiment_id",
        "student_id",
        "state_version",
        "status",
        "current_step",
        "highest_completed_step",
      ],
      additionalProperties: true,
    });
    expect(schemas.ExperimentRetryConflictCurrent.description).toContain("current_step >= 8");
    expect(schemas.ExperimentRetryConflictCurrent.description).toContain("缺失或为 null");
    expect(schemas.ExperimentRetryConflictCurrent.properties.current_step).toMatchObject({
      type: "integer",
      minimum: 1,
    });
    expect(schemas.ExperimentRetryConflictCurrent.properties.current_step.maximum)
      .toBeUndefined();
    for (const quizFlag of [
      "quiz_about_model_completed",
      "quiz_about_plan_completed",
    ]) {
      expect(schemas.ExperimentRetryConflictCurrent.required).not.toContain(quizFlag);
      expect(schemas.ExperimentRetryConflictCurrent.properties[quizFlag]).toMatchObject({
        type: "boolean",
        nullable: true,
      });
    }
    expect(schemas.ExperimentRetryConflictResponse.discriminator)
      .toEqual({ propertyName: "code" });
    const [notLatestConflict, otherConflict] = schemas.ExperimentRetryConflictResponse.oneOf;
    expect(notLatestConflict).toMatchObject({
      required: ["error", "code", "current"],
      properties: {
        code: { enum: ["REJECTED_REPORT_NOT_LATEST"] },
        current: { $ref: "#/components/schemas/ExperimentRetryConflictCurrent" },
      },
      additionalProperties: false,
    });
    expect(otherConflict).toMatchObject({
      required: ["error", "code"],
      properties: {
        code: {
          enum: [
            "REPORT_NOT_REJECTED",
            "REJECTED_EXPERIMENT_NOT_COMPLETED",
            "INVALID_RETRY_LINEAGE",
          ],
        },
      },
      additionalProperties: false,
    });
    expect(otherConflict.properties.current).toBeUndefined();
    expect(
      document.paths["/api/v1/students/me/experiment-runs"].get
        .responses["200"].content["application/json"].schema.$ref,
    ).toBe("#/components/schemas/ExperimentStateArrayResponse");
    expect(schemas.ExperimentStateResponse.properties.data.$ref)
      .toBe("#/components/schemas/ExperimentStateRecord");
    expect(schemas.ExperimentStateRecord.required).toEqual([
      "experiment_id",
      "student_id",
      "status",
      "current_step",
      "highest_completed_step",
      "state_version",
    ]);
    expect(schemas.ExperimentStateArrayResponse.properties.data.items.$ref)
      .toBe("#/components/schemas/ExperimentStateRecord");

    expect(schemas.ProductionPlanErrorResponse.properties.code.enum).toEqual([
      "PRODUCTION_PLAN_INVALID",
      "PRODUCTION_PLAN_MISMATCH",
      "PRODUCTION_FORECAST_LEGACY_REJECTED",
      "PRODUCTION_FORECAST_EVIDENCE_REQUIRED",
      "PRODUCTION_FORECAST_EVIDENCE_INVALID",
      "PRODUCTION_FORECAST_EVIDENCE_STALE",
    ]);
  });

  test("documents the exact lightweight experiment returned by latest report status", () => {
    const responseSchema = document.paths["/api/v1/my-latest-report-status"].get
      .responses["200"].content["application/json"].schema;
    expect(responseSchema.$ref)
      .toBe("#/components/schemas/LatestReportStatusResponse");
    expect(schemas.LatestReportStatusResponse.properties.data.$ref)
      .toBe("#/components/schemas/LatestReportStatus");
    expect(schemas.LatestReportStatus.properties.experiment.$ref)
      .toBe("#/components/schemas/LatestReportExperimentSummary");
    expect(schemas.LatestReportExperimentSummary.required).toEqual([
      "experiment_id",
      "student_id",
      "retry_of_experiment_id",
      "status",
      "current_step",
      "start_time",
      "last_activity_at",
      "completion_time",
    ]);
    expect(Object.keys(schemas.LatestReportExperimentSummary.properties))
      .toEqual(schemas.LatestReportExperimentSummary.required);
    expect(schemas.LatestReportExperimentSummary.additionalProperties).toBe(false);
    expect(schemas.LatestReportStatus.required)
      .toEqual(["is_rejected", "has_report", "redo_in_progress"]);
    expect(schemas.LatestReportStatus.description)
      .toContain("submitted 或 graded 后仍与当前 report 同时返回");
    expect(schemas.LatestReportStatus.properties.rejected_report.allOf[0].$ref)
      .toBe("#/components/schemas/LatestReportRecord");
  });

  test("documents archived parent context without replacing the current class report id", () => {
    const responseSchema = document.paths["/api/v1/classes/{classId}/reports"].get
      .responses["200"].content["application/json"].schema;
    expect(responseSchema.$ref)
      .toBe("#/components/schemas/ClassReportArrayResponse");
    expect(schemas.ClassReportArrayResponse.properties.data.$ref)
      .toBe("#/components/schemas/ClassReportArray");
    expect(schemas.ClassReportArray.items.$ref)
      .toBe("#/components/schemas/ClassReportRecord");
    expect(schemas.ClassReportRecord.required).toContain("rejected_report");
    expect(schemas.ClassReportRecord.required).not.toContain("rejected_report_id");
    expect(schemas.ClassReportRecord.properties.rejected_report_id).toBeUndefined();
    expect(schemas.ClassReportRecord.properties.report_id.description)
      .toContain("不会借用父驳回报告 ID");
    expect(schemas.ClassReportRecord.properties.rejected_report.allOf[0].$ref)
      .toBe("#/components/schemas/ClassReportArchivedRejection");
    expect(schemas.ClassReportRecord.properties.rejected_report.description)
      .toContain("submitted 或 graded 后仍保留");
    expect(schemas.ClassReportArchivedRejection.required).toEqual([
      "report_id",
      "experiment_id",
      "status",
      "feedback",
      "grader_name",
      "submitted_at",
      "pdf_file_path",
      "pdf_file_available",
    ]);
    expect(schemas.ClassReportArchivedRejection.properties.status.enum)
      .toEqual(["rejected"]);
    expect(schemas.ClassReportArchivedRejection.additionalProperties).toBe(false);
  });
});
