import { Hono } from "hono";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import { jwt } from "hono/jwt";

// Note: The business logic for forecasting (e.g., ARIMA, Exponential Smoothing)
// and statistical tests (e.g., ADF) is complex and typically requires specialized
// libraries. The following implementations are MOCKED to match the API specification
// but do not perform real calculations.

export default (JWT_SECRET: string) => {
  const forecastApi = new Hono();

  // Add JWT authentication middleware to all routes in this file
  forecastApi.use("*", jwt({ secret: JWT_SECRET }));

  // --- Zod Schemas for Forecasting API Validation ---

  // Base schema for all forecast requests
  const forecastBaseSchema = z.object({
    product_id: z.union([z.string(), z.number()]).optional(),
    forecast_periods: z.number().int().min(1),
    time_unit: z.string().optional(),
  });

  // Moving Average Request
  const forecastMovingAverageSchema = forecastBaseSchema.extend({
    method: z.literal("moving_average"),
    historical_data: z.array(z.number()).min(3),
    window_size: z.number().int().min(1),
  });

  // Exponential Smoothing Request
  const forecastExponentialSmoothingSchema = forecastBaseSchema.extend({
    method: z.literal("exponential_smoothing"),
    historical_data: z.array(z.number()).min(3),
    alpha: z.number().gt(0).lte(1),
  });

  // ARIMA Request
  const forecastArimaSchema = forecastBaseSchema.extend({
    method: z.literal("arima"),
    historical_data: z.array(z.number()).min(8),
    d: z.union([z.literal(0), z.literal(1), z.literal(2)]),
  });

  // LSTM Request
  const forecastLstmSchema = forecastBaseSchema.extend({
      method: z.literal("lstm"),
      data: z.object({
          rows: z.array(z.record(z.string(), z.any())).min(50),
      }),
      feature_config: z.object({
          time_column: z.string(),
          target: z.string(),
          feature_columns: z.array(z.string()).min(1),
          normalization: z.object({
              method: z.enum(["minmax", "zscore"]),
          }),
          missing: z.object({
              strategy: z.enum(["ffill", "bfill", "interpolate", "drop"]),
          }).optional(),
          time_features: z.object({
              enable: z.boolean(),
              components: z.array(z.enum(["dow", "month", "dom", "doy"])),
              cyclic: z.boolean(),
          }).optional(),
          categorical: z.object({
              columns: z.array(z.string()),
              encoding: z.enum(["onehot", "target"]),
          }).optional(),
      }),
  });


  // A discriminated union to handle the different forecast methods
  const forecastRequestSchema = z.discriminatedUnion("method", [
    forecastMovingAverageSchema,
    forecastExponentialSmoothingSchema,
    forecastArimaSchema,
    forecastLstmSchema,
  ]);

  // ADF Test Request
  const adfTestRequestSchema = z.object({
    product_id: z.union([z.string(), z.number()]).optional(),
    time_series: z.array(z.number()).min(8),
    significance_level: z.number().default(0.05),
    max_difference_order: z.number().int().default(2),
  });

  // Difference Test Request
  const differenceTestRequestSchema = z.object({
    product_id: z.union([z.string(), z.number()]).optional(),
    time_series: z.array(z.number()).min(8),
    difference_order: z.union([z.literal(0), z.literal(1), z.literal(2)]),
    significance_level: z.number().default(0.05),
  });

  // Metrics Request
  const metricsRequestSchema = z.object({
      y_true: z.array(z.number()).min(1),
      y_pred: z.array(z.number()).min(1),
      metrics: z.array(z.enum(["rmse", "mae", "r2"])).optional(),
      sample_weight: z.array(z.number()).optional(),
      nan_policy: z.enum(["omit", "raise"]).default("omit"),
      decimal_places: z.number().int().min(0).max(10).default(4),
  }).refine(data => data.y_true.length === data.y_pred.length, {
      message: "y_true and y_pred must have the same length",
      path: ["y_pred"],
  });


  // --- Forecasting API Routes ---

  forecastApi.post("/forecast", zValidator("json", forecastRequestSchema), async (c) => {
    const body = c.req.valid("json");

    // Mock logic based on method
    switch (body.method) {
      case "moving_average":
        return c.json({
          product_id: body.product_id,
          method: "moving_average",
          params: {
            window_size: body.window_size,
          },
          forecast: {
            horizon: body.forecast_periods,
            values: Array(body.forecast_periods).fill(0).map((_, i) => 145.5 + i * 2), // Mocked values
          },
        });
      case "exponential_smoothing":
        return c.json({
          product_id: body.product_id,
          method: "exponential_smoothing",
          params: {
            alpha: body.alpha,
          },
          forecast: {
            horizon: body.forecast_periods,
            values: Array(body.forecast_periods).fill(0).map((_, i) => 151.2 + i * 1.5), // Mocked values
          },
        });
      case "arima":
         return c.json({
              product_id: body.product_id,
              method: "arima",
              used_params: { p: 2, d: body.d, q: 1 },
              model_selection: {
                criterion: "aic",
                best_aic: 1175.22,
                search_space: { p_min: 0, p_max: 5, q_min: 0, q_max: 5 },
              },
              forecast: {
                horizon: body.forecast_periods,
                values: Array(body.forecast_periods).fill(0).map((_, i) => 160 + i * 2.5), // Mocked values
              },
              diagnostics: {
                residual_lb_pvalue: 0.21,
                is_residual_white_noise: true,
              },
            });
      case "lstm":
        // Generate mock forecast dates
        const forecastDates = Array(body.forecast_periods).fill(0).map((_, i) => {
            const d = new Date();
            d.setDate(d.getDate() + i + 1);
            return d.toISOString().split('T')[0];
        });
        return c.json({
            product_id: body.product_id,
            method: "lstm",
            forecast: {
                horizon: body.forecast_periods,
                timestamps: forecastDates,
                values: Array(body.forecast_periods).fill(0).map((_, i) => 170 + Math.random() * 5 + i * 2), // Mocked values
            },
            used_params: {
                normalization: {
                    method: body.feature_config.normalization.method,
                },
                feature_columns: body.feature_config.feature_columns,
                target: body.feature_config.target,
            },
        });
    }

    // This part should be unreachable due to the discriminated union, but as a fallback:
    return c.json({ error: "Invalid forecast method" }, 400);
  });

  forecastApi.post("/tools/adf-test", zValidator("json", adfTestRequestSchema), async (c) => {
      const body = c.req.valid("json");
      // Mock response
      return c.json({
          product_id: body.product_id,
          results: [
              {
                  difference_order: 0,
                  adf_statistic: -2.5,
                  p_value: 0.11,
                  critical_values: { "1%": -3.5, "5%": -2.9, "10%": -2.6 },
                  is_stationary: false,
              },
              {
                  difference_order: 1,
                  adf_statistic: -4.5,
                  p_value: 0.001,
                  critical_values: { "1%": -3.5, "5%": -2.9, "10%": -2.6 },
                  is_stationary: true,
              }
          ],
          recommended_difference_order: 1,
          recommendation: `Based on a significance level of ${body.significance_level}, a difference order of 1 is recommended to achieve stationarity.`
      });
  });

  forecastApi.post("/tools/difference-test", zValidator("json", differenceTestRequestSchema), async (c) => {
      const body = c.req.valid("json");
      const isStationary = body.difference_order > 0; // Mock logic
      // Mock response
      return c.json({
          product_id: body.product_id,
          difference_order: body.difference_order,
          adf_statistic: isStationary ? -4.8 : -2.1,
          p_value: isStationary ? 0.0001 : 0.23,
          critical_values: { "1%": -3.5, "5%": -2.9, "10%": -2.6 },
          is_stationary: isStationary,
          recommendation: isStationary
              ? "The time series is likely stationary at the chosen difference order."
              : "The time series is likely not stationary at the chosen difference order."
      });
  });

  forecastApi.post("/tools/metrics", zValidator("json", metricsRequestSchema), async (c) => {
      const body = c.req.valid("json");
      const n_samples = body.y_true.length;

      // Mock calculation
      const metrics_to_calculate = body.metrics || ["rmse", "mae", "r2"];
      const response: any = {
          n_samples: n_samples,
          effective_samples: n_samples, // Assuming no NaNs for mock
          input_summary: {
              len_y_true: n_samples,
              len_y_pred: body.y_pred.length,
              len_sample_weight: body.sample_weight?.length || null,
          }
      };

      if (metrics_to_calculate.includes("rmse")) {
          response.rmse = 1.4142;
      }
      if (metrics_to_calculate.includes("mae")) {
          response.mae = 1.2;
      }
      if (metrics_to_calculate.includes("r2")) {
          response.r2 = 0.9821;
      }

      return c.json(response);
  });

  return forecastApi;
}
