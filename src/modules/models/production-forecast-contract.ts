import { z } from "zod";

export const PRODUCTION_FORECAST_PERIODS_MIN = 2;
export const PRODUCTION_FORECAST_PERIODS_MAX = 12;

export const productionForecastPeriodsSchema = z.number()
  .int()
  .min(PRODUCTION_FORECAST_PERIODS_MIN)
  .max(PRODUCTION_FORECAST_PERIODS_MAX);

export const coercedProductionForecastPeriodsSchema = z.coerce
  .number()
  .int()
  .min(PRODUCTION_FORECAST_PERIODS_MIN)
  .max(PRODUCTION_FORECAST_PERIODS_MAX);
