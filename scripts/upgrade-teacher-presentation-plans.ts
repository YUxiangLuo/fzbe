import path from "node:path";
import {
  buildCanonicalProductionForecasts,
  buildCanonicalProductionMps,
  type ProductionForecastSeed,
} from "./production-plan-fixture";
import {
  CAPACITY_SCENARIO_VALUES,
  buildProductionCapacityScenarios,
  getCompletedProductionPlanValidationError,
} from "../src/modules/experiments/experiments.schema";

const INSERT_PREFIX = "INSERT INTO experiment_status (";
const VALUES_MARKER = ") VALUES\n";

const parseSqlList = (source: string): string[] => {
  const values: string[] = [];
  let start = 0;
  let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === "'" && quoted && source[index + 1] === "'") {
      index += 1;
      continue;
    }
    if (char === "'" && source[index - 1] !== "\\") {
      quoted = !quoted;
      continue;
    }
    if (char === "," && !quoted) {
      values.push(source.slice(start, index));
      start = index + 1;
    }
  }
  values.push(source.slice(start));
  return values;
};

const extractSqlTuples = (
  source: string,
): Array<{ start: number; end: number; body: string }> => {
  const tuples: Array<{ start: number; end: number; body: string }> = [];
  let quoted = false;
  let depth = 0;
  let tupleStart = -1;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === "'" && quoted && source[index + 1] === "'") {
      index += 1;
      continue;
    }
    if (char === "'" && source[index - 1] !== "\\") {
      quoted = !quoted;
      continue;
    }
    if (quoted) continue;
    if (char === "(") {
      if (depth === 0) tupleStart = index;
      depth += 1;
    } else if (char === ")") {
      depth -= 1;
      if (depth < 0) throw new Error("teacher SQL has an unmatched closing parenthesis");
      if (depth === 0 && tupleStart >= 0) {
        tuples.push({
          start: tupleStart,
          end: index + 1,
          body: source.slice(tupleStart + 1, index),
        });
        tupleStart = -1;
      }
    }
  }
  if (quoted || depth !== 0) {
    throw new Error("teacher SQL has an unterminated quote or tuple");
  }
  return tuples;
};

const decodeSqlString = (value: string, label: string): string => {
  const trimmed = value.trim();
  if (!trimmed.startsWith("'") || !trimmed.endsWith("'")) {
    throw new Error(`${label} must be a SQL string literal`);
  }
  return trimmed
    .slice(1, -1)
    .replace(/''/g, "'")
    .replace(/\\'/g, "'")
    .replace(/\\\\/g, "\\");
};

const encodeSqlString = (value: string): string =>
  `'${value.replace(/\\/g, "\\\\").replace(/'/g, "''")}'`;

const parseFiniteColumn = (
  fields: readonly string[],
  columnIndex: ReadonlyMap<string, number>,
  name: string,
) => {
  const index = columnIndex.get(name);
  if (index === undefined) throw new Error(`teacher SQL is missing ${name}`);
  const value = Number(fields[index]?.trim());
  if (!Number.isFinite(value)) {
    throw new Error(`teacher SQL ${name} must be finite`);
  }
  return value;
};

const parseNullableFiniteColumn = (
  fields: readonly string[],
  columnIndex: ReadonlyMap<string, number>,
  name: string,
): number | null => {
  const index = columnIndex.get(name);
  if (index === undefined) throw new Error(`teacher SQL is missing ${name}`);
  if (fields[index]?.trim().toUpperCase() === "NULL") return null;
  return parseFiniteColumn(fields, columnIndex, name);
};

const parseNullableStringColumn = (
  fields: readonly string[],
  columnIndex: ReadonlyMap<string, number>,
  name: string,
): string | null => {
  const index = columnIndex.get(name);
  if (index === undefined) throw new Error(`teacher SQL is missing ${name}`);
  const value = fields[index]!.trim();
  return value.toUpperCase() === "NULL"
    ? null
    : decodeSqlString(value, `teacher SQL ${name}`);
};

const parseForecastSeeds = (
  rawValue: string,
  experimentId: number,
): ProductionForecastSeed[] => {
  const decoded = decodeSqlString(
    rawValue,
    `experiment ${experimentId} production_forecast_results`,
  );
  const parsed = JSON.parse(decoded);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error(`experiment ${experimentId} has no production forecasts`);
  }
  return parsed.map((point, index) => {
    if (!point || typeof point !== "object" || Array.isArray(point)) {
      throw new Error(`experiment ${experimentId} forecast ${index + 1} is invalid`);
    }
    const record = point as Record<string, unknown>;
    const prediction = Number(record.prediction);
    const stdDev = Number(record.std_dev);
    if (
      !Number.isFinite(prediction)
      || prediction < 0
      || !Number.isFinite(stdDev)
      || stdDev < 0
    ) {
      throw new Error(`experiment ${experimentId} forecast ${index + 1} is invalid`);
    }
    return { prediction, std_dev: stdDev };
  });
};

const upgradeExperimentInsert = (statement: string) => {
  if (!statement.startsWith(INSERT_PREFIX)) {
    throw new Error("teacher SQL experiment_status insert is malformed");
  }
  const columnsStart = INSERT_PREFIX.length;
  const valuesMarker = statement.indexOf(VALUES_MARKER, columnsStart);
  if (valuesMarker < 0) throw new Error("teacher SQL experiment_status insert is malformed");
  const valuesStart = valuesMarker + VALUES_MARKER.length;
  const statementEnd = statement.lastIndexOf(";");
  if (statementEnd < 0) throw new Error("teacher SQL experiment_status insert has no terminator");

  const columns = statement
    .slice(columnsStart, valuesMarker)
    .split(",")
    .map((column) => column.trim());
  const columnIndex = new Map(columns.map((column, index) => [column, index]));
  const requiredColumns = [
    "experiment_id",
    "status",
    "production_plan_completed",
    "production_forecast_periods",
    "production_initial_inventory",
    "production_target_service_level",
    "production_safety_stock_z_score",
    "production_forecast_results",
    "production_mps_table",
    "production_capacity_mode",
    "production_capacity_scenario",
    "production_capacity",
    "production_custom_capacity",
  ];
  for (const column of requiredColumns) {
    if (!columnIndex.has(column)) {
      throw new Error(`teacher SQL experiment_status insert is missing ${column}`);
    }
  }

  const statementValues = statement.slice(valuesStart, statementEnd);
  const tuples = extractSqlTuples(statementValues);
  const upgradedTuples: string[] = [];
  let upgradedCount = 0;

  for (const tuple of tuples) {
    const fields = parseSqlList(tuple.body);
    if (fields.length !== columns.length) {
      throw new Error(
        `teacher SQL tuple has ${fields.length} fields; expected ${columns.length}`,
      );
    }
    const completedIndex = columnIndex.get("production_plan_completed")!;
    if (fields[completedIndex]?.trim() !== "1") {
      upgradedTuples.push(`(${tuple.body})`);
      continue;
    }

    const experimentId = parseFiniteColumn(
      fields,
      columnIndex,
      "experiment_id",
    );
    const forecastIndex = columnIndex.get("production_forecast_results")!;
    const forecasts = buildCanonicalProductionForecasts(
      parseForecastSeeds(fields[forecastIndex]!, experimentId),
    );
    const zScore = parseFiniteColumn(
      fields,
      columnIndex,
      "production_safety_stock_z_score",
    );
    const capacityMode = parseNullableStringColumn(
      fields,
      columnIndex,
      "production_capacity_mode",
    );
    const capacityScenario = parseNullableStringColumn(
      fields,
      columnIndex,
      "production_capacity_scenario",
    );
    let capacity = parseFiniteColumn(
      fields,
      columnIndex,
      "production_capacity",
    );
    if (capacityMode === "scenario") {
      if (!CAPACITY_SCENARIO_VALUES.includes(
        capacityScenario as typeof CAPACITY_SCENARIO_VALUES[number],
      )) {
        throw new Error(
          `experiment ${experimentId} has an invalid production capacity scenario`,
        );
      }
      capacity = buildProductionCapacityScenarios(forecasts, zScore)[
        capacityScenario as typeof CAPACITY_SCENARIO_VALUES[number]
      ];
      fields[columnIndex.get("production_capacity")!] = String(capacity);
    }
    const mpsTable = buildCanonicalProductionMps(forecasts, {
      initialInventory: parseFiniteColumn(
        fields,
        columnIndex,
        "production_initial_inventory",
      ),
      zScore,
      capacity,
    });
    const validationError = getCompletedProductionPlanValidationError({
      status: parseNullableStringColumn(fields, columnIndex, "status"),
      production_plan_completed: true,
      production_forecast_periods: parseFiniteColumn(
        fields,
        columnIndex,
        "production_forecast_periods",
      ),
      production_initial_inventory: parseFiniteColumn(
        fields,
        columnIndex,
        "production_initial_inventory",
      ),
      production_target_service_level: parseFiniteColumn(
        fields,
        columnIndex,
        "production_target_service_level",
      ),
      production_safety_stock_z_score: zScore,
      production_forecast_results: forecasts,
      production_mps_table: mpsTable,
      production_capacity_mode: capacityMode,
      production_capacity_scenario: capacityScenario,
      production_capacity: capacity,
      production_custom_capacity: parseNullableFiniteColumn(
        fields,
        columnIndex,
        "production_custom_capacity",
      ),
    });
    if (validationError) {
      throw new Error(
        `experiment ${experimentId} has an invalid canonical production plan: ${validationError}`,
      );
    }
    fields[forecastIndex] = encodeSqlString(JSON.stringify(forecasts));
    fields[columnIndex.get("production_mps_table")!] =
      encodeSqlString(JSON.stringify(mpsTable));
    upgradedTuples.push(`(${fields.join(",")})`);
    upgradedCount += 1;
  }

  if (tuples.length === 0) {
    throw new Error("teacher SQL experiment_status insert contains no tuples");
  }
  const upgradedValues = upgradedTuples.join(",\n");
  return {
    sql: statement.slice(0, valuesStart)
      + upgradedValues
      + statement.slice(statementEnd),
    upgradedCount,
  };
};

export function upgradeTeacherPresentationSql(sql: string) {
  const chunks: string[] = [];
  let cursor = 0;
  let upgradedCount = 0;
  let insertCount = 0;

  while (true) {
    const insertStart = sql.indexOf(INSERT_PREFIX, cursor);
    if (insertStart < 0) break;
    const statementEnd = sql.indexOf(";\n", insertStart);
    if (statementEnd < 0) {
      throw new Error("teacher SQL experiment_status insert has no terminator");
    }
    chunks.push(sql.slice(cursor, insertStart));
    const upgraded = upgradeExperimentInsert(
      sql.slice(insertStart, statementEnd + 1),
    );
    chunks.push(upgraded.sql);
    upgradedCount += upgraded.upgradedCount;
    insertCount += 1;
    cursor = statementEnd + 1;
  }

  if (insertCount === 0 || upgradedCount === 0) {
    throw new Error("teacher SQL did not contain completed production plans");
  }
  chunks.push(sql.slice(cursor));
  return {
    sql: chunks.join(""),
    upgradedCount,
  };
}

if (import.meta.main) {
  const sqlPath = path.resolve(
    process.cwd(),
    "sql",
    "teacher1_presentation_data.sql",
  );
  const source = await Bun.file(sqlPath).text();
  const upgraded = upgradeTeacherPresentationSql(source);
  const write = process.argv.includes("--write");
  if (write) {
    await Bun.write(sqlPath, upgraded.sql);
    console.log(`Upgraded ${upgraded.upgradedCount} teacher production plans.`);
  } else if (source !== upgraded.sql) {
    throw new Error(
      `${upgraded.upgradedCount} teacher production plans are not canonical; rerun with --write`,
    );
  } else {
    console.log(
      `Teacher production plans are canonical (${upgraded.upgradedCount} rows).`,
    );
  }
}
