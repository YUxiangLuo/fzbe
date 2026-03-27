import { spawn } from "node:child_process";
import process from "node:process";
import type { RowDataPacket } from "mysql2/promise";
import {
  PARALLEL_ACCOUNTS,
  PARALLEL_CLASS,
  PARALLEL_SCENARIOS,
  PARALLEL_STUDENT_USERNAMES,
  assertBaseEnv,
  createDbConnection,
  getCliPositionalArgs,
  type ParallelScenarioName,
} from "./parallel-seed.shared";

interface CountRow extends RowDataPacket {
  count: number;
}

interface TableRow extends RowDataPacket {
  table_name: string;
}

const REQUIRED_TABLES = [
  "users",
  "classes",
  "enrollments",
  "grade_weights",
  "question_bank",
  "experiment_manuals",
  "datasets",
  "dataset_segments",
  "experiment_status",
  "experiment_grade",
  "experiment_step_events",
  "experiment_reports",
] as const;
const MIN_QUESTION_BANK_COUNT = 16;

function hasForceFlag() {
  return process.argv.includes("--force");
}

function parseScenario(): ParallelScenarioName | null {
  const rawValue = getCliPositionalArgs()[0]?.trim();
  if (!rawValue) {
    return null;
  }

  if (PARALLEL_SCENARIOS.includes(rawValue as ParallelScenarioName)) {
    return rawValue as ParallelScenarioName;
  }

  throw new Error(
    `Invalid scenario "${rawValue}". Expected one of: ${PARALLEL_SCENARIOS.join(", ")}`,
  );
}

async function runSubcommand(input: { label: string; args: string[] }) {
  console.log(`[parallel-prepare] ${input.label}: bun ${input.args.join(" ")}`);

  const child = spawn("bun", input.args, {
    cwd: process.cwd(),
    env: process.env,
    stdio: "inherit",
  });

  await new Promise<void>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          `${input.label} failed (code=${String(code)}, signal=${String(signal)})`,
        ),
      );
    });
  });
}

async function hasParallelBaseReady() {
  assertBaseEnv();

  const connection = await createDbConnection();
  const requiredUsernames = [
    PARALLEL_ACCOUNTS.adminUsername,
    PARALLEL_ACCOUNTS.teacherUsername,
    PARALLEL_ACCOUNTS.assistantUsername,
    ...PARALLEL_STUDENT_USERNAMES,
  ];

  try {
    const [tableRows] = await connection.query<TableRow[]>(
      `SELECT table_name
         FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name IN (${REQUIRED_TABLES.map(() => "?").join(", ")})`,
      [...REQUIRED_TABLES],
    );
    const existingTables = new Set(tableRows.map((row) => row.table_name));
    const missingTables = REQUIRED_TABLES.filter((tableName) =>
      !existingTables.has(tableName),
    );
    if (missingTables.length > 0) {
      console.log(
        `[parallel-prepare] missing tables: ${missingTables.join(", ")}`,
      );
      return false;
    }

    const [userRows] = await connection.query<CountRow[]>(
      `SELECT COUNT(*) AS count
         FROM users
        WHERE username IN (${requiredUsernames.map(() => "?").join(", ")})`,
      requiredUsernames,
    );
    const userCount = userRows[0]?.count ?? 0;
    if (userCount !== requiredUsernames.length) {
      console.log(
        `[parallel-prepare] expected ${requiredUsernames.length} parallel users, found ${userCount}`,
      );
      return false;
    }

    const [classRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM classes WHERE class_id = ?",
      [PARALLEL_CLASS.id],
    );
    if ((classRows[0]?.count ?? 0) !== 1) {
      console.log(
        `[parallel-prepare] missing parallel class ${PARALLEL_CLASS.id}`,
      );
      return false;
    }

    const [enrollmentRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM enrollments WHERE class_id = ?",
      [PARALLEL_CLASS.id],
    );
    if ((enrollmentRows[0]?.count ?? 0) !== PARALLEL_STUDENT_USERNAMES.length) {
      console.log(
        `[parallel-prepare] expected ${PARALLEL_STUDENT_USERNAMES.length} enrollments for class ${PARALLEL_CLASS.id}, found ${enrollmentRows[0]?.count ?? 0}`,
      );
      return false;
    }

    const [assistantRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM class_assistants WHERE class_id = ? AND assistant_id = ?",
      [PARALLEL_CLASS.id, 900201],
    );
    if ((assistantRows[0]?.count ?? 0) !== 1) {
      console.log(
        `[parallel-prepare] expected assistant 900201 to be assigned to class ${PARALLEL_CLASS.id}`,
      );
      return false;
    }

    const [gradeWeightRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM grade_weights WHERE class_id = ?",
      [PARALLEL_CLASS.id],
    );
    if ((gradeWeightRows[0]?.count ?? 0) !== 1) {
      console.log(
        `[parallel-prepare] expected grade_weights for class ${PARALLEL_CLASS.id}`,
      );
      return false;
    }

    const [questionRows] = await connection.query<CountRow[]>(
      "SELECT COUNT(*) AS count FROM question_bank",
    );
    if ((questionRows[0]?.count ?? 0) < MIN_QUESTION_BANK_COUNT) {
      console.log(
        `[parallel-prepare] expected at least ${MIN_QUESTION_BANK_COUNT} question_bank rows, found ${questionRows[0]?.count ?? 0}`,
      );
      return false;
    }

    return true;
  } finally {
    await connection.end();
  }
}

async function main() {
  if (process.argv.includes("--help")) {
    console.log(`Usage: bun run scripts/prepare-parallel.ts [scenario] --force

Scenarios:
  start-only
  training-burst
  mixed-load

Behavior:
  1. Check whether the parallel database baseline exists
  2. Run db:reset:parallel --force if the baseline is missing
  3. Run seed:parallel:assets --force
  4. Optionally run seed:parallel:scenario <scenario> --force
`);
    return;
  }

  if (!hasForceFlag()) {
    throw new Error(
      "Parallel environment preparation requires --force.",
    );
  }

  const scenario = parseScenario();
  const baseReady = await hasParallelBaseReady();

  console.log(`[parallel-prepare] baseReady=${baseReady}`);
  if (!baseReady) {
    await runSubcommand({
      label: "reset-base",
      args: ["run", "db:reset:parallel", "--force"],
    });
  }

  await runSubcommand({
    label: "seed-assets",
    args: ["run", "seed:parallel:assets", "--force"],
  });

  if (scenario) {
    await runSubcommand({
      label: `seed-scenario:${scenario}`,
      args: ["run", "seed:parallel:scenario", scenario, "--force"],
    });
  } else {
    console.log("[parallel-prepare] no scenario requested");
  }
}

await main().catch((error) => {
  console.error("[parallel-prepare] failed", error);
  process.exit(1);
});
