import {
  type Connection,
  type ResultSetHeader,
  type RowDataPacket,
} from "mysql2/promise";
import { createMigrationConnection } from "./migration-database";

interface ColumnDefinitionRow extends RowDataPacket {
  data_type: string;
  column_type: string;
  is_nullable: string;
  column_default: unknown;
}

interface InventorySnapshotRow extends RowDataPacket {
  experiment_id: number;
  state_version: number;
  production_initial_inventory: number | string | null;
  production_mps_table: unknown;
}

const INVENTORY_COLUMN = "production_initial_inventory";
const INVENTORY_REPAIR_MAX_ATTEMPTS = 3;

const loadColumnDefinition = async (
  connection: Connection,
): Promise<ColumnDefinitionRow[]> => {
  const [rows] = await connection.query<ColumnDefinitionRow[]>(
    `SELECT DATA_TYPE AS data_type,
            COLUMN_TYPE AS column_type,
            IS_NULLABLE AS is_nullable,
            COLUMN_DEFAULT AS column_default
       FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'experiment_status'
        AND COLUMN_NAME = ?`,
    [INVENTORY_COLUMN],
  );
  return rows;
};

const assertExpectedColumnShape = (
  column: ColumnDefinitionRow,
  expectedType: "float" | "double",
): void => {
  // MySQL exposes a SQL NULL default as JavaScript null; MariaDB may expose
  // the same metadata value as the unquoted string "NULL".
  const hasNullDefault = column.column_default === null
    || String(column.column_default).toUpperCase() === "NULL";
  if (
    column.data_type.toLowerCase() !== expectedType
    || column.column_type.toLowerCase() !== expectedType
    || column.is_nullable.toUpperCase() !== "YES"
    || !hasNullDefault
  ) {
    throw new Error(
      `${INVENTORY_COLUMN} has an incompatible definition; manual repair is required.`,
    );
  }
};

const parseFirstBeginningInventory = (value: unknown): number | null => {
  let parsed = value;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      return null;
    }
  }
  if (!Array.isArray(parsed) || parsed.length === 0) return null;
  const firstRow = parsed[0];
  if (!firstRow || typeof firstRow !== "object" || Array.isArray(firstRow)) {
    return null;
  }
  const candidate = (firstRow as Record<string, unknown>).beginning_inventory;
  return typeof candidate === "number"
    && Number.isFinite(candidate)
    && candidate >= 0
    ? candidate
    : null;
};

const parseJsonValue = (value: unknown): unknown => {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};

const toFiniteNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const inventorySnapshotValuesEqual = (
  left: InventorySnapshotRow,
  right: InventorySnapshotRow,
): boolean => {
  const leftInventory = toFiniteNumber(left.production_initial_inventory);
  const rightInventory = toFiniteNumber(right.production_initial_inventory);
  if (
    leftInventory === null
    || rightInventory === null
    || !Object.is(leftInventory, rightInventory)
  ) {
    return false;
  }

  return JSON.stringify(parseJsonValue(left.production_mps_table))
    === JSON.stringify(parseJsonValue(right.production_mps_table));
};

const isCompatibleFloatRoundTrip = (stored: number, candidate: number): boolean => {
  const binaryFloat = Math.fround(candidate);
  // MariaDB's text protocol renders an unconstrained FLOAT with six
  // significant digits (for example 0.123456789 -> 0.123457). MySQL/mysql2
  // may instead expose the full binary-float value. Accept only either exact
  // representation of the same float, rather than a broad numeric tolerance.
  const legacyRepresentations = [
    binaryFloat,
    Number(binaryFloat.toPrecision(6)),
  ];
  return legacyRepresentations.some((value) => (
    Math.abs(stored - value)
      <= Number.EPSILON * Math.max(1, Math.abs(stored), Math.abs(value))
  ));
};

interface RepairableInventorySnapshot {
  stored: number;
  candidate: number;
}

const getRepairableInventorySnapshot = (
  row: InventorySnapshotRow,
): RepairableInventorySnapshot | null => {
  const stored = toFiniteNumber(row.production_initial_inventory);
  const candidate = parseFirstBeginningInventory(row.production_mps_table);
  if (
    stored === null
    || candidate === null
    || Math.abs(stored - candidate) <= 1e-9
    || !isCompatibleFloatRoundTrip(stored, candidate)
  ) {
    return null;
  }
  return { stored, candidate };
};

const isInventorySnapshotConsistent = (row: InventorySnapshotRow): boolean => {
  const stored = toFiniteNumber(row.production_initial_inventory);
  const candidate = parseFirstBeginningInventory(row.production_mps_table);
  return stored !== null
    && candidate !== null
    && Math.abs(stored - candidate) <= 1e-9;
};

const reconcileRoundedInventorySnapshots = async (
  connection: Connection,
): Promise<number> => {
  const [rows] = await connection.query<InventorySnapshotRow[]>(
    `SELECT experiment_id,
            state_version,
            production_initial_inventory,
            production_mps_table
       FROM experiment_status
      WHERE production_initial_inventory IS NOT NULL
        AND production_mps_table IS NOT NULL`,
  );

  let repaired = 0;
  for (const initialSnapshot of rows) {
    const initialRepair = getRepairableInventorySnapshot(initialSnapshot);
    if (!initialRepair) continue;

    let snapshot = initialSnapshot;
    let repair = initialRepair;
    for (let attempt = 1; attempt <= INVENTORY_REPAIR_MAX_ATTEMPTS; attempt += 1) {
      // state_version makes the repair safe while the legacy writer is still
      // live during staging. Re-snapshotting a still-repairable row avoids
      // requiring an unreachable post-stop rerun merely because an unrelated
      // state-versioned write won the first comparison-and-swap.
      const [result] = await connection.query<ResultSetHeader>(
        `UPDATE experiment_status
            SET production_initial_inventory = ?
          WHERE experiment_id = ?
            AND state_version = ?
            AND production_initial_inventory <=> ?`,
        [
          repair.candidate,
          snapshot.experiment_id,
          snapshot.state_version,
          repair.stored,
        ],
      );
      if (result.affectedRows > 0) {
        repaired += result.affectedRows;
        break;
      }

      const [currentRows] = await connection.query<InventorySnapshotRow[]>(
        `SELECT experiment_id,
                state_version,
                production_initial_inventory,
                production_mps_table
           FROM experiment_status
          WHERE experiment_id = ?
          LIMIT 1`,
        [snapshot.experiment_id],
      );
      const current = currentRows[0];
      if (!current) {
        // A concurrent business transaction deleted the complete experiment
        // graph. There is no remaining precision snapshot to reconcile.
        break;
      }
      if (isInventorySnapshotConsistent(current)) break;

      const stateVersionChanged = String(current.state_version)
        !== String(snapshot.state_version);
      const relatedValuesChanged = !inventorySnapshotValuesEqual(snapshot, current);
      const currentRepair = getRepairableInventorySnapshot(current);

      if (currentRepair) {
        if (attempt === INVENTORY_REPAIR_MAX_ATTEMPTS) {
          throw new Error(
            `Could not safely reconcile production inventory for experiment ${current.experiment_id} `
            + `after ${INVENTORY_REPAIR_MAX_ATTEMPTS} compare-and-swap attempts while concurrent `
            + "writes kept a repairable legacy FLOAT representation. Rerun the deployment after "
            + "write traffic subsides.",
          );
        }
        snapshot = current;
        repair = currentRepair;
        continue;
      }

      if (stateVersionChanged && relatedValuesChanged) {
        // Explicit policy: a state-versioned business write that materially
        // changed the scalar/MPS pair owns its new state. If that pair is no
        // longer an exact legacy-FLOAT round trip, guessing a repair from the
        // stale snapshot would be unsafe; leave it to normal integrity checks.
        break;
      }

      throw new Error(
        `Could not safely reconcile production inventory for experiment ${current.experiment_id}: `
        + `snapshot state_version ${snapshot.state_version}, current state_version ${current.state_version}, `
        + "and the current values cannot be attributed to a safely repairable or materially changed "
        + "business snapshot.",
      );
    }
  }
  return repaired;
};

export interface ProductionInventoryPrecisionMigrationOptions {
  log?: (message: string) => void;
}

export async function migrateProductionInventoryPrecision(
  connection: Connection,
  { log = console.log }: ProductionInventoryPrecisionMigrationOptions = {},
): Promise<void> {
  let columns = await loadColumnDefinition(connection);
  if (columns.length !== 1) {
    throw new Error(`Expected exactly one ${INVENTORY_COLUMN} column.`);
  }

  const currentType = columns[0]!.data_type.toLowerCase();
  if (currentType === "float") {
    assertExpectedColumnShape(columns[0]!, "float");
    await connection.query(
      `ALTER TABLE experiment_status
         MODIFY COLUMN production_initial_inventory DOUBLE NULL DEFAULT NULL`,
    );
    log(`Widened ${INVENTORY_COLUMN} from FLOAT to DOUBLE.`);
    columns = await loadColumnDefinition(connection);
  }

  if (columns.length !== 1) {
    throw new Error(`Expected exactly one ${INVENTORY_COLUMN} column after migration.`);
  }
  assertExpectedColumnShape(columns[0]!, "double");

  const repaired = await reconcileRoundedInventorySnapshots(connection);
  if (repaired > 0) {
    log(`Reconciled ${repaired} rounded production inventory snapshot(s).`);
  }
  log("Production initial-inventory precision migration complete.");
}

if (import.meta.main) {
  const connection = await createMigrationConnection();
  try {
    await migrateProductionInventoryPrecision(connection);
  } finally {
    await connection.end();
  }
}
