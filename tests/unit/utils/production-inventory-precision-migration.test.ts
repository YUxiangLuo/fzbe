import { describe, expect, mock, test } from "bun:test";
import type { Connection } from "mysql2/promise";
import { migrateProductionInventoryPrecision } from "../../../scripts/migrate-production-inventory-precision";

const floatColumn = {
  data_type: "float",
  column_type: "float",
  is_nullable: "YES",
  // MariaDB information_schema returns SQL NULL defaults as this string.
  column_default: "NULL",
};
const doubleColumn = {
  data_type: "double",
  column_type: "double",
  is_nullable: "YES",
  column_default: null,
};

describe("production initial-inventory precision migration", () => {
  test("widens FLOAT and repairs only a compatible rounded snapshot", async () => {
    const query = mock(async (sqlValue: string) => {
      const sql = String(sqlValue);
      if (sql.includes("information_schema.COLUMNS")) {
        const priorLoads = query.mock.calls.filter(([value]) =>
          String(value).includes("information_schema.COLUMNS")
        ).length;
        return [[priorLoads === 1 ? floatColumn : doubleColumn]];
      }
      if (sql.includes("ALTER TABLE experiment_status")) return [{ affectedRows: 0 }];
      if (sql.includes("SELECT experiment_id")) {
        return [[{
          experiment_id: 91,
          state_version: 7,
          production_initial_inventory: 0.123457,
          production_mps_table: JSON.stringify([{
            beginning_inventory: 0.123456789,
          }]),
        }]];
      }
      if (sql.includes("UPDATE experiment_status")) return [{ affectedRows: 1 }];
      throw new Error(`Unexpected query: ${sql}`);
    });
    const messages: string[] = [];

    await migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: (message) => messages.push(message) },
    );

    expect(query.mock.calls.some(([sql]) =>
      String(sql).includes("MODIFY COLUMN production_initial_inventory DOUBLE")
    )).toBeTrue();
    const repair = query.mock.calls.find(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    );
    expect(repair?.[1]).toEqual([0.123456789, 91, 7, 0.123457]);
    expect(messages).toContain("Reconciled 1 rounded production inventory snapshot(s).");
  });

  test("is idempotent and does not overwrite a materially different MPS snapshot", async () => {
    const query = mock(async (sqlValue: string) => {
      const sql = String(sqlValue);
      if (sql.includes("information_schema.COLUMNS")) return [[doubleColumn]];
      if (sql.includes("SELECT experiment_id")) {
        return [[
          {
            experiment_id: 92,
            state_version: 3,
            production_initial_inventory: 10,
            production_mps_table: [{ beginning_inventory: 12 }],
          },
          {
            experiment_id: 94,
            state_version: 2,
            production_initial_inventory: 0.123457,
            production_mps_table: [{ beginning_inventory: "0.123456789" }],
          },
        ]];
      }
      throw new Error(`Unexpected query: ${sql}`);
    });

    await migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: () => {} },
    );

    expect(query.mock.calls.some(([sql]) =>
      String(sql).includes("ALTER TABLE experiment_status")
      || String(sql).includes("UPDATE experiment_status")
    )).toBeFalse();
  });

  test("does not treat a merely close value as the same legacy FLOAT", async () => {
    const query = mock(async (sqlValue: string) => {
      const sql = String(sqlValue);
      if (sql.includes("information_schema.COLUMNS")) return [[doubleColumn]];
      if (sql.includes("SELECT experiment_id")) {
        return [[{
          experiment_id: 93,
          state_version: 4,
          production_initial_inventory: 0.123457,
          production_mps_table: [{ beginning_inventory: 0.123456 }],
        }]];
      }
      throw new Error(`Unexpected query: ${sql}`);
    });

    await migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: () => {} },
    );

    expect(query.mock.calls.some(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    )).toBeFalse();
  });

  test("re-snapshots and repairs after a concurrent version-only write", async () => {
    let snapshotReadCount = 0;
    let updateCount = 0;
    const legacySnapshot = {
      experiment_id: 95,
      state_version: 7,
      production_initial_inventory: 0.12345679104328156,
      production_mps_table: [{ beginning_inventory: 0.123456789 }],
    };
    const query = mock(async (sqlValue: string) => {
      const sql = String(sqlValue);
      if (sql.includes("information_schema.COLUMNS")) return [[doubleColumn]];
      if (sql.includes("SELECT experiment_id")) {
        snapshotReadCount += 1;
        return [[snapshotReadCount === 1
          ? legacySnapshot
          : { ...legacySnapshot, state_version: 8 }]];
      }
      if (sql.includes("UPDATE experiment_status")) {
        updateCount += 1;
        return [{ affectedRows: updateCount === 1 ? 0 : 1 }];
      }
      throw new Error(`Unexpected query: ${sql}`);
    });
    const messages: string[] = [];

    await migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: (message) => messages.push(message) },
    );

    expect(snapshotReadCount).toBe(2);
    expect(updateCount).toBe(2);
    const reload = query.mock.calls.find(([sql, values]) =>
      String(sql).includes("WHERE experiment_id = ?")
      && Array.isArray(values)
      && values.length === 1
    );
    expect(reload?.[1]).toEqual([95]);
    const repairAttempts = query.mock.calls.filter(([sql]) =>
      String(sql).includes("UPDATE experiment_status")
    );
    expect(repairAttempts[1]?.[1]).toEqual([0.123456789, 95, 8, 0.12345679104328156]);
    expect(messages).toContain("Reconciled 1 rounded production inventory snapshot(s).");
  });

  test("fails closed after bounded retries keep finding a repairable legacy value", async () => {
    let snapshotReadCount = 0;
    let updateCount = 0;
    const legacySnapshot = {
      experiment_id: 97,
      state_version: 10,
      production_initial_inventory: 0.12345679104328156,
      production_mps_table: [{ beginning_inventory: 0.123456789 }],
    };
    const query = mock(async (sqlValue: string) => {
      const sql = String(sqlValue);
      if (sql.includes("information_schema.COLUMNS")) return [[doubleColumn]];
      if (sql.includes("SELECT experiment_id")) {
        snapshotReadCount += 1;
        return [[{
          ...legacySnapshot,
          state_version: legacySnapshot.state_version + snapshotReadCount - 1,
        }]];
      }
      if (sql.includes("UPDATE experiment_status")) {
        updateCount += 1;
        return [{ affectedRows: 0 }];
      }
      throw new Error(`Unexpected query: ${sql}`);
    });

    await expect(migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: () => {} },
    )).rejects.toThrow("after 3 compare-and-swap attempts");

    expect(updateCount).toBe(3);
    expect(snapshotReadCount).toBe(4);
  });

  test("accepts a concurrent business write that already persisted the exact value", async () => {
    let snapshotReadCount = 0;
    const mpsTable = [{ beginning_inventory: 0.123456789 }];
    const query = mock(async (sqlValue: string) => {
      const sql = String(sqlValue);
      if (sql.includes("information_schema.COLUMNS")) return [[doubleColumn]];
      if (sql.includes("SELECT experiment_id")) {
        snapshotReadCount += 1;
        return [[snapshotReadCount === 1
          ? {
            experiment_id: 96,
            state_version: 7,
            production_initial_inventory: 0.12345679104328156,
            production_mps_table: mpsTable,
          }
          : {
            experiment_id: 96,
            state_version: 8,
            production_initial_inventory: 0.123456789,
            production_mps_table: mpsTable,
          }]];
      }
      if (sql.includes("UPDATE experiment_status")) return [{ affectedRows: 0 }];
      throw new Error(`Unexpected query: ${sql}`);
    });

    await expect(migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: () => {} },
    )).resolves.toBeUndefined();
    expect(snapshotReadCount).toBe(2);
  });

  test("leaves a concurrent materially different scalar and MPS pair to integrity checks", async () => {
    let snapshotReadCount = 0;
    let updateCount = 0;
    const query = mock(async (sqlValue: string) => {
      const sql = String(sqlValue);
      if (sql.includes("information_schema.COLUMNS")) return [[doubleColumn]];
      if (sql.includes("SELECT experiment_id")) {
        snapshotReadCount += 1;
        return [[snapshotReadCount === 1
          ? {
            experiment_id: 98,
            state_version: 20,
            production_initial_inventory: 0.12345679104328156,
            production_mps_table: [{ beginning_inventory: 0.123456789 }],
          }
          : {
            experiment_id: 98,
            state_version: 21,
            production_initial_inventory: 25,
            production_mps_table: [{ beginning_inventory: 31 }],
          }]];
      }
      if (sql.includes("UPDATE experiment_status")) {
        updateCount += 1;
        return [{ affectedRows: 0 }];
      }
      throw new Error(`Unexpected query: ${sql}`);
    });

    await expect(migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: () => {} },
    )).resolves.toBeUndefined();

    expect(snapshotReadCount).toBe(2);
    expect(updateCount).toBe(1);
  });

  test("fails closed for an unexpected existing column definition", async () => {
    const query = mock(async () => [[{
      ...doubleColumn,
      column_type: "double unsigned",
    }]]);

    await expect(migrateProductionInventoryPrecision(
      { query } as unknown as Connection,
      { log: () => {} },
    )).rejects.toThrow("incompatible definition");
  });
});
