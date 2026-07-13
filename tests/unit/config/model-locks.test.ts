import { beforeEach, describe, expect, mock, test } from "bun:test";
import type { PoolConnection } from "mysql2/promise";
import { createModelLockManager } from "../../../src/modules/models/model-lock-manager";
import { ModelServiceUnavailableError } from "../../../src/modules/models/model-errors";

const poolGetConnection = mock(async () => null as never);
const createManager = (heartbeatIntervalMs?: number) => createModelLockManager({
  pool: { getConnection: poolGetConnection },
  maxConcurrentJobs: 2,
  connectionAcquireTimeoutMs: 10,
  queryTimeoutMs: 20,
  heartbeatIntervalMs,
});
let manager = createManager();

const makeConnection = () => {
  let connectionId = 101;
  let releaseFails = false;
  let acquisitionFailure: unknown = null;
  let acquisitionQuery: Promise<unknown> | null = null;
  let verificationFailure: unknown = null;
  let verificationQuery: Promise<unknown> | null = null;
  let reportedOwner: number | null = connectionId;
  const release = mock(() => undefined);
  const destroy = mock(() => undefined);
  const query = mock(async (options: { sql: string; timeout?: number }) => {
    expect(options.timeout).toBe(20);
    if (options.sql.includes("GET_LOCK")) {
      if (acquisitionQuery) return await acquisitionQuery;
      if (acquisitionFailure) throw acquisitionFailure;
      return [[{ acquired: 1, connection_id: connectionId }]];
    }
    if (options.sql.includes("IS_USED_LOCK")) {
      if (verificationQuery) return await verificationQuery;
      if (verificationFailure) throw verificationFailure;
      return [[{ owner_id: reportedOwner }]];
    }
    if (options.sql.includes("RELEASE_LOCK")) {
      if (releaseFails) throw new Error("connection reset during release");
      return [[{ released: 1 }]];
    }
    return [[]];
  });

  return {
    connection: { query, release, destroy } as unknown as PoolConnection,
    query,
    release,
    destroy,
    setAcquisitionFailure(value: unknown) { acquisitionFailure = value; },
    setAcquisitionQuery(value: Promise<unknown> | null) { acquisitionQuery = value; },
    setReleaseFails(value: boolean) { releaseFails = value; },
    setReportedOwner(value: number | null) { reportedOwner = value; },
    setVerificationQuery(value: Promise<unknown> | null) { verificationQuery = value; },
    setVerificationFailure(value: unknown) { verificationFailure = value; },
    setConnectionId(value: number) { connectionId = value; reportedOwner = value; },
  };
};

describe("model advisory locks", () => {
  beforeEach(() => {
    poolGetConnection.mockReset();
    manager = createManager();
  });

  test("acquires and releases a global slot on one dedicated connection", async () => {
    const fake = makeConnection();
    poolGetConnection.mockResolvedValue(fake.connection as never);

    const handle = await manager.acquireModelSlot();
    expect(handle?.ownerConnectionId).toBe(101);
    expect(handle?.name).toContain(":slot:1");

    await manager.releaseModelSlot(handle);
    expect(fake.release).toHaveBeenCalledTimes(1);
    expect(fake.destroy).not.toHaveBeenCalled();
    expect(manager.getModelLockStats().active).toBe(0);
  });

  test("actively verifies the MariaDB connection still owns the lock", async () => {
    const fake = makeConnection();
    poolGetConnection.mockResolvedValue(fake.connection as never);
    const handle = await manager.acquireModelExecutionLock(9, "ma");
    fake.setReportedOwner(999);

    await expect(manager.verifyModelLockOwned(handle)).rejects.toMatchObject({
      code: "MODEL_LOCK_LOST",
    });
    await manager.releaseModelExecutionLock(handle);
  });

  test("destroys rather than pools a connection when RELEASE_LOCK fails", async () => {
    const fake = makeConnection();
    poolGetConnection.mockResolvedValue(fake.connection as never);
    const handle = await manager.acquireModelExecutionLock(9, "ma");
    fake.setReleaseFails(true);

    await manager.releaseModelExecutionLock(handle);
    expect(fake.destroy).toHaveBeenCalledTimes(1);
    expect(fake.release).not.toHaveBeenCalled();
  });

  test("destroys an unusable handle when ownership verification times out", async () => {
    const fake = makeConnection();
    poolGetConnection.mockResolvedValue(fake.connection as never);
    const handle = await manager.acquireModelExecutionLock(9, "ma");
    fake.setVerificationFailure(Object.assign(new Error("Query inactivity timeout"), {
      code: "PROTOCOL_SEQUENCE_TIMEOUT",
    }));

    await expect(manager.verifyModelLockOwned(handle)).rejects.toMatchObject({
      code: "MODEL_LOCK_VERIFY_FAILED",
    });
    expect(fake.destroy).toHaveBeenCalledTimes(1);

    const queryCallsBeforeRelease = fake.query.mock.calls.length;
    await manager.releaseModelExecutionLock(handle);
    expect(fake.query).toHaveBeenCalledTimes(queryCallsBeforeRelease);
    expect(fake.release).not.toHaveBeenCalled();
  });

  test("destroys the dedicated connection instead of queuing release behind a heartbeat", async () => {
    manager = createManager(1);
    const fake = makeConnection();
    fake.setVerificationQuery(new Promise(() => {}));
    poolGetConnection.mockResolvedValue(fake.connection as never);
    const handle = await manager.acquireModelExecutionLock(9, "ma");

    for (
      let attempt = 0;
      attempt < 20 && !fake.query.mock.calls.some(([options]) => options.sql.includes("IS_USED_LOCK"));
      attempt += 1
    ) {
      await Bun.sleep(1);
    }
    expect(fake.query.mock.calls.some(([options]) => options.sql.includes("IS_USED_LOCK"))).toBe(true);

    await manager.releaseModelExecutionLock(handle);

    expect(fake.destroy).toHaveBeenCalledTimes(1);
    expect(fake.release).not.toHaveBeenCalled();
    expect(fake.query.mock.calls.some(([options]) => options.sql.includes("RELEASE_LOCK"))).toBe(false);
    expect(manager.getModelLockStats().active).toBe(0);
  });

  test("wraps GET_LOCK query failures as structured service-unavailable errors", async () => {
    const fake = makeConnection();
    const failuresBefore = manager.getModelLockStats().acquisitionFailures;
    poolGetConnection.mockResolvedValue(fake.connection as never);
    fake.setAcquisitionFailure(Object.assign(new Error("Query inactivity timeout"), {
      code: "PROTOCOL_SEQUENCE_TIMEOUT",
    }));

    const error = await manager.acquireModelExecutionLock(9, "ma").catch((caught) => caught);
    expect(error).toBeInstanceOf(ModelServiceUnavailableError);
    expect(error).toMatchObject({ code: "MODEL_LOCK_QUERY_TIMEOUT" });
    expect(fake.destroy).toHaveBeenCalledTimes(1);
    expect(manager.getModelLockStats().acquisitionFailures).toBe(failuresBefore + 1);
  });

  test("wraps global-slot query failures and counts them", async () => {
    const fake = makeConnection();
    const failuresBefore = manager.getModelLockStats().acquisitionFailures;
    poolGetConnection.mockResolvedValue(fake.connection as never);
    fake.setAcquisitionFailure(new Error("database connection reset"));

    const error = await manager.acquireModelSlot().catch((caught) => caught);
    expect(error).toBeInstanceOf(ModelServiceUnavailableError);
    expect(error).toMatchObject({ code: "MODEL_LOCK_ACQUISITION_FAILED" });
    expect(fake.destroy).toHaveBeenCalledTimes(1);
    expect(manager.getModelLockStats().acquisitionFailures).toBe(failuresBefore + 1);
  });

  test("bounds pool acquisition and releases a healthy connection that arrives late", async () => {
    const fake = makeConnection();
    let deliverConnection!: (connection: unknown) => void;
    poolGetConnection.mockImplementation(() => new Promise((resolve) => {
      deliverConnection = resolve;
    }) as never);

    const acquisition = manager.acquireModelSlot();
    const error = await acquisition.catch((caught) => caught);
    expect(error).toBeInstanceOf(ModelServiceUnavailableError);
    expect(error).toMatchObject({ code: "MODEL_LOCK_CONNECTION_TIMEOUT" });

    deliverConnection(fake.connection);
    await Bun.sleep(1);
    expect(fake.release).toHaveBeenCalledTimes(1);
    expect(fake.destroy).not.toHaveBeenCalled();
    expect(fake.query).not.toHaveBeenCalled();
  });

  test("aborts a pending pool acquisition at the operation deadline and releases a late connection", async () => {
    const fake = makeConnection();
    let deliverConnection!: (connection: unknown) => void;
    poolGetConnection.mockImplementation(() => new Promise((resolve) => {
      deliverConnection = resolve;
    }) as never);
    const controller = new AbortController();
    const deadline = new Error("operation deadline");

    const acquisition = manager.acquireModelSlot(controller.signal);
    controller.abort(deadline);

    await expect(acquisition).rejects.toBe(deadline);
    deliverConnection(fake.connection);
    await Bun.sleep(1);
    expect(fake.release).toHaveBeenCalledTimes(1);
    expect(fake.destroy).not.toHaveBeenCalled();
    expect(fake.query).not.toHaveBeenCalled();
  });

  test("destroys the acquisition connection when the operation aborts during a slot query", async () => {
    const fake = makeConnection();
    fake.setAcquisitionQuery(new Promise(() => {}));
    poolGetConnection.mockResolvedValue(fake.connection as never);
    const controller = new AbortController();
    const deadline = new Error("operation deadline");

    const acquisition = manager.acquireModelSlot(controller.signal);
    for (let attempt = 0; attempt < 20 && fake.query.mock.calls.length === 0; attempt += 1) {
      await Bun.sleep(1);
    }
    controller.abort(deadline);

    await expect(acquisition).rejects.toBe(deadline);
    expect(fake.destroy).toHaveBeenCalledTimes(1);
    expect(fake.release).not.toHaveBeenCalled();
  });
});
