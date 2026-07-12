import { beforeEach, describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const modelLocksModulePath = path.join(srcRoot, "modules", "models", "model-locks.ts");
const lockResults: Array<0 | 1> = [];
const lockNames: string[] = [];
const releaseNames: string[] = [];
const releasedConnections: string[] = [];
const destroyedConnections: string[] = [];
let failNextGetLock = false;
let failNextReleaseLock = false;

const getConnection = mock(async () => {
  const connectionId = `connection-${releasedConnections.length + lockNames.length + 1}`;
  return {
    query: mock(async (sql: string, params: unknown[]) => {
      if (sql.includes("GET_LOCK")) {
        if (failNextGetLock) {
          failNextGetLock = false;
          throw new Error("GET_LOCK response lost");
        }
        lockNames.push(String(params[0]));
        return [[{ acquired: lockResults.shift() ?? 1 }]];
      }
      if (sql.includes("RELEASE_LOCK")) {
        if (failNextReleaseLock) {
          failNextReleaseLock = false;
          throw new Error("RELEASE_LOCK response lost");
        }
        releaseNames.push(String(params[0]));
        return [[{ released: 1 }]];
      }
      return [[]];
    }),
    release: mock(() => {
      releasedConnections.push(connectionId);
    }),
    destroy: mock(() => {
      destroyedConnections.push(connectionId);
    }),
  };
});

mock.module(configModulePath, () => ({
  pool: { getConnection },
}));

const {
  acquireModelArtifactMutationLocks,
  releaseModelArtifactMutationLocks,
} = await import(`${modelLocksModulePath}?model-locks-test`);

describe("model artifact mutation locks", () => {
  beforeEach(() => {
    lockResults.length = 0;
    lockNames.length = 0;
    releaseNames.length = 0;
    releasedConnections.length = 0;
    destroyedConnections.length = 0;
    failNextGetLock = false;
    failNextReleaseLock = false;
    getConnection.mockClear();
  });

  test("uses one experiment-wide lock for a base-model artifact mutation", async () => {
    const handles = await acquireModelArtifactMutationLocks(42, "ma");

    expect(handles).toHaveLength(1);
    expect(lockNames).toEqual(["fangzhen-model:exp:42"]);

    await releaseModelArtifactMutationLocks(handles);
    expect(releaseNames.toSorted()).toEqual(lockNames.toSorted());
    expect(releasedConnections).toHaveLength(1);
  });

  test("uses the same experiment-wide lock for an ensemble mutation", async () => {
    const handles = await acquireModelArtifactMutationLocks(42, "weighted_avg");

    expect(handles).toHaveLength(1);
    expect(lockNames).toEqual(["fangzhen-model:exp:42"]);
  });

  test("returns null without retaining a connection when the experiment is busy", async () => {
    lockResults.push(0);

    const handles = await acquireModelArtifactMutationLocks(42, "ma");

    expect(handles).toBeNull();
    expect(lockNames).toEqual(["fangzhen-model:exp:42"]);
    expect(releaseNames).toEqual([]);
    expect(releasedConnections).toHaveLength(1);
  });

  test("destroys a connection when GET_LOCK ownership is unknown", async () => {
    failNextGetLock = true;

    await expect(
      acquireModelArtifactMutationLocks(42, "ma"),
    ).rejects.toThrow("GET_LOCK response lost");

    expect(releasedConnections).toHaveLength(0);
    expect(destroyedConnections).toHaveLength(1);
  });

  test("destroys a connection when RELEASE_LOCK ownership is unknown", async () => {
    const handles = await acquireModelArtifactMutationLocks(42, "ma");
    failNextReleaseLock = true;

    await releaseModelArtifactMutationLocks(handles);

    expect(releasedConnections).toHaveLength(0);
    expect(destroyedConnections).toHaveLength(1);
  });
});
