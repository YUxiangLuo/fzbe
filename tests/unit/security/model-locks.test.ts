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

const getConnection = mock(async () => {
  const connectionId = `connection-${releasedConnections.length + lockNames.length + 1}`;
  return {
    query: mock(async (sql: string, params: unknown[]) => {
      if (sql.includes("GET_LOCK")) {
        lockNames.push(String(params[0]));
        return [[{ acquired: lockResults.shift() ?? 1 }]];
      }
      if (sql.includes("RELEASE_LOCK")) {
        releaseNames.push(String(params[0]));
        return [[{ released: 1 }]];
      }
      return [[]];
    }),
    release: mock(() => {
      releasedConnections.push(connectionId);
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
    getConnection.mockClear();
  });

  test("locks a base model and every dependent ensemble in stable order", async () => {
    const handles = await acquireModelArtifactMutationLocks(42, "ma");

    expect(handles).toHaveLength(4);
    expect(lockNames).toEqual([
      "fangzhen-model:exp:42:boosting",
      "fangzhen-model:exp:42:ma",
      "fangzhen-model:exp:42:stacking",
      "fangzhen-model:exp:42:weighted_avg",
    ]);

    await releaseModelArtifactMutationLocks(handles);
    expect(releaseNames.toSorted()).toEqual(lockNames.toSorted());
    expect(releasedConnections).toHaveLength(4);
  });

  test("locks only the selected ensemble when its artifact changes", async () => {
    const handles = await acquireModelArtifactMutationLocks(42, "weighted_avg");

    expect(handles).toHaveLength(1);
    expect(lockNames).toEqual(["fangzhen-model:exp:42:weighted_avg"]);
  });

  test("releases the partial lock set when a later lock is busy", async () => {
    lockResults.push(1, 0);

    const handles = await acquireModelArtifactMutationLocks(42, "ma");

    expect(handles).toBeNull();
    expect(lockNames).toEqual([
      "fangzhen-model:exp:42:boosting",
      "fangzhen-model:exp:42:ma",
    ]);
    expect(releaseNames).toEqual(["fangzhen-model:exp:42:boosting"]);
    expect(releasedConnections).toHaveLength(2);
  });
});
