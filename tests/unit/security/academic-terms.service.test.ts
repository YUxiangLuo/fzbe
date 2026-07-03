import { describe, expect, mock, test } from "bun:test";
import path from "node:path";

mock.restore();

const srcRoot = path.resolve(import.meta.dir, "../../../src");
const configModulePath = path.join(srcRoot, "config", "index.ts");
const academicTermsServiceModulePath = path.join(srcRoot, "modules", "academic-terms", "academic-terms.service.ts");
const freshAcademicTermsServiceModulePath = `${academicTermsServiceModulePath}?academic-terms-service-test`;

const poolQuery = mock(async () => [[]]);

mock.module(configModulePath, () => ({
  env: {
    DB_HOST: process.env.DB_HOST ?? "127.0.0.1",
    DB_USER: process.env.DB_USER ?? "test",
    DB_PASSWORD: process.env.DB_PASSWORD ?? "test",
    DB_DATABASE: process.env.DB_DATABASE ?? "test",
    JWT_SECRET: "test-jwt-secret",
    PORT: Number(process.env.PORT ?? "3001"),
  },
  connectionLimit: 64,
  pool: {
    query: poolQuery,
  },
}));

const {
  formatAcademicTermLabel,
  getDefaultClassTermId,
  mapAcademicTermRow,
  resolveAcademicTermId,
} = await import(freshAcademicTermsServiceModulePath);

const activeTermRow = {
  term_id: 7,
  academic_year: "2026-2027",
  semester: 1,
  is_active: 1,
  created_at: new Date("2026-07-01T00:00:00Z"),
};

function createQueryable(...rowSets: unknown[][]) {
  const query = mock(async () => [rowSets.shift() ?? []]);
  return {
    db: { query } as never,
    query,
  };
}

describe("academic terms service", () => {
  test("formats and maps academic term rows consistently", () => {
    expect(formatAcademicTermLabel("2026-2027", 2)).toBe("2026-2027 第2学期");
    expect(mapAcademicTermRow({ ...activeTermRow, class_count: "3" })).toMatchObject({
      term_id: 7,
      academic_year: "2026-2027",
      semester: 1,
      term_label: "2026-2027 第1学期",
      is_active: true,
      class_count: 3,
    });
  });

  test("leaves term filters unset when no term id is provided", async () => {
    const { db, query } = createQueryable();

    await expect(resolveAcademicTermId(undefined, db)).resolves.toEqual({ termId: undefined });
    await expect(resolveAcademicTermId("", db)).resolves.toEqual({ termId: undefined });
    expect(query).not.toHaveBeenCalled();
  });

  test("resolves the current academic term from the active row", async () => {
    const { db, query } = createQueryable([activeTermRow]);

    await expect(resolveAcademicTermId("current", db)).resolves.toEqual({ termId: 7 });
    expect(query).toHaveBeenCalledTimes(1);
  });

  test("rejects invalid and missing term ids", async () => {
    const invalidDb = createQueryable();
    await expect(resolveAcademicTermId("abc", invalidDb.db)).resolves.toEqual({ error: "invalid_term" });
    expect(invalidDb.query).not.toHaveBeenCalled();

    const missingDb = createQueryable([]);
    await expect(resolveAcademicTermId("99", missingDb.db)).resolves.toEqual({ error: "invalid_term" });
    expect(missingDb.query).toHaveBeenCalledTimes(1);
  });

  test("requires a current academic term when defaulting class creation", async () => {
    const { db } = createQueryable([]);

    await expect(getDefaultClassTermId(db)).resolves.toEqual({ error: "no_current_term" });
  });
});
