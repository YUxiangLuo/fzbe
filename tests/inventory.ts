import { readdir } from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(import.meta.dir, "..");
const IGNORED_DIRECTORIES = new Set([
  ".git",
  "__pycache__",
  "node_modules",
  "venv",
]);

async function walk(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    if (entry.isDirectory() && IGNORED_DIRECTORIES.has(entry.name)) {
      continue;
    }
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walk(absolute));
    } else if (entry.isFile()) {
      files.push(path.relative(ROOT, absolute).split(path.sep).join("/"));
    }
  }
  return files;
}

function countByDirectory(files: string[], prefix: string, segment: number): Map<string, number> {
  const counts = new Map<string, number>();
  for (const file of files.filter((candidate) => candidate.startsWith(prefix))) {
    const key = file.split("/")[segment] ?? "(root)";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

function printGroup(title: string, counts: Map<string, number>): void {
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  console.log(`\n${title} (${total})`);
  for (const [name, count] of [...counts].sort(([left], [right]) => left.localeCompare(right))) {
    console.log(`  ${name.padEnd(18)} ${String(count).padStart(3)}`);
  }
}

const files = await walk(ROOT);
const bunTests = files.filter((file) => file.endsWith(".test.ts") || file.endsWith(".test.tsx"));
const deepTests = files.filter(
  (file) => file.startsWith("py/deep_tests/") && /^test_.+\.py$/.test(path.posix.basename(file)),
);
const workflowTests = files.filter(
  (file) =>
    file.startsWith("py/test/")
    && path.posix.dirname(file) === "py/test"
    && file.endsWith(".py")
    && !["py/test/__init__.py", "py/test/run.py"].includes(file),
);
const systemRunners = files.filter(
  (file) => file.startsWith("tests/system/") && file.endsWith(".ts"),
);

printGroup("Bun unit suites", countByDirectory(bunTests, "tests/unit/", 2));
printGroup("Python deep modules", new Map([["deep", deepTests.length]]));
printGroup("Python workflow scripts", new Map([["workflow", workflowTests.length]]));
printGroup("System test runners", countByDirectory(systemRunners, "tests/system/", 2));

const unclassified = [
  ...bunTests.filter((file) => !file.startsWith("tests/unit/")),
  ...files.filter(
    (file) =>
      /^test_.+\.py$/.test(path.posix.basename(file))
      && !file.startsWith("py/deep_tests/"),
  ),
];

const knownUnitDirectories = new Set([
  "config",
  "controllers",
  "models",
  "policies",
  "routing",
  "schemas",
  "security",
  "services",
  "utils",
]);
const unknownUnitDirectories = bunTests
  .filter((file) => file.startsWith("tests/unit/"))
  .map((file) => file.split("/")[2])
  .filter((directory): directory is string => Boolean(directory) && !knownUnitDirectories.has(directory));

if (systemRunners.length === 0) {
  console.error(
    "\nExpected at least one TypeScript system test runner in tests/system/.",
  );
  process.exitCode = 1;
}

if (unclassified.length > 0 || unknownUnitDirectories.length > 0) {
  console.error("\nUnclassified tests:");
  for (const file of [...new Set([...unclassified, ...unknownUnitDirectories])].sort()) {
    console.error(`  ${file}`);
  }
  process.exitCode = 1;
} else if (!process.exitCode) {
  console.log(
    `\nInventory OK: ${bunTests.length + deepTests.length + workflowTests.length + systemRunners.length} `
    + "test files/scripts are classified.",
  );
}
