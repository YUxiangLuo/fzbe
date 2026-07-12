import path from "node:path";

const projectRoot = path.resolve(import.meta.dir, "..");

async function runMigration(scriptPath: string) {
  const processHandle = Bun.spawn(
    [process.execPath, "run", scriptPath],
    {
      cwd: projectRoot,
      env: process.env,
      stdout: "inherit",
      stderr: "inherit",
    },
  );
  const exitCode = await processHandle.exited;
  if (exitCode !== 0) {
    throw new Error(`Migration failed (${scriptPath}) with exit code ${exitCode}`);
  }
}

await runMigration("scripts/migrate-guided-training-sessions.ts");
await runMigration("scripts/migrate-model-training-integrity.ts");
console.log("Model training migrations complete.");
