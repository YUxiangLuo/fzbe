import path from "node:path";

const backendRoot = path.resolve(import.meta.dir, "..");
const pythonRoot = path.join(backendRoot, "py");
const outputPath = path.join(pythonRoot, "production-dependencies.lock");
const uvRunner = path.join(pythonRoot, "run-pinned-uv.sh");
const checkOnly = process.argv.includes("--check");

const child = Bun.spawn([
  uvRunner,
  "export",
  "--project",
  pythonRoot,
  "--locked",
  "--format",
  "requirements.txt",
  "--no-dev",
  "--no-hashes",
  "--no-header",
  "--no-annotate",
  "--no-emit-project",
], {
  cwd: backendRoot,
  stdout: "pipe",
  stderr: "pipe",
});

const [exitCode, stdout, stderr] = await Promise.all([
  child.exited,
  new Response(child.stdout).text(),
  new Response(child.stderr).text(),
]);
if (exitCode !== 0) {
  throw new Error(`Unable to export production Python dependencies:\n${stderr.trim()}`);
}

const generated = stdout.endsWith("\n") ? stdout : `${stdout}\n`;
if (checkOnly) {
  const committed = await Bun.file(outputPath).text().catch(() => null);
  if (committed !== generated) {
    throw new Error(
      "Production Python dependency lock is stale. Run bun run python:lock:generate.",
    );
  }
  console.log("Production Python dependency lock matches uv.lock without development groups.");
} else {
  await Bun.write(outputPath, generated);
  console.log(`Generated ${outputPath}`);
}
