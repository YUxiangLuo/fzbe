import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dir, "../../..");
const helperPath = path.join(projectRoot, "scripts/deployment-environment.sh");
const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) => rm(directory, {
      recursive: true,
      force: true,
    })),
  );
});

function runBash(script: string, args: string[] = [], env = process.env) {
  return Bun.spawnSync(["bash", "-c", script, "deployment-env-test", ...args], {
    cwd: projectRoot,
    env,
    stdout: "pipe",
    stderr: "pipe",
  });
}

describe("controlled deployment environment", () => {
  test("loads sensitive values only from the explicit service environment file", async () => {
    const workingDirectory = await mkdtemp(path.join(tmpdir(), "fzbe-deploy-env-"));
    temporaryDirectories.push(workingDirectory);
    const fakeBinDirectory = path.join(workingDirectory, "bin");
    await mkdir(fakeBinDirectory);
    const fakeSudo = path.join(fakeBinDirectory, "sudo");
    await writeFile(
      fakeSudo,
      `#!/usr/bin/env bash
set -euo pipefail
[[ "\${1-}" == "systemd-run" ]]
shift
environment_file=""
working_directory=""
declare -a controlled_environment=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --property=EnvironmentFile=*) environment_file="\${1#--property=EnvironmentFile=}" ;;
    --working-directory=*) working_directory="\${1#*=}" ;;
    --setenv=*) controlled_environment+=("\${1#*=}") ;;
    --quiet|--wait|--pipe|--collect|--service-type=exec|--uid=*|--gid=*) ;;
    *) break ;;
  esac
  shift
done
declare -a file_environment=()
while IFS= read -r line || [[ -n "$line" ]]; do
  case "\${line:0:1}" in
    ""|"#"|";") continue ;;
  esac
  file_environment+=("$line")
done < "$environment_file"
cd "$working_directory"
exec env -i "\${controlled_environment[@]}" "\${file_environment[@]}" "$@"
`,
    );
    await chmod(fakeSudo, 0o755);
    const environmentFile = path.join(workingDirectory, "service.env");
    await writeFile(
      environmentFile,
      [
        "DB_DATABASE=controlled_database",
        "HOST=0.0.0.0",
        "PORT=4317",
        "JWT_SECRET=controlled-jwt-secret",
        "ALLOW_LEGACY_PRODUCTION_PLAN_WRITES=false",
      ].join("\n"),
    );
    await writeFile(
      path.join(workingDirectory, ".env.local"),
      "DB_DATABASE=automatic-dotenv-must-not-load\n",
    );

    const result = runBash(
      `source "$1"
       run_bun_with_service_environment "$2" "$3" -e '
         process.stdout.write(JSON.stringify({
           database: process.env.DB_DATABASE,
           host: process.env.HOST,
           port: process.env.PORT,
           jwt: process.env.JWT_SECRET,
           rollout: process.env.ALLOW_LEGACY_PRODUCTION_PLAN_WRITES,
           callerOnly: process.env.ACADEMIC_TERM_BACKFILL_YEAR,
         }));
       '`,
      [helperPath, environmentFile, workingDirectory],
      {
        ...process.env,
        PATH: `${fakeBinDirectory}:${process.env.PATH ?? "/usr/bin:/bin"}`,
        DB_DATABASE: "caller_database",
        HOST: "attacker.invalid",
        PORT: "9999",
        JWT_SECRET: "caller-jwt-secret",
        ALLOW_LEGACY_PRODUCTION_PLAN_WRITES: "true",
        ACADEMIC_TERM_BACKFILL_YEAR: "caller-only-value",
      },
    );

    expect([result.exitCode, result.stderr.toString()]).toEqual([0, ""]);
    expect(JSON.parse(result.stdout.toString())).toEqual({
      database: "controlled_database",
      host: "0.0.0.0",
      port: "4317",
      jwt: "controlled-jwt-secret",
      rollout: "false",
    });
  });

  test("uses systemd's EnvironmentFile parser and detects snapshot drift", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fzbe-deploy-env-snapshot-"));
    temporaryDirectories.push(directory);
    const snapshot = path.join(directory, "snapshot.env");
    const current = path.join(directory, "current.env");
    await writeFile(snapshot, "DB_DATABASE=fz_pr_snapshot\nJWT_SECRET=literal-$VALUE\n");
    await writeFile(current, "DB_DATABASE=fz_pr_snapshot\nJWT_SECRET=literal-$VALUE\n");

    const matching = runBash(
      `source "$1"
       deployment_environment_files_match "$2" "$3"`,
      [helperPath, snapshot, current],
    );
    expect(matching.exitCode).toBe(0);

    await writeFile(current, "DB_DATABASE=fz_pr_changed\nJWT_SECRET=literal-$VALUE\n");
    const changed = runBash(
      `source "$1"
       deployment_environment_files_match "$2" "$3"`,
      [helperPath, snapshot, current],
    );
    expect(changed.exitCode).not.toBe(0);

    const helper = await Bun.file(helperPath).text();
    expect(helper).toContain("sudo systemd-run");
    expect(helper).toContain("--property=EnvironmentFile=$environment_file");
    expect(helper).not.toContain('--env-file="$environment_file"');
  });

  test("rejects extra EnvironmentFiles and sensitive systemd Environment overrides", () => {
    const extraFileResult = runBash(
      `source "$1"
       deployment_single_environment_file '/srv/fz/.env (ignore_errors=no) /tmp/override.env (ignore_errors=no)'`,
      [helperPath],
    );
    expect(extraFileResult.exitCode).not.toBe(0);
    expect(extraFileResult.stderr.toString()).toContain("Expected exactly one");

    for (const configuredEnvironment of [
      "NODE_ENV=production DB_HOST=wrong-db",
      'NODE_ENV=production "PORT=9000"',
      "JWT_SECRET=wrong-secret",
      "ALLOW_LEGACY_PRODUCTION_PLAN_WRITES=true",
    ]) {
      const result = runBash(
        `source "$1"
         deployment_environment_has_protected_override "$2"`,
        [helperPath, configuredEnvironment],
      );
      expect(result.exitCode).toBe(0);
    }

    const safeResult = runBash(
      `source "$1"
       deployment_environment_has_protected_override "$2"`,
      [helperPath, "NODE_ENV=production LOG_LEVEL=info"],
    );
    expect(safeResult.exitCode).not.toBe(0);
  });
});
