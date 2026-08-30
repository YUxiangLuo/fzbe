import path from "node:path";

const versionFile = path.resolve(import.meta.dir, "../.bun-version");
const expectedVersion = (await Bun.file(versionFile).text()).trim();

if (!/^\d+\.\d+\.\d+$/.test(expectedVersion)) {
  throw new Error(`Invalid Bun version in ${versionFile}: ${JSON.stringify(expectedVersion)}`);
}

if (Bun.version !== expectedVersion) {
  throw new Error(
    `Bun ${expectedVersion} is required, but the current runtime is ${Bun.version}.`,
  );
}

console.log(`Bun runtime ${Bun.version} matches .bun-version.`);
