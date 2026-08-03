process.env.E2E_SERVER ??= "1";

const { ensureE2eDatabaseExists } = await import("../scripts/ensure-e2e-database");

// Must run before ./index is imported: importing it builds the connection pool
// and the startup preflight immediately queries the target schema.
const bootstrap = await ensureE2eDatabaseExists();
if (bootstrap === "created") {
  console.log(`Created empty E2E database ${process.env.DB_DATABASE}; globalSetup will load schema and seed data.`);
}

const { runServerStartupPreflight, serverConfig } = await import("./index");

await runServerStartupPreflight();
const server = Bun.serve(serverConfig);

console.log(`E2E backend server listening on ${server.url.href}`);
