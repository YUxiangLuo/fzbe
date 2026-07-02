process.env.E2E_SERVER ??= "1";

const { runServerStartupPreflight, serverConfig } = await import("./index");

await runServerStartupPreflight();
const server = Bun.serve(serverConfig);

console.log(`E2E backend server listening on ${server.url.href}`);
