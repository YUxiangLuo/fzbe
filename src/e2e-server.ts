import { serverConfig } from "./index";

const server = Bun.serve(serverConfig);

console.log(`E2E backend server listening on ${server.url.href}`);
