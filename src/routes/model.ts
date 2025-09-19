import { Hono } from "hono";
import { jwt } from "hono/jwt";

export default (JWT_SECRET: string) => {
  const modelApi = new Hono();

  modelApi.use("*", jwt({ secret: JWT_SECRET }));

  modelApi.get("/model/hello", (c) => {
    return c.json({ message: "Model service placeholder" });
  });

  return modelApi;
};
