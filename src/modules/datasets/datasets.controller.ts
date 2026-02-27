import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { jwtAuth, requireAdmin } from "../../middleware";
import {
  industryParamSchema, industryAndCompanyParamSchema, industryCompanyProductParamSchema,
  datasetIdParamSchema, updateDatasetSchema,
} from "./datasets.schema";
import {
  uploadDataset, getAllDatasets, getDatasetById, updateDataset, deleteDataset,
  getIndustries, getCompaniesByIndustry, getProductsByCompany, getProductFields, getProductSales,
  FileValidationError,
} from "./datasets.service";
import { handleUnexpectedError } from "../../utils/error-handler";

const datasetsController = new Hono();

datasetsController.post("/api/v1/datasets", jwtAuth, requireAdmin, async (c) => {
  try {
    const body = await c.req.parseBody();
    const file = body["file"];
    const dataName = body["data_name"];
    const description = body["description"];

    if (!(file instanceof File)) return c.json({ error: "需要 CSV 文件" }, 400);
    if (!dataName || typeof dataName !== "string") return c.json({ error: "数据集名称不能为空" }, 400);

    const result = await uploadDataset(file, dataName, typeof description === "string" ? description : undefined);

    if ("error" in result) {
      const response: Record<string, unknown> = { error: result.error };
      if ("details" in result) Object.assign(response, result.details);
      return c.json(response, result.status);
    }
    return c.json(result.data, 201);
  } catch (error) {
    if (error instanceof FileValidationError) return c.json({ error: error.message }, 400);
    return handleUnexpectedError(c, error, "上传数据集失败");
  }
});

datasetsController.get("/api/v1/datasets", jwtAuth, requireAdmin, async (c) => {
  try {
    const datasets = await getAllDatasets();
    return c.json(datasets);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取数据集列表失败");
  }
});

datasetsController.get("/api/v1/datasets/industries", jwtAuth, async (c) => {
  try {
    const industries = await getIndustries();
    return c.json(industries);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取行业列表失败");
  }
});

datasetsController.get("/api/v1/datasets/industries/:industry/companies", jwtAuth, zValidator("param", industryParamSchema), async (c) => {
  try {
    const { industry } = c.req.valid("param");
    const companies = await getCompaniesByIndustry(industry);
    return c.json(companies);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取公司列表失败");
  }
});

datasetsController.get("/api/v1/datasets/industries/:industry/companies/:company/products", jwtAuth, zValidator("param", industryAndCompanyParamSchema), async (c) => {
  try {
    const { industry, company } = c.req.valid("param");
    const products = await getProductsByCompany(industry, company);
    return c.json(products);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取产品列表失败");
  }
});

datasetsController.get("/api/v1/datasets/industries/:industry/companies/:company/products/:product/fields", jwtAuth, zValidator("param", industryCompanyProductParamSchema), async (c) => {
  try {
    const { industry, company, product } = c.req.valid("param");
    const result = await getProductFields(industry, company, product);
    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "未找到产品分段" }, 404);
      return c.json({ error: "产品分段文件丢失" }, 404);
    }
    return c.json({ fields: result.data });
  } catch (error) {
    return handleUnexpectedError(c, error, "获取产品字段失败");
  }
});

datasetsController.get("/api/v1/datasets/industries/:industry/companies/:company/products/:product/sales", jwtAuth, zValidator("param", industryCompanyProductParamSchema), async (c) => {
  try {
    const { industry, company, product } = c.req.valid("param");
    const result = await getProductSales(industry, company, product);
    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "未找到产品分段" }, 404);
      return c.json({ error: "产品销售数据文件丢失" }, 404);
    }
    return c.json(result.data);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取产品销售数据失败");
  }
});

datasetsController.get("/api/v1/datasets/:datasetId", jwtAuth, requireAdmin, zValidator("param", datasetIdParamSchema), async (c) => {
  try {
    const { datasetId } = c.req.valid("param");
    const dataset = await getDatasetById(parseInt(datasetId, 10));
    if (!dataset) return c.json({ error: "数据集不存在" }, 404);
    return c.json(dataset);
  } catch (error) {
    return handleUnexpectedError(c, error, "获取数据集失败");
  }
});

datasetsController.put("/api/v1/datasets/:datasetId", jwtAuth, requireAdmin, zValidator("param", datasetIdParamSchema), zValidator("json", updateDatasetSchema), async (c) => {
  try {
    const { datasetId } = c.req.valid("param");
    const input = c.req.valid("json");
    const result = await updateDataset(parseInt(datasetId, 10), input);

    if ("error" in result) {
      if (result.error === "not_found") return c.json({ error: "数据集不存在" }, 404);
      return c.json({ error: "没有需要更新的字段" }, 400);
    }
    return c.json(result.data);
  } catch (error) {
    return handleUnexpectedError(c, error, "更新数据集失败");
  }
});

datasetsController.delete("/api/v1/datasets/:datasetId", jwtAuth, requireAdmin, zValidator("param", datasetIdParamSchema), async (c) => {
  try {
    const { datasetId } = c.req.valid("param");
    const result = await deleteDataset(parseInt(datasetId, 10));
    if ("error" in result) return c.json({ error: "数据集不存在" }, 404);
    return c.json({ message: "数据集及所有相关文件删除成功" });
  } catch (error) {
    return handleUnexpectedError(c, error, "删除数据集失败");
  }
});

export default datasetsController;
