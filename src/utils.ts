import fs from "fs";
import path from "path";
import csv from "csv-parser";

export interface Segment {
  行业名称: string;
  公司名称: string;
  产品名称: string;
  起始位置: number; // 不含表头，从 1 开始
  结束位置: number;
  filePath: string; // ✅ 新增：源文件绝对路径
}

export interface ParseOptions {
  separator?: string;
  strict?: boolean;
  makeKey?: (row: Record<string, string>) => string;
}

function defaultMakeKey(row: Record<string, string>) {
  const industry = (row["行业名称"] ?? "").trim();
  const company = (row["公司名称"] ?? "").trim();
  const product = (row["产品名称"] ?? "").trim();
  return `${industry}|${company}|${product}`;
}

export function makeKeyWithCodes(row: Record<string, string>) {
  const parts = [
    (row["行业名称"] ?? "").trim(),
    (row["公司代码"] ?? "").trim(),
    (row["公司名称"] ?? "").trim(),
    (row["产品编码"] ?? "").trim(),
    (row["产品名称"] ?? "").trim(),
  ];
  return parts.join("|");
}

/**
 * 解析 CSV 为“连续分段”：
 * - 同一键的相邻行归为同一段；
 * - strict=true 时，若同一键非连续重复出现，立刻抛错；
 */
export function parseCsvSegments(
  filePath: string,
  options: ParseOptions = {},
): Promise<Segment[]> {
  const { separator = ",", strict = true, makeKey = defaultMakeKey } = options;

  const absPath = path.resolve(filePath);

  return new Promise((resolve, reject) => {
    let rowIndex = 0;
    let prevKey: string | undefined;

    const segmentsByKey = new Map<string, Segment[]>();
    const lastActiveByKey = new Map<string, number>();
    const closedKeys = new Set<string>();

    fs.createReadStream(absPath)
      .pipe(
        csv({
          separator,
          mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
        }),
      )
      .on("data", (row: Record<string, string>) => {
        const industry = (row["行业名称"] ?? "").trim();
        const company = (row["公司名称"] ?? "").trim();
        const product = (row["产品名称"] ?? "").trim();
        if (!industry || !company || !product) return;

        rowIndex += 1;
        const key = makeKey(row);

        if (prevKey === key) {
          const list = segmentsByKey.get(key);
          const idx = lastActiveByKey.get(key);

          if (!list || idx === undefined || idx < 0 || idx >= list.length) {
            reject(new Error(`内部状态异常：key=${key}`));
            return;
          }

          const seg = list[idx];
          if (!seg) {
            reject(new Error(`内部状态异常：list[${idx}] 不存在，key=${key}`));
            return;
          }
          seg.结束位置 = rowIndex;
        } else {
          if (prevKey !== undefined) closedKeys.add(prevKey);
          if (strict && closedKeys.has(key)) {
            reject(new Error(`检测到不连续：${key} 在 ${absPath}:${rowIndex}`));
            return;
          }

          let list = segmentsByKey.get(key);
          if (!list) {
            list = [];
            segmentsByKey.set(key, list);
          }

          const seg: Segment = {
            行业名称: industry,
            公司名称: company,
            产品名称: product,
            起始位置: rowIndex,
            结束位置: rowIndex,
            filePath: absPath, // ✅ 带上文件路径
          };
          list.push(seg);
          lastActiveByKey.set(key, list.length - 1);
        }

        prevKey = key;
      })
      .on("end", () => {
        const all: Segment[] = [];
        for (const list of segmentsByKey.values()) {
          all.push(...list);
        }
        all.sort((a, b) => a.起始位置 - b.起始位置);
        resolve(all);
      })
      .on("error", reject);
  });
}

/**
 * 宽松模式解析后统一断言：若任一键出现多段（不连续），抛出错误并列出问题项及区间预览。
 */
export async function assertCsvContiguous(
  filePath: string,
  options: Omit<ParseOptions, "strict"> = {},
): Promise<void> {
  const absPath = path.resolve(filePath);
  const segs = await parseCsvSegments(absPath, { ...options, strict: false });

  const byKey = new Map<string, Segment[]>();
  for (const s of segs) {
    const key = (options.makeKey ?? defaultMakeKey)({
      行业名称: s.行业名称,
      公司名称: s.公司名称,
      产品名称: s.产品名称,
    } as any);
    const list = byKey.get(key);
    if (list) list.push(s);
    else byKey.set(key, [s]);
  }

  const offenders: Array<{ key: string; segments: Segment[] }> = [];
  for (const [key, list] of byKey.entries()) {
    if (list.length > 1) {
      list.sort((a, b) => a.起始位置 - b.起始位置);
      offenders.push({ key, segments: list });
    }
  }

  if (offenders.length > 0) {
    const preview = offenders
      .slice(0, 10)
      .map(({ key, segments }) => {
        const ranges = segments
          .slice(0, 5)
          .map((s) => `[${s.起始位置}-${s.结束位置}]`)
          .join(", ");
        const more = segments.length > 5 ? ` ...共 ${segments.length} 段` : "";
        return `- ${key} => ${ranges}${more}`;
      })
      .join("\n");

    const moreKeys =
      offenders.length > 10
        ? `\n... 共 ${offenders.length} 个不连续的商品`
        : "";

    throw new Error(
      `文件: ${absPath}\n检测到不连续的同一商品销量数据（同一键出现多段）：\n${preview}${moreKeys}`,
    );
  }
}
