import fs from "fs";
import csv from "csv-parser";

// Represents a segment of data for a specific product, including its metadata and all its data rows.
export interface SegmentData {
  industryName: string;
  companyName: string;
  productName: string;
  rows: Record<string, string>[]; // All data rows for this segment
}

/**
 * Parses a CSV file and groups its rows by a composite key (Industry|Company|Product).
 * This function reads the entire file into memory and returns a map where each entry
 * corresponds to a unique product segment.
 *
 * @param filePath The absolute path to the CSV file.
 * @returns A Promise that resolves to a Map<string, SegmentData>.
 */
export function parseCsvSegments(
  filePath: string,
): Promise<Map<string, SegmentData>> {
  return new Promise((resolve, reject) => {
    const segments = new Map<string, SegmentData>();

    fs.createReadStream(filePath)
      .pipe(
        csv({
          mapHeaders: ({ header }) => header.replace(/^\uFEFF/, "").trim(),
        }),
      )
      .on("data", (row: Record<string, string>) => {
        const industryName = (row["行业名称"] ?? "").trim();
        const companyName = (row["公司名称"] ?? "").trim();
        const productName = (row["产品名称"] ?? "").trim();

        if (!industryName || !companyName || !productName) {
          return; // Skip rows that don't have the required identifiers
        }

        const key = `${industryName}|${companyName}|${productName}`;
        let segment = segments.get(key);

        if (!segment) {
          segment = {
            industryName,
            companyName,
            productName,
            rows: [],
          };
          segments.set(key, segment);
        }

        segment.rows.push(row);
      })
      .on("end", () => {
        resolve(segments);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}
