import { describe, expect, test } from "bun:test";
import { Buffer } from "node:buffer";
import {
  FILESYSTEM_FILENAME_MAX_BYTES,
  fitFilenameToUtf8Bytes,
  generateUniqueFilename,
} from "../../../src/utils/file-validator";

describe("filesystem-safe upload filenames", () => {
  test("bounds long CJK upload names by UTF-8 bytes and preserves the extension", () => {
    const filename = generateUniqueFilename(`${"超长教学资料".repeat(100)}.pdf`);

    expect(Buffer.byteLength(filename, "utf8")).toBeLessThanOrEqual(
      FILESYSTEM_FILENAME_MAX_BYTES,
    );
    expect(filename).toEndWith(".pdf");
    expect(filename).not.toContain("�");
  });

  test("never splits a multibyte code point at an exact byte boundary", () => {
    const filename = fitFilenameToUtf8Bytes("教学资料.csv", 10);

    expect(filename).toBe("教学.csv");
    expect(Buffer.byteLength(filename, "utf8")).toBe(10);
  });

  test("also bounds the complete absolute path stored in VARCHAR(255)", () => {
    const directory = "/srv/fangzhen/uploads/manuals/.db/0123456789abcdef";
    const filename = generateUniqueFilename(`${"a".repeat(500)}.pdf`, { directory });
    const storedPath = `${directory}/${filename}`;

    expect(Array.from(storedPath).length).toBeLessThanOrEqual(255);
    expect(Buffer.byteLength(filename, "utf8")).toBeLessThanOrEqual(255);
    expect(filename).toEndWith(".pdf");
  });
});
