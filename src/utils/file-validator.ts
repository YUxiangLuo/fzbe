import path from "path";
import { Buffer } from "node:buffer";
import { randomUUID } from "node:crypto";

export const FILESYSTEM_FILENAME_MAX_BYTES = 255;

/**
 * File validation utilities for secure file uploads
 * Implements multiple layers of validation:
 * 1. File extension whitelist
 * 2. MIME type validation
 * 3. Magic bytes (file signature) verification
 */

// Allowed file extensions by category
const ALLOWED_EXTENSIONS = {
  pdf: [".pdf"],
  csv: [".csv"],
} as const;

// Allowed MIME types by category
const ALLOWED_MIME_TYPES = {
  pdf: ["application/pdf"],
  csv: ["text/csv", "application/vnd.ms-excel", "text/plain"],
} as const;

// File magic bytes (signatures) for verification
const FILE_SIGNATURES = {
  pdf: [
    [0x25, 0x50, 0x44, 0x46], // %PDF
  ],
  csv: null, // CSV files are plain text, no reliable magic bytes
} as const;

export type FileType = keyof typeof ALLOWED_EXTENSIONS;

export interface FileValidationOptions {
  /** Maximum file size in bytes */
  maxSize?: number;
  /** Whether to check file magic bytes */
  checkMagicBytes?: boolean;
}

export class FileValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FileValidationError";
  }
}

/**
 * Validates file extension against whitelist
 */
function validateExtension(fileName: string, fileType: FileType): void {
  const ext = path.extname(fileName).toLowerCase();
  const allowedExts = ALLOWED_EXTENSIONS[fileType];

  if (!ext) {
    throw new FileValidationError("文件没有扩展名");
  }

  if (!(allowedExts as readonly string[]).includes(ext)) {
    throw new FileValidationError(
      `无效的文件扩展名 "${ext}"。允许的扩展名: ${allowedExts.join(", ")}`
    );
  }
}

/**
 * Validates MIME type against whitelist
 */
function validateMimeType(mimeType: string, fileType: FileType): void {
  const allowedMimes = ALLOWED_MIME_TYPES[fileType];

  if (!mimeType) {
    throw new FileValidationError("文件 MIME 类型缺失");
  }

  // Check if MIME type matches any allowed pattern
  const isValid = allowedMimes.some((allowed) => {
    const subtype = allowed.split("/")[1];
    // Exact match or contains check for flexibility
    return mimeType === allowed || (subtype && mimeType.includes(subtype));
  });

  if (!isValid) {
    throw new FileValidationError(
      `无效的 MIME 类型 "${mimeType}"。允许的类型: ${allowedMimes.join(", ")}`
    );
  }
}

/**
 * Validates file magic bytes (file signature)
 */
async function validateMagicBytes(
  file: File,
  fileType: FileType
): Promise<void> {
  const signatures = FILE_SIGNATURES[fileType];

  // Skip if no signature defined for this file type
  if (!signatures) {
    return;
  }

  // Read first few bytes of the file
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  // Check if file starts with any of the valid signatures
  const isValid = signatures.some((signature) => {
    if (bytes.length < signature.length) {
      return false;
    }
    return signature.every((byte, index) => bytes[index] === byte);
  });

  if (!isValid) {
    throw new FileValidationError(
      `文件签名不符合预期的 ${fileType.toUpperCase()} 格式`
    );
  }
}

/**
 * Validates file size
 */
function validateSize(fileSize: number, maxSize?: number): void {
  if (maxSize && fileSize > maxSize) {
    const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(2);
    const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(2);
    throw new FileValidationError(
      `文件大小 ${fileSizeMB}MB 超过了允许的最大值 ${maxSizeMB}MB`
    );
  }

  if (fileSize === 0) {
    throw new FileValidationError("文件为空");
  }
}

/**
 * Comprehensive file validation
 * Performs multiple security checks on uploaded files
 *
 * @param file - The file to validate
 * @param fileType - Expected file type (pdf, csv)
 * @param options - Additional validation options
 * @throws {FileValidationError} If validation fails
 */
export async function validateFile(
  file: File,
  fileType: FileType,
  options: FileValidationOptions = {}
): Promise<void> {
  const { maxSize, checkMagicBytes = true } = options;

  // 1. Validate file size
  validateSize(file.size, maxSize);

  // 2. Validate file extension
  validateExtension(file.name, fileType);

  // 3. Validate MIME type
  validateMimeType(file.type, fileType);

  // 4. Validate magic bytes (file signature)
  if (checkMagicBytes) {
    await validateMagicBytes(file, fileType);
  }
}

/**
 * Sanitizes filename to prevent directory traversal and other attacks
 */
export function sanitizeFilename(filename: string): string {
  // Remove any path components
  const basename = path.basename(filename);

  // Remove or replace dangerous characters
  // Allow: alphanumeric, dash, underscore, dot, Chinese characters
  const sanitized = basename.replace(/[^\w\u4e00-\u9fa5.-]/g, "_");

  // Prevent double extensions that could bypass filters
  const parts = sanitized.split(".");
  if (parts.length > 2) {
    // Keep only the last extension
    const ext = parts.pop();
    const name = parts.join("_");
    return `${name}.${ext}`;
  }

  return sanitized;
}

/** Keeps a filename within a filesystem component byte budget without
 * splitting a Unicode code point, while preserving its validated extension. */
export function fitFilenameToUtf8Bytes(filename: string, maxBytes: number): string {
  if (!Number.isInteger(maxBytes) || maxBytes <= 0) {
    throw new Error("Filename byte budget must be a positive integer");
  }

  const extension = path.extname(filename);
  const extensionBytes = Buffer.byteLength(extension, "utf8");
  if (extensionBytes >= maxBytes) {
    throw new Error("Filename extension exceeds the filesystem byte budget");
  }

  const stem = path.basename(filename, extension);
  const stemBudget = maxBytes - extensionBytes;
  let boundedStem = "";
  let usedBytes = 0;
  for (const character of stem) {
    const characterBytes = Buffer.byteLength(character, "utf8");
    if (usedBytes + characterBytes > stemBudget) break;
    boundedStem += character;
    usedBytes += characterBytes;
  }
  if (!boundedStem) {
    boundedStem = "file".slice(0, stemBudget);
  }
  return `${boundedStem}${extension}`;
}

function fitFilenameToCharacterCount(filename: string, maxCharacters: number): string {
  if (!Number.isInteger(maxCharacters) || maxCharacters <= 0) {
    throw new Error("Filename character budget must be a positive integer");
  }
  const extension = path.extname(filename);
  const extensionCharacters = Array.from(extension).length;
  if (extensionCharacters >= maxCharacters) {
    throw new Error("Filename extension exceeds the storage path character budget");
  }
  const stemBudget = maxCharacters - extensionCharacters;
  const boundedStem = Array.from(path.basename(filename, extension))
    .slice(0, stemBudget)
    .join("") || "file".slice(0, stemBudget);
  return `${boundedStem}${extension}`;
}

export function fitFilenameToStoragePathLimits(
  filename: string,
  options: {
    directory: string;
    fixedPrefix?: string;
    maxStoredPathCharacters?: number;
  },
): string {
  const fixedPrefix = options.fixedPrefix ?? "";
  if (fixedPrefix.includes("/") || fixedPrefix.includes("\\")) {
    throw new Error("Stored filename prefix cannot contain path separators");
  }
  const maxStoredPathCharacters = options.maxStoredPathCharacters ?? 255;
  const pathPrefix = `${path.resolve(options.directory)}${path.sep}${fixedPrefix}`;
  const remainingCharacters = maxStoredPathCharacters - Array.from(pathPrefix).length;
  const remainingComponentBytes = (
    FILESYSTEM_FILENAME_MAX_BYTES - Buffer.byteLength(fixedPrefix, "utf8")
  );

  const byteBounded = fitFilenameToUtf8Bytes(filename, remainingComponentBytes);
  const characterBounded = fitFilenameToCharacterCount(byteBounded, remainingCharacters);
  return `${fixedPrefix}${characterBounded}`;
}

/**
 * Generates a unique, safe filename
 */
export function generateUniqueFilename(
  originalFilename: string,
  options?: { directory: string; maxStoredPathCharacters?: number },
): string {
  const sanitized = sanitizeFilename(originalFilename);
  const timestamp = Date.now();
  const prefix = `${timestamp}-${randomUUID()}-`;
  const boundedFilename = options
    ? fitFilenameToStoragePathLimits(sanitized, {
      ...options,
      fixedPrefix: prefix,
    }).slice(prefix.length)
    : fitFilenameToUtf8Bytes(
      sanitized,
      FILESYSTEM_FILENAME_MAX_BYTES - Buffer.byteLength(prefix, "utf8"),
    );

  return `${prefix}${boundedFilename}`;
}
