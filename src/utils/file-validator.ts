import path from "path";

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
    throw new FileValidationError("File has no extension");
  }

  if (!allowedExts.includes(ext as any)) {
    throw new FileValidationError(
      `Invalid file extension "${ext}". Allowed: ${allowedExts.join(", ")}`
    );
  }
}

/**
 * Validates MIME type against whitelist
 */
function validateMimeType(mimeType: string, fileType: FileType): void {
  const allowedMimes = ALLOWED_MIME_TYPES[fileType];

  if (!mimeType) {
    throw new FileValidationError("File MIME type is missing");
  }

  // Check if MIME type matches any allowed pattern
  const isValid = allowedMimes.some((allowed) => {
    // Exact match or contains check for flexibility
    return mimeType === allowed || mimeType.includes(allowed.split("/")[1]);
  });

  if (!isValid) {
    throw new FileValidationError(
      `Invalid MIME type "${mimeType}". Allowed: ${allowedMimes.join(", ")}`
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
      `File signature does not match expected ${fileType.toUpperCase()} format`
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
      `File size ${fileSizeMB}MB exceeds maximum allowed size of ${maxSizeMB}MB`
    );
  }

  if (fileSize === 0) {
    throw new FileValidationError("File is empty");
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

/**
 * Generates a unique, safe filename
 */
export function generateUniqueFilename(originalFilename: string): string {
  const sanitized = sanitizeFilename(originalFilename);
  const ext = path.extname(sanitized);
  const nameWithoutExt = path.basename(sanitized, ext);
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);

  return `${timestamp}-${random}-${nameWithoutExt}${ext}`;
}
