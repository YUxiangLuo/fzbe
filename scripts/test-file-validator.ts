#!/usr/bin/env bun
/**
 * Test script for file validation utilities
 * Run with: bun run scripts/test-file-validator.ts
 */

import { validateFile, sanitizeFilename, generateUniqueFilename, FileValidationError } from "../src/utils/file-validator";
import { sanitizeHTML, HTMLSanitizationError } from "../src/utils/html-sanitizer";

console.log("=".repeat(60));
console.log("File Validation Utilities Test Suite");
console.log("=".repeat(60));

// Test 1: Valid PDF file
console.log("\n[Test 1] Valid PDF file");
try {
  const pdfContent = new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2D, 0x31, 0x2E, 0x34]); // %PDF-1.4
  const validPdf = new File([pdfContent], "test.pdf", { type: "application/pdf" });

  await validateFile(validPdf, "pdf", { maxSize: 1024 * 1024, checkMagicBytes: true });
  console.log("✅ PASS: Valid PDF accepted");
} catch (error) {
  console.log(`❌ FAIL: ${error instanceof Error ? error.message : String(error)}`);
}

// Test 2: Invalid PDF (wrong magic bytes)
console.log("\n[Test 2] Invalid PDF (wrong magic bytes)");
try {
  const fakePdfContent = new Uint8Array([0x00, 0x01, 0x02, 0x03]);
  const fakePdf = new File([fakePdfContent], "fake.pdf", { type: "application/pdf" });

  await validateFile(fakePdf, "pdf", { maxSize: 1024 * 1024, checkMagicBytes: true });
  console.log("❌ FAIL: Should have rejected fake PDF");
} catch (error) {
  if (error instanceof FileValidationError) {
    console.log(`✅ PASS: Correctly rejected - ${error.message}`);
  } else {
    console.log(`❌ FAIL: Wrong error type`);
  }
}

// Test 3: Wrong file extension
console.log("\n[Test 3] Wrong file extension");
try {
  const content = new Uint8Array([0x25, 0x50, 0x44, 0x46]);
  const wrongExt = new File([content], "test.txt", { type: "application/pdf" });

  await validateFile(wrongExt, "pdf", { maxSize: 1024 * 1024 });
  console.log("❌ FAIL: Should have rejected wrong extension");
} catch (error) {
  if (error instanceof FileValidationError && error.message.includes("extension")) {
    console.log(`✅ PASS: Correctly rejected - ${error.message}`);
  } else {
    console.log(`❌ FAIL: Wrong error: ${error instanceof Error ? error.message : String(error)}`);
  }
}

// Test 4: File too large
console.log("\n[Test 4] File too large");
try {
  const largeContent = new Uint8Array(10 * 1024 * 1024); // 10MB
  const largePdf = new File([largeContent], "large.pdf", { type: "application/pdf" });

  await validateFile(largePdf, "pdf", { maxSize: 5 * 1024 * 1024, checkMagicBytes: false });
  console.log("❌ FAIL: Should have rejected large file");
} catch (error) {
  if (error instanceof FileValidationError && error.message.includes("size")) {
    console.log(`✅ PASS: Correctly rejected - ${error.message}`);
  } else {
    console.log(`❌ FAIL: Wrong error`);
  }
}

// Test 5: Empty file
console.log("\n[Test 5] Empty file");
try {
  const emptyFile = new File([], "empty.pdf", { type: "application/pdf" });

  await validateFile(emptyFile, "pdf", { maxSize: 1024 * 1024 });
  console.log("❌ FAIL: Should have rejected empty file");
} catch (error) {
  if (error instanceof FileValidationError && error.message.includes("empty")) {
    console.log(`✅ PASS: Correctly rejected - ${error.message}`);
  } else {
    console.log(`❌ FAIL: Wrong error`);
  }
}

// Test 6: Valid CSV file
console.log("\n[Test 6] Valid CSV file");
try {
  const csvContent = "name,age\nJohn,30\nJane,25";
  const validCsv = new File([csvContent], "data.csv", { type: "text/csv" });

  await validateFile(validCsv, "csv", { maxSize: 1024 * 1024, checkMagicBytes: false });
  console.log("✅ PASS: Valid CSV accepted");
} catch (error) {
  console.log(`❌ FAIL: ${error instanceof Error ? error.message : String(error)}`);
}

// Test 7: Sanitize filename with special characters
console.log("\n[Test 7] Sanitize filename with special characters");
const dirtyFilenames = [
  "../../../etc/passwd",
  "file<>with|bad:chars.pdf",
  "double..extension.pdf.exe",
  "中文文件名.pdf",
  "normal-file_name.pdf"
];

for (const filename of dirtyFilenames) {
  const sanitized = sanitizeFilename(filename);
  console.log(`  "${filename}" → "${sanitized}"`);
}

// Test 8: Generate unique filenames
console.log("\n[Test 8] Generate unique filenames");
const originals = ["report.pdf", "data.csv", "文档.pdf"];
for (const original of originals) {
  const unique = generateUniqueFilename(original);
  console.log(`  "${original}" → "${unique}"`);
}

// Test 9: HTML Sanitization - Safe HTML
console.log("\n[Test 9] HTML Sanitization - Safe HTML");
try {
  const safeHTML = "<h1>Title</h1><p>This is <strong>safe</strong> content.</p>";
  const sanitized = sanitizeHTML(safeHTML);
  console.log("✅ PASS: Safe HTML accepted");
  console.log(`  Input:  ${safeHTML.substring(0, 50)}...`);
  console.log(`  Output: ${sanitized.substring(0, 50)}...`);
} catch (error) {
  console.log(`❌ FAIL: ${error instanceof Error ? error.message : String(error)}`);
}

// Test 10: HTML Sanitization - Malicious script
console.log("\n[Test 10] HTML Sanitization - Malicious script");
try {
  const maliciousHTML = '<h1>Title</h1><script>alert("XSS")</script><p>Content</p>';
  const sanitized = sanitizeHTML(maliciousHTML);
  console.log("❌ FAIL: Should have rejected malicious HTML");
} catch (error) {
  if (error instanceof HTMLSanitizationError) {
    console.log(`✅ PASS: Correctly rejected - ${error.message}`);
  } else {
    console.log(`❌ FAIL: Wrong error type`);
  }
}

// Test 11: HTML Sanitization - Event handlers
console.log("\n[Test 11] HTML Sanitization - Event handlers");
try {
  const xssHTML = '<img src=x onerror="alert(1)"><p>Content</p>';
  const sanitized = sanitizeHTML(xssHTML);
  console.log("❌ FAIL: Should have rejected event handlers");
} catch (error) {
  if (error instanceof HTMLSanitizationError) {
    console.log(`✅ PASS: Correctly rejected - ${error.message}`);
  } else {
    console.log(`❌ FAIL: Wrong error type`);
  }
}

// Test 12: HTML Sanitization - javascript: protocol
console.log("\n[Test 12] HTML Sanitization - javascript: protocol");
try {
  const jsProtocol = '<a href="javascript:alert(1)">Click</a>';
  const sanitized = sanitizeHTML(jsProtocol);
  console.log("❌ FAIL: Should have rejected javascript: protocol");
} catch (error) {
  if (error instanceof HTMLSanitizationError) {
    console.log(`✅ PASS: Correctly rejected - ${error.message}`);
  } else {
    console.log(`❌ FAIL: Wrong error type`);
  }
}

// Test 13: HTML Sanitization - Dangerous tags removed
console.log("\n[Test 13] HTML Sanitization - Filter dangerous tags");
try {
  const mixedHTML = '<h1>Title</h1><iframe src="evil"></iframe><p>Safe content</p>';
  const sanitized = sanitizeHTML(mixedHTML);

  if (!sanitized.includes("iframe") && sanitized.includes("Title")) {
    console.log("✅ PASS: Dangerous tags removed, safe content preserved");
    console.log(`  Output: ${sanitized}`);
  } else {
    console.log("❌ FAIL: Sanitization not working correctly");
  }
} catch (error) {
  console.log(`❌ FAIL: ${error instanceof Error ? error.message : String(error)}`);
}

console.log("\n" + "=".repeat(60));
console.log("Test Suite Complete");
console.log("=".repeat(60));
