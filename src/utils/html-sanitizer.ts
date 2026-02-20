/**
 * HTML Sanitizer for Report Content
 * Uses Bun's HTMLRewriter to avoid regex-based HTML parsing.
 */

/**
 * List of allowed HTML tags for report content
 */
const ALLOWED_TAGS = new Set([
  // Text formatting
  "p",
  "br",
  "span",
  "div",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "strong",
  "em",
  "u",
  "s",
  "pre",
  "code",
  "blockquote",
  // Lists
  "ul",
  "ol",
  "li",
  // Tables
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  // Other
  "hr",
]);

const BLOCKED_TAGS = new Set([
  "script",
  "style",
  "iframe",
  "object",
  "embed",
  "link",
  "meta",
  "base",
  "form",
  "input",
  "button",
  "textarea",
  "select",
  "option",
]);

/**
 * Allowed attributes for HTML tags
 */
const ALLOWED_ATTRIBUTES: Record<string, Set<string>> = {
  "*": new Set(["class", "style"]),
  table: new Set(["border", "cellpadding", "cellspacing"]),
  td: new Set(["colspan", "rowspan"]),
  th: new Set(["colspan", "rowspan"]),
};

/**
 * Dangerous patterns in style attributes
 */
const DANGEROUS_STYLE_PATTERNS = [
  /url\s*\(/i,
  /@import/i,
  /javascript:/i,
  /expression\s*\(/i,
  /behavior\s*:/i,
];

/**
 * Dangerous URL protocols
 */
const DANGEROUS_PROTOCOLS = ["javascript:", "data:", "vbscript:", "file:", "about:"];

export class HTMLSanitizationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HTMLSanitizationError";
  }
}

function isSafeStyleValue(value: string): boolean {
  return !DANGEROUS_STYLE_PATTERNS.some((pattern) => pattern.test(value));
}

function isSafeUrl(url: string): boolean {
  const lowerUrl = url.toLowerCase().trim();
  return !DANGEROUS_PROTOCOLS.some((protocol) => lowerUrl.startsWith(protocol));
}

function sanitizeElementAttributes(element: HTMLRewriterTypes.Element): void {
  const tag = element.tagName.toLowerCase();
  const allowedForTag = ALLOWED_ATTRIBUTES[tag] || new Set();
  const allowedCommon = ALLOWED_ATTRIBUTES["*"] || new Set();

  // Snapshot attributes first: Bun's attribute iterator is live.
  const attributes = Array.from(element.attributes);
  for (const [rawName, rawValue] of attributes) {
    const name = rawName.toLowerCase();

    if (name.startsWith("on")) {
      element.removeAttribute(rawName);
      continue;
    }

    if (!allowedForTag.has(name) && !allowedCommon.has(name)) {
      element.removeAttribute(rawName);
      continue;
    }

    if (name === "style" && !isSafeStyleValue(rawValue)) {
      element.removeAttribute(rawName);
      continue;
    }

    if ((name === "href" || name === "src") && !isSafeUrl(rawValue)) {
      element.removeAttribute(rawName);
    }
  }
}

/**
 * Sanitize HTML content to prevent XSS and SSRF attacks
 *
 * @param html - Raw HTML content from user
 * @returns Sanitized HTML safe for rendering
 */
export function sanitizeHTML(html: string): string {
  if (!html || typeof html !== "string") {
    throw new HTMLSanitizationError("Invalid HTML input");
  }

  try {
    const rewriter = new HTMLRewriter().on("*", {
      element(element) {
        const tag = element.tagName.toLowerCase();

        if (BLOCKED_TAGS.has(tag)) {
          element.remove();
          return;
        }

        if (!ALLOWED_TAGS.has(tag)) {
          element.removeAndKeepContent();
          return;
        }

        sanitizeElementAttributes(element);
      },
    });

    return rewriter.transform(html);
  } catch (error) {
    if (error instanceof HTMLSanitizationError) {
      throw error;
    }
    throw new HTMLSanitizationError("Failed to sanitize HTML content");
  }
}

/**
 * Validate and sanitize Base64 encoded HTML
 */
export function sanitizeBase64HTML(base64Content: string): string {
  try {
    const decodedHTML = Buffer.from(base64Content, "base64").toString("utf-8");
    return sanitizeHTML(decodedHTML);
  } catch (error) {
    if (error instanceof HTMLSanitizationError) {
      throw error;
    }
    throw new HTMLSanitizationError("Failed to decode or sanitize HTML");
  }
}
