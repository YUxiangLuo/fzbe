/**
 * HTML Sanitizer for Report Content
 * Prevents XSS and SSRF attacks by sanitizing user-provided HTML
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

/**
 * Allowed attributes for HTML tags
 */
const ALLOWED_ATTRIBUTES: Record<string, Set<string>> = {
  "*": new Set(["class", "style"]), // Common safe attributes
  table: new Set(["border", "cellpadding", "cellspacing"]),
  td: new Set(["colspan", "rowspan"]),
  th: new Set(["colspan", "rowspan"]),
};

/**
 * Dangerous patterns in style attributes
 */
const DANGEROUS_STYLE_PATTERNS = [
  /url\s*\(/i, // CSS url() can trigger SSRF
  /@import/i, // CSS imports
  /javascript:/i, // javascript: protocol
  /expression\s*\(/i, // IE expression()
  /behavior\s*:/i, // IE behavior
];

/**
 * Dangerous URL protocols
 */
const DANGEROUS_PROTOCOLS = [
  "javascript:",
  "data:",
  "vbscript:",
  "file:",
  "about:",
];

export class HTMLSanitizationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HTMLSanitizationError";
  }
}

/**
 * Check if a style value is safe
 */
function isSafeStyleValue(value: string): boolean {
  return !DANGEROUS_STYLE_PATTERNS.some((pattern) => pattern.test(value));
}

/**
 * Check if a URL is safe
 */
function isSafeUrl(url: string): boolean {
  const lowerUrl = url.toLowerCase().trim();
  return !DANGEROUS_PROTOCOLS.some((protocol) =>
    lowerUrl.startsWith(protocol)
  );
}

/**
 * Sanitize HTML attributes
 */
function sanitizeAttributes(
  tag: string,
  attributes: Record<string, string>
): Record<string, string> {
  const allowedForTag = ALLOWED_ATTRIBUTES[tag] || new Set();
  const allowedCommon = ALLOWED_ATTRIBUTES["*"] || new Set();
  const sanitized: Record<string, string> = {};

  for (const [key, value] of Object.entries(attributes)) {
    const lowerKey = key.toLowerCase();

    // Check if attribute is allowed
    if (!allowedForTag.has(lowerKey) && !allowedCommon.has(lowerKey)) {
      continue;
    }

    // Special handling for style attribute
    if (lowerKey === "style") {
      if (isSafeStyleValue(value)) {
        sanitized[key] = value;
      }
      continue;
    }

    // Special handling for URL attributes (even though not in allowed list currently)
    if (["href", "src"].includes(lowerKey)) {
      if (isSafeUrl(value)) {
        sanitized[key] = value;
      }
      continue;
    }

    // Add other allowed attributes
    sanitized[key] = value;
  }

  return sanitized;
}

/**
 * Simple HTML tag parser
 * Note: This is a basic implementation. For production use, consider a proper HTML parser library.
 */
interface ParsedTag {
  type: "open" | "close" | "self-closing" | "text";
  tag?: string;
  attributes?: Record<string, string>;
  text?: string;
}

function parseHTML(html: string): ParsedTag[] {
  const tokens: ParsedTag[] = [];
  const tagRegex = /<\/?([a-zA-Z][a-zA-Z0-9]*)\s*([^>]*)>/g;
  let lastIndex = 0;
  let match;

  while ((match = tagRegex.exec(html)) !== null) {
    // Add text before tag
    if (match.index > lastIndex) {
      const text = html.substring(lastIndex, match.index);
      if (text.trim()) {
        tokens.push({ type: "text", text });
      }
    }

    const fullTag = match[0];
    const tagName = match[1].toLowerCase();
    const attributesStr = match[2];

    // Parse tag type
    if (fullTag.startsWith("</")) {
      tokens.push({ type: "close", tag: tagName });
    } else {
      const isSelfClosing = fullTag.endsWith("/>");
      const attributes = parseAttributes(attributesStr);

      tokens.push({
        type: isSelfClosing ? "self-closing" : "open",
        tag: tagName,
        attributes,
      });
    }

    lastIndex = tagRegex.lastIndex;
  }

  // Add remaining text
  if (lastIndex < html.length) {
    const text = html.substring(lastIndex);
    if (text.trim()) {
      tokens.push({ type: "text", text });
    }
  }

  return tokens;
}

/**
 * Parse HTML attributes from a string
 */
function parseAttributes(attrStr: string): Record<string, string> {
  const attributes: Record<string, string> = {};
  const attrRegex = /([a-zA-Z][a-zA-Z0-9-]*)\s*=\s*["']([^"']*)["']/g;
  let match;

  while ((match = attrRegex.exec(attrStr)) !== null) {
    attributes[match[1]] = match[2];
  }

  return attributes;
}

/**
 * Reconstruct HTML from parsed tokens
 */
function reconstructHTML(tokens: ParsedTag[]): string {
  return tokens
    .map((token) => {
      if (token.type === "text") {
        return escapeHTML(token.text || "");
      }

      if (token.type === "close") {
        return `</${token.tag}>`;
      }

      const attrStr = Object.entries(token.attributes || {})
        .map(([key, value]) => `${key}="${escapeAttributeValue(value)}"`)
        .join(" ");

      if (token.type === "self-closing") {
        return attrStr ? `<${token.tag} ${attrStr} />` : `<${token.tag} />`;
      }

      return attrStr ? `<${token.tag} ${attrStr}>` : `<${token.tag}>`;
    })
    .join("");
}

/**
 * Escape HTML special characters in text content
 */
function escapeHTML(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

/**
 * Escape attribute values
 */
function escapeAttributeValue(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
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

  // Detect potential script injection attempts
  if (
    /<script/i.test(html) ||
    /javascript:/i.test(html) ||
    /onerror\s*=/i.test(html) ||
    /onload\s*=/i.test(html)
  ) {
    throw new HTMLSanitizationError(
      "HTML contains potentially malicious content"
    );
  }

  // Parse HTML
  const tokens = parseHTML(html);

  // Filter and sanitize tokens
  const sanitizedTokens: ParsedTag[] = [];
  const tagStack: string[] = [];

  for (const token of tokens) {
    if (token.type === "text") {
      sanitizedTokens.push(token);
      continue;
    }

    if (token.type === "open") {
      const tag = token.tag!;
      if (ALLOWED_TAGS.has(tag)) {
        const sanitizedAttrs = sanitizeAttributes(tag, token.attributes || {});
        sanitizedTokens.push({
          type: "open",
          tag,
          attributes: sanitizedAttrs,
        });
        tagStack.push(tag);
      }
      continue;
    }

    if (token.type === "close") {
      const tag = token.tag!;
      if (ALLOWED_TAGS.has(tag) && tagStack.includes(tag)) {
        sanitizedTokens.push({ type: "close", tag });
        // Remove from stack
        const index = tagStack.lastIndexOf(tag);
        if (index !== -1) {
          tagStack.splice(index, 1);
        }
      }
      continue;
    }

    if (token.type === "self-closing") {
      const tag = token.tag!;
      if (ALLOWED_TAGS.has(tag)) {
        const sanitizedAttrs = sanitizeAttributes(tag, token.attributes || {});
        sanitizedTokens.push({
          type: "self-closing",
          tag,
          attributes: sanitizedAttrs,
        });
      }
    }
  }

  // Close any unclosed tags
  while (tagStack.length > 0) {
    const tag = tagStack.pop()!;
    sanitizedTokens.push({ type: "close", tag });
  }

  return reconstructHTML(sanitizedTokens);
}

/**
 * Validate and sanitize Base64 encoded HTML
 */
export function sanitizeBase64HTML(base64Content: string): string {
  try {
    // Decode Base64
    const decodedHTML = Buffer.from(base64Content, "base64").toString("utf-8");

    // Sanitize HTML
    return sanitizeHTML(decodedHTML);
  } catch (error) {
    if (error instanceof HTMLSanitizationError) {
      throw error;
    }
    throw new HTMLSanitizationError("Failed to decode or sanitize HTML");
  }
}
