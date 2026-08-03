function escapePdfText(value: string): string {
  return value
    .replace(/[^\x20-\x7E]/g, "?")
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)");
}

export type MinimalPdfInspection =
  | {
      ok: true;
      visibleText: string;
      objectCount: number;
    }
  | {
      ok: false;
      error: string;
    };

interface MinimalPdfInspectionOptions {
  expectedTextMarkers?: string[];
}

function unescapePdfText(value: string): string {
  return value.replace(/\\([\\()])/g, "$1");
}

/**
 * Validate the deliberately small PDF subset emitted by buildMinimalPdf.
 *
 * This is not a general-purpose PDF parser. It gives fixture verification a
 * dependency-free way to reject the failures that make a browser render a
 * blank page: missing page/font resources, empty text operations, incorrect
 * stream lengths, and stale xref/startxref offsets.
 */
export function inspectMinimalPdf(
  input: string | Uint8Array,
  options: MinimalPdfInspectionOptions = {},
): MinimalPdfInspection {
  const bytes = typeof input === "string"
    ? Buffer.from(input, "utf8")
    : Buffer.from(input);
  const pdf = bytes.toString("latin1");
  const fail = (error: string): MinimalPdfInspection => ({ ok: false, error });

  if (!pdf.startsWith("%PDF-1.4\n")) {
    return fail("missing PDF 1.4 header");
  }
  if (!/%%EOF\n?$/.test(pdf)) {
    return fail("missing PDF EOF marker");
  }
  if (!pdf.includes("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>")) {
    return fail("missing PDF catalog");
  }
  if (!pdf.includes("/Type /Page ") || !pdf.includes("/Contents 5 0 R")) {
    return fail("missing page content reference");
  }
  if (!pdf.includes("/Resources << /Font << /F1 4 0 R >> >>")) {
    return fail("missing page font resource");
  }
  if (!pdf.includes("/Subtype /Type1 /BaseFont /Helvetica")) {
    return fail("missing built-in Helvetica font");
  }

  const contentMatch = pdf.match(
    /5 0 obj\n<< \/Length (\d+) >>\nstream\n([\s\S]*?)\nendstream\nendobj/,
  );
  if (!contentMatch) {
    return fail("missing content stream");
  }
  const declaredContentLength = Number(contentMatch[1]);
  const content = contentMatch[2] ?? "";
  if (Buffer.byteLength(content, "latin1") !== declaredContentLength) {
    return fail("content stream length does not match its declaration");
  }

  const visibleLines = Array.from(
    content.matchAll(
      /BT\s+\/F1\s+\d+(?:\.\d+)?\s+Tf\s+-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s+Td\s+\(((?:\\.|[^\\)])*)\)\s+Tj\s+ET/g,
    ),
    (match) => unescapePdfText(match[1] ?? ""),
  );
  const visibleText = visibleLines.join("\n");
  if (!/[A-Za-z0-9]/.test(visibleText)) {
    return fail("content stream has no visible ASCII text");
  }

  const startXrefMatch = pdf.match(/startxref\n(\d+)\n%%EOF\n?$/);
  if (!startXrefMatch) {
    return fail("missing startxref offset");
  }
  const xrefOffset = Number(startXrefMatch[1]);
  if (!Number.isSafeInteger(xrefOffset) || !pdf.startsWith("xref\n", xrefOffset)) {
    return fail("startxref does not point to the xref table");
  }

  const xrefLines = pdf.slice(xrefOffset).split("\n");
  const subsectionMatch = xrefLines[1]?.match(/^0 (\d+)$/);
  if (!subsectionMatch) {
    return fail("invalid xref subsection header");
  }
  const objectCount = Number(subsectionMatch[1]);
  if (objectCount !== 6) {
    return fail(`unexpected xref object count: ${objectCount}`);
  }
  const entries = xrefLines.slice(2, 2 + objectCount);
  if (entries.length !== objectCount || entries[0] !== "0000000000 65535 f ") {
    return fail("invalid xref entry count or free entry");
  }
  for (let objectNumber = 1; objectNumber < objectCount; objectNumber += 1) {
    const entry = entries[objectNumber];
    const entryMatch = entry?.match(/^(\d{10}) 00000 n $/);
    if (!entryMatch) {
      return fail(`invalid xref entry for object ${objectNumber}`);
    }
    const objectOffset = Number(entryMatch[1]);
    if (!pdf.startsWith(`${objectNumber} 0 obj\n`, objectOffset)) {
      return fail(`xref entry for object ${objectNumber} has the wrong offset`);
    }
  }
  if (!pdf.slice(xrefOffset).includes("trailer\n<< /Size 6 /Root 1 0 R >>")) {
    return fail("missing or invalid PDF trailer");
  }

  for (const marker of options.expectedTextMarkers ?? []) {
    if (!visibleText.includes(marker)) {
      return fail(`visible text is missing marker: ${marker}`);
    }
  }

  return { ok: true, visibleText, objectCount };
}

export function buildMinimalPdf(lines: string[]): string {
  const content = lines
    .map((line, index) => `BT /F1 12 Tf 50 ${780 - index * 20} Td (${escapePdfText(line)}) Tj ET`)
    .join("\n");
  const contentLength = Buffer.byteLength(content, "utf8");

  const objects = [
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
    "2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj",
    "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
    `5 0 obj\n<< /Length ${contentLength} >>\nstream\n${content}\nendstream\nendobj`,
  ];

  let pdf = "%PDF-1.4\n";
  const offsets = [0];

  for (const object of objects) {
    offsets.push(Buffer.byteLength(pdf, "utf8"));
    pdf += `${object}\n`;
  }

  const xrefOffset = Buffer.byteLength(pdf, "utf8");
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += "0000000000 65535 f \n";
  for (let index = 1; index < offsets.length; index += 1) {
    pdf += `${String(offsets[index]).padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;

  return pdf;
}
