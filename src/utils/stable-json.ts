export const stableJsonStringify = (value: unknown): string => {
  if (value instanceof Date) {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => (item === undefined ? "null" : stableJsonStringify(item))).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .filter((key) => record[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJsonStringify(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "undefined";
};
