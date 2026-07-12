import { describe, expect, test } from "bun:test";
import { KeyedLock } from "../../../src/utils/keyed-lock";

describe("KeyedLock", () => {
  test("acquires a free key and reports it held", () => {
    const lock = new KeyedLock();

    expect(lock.tryAcquire("exp1:lstm")).toBe(true);
    expect(lock.isHeld("exp1:lstm")).toBe(true);
    expect(lock.size).toBe(1);
  });

  test("re-acquiring the same held key fails (409 semantics)", () => {
    const lock = new KeyedLock();

    expect(lock.tryAcquire("exp1:lstm")).toBe(true);
    expect(lock.tryAcquire("exp1:lstm")).toBe(false);
  });

  test("different keys are independent", () => {
    const lock = new KeyedLock();

    expect(lock.tryAcquire("exp1:lstm")).toBe(true);
    expect(lock.tryAcquire("exp1:arima")).toBe(true);
    expect(lock.tryAcquire("exp2:lstm")).toBe(true);
    expect(lock.size).toBe(3);
  });

  test("release lets the key be acquired again", () => {
    const lock = new KeyedLock();

    expect(lock.tryAcquire("exp1:lstm")).toBe(true);
    lock.release("exp1:lstm");
    expect(lock.isHeld("exp1:lstm")).toBe(false);
    expect(lock.tryAcquire("exp1:lstm")).toBe(true);
  });

  test("releasing an unheld key is a safe no-op", () => {
    const lock = new KeyedLock();

    expect(() => lock.release("never-held")).not.toThrow();
    lock.tryAcquire("exp1:lstm");
    lock.release("exp1:lstm");
    expect(() => lock.release("exp1:lstm")).not.toThrow();
    expect(lock.size).toBe(0);
  });
});
