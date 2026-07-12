import { describe, expect, test } from "bun:test";
import { Semaphore, SemaphoreTimeoutError } from "../../../src/utils/semaphore";

describe("Semaphore", () => {
  test("acquire and release within limit", async () => {
    const sem = new Semaphore(2);

    await sem.acquire();
    await sem.acquire();
    expect(sem.active).toBe(2);

    sem.release();
    expect(sem.active).toBe(1);

    sem.release();
    expect(sem.active).toBe(0);
  });

  test("queues when at capacity and resolves on release", async () => {
    const sem = new Semaphore(1);

    await sem.acquire();
    expect(sem.active).toBe(1);

    let acquired = false;
    const pending = sem.acquire().then(() => { acquired = true; });
    expect(sem.pending).toBe(1);
    expect(acquired).toBe(false);

    sem.release();
    await pending;
    expect(acquired).toBe(true);
    expect(sem.active).toBe(1);
    expect(sem.pending).toBe(0);

    sem.release();
  });

  test("timeout rejects with SemaphoreTimeoutError", async () => {
    const sem = new Semaphore(1);
    await sem.acquire();

    await expect(sem.acquire(10)).rejects.toBeInstanceOf(SemaphoreTimeoutError);

    expect(sem.pending).toBe(0);
    expect(sem.active).toBe(1);

    sem.release();
  });

  test("timeout does not fire after release resolves the waiter", async () => {
    const sem = new Semaphore(1);
    await sem.acquire();

    const pending = sem.acquire(500);
    sem.release();
    await pending;

    // Wait past the original timeout — should not throw
    await new Promise((r) => setTimeout(r, 600));
    expect(sem.active).toBe(1);

    sem.release();
  });

  test("FIFO ordering of queued waiters", async () => {
    const sem = new Semaphore(1);
    await sem.acquire();

    const order: number[] = [];
    const p1 = sem.acquire().then(() => order.push(1));
    const p2 = sem.acquire().then(() => order.push(2));
    const p3 = sem.acquire().then(() => order.push(3));

    expect(sem.pending).toBe(3);

    sem.release(); await p1;
    sem.release(); await p2;
    sem.release(); await p3;

    expect(order).toEqual([1, 2, 3]);

    sem.release();
  });

  test("timed-out entry is removed from queue without affecting others", async () => {
    const sem = new Semaphore(1);
    await sem.acquire();

    const order: string[] = [];
    const p1 = sem.acquire(10).catch(() => order.push("timeout"));
    const p2 = sem.acquire().then(() => order.push("second"));

    // Wait for p1 to timeout
    await p1.catch(() => {});
    expect(sem.pending).toBe(1);

    sem.release();
    await p2;
    expect(order).toEqual(["timeout", "second"]);

    sem.release();
  });

  test("release without acquire throws", () => {
    const sem = new Semaphore(1);
    expect(() => sem.release()).toThrow("Semaphore released without matching acquire");
  });

  test("double release past zero throws", async () => {
    const sem = new Semaphore(1);
    await sem.acquire();
    sem.release();
    expect(() => sem.release()).toThrow("Semaphore released without matching acquire");
  });

  test("acquire without timeout waits indefinitely (no rejection)", async () => {
    const sem = new Semaphore(1);
    await sem.acquire();

    let resolved = false;
    const pending = sem.acquire().then(() => { resolved = true; });

    await new Promise((r) => setTimeout(r, 50));
    expect(resolved).toBe(false);

    sem.release();
    await pending;
    expect(resolved).toBe(true);

    sem.release();
  });

  test("SemaphoreTimeoutError message formats correctly", async () => {
    const sem = new Semaphore(1);
    await sem.acquire();

    try {
      await sem.acquire(10);
    } catch (e) {
      expect(e).toBeInstanceOf(SemaphoreTimeoutError);
      expect((e as Error).message).toContain("10ms");
      sem.release();
      return;
    }
    throw new Error("expected SemaphoreTimeoutError");
  });

  test("constructor rejects zero", () => {
    expect(() => new Semaphore(0)).toThrow("positive integer");
  });

  test("constructor rejects negative", () => {
    expect(() => new Semaphore(-1)).toThrow("positive integer");
  });

  test("constructor rejects non-integer", () => {
    expect(() => new Semaphore(1.5)).toThrow("positive integer");
  });

  test("capacity reports the configured max", () => {
    expect(new Semaphore(3).capacity).toBe(3);
  });

  describe("tryAcquire (non-blocking)", () => {
    test("succeeds while below capacity, fails when full", () => {
      const sem = new Semaphore(2);

      expect(sem.tryAcquire()).toBe(true);
      expect(sem.tryAcquire()).toBe(true);
      expect(sem.active).toBe(2);

      // Full — must fail immediately without queueing.
      expect(sem.tryAcquire()).toBe(false);
      expect(sem.active).toBe(2);
      expect(sem.pending).toBe(0);
    });

    test("a release frees exactly one tryAcquire slot", () => {
      const sem = new Semaphore(1);

      expect(sem.tryAcquire()).toBe(true);
      expect(sem.tryAcquire()).toBe(false);

      sem.release();
      expect(sem.tryAcquire()).toBe(true);
      expect(sem.tryAcquire()).toBe(false);
    });

    test("tryAcquire does not steal a slot from a queued acquire waiter", async () => {
      const sem = new Semaphore(1);
      expect(sem.tryAcquire()).toBe(true);

      let waiterResolved = false;
      const waiter = sem.acquire().then(() => { waiterResolved = true; });

      // Releasing hands the slot to the queued waiter, so a subsequent tryAcquire fails.
      sem.release();
      expect(sem.tryAcquire()).toBe(false);

      await waiter;
      expect(waiterResolved).toBe(true);

      sem.release();
    });
  });
});
