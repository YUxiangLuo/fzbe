export class SemaphoreTimeoutError extends Error {
  constructor(public readonly timeoutMs: number) {
    super(`Semaphore acquire timed out after ${timeoutMs}ms`);
  }
}

export class Semaphore {
  private running = 0;
  private readonly queue: Array<{
    resolve: () => void;
    reject: (err: Error) => void;
    timerId?: ReturnType<typeof setTimeout>;
  }> = [];

  constructor(private readonly max: number) {
    if (!Number.isInteger(max) || max <= 0) {
      throw new Error(`Semaphore max must be a positive integer, got: ${max}`);
    }
  }

  get pending(): number {
    return this.queue.length;
  }

  get active(): number {
    return this.running;
  }

  async acquire(timeoutMs?: number): Promise<void> {
    if (this.running < this.max) {
      this.running++;
      return;
    }
    return new Promise<void>((resolve, reject) => {
      const entry: (typeof this.queue)[number] = { resolve, reject };
      this.queue.push(entry);

      if (timeoutMs !== undefined) {
        entry.timerId = setTimeout(() => {
          const index = this.queue.indexOf(entry);
          if (index !== -1) {
            this.queue.splice(index, 1);
            reject(new SemaphoreTimeoutError(timeoutMs));
          }
        }, timeoutMs);
      }
    });
  }

  release(): void {
    if (this.running <= 0) {
      throw new Error("Semaphore released without matching acquire");
    }
    this.running--;
    const next = this.queue.shift();
    if (next) {
      this.running++;
      if (next.timerId !== undefined) clearTimeout(next.timerId);
      next.resolve();
    }
  }
}
