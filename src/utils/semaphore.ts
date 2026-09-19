export class SemaphoreTimeoutError extends Error {
  constructor(public readonly timeoutMs: number) {
    super(`Semaphore acquire timed out after ${timeoutMs}ms`);
  }
}

export class SemaphoreQueueFullError extends Error {
  constructor(public readonly maxPending: number) {
    super(`Semaphore wait queue is full at ${maxPending} pending acquisitions`);
  }
}

export class Semaphore {
  private running = 0;
  private readonly queue: Array<{
    resolve: () => void;
    reject: (err: Error) => void;
    timerId?: ReturnType<typeof setTimeout>;
  }> = [];

  constructor(
    private readonly max: number,
    private readonly maxPending = Number.POSITIVE_INFINITY,
  ) {
    if (!Number.isInteger(max) || max <= 0) {
      throw new Error(`Semaphore max must be a positive integer, got: ${max}`);
    }
    if (
      maxPending !== Number.POSITIVE_INFINITY
      && (!Number.isInteger(maxPending) || maxPending <= 0)
    ) {
      throw new Error(`Semaphore maxPending must be a positive integer, got: ${maxPending}`);
    }
  }

  get capacity(): number {
    return this.max;
  }

  get pending(): number {
    return this.queue.length;
  }

  get active(): number {
    return this.running;
  }

  /**
   * 立即尝试获取额度，不进入等待队列。
   *
   * 模型请求使用该方法实现 fail-fast admission：超过容量的 HTTP 请求直接
   * 返回 429，而不是在 Bun 进程里堆积并继续争抢数据库连接。
   */
  tryAcquire(): boolean {
    if (this.running >= this.max) return false;
    this.running++;
    return true;
  }

  async acquire(timeoutMs?: number): Promise<void> {
    if (this.running < this.max) {
      this.running++;
      return;
    }
    if (this.queue.length >= this.maxPending) {
      throw new SemaphoreQueueFullError(this.maxPending);
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
