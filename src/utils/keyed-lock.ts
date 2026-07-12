/**
 * 进程内按键互斥锁（非阻塞）。
 *
 * 用于模型执行去重：同一 `(experimentId, modelType)` 同时只能有一个训练/生产/预测
 * 任务在执行。语义对应旧的 MySQL `GET_LOCK(execName, 0)`——立即成功或立即失败，
 * 不排队。失败时上层返回 409。
 *
 * 单实例部署下，进程内 Set 即可提供与 advisory lock 等价的去重保证，且不占用数据库连接。
 */
export class KeyedLock {
  private readonly held = new Set<string>();

  /** 尝试占用 key：未被持有则占用并返回 true，否则立即返回 false。 */
  tryAcquire(key: string): boolean {
    if (this.held.has(key)) {
      return false;
    }
    this.held.add(key);
    return true;
  }

  /** 释放 key。重复释放或释放未持有的 key 是安全的 no-op。 */
  release(key: string): void {
    this.held.delete(key);
  }

  /** 当前是否持有该 key（主要用于测试与观测）。 */
  isHeld(key: string): boolean {
    return this.held.has(key);
  }

  /** 当前持有的 key 数量。 */
  get size(): number {
    return this.held.size;
  }
}
