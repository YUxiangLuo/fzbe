const explicitConnectionQueryPools = new WeakSet<object>();

/**
 * Marks a pool whose queries may borrow and destroy an explicit connection on
 * timeout. Structural Pool-like mocks stay unmarked and retain direct-query
 * behavior for focused unit tests.
 */
export const registerExplicitConnectionQueryPool = <T extends object>(pool: T): T => {
  explicitConnectionQueryPools.add(pool);
  return pool;
};

export const isExplicitConnectionQueryPool = (pool: object): boolean =>
  explicitConnectionQueryPools.has(pool);
