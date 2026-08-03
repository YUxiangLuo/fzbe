import { Semaphore, SemaphoreTimeoutError } from "./semaphore";

const MANAGED_DELETION_CAPACITY = 1;
export const MANAGED_DELETION_QUEUE_TIMEOUT_MS = 10_000;

// Dataset and user deletion deliberately share one process-wide permit. Each
// request owns a trusted advisory-lock session plus a transaction session, so
// serial admission prevents low-frequency admin operations from exhausting
// the pool while still leaving bulk deletion available as one transaction.
const managedDeletionAdmission = new Semaphore(MANAGED_DELETION_CAPACITY);

export interface ManagedDeletionPermit {
  release: () => void;
}

export async function acquireManagedDeletionPermit(
  timeoutMs = MANAGED_DELETION_QUEUE_TIMEOUT_MS,
): Promise<ManagedDeletionPermit | null> {
  try {
    await managedDeletionAdmission.acquire(timeoutMs);
  } catch (error) {
    if (error instanceof SemaphoreTimeoutError) return null;
    throw error;
  }

  let released = false;
  return {
    release: () => {
      if (released) return;
      released = true;
      managedDeletionAdmission.release();
    },
  };
}

export function tryAcquireManagedDeletionPermit(): ManagedDeletionPermit | null {
  if (!managedDeletionAdmission.tryAcquire()) return null;
  let released = false;
  return {
    release: () => {
      if (released) return;
      released = true;
      managedDeletionAdmission.release();
    },
  };
}

export function getManagedDeletionAdmissionStats() {
  return {
    capacity: managedDeletionAdmission.capacity,
    active: managedDeletionAdmission.active,
    pending: managedDeletionAdmission.pending,
  };
}
