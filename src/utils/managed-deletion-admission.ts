import { Semaphore, SemaphoreTimeoutError } from "./semaphore";

const MANAGED_DELETION_CAPACITY = 1;
export const MANAGED_DELETION_QUEUE_TIMEOUT_MS = 10_000;

// Dataset, manual, and user deletion deliberately share one process-wide
// permit. The single-instance deployment contract makes this permit the global
// filesystem-deletion mutex; database row locks still protect each transaction.
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
