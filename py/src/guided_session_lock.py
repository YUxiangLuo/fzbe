"""Process-level fencing for guided-session workspace mutations."""

from contextlib import contextmanager
import os
from typing import Iterator


@contextmanager
def guided_session_lock(artifact_dir: str) -> Iterator[None]:
    """Prevent an orphaned and a replacement worker from mutating one session together."""

    os.makedirs(artifact_dir, exist_ok=True)
    lock_path = os.path.join(artifact_dir, ".step.lock")
    handle = open(lock_path, "a+b")

    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "[CONFLICT] Another guided worker still owns this session workspace."
                ) from error
        yield
    finally:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()
