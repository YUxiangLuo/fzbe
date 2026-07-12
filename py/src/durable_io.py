"""Small durability helpers shared by model and guided-state writers."""

import os


def fsync_file(path: str) -> None:
    """Flush a fully written temporary file before publishing it with rename."""

    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def fsync_parent_directory(path: str) -> None:
    """Persist a rename in the parent directory on POSIX filesystems."""

    if os.name != "posix":
        return

    parent = os.path.dirname(os.path.abspath(path)) or "."
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
