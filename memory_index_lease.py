"""Cross-process lease for derived-index maintenance; never a Memory authority.

The permanent lock inode is not unlinked, including after a failed or killed job.
The OS releases it on process exit. Readers do not take this write lease.
"""
from __future__ import annotations

import os
from pathlib import Path


class MemoryIndexLease:
    def __init__(self, state_dir, name="memory-index-writer.lock"):
        self.path = Path(state_dir) / name
        self.file = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            handle.close()
            return False
        self.file = handle
        return True

    def release(self):
        handle, self.file = self.file, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
