"""Load the current directory-mounted env before either service starts.

The runtime directory contains this file and the existing credential_file.py.
It is mounted read-only, including when rolling back to a compatible old image.
"""
import os
from pathlib import Path
import sys

from credential_file import file_lock, inode_lock, read_values


def current_environment(path):
    # The stable sidecar and inode locks also cover the legacy in-place writer.
    # Reading must fail closed: starting with stale Docker creation env is wrong.
    target = Path(path)
    if not target.is_file():
        raise OSError("credential source unavailable")
    with file_lock(target), inode_lock(target, True):
        values, _ = read_values(target)
    values["OMBRE_ENV_PATH"] = str(target)
    return values


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise ValueError("service command required")
    path = os.environ.get("OMBRE_ENV_PATH", "/run/ombre-credentials/.env")
    os.environ.update(current_environment(path))
    os.execvp(args[0], args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError):
        # Neither exceptions nor subprocess arguments may reveal credentials.
        print("credential bootstrap failed; service was not started", file=sys.stderr)
        sys.exit(1)
