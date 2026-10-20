"""Private env updates shared by every credential action.

Locks a stable sidecar of the resolved target, not the replaceable inode.
Generation markers are random operation identities, never secret fingerprints.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

_locks = {}
_guard = threading.Lock()
_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_assignment = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_marker = "# guyan-credential-generation "


@contextmanager
def file_lock(path):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _guard:
        lock = _locks.setdefault(str(target), threading.RLock())
    with lock:
        fd = os.open(str(target) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield target
        finally:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def validate_values(updates):
    for name, value in updates.items():
        if not _name.fullmatch(name):
            raise ValueError("invalid credential variable name")
        if value is not None and (not isinstance(value, str) or
                any(ord(ch) < 32 or ord(ch) == 127 for ch in value)):
            raise ValueError("credential must be a single line without control characters")


@contextmanager
def inode_lock(path, enabled=False):
    if not enabled or os.name != "posix":
        yield
        return
    import fcntl
    fd = os.open(Path(path).resolve(), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def read_values(path):
    """Last assignment wins, including empty quoted values, like dotenv."""
    values, generations = {}, {}
    target = Path(path).resolve()
    if not target.exists():
        return values, generations
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.startswith(_marker):
            parts = line[len(_marker):].split()
            if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{32}", parts[1]):
                generations[parts[0]] = parts[1]
        match = _assignment.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip()
        if value.startswith('"') and value.endswith('"'):
            try:
                value = json.loads(value)
            except ValueError:
                value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[name] = value
    return values, generations


def write_values(path, updates, *, preserve_inode=False):
    validate_values(updates)  # before touching disk or the process environment
    generations = {name: uuid.uuid4().hex for name in updates}
    with file_lock(path) as target, inode_lock(path, preserve_inode):
        lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
        output, seen = [], set()
        for line in lines:
            match = _assignment.match(line)
            name = match.group(1) if match else None
            if line.startswith(_marker) and line[len(_marker):].split(" ", 1)[0] in updates:
                continue
            if name in updates:
                if name in seen:
                    continue
                seen.add(name)
                if updates[name] is not None:
                    output.append(f"{name}={quote_value(updates[name])}")
            else:
                output.append(line)
        for name, value in updates.items():
            if name not in seen and value is not None:
                output.append(f"{name}={quote_value(value)}")
            output.append(f"{_marker}{name} {generations[name]}")
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        recovery = None
        keep_recovery = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                fd = None
                handle.write("\n".join(output) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if preserve_inode and target.exists():
                # A file bind mount pins this inode. A rename on either side
                # cannot update the other side's mount. Stage and fsync first,
                # then update the shared inode under the stable writer lock.
                # A failure here is explicitly unconfirmed, never "saved".
                backup_fd, recovery = tempfile.mkstemp(prefix=target.name + ".", suffix=".recovery", dir=target.parent)
                with os.fdopen(backup_fd, "wb") as backup:
                    backup.write(target.read_bytes())
                    backup.flush()
                    os.fsync(backup.fileno())
                keep_recovery = True
                try:
                    with open(temporary, "rb") as source, open(target, "r+b") as destination:
                        if os.name == "posix":
                            os.fchmod(destination.fileno(), 0o600)
                        destination.write(source.read())
                        destination.truncate()
                        destination.flush()
                        os.fsync(destination.fileno())
                except Exception:
                    # Keep both private old/new files for explicit recovery.
                    # Do not replay or restore over another writer's work.
                    raise OSError("credential write unconfirmed; private recovery files retained") from None
                keep_recovery = False
            else:
                os.replace(temporary, target)
        finally:
            if fd is not None:
                os.close(fd)
            if not keep_recovery:
                Path(temporary).unlink(missing_ok=True)
                if recovery:
                    Path(recovery).unlink(missing_ok=True)
        # Ordered with the file commit: another writer cannot set the process
        # environment and then be overwritten by this older operation.
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return generations


def quote_value(value):
    if re.fullmatch(r"[A-Za-z0-9_./:+@=-]*", value):
        return value
    return json.dumps(value, ensure_ascii=False)
