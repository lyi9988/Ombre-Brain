"""Prepare a directory bind migration; the release controller alone recreates services.

Linux host, Docker Compose v2, root service users (the confirmed deployment).
Private snapshots/configs never go to stdout. There is no up/stop/restart command.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from credential_file import file_lock, inode_lock, read_values

SERVICES = ("ombre-gateway", "ombre-brain")
CONTAINER_DIR = "/run/ombre-credentials"
RUNTIME_DIR = "/opt/ombre-credential-runtime"


class Refused(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def docker(*args):
    result = subprocess.run(["docker", *args], capture_output=True)
    require(result.returncode == 0, "docker command failed; inspect original operation privately")
    return result.stdout


def private_write(path, data):
    path = Path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_state(directory, state):
    temporary = directory / ("state." + uuid.uuid4().hex + ".tmp")
    private_write(temporary, (json.dumps(state, indent=2) + "\n").encode())
    os.replace(temporary, directory / "state.json")
    sync_directory(directory)


def inspect(name):
    data = json.loads(docker("inspect", name))[0]
    require(data["Name"].lstrip("/") == name, "unexpected container name")
    return data


def snapshot(source, destination):
    # /proc/PID/root/... intentionally retains the original mount namespace.
    # Do not resolve it into a host /app path before opening it.
    import fcntl
    with open(source, "rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "source changed during snapshot")
    private_write(destination, raw)
    return {"device": before.st_dev, "inode": before.st_ino,
            "mode": before.st_mode & 0o777, "uid": before.st_uid, "gid": before.st_gid}


def compose_model(info):
    labels = info["Config"]["Labels"]
    project = labels["com.docker.compose.project"]
    working = labels["com.docker.compose.project.working_dir"]
    files = labels["com.docker.compose.project.config_files"].split(",")
    require(all(Path(f).is_absolute() and Path(f).is_file() for f in files), "Compose chain unavailable")
    command = ["compose", "--project-directory", working, "-p", project]
    for file in files:
        command.extend(["-f", file])
    model = json.loads(docker(*command, "config", "--format", "json"))
    return project, working, files, model


def prepare(args):
    require(args.writers_quiesced, "coordinate configuration writers before preparation")
    entry = Path(args.host_env).absolute()
    target_dir = Path(args.credentials_dir).absolute()
    state_dir = Path(args.operation_dir).absolute()
    require(entry.is_file() and not target_dir.exists(), "source missing or destination already exists")
    require(target_dir.parent == entry.parent and target_dir != entry, "credential directory must be a dedicated sibling")
    require(not state_dir.exists(), "operation already exists; inspect its receipt instead of replaying")
    state_dir.mkdir(mode=0o700)
    infos = {name: inspect(name) for name in SERVICES}
    state = {"operation": uuid.uuid4().hex, "phase": "snapshots_only", "host_entry": str(entry),
             "host_resolved": str(entry.resolve()), "credentials_dir": str(target_dir), "services": {}}
    host = state_dir / "host-current.env"
    with file_lock(entry):
        state["host_identity"] = snapshot(entry, host)
    values = read_values(host)[0]
    require(values, "empty credential source")
    for name, info in infos.items():
        require(info["State"]["Running"], "both original containers must be readable before destruction")
        require(info["Config"].get("User", "") in ("", "0", "root", "0:0", "root:root"),
                "non-root service ownership requires an explicit migration adaptation")
        mounts = [v for v in info["Mounts"] if v["Destination"] == "/app/.env"]
        require(len(mounts) == 1 and mounts[0]["Type"] == "bind" and mounts[0]["RW"] and
                Path(mounts[0]["Source"]).resolve() == entry.resolve(), "unexpected original credential mount")
        saved = state_dir / (name + ".env")
        identity = snapshot(Path("/proc") / str(info["State"]["Pid"]) / "root/app/.env", saved)
        state["services"][name] = {"container": info["Id"], "image": info["Image"], "identity": identity}
        save_state(state_dir, state)
        # Preserve each orphan first. Differences require field-level owner resolution,
        # never selecting one complete env file just because it is newer.
        require(read_values(saved)[0] == values, "source values differ; private snapshots retained for reconciliation")
        project, working, files, model = compose_model(info)
        require(model["services"][name].get("entrypoint") in (None, [], ""), "custom entrypoint requires adaptation")
        private_write(state_dir / (name + ".original.json"), json.dumps(model).encode())
        state["services"][name].update(project=project, working_dir=working, compose_files=files)
    require(len({v["project"] for v in state["services"].values()}) == 1, "unexpected project split")
    target_dir.mkdir(mode=0o700)
    private_write(target_dir / ".env", host.read_bytes())
    os.chown(target_dir, state["host_identity"]["uid"], state["host_identity"]["gid"])
    os.chown(target_dir / ".env", state["host_identity"]["uid"], state["host_identity"]["gid"])
    runtime = state_dir / "runtime"
    runtime.mkdir(mode=0o700)
    for source in (Path(__file__).with_name("credential_bootstrap.py"),
                   Path(__file__).resolve().parents[1] / "credential_file.py"):
        private_write(runtime / source.name, source.read_bytes())
    sync_directory(target_dir)
    sync_directory(runtime)
    state["phase"] = "prepared"
    save_state(state_dir, state)
    return state


def render_model(original, name, state, operation_dir, image):
    model = deepcopy(original)
    service = model["services"][name]
    values = read_values(operation_dir / "host-current.env")[0]
    environment = service.setdefault("environment", {})
    # Values resolved into Compose environment from the old env_file must not
    # become an immutable startup snapshot. A deliberate override is a conflict.
    for key, value in values.items():
        if key in environment:
            require(str(environment[key]) == value, "startup override conflicts with persistent source")
            del environment[key]
    environment["OMBRE_ENV_PATH"] = CONTAINER_DIR + "/.env"
    service.pop("env_file", None)
    volumes = service.setdefault("volumes", [])
    require(all(v.get("target") not in (CONTAINER_DIR, RUNTIME_DIR) for v in volumes), "new mount target already in use")
    require(sum(v.get("target") == "/app/.env" for v in volumes) == 1, "original file bind not found exactly once")
    service["volumes"] = [v for v in volumes if v.get("target") != "/app/.env"] + [
        {"type": "bind", "source": state["credentials_dir"], "target": CONTAINER_DIR,
         "bind": {"create_host_path": False}},
        {"type": "bind", "source": str(operation_dir / "runtime"), "target": RUNTIME_DIR,
         "read_only": True, "bind": {"create_host_path": False}},
    ]
    service["entrypoint"] = ["python", RUNTIME_DIR + "/credential_bootstrap.py"]
    require(service.get("command"), "explicit original service command required")
    service.pop("build", None)
    service["image"] = image
    # This is a full resolved model used ALONE, never an additive volumes overlay.
    # Escape interpolation, including inline credentials outside the moved file.
    def escape(value):
        if isinstance(value, str):
            return value.replace("$", "$$")
        if isinstance(value, list):
            return [escape(v) for v in value]
        if isinstance(value, dict):
            return {k: escape(v) for k, v in value.items()}
        return value
    return escape(model)


def activate(state, directory):
    require(state["phase"] in ("prepared", "activating", "active"), "operation is not prepared")
    entry = Path(state["host_entry"])
    target = Path(state["credentials_dir"]) / ".env"
    if entry.is_symlink() and entry.resolve() == target:
        state["phase"] = "active"
        save_state(directory, state)
        return state
    for name, original in state["services"].items():
        info = inspect(name)
        require(info["Id"] == original["container"] and not info["State"]["Running"],
                "stop both original services through the release controller before activating")
    with file_lock(entry), inode_lock(entry, True):
        require(str(entry.resolve()) == state["host_resolved"], "host source path changed after preparation")
        require(entry.read_bytes() == (directory / "host-current.env").read_bytes() == target.read_bytes(),
                "source changed after preparation; preserve current values and prepare a fresh operation")
        state["phase"] = "activating"
        save_state(directory, state)
        link = entry.with_name(entry.name + "." + state["operation"] + ".link")
        if link.is_symlink():
            require(link.resolve() == target, "unexpected partial activation link")
        else:
            link.symlink_to(os.path.relpath(target, entry.parent))
        os.replace(link, entry)
        sync_directory(entry.parent)
    state["phase"] = "active"
    save_state(directory, state)
    return state


def verify(state):
    target = Path(state["credentials_dir"]) / ".env"
    require(Path(state["host_entry"]).is_symlink() and Path(state["host_entry"]).resolve() == target,
            "legacy host entry is not the canonical symlink")
    canonical = target.stat()
    require((canonical.st_mode & 0o777) == 0o600, "credential mode must be 0600")
    require((target.parent.stat().st_mode & 0o777) == 0o700, "credential directory mode must be 0700")
    for name in SERVICES:
        info = inspect(name)
        mounts = {v["Destination"]: v for v in info["Mounts"]}
        require(info["State"]["Running"] and "/app/.env" not in mounts, "service still uses original file bind")
        mounted = mounts.get(CONTAINER_DIR, {})
        require(mounted.get("Type") == "bind" and mounted.get("RW") and
                mounted.get("Source") == state["credentials_dir"], "service directory source differs")
        visible = Path("/proc") / str(info["State"]["Pid"]) / "root" / CONTAINER_DIR.lstrip("/") / ".env"
        actual = visible.stat()
        require((actual.st_dev, actual.st_ino) == (canonical.st_dev, canonical.st_ino), "service sees another inode")
        require(visible.read_bytes() == target.read_bytes(), "service file differs")
        env = dict(v.partition("=")[::2] for v in info["Config"]["Env"])
        require(env.get("OMBRE_ENV_PATH") == CONTAINER_DIR + "/.env", "startup path differs")
    return {"phase": "mounts_verified", "consumers": "verify application receipts separately",
            "provider_auth": "not_called"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "render", "activate", "verify", "rollback"))
    parser.add_argument("--operation-dir", required=True)
    parser.add_argument("--host-env", default="/srv/ombre-brain/.env")
    parser.add_argument("--credentials-dir", default="/srv/ombre-brain/credentials")
    parser.add_argument("--writers-quiesced", action="store_true")
    parser.add_argument("--service", choices=SERVICES)
    parser.add_argument("--image", help="immutable image ID/tag frozen by the release controller")
    args = parser.parse_args()
    require(sys.platform == "linux", "run on the Linux Docker host")
    directory = Path(args.operation_dir).absolute()
    if args.action == "prepare":
        state = prepare(args)
    else:
        with file_lock(directory / "operation"):
            state = json.loads((directory / "state.json").read_text())
            if args.action == "activate":
                state = activate(state, directory)
            elif args.action == "verify":
                return verify(state)
            else:
                require(args.service, "service required")
                image = state["services"][args.service]["image"] if args.action == "rollback" else args.image
                require(image, "explicit release image required")
                original = json.loads((directory / (args.service + ".original.json")).read_text())
                model = render_model(original, args.service, state, directory, image)
                name = args.service + (".rollback.json" if args.action == "rollback" else ".directory.json")
                private_write(directory / name, (json.dumps(model, indent=2) + "\n").encode())
                return {"phase": "rendered", "compose_file": str(directory / name),
                        "service": args.service, "project": state["services"][args.service]["project"],
                        "rollback_keeps_current_credentials": True}
    return {"operation": state["operation"], "phase": state["phase"]}


if __name__ == "__main__":
    try:
        print(json.dumps(main()))
    except Refused as error:
        print(json.dumps({"phase": "unconfirmed", "error": str(error)}))
        sys.exit(1)
    except (OSError, ValueError, KeyError):
        # No traceback or Docker stderr: either can contain resolved secrets.
        print(json.dumps({"phase": "unconfirmed", "error": "migration refused; inspect private state and current topology; do not replay cutover"}))
        sys.exit(1)
