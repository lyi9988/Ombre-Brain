import asyncio

from import_memory import ImportEngine
from memory_authority import MemoryAuthorityStore
from memory_commit_service import BucketProjectionResult, MemoryCommitService, body_sha256


class FakeProjection:
    def __init__(self):
        self.revisions = []

    async def write_revision(self, **kwargs):
        self.revisions.append(dict(kwargs))
        return BucketProjectionResult(
            bucket_id=kwargs["bucket_id"], revision=kwargs["revision"],
            operation_id=kwargs["operation_id"], body_sha256=body_sha256(kwargs["body"]),
            snapshot_path=kwargs["snapshot_path"],
        )

    async def append_ring(self, **_kwargs):
        raise AssertionError("not used")

    async def retract_ring(self, **_kwargs):
        raise AssertionError("not used")

    async def delete_memory(self, **_kwargs):
        raise AssertionError("not used")


class FakeBucketManager:
    def __init__(self):
        self.created = []
        self.updated = []
        self.buckets = []

    async def list_all(self, **_kwargs):
        return list(self.buckets)

    async def search(self, *_args, **_kwargs):
        return []

    async def create(self, content, **kwargs):
        bucket_id = f"legacy-{len(self.created) + 1}"
        self.created.append((bucket_id, content, kwargs))
        return bucket_id

    async def update(self, bucket_id, **kwargs):
        self.updated.append((bucket_id, kwargs))
        return True


class FakeDehydrator:
    prompt_plan_mirror = None

    async def merge(self, existing, incoming):
        return f"{existing}\n{incoming}"


def config(tmp_path, *, enabled):
    return {
        "buckets_dir": str(tmp_path / "buckets"),
        "state_dir": str(tmp_path / "state"),
        "gateway": {"prompt_plan_mirror_path": str(tmp_path / "prompt.sqlite3")},
        "memory_authority": {
            "enabled": enabled,
            "db_path": str(tmp_path / "memory-authority.sqlite3"),
        },
        "import": {"auto_merge_enabled": False},
    }


def item(content="导入的长期事实"):
    return {
        "content": content,
        "name": "导入事实",
        "tags": ["imported"],
        "importance": 6,
        "domain": ["历史"],
        "valence": 0.5,
        "arousal": 0.3,
        "source_chunk_ids": ["sourcehash:00001"],
        "source_refs": [{"chunk_id": "sourcehash:00001", "source_hash": "sourcehash"}],
        "import_source_hash": "sourcehash",
    }


def install_authority(engine):
    projection = FakeProjection()
    engine.memory_commit_service = MemoryCommitService(engine.memory_authority_store, projection)
    return projection


def force_no_duplicate(engine):
    async def no_duplicate(_content):
        return None
    engine._find_duplicate_bucket = no_duplicate


def test_import_create_retry_is_one_memory(tmp_path):
    manager = FakeBucketManager()
    engine = ImportEngine(config(tmp_path, enabled=True), manager, FakeDehydrator())
    projection = install_authority(engine)
    force_no_duplicate(engine)
    first = asyncio.run(engine._merge_or_create_item(item()))
    second = asyncio.run(engine._merge_or_create_item(item()))
    assert first == second == "created"
    assert len(projection.revisions) == 1
    memory_id = projection.revisions[0]["memory_id"]
    assert engine.memory_authority_store.get_memory(memory_id)["active_revision"] == 1
    assert len(engine.memory_authority_store.list_candidates(status="committed")) == 1


def test_import_merge_creates_target_revision_with_stable_source_refs(tmp_path):
    manager = FakeBucketManager()
    cfg = config(tmp_path, enabled=True)
    cfg["import"]["auto_merge_enabled"] = True
    engine = ImportEngine(cfg, manager, FakeDehydrator())
    projection = install_authority(engine)
    seed = asyncio.run(engine.memory_commit_service.commit_memory(
        memory_id="existing-1", bucket_id="existing-1", expected_revision=0,
        body="旧正文", metadata={"domain": ["历史"], "tags": []},
        source_refs=["seed:1"], decision_source="migration",
        idempotency_key="seed-existing", actor="migration",
    ))
    assert seed["revision"] == 1
    target = {
        "id": "existing-1",
        "content": "旧正文",
        "metadata": {"domain": ["历史"], "tags": [], "importance": 5,
                     "valence": 0.5, "arousal": 0.3, "name": "旧记忆"},
    }

    async def no_duplicate(_content):
        return None

    async def merge_target(_item):
        return target

    engine._find_duplicate_bucket = no_duplicate
    engine._find_import_merge_candidate = merge_target
    result = asyncio.run(engine._merge_or_create_item(item("新增补充")))
    memory = engine.memory_authority_store.get_memory("existing-1")
    revision = engine.memory_authority_store.get_memory_revision("existing-1", 2)
    assert result == "merged"
    assert memory["active_revision"] == 2
    assert set(revision["source_refs"]) == {"import_chunk:sourcehash:00001", "import_source:sourcehash"}
    assert len(projection.revisions) == 2


def test_import_authority_off_preserves_legacy_create(tmp_path):
    manager = FakeBucketManager()
    engine = ImportEngine(config(tmp_path, enabled=False), manager, FakeDehydrator())
    force_no_duplicate(engine)
    result = asyncio.run(engine._merge_or_create_item(item()))
    assert result == "created"
    assert len(manager.created) == 1
    assert manager.updated == []
