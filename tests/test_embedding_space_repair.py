"""SQL regressions for dimension/model-aware projection repair selection."""

import json
import sqlite3
from pathlib import Path

from memory_authority import MemoryAuthorityStore
from memory_projection_worker import embedding_projection_needs_upgrade
from scripts import repair_recall_r1_embeddings as repair


def test_sql_repair_selector_uses_expected_space_complete_status_and_cooldown(
    tmp_path, monkeypatch,
):
    now_ms = 2_000_000_000_000
    monkeypatch.setattr(repair.time, "time", lambda: now_ms / 1000)
    authority = MemoryAuthorityStore({"state_dir": str(tmp_path / "state")})
    path = Path(authority.path)
    expected_space = repair._expected_embedding_space({
        "embedding": {
            "enabled": True,
            "model": "current-model",
            "base_url": "https://embedding.invalid/v1",
            "dimensions": 4,
        }
    })
    assert expected_space == {
        "model": "current-model",
        "provider": "embedding.invalid",
        "dimension": 4,
    }

    same_space = {
        "metadata_complete": True,
        **expected_space,
    }
    stale_space = {
        "metadata_complete": True,
        "model": "old-model",
        "provider": expected_space["provider"],
        "dimension": expected_space["dimension"],
    }
    degraded_details = {**stale_space, "error": "synthetic failure"}

    assert embedding_projection_needs_upgrade("projected", stale_space, expected_space)
    assert not embedding_projection_needs_upgrade("projected", same_space, expected_space)
    assert not embedding_projection_needs_upgrade("degraded", degraded_details, expected_space)

    rows = (
        ("same-space", "projected", same_space, 0, "2026-10-01T13:00:00Z"),
        ("old-model-cooling", "projected", stale_space, now_ms + 60_000, "2026-10-01T12:00:00Z"),
        ("old-model-ready", "projected", stale_space, 0, "2026-10-01T11:00:00Z"),
        ("degraded-old-model", "degraded", degraded_details, 0, "2026-10-01T10:00:00Z"),
    )
    with sqlite3.connect(path) as conn:
        for memory_id, status, details, retry_after, updated_at in rows:
            encoded = json.dumps({**details, "retry_after_ms": retry_after})
            conn.execute(
                "INSERT INTO memories(memory_id,bucket_id,active_revision,state,recall_policy,updated_at) "
                "VALUES (?,?,1,'active','enabled',?)",
                (memory_id, memory_id, updated_at),
            )
            conn.execute(
                "INSERT INTO memory_projection_status"
                "(memory_id,memory_revision,projector,status,details_json,updated_at) "
                "VALUES (?,1,'embedding',?,?,?)",
                (memory_id, status, encoded, updated_at),
            )

    assert repair._upgrade_candidate_count(path, expected_space=expected_space) == 2
    candidate = repair._next_candidate(path, expected_space=expected_space)
    assert candidate["memory_id"] == "old-model-ready"

    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE memory_projection_status SET details_json=? WHERE memory_id=?",
            (json.dumps({**stale_space, "retry_after_ms": now_ms + 60_000}), "old-model-ready"),
        )
    assert repair._next_candidate(path, expected_space=expected_space) is None
    # Inventory/count intentionally includes cooling candidates; selection does not.
    assert repair._upgrade_candidate_count(path, expected_space=expected_space) == 2
