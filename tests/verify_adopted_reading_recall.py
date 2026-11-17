"""Offline ordinary Gateway recall check for an already-adopted reading card.

The manifest must point at the isolated Brain authority/bucket snapshot and
contain the complete successful response from Brain's recall-sources POST.
This runner never creates or commits a memory card. Gateway's mutable state,
embedding index, and bucket view are temporary; the authority database is
opened by MemoryAuthorityRecallView in read-only mode.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

from attachment_adoption_support import openai_response
from gateway import GatewayService, create_gateway_app
from memory_commit_service import body_sha256
from memory_delivery import payload_texts, sha
from memory_l1_v2 import TEXT_PROJECTION_VERSION, retrieval_text
from test_memory_local_first_settings import _base_config
import memory_recall_runtime as memory_use


class VerificationError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise VerificationError(code)


def read_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        raise VerificationError("manifest_unreadable") from None
    required = {
        "state_dir", "buckets_dir", "authority_db", "memory_id", "bucket_id",
        "revision", "body", "query", "source_verification",
    }
    require(isinstance(value, dict) and required <= set(value), "manifest_fields_missing")
    require(isinstance(value["memory_id"], str) and value["memory_id"], "memory_id_invalid")
    require(isinstance(value["bucket_id"], str) and value["bucket_id"], "bucket_id_invalid")
    require(type(value["revision"]) is int and value["revision"] > 0, "revision_invalid")
    require(isinstance(value["body"], str) and value["body"], "body_invalid")
    require(isinstance(value["query"], str) and value["query"], "query_invalid")
    for key in ("state_dir", "buckets_dir", "authority_db"):
        require(isinstance(value[key], str) and value[key], key + "_invalid")
    return value


def locate_bucket_file(buckets_dir: Path, bucket_id: str) -> Path:
    matches = [
        path for path in buckets_dir.rglob("*.md")
        if path.stem == bucket_id or path.stem.endswith("_" + bucket_id)
    ]
    require(len(matches) == 1, "adopted_bucket_file_not_unique")
    path = matches[0].resolve(strict=True)
    require(path.is_relative_to(buckets_dir.resolve(strict=True)), "bucket_path_outside_root")
    return path


def validate_source_verification(
    response: dict, memory_id: str, revision: int, binding_ids: set[str],
) -> dict:
    require(isinstance(response, dict), "source_verification_invalid")
    items = response.get("items")
    require(isinstance(items, list) and len(items) == 1, "source_verification_scope_invalid")
    item = items[0]
    require(isinstance(item, dict), "source_verification_item_invalid")
    require(item.get("memory_id") == memory_id, "source_verification_memory_mismatch")
    require(item.get("revision") == revision, "source_verification_revision_mismatch")
    require(item.get("state") == "ok", "source_verification_not_ok")
    require(item.get("historical") is False, "source_verification_not_current")
    require(item.get("l1_sync_required") is False, "source_verification_sync_required")
    require(item.get("sync_scope") is None, "source_verification_sync_scope")
    sources = item.get("sources")
    require(isinstance(sources, dict) and set(sources) == binding_ids, "source_verification_bindings_mismatch")
    require(all(isinstance(value, dict) and value.get("eligible_source") is True
                for value in sources.values()), "source_verification_source_ineligible")
    return item


def file_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


async def run_check(manifest: dict) -> dict:
    state_root = Path(manifest["state_dir"]).expanduser().resolve(strict=True)
    source_buckets = Path(manifest["buckets_dir"]).expanduser().resolve(strict=True)
    authority_db = Path(manifest["authority_db"]).expanduser().resolve(strict=True)
    require(state_root.is_dir(), "state_dir_not_directory")
    require(source_buckets.is_dir(), "buckets_dir_not_directory")
    require(authority_db.is_file(), "authority_db_not_file")

    memory_id = manifest["memory_id"]
    bucket_id = manifest["bucket_id"]
    revision = manifest["revision"]
    body = manifest["body"]
    query = manifest["query"]
    source_file = locate_bucket_file(source_buckets, bucket_id)
    authority_stamp = file_stamp(authority_db)
    bucket_stamp = file_stamp(source_file)

    run_id = uuid.uuid4().hex
    conversation_id = "reading-recall-" + run_id
    user_event_id = "reading-user-" + run_id
    assistant_event_id = "reading-assistant-" + run_id
    turn = {
        "conversation_id": conversation_id,
        "user_source_event_id": user_event_id,
        "assistant_source_event_id": assistant_event_id,
    }

    provider_requests: list[dict] = []
    source_requests: list[dict] = []
    embedding_inputs: list[str] = []
    rerank_calls: list[int] = []

    # Keep every Gateway-created database and index outside the source Brain
    # tree. Copy only the adopted card's Markdown so BucketManager reads the
    # exact existing body without exposing or mutating other buckets.
    with tempfile.TemporaryDirectory(prefix="gateway-reading-recall-", dir=str(state_root)) as temp:
        scratch_root = Path(temp)
        config = _base_config(scratch_root)
        scratch_buckets = Path(config["buckets_dir"])
        copied_bucket = scratch_buckets / source_file.relative_to(source_buckets)
        copied_bucket.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, copied_bucket)
        require(file_stamp(source_file)[0] == file_stamp(copied_bucket)[0], "bucket_copy_size_mismatch")
        require(hashlib.sha256(source_file.read_bytes()).digest()
                == hashlib.sha256(copied_bucket.read_bytes()).digest(), "bucket_copy_hash_mismatch")

        config["memory_authority"] = {"enabled": True, "db_path": str(authority_db)}
        config["gateway"].update(
            upstream_base_url="https://provider.fixture/v1",
            upstream_default_model="offline-reading-recall",
            core_memory_interval_rounds=0,
            favorite_memory_interval_rounds=0,
        )
        config["memory_system"]["recall"] = {"mode": "fast_hybrid"}
        config["embedding"].update(
            enabled=True,
            api_key="offline-only",
            base_url="https://embedding.fixture",
            model="offline-reading-recall",
        )

        # Gateway's authority view uses SQLite mode=ro. A local deterministic
        # vector/rerank path and MockTransport keep this request fully offline.
        os.environ["OMBRE_GATEWAY_TOKEN"] = "offline-reading-recall-token"
        os.environ["OMBRE_GATEWAY_UPSTREAM_API_KEY"] = "offline-only"
        os.environ["OMBRE_MEMORY_INDEX_ADMIN_URL"] = "http://source-transport-disabled.invalid"
        service = GatewayService(config)
        app = create_gateway_app(config=config, service=service)
        app.state.gateway_service = service
        record = service.memory_authority_view.memory_l1_recall_map([bucket_id]).get(bucket_id)
        require(isinstance(record, dict), "adopted_memory_not_readable")
        require(record.get("state") == "l1_v2", "adopted_memory_not_l1_v2")
        require(record.get("memory_id") == memory_id, "adopted_memory_id_mismatch")
        require(record.get("revision") == revision, "adopted_memory_revision_mismatch")
        document = record.get("document")
        require(isinstance(document, dict), "adopted_memory_document_missing")
        binding_ids = {
            str(row.get("binding_id"))
            for row in document.get("evidence_bindings", [])
            if isinstance(row, dict) and row.get("binding_id")
        }
        require(bool(binding_ids), "adopted_memory_bindings_missing")
        require(record.get("body_sha256") == body_sha256(body), "authority_body_hash_mismatch")

        source_bucket_path = service.bucket_mgr._find_bucket_file(bucket_id)
        require(bool(source_bucket_path), "copied_bucket_not_found")
        bucket = service.bucket_mgr._load_bucket(source_bucket_path, apply_activation=False)
        require(isinstance(bucket, dict), "copied_bucket_unreadable")
        require(bucket.get("id") == bucket_id, "bucket_frontmatter_id_mismatch")
        require(bucket.get("content") == body, "manifest_body_mismatch")

        verification_item = validate_source_verification(
            manifest["source_verification"], memory_id, revision, binding_ids,
        )

        async def record_source_read(_service, _request, _stage, path, payload):
            require(path == "/api/memory-authority/recall-sources", "unexpected_source_endpoint")
            items = payload.get("items") if isinstance(payload, dict) else None
            require(isinstance(items, list) and len(items) == 1, "gateway_source_scope_invalid")
            requested = items[0]
            require(requested.get("memory_id") == memory_id, "gateway_source_memory_mismatch")
            require(requested.get("revision") == revision, "gateway_source_revision_mismatch")
            require(set(requested.get("binding_ids") or []) == binding_ids,
                    "gateway_source_binding_ids_mismatch")
            require(requested.get("allow_historical", False) is False,
                    "gateway_requested_historical_source")
            source_requests.append(copy.deepcopy(payload))
            # This is the complete response already obtained and verified from
            # Brain; no Brain/production request is made by this runner.
            return copy.deepcopy(manifest["source_verification"])

        async def query_embeddings(texts, **_kwargs):
            embedding_inputs.extend(str(value) for value in texts)
            return [[1.0, 0.0] for _ in texts]

        async def rerank(_query, items, **kwargs):
            rerank_calls.append(len(items))
            documents_override = kwargs.get("documents_override") or {}
            target = [item for item in items if item.get("bucket", {}).get("id") == bucket_id]
            require(len(target) == 1, "adopted_card_not_in_recall_candidates")
            require(body in documents_override.get(bucket_id, ""),
                    "rerank_document_missing_adopted_body")
            for item in items:
                item["rerank_score"] = 0.95 if item in target else 0.0
            return items

        service.embedding_engine.query_embeddings = query_embeddings
        service._rerank_scored_bucket_candidates = rerank
        service.embedding_engine._store_embedding(
            bucket_id,
            [1.0, 0.0],
            source_revision=revision,
            source_sha256=body_sha256(body),
            parent_memory_id=memory_id,
            input_text=retrieval_text(body, document),
            text_projection_version=TEXT_PROJECTION_VERSION,
        )
        plan = {"scope": "talk.initial", "gateway_slice": {"blocks": [{
            "source_id": "ombre.memory_recall",
            "block_id": "l1",
            "scope": "talk.initial",
            "anchor": "current_user.before",
            "mode": "live_source",
            "role": "system",
            "wrapper_text": "顾衍的记忆\n{{content}}",
        }]}}
        service._prompt_plan_for_request = lambda *_args, **_kwargs: plan

        def offline_provider(request):
            require(request.url.host == "provider.fixture", "provider_transport_not_mocked")
            provider_requests.append(json.loads(request.content))
            response = openai_response(finish_reason="stop")
            response["id"] = "chatcmpl-reading-" + run_id
            return httpx.Response(200, json=response)

        await service.http_client.aclose()
        service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(offline_provider))
        original_read_batch = memory_use.read_batch
        memory_use.read_batch = record_source_read
        try:
            auth = {"Authorization": "Bearer offline-reading-recall-token"}
            async with service.http_client, httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway.fixture",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        **auth,
                        "X-Guyan-Conversation-Id": conversation_id,
                        "X-Ombre-Canonical-Source-Event-Id": user_event_id,
                        "X-Ombre-Canonical-Assistant-Source-Event-Id": assistant_event_id,
                        "X-Guyan-Trace-Id": "reading-recall-" + run_id,
                    },
                    json={
                        "model": "offline-reading-recall",
                        "messages": [{"role": "user", "content": query}],
                    },
                )
                require(response.status_code == 200, "ordinary_chat_http_failed")
                response_body = response.json()
                choices = response_body.get("choices") or []
                require(bool(choices), "ordinary_chat_response_missing")
                answer = (choices[0].get("message") or {}).get("content")
                require(isinstance(answer, str) and answer, "ordinary_chat_answer_missing")
                require(len(provider_requests) == 1, "unexpected_provider_request_count")

                wire_text = "\n".join(payload_texts(provider_requests[0]))
                require(wire_text.count(body) == 1, "adopted_body_not_injected_once")
                require("evidence_bindings" not in wire_text, "semantic_source_metadata_leaked_to_provider")
                require(bool(source_requests), "brain_source_verification_not_used")
                for payload in source_requests:
                    items = payload.get("items") or []
                    require(len(items) == 1, "source_verification_request_count_changed")
                    require(items[0].get("memory_id") == memory_id
                            and items[0].get("revision") == revision
                            and set(items[0].get("binding_ids") or []) == binding_ids,
                            "source_verification_request_identity_changed")

                lookup_response = await client.post(
                    "/internal/memory-delivery/lookup", json=turn, headers=auth,
                )
                require(lookup_response.status_code == 200, "delivery_lookup_http_failed")
                lookup = lookup_response.json()
                require(lookup.get("state") == "available", "delivery_receipt_unavailable")
                context = lookup.get("context") or {}
                fragments = context.get("fragments") or []
                require(len(fragments) == 1, "delivery_fragment_count_changed")
                fragment = fragments[0]
                require(fragment.get("memory_id") == memory_id, "delivery_memory_id_mismatch")
                require(fragment.get("bucket_id") == bucket_id, "delivery_bucket_id_mismatch")
                require(fragment.get("l1_revision") == revision, "delivery_revision_mismatch")
                require(fragment.get("coverage", {}).get("l1_body_sha256") == body_sha256(body),
                        "delivery_body_hash_mismatch")
                require(fragment.get("text_sha256") == sha(fragment.get("text", "")),
                        "delivery_text_hash_mismatch")
                states = lookup.get("fragment_states") or []
                require(len(states) == 1 and states[0].get("fragment_id") == fragment.get("fragment_id")
                        and states[0].get("state") == "available", "delivery_source_state_unavailable")
                requests = lookup.get("requests") or []
                require(len(requests) == 1, "delivery_request_count_changed")
                require(fragment.get("fragment_id") in requests[0].get("fragment_ids", []),
                        "delivery_fragment_not_bound_to_request")

                require(file_stamp(authority_db) == authority_stamp, "authority_db_changed_during_check")
                require(file_stamp(source_file) == bucket_stamp, "source_bucket_changed_during_check")
                return {
                    "status": "passed",
                    "scope": "isolated_offline_gateway",
                    "provider_transport": "MockTransport",
                    "source_verification": "supplied_brain_response",
                    "memory_id": memory_id,
                    "bucket_id": bucket_id,
                    "revision": revision,
                    "body_sha256": body_sha256(body),
                    "request_bound_ids_match": True,
                    "source_binding_count": len(binding_ids),
                    "body_occurrences_in_provider_request": wire_text.count(body),
                    "provider_request_count": len(provider_requests),
                    "source_verification_request_count": len(source_requests),
                    "conversation_id": conversation_id,
                    "user_source_event_id": user_event_id,
                    "assistant_source_event_id": assistant_event_id,
                    "assistant_response_id": response_body.get("id"),
                    "assistant_content_sha256": sha(answer),
                    "delivery_receipt_id": context.get("delivery_receipt_id"),
                    "delivery_sha256": context.get("delivery_sha256"),
                    "delivery_attempt_id": requests[0].get("attempt_id"),
                    "delivery_trace_id": requests[0].get("trace_id"),
                    "fragment_id": fragment.get("fragment_id"),
                    "fragment_state": states[0].get("state"),
                    "authority_db_unchanged": True,
                    "source_bucket_unchanged": True,
                    "created_or_committed_memory": False,
                }
        finally:
            memory_use.read_batch = original_read_batch
            if not service.http_client.is_closed:
                await service.http_client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="adopted reading recall JSON manifest")
    args = parser.parse_args()
    try:
        manifest = read_manifest(args.manifest.expanduser().resolve(strict=True))
        result = asyncio.run(run_check(manifest))
    except VerificationError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}, ensure_ascii=False))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}, ensure_ascii=False))
        return 3
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
