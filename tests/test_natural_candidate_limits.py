"""Independent channel limits through the ordinary-chat candidate path; local only."""
import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from test_memory_local_first_settings import _app, _base_config


@pytest.mark.parametrize("local_limit,vector_limit,scene_limit", [(3, 5, 7), (6, 4, 2)])
def test_natural_chat_consumes_independent_candidate_limits(tmp_path, monkeypatch,
                                                           local_limit, vector_limit, scene_limit):
    config = _base_config(tmp_path)
    config["gateway"].update(dynamic_top_k=local_limit,
                              semantic_candidate_top_k=vector_limit,
                              moment_search_limit=scene_limit)
    service, _application = _app(config)
    query = "The quiet coastal train trip helped me relax."
    buckets = [{"id": f"trip-{i}", "content": f"On quiet coastal train trip {i}, I relaxed by the sea.",
                "metadata": {"type": "dynamic", "name": f"Coastal train trip {i}"}}
               for i in range(12)]
    metadata = {bucket["id"]: {"memory_id": "memory:" + bucket["id"], "revision": 1,
                "body_sha256": hashlib.sha256(bucket["content"].encode()).hexdigest()}
                for bucket in buckets}
    service.memory_authority_view = SimpleNamespace(enabled=False,
        memory_index_metadata_map=lambda _ids: metadata)
    # Deterministic local scores still flow through the production Top-K consumer.
    monkeypatch.setattr(service.bucket_mgr, "calc_topic_scores", lambda _query, eligible:
                        {bucket["id"]: .99 - i * .01 for i, bucket in enumerate(eligible)})
    observed = {"local": [], "vector": [], "scene": []}
    keyword_search = service._get_keyword_candidates

    def local_search(text, eligible):
        result = keyword_search(text, eligible)
        observed["local"].append(result)
        return result

    async def vector_search(queries, *, top_k, eligible_ids, **_options):
        result = [{"bucket_id": bucket["id"], "score": .93, "unit_id": bucket["id"] + ":unit",
                   "index_status": "verified"}
                  for bucket in buckets if bucket["id"] in eligible_ids][:top_k]
        observed["vector"].append((queries, top_k, result))
        return result

    moment_search = service.memory_moment_store.search_moment_items

    def scene_search(text, moments, *, limit, **options):
        result = moment_search(text, moments, limit=limit, **options)
        observed["scene"].append((text, limit, len(moments), result))
        return result

    monkeypatch.setattr(service, "_get_keyword_candidates", local_search)
    monkeypatch.setattr(service, "embedding_engine", SimpleNamespace(enabled=True,
        search_similar_queries=vector_search))
    monkeypatch.setattr(service.memory_moment_store, "search_moment_items", scene_search)
    # Exercise the same natural_input entry used by ordinary chat, including
    # the real candidate builder, local Top-K, fresh moment parser/search and finish.
    request = {"q_current": query, "q_context": query, "metadata": {}, "route": "ordinary"}
    result = asyncio.run(service._select_dynamic_moments(
        query, "independent-limits", buckets, {}, natural_input=request,
        allow_semantic=True, allow_query_planner=False, allow_rerank=False))
    assert len(result) == 5
    assert len(observed["local"]) == 1
    assert len(observed["local"][0]) == local_limit
    assert len(observed["vector"]) == 1
    assert observed["vector"][0][0] == [query]
    assert observed["vector"][0][1] == vector_limit
    assert len(observed["vector"][0][2]) == vector_limit
    assert len(observed["scene"]) == 1
    text, actual_limit, input_count, hits = observed["scene"][0]
    assert text == query and actual_limit == scene_limit
    assert input_count > max(local_limit, vector_limit, scene_limit)
    assert len(hits) == scene_limit
