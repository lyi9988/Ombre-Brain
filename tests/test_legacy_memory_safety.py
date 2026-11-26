"""Exercise the actual legacy-hook route handlers and enrichment entrypoints."""
import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from starlette.applications import Starlette
from starlette.routing import Route


def isolated_handlers(reflection_config):
    path = Path(__file__).resolve().parents[1] / "server.py"
    names = {
        "dream_hook",
        "_legacy_memory_enrichment_enabled",
        "_queue_memory_enrichment",
        "_enrich_memory_async",
        "_bucket_needs_memory_enrichment",
        "_backfill_memory_enrichment",
        "enrich_backfill",
        "_int_between",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert names == {node.name for node in functions}
    routes = []
    for fn in functions:
        if fn.name == "dream_hook":
            routes = [(ast.literal_eval(d.args[0]),
                       ast.literal_eval(next(k.value for k in d.keywords if k.arg == "methods")))
                      for d in fn.decorator_list]
        fn.decorator_list = []
    bucket = {"id": "fixture", "metadata": {"name": "Should never leak"}}
    namespace = dict(
        config={"reflection": reflection_config}, asyncio=asyncio, logger=logging.getLogger(__name__),
        bucket_mgr=SimpleNamespace(get=AsyncMock(return_value=bucket),
                                   list_all=AsyncMock(return_value=[bucket])),
        reflection_engine=SimpleNamespace(enrich_bucket=AsyncMock(return_value={})),
        embedding_engine=object(), memory_edge_store=object(), is_self_anchor_bucket=lambda _: False,
        _refresh_entity_edges_for_bucket_id=AsyncMock(return_value=0),
    )
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    app = Starlette(routes=[Route(route, namespace["dream_hook"], methods=methods) for route, methods in routes])
    return namespace, app


def test_retired_hooks_return_410_without_bucket_access():
    ns, app = isolated_handlers({})
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            for path in ("/dream-hook", "/introspection-hook"):
                response = await client.get(path)
                assert response.status_code == 410
                assert response.text == "Retired legacy memory hook."
    asyncio.run(run())
    ns["bucket_mgr"].list_all.assert_not_called()


def test_shared_runtime_flag_guards_queue_and_direct_backfill():
    ns, _ = isolated_handlers({"enrich_backfill_enabled": False})
    async def run():
        async_spy = AsyncMock(wraps=ns["_enrich_memory_async"])
        ns["_enrich_memory_async"] = async_spy
        ns["_queue_memory_enrichment"]("fixture", force=True)
        async_spy.assert_not_called()
        # A previously queued coroutine must still recheck when it starts.
        await async_spy("fixture", force=True)
        assert await ns["enrich_backfill"](10) == {"processed": 0, "ids": [], "errors": []}
        assert (await ns["_backfill_memory_enrichment"](limit=5, bucket_mgr_arg=ns["bucket_mgr"],
                    reflection_engine_arg=ns["reflection_engine"]))["processed"] == 0
    asyncio.run(run())
    ns["bucket_mgr"].list_all.assert_not_called()
    ns["bucket_mgr"].get.assert_not_called()
    ns["reflection_engine"].enrich_bucket.assert_not_called()
    ns["_refresh_entity_edges_for_bucket_id"].assert_not_called()


def test_default_enrichment_still_runs_and_rechecks_after_async_reads():
    ns, _ = isolated_handlers({})
    async def run():
        ns["_queue_memory_enrichment"]("fixture", force=True)
        await asyncio.sleep(0)
        ns["reflection_engine"].enrich_bucket.assert_awaited_once()
        result = await ns["enrich_backfill"](1)
        assert result["ids"] == ["fixture"]
        assert ns["reflection_engine"].enrich_bucket.await_count == 2

        async def disable_during_read(*args, **kwargs):
            ns["config"]["reflection"]["enrich_backfill_enabled"] = False
            return {"id": "fixture", "metadata": {}}
        ns["bucket_mgr"].get.side_effect = disable_during_read
        await ns["_enrich_memory_async"]("fixture")
        assert ns["reflection_engine"].enrich_bucket.await_count == 2
    asyncio.run(run())
