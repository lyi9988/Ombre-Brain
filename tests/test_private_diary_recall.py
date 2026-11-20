"""Offline date route, adopted version, Composer and final-wire checks."""
import asyncio
import json
from types import SimpleNamespace
from datetime import timezone

import httpx
import pytest

from attachment_adoption_support import openai_response
from gateway import GatewayService, create_gateway_app
from memory_delivery import payload_texts, sha
from private_diary_recall import queries, read
from test_memory_local_first_settings import _base_config


DATE = "2026-11-08"
ORIGINAL = "开头没有提及这个问题。\n\n"
MATCH = "那天我写到蓝桦工程的桥面安排，心里有些担心。"
BODY = ORIGINAL + MATCH + "\n\n" + "后面还有很多内容。" * 130
QUESTION = "2026年11月8日你的私人日记里关于蓝桦工程写了什么？"
SCOPE = {"conversation_id": "owner-conversation", "user_source_event_id": "user-a",
         "assistant_source_event_id": "assistant-a"}


def _bridge_response(request, calls, *, changed=False):
    assert request.url.path == "/bridge/v1/private-diary/" + DATE
    assert request.url.params["conversation_id"] == SCOPE["conversation_id"]
    assert request.headers["Authorization"] == "Bearer bridge-fixture"
    calls.append(dict(request.url.params))
    source = {"source_namespace": "snapshot.aiz.diary.private", "source_id": f"diary:private:{DATE}",
              "source_revision": "adopted-r1", "state": "available", "eligible_source": True,
              "permission": {"authorized": True, "scope": "owner.private_diary"},
              "content_sha256": sha(BODY), "content_chars": len(BODY), "adopted_at_ms": 1794115200000,
              "author": {"kind": "assistant", "name": "顾衍"}}
    if "revision" in request.url.params:
        assert request.url.params["revision"] == "adopted-r1"
        start, end = (int(request.url.params[k]) for k in ("start", "end"))
        if changed:
            source.update(state="changed", eligible_source=False, source_revision="adopted-r2")
        else:
            source.update(text=BODY[start:end], range={"start": start, "end": end, "coordinate_unit": "unicode_codepoint"})
    else:
        source["paragraphs"] = [{"start": len(ORIGINAL), "end": len(ORIGINAL) + len(MATCH),
                                 "match_start": len(ORIGINAL) + 6}]
    return httpx.Response(200, json={"identity_id": "owner", "conversation_id": SCOPE["conversation_id"],
                                    "source": source})


def test_dated_adopted_diary_excerpt_has_actual_range_and_full_text_continuation():
    first = queries(QUESTION + " 2026年11月7日我们聊过什么？", tz=timezone.utc)
    assert [row["date"] for row in first] == [DATE]
    assert "蓝桦工程" in first[0]["focus"]
    assert not queries("2026年11月8日我们谈过什么？", tz=timezone.utc)
    assert not queries("你的私人日记写了什么？", tz=timezone.utc)

    async def scenario():
        calls = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: _bridge_response(req, calls))) as client:
            adapter = SimpleNamespace(base_url="http://aiz.fixture", conversation_id=SCOPE["conversation_id"],
                                      http_client=client, _headers=lambda: {"Authorization": "Bearer bridge-fixture"})
            text, evidence = await read(adapter, first[0], budget=250, timeout=2)
            assert MATCH in text and ORIGINAL not in text
            assert evidence["range"] == {"start": len(ORIGINAL), "end": len(ORIGINAL) + len(MATCH)}
            assert evidence["excerpt_sha256"] == sha(MATCH) and evidence["complete"] is False
            full = queries("请逐字读2026年11月8日私人日记全文", tz=timezone.utc)[0]
            _, page = await read(adapter, full, budget=145, timeout=2)
            assert page["range"]["start"] == 0 and page["range"]["end"] < len(BODY)
            next_char = page["range"]["end"] + 1
            next_page = queries(f"2026年11月8日日记，从第{next_char}个字继续", tz=timezone.utc)[0]
            text, continued = await read(adapter, next_page, budget=145, timeout=2)
            assert continued["range"]["start"] == next_char - 1 and "如需续读" in text
            assert "focus" not in calls[-2] and int(calls[-1]["start"]) == next_char - 1

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["available", "restart_from_first_char", "changed", "other_conversation", "composer_off", "ordinary"])
def test_diary_is_owner_scoped_and_only_receipted_on_actual_talk_wire(tmp_path, monkeypatch, case):
    config = _base_config(tmp_path)
    config["gateway"].update(upstream_base_url="https://provider.fixture/v1", upstream_default_model="public-model",
                             date_recall_enabled=True, date_recall_budget=520)
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-v2-test")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "fixture")
    service = GatewayService(config)
    calls, sent = [], []

    def provider(request):
        assert request.url.host == "provider.fixture"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=openai_response(finish_reason="stop"))

    async def canonical(request, payload, session_id, query):
        # Only external conversation mirroring is stubbed. Talk and Composer
        # consume the ordinary HTTP trace produced from the incoming headers.
        return payload, {}

    plan = {"scope": "talk.initial", "gateway_slice": {"blocks": [{
        "source_id": "ombre.date_recall", "block_id": "diary", "scope": "talk.initial",
        "anchor": "current_user.before", "mode": "live_source", "role": "system",
        "enabled": case != "composer_off", "wrapper_text": "历史资料\n{{content}}",
    }]}}
    monkeypatch.setattr(service, "_prompt_plan_for_request", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(service, "_prepare_canonical_turn", canonical)
    monkeypatch.setattr(service, "_canonical_continuation_active", lambda _: True)

    async def scenario():
        await service.http_client.aclose()
        service.http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        app = create_gateway_app(config=config, service=service)
        app.state.gateway_service = service
        auth = {"Authorization": "Bearer gateway-v2-test"}
        scope = {**SCOPE, "conversation_id": "somewhere-else"} if case == "other_conversation" else SCOPE
        async with service.http_client, httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: _bridge_response(req, calls, changed=case == "changed"))
        ) as bridge, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.fixture") as client:
            service.canonical_adapter = SimpleNamespace(enabled=True, conversation_id=SCOPE["conversation_id"],
                base_url="http://aiz.fixture", http_client=bridge,
                _headers=lambda: {"Authorization": "Bearer bridge-fixture"})
            question = {
                "ordinary": "今天早上好。",
                "restart_from_first_char": "2026年11月8日的私人日记，从第1个字继续。",
            }.get(case, QUESTION)
            response = await client.post("/v1/chat/completions", headers={**auth,
                "X-Guyan-Conversation-Id": scope["conversation_id"],
                "X-Ombre-Canonical-Source-Event-Id": scope["user_source_event_id"],
                "X-Ombre-Canonical-Assistant-Source-Event-Id": scope["assistant_source_event_id"]},
                json={"model": "public-model", "messages": [{"role": "user",
                    "content": question}]})
            assert response.status_code == 200, response.text
            wire = "\n".join(payload_texts(sent[-1]))
            lookup = (await client.post("/internal/memory-delivery/lookup", headers=auth, json=scope)).json()
            requests = lookup.get("requests") or []
            diary_sources = requests[0].get("private_diary_sources") if requests else None
            if case == "available":
                assert wire.count(MATCH) == 1 and "历史资料" in wire and ORIGINAL not in wire
                assert diary_sources[0]["source_revision"] == "adopted-r1"
                assert diary_sources[0]["excerpt_sha256"] == sha(MATCH)
                assert lookup["context"]["fragments"] == [], "a diary is not a formal L1 card"
                debug = service.state_store.list_injection_debug(limit=1)[0]
                assert MATCH not in json.dumps(debug, ensure_ascii=False)
            elif case == "restart_from_first_char":
                assert "focus" not in calls[0] and calls[1]["start"] == "0"
                page = diary_sources[0]
                end = page["range"]["end"]
                assert page["range"]["start"] == 0 and 0 < end < len(BODY)
                assert BODY[:end] in wire and f"从第 {end + 1} 个字继续" in wire
                assert page["source_revision"] == "adopted-r1" and page["excerpt_sha256"] == sha(BODY[:end])
            else:
                assert MATCH not in wire and not diary_sources
            assert len(calls) == (2 if case in {"available", "restart_from_first_char", "changed"} else 0)

    asyncio.run(scenario())
