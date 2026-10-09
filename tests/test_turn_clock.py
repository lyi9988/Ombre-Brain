import asyncio
import hashlib
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from gateway import GatewayService
from turn_clock import use_clock, reference_now, clock_metadata, read_clock


def request(instant="2026-10-09T15:59:59+00:00", zone="Asia/Shanghai"):
    return SimpleNamespace(headers={"X-Ombre-Canonical-Source-Event-Id": "user",
        "X-Ombre-Canonical-Assistant-Source-Event-Id": "assistant",
        "X-Guyan-Turn-Time-Utc": instant, "X-Guyan-Turn-Timezone": zone})


def test_relative_dates_use_turn_clock_not_wall_clock():
    service = GatewayService.__new__(GatewayService)
    service.gateway_tz = ZoneInfo("Asia/Shanghai")
    with use_clock(request()):
        assert service._query_date_hint("昨天")['date'] == "2026-10-08"
        assert service._query_date_recall_hint("昨天")['date'] == "2026-10-08"
    with use_clock(request("2026-10-09T16:00:01+00:00")):
        assert service._query_date_hint("昨天")['date'] == "2026-10-09"
    assert clock_metadata() == {"source": "gateway_clock"}


def test_date_range_uses_selected_zone_and_dst_boundary():
    service = GatewayService.__new__(GatewayService)
    service.gateway_tz = ZoneInfo("Asia/Shanghai")
    with use_clock(request("2026-11-01T06:30:00+00:00", "America/New_York")):
        start, end = service._date_recall_range("2026-11-01")
        assert str(start.tzinfo) == "America/New_York"
        assert end.timestamp() - start.timestamp() == 25 * 3600


def test_untrusted_missing_or_malformed_clock():
    assert read_clock(None) is None
    assert read_clock(SimpleNamespace(headers={"X-Guyan-Turn-Time-Utc": "fake"})) is None
    with pytest.raises(ValueError):
        read_clock(request("2026-10-10T01:00:00"))
    with pytest.raises(ValueError):
        read_clock(request(zone="bad/unknown"))


def test_concurrent_clocks_reset_on_failure_and_internal_scope():
    async def task(instant):
        with use_clock(request(instant)):
            await asyncio.sleep(0)
            return clock_metadata()["reference_utc"]
    async def run():
        return await asyncio.gather(task("2026-10-09T00:00:00+00:00"), task("2026-10-10T00:00:00+00:00"))
    assert asyncio.run(run()) == ["2026-10-09T00:00:00+00:00", "2026-10-10T00:00:00+00:00"]
    with pytest.raises(RuntimeError):
        with use_clock(request()):
            raise RuntimeError()
    assert clock_metadata() == {"source": "gateway_clock"}
    with use_clock(request(), enabled=False):
        assert clock_metadata() == {"source": "gateway_clock"}


def test_prepare_wrapper_exposes_snapshot_and_never_modifies_input():
    service = GatewayService.__new__(GatewayService)
    async def inner(payload, session_id, **kwargs):
        await asyncio.sleep(0)
        assert clock_metadata()["source"] == "aizizhu_turn_snapshot"
        return dict(payload), [], {}
    service._prepare_payload_with_clock = inner
    payload = {"messages": [{"role": "user", "content": "昨天"}]}
    result = asyncio.run(service.prepare_payload(payload, "s", request=request(), include_debug=True))
    assert result[0] == payload
    assert result[2]["time_context"]["reference_utc"] == "2026-10-09T15:59:59+00:00"
    assert clock_metadata() == {"source": "gateway_clock"}


def test_physical_clock_match_and_prepare_cache_identity(tmp_path):
    from test_gateway_sdk_tool_phase import _snapshot_service, _snapshot_headers, _snapshot_payload, make_request
    service = _snapshot_service(tmp_path)
    clock_body = "synthetic complete clock data"
    headers = {**_snapshot_headers(), **request().headers,
               "X-Guyan-Time-Context-Sha256": hashlib.sha256(clock_body.encode()).hexdigest()}
    payload = _snapshot_payload([{"role": "system", "content": "instructions"},
                                {"role": "system", "content": clock_body},
                                {"role": "user", "content": "ordinary input"}])
    async def run():
        first = await service.prepare_payload(payload, "s", include_debug=True, request=make_request(headers))
        again = await service.prepare_payload(payload, "s", include_debug=True, request=make_request(headers))
        changed = {**headers, "X-Guyan-Turn-Time-Utc": "2026-10-09T16:00:01+00:00"}
        next_clock = await service.prepare_payload(payload, "s", include_debug=True, request=make_request(changed))
        return first, again, next_clock
    first, again, changed = asyncio.run(run())
    assert first[2]["time_context"]["physical_verified"] is True
    assert first[2]["time_context"]["physical_matches"] == [{"message_index": 1, "role": "system"}]
    assert again[2]["prepare_snapshot_cache"]["status"] == "hit"
    assert first[2]["prepare_snapshot_cache"]["key_hash"] != changed[2]["prepare_snapshot_cache"]["key_hash"]
    assert "time_context" not in first[0]  # sidecar never becomes provider content
    assert clock_metadata() == {"source": "gateway_clock"}
