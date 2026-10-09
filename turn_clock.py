"""Authenticated Aiz turn clock, scoped to one prepare call (never global state)."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_CLOCK = ContextVar("gateway_turn_clock", default=None)
TIME_HEADER = "X-Guyan-Turn-Time-Utc"
ZONE_HEADER = "X-Guyan-Turn-Timezone"
BODY_HEADER = "X-Guyan-Time-Context-Sha256"


def read_clock(request):
    headers = getattr(request, "headers", {}) or {}
    if not (headers.get("X-Ombre-Canonical-Source-Event-Id")
            and headers.get("X-Ombre-Canonical-Assistant-Source-Event-Id")):
        return None
    raw = headers.get(TIME_HEADER, "")
    zone = headers.get(ZONE_HEADER, "")
    if not raw and not zone:
        return None
    try:
        if len(raw) > 64 or len(zone) > 80:
            raise ValueError()
        instant = datetime.fromisoformat(raw)
        if instant.tzinfo is None:
            raise ValueError()
        tz = ZoneInfo(zone)
        body_sha = headers.get(BODY_HEADER, "")
        if body_sha and not re.fullmatch(r"[a-f0-9]{64}", body_sha):
            raise ValueError()
        return {"reference_utc": instant.astimezone(timezone.utc).isoformat(timespec="seconds"),
                "timezone": zone, "now": instant.astimezone(tz), "body_sha256": body_sha}
    except (ValueError, KeyError, TypeError):
        raise ValueError("invalid canonical turn clock") from None


@contextmanager
def use_clock(request, *, enabled=True):
    value = read_clock(request) if enabled else None
    token = _CLOCK.set(value)
    try:
        yield clock_metadata()
    finally:
        _CLOCK.reset(token)


def clock_metadata(messages=None):
    value = _CLOCK.get()
    if not value:
        return {"source": "gateway_clock"}
    result = {"source": "aizizhu_turn_snapshot", "reference_utc": value["reference_utc"],
              "timezone": value["timezone"], "body_sha256": value["body_sha256"]}
    if messages is not None:
        matches = [{"message_index": i, "role": item.get("role")}
                   for i, item in enumerate(messages) if isinstance(item, dict)
                   and isinstance(item.get("content"), str)
                   and hashlib.sha256(item["content"].encode()).hexdigest() == value["body_sha256"]]
        result.update(physical_matches=matches, physical_count=len(matches),
                      physical_verified=bool(value["body_sha256"]) and len(matches) == 1)
    return result


def reference_now(default_tz):
    value = _CLOCK.get()
    return value["now"] if value else datetime.now(default_tz)
