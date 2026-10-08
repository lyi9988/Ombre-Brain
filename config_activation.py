"""Owner-safe save/activation receipts; never call an inference provider here."""

import asyncio
from urllib.parse import urlsplit, urlunsplit

import httpx


def activation_receipt(fields=(), *, state="not_required", reason=None, attempted=False):
    return {
        "state": state,
        "reason": reason,
        "attempted": attempted,
        "requested_fields": sorted(fields),
        "confirmed_fields": [],
        "unconfirmed_fields": sorted(fields),
        # A config acknowledgement is not an authentication/inference probe.
        "provider_verified": False,
    }


def _health_url(admin_url):
    parts = urlsplit(admin_url)
    path = parts.path.rstrip("/")
    if not path.endswith("/api/config"):
        return None
    return urlunsplit((parts.scheme, parts.netloc, path[:-11] + "/health", "", ""))


def _embedding_matches(health, expected, fields):
    """Confirm a runtime snapshot, not the identity/validity of an external key."""
    if not isinstance(health, dict) or health.get("status") != "ok":
        return False
    gateway = health.get("gateway")
    if not isinstance(gateway, dict):
        return False
    observed = gateway.get("embedding")
    runtime = gateway.get("retrieval_runtime")
    if not isinstance(observed, dict) or not isinstance(runtime, dict):
        return False
    overlay = runtime.get("runtime_overlay")
    if not isinstance(overlay, dict) or overlay.get("last_reload_status") not in ("observed", "reloaded"):
        return False
    for key in ("enabled", "model", "base_url"):
        wanted, actual = expected.get(key), observed.get(key)
        if key == "base_url" and isinstance(wanted, str) and isinstance(actual, str):
            wanted, actual = wanted.rstrip("/"), actual.rstrip("/")
        if wanted is None or actual != wanted:
            return False
    if expected.get("enabled") is True and observed.get("configured") is not True:
        return False
    if "embedding.dimensions" in fields:
        if ("requested_dimension" not in observed
                or observed["requested_dimension"] != expected.get("dimensions")):
            return False
    if "embedding.api_key" in fields:
        name = "OMBRE_EMBEDDING_API_KEY"
        sources, matches = overlay.get("credential_sources"), overlay.get("credential_matches_observed_mount")
        if (overlay.get("credential_source_readable") is not True
                or not isinstance(sources, dict) or sources.get(name) != "mounted_env"
                or not isinstance(matches, dict) or matches.get(name) is not True
                or observed.get("configured") is not True):
            return False
    return True


def _reranker_matches(observed, expected):
    if not isinstance(observed, dict):
        return False
    for key, wanted in expected.items():
        if key == "api_key":
            if wanted and observed.get("api_ready") is not True:
                return False
            continue
        actual = observed.get(key)
        if key == "base_url" and isinstance(wanted, str) and isinstance(actual, str):
            wanted, actual = wanted.rstrip("/"), actual.rstrip("/")
        if actual != wanted:
            return False
    return True


async def activate_gateway(payload, *, admin_url, token, embedding_fields=(),
                           embedding_persisted=False, embedding_expected=None):
    """One bounded save sync; POST ACK and shared-mount readback are distinct.

    Embedding has no POST handler. Its persisted values are confirmed using the
    existing health endpoint, which triggers overlay reload but no inference.
    No response body, key, key fingerprint or arbitrary remote string is returned.
    """
    post_fields = {f"{section}.{key}" for section, values in payload.items()
                   for key in values}
    embedding_fields = set(embedding_fields)
    fields = post_fields | embedding_fields
    receipt = activation_receipt(fields)
    if not fields:
        return receipt
    receipt.update(state="unconfirmed", reason="not_attempted")
    if not payload and not embedding_persisted:
        receipt["reason"] = "embedding_requires_persistence"
        return receipt
    if not admin_url or not token:
        receipt.update(state="not_configured", reason="gateway_admin_not_configured")
        return receipt

    async def sync():
        async with httpx.AsyncClient(timeout=2.0, follow_redirects=False, trust_env=False) as client:
            headers = {"Authorization": f"Bearer {token}"}
            if payload:
                receipt["attempted"] = True
                response = await client.post(admin_url, headers=headers, json=payload)
                receipt["http_status"] = response.status_code
                if not 200 <= response.status_code < 300:
                    receipt.update(state="failed", reason="gateway_http_error")
                    return
                ack = response.json()
                acknowledged = ack.get("updated") if isinstance(ack, dict) else None
                if (not isinstance(ack, dict) or ack.get("ok") is not True
                        or not isinstance(acknowledged, list)
                        or any(not isinstance(field, str) for field in acknowledged)):
                    receipt["reason"] = "invalid_acknowledgement"
                    return
                confirmed = post_fields.intersection(acknowledged)
                receipt["confirmed_fields"] = sorted(confirmed)
                receipt["unconfirmed_fields"] = sorted(fields - confirmed)
                receipt["post_acknowledged"] = not bool(post_fields - confirmed)
                if post_fields - confirmed:
                    receipt["reason"] = "incomplete_acknowledgement"
                    return
                if "reranker" in payload:
                    if not _reranker_matches(ack.get("reranker"), payload["reranker"]):
                        confirmed -= {f for f in post_fields if f.startswith("reranker.")}
                        receipt["confirmed_fields"] = sorted(confirmed)
                        receipt["unconfirmed_fields"] = sorted(fields - confirmed)
                        receipt["reason"] = "reranker_runtime_not_confirmed"
                        return
                    receipt["reranker_confirmation"] = "admin_runtime_snapshot"
            if embedding_fields:
                if not embedding_persisted:
                    receipt["reason"] = "embedding_requires_persistence"
                    return
                health_url = _health_url(admin_url)
                if not health_url:
                    receipt["reason"] = "health_endpoint_unavailable"
                    return
                receipt["attempted"] = True
                response = await client.get(health_url, headers=headers)
                receipt["health_http_status"] = response.status_code
                if not 200 <= response.status_code < 300:
                    receipt["reason"] = "embedding_readback_failed"
                    return
                if not _embedding_matches(response.json(), embedding_expected or {}, embedding_fields):
                    receipt["reason"] = "embedding_runtime_not_confirmed"
                    return
                receipt["embedding_confirmation"] = "shared_mount_runtime_snapshot"
                receipt["confirmed_fields"] = sorted(fields)
                receipt["unconfirmed_fields"] = []
            receipt.update(state="confirmed", reason=None)

    try:
        # Bound the entire POST + optional readback, not 2s per network phase.
        await asyncio.wait_for(sync(), timeout=2.0)
    except (httpx.TimeoutException, asyncio.TimeoutError):
        # A timed-out write may have been applied. No blind automatic retries.
        receipt["reason"] = "gateway_timeout"
    except httpx.HTTPError:
        receipt["reason"] = "gateway_transport_error"
    except (TypeError, ValueError):
        receipt["reason"] = "invalid_acknowledgement"
    except Exception:
        receipt["reason"] = "gateway_sync_error"
    return receipt


def config_save_result(updated, persistence, gateway_activation, *, error=None):
    """Keep legacy ok for the local save, with a distinct activation contract."""
    attention = bool(error) or gateway_activation["state"] not in ("confirmed", "not_required")
    result = {
        "ok": error is None,
        "updated": list(updated),
        "local_update": "applied",
        "persistence": dict(persistence),
        "gateway_activation": gateway_activation,
        "attention_required": attention,
    }
    if error:
        result["error"] = error
    return result
