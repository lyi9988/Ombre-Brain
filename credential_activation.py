"""Compare requested credentials with consumers in memory, expose only a nonce."""
import re


def adoption_receipt(service, body):
    generation = body.get("_credential_generation")
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
        return None
    confirmed = []
    for section, attribute in (("reranker", "reranker_engine"),
                               ("dehydration", "dehydrator"),
                               ("persona", "persona_engine"),
                               ("dream", "dream_engine")):
        expected = body.get(section) or {}
        if expected.get("api_key") and getattr(getattr(service, attribute, None), "api_key", None) == expected["api_key"]:
            confirmed.append(section + ".api_key")
    gateway = body.get("gateway") or {}
    if gateway.get("domain_sentinel_api_key") and getattr(service, "domain_sentinel_api_key", None) == gateway["domain_sentinel_api_key"]:
        confirmed.append("gateway.domain_sentinel_api_key")
    upstreams = gateway.get("upstreams")
    if isinstance(upstreams, list):
        observed = {item["name"]: item for item in service.upstreams}
        matches = True
        for upstream in upstreams:
            actual = observed.get(upstream.get("name"), {})
            requested = upstream.get("api_keys") or []
            values = [item.get("api_key") if isinstance(item, dict) else item for item in requested]
            actual_values = [item.get("value") for item in actual.get("api_keys", [])]
            if any(value and value not in actual_values for value in values):
                matches = False
            primary = upstream.get("credential_primary_env")
            primary_value = next((item.get("api_key") for item in requested
                if isinstance(item, dict) and item.get("label") == f"env:{primary}"), None)
            if primary_value and (not actual_values or actual_values[0] != primary_value):
                matches = False
        if matches:
            confirmed.append("gateway.upstreams")
    return {"generation": generation, "confirmed_fields": confirmed,
            "state": "consumer_snapshot", "provider_verified": False,
            "host_target_confirmed": False}
