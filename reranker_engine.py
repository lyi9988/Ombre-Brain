from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("ombre_brain.reranker")


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class RerankerEngine:
    """Small SiliconFlow-compatible rerank client."""

    def __init__(self, config: dict):
        config = config or {}
        embed_cfg = config.get("embedding", {}) or {}
        rerank_cfg = config.get("reranker", {}) or {}
        dehy_cfg = config.get("dehydration", {}) or {}

        self.model = str(rerank_cfg.get("model") or "Qwen/Qwen3-Reranker-4B")
        self.base_url = str(
            rerank_cfg.get("base_url")
            or embed_cfg.get("base_url")
            or dehy_cfg.get("base_url")
            or ""
        ).rstrip("/")
        self.api_key = str(
            rerank_cfg.get("api_key")
            or embed_cfg.get("api_key")
            or dehy_cfg.get("api_key")
            or ""
        )
        self.enabled = bool(self.api_key and self.base_url) and _bool_value(
            rerank_cfg.get("enabled", True)
        )
        self.timeout = _float_between(rerank_cfg.get("timeout_seconds", 12), 12, 1, 120)
        self.candidate_limit = _int_between(rerank_cfg.get("candidate_limit", 20), 20, 1, 100)
        self.score_weight = _float_between(rerank_cfg.get("score_weight", 0.65), 0.65, 0.0, 1.0)
        self._runtime = {
            "last_status": "not_requested",
            "last_error_type": "",
            "last_http_status": None,
            "last_latency_ms": None,
            "last_result_count": None,
            "transport": "per_call_httpx",
            "connection_fallback_count": 0,
            "last_request_model": None,
        }
        self._last_request_fingerprint: str | None = None

    def _configuration_fingerprint(self, identity: tuple[str, str, str] | None = None) -> str:
        # Process-local comparison only; neither credential nor digest is
        # exposed by runtime_debug or request diagnostics.
        material = identity if identity is not None else (self.model, self.base_url, self.api_key)
        return hashlib.sha256(json.dumps(material).encode("utf-8")).hexdigest()

    def runtime_debug(self) -> dict:
        """Return owner-safe rerank health without credentials or documents."""
        return {
            "enabled": bool(self.enabled),
            "configured": bool(self.api_key and self.base_url),
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout,
            "candidate_limit": self.candidate_limit,
            "last_result_matches_current_config": (
                self._last_request_fingerprint == self._configuration_fingerprint()
                if self._last_request_fingerprint is not None else None
            ),
            **dict(self._runtime),
        }

    async def rerank_with_diagnostics(
        self, query: str, documents: list[str], top_n: int | None = None,
    ) -> tuple[list[RerankResult], dict[str, Any]]:
        diagnostics: dict[str, Any] = {}
        results = await self.rerank(query, documents, top_n=top_n, diagnostics=diagnostics)
        return results, diagnostics

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None,
        *, diagnostics: dict[str, Any] | None = None,
    ) -> list[RerankResult]:
        # Each call owns its outcome. Concurrent calls and cached scores must
        # never borrow another request's mutable runtime health snapshot.
        model, base_url, api_key = self.model, self.base_url, self.api_key
        enabled, timeout = self.enabled, self.timeout
        state = {"last_status": "not_requested", "last_error_type": "",
                 "last_http_status": None, "last_latency_ms": None,
                 "last_result_count": 0, "last_request_model": model}
        fingerprint = self._configuration_fingerprint((model, base_url, api_key))

        def publish(changes: dict[str, Any]) -> None:
            state.update(changes)
            self._runtime.update(state)
            self._last_request_fingerprint = fingerprint
            if diagnostics is not None:
                diagnostics.update(state)

        if not enabled or not query or not documents:
            publish({
                "last_status": "disabled" if not enabled else "empty_input",
                "last_error_type": "",
                "last_http_status": None,
                "last_latency_ms": 0,
                "last_result_count": 0,
            })
            return []
        endpoint = f"{base_url}/rerank"
        payload: dict[str, Any] = {
            "model": model,
            "query": str(query),
            "documents": [str(document or "") for document in documents],
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = max(1, min(int(top_n), len(documents)))

        started = time.perf_counter()
        publish({
            "last_status": "started",
            "last_error_type": "",
            "last_http_status": None,
            "last_latency_ms": None,
            "last_result_count": None,
        })
        try:
            # The persistent transport was measured against the live provider
            # and returned stable 500s.  Keep the known-good per-call client
            # until that provider behavior is separately explained.
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                publish({"last_http_status": response.status_code})
                response.raise_for_status()
                body = response.json()
        except asyncio.CancelledError:
            publish({
                "last_status": "cancelled",
                "last_error_type": "CancelledError",
                "last_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "last_result_count": 0,
            })
            raise
        except Exception as exc:
            publish({
                "last_status": "error",
                "last_error_type": type(exc).__name__,
                "last_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "last_result_count": 0,
            })
            logger.warning("Reranker request failed | type=%s status=%s",
                           type(exc).__name__, state["last_http_status"])
            return []

        results = []
        raw_results = body.get("results") if isinstance(body, dict) else None
        invalid = not isinstance(raw_results, list)
        seen = set()
        for item in raw_results if isinstance(raw_results, list) else []:
            try:
                if not isinstance(item, dict) or type(item.get("index")) not in (int, str):
                    raise ValueError("invalid index")
                index = int(item["index"])
                raw_score = item.get("relevance_score")
                if type(raw_score) not in (int, float, str):
                    raise ValueError("invalid score")
                score = float(raw_score)
                if not math.isfinite(score) or index in seen or not 0 <= index < len(documents):
                    raise ValueError("invalid result")
            except (TypeError, ValueError, OverflowError):
                invalid = True
                continue
            seen.add(index)
            results.append(RerankResult(index=index, score=max(0.0, min(1.0, score))))
        results.sort(key=lambda item: item.score, reverse=True)
        expected = min(len(documents), max(1, int(top_n))) if top_n is not None else len(documents)
        complete = not invalid and len(results) >= expected
        publish({
            "last_status": "ok" if complete else "partial" if results else "invalid_response",
            "last_error_type": "" if complete else "InvalidRerankResponse",
            "last_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
            "last_result_count": len(results),
        })
        return results


def _bool_value(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_between(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))


def _float_between(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))
