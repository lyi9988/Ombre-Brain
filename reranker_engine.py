from __future__ import annotations

import asyncio
import logging
import threading
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
            "transport": "persistent_httpx",
            "connection_fallback_count": 0,
        }
        self._client_lock = threading.RLock()
        self._client_signature = None
        self.client = None

    def runtime_debug(self) -> dict:
        """Return owner-safe rerank health without credentials or documents."""
        return {
            "enabled": bool(self.enabled),
            "configured": bool(self.api_key and self.base_url),
            "model": self.model,
            "base_url": self.base_url,
            "client_open": bool(
                self.client is not None
                and not getattr(self.client, "is_closed", False)
            ),
            "timeout_seconds": self.timeout,
            "candidate_limit": self.candidate_limit,
            **dict(self._runtime),
        }

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        if not self.enabled or not query or not documents:
            self._runtime.update({
                "last_status": "disabled" if not self.enabled else "empty_input",
                "last_error_type": "",
                "last_http_status": None,
                "last_latency_ms": 0,
                "last_result_count": 0,
            })
            return []
        endpoint = f"{self.base_url}/rerank"
        payload: dict[str, Any] = {
            "model": self.model,
            "query": str(query),
            "documents": [str(document or "") for document in documents],
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = max(1, min(int(top_n), len(documents)))

        started = time.perf_counter()
        self._runtime.update({
            "last_status": "started",
            "last_error_type": "",
            "last_http_status": None,
            "last_latency_ms": None,
            "last_result_count": None,
        })
        try:
            response = await self._post(endpoint, payload)
            self._runtime["last_http_status"] = response.status_code
            response.raise_for_status()
            body = response.json()
        except asyncio.CancelledError:
            self._runtime.update({
                "last_status": "cancelled",
                "last_error_type": "CancelledError",
                "last_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "last_result_count": 0,
            })
            raise
        except Exception as exc:
            self._runtime.update({
                "last_status": "error",
                "last_error_type": type(exc).__name__,
                "last_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "last_result_count": 0,
            })
            logger.warning("Reranker request failed: %s", exc)
            return []

        results = []
        for item in body.get("results", []) if isinstance(body, dict) else []:
            try:
                index = int(item.get("index"))
                score = float(item.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(documents):
                results.append(RerankResult(index=index, score=max(0.0, min(1.0, score))))
        results.sort(key=lambda item: item.score, reverse=True)
        self._runtime.update({
            "last_status": "ok",
            "last_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
            "last_result_count": len(results),
        })
        return results

    def _client_signature_value(self):
        return (bool(self.enabled), str(self.base_url or "").rstrip("/"),
                str(self.api_key or ""))

    def _new_client(self):
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.timeout,
                connect=min(8.0, self.timeout),
                write=min(15.0, self.timeout),
                pool=min(8.0, self.timeout),
                read=self.timeout,
            ),
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=4,
                keepalive_expiry=30.0,
            ),
        )

    async def _ensure_client(self):
        signature = self._client_signature_value()
        old = None
        with self._client_lock:
            if (
                self.client is not None
                and not getattr(self.client, "is_closed", False)
                and self._client_signature == signature
            ):
                return self.client
            old = self.client
            self.client = self._new_client() if signature[0] else None
            self._client_signature = signature
            client = self.client
        if old is not None and not getattr(old, "is_closed", False):
            try:
                await old.aclose()
            except Exception:
                logger.debug("Reranker previous HTTP client close failed", exc_info=True)
        return client

    async def _post(self, endpoint: str, payload: dict[str, Any]):
        client = await self._ensure_client()
        if client is None:
            raise RuntimeError("reranker client unavailable")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            return await client.post(endpoint, headers=headers, json=payload,
                                     timeout=self.timeout)
        except httpx.PoolTimeout:
            self._runtime["connection_fallback_count"] = int(
                self._runtime.get("connection_fallback_count") or 0
            ) + 1
            async with httpx.AsyncClient(timeout=self.timeout) as fallback:
                return await fallback.post(endpoint, headers=headers, json=payload)

    async def reconfigure(self, config: dict) -> None:
        config = config or {}
        embed_cfg = config.get("embedding", {}) or {}
        rerank_cfg = config.get("reranker", {}) or {}
        dehy_cfg = config.get("dehydration", {}) or {}
        self.model = str(rerank_cfg.get("model") or self.model)
        self.base_url = str(
            rerank_cfg.get("base_url")
            or embed_cfg.get("base_url")
            or dehy_cfg.get("base_url")
            or self.base_url
        ).rstrip("/")
        self.api_key = str(
            rerank_cfg.get("api_key")
            or embed_cfg.get("api_key")
            or dehy_cfg.get("api_key")
            or self.api_key
        )
        self.enabled = bool(self.api_key and self.base_url) and _bool_value(
            rerank_cfg.get("enabled", True)
        )
        self.timeout = _float_between(
            rerank_cfg.get("timeout_seconds", self.timeout), self.timeout, 1, 120
        )
        self.candidate_limit = _int_between(
            rerank_cfg.get("candidate_limit", self.candidate_limit),
            self.candidate_limit, 1, 100,
        )
        self.score_weight = _float_between(
            rerank_cfg.get("score_weight", self.score_weight), self.score_weight, 0.0, 1.0
        )
        await self._ensure_client()

    async def close(self) -> None:
        with self._client_lock:
            client = self.client
            self.client = None
            self._client_signature = None
        if client is not None and not getattr(client, "is_closed", False):
            await client.aclose()


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
