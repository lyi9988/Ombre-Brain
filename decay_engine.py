# ============================================================
# Module: Memory Decay Engine (decay_engine.py)
# 模块：记忆衰减引擎
#
# Simulates human forgetting curve; auto-decays inactive memories and archives them.
# 模拟人类遗忘曲线，自动衰减不活跃记忆并归档。
#
# Core formula (improved Ebbinghaus + emotion coordinates):
# 核心公式（改进版艾宾浩斯遗忘曲线 + 情感坐标）：
#   Score = Importance × (activation_count^0.3) × e^(-λ×days) × emotion_weight
#
# Emotion weight (continuous coordinate, not discrete labels):
# 情感权重（基于连续坐标而非离散列举）：
#   emotion_weight = base + (arousal × arousal_boost)
#   Higher arousal → higher emotion weight → slower decay
#   唤醒度越高 → 情感权重越大 → 记忆衰减越慢
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================

import math
import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger("ombre_brain.decay")


class DecayEngine:
    """
    Memory decay engine — periodically scans all dynamic buckets,
    calculates decay scores, auto-archives low-activity buckets
    to simulate natural forgetting.
    记忆衰减引擎 —— 定期扫描所有动态桶，
    计算衰减得分，将低活跃桶自动归档，模拟自然遗忘。
    """

    def __init__(self, config: dict, bucket_mgr, memory_state_service=None):
        # --- Load decay parameters / 加载衰减参数 ---
        decay_cfg = config.get("decay", {})
        self.decay_lambda = decay_cfg.get("lambda", 0.05)
        self.threshold = decay_cfg.get("threshold", 0.3)
        self.check_interval = decay_cfg.get("check_interval_hours", 24)

        # --- Emotion weight params (continuous arousal coordinate) ---
        # --- 情感权重参数（基于连续 arousal 坐标）---
        emotion_cfg = decay_cfg.get("emotion_weights", {})
        self.emotion_base = emotion_cfg.get("base", 1.0)
        self.arousal_boost = emotion_cfg.get("arousal_boost", 0.8)

        self.bucket_mgr = bucket_mgr
        self.memory_state_service = memory_state_service
        authority_cfg = config.get("memory_authority", {})
        if not isinstance(authority_cfg, dict):
            authority_cfg = {}
        self.memory_authority_enabled = bool(authority_cfg.get("enabled", False))

        # --- Background task control / 后台任务控制 ---
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the decay engine is running in the background.
        衰减引擎是否正在后台运行。"""
        return self._running

    # ---------------------------------------------------------
    # Core: calculate decay score for a single bucket
    # 核心：计算单个桶的衰减得分
    #
    # Higher score = more vivid memory; below threshold → archive
    # 得分越高 = 记忆越鲜活，低于阈值则归档
    # Permanent buckets never decay / 固化桶永远不衰减
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Freshness bonus: continuous exponential decay
    # 新鲜度加成：连续指数衰减
    # bonus = 1.0 + 1.0 × e^(-t/36), t in hours
    # t=0 → 2.0×, t≈25h(半衰) → 1.5×, t≈72h → ≈1.14×, t→∞ → 1.0×
    # ---------------------------------------------------------
    @staticmethod
    def _calc_time_weight(days_since: float) -> float:
        """
        Freshness bonus multiplier: 1.0 + e^(-t/36), t in hours.
        新鲜度加成乘数：刚存入×2.0，~36小时半衰，72小时后趋近×1.0。
        """
        hours = days_since * 24.0
        return 1.0 + 1.0 * math.exp(-hours / 36.0)

    @staticmethod
    def _parse_datetime(value) -> datetime:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _now_naive_utc() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def calculate_score(self, metadata: dict) -> float:
        """
        Calculate current activity score for a memory bucket.
        计算一个记忆桶的当前活跃度得分。

        New model: short-term vs long-term weight separation.
        新模型：短期/长期权重分离。
        - Short-term (≤3 days): time_weight dominates, emotion amplifies
        - Long-term (>3 days): emotion_weight dominates, time decays to floor
        短期（≤3天）：时间权重主导，情感放大
        长期（>3天）：情感权重主导，时间衰减到底线
        """
        if not isinstance(metadata, dict):
            return 0.0

        # --- Pinned/protected buckets: never decay, importance locked to 10 ---
        if metadata.get("pinned") or metadata.get("protected"):
            return 999.0

        # --- Permanent buckets never decay ---
        if metadata.get("type") == "permanent":
            return 999.0

        # --- Feel buckets: never decay, fixed moderate score ---
        if metadata.get("type") == "feel":
            return 50.0

        importance = max(1, min(10, int(metadata.get("importance", 5))))
        activation_count = max(1, int(metadata.get("activation_count", 1)))

        # --- Days since last activation ---
        last_active_str = metadata.get("last_active", metadata.get("created", ""))
        try:
            last_active = self._parse_datetime(last_active_str)
            days_since = max(0.0, (self._now_naive_utc() - last_active).total_seconds() / 86400)
        except (ValueError, TypeError):
            days_since = 30

        # --- Emotion weight ---
        try:
            arousal = max(0.0, min(1.0, float(metadata.get("arousal", 0.3))))
        except (ValueError, TypeError):
            arousal = 0.3
        emotion_weight = self.emotion_base + arousal * self.arousal_boost

        # --- Time weight ---
        time_weight = self._calc_time_weight(days_since)

        # --- Short-term vs Long-term weight separation ---
        # 短期（≤3天）：time_weight 占 70%，emotion 占 30%
        # 长期（>3天）：emotion 占 70%，time_weight 占 30%
        if days_since <= 3.0:
            # Short-term: time dominates, emotion amplifies
            combined_weight = time_weight * 0.7 + emotion_weight * 0.3
        else:
            # Long-term: emotion dominates, time provides baseline
            combined_weight = emotion_weight * 0.7 + time_weight * 0.3

        # --- Base score ---
        base_score = (
            importance
            * (activation_count ** 0.3)
            * math.exp(-self.decay_lambda * days_since)
            * combined_weight
        )

        # --- Weight pool modifiers ---
        # resolved + digested (has feel) → accelerated fade: ×0.02
        # resolved only → ×0.05
        # 已处理+已消化（写过feel）→ 加速淡化：×0.02
        # 仅已处理 → ×0.05
        resolved = metadata.get("resolved", False)
        digested = metadata.get("digested", False)  # set when feel is written for this memory
        if resolved and digested:
            resolved_factor = 0.02
        elif resolved:
            resolved_factor = 0.05
        else:
            resolved_factor = 1.0
        urgency_boost = 1.5 if (arousal > 0.7 and not resolved) else 1.0

        return round(base_score * resolved_factor * urgency_boost, 4)

    def _authority_memories_by_bucket(self) -> dict[str, dict]:
        """Read managed rows once, keyed by their physical bucket id."""
        service = self.memory_state_service
        if service is None:
            if self.memory_authority_enabled:
                raise RuntimeError("memory_state_service_unavailable")
            return {}
        authority = getattr(service, "authority", None)
        if authority is None:
            raise RuntimeError("memory_authority_unavailable")
        list_memories = getattr(authority, "list_memories", None)
        if not callable(list_memories):
            raise RuntimeError("memory_authority_lookup_unavailable")

        output: dict[str, dict] = {}
        limit = 1000
        offset = 0
        while True:
            rows = list_memories(state="all", limit=limit, offset=offset)
            if not isinstance(rows, list):
                raise RuntimeError("memory_authority_lookup_invalid")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                bucket_id = str(row.get("bucket_id") or "").strip()
                memory_id = str(row.get("memory_id") or "").strip()
                if not bucket_id or not memory_id:
                    continue
                prior = output.get(bucket_id)
                if prior is not None and str(prior.get("memory_id")) != memory_id:
                    raise RuntimeError("memory_authority_bucket_ambiguous")
                output[bucket_id] = row
            if len(rows) < limit:
                return output
            offset += len(rows)

    async def _archive_for_decay(
        self, bucket_id: str, authority_memories: dict[str, dict] | None
    ) -> tuple[bool, str]:
        """Archive managed buckets through authority and preserve legacy fallback."""
        if self.memory_state_service is None and not self.memory_authority_enabled:
            return bool(await self.bucket_mgr.archive(bucket_id)), "legacy"
        if authority_memories is None:
            return False, "authority_unavailable"

        listed = authority_memories.get(bucket_id)
        if listed is None:
            # A successful authority lookup proves this bucket is legacy.
            return bool(await self.bucket_mgr.archive(bucket_id)), "legacy"

        memory_id = str(listed.get("memory_id") or "").strip()
        authority = getattr(self.memory_state_service, "authority", None)
        get_snapshot = getattr(authority, "get_memory_state_snapshot", None)
        if not memory_id or not callable(get_snapshot):
            return False, "authority_unavailable"
        memory, generation = get_snapshot(memory_id)
        if not memory or str(memory.get("bucket_id") or "") != bucket_id:
            return False, "authority_unavailable"

        state = str(memory.get("state") or "").strip().lower()
        if state == "archived":
            return False, "already_archived"
        if state != "active":
            return False, "not_active"

        revision = int(memory.get("active_revision") or 0)
        recall_policy = str(memory.get("recall_policy") or "enabled")
        idempotency_key = (
            f"decay-archive:{memory_id}:r{revision}:g{generation}"
        )
        try:
            await self.memory_state_service.change_memory_state(
                memory_id=memory_id,
                state="archived",
                recall_policy=recall_policy,
                idempotency_key=idempotency_key,
                actor="decay",
                expected_revision=revision,
                expected_state="active",
                expected_state_generation=generation,
            )
            return True, "authority"
        except Exception as exc:
            # Keep failure logs free of memory names and body text.
            from memory_authority import RevisionConflict

            if isinstance(exc, RevisionConflict):
                logger.info("Decay archive skipped: authority CAS conflict")
                return False, "cas_conflict"
            logger.warning(
                "Decay archive blocked: authority transition failed (%s)",
                type(exc).__name__,
            )
            return False, "authority_failed"

    # ---------------------------------------------------------
    # Execute one decay cycle
    # 执行一轮衰减周期
    # Scan all dynamic buckets → score → archive those below threshold
    # 扫描所有动态桶 → 算分 → 低于阈值的归档
    # ---------------------------------------------------------
    async def run_decay_cycle(self) -> dict:
        """
        Execute one decay cycle: iterate dynamic buckets, archive those
        scoring below threshold.
        执行一轮衰减：遍历动态桶，归档得分低于阈值的桶。

        Returns stats: {"checked": N, "archived": N, "lowest_score": X}
        """
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for decay / 衰减周期列桶失败: {e}")
            return {"checked": 0, "archived": 0, "lowest_score": 0, "error": str(e)}

        checked = 0
        archived = 0
        auto_resolved = 0
        lowest_score = float("inf")
        authority_memories: dict[str, dict] | None = {}
        if self.memory_state_service is not None or self.memory_authority_enabled:
            try:
                authority_memories = self._authority_memories_by_bucket()
            except Exception as exc:
                authority_memories = None
                logger.warning(
                    "Decay archive blocked: authority lookup failed (%s)",
                    type(exc).__name__,
                )

        for bucket in buckets:
            meta = bucket.get("metadata", {})

            # Skip permanent / pinned / protected / feel buckets
            # 跳过固化桶、钉选/保护桶和 feel 桶
            if meta.get("type") in ("permanent", "feel") or meta.get("pinned") or meta.get("protected"):
                continue

            checked += 1

            # --- Auto-resolve: imp≤4 + >30 days old + not resolved → auto resolve ---
            # --- 自动结案：重要度≤4 + 超过30天 + 未解决 → 自动 resolve ---
            if not meta.get("resolved", False):
                imp = int(meta.get("importance", 5))
                last_active_str = meta.get("last_active", meta.get("created", ""))
                try:
                    last_active = self._parse_datetime(last_active_str)
                    days_since = (self._now_naive_utc() - last_active).total_seconds() / 86400
                except (ValueError, TypeError):
                    days_since = 999
                if imp <= 4 and days_since > 30:
                    try:
                        await self.bucket_mgr.update(bucket["id"], resolved=True)
                        auto_resolved += 1
                        logger.info(
                            f"Auto-resolved / 自动结案: "
                            f"{meta.get('name', bucket['id'])} "
                            f"(imp={imp}, days={days_since:.0f})"
                        )
                    except Exception as e:
                        logger.warning(f"Auto-resolve failed / 自动结案失败: {e}")

            try:
                score = self.calculate_score(meta)
            except Exception as e:
                logger.warning(
                    f"Score calculation failed for {bucket.get('id', '?')} / "
                    f"计算得分失败: {e}"
                )
                continue

            lowest_score = min(lowest_score, score)

            # --- Below threshold → archive (simulate forgetting) ---
            # --- 低于阈值 → 归档（模拟遗忘）---
            if score < self.threshold:
                try:
                    bucket_id = str(bucket.get("id") or "")
                    success, archive_path = await self._archive_for_decay(
                        bucket_id, authority_memories
                    )
                    if success:
                        archived += 1
                        logger.info(
                            "Decay archived: bucket_id=%s path=%s score=%.4f threshold=%.4f",
                            bucket_id, archive_path, score, self.threshold,
                        )
                except Exception as e:
                    logger.warning("Decay archive failed (%s)", type(e).__name__)

        result = {
            "checked": checked,
            "archived": archived,
            "auto_resolved": auto_resolved,
            "lowest_score": lowest_score if checked > 0 else 0,
        }
        logger.info(f"Decay cycle complete / 衰减周期完成: {result}")
        return result

    # ---------------------------------------------------------
    # Background decay task management
    # 后台衰减任务管理
    # ---------------------------------------------------------
    async def ensure_started(self) -> None:
        """
        Ensure the decay engine is started (lazy init on first call).
        确保衰减引擎已启动（懒加载，首次调用时启动）。
        """
        if not self._running:
            await self.start()

    async def start(self) -> None:
        """Start the background decay loop.
        启动后台衰减循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info(
            f"Decay engine started, interval: {self.check_interval}h / "
            f"衰减引擎已启动，检查间隔: {self.check_interval} 小时"
        )

    async def stop(self) -> None:
        """Stop the background decay loop.
        停止后台衰减循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Decay engine stopped / 衰减引擎已停止")

    async def _background_loop(self) -> None:
        """Background loop: run decay → sleep → repeat.
        后台循环体：执行衰减 → 睡眠 → 重复。"""
        while self._running:
            try:
                await self.run_decay_cycle()
            except Exception as e:
                logger.error(f"Decay cycle error / 衰减周期出错: {e}")
            # --- Wait for next cycle / 等待下一个周期 ---
            try:
                await asyncio.sleep(self.check_interval * 3600)
            except asyncio.CancelledError:
                break
