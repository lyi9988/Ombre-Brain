"""Loop-owned Reflection clients, refreshed only between completed batches."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import wraps
import logging
import os
import threading

logger = logging.getLogger('ombre_brain.reflection')

_REFLECTION_CONNECTION_FIELDS = (
    'enabled', 'base_url', 'api_key', 'model', 'thinking_mode', 'temperature', 'max_tokens',
    'daily_chat_memory_api_key_env', 'daily_chat_memory_summary_api_key_env',
    'daily_chat_memory_api_key', 'daily_chat_memory_summary_api_key',
    'daily_chat_memory_base_url', 'daily_chat_memory_summary_base_url',
    'daily_chat_memory_summary_model', 'daily_chat_memory_candidate_model',
    'daily_chat_memory_timeout_seconds', 'daily_chat_memory_thinking_mode', 'memory_task_routes',
)
_CLIENTS = ('client', 'daily_chat_memory_client', 'dehydration_client',
            'daily_activity_summary_dehydration_client')


def capture_configuration(config):
    snapshot = deepcopy(config)
    reflection = snapshot.get('reflection') or {}
    names = {'OMBRE_REFLECTION_API_KEY', 'OMBRE_EMBEDDING_API_KEY',
             'OMBRE_PERSONA_API_KEY', 'OMBRE_API_KEY'}
    daily = (reflection.get('daily_chat_memory_api_key_env')
             or reflection.get('daily_chat_memory_summary_api_key_env'))
    if daily:
        names.add(str(daily).strip())
    credentials = {name: os.environ.get(name, '') for name in names}
    signature = {
        'reflection': {k: deepcopy(reflection.get(k)) for k in _REFLECTION_CONNECTION_FIELDS},
        'fallbacks': {section: {k: deepcopy((snapshot.get(section) or {}).get(k))
            for k in fields} for section, fields in (
                ('dehydration', ('base_url', 'api_key', 'model', 'thinking_mode')),
                ('embedding', ('base_url', 'api_key')),
                ('persona', ('base_url', 'api_key', 'model', 'thinking_mode')))},
        'credentials': credentials,
        # Rebuilding must also preserve the original policy and stores.
        'authority': {k: deepcopy((snapshot.get('memory_authority') or {}).get(k))
            for k in ('enabled', 'db_path', 'ingestion_policy')},
        'daily_mode': reflection.get('daily_chat_memory_mode'),
        'paths': {k: snapshot.get(k) for k in ('state_dir', 'buckets_dir', 'memory_authority_db_path')},
    }
    return snapshot, credentials, signature


class ReflectionConfigurationSource:
    """Snapshot only applied config; an owner save never exposes its midpoint."""
    def __init__(self):
        self._guard = threading.RLock()
        self._update_lock = asyncio.Lock()
        self._before = None

    def snapshot(self, config):
        with self._guard:
            return deepcopy(self._before) if self._before is not None else capture_configuration(config)

    def applied_update(self, config_getter):
        def decorate(function):
            @wraps(function)
            async def update(*args, **kwargs):
                # This serializes settings saves only, never model/chat work.
                async with self._update_lock:
                    with self._guard:
                        self._before = capture_configuration(config_getter())
                    try:
                        return await function(*args, **kwargs)
                    finally:
                        with self._guard:
                            self._before = None
            return update
        return decorate


class BackgroundReflectionRuntime:
    """One instance per sequential worker, never the global HTTP engine.

    Signatures contain credentials only in memory, with direct equality; they
    are never hashed, logged, persisted, or included in a diagnostic repr.
    """
    def __init__(self, prompt_plan_mirror=None, *, engine_factory=None, configuration_source=None):
        if engine_factory is None:
            from reflection_engine import ReflectionEngine
            engine_factory = ReflectionEngine
        self._factory = engine_factory
        self._configuration_source = configuration_source
        self._mirror = prompt_plan_mirror
        self._engine = None
        self._signature = None
        self._loop = None
        self._lock = asyncio.Lock()

    async def _close_engine(self, engine):
        seen = set()
        for name in _CLIENTS:
            client = getattr(engine, name, None)
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            try:
                await client.close()
            except Exception as exc:
                logger.warning('Background Reflection client close failed: %s', type(exc).__name__)

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError('background_reflection_runtime_loop_changed')

    @asynccontextmanager
    async def batch(self, config):
        self._check_loop()
        async with self._lock:
            snapshot, credentials, signature = (self._configuration_source.snapshot(config)
                if self._configuration_source is not None else capture_configuration(config))
            if self._engine is None or signature != self._signature:
                replacement = self._factory(snapshot, self._mirror, credential_environment=credentials)
                previous = self._engine
                if previous is not None and hasattr(previous, '_memory_preparation_cursor'):
                    replacement._memory_preparation_cursor = previous._memory_preparation_cursor
                self._engine, self._signature = replacement, signature
                if previous is not None:
                    await self._close_engine(previous)
            else:
                self._engine.config = snapshot
                self._engine.refresh_daily_chat_memory_options(snapshot)
            yield self._engine

    async def close(self):
        self._check_loop()
        async with self._lock:
            if self._engine is not None:
                await self._close_engine(self._engine)
                self._engine = None
                self._signature = None
