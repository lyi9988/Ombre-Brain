"""Run Brain's background worker inside the MCP application's ASGI lifespan."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging


logger = logging.getLogger(__name__)


def attach_runtime_lifespan(app, *, worker, on_startup, on_shutdown):
    # FastMCP owns a lifespan for its session manager. Preserve that context
    # (including its yielded state); deprecated startup handlers bypass it.
    mcp_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def runtime_lifespan(application):
        async with mcp_lifespan(application) as state:
            await on_startup()
            task = asyncio.create_task(worker(), name="chat-memory-preparation")
            logger.info("Chat Memory preparation worker started")
            try:
                yield state
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                finally:
                    await on_shutdown()
                    logger.info("Chat Memory preparation worker stopped")

    app.router.lifespan_context = runtime_lifespan
    return app
