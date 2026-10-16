import asyncio
from contextlib import asynccontextmanager
from functools import wraps

import pytest
from mcp.server.fastmcp import FastMCP

from chat_memory_tool import ChatMemoryTool, candidate_id
from http_runtime_lifespan import attach_runtime_lifespan
from memory_chat_preparation import ChatMemoryPreparation, prepare_chat_tasks
from test_chat_memory_preparation import setup as setup_chat_memory


TRANSPORTS = ["streamable-http"]
if callable(getattr(FastMCP, "sse_app", None)):
    TRANSPORTS.append("sse")


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def exception_tree(error):
    yield error
    for nested in getattr(error, "exceptions", ()):
        yield from exception_tree(nested)


class LifespanDriver:
    """Drive a real Starlette/FastMCP ASGI app's lifespan protocol."""

    def __init__(self, app):
        self.app = app
        self.scope = {
            "type": "lifespan",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "state": {},
        }
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(
            self.app(self.scope, self.incoming.get, self.outgoing.put)
        )
        await self.incoming.put({"type": "lifespan.startup"})
        return await asyncio.wait_for(self.outgoing.get(), timeout=3)

    async def stop(self):
        await self.incoming.put({"type": "lifespan.shutdown"})
        message = await asyncio.wait_for(self.outgoing.get(), timeout=3)
        error = None
        try:
            await asyncio.wait_for(self.task, timeout=3)
        except BaseException as exc:
            error = exc
        return message, error


def make_app(transport, events, worker, on_startup, on_shutdown):
    server = FastMCP("lifespan-test", host="127.0.0.1", port=0, stateless_http=True)
    app = server.streamable_http_app() if transport == "streamable-http" else server.sse_app()

    # Keep FastMCP's real lifespan active and add an observable state value to
    # verify attach_runtime_lifespan passes the existing context through.
    fastmcp_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def observed_fastmcp_lifespan(application):
        events.append("fastmcp_enter")
        async with fastmcp_lifespan(application):
            events.append("fastmcp_ready")
            try:
                yield {"fastmcp_state": "preserved"}
            finally:
                events.append("fastmcp_exit")

    app.router.lifespan_context = observed_fastmcp_lifespan
    return attach_runtime_lifespan(
        app,
        worker=worker,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
    )


@pytest.mark.parametrize("transport", TRANSPORTS)
@async_test
async def test_fastmcp_http_lifespan_starts_and_cancels_worker_in_order(transport):
    events = []
    worker_started = asyncio.Event()
    worker_cancelled = asyncio.Event()

    async def worker():
        events.append("worker_started")
        worker_started.set()
        try:
            await asyncio.Future()
        finally:
            events.append("worker_cancelled")
            worker_cancelled.set()

    async def on_startup():
        events.append("on_startup")

    async def on_shutdown():
        events.append("on_shutdown")

    driver = LifespanDriver(make_app(transport, events, worker, on_startup, on_shutdown))
    assert (await driver.start())["type"] == "lifespan.startup.complete"
    await asyncio.wait_for(worker_started.wait(), timeout=3)
    assert driver.scope["state"]["fastmcp_state"] == "preserved"

    response, error = await driver.stop()
    assert response["type"] == "lifespan.shutdown.complete"
    assert error is None
    assert worker_cancelled.is_set()
    assert events == [
        "fastmcp_enter",
        "fastmcp_ready",
        "on_startup",
        "worker_started",
        "worker_cancelled",
        "on_shutdown",
        "fastmcp_exit",
    ]


@pytest.mark.parametrize("transport", TRANSPORTS)
@async_test
async def test_startup_failure_exits_fastmcp_context_without_starting_worker(transport):
    events = []

    async def worker():
        events.append("worker_started")

    async def on_startup():
        events.append("on_startup")
        raise RuntimeError("startup failed")

    async def on_shutdown():
        events.append("on_shutdown")

    driver = LifespanDriver(make_app(transport, events, worker, on_startup, on_shutdown))
    response = await driver.start()
    assert response["type"] == "lifespan.startup.failed"
    with pytest.raises(Exception) as raised:
        await driver.task
    assert any(
        isinstance(error, RuntimeError) and str(error) == "startup failed"
        for error in exception_tree(raised.value)
    )
    assert events == ["fastmcp_enter", "fastmcp_ready", "on_startup", "fastmcp_exit"]


@pytest.mark.parametrize("transport", TRANSPORTS)
@async_test
async def test_worker_error_still_runs_shutdown_and_exits_fastmcp_context(transport):
    events = []
    worker_started = asyncio.Event()

    async def worker():
        events.append("worker_started")
        worker_started.set()
        raise RuntimeError("worker failed")

    async def on_startup():
        events.append("on_startup")

    async def on_shutdown():
        events.append("on_shutdown")

    driver = LifespanDriver(make_app(transport, events, worker, on_startup, on_shutdown))
    assert (await driver.start())["type"] == "lifespan.startup.complete"
    await asyncio.wait_for(worker_started.wait(), timeout=3)
    await asyncio.sleep(0)

    response, error = await driver.stop()
    assert response["type"] == "lifespan.shutdown.failed"
    assert any(
        isinstance(nested, RuntimeError) and str(nested) == "worker failed"
        for nested in exception_tree(error)
    )
    assert events == [
        "fastmcp_enter",
        "fastmcp_ready",
        "on_startup",
        "worker_started",
        "on_shutdown",
        "fastmcp_exit",
    ]


@pytest.mark.parametrize("transport", TRANSPORTS)
@async_test
async def test_lifespan_worker_prepares_queued_submission_into_review_candidate(tmp_path, transport):
    env, engine, envelope, calls = setup_chat_memory(tmp_path)
    submitter = ChatMemoryTool(engine, env["buckets"])
    accepted = await submitter.submit(envelope)
    memory_id = candidate_id(envelope["operation_id"])
    tasks = ChatMemoryPreparation(env["authority"])
    assert accepted["memory_status"] == "preparing"
    assert tasks.get(memory_id)["phase"] == "queued"
    assert calls == []

    events = []
    worker_started = asyncio.Event()
    task_prepared = asyncio.Event()
    worker_cancelled = asyncio.Event()

    async def worker():
        events.append("worker_started")
        worker_started.set()
        try:
            while True:
                result = await prepare_chat_tasks(engine, env["buckets"])
                if result.get("results"):
                    task_prepared.set()
                await asyncio.sleep(0.01)
        finally:
            events.append("worker_cancelled")
            worker_cancelled.set()

    async def on_startup():
        events.append("on_startup")

    async def on_shutdown():
        events.append("on_shutdown")

    driver = LifespanDriver(make_app(transport, events, worker, on_startup, on_shutdown))
    assert (await driver.start())["type"] == "lifespan.startup.complete"
    await asyncio.wait_for(worker_started.wait(), timeout=3)
    await asyncio.wait_for(task_prepared.wait(), timeout=8)

    candidate = env["authority"].get_candidate(memory_id)
    assert candidate["status"] == "pending"
    assert candidate["proposal"]["proposed_body"] == envelope["arguments"]["narrative"]["body"]
    assert candidate["proposal"]["metadata"]["memory_l1"]["generation"]["actor"] == "chat_memory_preparation"
    assert len(calls) == 1
    assert tasks.get(memory_id)["phase"] == "prepared"
    assert tasks.receipt(memory_id)["memory_status"] == "pending"
    assert env["authority"].list_memories() == []

    response, error = await driver.stop()
    assert response["type"] == "lifespan.shutdown.complete"
    assert error is None
    assert worker_cancelled.is_set()
