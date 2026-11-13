"""Execute the current comment handler with real temporary Authority/projection.

Extract only the handler and its two helpers to avoid server module startup and
real configuration. This proves the Python receiver, not MCP wire registration.
"""
import ast
import asyncio
from pathlib import Path
import re
import secrets

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from test_memory_legacy_projection import setup


def receiver(tmp_path):
    config = {'buckets_dir': str(tmp_path / 'buckets'), 'state_dir': str(tmp_path / 'state')}
    authority, buckets = MemoryAuthorityStore(config), BucketManager(config)
    service = MemoryCommitService(authority, BucketMemoryProjection(config, buckets))
    source = Path(__file__).resolve().parents[1] / 'server.py'
    names = {'_memory_operation_key', '_append_memory_ring', 'comment_bucket'}
    nodes = [node for node in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    for node in nodes:
        node.decorator_list = []
    scope = {'secrets': secrets, 'memory_commit_service': service,
             'memory_authority_store': authority, 'bucket_mgr': buckets,
             '_coerce_memory_id': lambda value: value, 'MEMORY_ID_RE': re.compile(r'[a-z0-9-]+'),
             '_ai_author_name': lambda: 'synthetic-agent',
             '_bucket_read_payload': lambda bucket: bucket,
             '_queue_embedding_refresh': lambda value: False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope['comment_bucket'], authority, buckets


def test_same_operation_comment_concurrency_and_reopen_keep_one_ring(tmp_path):
    setup(tmp_path)
    first, _, _ = receiver(tmp_path)
    second, _, _ = receiver(tmp_path)
    kwargs = {'bucket_id': 'old-memory', 'content': 'Successful result: Session not found',
              'idempotency_key': 'synthetic-operation-1'}

    async def concurrent():
        return await asyncio.gather(first(**kwargs), second(**kwargs))

    responses = asyncio.run(concurrent())
    assert all(row['status'] == 'commented' for row in responses)
    assert responses[0]['comment']['id'] == responses[1]['comment']['id']
    reopened, authority, buckets = receiver(tmp_path)
    recovered = asyncio.run(reopened(**kwargs))
    assert recovered['comment']['id'] == responses[0]['comment']['id']
    bucket = asyncio.run(buckets.get('old-memory'))
    matching = [row for row in bucket['metadata']['comments'] if row['content'] == kwargs['content']]
    assert len(matching) == 1
    assert authority.get_memory('old-memory')['active_revision'] == 1
