"""Validated first-pass diary checkpoints, not adopted memory or owner proposals.

Writes use the existing Authority database and optimistic CAS. No model calls,
expiry-based retries, candidate transitions or new scheduling are performed.
"""
import json
import re
from memory_authority import RevisionConflict


def _provider_issue(body):
    """Parse common nested gateway shapes. Only fixed technical phrases leave this function."""
    pending = [body]
    fragments = []
    while pending and len(fragments) < 12:
        node = pending.pop()
        if isinstance(node, dict):
            pending.extend(node[key] for key in ('error', 'detail', 'message', 'code', 'type') if key in node)
        elif isinstance(node, str):
            value = node[:1000].strip()
            fragments.append(value.casefold())
            if value.startswith('{'):
                try:
                    pending.append(json.loads(value))
                except ValueError:
                    pass
    words = ' '.join(fragments)
    for category, markers in (
        ('content_filter', ('content exists risk', 'content_filter', 'content filter', 'content policy',
                            'inappropriate content', 'data_inspection_failed', 'sensitive content',
                            '内容安全', '敏感内容', '安全策略', '风险内容', '内容审核')),
        ('context_length', ('context length', 'context_length_exceeded', 'maximum context', '上下文长度')),
        ('response_format', ('response_format', 'json schema', 'json_schema', '结构化输出')),
        ('provider_route', ('no deployments', 'no available deployment', 'model not found',
                            'no available channel', '无可用渠道', '无可用模型')),
        ('rate_limit', ('rate limit', 'rate_limit', 'too many requests', '请求频率')),
        ('upstream_capacity', ('overloaded', 'server busy', 'temporarily unavailable', '服务繁忙')),
        ('invalid_parameter', ('unsupported parameter', 'invalid parameter', 'max_tokens', '参数错误'))):
        for marker in markers:
            if marker in words:
                return category, marker
    return None, None


def window_failure(exc, index):
    """Keep provider correlation and the failing field without storing the diary or provider body."""
    def identifier(value):
        return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:\[\]-]{1,160}', value) else None
    failure = {'window': index, 'reason': type(exc).__name__, 'code': identifier(getattr(exc, 'code', None))}
    response = getattr(exc, 'response', None)
    status = getattr(exc, 'status_code', None)
    if type(status) is int:
        failure['http_status'] = status
    headers = getattr(response, 'headers', {})
    body = getattr(exc, 'body', None)
    upstream = body.get('error', body) if isinstance(body, dict) else {}
    if isinstance(upstream, dict) and identifier(upstream.get('type')):
        failure['provider_category'] = upstream['type']
    if callable(getattr(response, 'json', None)) and type(status) is int and 400 <= status <= 599:
        try:
            body = response.json()
        except (ValueError, RuntimeError):
            pass
    category, matched = _provider_issue(body)
    if category:
        failure.update(provider_category=category, provider_reason_marker=matched)
    for field, value in (('field_path', getattr(exc, 'field_path', None)),
                         ('provider_param', getattr(exc, 'param', None) or
                          (upstream.get('param') if isinstance(upstream, dict) else None)),
                         ('provider_trace_id', headers.get('x-trace-id')),
                         ('upstream_request_id', headers.get('x-upstream-request-id'))):
        if identifier(value):
            failure[field] = value
    return failure


def rejected_request(failure):
    return failure.get('http_status') in (400, 401, 403, 404, 413, 422, 429) or failure.get('reason') == 'BadRequestError'


class DiaryProgress:
    def __init__(self, authority):
        self.authority = authority
        db = authority._connect()
        try:
            db.execute('CREATE TABLE IF NOT EXISTS memory_diary_task_progress ('
                       'task_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, progress_json TEXT NOT NULL)')
            db.commit()
        finally:
            db.close()

    def get(self, task_id):
        db = self.authority._connect()
        try:
            row = db.execute('SELECT * FROM memory_diary_task_progress WHERE task_id=?', (task_id,)).fetchone()
            return {'revision': row['revision'], 'progress': json.loads(row['progress_json'])} if row else None
        finally:
            db.close()

    def original_first_failure(self, day, task_id, record):
        # Old first-dispatch checkpoints predate stored HTTP outcomes. Revision
        # one and window zero identify that single dispatch. Never borrow an
        # older failure for a later attempt of the same window.
        if record['revision'] != 1 or record['progress'].get('uncertain_window') != 0:
            return {}
        for projection in self.authority.list_projection_status('diary_task_' + day, revision=0):
            detail = projection['details']
            if (projection['projector'] == 'memory_l1_first_pass' and projection['status'] == 'degraded'
                    and detail.get('task_id') == task_id):
                failures = [f for f in detail.get('failures', []) if f.get('window') == 0]
                if len(failures) == 1:
                    return failures[0]
        return {}

    def save(self, task_id, progress, *, expected_revision):
        db = self.authority._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM memory_diary_task_progress WHERE task_id=?', (task_id,)).fetchone()
            current = row['revision'] if row else 0
            if current != expected_revision:
                raise RevisionConflict('diary progress changed during generation')
            next_revision = current + 1
            payload = json.dumps(progress, ensure_ascii=False, sort_keys=True)
            db.execute('INSERT INTO memory_diary_task_progress VALUES(?,?,?) '
                       'ON CONFLICT(task_id) DO UPDATE SET revision=excluded.revision, progress_json=excluded.progress_json',
                       (task_id, next_revision, payload))
            db.commit()
            return {'revision': next_revision, 'progress': progress}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
