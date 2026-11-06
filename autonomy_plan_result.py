"""Private handoff of one original, non-streaming planning answer to its originating Aiz cycle."""
import hashlib
import json
import re
from contextlib import closing

from starlette.responses import JSONResponse


HEADER = 'X-Guyan-Plan-Receipt'
HEX = re.compile(r'[0-9a-f]{64}')


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def initialize(db):
    db.execute('''CREATE TABLE IF NOT EXISTS autonomy_plan_results (
        ticket_hash TEXT PRIMARY KEY, cycle_id TEXT NOT NULL, input_sha256 TEXT NOT NULL,
        output_text TEXT, output_sha256 TEXT, acknowledged INTEGER NOT NULL DEFAULT 0)''')


def identity(request, payload):
    token = request.headers[HEADER]
    cycle = request.headers.get('X-Guyan-Request-Id', '')
    trace = request.headers.get('X-Guyan-Trace-Id')
    if (request.headers.get('X-Guyan-Runtime-Scope') != 'internal:autonomy_chain'
            or not HEX.fullmatch(token) or not cycle or len(cycle) > 200
            or trace != 'trace-autonomy-' + _hash(cycle)[:32]
            or not isinstance(payload, dict) or payload.get('stream') or not isinstance(payload.get('messages'), list)):
        raise ValueError('invalid_planning_receipt_request')
    wire = json.dumps(payload['messages'], ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return (_hash(token), cycle, _hash(wire))


def _read(db, key):
    row = db.execute('SELECT * FROM autonomy_plan_results WHERE ticket_hash=?', key[:1]).fetchone()
    if row and (row['cycle_id'], row['input_sha256']) != key[1:]:
        raise ValueError('planning_receipt_identity_conflict')
    return row


def reserve(store, key):
    with closing(store._connect()) as db, db:
        if db.execute('INSERT OR IGNORE INTO autonomy_plan_results(ticket_hash,cycle_id,input_sha256) '
                      'VALUES(?,?,?)', key).rowcount:
            return None
        row = _read(db, key)
        if row['output_text'] is not None:
            return JSONResponse({'choices': [{'message': {'role': 'assistant', 'content': row['output_text']}}]})
        return JSONResponse({'error': {'code': 'planning_receipt_already_dispatched'}}, status_code=409)


def record(store, key, response):
    with closing(store._connect()) as db, db:
        output = None
        if 200 <= response.status_code < 300:
            try:
                body = json.loads(response.body)
                choices = body.get('choices') or []
                output = choices[0]['message'].get('content') if choices else None
            except (ValueError, AttributeError, TypeError, KeyError, IndexError):
                pass
        if not isinstance(output, str) or not output.strip():
            # A rejected or empty answer leaves the existing candidate fallback available.
            db.execute('DELETE FROM autonomy_plan_results WHERE ticket_hash=? AND output_sha256 IS NULL', key[:1])
            return
        # Only visible answer text is transferable. The worker never consumes reasoning as a plan.
        db.execute('UPDATE autonomy_plan_results SET output_text=?,output_sha256=? '
                   'WHERE ticket_hash=? AND output_sha256 IS NULL', (output, _hash(output), key[0]))


def lookup(store, query):
    expected = {'ticket', 'cycle_id', 'input_sha256'}
    if (not isinstance(query, dict) or not expected <= query.keys()
            or set(query) - (expected | {'acknowledge'})
            or any(not isinstance(query[k], str) for k in expected)
            or not HEX.fullmatch(query['ticket']) or not HEX.fullmatch(query['input_sha256'])
            or not query['cycle_id'] or len(query['cycle_id']) > 200):
        raise ValueError('invalid_planning_receipt_lookup')
    key = (_hash(query['ticket']), query['cycle_id'], query['input_sha256'])
    with closing(store._connect()) as db, db:
        row = _read(db, key)
        if row is None or row['output_sha256'] is None:
            return {'state': 'pending' if row else 'not_found'}
        if 'acknowledge' in query:
            if query['acknowledge'] != row['output_sha256']:
                raise ValueError('planning_receipt_output_conflict')
            db.execute('UPDATE autonomy_plan_results SET acknowledged=1,output_text=NULL WHERE ticket_hash=?', key[:1])
            return {'state': 'acknowledged'}
        if row['acknowledged']:
            return {'state': 'acknowledged'}
        return {'state': 'available', 'cycle_id': row['cycle_id'], 'input_sha256': row['input_sha256'],
                'output': row['output_text'], 'output_sha256': row['output_sha256']}
