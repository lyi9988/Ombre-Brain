"""Revision-bound adopted relations and their rebuildable dependency projection.

Relations never grant source access or change another memory's stored facts.
Adoption receipts are Authority operations; edge/guard tables only locate the
material and restrictions the existing source/subject/time resolver must use.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from memory_l1_contract import canonical_json, semantic_sha256

RESTRICTING = {'supersedes', 'revises', 'limits', 'possible_conflict'}
SYNC_CLOSED = {'synchronized', 'superseded', 'withdrawn', 'dismissed', 'owner_confirmed_after_l1_revision'}


def pending_sync_scope(conn, memory_id):
    """Shared read-only L2 sync contract; no worker/module import on Gateway."""
    try:
        rows = conn.execute('SELECT presentation_revision,state,proposal_json FROM memory_l2_sync_proposals '
            'WHERE memory_id=? ORDER BY presentation_revision', (memory_id,)).fetchall()
    except sqlite3.OperationalError:
        return None
    open_rows = [(row, json.loads(row['proposal_json'])) for row in rows if row['state'] not in SYNC_CLOSED]
    if not open_rows:
        return None
    ids, unknown, generations = set(), False, []
    for row, proposal in open_rows:
        process = proposal.get('processing') or {'revision': 1, 'scope': {'statement_ids': [], 'unknown_scope': True}}
        scope = process.get('scope') or {'statement_ids': [], 'unknown_scope': True}
        ids.update(scope.get('statement_ids', []))
        unknown |= scope.get('unknown_scope') is not False
        generations.append({'presentation_revision': row['presentation_revision'], 'revision': process['revision'],
            'state': row['state'], 'scope': scope, 'target': process.get('material_identity')})
    return {'statement_ids': sorted(ids), 'unknown_scope': unknown,
        'proposal_revisions': [row['presentation_revision'] for row, _ in open_rows],
        'generation': semantic_sha256({'proposals': generations})}


def initialize(conn):
    for sql in (
        "CREATE INDEX IF NOT EXISTS memory_revision_parent_dependency ON memory_revisions(json_extract(metadata_json,'$.memory_l1.entry.parent_ref.memory_id'),memory_id,revision)",
        '''CREATE TABLE IF NOT EXISTS memory_relation_edges (
          host_memory_id TEXT NOT NULL,host_revision INTEGER NOT NULL,relation_id TEXT NOT NULL,
          relation_sha256 TEXT NOT NULL,kind TEXT NOT NULL,
          from_memory_id TEXT NOT NULL,from_revision INTEGER NOT NULL,from_type TEXT NOT NULL,from_id TEXT NOT NULL,
          to_memory_id TEXT NOT NULL,to_revision INTEGER NOT NULL,to_type TEXT NOT NULL,to_id TEXT NOT NULL,
          relation_json TEXT NOT NULL,PRIMARY KEY(host_memory_id,host_revision,relation_id))''',
        'CREATE INDEX IF NOT EXISTS memory_relation_from ON memory_relation_edges(from_memory_id,from_revision,from_type,from_id)',
        'CREATE INDEX IF NOT EXISTS memory_relation_to ON memory_relation_edges(to_memory_id,to_revision,to_type,to_id)',
        '''CREATE TABLE IF NOT EXISTS memory_relation_adoptions (
          host_memory_id TEXT NOT NULL,relation_id TEXT NOT NULL,relation_sha256 TEXT NOT NULL,
          operation_id TEXT NOT NULL,request_id TEXT NOT NULL,actor TEXT NOT NULL,adopted_at TEXT NOT NULL,
          PRIMARY KEY(host_memory_id,relation_id,relation_sha256))''',
        '''CREATE TABLE IF NOT EXISTS memory_relation_generations (
          memory_id TEXT PRIMARY KEY,generation INTEGER NOT NULL,operation_id TEXT NOT NULL)''',
        '''CREATE TABLE IF NOT EXISTS memory_relation_guards (
          guard_id TEXT PRIMARY KEY,host_memory_id TEXT NOT NULL,host_revision INTEGER NOT NULL,
          relation_id TEXT NOT NULL,relation_sha256 TEXT NOT NULL,target_memory_id TEXT NOT NULL,
          target_revision INTEGER NOT NULL,object_type TEXT NOT NULL,object_id TEXT NOT NULL,
          kind TEXT NOT NULL,state TEXT NOT NULL,operation_id TEXT NOT NULL)''',
        'CREATE INDEX IF NOT EXISTS memory_relation_guard_target ON memory_relation_guards(target_memory_id,target_revision)',
    ):
        conn.execute(sql)


def project(conn, memory_id, revision, semantic):
    conn.execute('DELETE FROM memory_relation_edges WHERE host_memory_id=? AND host_revision=?', (memory_id, revision))
    host = semantic.get('entry', {}).get('parent_ref') or {'memory_id': memory_id, 'revision': revision}
    for relation in semantic.get('relations', []):
        endpoints = []
        for name in ('from', 'to'):
            endpoint = relation[name]
            mid = endpoint.get('memory_id') or host['memory_id']
            rev = host['revision'] if endpoint.get('revision') == 'this_revision' else endpoint['revision']
            if type(rev) is not int:
                break
            endpoints.extend([mid, rev, endpoint['object_type'], endpoint['object_id']])
        if len(endpoints) != 8:
            continue
        conn.execute('INSERT INTO memory_relation_edges VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (memory_id, revision, relation['relation_id'], semantic_sha256(relation), relation['kind'],
             *endpoints, canonical_json(relation)))


def owner_adoption_ids(previous, current, explicit=()):
    """Only an actual confirmation transition/new relation or explicit IDs."""
    old = {row['relation_id']: row for row in previous.get('relations', [])}
    new = {row['relation_id']: row for row in current.get('relations', [])}
    if not isinstance(explicit, (list, tuple)) or any(not isinstance(key, str) or key not in new for key in explicit):
        raise ValueError('invalid_owner_relation_adoptions')
    return sorted({row['relation_id'] for row in new.values() if row.get('confirmation_state') == 'confirmed'
        and (row['relation_id'] in explicit or row['relation_id'] not in old
             or old[row['relation_id']].get('confirmation_state') != 'confirmed')})


def _diary_review_proof(conn, memory_id, actor, decision_source, metadata):
    """Resolve the real owner Review decision; a worker metadata flag is no proof."""
    pointer = metadata.get('diary_l1_review') or {}
    if actor != 'diary_first_pass' or decision_source != 'diary_first_pass':
        return None
    candidate_id, revision = pointer.get('candidate_id'), pointer.get('candidate_revision')
    if not isinstance(candidate_id, str) or type(revision) is not int:
        return None
    row = conn.execute('SELECT proposal_json,status FROM candidate_revisions '
        'WHERE candidate_id=? AND revision=?', (candidate_id, revision)).fetchone()
    decision = conn.execute("SELECT request_id FROM candidate_decisions WHERE candidate_id=? "
        "AND candidate_revision=? AND to_status='accepted' AND actor='owner'",
        (candidate_id, revision)).fetchone()
    if not row or row['status'] != 'accepted' or not decision:
        return None
    proposal = json.loads(row['proposal_json'])
    source_metadata = proposal.get('metadata') or {}
    bundle = source_metadata.get('diary_l1_bundle') or {}
    if (proposal.get('source_type') != 'diary' or bundle.get('memory_id') != memory_id
            or semantic_sha256(source_metadata.get('memory_l1') or {}) !=
               semantic_sha256(metadata.get('memory_l1') or {})):
        return None
    return {'candidate_id': candidate_id, 'candidate_revision': revision, 'request_id': decision['request_id']}


def _adoption_intent(conn, memory_id, revision, actor, decision_source, metadata):
    semantic = metadata.get('memory_l1') or {}
    relations = {row['relation_id']: row for row in semantic.get('relations', [])}
    request = metadata.get('memory_l1_owner_request') or {}
    ids = set(request.get('owner_relation_adoptions') or []) if actor == 'owner' else set()
    if (actor == 'owner' and decision_source == 'owner_l1_edit' and revision > 1
            and request.get('request_sha256')):
        prior = conn.execute('SELECT metadata_json FROM memory_revisions WHERE memory_id=? AND revision<? '
            'ORDER BY revision DESC LIMIT 1', (memory_id, revision)).fetchone()
        previous = json.loads(prior[0]).get('memory_l1', {}) if prior else {}
        ids.update(owner_adoption_ids(previous, semantic))
    diary_review = _diary_review_proof(conn, memory_id, actor, decision_source, metadata)
    if actor == 'owner' or diary_review:
        # Review preserves the actual owner-edited candidate revisions. A title
        # edit of an already-confirmed model relation has no adoption transition.
        rows = conn.execute('SELECT revision,proposal_json,actor FROM candidate_revisions '
            'WHERE candidate_id=? AND revision<=? ORDER BY revision',
            (diary_review['candidate_id'], diary_review['candidate_revision']) if diary_review
            else (memory_id, 9223372036854775807)).fetchall()
        previous = {}
        for row in rows:
            current = json.loads(row['proposal_json']).get('metadata', {}).get('memory_l1') or {}
            if row['actor'] == 'owner':
                for key in owner_adoption_ids(previous, current):
                    if key in relations and semantic_sha256(relations[key]) == semantic_sha256(
                            next(r for r in current['relations'] if r['relation_id'] == key)):
                        ids.add(key)
            previous = current
    return ids


def record_adoptions(conn, memory_id, revision, operation_id, actor, decision_source, metadata):
    semantic = metadata.get('memory_l1') or {}
    relations = {row['relation_id']: row for row in semantic.get('relations', [])}
    request = metadata.get('memory_l1_owner_request') or {}
    ids = _adoption_intent(conn, memory_id, revision, actor, decision_source, metadata)
    diary_review = _diary_review_proof(conn, memory_id, actor, decision_source, metadata)
    for key, relation in relations.items():
        # Program-authored relationships retain their existing exact trust
        # source. Model relationships require the specific owner operation.
        adopted = (key in ids and (actor == 'owner' or diary_review is not None)) or (
            relation.get('origin') == 'program' and relation.get('confirmation_state') == 'confirmed'
            and actor not in {'model', 'first_pass_model'})
        if not adopted or relation.get('confirmation_state') != 'confirmed':
            continue
        relation_sha = semantic_sha256(relation)
        conn.execute('INSERT OR IGNORE INTO memory_relation_adoptions VALUES(?,?,?,?,?,?,?)',
            (memory_id, key, relation_sha, operation_id,
             str(diary_review['request_id'] if diary_review else request.get('request_id') or operation_id),
             'owner' if diary_review else actor, datetime.now(timezone.utc).isoformat()))
        edge = conn.execute('SELECT * FROM memory_relation_edges WHERE host_memory_id=? AND host_revision=? AND relation_id=?',
            (memory_id, revision, key)).fetchone()
        if edge and relation['kind'] in RESTRICTING:
            guard = semantic_sha256({'host': memory_id, 'relation_id': key, 'relation_sha256': relation_sha})
            conn.execute('INSERT OR IGNORE INTO memory_relation_guards VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (guard, memory_id, revision, key, relation_sha, edge['to_memory_id'], edge['to_revision'],
                 edge['to_type'], edge['to_id'], edge['kind'], 'active', operation_id))


def _edges(conn, memory_ids, *, adopted=True):
    ids = list(dict.fromkeys(memory_ids))
    if not ids:
        return []
    result = {}
    for mid in ids:
        for column in ('host_memory_id', 'from_memory_id', 'to_memory_id'):
            query = ('SELECT e.*,m.state AS host_state,m.recall_policy AS host_policy '
                'FROM memory_relation_edges e JOIN memories m ON m.memory_id=e.host_memory_id '
                'AND m.active_revision=e.host_revision ')
            if adopted:
                query += ('JOIN memory_relation_adoptions a ON a.host_memory_id=e.host_memory_id '
                    'AND a.relation_id=e.relation_id AND a.relation_sha256=e.relation_sha256 ')
            query += 'WHERE e.' + column + '=?'
            for row in conn.execute(query, (mid,)):
                result[(row['host_memory_id'], row['host_revision'], row['relation_id'])] = dict(row)
    return list(result.values())


def _adoption_proof(conn, host, relation):
    relation_sha = semantic_sha256(relation)
    row = conn.execute('SELECT * FROM memory_relation_adoptions WHERE host_memory_id=? AND relation_id=? '
        'AND relation_sha256=?', (host, relation['relation_id'], relation_sha)).fetchone()
    if row:
        return dict(row)
    # Bounded historical compatibility: inspect only the located host's own
    # immutable operations, never upgrade a confirmed flag by itself.
    for row in conn.execute('SELECT * FROM memory_revisions WHERE memory_id=? ORDER BY revision DESC', (host,)):
        metadata = json.loads(row['metadata_json'])
        candidate = next((r for r in metadata.get('memory_l1', {}).get('relations', [])
                          if r['relation_id'] == relation['relation_id']), None)
        if not candidate or semantic_sha256(candidate) != relation_sha:
            continue
        ids = _adoption_intent(conn, host, row['revision'], row['created_by'], row['decision_source'], metadata)
        program = relation.get('origin') == 'program' and row['created_by'] not in {'model', 'first_pass_model'}
        if relation.get('confirmation_state') == 'confirmed' and (relation['relation_id'] in ids or program):
            return {'operation_id': row['operation_id'], 'actor': row['created_by'], 'relation_sha256': relation_sha}
    return None


def _legacy_edges(conn, memory_ids):
    """Use the existing exact typed-feature index until a located host is rebuilt."""
    endpoints, located = [], set()
    for mid in dict.fromkeys(memory_ids):
        for row in conn.execute("SELECT f.memory_id,f.revision,f.object_id FROM memory_l1_features f JOIN memories m "
                "ON m.memory_id=f.memory_id AND m.active_revision=f.revision WHERE f.memory_id=? AND f.object_type='relation'", (mid,)):
            located.add((row[0], row[1], row[2]))
        revisions = [row[0] for row in conn.execute('SELECT revision FROM memory_l1_index_revisions WHERE memory_id=?', (mid,))]
        for revision in revisions:
            endpoints.append({'memory_id': mid, 'revision': revision, 'object_type': 'memory', 'object_id': mid})
        for row in conn.execute("SELECT revision,module,object_id FROM memory_l1_members WHERE memory_id=? AND module IN ('statements','scenes')", (mid,)):
            endpoints.append({'memory_id': mid, 'revision': row['revision'],
                'object_type': 'statement' if row['module'] == 'statements' else 'scene', 'object_id': row['object_id']})
    for offset in range(0, len(endpoints), 300):
        chunk = [canonical_json(e) for e in endpoints[offset:offset + 300]]
        marks = ','.join('?' for _ in chunk)
        for row in conn.execute("SELECT f.memory_id,f.revision,f.object_id FROM memory_l1_features f JOIN memories m "
                "ON m.memory_id=f.memory_id AND m.active_revision=f.revision WHERE f.feature_path IN ('from','to') "
                "AND f.value_state='known' AND f.value_json IN (" + marks + ')', chunk):
            located.add((row[0], row[1], row[2]))
    result = []
    for host, revision, ident in sorted(located):
        row = conn.execute('SELECT r.metadata_json,m.state,m.recall_policy FROM memory_revisions r JOIN memories m '
            'ON m.memory_id=r.memory_id WHERE r.memory_id=? AND r.revision=?', (host, revision)).fetchone()
        semantic = json.loads(row['metadata_json']).get('memory_l1') or {}
        relation = next((r for r in semantic.get('relations', []) if r['relation_id'] == ident), None)
        proof = _adoption_proof(conn, host, relation) if relation else None
        if not proof:
            continue
        parent = semantic.get('entry', {}).get('parent_ref') or {'memory_id': host, 'revision': revision}
        edge = {'host_memory_id': host, 'host_revision': revision, 'relation_id': ident,
            'relation_sha256': semantic_sha256(relation), 'kind': relation['kind'],
            'relation_json': canonical_json(relation), 'host_state': row['state'], 'host_policy': row['recall_policy']}
        for side in ('from', 'to'):
            endpoint = relation[side]
            edge.update({side + '_memory_id': endpoint.get('memory_id') or parent['memory_id'],
                side + '_revision': parent['revision'] if endpoint['revision'] == 'this_revision' else endpoint['revision'],
                side + '_type': endpoint['object_type'], side + '_id': endpoint['object_id']})
        result.append(edge)
    return result


def relation_context(conn, memory_ids):
    edges = {(r['host_memory_id'], r['host_revision'], r['relation_id']): r for r in _edges(conn, memory_ids)}
    for row in _legacy_edges(conn, memory_ids):
        edges.setdefault((row['host_memory_id'], row['host_revision'], row['relation_id']), row)
    edges = list(edges.values())
    guards = {}
    for mid in dict.fromkeys(memory_ids):
        for row in conn.execute('SELECT * FROM memory_relation_guards WHERE target_memory_id=?', (mid,)):
            guards[row['guard_id']] = dict(row)
    for row in edges:
        if row['kind'] in RESTRICTING and row['to_memory_id'] in memory_ids:
            guard_id = semantic_sha256({'host': row['host_memory_id'], 'relation_id': row['relation_id'],
                'relation_sha256': row['relation_sha256']})
            guards.setdefault(guard_id, {'guard_id': guard_id, 'host_memory_id': row['host_memory_id'],
                'host_revision': row['host_revision'], 'relation_id': row['relation_id'],
                'relation_sha256': row['relation_sha256'], 'target_memory_id': row['to_memory_id'],
                'target_revision': row['to_revision'], 'object_type': row['to_type'], 'object_id': row['to_id'],
                'kind': row['kind'], 'state': 'active' if row['host_state'] == 'active' and row['host_policy'] == 'enabled'
                    else 'dependency_withdrawn', 'operation_id': (_adoption_proof(conn, row['host_memory_id'],
                        json.loads(row['relation_json'])) or {}).get('operation_id', '')})
    for guard in guards.values():
        guard['scope'] = guard_scope(conn, guard)
    return {'relations': [{**{k: v for k, v in row.items() if k != 'relation_json'},
        'relation': json.loads(row['relation_json']), 'adoption': {'state': 'adopted',
            'relation_sha256': row['relation_sha256']}} for row in sorted(edges,
                key=lambda r: (r['host_memory_id'], r['host_revision'], r['relation_id']))],
        'guards': sorted(guards.values(), key=lambda r: r['guard_id'])}


def guard_scope(conn, guard):
    mid = guard['target_memory_id']
    original = conn.execute('SELECT metadata_json FROM memory_revisions WHERE memory_id=? AND revision=?',
        (mid, guard['target_revision'])).fetchone()
    latest = conn.execute('SELECT r.metadata_json FROM memories m JOIN memory_revisions r '
        'ON r.memory_id=m.memory_id AND r.revision=m.active_revision WHERE m.memory_id=?', (mid,)).fetchone()
    if not original or not latest:
        return {'statement_ids': [], 'unknown_scope': True}
    old = json.loads(original[0]).get('memory_l1') or {}
    current = json.loads(latest[0]).get('memory_l1') or {}
    before = {s['statement_id']: s for s in old.get('statements', [])}
    after = {s['statement_id']: s for s in current.get('statements', [])}
    if guard['object_type'] == 'statement':
        ids = [guard['object_id']]
    elif guard['object_type'] == 'scene':
        scene = next((s for s in old.get('scenes', []) if s['scene_id'] == guard['object_id']), None)
        if not scene:
            return {'statement_ids': [], 'unknown_scope': True}
        ids = scene['statement_refs']
    else:
        ids = list(before)
    if not ids or any(key not in before for key in ids):
        return {'statement_ids': [], 'unknown_scope': True}
    affected = [key for key in ids if key in after and (
        before[key].get('statement_revision_id') is None or after[key].get('statement_revision_id') is None
        or before[key]['statement_revision_id'] == after[key]['statement_revision_id'])]
    return {'statement_ids': affected, 'unknown_scope': False, 'historical_only': not bool(affected)}


def dependency_snapshot(conn, memory_id):
    row = conn.execute('SELECT generation FROM memory_relation_generations WHERE memory_id=?', (memory_id,)).fetchone()
    context = relation_context(conn, [memory_id])
    sync = pending_sync_scope(conn, memory_id)
    result = {'generation': row[0] if row else 0,
        'relations': [{key: r[key] for key in ('host_memory_id', 'host_revision', 'relation_id',
            'relation_sha256', 'host_state', 'host_policy')} for r in context['relations']],
        'guards': context['guards'], 'l1_sync_required': sync is not None, 'sync_scope': sync}
    result['token'] = semantic_sha256(result)
    return result


def cache_versions(conn, memory_ids):
    """Small per-memory versions; no prose or whole-library invalidation."""
    ids = list(dict.fromkeys(memory_ids))
    values = {mid: {'relations': None, 'presentation': None, 'sync': []} for mid in ids}
    for offset in range(0, len(ids), 300):
        chunk = ids[offset:offset + 300]
        marks = ','.join('?' for _ in chunk)
        for table, columns, key in (('memory_relation_generations', 'memory_id,generation', 'relations'),
                                    ('memory_l2_heads', 'memory_id,presentation_revision', 'presentation')):
            try:
                for row in conn.execute('SELECT ' + columns + ' FROM ' + table + ' WHERE memory_id IN (' + marks + ')', chunk):
                    values[row[0]][key] = row[1]
            except sqlite3.OperationalError:
                pass  # Older databases remain distinguishable from generation 0.
        try:
            for row in conn.execute('SELECT memory_id,presentation_revision,state FROM memory_l2_sync_proposals '
                    'WHERE memory_id IN (' + marks + ') ORDER BY memory_id,presentation_revision', chunk):
                values[row[0]]['sync'].append([row[1], row[2]])
        except sqlite3.OperationalError:
            pass
    return {mid: semantic_sha256(value) for mid, value in values.items()}


def propagate_change(conn, memory_id, revision, operation_id, reason='revision_committed'):
    """Walk indexed actual dependencies, not the library; preserve withdrawal guards."""
    pending, affected = [memory_id], set()
    while pending:
        mid = pending.pop()
        if mid in affected:
            continue
        affected.add(mid)
        for edge in _edges(conn, [mid]):
            pending.extend([edge['host_memory_id'], edge['from_memory_id'], edge['to_memory_id']])
        for row in conn.execute('SELECT target_memory_id FROM memory_relation_guards WHERE host_memory_id=?', (mid,)):
            pending.append(row[0])
        # Shared diary scene members are explicit dependencies of their root.
        pending.extend(row[0] for row in conn.execute('SELECT r.memory_id FROM memory_revisions r JOIN memories m '
            'ON m.memory_id=r.memory_id AND m.active_revision=r.revision '
            "WHERE json_extract(r.metadata_json,'$.memory_l1.entry.parent_ref.memory_id')=?", (mid,)))
    for mid in sorted(affected):
        conn.execute('INSERT INTO memory_relation_generations VALUES(?,1,?) ON CONFLICT(memory_id) '
            'DO UPDATE SET generation=generation+1,operation_id=excluded.operation_id', (mid, operation_id))
        host = conn.execute('SELECT state,recall_policy FROM memories WHERE memory_id=?', (mid,)).fetchone()
        if host and (host['state'] != 'active' or host['recall_policy'] != 'enabled'):
            conn.execute("UPDATE memory_relation_guards SET state='dependency_withdrawn',operation_id=? WHERE host_memory_id=?",
                (operation_id, mid))
        row = conn.execute('SELECT m.active_revision,r.metadata_json FROM memories m JOIN memory_revisions r '
            'ON r.memory_id=m.memory_id AND r.revision=m.active_revision WHERE m.memory_id=? '
            "AND m.state='active' AND m.recall_policy='enabled'", (mid,)).fetchone()
        if row and 'memory_l1' in json.loads(row['metadata_json']) and (mid != memory_id or reason != 'revision_committed'):
            from memory_l2_store import enqueue_in_transaction
            enqueue_in_transaction(conn, mid, row['active_revision'], request_id='dependency:' + operation_id)
    return sorted(affected)
