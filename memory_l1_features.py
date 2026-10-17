"""Typed, rebuildable L1 features; no ranking or fact authority lives here."""
from __future__ import annotations

from memory_l1_contract import canonical_json


def initialize(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS memory_l1_features (
            memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
            object_type TEXT NOT NULL, object_id TEXT NOT NULL,
            feature_path TEXT NOT NULL, ordinal INTEGER NOT NULL,
            value_state TEXT NOT NULL, value_json TEXT NOT NULL,
            PRIMARY KEY(memory_id,revision,object_type,object_id,feature_path,ordinal),
            FOREIGN KEY(memory_id,revision) REFERENCES memory_l1_index_revisions(memory_id,revision) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_l1_feature_lookup ON memory_l1_features(feature_path,value_state,value_json,memory_id,revision);
    ''')


def project(conn, memory_id, revision, semantic):
    # Explicit delete also covers SQLite callers with foreign_keys disabled.
    conn.execute('DELETE FROM memory_l1_features WHERE memory_id=? AND revision=?', (memory_id, revision))
    def put(kind, ident, path, value, ordinal=0):
        state = value.get('state', 'known') if isinstance(value, dict) else 'known'
        stored = value.get('value') if isinstance(value, dict) and 'state' in value else value
        conn.execute('INSERT INTO memory_l1_features VALUES(?,?,?,?,?,?,?,?)',
            (memory_id, revision, kind, ident, path, ordinal, state, canonical_json(stored)))
    def attributes(kind, ident, values):
        for index, item in enumerate(values):
            put(kind, ident, 'attributes.' + item['definition_id'], item['value'], index)
    for field in ('keywords', 'tags'):
        for ordinal, value in enumerate(semantic['retrieval'][field]):
            put('memory', memory_id, 'retrieval.' + field, value, ordinal)
    put('memory', memory_id, 'entry.form', semantic['entry']['form'])
    attributes('memory', memory_id, semantic['attributes'])
    for item in semantic['entities']:
        for key in ('entity_type', 'name'): put('entity', item['entity_id'], key, item[key])
        for i, value in enumerate(item['aliases']): put('entity', item['entity_id'], 'aliases', value, i)
    for item in semantic['statements']:
        ident = item['statement_id']
        put('statement', ident, 'subject_ref', item['subject_ref'])
        for field, value in item['classification'].items():
            if field == 'topics':
                for i, topic in enumerate(value): put('statement', ident, 'classification.topics', topic, i)
            else: put('statement', ident, 'classification.' + field, value)
        for group, values in item['time'].items():
            for field, value in values.items(): put('statement', ident, 'time.' + group + '.' + field, value)
        put('statement', ident, 'conditions', item['conditions'])
        cognition = item.get('cognition')
        if cognition:
            for key in ('role', 'epistemic_status'):
                put('statement', ident, 'cognition.' + key, cognition[key])
            for key in ('about_refs', 'open_questions'):
                for ordinal, value in enumerate(cognition[key]):
                    put('statement', ident, 'cognition.' + key, value, ordinal)
        if item['property'] is not None:
            prop = item['property']
            put('statement', ident, 'property.' + prop['key'], prop['value'])
            put('statement', ident, 'property_polarity.' + prop['key'], prop['polarity'])
        attributes('statement', ident, item['attributes'])
    for scene in semantic['scenes']: attributes('scene', scene['scene_id'], scene.get('attributes', []))
    for relation in semantic['relations']:
        for field in ('from', 'to', 'kind', 'origin', 'confirmation_state'):
            put('relation', relation['relation_id'], field, relation[field])


def read_features(authority, memory_id, revision):
    """A bounded record read; callers still apply access/lifecycle/source gates."""
    from memory_l1_index import MemoryL1Index
    if not MemoryL1Index(authority).ensure(memory_id, revision): return []
    import json
    conn = authority._connect()
    try:
        rows = conn.execute('SELECT * FROM memory_l1_features WHERE memory_id=? AND revision=? '
            'ORDER BY object_type,object_id,feature_path,ordinal', (memory_id, revision)).fetchall()
        return [{**dict(row), 'value': json.loads(row['value_json'])} for row in rows]
    finally: conn.close()
