"""Existing unknown-attribution reads survive only an unchanged raw original."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from memory_authority import MemoryAuthorityStore
from memory_l0_sources import MemoryL0Sources
from memory_l1_ingestion import bind_sources
from memory_l1_source_readers import configured_source_readers
from raw_events import RawEventStore
from test_memory_legacy_participants import enable_exact_bindings


def fixture(tmp_path, kind='raw_event'):
    config={'state_dir':str(tmp_path/'state'),'buckets_dir':str(tmp_path/'buckets'),
        'persona':{'profile_id':'jiajia-main'},'gateway':{'canonical_continuation':{'enabled':True,'conversation_id':'conv-jiajia-main'}}}
    authority=MemoryAuthorityStore(config)
    source={'system':'ombre','namespace':kind,'profile_id':'jiajia-main','conversation_id':'jiajia-main',
        'event_id':'1','role':'user','version_id':'','recorded_at':'2026-10-22T00:00:00Z','text':'这是完整的旧原话，不能改成别的。'}
    if kind=='raw_event':
        store=RawEventStore(config);path=Path(store.db_path)
        with sqlite3.connect(path) as db:
            db.execute('INSERT INTO raw_events(id,source,source_event_id,event_hash,role,text,created_at,ingested_at,conversation_id,session_id,client,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (1,'gateway','',hashlib.sha256(b'old-private-original').hexdigest(),'user',source['text'],source['recorded_at'],source['recorded_at'],'jiajia-main','','private-test',json.dumps({'profile_id':'jiajia-main'})))
    else:
        path=tmp_path/'buckets/gateway_state.db';path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE conversation_turns(id INTEGER PRIMARY KEY,profile_id TEXT,session_id TEXT,user_text TEXT,assistant_text TEXT,created_at TEXT)')
            db.execute('INSERT INTO conversation_turns VALUES(?,?,?,?,?,?)',(1,'jiajia-main','jiajia-main',source['text'],'',source['recorded_at']))
    _,bindings=bind_sources([source],config=config)
    readers=MemoryL0Sources(authority,configured_source_readers(config,authority))
    assert asyncio.run(readers.locate(bindings[0]))['eligible_source']
    # Retain the authenticated reader's normalized recorded_at, exactly as
    # the original admission path does, rather than a caller-made sidecar.
    snapshot=readers.copy(bindings[0])
    return config,authority,path,source,bindings[0],snapshot,readers


@pytest.mark.parametrize('kind',['raw_event','conversation_turn'])
def test_historical_read_preserves_unknown_and_new_binding_still_uses_known_actor(tmp_path,kind):
    config,authority,_,source,old,snapshot,_=fixture(tmp_path,kind)
    actors=enable_exact_bindings(config);readers=MemoryL0Sources(authority,configured_source_readers(config,authority))
    read=asyncio.run(readers.locate(old));assert read['eligible_source'] and read['state']=='verified_snapshot'
    assert read['participant']['subject_id'] is None and read['participant']['kind']=='unknown'
    assert asyncio.run(readers.locate(old,verify_only=True))['eligible_source']
    assert readers.copy(old)==snapshot
    new=bind_sources([source],config=config)[1][0]
    assert new['source_revision']!=old['source_revision'] and new['author_ref']==actors['user']['subject_id']
    assert asyncio.run(readers.locate(new))['participant']['subject_id']==actors['user']['subject_id']


@pytest.mark.parametrize('change',['body','shortened','time','role','peer','missing_copy','span'])
def test_historical_compatibility_does_not_qualify_changed_or_cross_actor_sources(tmp_path,change):
    config,authority,path,source,old,_,_=fixture(tmp_path);enable_exact_bindings(config)
    if change in {'body','shortened','time','role'}:
        field,value={'body':('text','不同的正文。'),'shortened':('text',source['text'][:-1]),
            'time':('created_at','2026-10-23T00:00:00Z'),'role':('role','assistant')}[change]
        with sqlite3.connect(path) as db:db.execute('UPDATE raw_events SET '+field+'=? WHERE id=1',(value,))
    elif change=='peer':
        for item in config['identity']['source_participants']:
            if item['role']=='user':item['participant']['subject_id']='peer:someone-else'
    elif change=='missing_copy':
        with authority._connect() as db,db:db.execute('DELETE FROM memory_l0_source_copies')
    else:
        old=deepcopy(old);old['anchors'][0]['end']-=1
    readers=MemoryL0Sources(authority,configured_source_readers(config,authority))
    assert not asyncio.run(readers.locate(old,verify_only=True))['eligible_source']
