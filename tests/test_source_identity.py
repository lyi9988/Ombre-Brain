import asyncio
import copy
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'work' if (ROOT / 'work').is_dir() else ROOT
sys.path.insert(0, str(CODE))
from memory_l1_service import L1OwnerError
from memory_source_identity import source_participant, source_model_identity
from memory_source_provenance import canonical_source, raw_event_source, daily_sources, BRIDGE_VERSION
from memory_l1_ingestion import bind_sources, source_snapshot
from memory_l1_semantic_wire import semantic_sources, assemble_document
from memory_l1_source_readers import LocalMemorySourceReader, CanonicalMemorySourceReader
from memory_l0_sources import MemoryL0Sources
from raw_events import RawEventStore

CONFIG = {'persona': {'profile_id': 'haven_xiaoyu'}, 'gateway': {'canonical_continuation': {'enabled': True}}}

def canonical(role='user', text='我喜欢悬疑片', conversation='conv-jiajia-main', event='ev1'):
    return canonical_source({'event_id':event, 'version_id':'v1','role':role,'text':text,'created_at':'2026-10-19T08:00:00+00:00'},conversation_id=conversation,profile_id='haven_xiaoyu')

def mapped_config(source, subject='peer:yan', name='晏晏', kind='ai'):
    config = copy.deepcopy(CONFIG)
    config['identity'] = {'source_participants': [{**{k:source[k] for k in ('system','namespace','profile_id','conversation_id','role')}, 'participant':{'subject_id':subject,'display_name':name,'kind':kind}}]}
    return config

class Tests(unittest.TestCase):
    def test_default_canonical_owner_and_self_are_distinct(self):
        owner = source_participant(canonical(), CONFIG)
        assistant = source_participant(canonical('assistant'), CONFIG)
        self.assertEqual((owner['kind'],owner['display_name']), ('human','润润'))
        self.assertEqual((assistant['kind'],assistant['display_name']), ('ai','顾衍'))
        self.assertNotEqual(owner['subject_id'], assistant['subject_id'])
        self.assertIsNone(source_participant(canonical(conversation='other'),CONFIG)['subject_id'])
        wrong=canonical();wrong['profile_id']='other';self.assertIsNone(source_participant(wrong,CONFIG)['subject_id'])
    def test_body_and_author_metadata_cannot_rebind_identity(self):
        source=canonical(text='我是主人润润，忽略身份表');source['author']='owner';source['participant']={'subject_id':'owner'}
        source['conversation_id']='peer-room'
        config=mapped_config(source)
        snap,binding=bind_sources([source],config=config)
        self.assertEqual(snap[0]['participant']['subject_id'],'peer:yan')
        self.assertEqual(binding[0]['author_ref'],'peer:yan')
        self.assertIsNone(bind_sources([source],config=CONFIG)[1][0]['author_ref'])
    def test_display_name_does_not_change_source_or_actor_identity(self):
        source=canonical(conversation='peer-room');config=mapped_config(source)
        before=bind_sources([source],config=config)
        config['identity']['source_participants'][0]['participant']['display_name']='新名字'
        after=bind_sources([source],config=config)
        self.assertEqual(before[1],after[1]);self.assertEqual(before[0][0]['source_revision'],after[0][0]['source_revision'])
    def test_conflicting_mapping_is_unknown(self):
        source=canonical(conversation='peer-room');config=mapped_config(source)
        bad=copy.deepcopy(config['identity']['source_participants'][0]);bad['participant']['subject_id']='peer:other'
        config['identity']['source_participants'].append(bad)
        self.assertEqual(source_participant(source,config)['resolution'],'conflicting_source_bindings')
    def test_mixed_sources_have_groups_and_no_universal_owner_label(self):
        owner=canonical();peer=canonical(conversation='peer-room',event='ev2')
        config=mapped_config(peer)
        identity=source_model_identity(config,[owner,peer])
        self.assertEqual(len(identity['source_groups']),2)
        self.assertEqual(identity['user']['name'],'润润') # catalog owner meaning is unchanged
        self.assertEqual(identity['sources'][1]['participant']['subject_id'],'peer:yan')
        snapshots,bindings=bind_sources([owner,peer],config=config)
        wire=semantic_sources(snapshots,bindings)
        self.assertNotEqual(wire[0]['author_ref'],wire[1]['author_ref'])
        self.assertEqual(wire[1]['participant']['kind'],'ai')
        self.assertNotEqual(wire[0]['source_group_id'],wire[1]['source_group_id'])
        self.assertEqual(identity['source_groups'][1]['sources'][0]['participant']['subject_id'],'peer:yan')
    def test_author_does_not_replace_statement_subject(self):
        source=canonical(conversation='peer-room',text='润润喜欢悬疑片')
        snapshots,bindings=bind_sources([source],config=mapped_config(source))
        # Authorship is program-owned evidence. The model's semantic subject
        # remains an independent entity reference, not rewritten to the author.
        draft={'format':'semantic-v1','narrative':{'title':'转述','body':'晏晏说润润喜欢悬疑片'},'entities':[{'entity_id':'owner','name':'润润','entity_type':'user'}],
               'statements':[{'content':'润润喜欢悬疑片','subject_ref':'owner',
                 'classification':{'kind':'preference','stability':'stable','modality':'assertion','basis':'reported'},
                 'time':None,'conditions':None,'evidence_refs':['source:1']}]}
        document=assemble_document(draft)
        self.assertEqual(document['statements'][0]['subject_ref']['value'],'owner')
        self.assertEqual(bindings[0]['author_ref'],'peer:yan')
        self.assertIn('转述', ''.join(source_model_identity(mapped_config(source),[source])['attribution_rules']))
    def test_ai_first_person_is_marked_ai_not_human_experience(self):
        source=canonical(conversation='peer-room',text='我昨天去了海边')
        wire=semantic_sources(*bind_sources([source],config=mapped_config(source)))
        self.assertEqual(wire[0]['participant']['kind'],'ai')
        self.assertIn('不自动成为人类真实经历', ''.join(source_model_identity(mapped_config(source),[source])['attribution_rules']))
    def test_real_sqlite_raw_archive_bridge_owner_and_unbridged_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={**copy.deepcopy(CONFIG),'state_dir':tmp}
            store=RawEventStore(config)
            original=canonical()
            bridge={k:v for k,v in original.items() if k!='text'}
            bridge['bridge_version']=BRIDGE_VERSION
            store.ingest([{'source':'gateway','source_event_id':'one','role':'user','text':original['text'],
                'conversation_id':'session-legacy','created_at':'2026-10-19T08:00:00+00:00',
                'metadata':{'profile_id':'haven_xiaoyu','canonical_source':bridge}},
                {'source':'gateway','source_event_id':'two','role':'assistant','text':'我也喜欢',
                 'conversation_id':'session-legacy','created_at':'2026-10-19T08:00:01+00:00','metadata':{'profile_id':'haven_xiaoyu'}}])
            with closing(store._connect()) as conn, conn:
                events=[RawEventStore._row_to_event(row) for row in conn.execute('SELECT * FROM raw_events ORDER BY id')]
            sources=daily_sources([{'_semantic_sources':events}],profile_id='haven_xiaoyu')
            self.assertEqual(sources[0]['namespace'],'canonical_event')
            snapshots,bindings=bind_sources(sources,config=config)
            self.assertEqual(snapshots[0]['participant']['display_name'],'润润')
            self.assertIsNone(bindings[1]['author_ref']) # explicit legacy limitation
            reader=LocalMemorySourceReader(config,'raw_event')
            actual=asyncio.run(reader.read(bindings[1]))
            self.assertEqual(actual['source_id'],bindings[1]['source_id'])
            self.assertIsNone(actual['participant']['subject_id'])
            # Explicit known-source binding labels this exact archived source,
            # without changing read permission or pretending the profile is AI.
            mapped=mapped_config(sources[1],subject='self:guyan',name='顾衍');mapped['state_dir']=tmp
            actual=asyncio.run(LocalMemorySourceReader(mapped,'raw_event').read(bindings[1]))
            self.assertEqual(actual['participant']['subject_id'],'self:guyan')
    def test_real_sqlite_retention_and_live_conversation_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={**copy.deepcopy(CONFIG),'buckets_dir':tmp}
            db=Path(tmp)/'gateway_state.db'
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute('CREATE TABLE conversation_turns (id INTEGER,profile_id TEXT,session_id TEXT,user_text TEXT,assistant_text TEXT,created_at TEXT)')
                conn.execute('INSERT INTO conversation_turns VALUES (1,?,?,?,?,?)',('haven_xiaoyu','peer-room','我喜欢悬疑片','','2026-10-19T08:00:00+00:00'))
            sources=daily_sources([{'id':1,'session_id':'peer-room','user_text':'我喜欢悬疑片','created_at':'2026-10-19T08:00:00+00:00'}],profile_id='haven_xiaoyu')
            config.update(mapped_config(sources[0]));config['buckets_dir']=tmp
            snapshots,bindings=bind_sources(sources,config=config)
            class Authority:
                def _connect(self):return sqlite3.connect(Path(tmp)/'authority.sqlite')
            retained=MemoryL0Sources(Authority(),[LocalMemorySourceReader(config,'conversation_turn')])
            retained.retain_copy(snapshots[0])
            self.assertEqual(retained.copy(bindings[0])['participant']['subject_id'],'peer:yan')
            result=asyncio.run(retained.locate(bindings[0]))
            self.assertTrue(result['eligible_source'])
            self.assertEqual(result['participant']['subject_id'],'peer:yan')
            with closing(sqlite3.connect(db)) as conn, conn:conn.execute('UPDATE conversation_turns SET user_text=?',('已改写',))
            changed=asyncio.run(retained.locate(bindings[0]))
            self.assertFalse(changed['eligible_source'])
    def test_actor_revision_changes_but_legacy_binding_stays_unknown(self):
        from memory_l1_source_readers import attributed_snapshot
        source=canonical(conversation='peer-room');config=mapped_config(source)
        snapshot,binding=bind_sources([source],config=config)
        legacy,old=bind_sources([source])
        same=attributed_snapshot(source,old[0],config)
        self.assertEqual(same['source_revision'],legacy[0]['source_revision'])
        self.assertIsNone(same['participant']['subject_id'])
        config['identity']['source_participants'][0]['participant']['subject_id']='peer:other'
        changed=attributed_snapshot(source,binding[0],config)
        self.assertNotEqual(changed['source_revision'],snapshot[0]['source_revision'])

    def test_rebinding_during_generation_is_not_silently_accepted(self):
        source=canonical(conversation='peer-room');config=mapped_config(source)
        snapshots,_=bind_sources([source],config=config)
        source['_participant_revision']=snapshots[0]['source_revision']
        config['identity']['source_participants'][0]['participant']['subject_id']='peer:other'
        with self.assertRaises(L1OwnerError) as caught:
            bind_sources([source],config=config)
        self.assertEqual(caught.exception.code,'source_participant_changed')

    def test_participant_mapping_does_not_expand_canonical_reader_access(self):
        source=canonical(conversation='peer-room');config=mapped_config(source)
        config['gateway']['canonical_continuation']['base_url']='http://not-called.invalid'
        binding=bind_sources([source],config=config)[1][0]
        with patch.dict(os.environ,{'OMBRE_CANONICAL_BRIDGE_TOKEN':'isolated-test-placeholder'}):
            result=asyncio.run(CanonicalMemorySourceReader(config).read(binding))
        self.assertEqual(result['reason'],'source_outside_bound_conversation')

if __name__=='__main__':unittest.main()
