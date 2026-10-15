import asyncio
import copy
import hashlib

import pytest

from bucket_manager import BucketManager
from memory_authority import MemoryAuthorityStore
from memory_commit_service import BucketMemoryProjection, MemoryCommitService
from memory_l1_contract import L1ContractError
from memory_l1_ingestion import model_template, prepare_model_document
from memory_l1_schema import catalog_with_defaults, statement_template, unknown
from memory_l1_service import L1OwnerError, MemoryL1RevisionService
from memory_source_provenance import canonical_source


def setup(tmp_path):
    config = {'buckets_dir': str(tmp_path / 'buckets'), 'state_dir': str(tmp_path / 'state')}
    authority = MemoryAuthorityStore(config)
    buckets = BucketManager(config)
    commit = MemoryCommitService(authority, BucketMemoryProjection(config, buckets))
    catalog = catalog_with_defaults()
    service = MemoryL1RevisionService(authority, commit, buckets, lambda: (catalog, 'catalog-test'))
    source = canonical_source({'event_id': 'e1', 'version_id': 'v1', 'role': 'user',
        'text': '两个月前肚子痛，那次想要抱抱。🙂', 'created_at': '2026-10-15T00:00:00Z'},
        conversation_id='c1', profile_id='p1')
    draft = model_template(catalog=catalog)
    draft['narrative'] = {'title': '一次旧事', 'body': '那次她说肚子痛，希望我抱抱。' + '这是完整的合成回忆。' * 900}
    draft['entities'] = [{'entity_id': 'new:user', 'entity_type': unknown('known', 'user', ['source:current']),
        'name': '主人', 'aliases': []}]
    symptom = statement_template()
    symptom.pop('statement_revision_id')
    symptom.update(statement_id='new:symptom', content='那次肚子痛。', evidence_refs=['source:current'],
        subject_ref=unknown('known', 'new:user', ['source:current']))
    symptom['classification']['kind'] = unknown('known', 'state', ['source:current'])
    comfort = copy.deepcopy(symptom)
    comfort.update(statement_id='new:comfort', content='那次希望抱抱。')
    comfort['classification']['kind'] = unknown('known', 'preference', ['source:current'])
    comfort['classification']['stability'] = unknown('known', 'episode', ['source:current'])
    draft['statements'] = [symptom, comfort]
    draft['scenes'] = [{'scene_id': 'new:scene', 'title': '那次陪伴', 'statement_refs': ['new:symptom', 'new:comfort'],
        'attributes': [], 'source_refs': [], 'source_ranges': [{'source_ref': 'source:current', 'start': 0, 'end': 9}]}]
    generation = {'prompt_revision': 'prompt-r1', 'model_route_revision': 'route-r1', 'task_id': 't1', 'actor': 'model'}
    return service, source, draft, generation


def prepare(env):
    service, source, draft, generation = env
    return prepare_model_document(service, memory_id='m1', document=draft, sources=[source], generation=generation, request_id='t1')


def test_full_model_body_and_independent_statements_round_trip_without_time_promotion(tmp_path):
    env = setup(tmp_path)
    doc = prepare(env)
    assert len(doc['narrative']['body']) > 4000
    assert doc['narrative'] == env[2]['narrative']
    assert doc['statements'][0]['classification']['kind']['value'] == 'state'
    assert doc['statements'][1]['classification']['stability']['value'] == 'episode'
    assert doc['statements'][0]['time']['event']['start']['state'] == 'not_stated'
    assert doc['statements'][0]['time']['applicability']['temporal_scope'] == 'unknown'
    assert doc['statements'][0]['evidence_refs'] == ['source:1']
    assert doc['scenes'][0]['statement_refs'] == [s['statement_id'] for s in doc['statements']]
    assert doc['generation']['origin'] == 'first_pass_model'
    assert env[0].authority.list_memories() == []


def test_scene_anchor_is_program_computed_in_codepoints_and_l0_copy_is_exact(tmp_path):
    env = setup(tmp_path)
    doc = prepare(env)
    anchor = doc['scenes'][0]['anchors'][0]
    assert anchor['start'] == 0 and anchor['end'] == 9
    assert anchor['quote_sha256'] == hashlib.sha256(env[1]['text'][:9].encode()).hexdigest()
    from memory_l0_sources import MemoryL0Sources
    snapshot = MemoryL0Sources(env[0].authority).copy(doc['evidence_bindings'][0])
    assert snapshot['text'] == env[1]['text']
    assert snapshot['upstream_revision'] == 'v1'


@pytest.mark.parametrize('field', ['generation', 'evidence_bindings', 'schema_version'])
def test_model_cannot_supply_managed_provenance(tmp_path, field):
    env = setup(tmp_path)
    env[2][field] = {}
    with pytest.raises(L1OwnerError, match='model_document_fields_required'): prepare(env)


def test_null_actual_model_route_is_recorded_unknown_instead_of_invented(tmp_path):
    env = setup(tmp_path)
    env[3].update(prompt_revision=None, model_route_revision=None)
    assert prepare(env)['generation']['model_route_revision'] is None


def test_supported_predicate_grammar_is_bounded_strict_and_catalog_typed(tmp_path):
    from jsonschema import Draft202012Validator
    from memory_l1_schema import MAX_PREDICATE_DEPTH
    from memory_l1_wire_schema import MAX_PREDICATE_DEPTH as WIRE_PREDICATE_DEPTH, MEMORY_WRITE_PARAMETERS

    env = setup(tmp_path)
    service, _source, draft, _generation = env
    assert MAX_PREDICATE_DEPTH == WIRE_PREDICATE_DEPTH
    catalog, revision = service.catalog_provider()
    catalog['attributes'] = [{'id': 'temperature', 'value_type': 'number', 'scope': 'any'}]
    service.catalog_provider = lambda: (catalog, revision)

    def predicate(depth):
        node = {'op': 'eq', 'field': 'temperature', 'value': 2.5}
        for _ in range(depth):
            node = {'op': 'all', 'args': [node, {'op': 'le', 'field': 'temperature', 'value': 4}]}
        return node

    def use_predicate(value):
        result = copy.deepcopy(draft)
        result['statements'][0]['conditions'] = {
            'state': 'known', 'text': None, 'predicate': value,
            'evidence_refs': ['source:current'], 'exceptions': [],
        }
        return result

    at_limit = use_predicate(predicate(MAX_PREDICATE_DEPTH))
    Draft202012Validator.check_schema(MEMORY_WRITE_PARAMETERS)
    wire = Draft202012Validator(MEMORY_WRITE_PARAMETERS)
    assert wire.is_valid({'document': at_limit})
    prepared = prepare_model_document(service, memory_id='predicate-ok', document=at_limit,
        sources=[env[1]], generation=env[3], request_id='predicate-ok')
    actual = prepared['statements'][0]['conditions']['predicate']
    assert actual['op'] == 'all' and actual['args'][0]['op'] == 'all'
    assert prepared['statements'][0]['conditions']['evidence_refs'] == ['source:1']

    over_limit = use_predicate(predicate(MAX_PREDICATE_DEPTH + 1))
    assert Draft202012Validator(MEMORY_WRITE_PARAMETERS).is_valid({'document': over_limit})
    with pytest.raises(L1ContractError, match='predicate_depth_exceeded'):
        prepare_model_document(service, memory_id='predicate-deep', document=over_limit,
            sources=[env[1]], generation=env[3], request_id='predicate-deep')

    malformed_cases = [
        ({'op': 'eval', 'expression': 'anything'}, 'unsupported_predicate'),
        ({'op': 'all', 'args': [], 'expression': 'anything'}, 'invalid_object_fields'),
        ({'op': 'not', 'args': [
            {'op': 'eq', 'field': 'temperature', 'value': 1},
            {'op': 'eq', 'field': 'temperature', 'value': 2}]}, 'invalid_predicate_arity'),
        ({'op': 'eq', 'field': 'unconfigured', 'value': 1}, 'unknown_condition_field'),
    ]
    for index, (value, code) in enumerate(malformed_cases):
        bad = use_predicate(value)
        if code != 'unknown_condition_field':
            assert not wire.is_valid({'document': bad})
        else:
            # Vocabulary membership depends on the live catalog, so only the
            # owner normalizer can reject a structurally valid unknown field.
            assert wire.is_valid({'document': bad})
        with pytest.raises(L1ContractError, match=code):
            prepare_model_document(service, memory_id=f'predicate-bad-{index}', document=bad,
                sources=[env[1]], generation=env[3], request_id=f'predicate-bad-{index}')


def test_model_supplied_revision_or_unknown_source_is_rejected(tmp_path):
    env = setup(tmp_path)
    env[2]['statements'][0]['statement_revision_id'] = 'fake'
    with pytest.raises(L1OwnerError, match='model_managed_field'): prepare(env)
    env[2]['statements'][0].pop('statement_revision_id')
    env[2]['statements'][0]['evidence_refs'] = ['source:999']
    with pytest.raises(L1ContractError, match='dangling_evidence_ref'): prepare(env)


def test_scene_range_cannot_claim_unseen_coordinates(tmp_path):
    env = setup(tmp_path)
    env[2]['scenes'][0]['source_ranges'][0]['end'] = 999
    with pytest.raises(L1OwnerError, match='scene_range_outside_observed_input'): prepare(env)


def test_model_relations_remain_proposed_and_ids_are_stable_on_replay(tmp_path):
    env = setup(tmp_path)
    endpoint = lambda ident: {'memory_id': None, 'revision': 'this_revision', 'object_type': 'statement', 'object_id': ident}
    env[2]['relations'] = [{'relation_id': 'new:r1', 'from': endpoint('new:symptom'), 'to': endpoint('new:comfort'),
        'kind': 'supplements', 'evidence_refs': ['source:current']}]
    a, b = prepare(env), prepare(env)
    assert a == b
    assert a['relations'][0]['origin'] == 'model'
    assert a['relations'][0]['confirmation_state'] == 'proposed'
