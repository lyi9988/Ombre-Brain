import asyncio

from memory_l1_features import read_features
from memory_l1_ingestion import prepare_model_document
from memory_l1_schema import split_document, unknown
from test_memory_l1_ingestion_p2c import setup


def test_projection_preserves_known_zero_unknown_and_six_axes_without_ranking(tmp_path):
    service, source, draft, generation = setup(tmp_path)
    catalog, revision = service.catalog_provider()
    catalog['attributes'] = [{'id': 'measured_value', 'value_type': 'number', 'scope': 'any'}]
    service.catalog_provider = lambda: (catalog, revision)
    draft['attributes'] = [{'definition_id': 'measured_value', 'value': unknown('known', 0, ['source:current']),
        'evidence_refs': ['source:current']}]
    draft['statements'][0]['property'] = {'key': 'measured_value', 'value': unknown('known', 0, ['source:current']),
        'polarity': 'negated', 'evidence_refs': ['source:current']}
    doc = prepare_model_document(service, memory_id='f1', document=draft, sources=[source],
        generation=generation, request_id='feature-test')
    body, semantic = split_document(doc)
    asyncio.run(service.commit.commit_memory(memory_id='f1', bucket_id='f1', expected_revision=0, body=body,
        metadata={'memory_l1': semantic}, source_refs=['synthetic'], decision_source='test', idempotency_key='features', actor='test'))
    features = read_features(service.authority, 'f1', 1)
    zero = [f for f in features if f['feature_path'] == 'attributes.measured_value'][0]
    assert zero['value_state'] == 'known' and zero['value'] == 0
    missing = [f for f in features if f['feature_path'] == 'classification.modality'][0]
    assert missing['value_state'] == 'not_stated' and missing['value'] is None
    assert [f for f in features if f['feature_path'] == 'property_polarity.measured_value'][0]['value'] == 'negated'
    assert all('score' not in f for f in features)
    # A derived-index rebuild is exact and cannot change/adopt any source.
    db = service.authority._connect()
    db.execute('DELETE FROM memory_l1_features WHERE memory_id=?', ('f1',))
    db.execute('DELETE FROM memory_l1_index_revisions WHERE memory_id=?', ('f1',))
    db.commit(); db.close()
    assert read_features(service.authority, 'f1', 1) == features
    assert service.authority.get_memory('f1')['active_revision'] == 1
