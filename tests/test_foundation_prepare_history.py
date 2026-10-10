from model_request_trace import ModelRequestTraceStore


def test_initial_prepare_survives_retry_and_has_attempt_identity(tmp_path):
    store = ModelRequestTraceStore(tmp_path / 'trace.sqlite3')
    for total, status in [(10607, 'miss'), (6, 'hit')]:
        trace = store.begin_logical({'trace_id': 'same-trace', 'request_id': 'r', 'metadata': {
            'prepare_timing_debug': {'total_ms': total, 'steps_ms': {'selection': total},
                'prepare_snapshot_cache': {'status': status}, 'private_outline': 'do not project'}}})
        store.record_attempt(trace_id=trace, ordinal=1, provider='p', upstream='u', model='m', payload={})
    result = store.get(trace)
    assert result['metadata']['prepare_timing_debug']['total_ms'] == 6
    assert [p['total_ms'] for p in result['preparations']] == [10607, 6]
    assert [p['snapshot_status'] for p in result['preparations']] == ['miss', 'hit']
    assert [p['attempt_ids'][0] for p in result['preparations']] == [a['attempt_id'] for a in result['attempts']]
    assert 'private_outline' not in str(result['preparations'])


def test_missing_prepare_is_not_fabricated(tmp_path):
    store = ModelRequestTraceStore(tmp_path / 'trace.sqlite3')
    trace = store.begin_logical({'trace_id': 'old', 'request_id': 'r'})
    assert store.get(trace)['preparations'] == []


def test_malformed_timing_and_missing_phase_do_not_crash_or_reassign(tmp_path):
    store = ModelRequestTraceStore(tmp_path / 'trace.sqlite3')
    trace = store.begin_logical({'trace_id': 'mixed', 'request_id': 'r'})
    with store._connect() as db:
        store._event(db, trace, 'request.started', {'metadata': ['bad']})
        store._event(db, trace, 'request.started', {'metadata': {'prepare_timing_debug': {
            'total_ms': True, 'steps_ms': ['bad'], 'prepare_snapshot_cache': 'bad'}}})
        store._event(db, trace, 'attempt', {'attempt_id': 'a'})
        store._event(db, trace, 'request.started', [])
        store._event(db, trace, 'attempt', {'attempt_id': 'b'})
    result = store.get(trace)['preparations']
    assert len(result) == 1
    assert result[0]['prepare_ordinal'] == 3
    assert result[0]['total_ms'] is None
    assert result[0]['steps_ms'] == {}
    assert result[0]['snapshot_status'] is None
    assert result[0]['attempt_ids'] == ['a']


def test_only_valid_nonnegative_numeric_durations_are_projected(tmp_path):
    store = ModelRequestTraceStore(tmp_path / 'trace.sqlite3')
    trace = store.begin_logical({'trace_id': 'numeric', 'request_id': 'r', 'metadata': {
        'prepare_timing_debug': {'total_ms': 'private text', 'steps_ms': {
            'valid': 1.5, 'zero': 0, 'negative': -1, 'boolean': True, 'text': 'private'},
            'prepare_snapshot_cache': {'status': ['bad']}}}})
    result = store.get(trace)['preparations'][0]
    assert result['total_ms'] is None
    assert result['steps_ms'] == {'valid': 1.5, 'zero': 0}
    assert result['snapshot_status'] is None
