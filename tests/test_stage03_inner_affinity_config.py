"""Only the new setting, the original overlay, and the real signal consumer."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from memory_system_config import (ConfigurationError, configuration, merge_config, preview,
    read_overlay, save_configuration, update_overlay, validate_values)


def test_defaults_and_registered_dimensions_are_exposed_without_expanding_owner_weights():
    from memory_inner_affinity import DEFAULT_WEIGHTS
    result = configuration({})
    assert result['validation']['valid']
    assert result['values']['recall']['inner_state_affinity'] == {'enabled': True, 'max_bonus': .05, 'weights': {}}
    metadata = result['bindings']['inner_state_affinity']
    assert len(metadata['dimension_keys']) == 35 and set(metadata['dimension_keys']) == set(DEFAULT_WEIGHTS)
    assert sum(value == 1 for value in metadata['default_weights'].values()) == 21
    assert metadata['default_weights']['curiosity'] == 1 and metadata['default_weights']['libido'] == 1
    assert metadata['default_weights']['energy'] == 0 and metadata['default_weights']['slow_security'] == 0
    assert metadata['default_weights']['rest'] == 0


@pytest.mark.parametrize('maximum', [.050001, .08])
def test_bonus_above_approved_maximum_is_rejected_explicitly(maximum):
    values = configuration({})['values']
    values['recall']['inner_state_affinity']['max_bonus'] = maximum
    with pytest.raises(ConfigurationError) as failure:
        validate_values(values)
    assert failure.value.code == 'inner_affinity_bonus_out_of_range'


def test_original_overlay_save_reload_zero_values_and_older_client_preserve_owner_policy(tmp_path):
    path = tmp_path / 'config.runtime.yaml'
    base = {'gateway': {'inject_total_budget': 10000}, 'reflection': {'model': 'original-model'}}
    update_overlay(path, lambda old: {'other_owner_settings': {'keep': 'unchanged'}})
    current = configuration(base)
    values = deepcopy(current['values'])
    wanted = {'enabled': False, 'max_bonus': 0, 'weights': {'curiosity': 0, 'libido': .8, 'energy': 0}}
    values['recall']['inner_state_affinity'] = wanted
    values['recall']['signal_bindings'] = [{'id': 'curiosity', 'producer': 'guyan.inner_state', 'key': 'curiosity', 'max_age_seconds': None}]
    values['recall']['local_match_weights']['keyword_match'] = .23
    payload = {'expected_revision': current['revision'], 'request_id': 'owner-inner-affinity', 'values': values}
    saved, _ = save_configuration(path, base, payload)
    reloaded = configuration(merge_config(base, read_overlay(path)))
    assert reloaded['values']['recall']['inner_state_affinity'] == wanted
    assert reloaded['values']['recall']['local_match_weights']['keyword_match'] == .23
    assert reloaded['values']['limits']['inject_total_budget'] == 10000
    assert read_overlay(path)['other_owner_settings'] == {'keep': 'unchanged'}
    assert base['reflection']['model'] == 'original-model'
    assert saved['revision'] == reloaded['revision']
    assert save_configuration(path, base, payload)[0]['replayed']
    older = deepcopy(reloaded['values'])
    older['recall'].pop('inner_state_affinity')
    again, _ = save_configuration(path, base, {'expected_revision': saved['revision'], 'request_id': 'older-client', 'values': older})
    assert again['values']['recall']['inner_state_affinity'] == wanted


@pytest.mark.parametrize('policy', [None, {'enabled': None}, {'enabled': 1}, {'max_bonus': None},
    {'max_bonus': -.01}, {'max_bonus': 1.01}, {'max_bonus': True}, {'weights': None},
    {'weights': {'not_registered': .5}}, {'weights': {'libido': None}}, {'weights': {'libido': float('nan')}},
    {'weights': {'curiosity': 1.01}}, {'weights': {'curiosity': True}}])
def test_invalid_inner_policy_is_rejected_without_coercion(policy):
    values = configuration({})['values']
    values['recall']['inner_state_affinity'] = policy
    with pytest.raises(ConfigurationError):
        validate_values(values)


def test_new_producer_known_zero_and_unknown_null_reach_existing_signal_consumer():
    from memory_affinity import resolve_signals
    from recall_signal_protocol import INNER_DIMENSIONS, snapshot_from_inner
    values = configuration({})['values']
    values['recall']['signal_bindings'] = [{'id': key, 'producer': 'guyan.inner_state', 'key': key, 'max_age_seconds': None}
        for key in ('curiosity', 'libido')]
    settings = validate_values(values)
    at = '2026-10-26T10:00:00+00:00'
    dimensions = {key: {'value': None, 'status': 'unknown', 'reason': 'not_measured', 'unit': '0..1', 'owner': owner, 'kind': kind}
        for key, (owner, kind) in INNER_DIMENSIONS.items()}
    dimensions['curiosity'].update(value=0, status='known', reason=None)
    snapshot = snapshot_from_inner({'schema': 'inner-state-v1', 'availability': 'complete', 'subject_id': 'synthetic',
        'checkpoint_id': 'config-consumer-check', 'rules_version': 'existing-rules', 'revision': {'aiz': 1, 'brain': 1},
        'settled_at': at, 'dimensions': dimensions})
    resolved = resolve_signals(settings, [snapshot], datetime.fromisoformat(at))
    assert resolved['curiosity']['state'] == 'known' and resolved['curiosity']['value'] == 0
    assert resolved['libido']['state'] == 'unknown' and resolved['libido']['value'] is None
    values['features'] = [{'id': 'signal_value', 'value_type': 'number', 'source': 'runtime', 'input_key': 'signal_libido', 'missing': 'unknown'}]
    values['formulas'] = [{'id': 'final', 'expression': 'signal_value', 'weights': {}}]
    assert preview(values, {})['formulas']['final'] == {'state': 'unknown', 'value': None}
