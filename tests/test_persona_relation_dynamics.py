"""Brain candidate checks through the real SQLite ledger and HTTP ingress."""
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette

from b1_protocol import digest
from persona_b1 import Conflict, FIELDS, PersonaB1Adapter, PersonaB1Store
from persona_b1_http import register_candidate_route
from persona_engine import PersonaStateEngine
from persona_inner_state import upgrade_policy_v2
from persona_relation_dynamics import LOCAL_REVISION, NEEDS, local_policy
from test_persona_b1_http import RouteRegistry
from test_persona_inner_state_v2 import WHEN, effect, envelope, fixture_policy, observation, source, span, turn
from test_stage03_memory_reactions import delivered, reaction


def when(hours, start=WHEN):
    return (datetime.fromisoformat(start) + timedelta(hours=hours)).isoformat(timespec='microseconds')


def relation_row(src, dimension, phase, excerpt=None):
    spans = [span(src, 'assistant', excerpt)]
    evidence = {'source_hash': digest(src), 'evidence_spans': spans}
    return {'dimension': dimension, 'phase': phase, 'evidence_spans': spans, 'evidence_hash': digest(evidence),
        'observation_id': 'rel:' + digest({**evidence, 'dimension': dimension, 'phase': phase})}


def setup(tmp, seed=None, *, base=None, revision=LOCAL_REVISION, candidate=True, start=WHEN):
    base = base or fixture_policy()
    cfg = local_policy(base, revision) if candidate else upgrade_policy_v2(base, revision)
    store = PersonaB1Store(tmp / 'candidate.sqlite', legacy_path=tmp / 'unused-legacy.sqlite')
    values = {key: .60 for key in FIELDS}
    values.update(seed or {})
    store.initialize('fixture', values, start, cfg)
    return PersonaB1Adapter(store, 'fixture', {}), cfg


def payload(src, stamp, observations=None, *, revision=LOCAL_REVISION, **options):
    result = envelope(src, options.pop('interactions', []), **options)
    result.update(occurred_at=stamp, policy_revision=revision)
    if observations is not None:
        result['relation_observations'] = observations
    return result


def send(adapter, src, stamp, observations=None, **options):
    return adapter.settle(payload(src, stamp, observations, **options), turn(src))['b1_receipt']['receipt']


def move(adapter, start, hours):
    return adapter.advance_clock(f'local-clock:{hours}', when(hours, start), LOCAL_REVISION)


def test_receipts_isolate_bad_and_same_source_relations_and_preserve_other_paths(tmp_path):
    adapter, cfg = setup(tmp_path)
    src = source('我们靠近一点。', '有点想你。手心也想贴近。现在我比较安心。')
    longing = relation_row(src, 'longing', 'rise', '有点想你')
    duplicate = relation_row(src, 'longing', 'rise', '有点想你。')
    security = relation_row(src, 'security', 'rise', '我比较安心')
    bad = {**relation_row(src, 'possessiveness', 'rise', '手心也想贴近'), 'evidence_hash': digest('foreign')}
    libido = relation_row(src, 'libido', 'rise', '手心也想贴近')
    conflicting = {**relation_row(src, 'libido', 'satisfy', '手心也想贴近。')}
    ordinary = observation(src)
    receipt = send(adapter, src, WHEN, [duplicate, longing, bad, libido, conflicting, security], interactions=[ordinary])
    results = receipt['relation_observations']
    assert sorted(r['status'] for r in results) == ['applied', 'applied',
        'conflicting_source_phases', 'conflicting_source_phases', 'rejected', 'same_cause_already_settled']
    assert receipt['after']['longing'] == pytest.approx(.62)
    assert receipt['after']['possessiveness'] == .6
    assert receipt['after']['libido'] == .6
    assert receipt['after']['security'] == pytest.approx(.68)
    assert receipt['after']['slow_security'] == pytest.approx(.61)
    assert receipt['after']['valence'] > .6
    assert 'tenderness' not in adapter.snapshot()['affect']
    assert 'session_defensiveness' not in adapter.store.numeric_at('fixture')['values']
    assert adapter.store.read('fixture')['values']['tenderness'] == .6
    assert all(set(rule).isdisjoint(set(NEEDS) | {'security'}) for rule in cfg['events'].values())


def test_reinterpreting_coarse_original_cannot_post_a_new_pulse(tmp_path):
    adapter, _ = setup(tmp_path)
    src = source('你现在怎么样？', '我想靠近你。另有一种说不上来、想再挨近一点的感觉。')
    first = relation_row(src, 'yearning', 'felt', '我想靠近你')
    second = relation_row(src, 'yearning', 'felt', '想再挨近一点')
    coarse = payload(src, WHEN, [first, second])
    assert len(adapter.settle(coarse, turn(src))['affect']['label_sources']['yearning']) == 2
    assert adapter.store.numeric_at('fixture')['values']['longing'] == .6
    assert adapter.settle(coarse, turn(src))['b1_receipt']['replayed']
    reunion = relation_row(src, 'longing', 'rise', '我想靠近你')
    with pytest.raises(Conflict, match='identity reused'):
        adapter.settle(payload(src, when(1), [reunion]), turn(src))
    with pytest.raises(ValueError, match='internal v2 completed-turn'):
        adapter.settle(payload(src, when(1), [reunion], relation_refinement_of=first['observation_id']), turn(src))
    fresh = source('你是说想回到那时，还是想见我？', '我现在想清楚了，我确实想下一次见你。', event_id='new-evidence')
    view = adapter.settle(payload(fresh, when(1), [relation_row(fresh, 'longing', 'rise')]), turn(fresh))
    assert view['affect']['longing'] == pytest.approx(.62)
    # Without a verified cross-turn causal link, the new evidence cannot
    # retrospectively edit either old coarse observation or its original receipt.
    assert len(view['affect']['label_sources']['yearning']) == 2
    assert view['affect']['label_until']['yearning'] == when(2)
    expired = move(adapter, WHEN, 2)
    assert 'yearning' not in expired['affect']['labels']
    assert {row['observation_id'] for row in expired['b1_receipt']['receipt']['relation_observations']
        if row['status'] == 'expired'} == {first['observation_id'], second['observation_id']}
    assert adapter.store.receipt('fixture', src['user_id'])['relation_observations'][0]['status'] == 'applied'


def test_same_turn_only_one_root_refines_and_independent_legacy_label_survives(tmp_path):
    adapter, _ = setup(tmp_path)
    src = source('你心里怎么样？', '我很想再见你。另有种想贴近但没想清的感受，我也真担心你。')
    precise = relation_row(src, 'longing', 'rise', '我很想再见你')
    same = relation_row(src, 'yearning', 'felt', '我很想再见你。')
    independent_text = '另有种想贴近但没想清的感受，我也真担心你'
    independent = observation(src, actor='assistant', target='owner',
        spans=[span(src, 'assistant', independent_text)])
    concern = relation_row(src, 'protective_drive', 'rise', '我也真担心你')
    ack = send(adapter, src, WHEN, [same, precise, concern], interactions=[independent],
        labels=['yearning'], feeling_sources=[{'label':'yearning', **{
            key: independent[key] for key in ('type','observation_id','evidence_hash')}}])
    assert ack['after']['longing'] == pytest.approx(.62)
    assert ack['after']['protective_drive'] == pytest.approx(.65)
    active = adapter.snapshot()['affect']['label_sources']['yearning']
    by_id = {row['observation_id']: row for row in ack['relation_observations']}
    assert by_id[same['observation_id']]['status'] == 'refined_by_supported_dimension'
    assert by_id[active[0]['observation_id']]['legacy_observation_id'] == independent['observation_id']
    assert len(active) == 1


@pytest.mark.parametrize('key', NEEDS)
def test_four_needs_have_independent_phases_and_time_recovery(tmp_path, key):
    adapter, config = setup(tmp_path)
    law = config['laws'][key]
    home = law['target']
    rules = config['relation_dynamics']['needs'][key]
    assert adapter.store.numeric_at('fixture')['values'][key] == .6
    for i, (phase, experience) in enumerate((
            ('rise', '这件事刚刚让我起了一个念头。'),
            ('ease', '听完近况，我之前的那份感受缓了。'),
            ('satisfy', '现在确实觉得够了，已经满足。'))):
        src = source('现在怎么样？', experience, event_id=f'need-{i}')
        ack = send(adapter, src, WHEN, [relation_row(src, key, phase)])
        assert ack['event_deltas'][key] == pytest.approx(rules[phase] * (1 if phase == 'rise' else -1))
        assert all(ack['event_deltas'][other] == 0 for other in NEEDS if other != key)
        assert ack['event_deltas']['slow_security'] == 0
    buffered = ack['after'][key]
    assert buffered < home
    assert move(adapter, WHEN, rules['buffer_hours'])['affect'][key] == pytest.approx(buffered)
    after_half_life = move(adapter, WHEN, rules['buffer_hours'] + law['half_life_hours'])['affect'][key]
    assert after_half_life == pytest.approx((buffered + home) / 2)


def test_stance_unknown_two_directions_and_no_action_or_pressure_pulse(tmp_path):
    base = fixture_policy()
    base['proof_policy'] = {'gains': {'her_ok': .05, 'first': .05, 'hard': .05, 'helped_ack': .05},
        'daily_factor': .9, 'high_threshold': .9, 'high_factor': .6}
    adapter, _ = setup(tmp_path, base=base)
    assert adapter.snapshot()['affect']['relational_stance'] is None
    src = source('我们今晚怎么办？', '今晚我来定，跟我走。')
    send(adapter, src, WHEN, [relation_row(src, 'relational_stance', 'lead')])
    assert adapter.snapshot()['affect']['relational_stance'] == .75
    assert 'relational_stance' not in adapter.store.numeric_at('fixture')['values']
    adapter._record_verified_operation({'scope': 'fixture', 'operation_id': 'verified-work-local',
        'status': 'succeeded', 'revision': 1, 'result_hash': digest('fixture proof'), 'proof': 'hard',
        'occurred_at': WHEN}, LOCAL_REVISION)
    assert adapter.snapshot()['affect']['relational_stance'] == .75
    assert move(adapter, WHEN, 3)['affect']['relational_stance'] == .625
    next_src = source('这一阵很累。', '我想慢下来，陪着你就好。', event_id='soften')
    send(adapter, next_src, when(3), [relation_row(next_src, 'relational_stance', 'soften'),
        relation_row(next_src, 'security', 'ease')])
    view = adapter.snapshot()['affect']
    assert view['relational_stance'] == .375
    assert view['security'] == pytest.approx(.52)
    assert view['stance_source_refs'][0]['event_id'] == 'soften'
    assert adapter.store.numeric_at('fixture')['values']['slow_security'] == .6


def test_qualified_memory_applies_independently_but_cannot_double_count_same_span(tmp_path):
    adapter, _ = setup(tmp_path)
    src = source('今天翻到了那张照片。', '我这一刻想见你，也突然担心你。')
    ctx, original_turn = delivered(src)
    rows = [reaction(src, ctx, dimension=key, strength='light') for key in ('longing', 'protective_drive', 'tenderness')]
    env = payload(src, WHEN, [relation_row(src, 'longing', 'rise', '我这一刻想见你')],
        memory_context=ctx, memory_reactions={'schema': 'memory-reactions-v1', 'observations': rows})
    eligible = {'state': 'available', 'eligible_fragment_ids': [ctx['fragments'][0]['fragment_id']],
        'delivery_receipt_id': ctx['delivery_receipt_id'], 'delivery_sha256': ctx['delivery_sha256']}
    ack = adapter.settle(env, original_turn, memory_validation=eligible)['b1_receipt']['receipt']
    status = {row['dimension']: row['status'] for row in ack['memory_reactions']['observations']}
    assert status == {'longing': 'same_cause_already_settled', 'protective_drive': 'applied',
                      'tenderness': 'retired_dimension'}
    assert ack['after']['longing'] == pytest.approx(.62)
    assert ack['after']['protective_drive'] == pytest.approx(.62)
    assert adapter.store.read('fixture')['values']['tenderness'] == .6


def test_authenticated_real_ingress_r6_and_r5_policy_fence(tmp_path):
    adapter, _ = setup(tmp_path / 'r6')
    old, _ = setup(tmp_path / 'r5', candidate=False, revision='r5')
    with pytest.raises(Conflict, match='freshly initialized'):
        old.store.upgrade_policy('fixture', local_policy(fixture_policy()), expected_revision=0)
    with pytest.raises(Conflict, match='versioned'):
        src = source()
        send(old, src, WHEN, [relation_row(src, 'yearning', 'felt')], revision='r5')

    engine = PersonaStateEngine({'buckets_dir': str(tmp_path), 'persona': {'profile_id': 'fixture', 'mode': 'disabled'}},
        str(tmp_path / 'legacy.sqlite'), b1_candidate=adapter)
    routes = RouteRegistry()
    register_candidate_route(routes, lambda: engine, lambda request: request.headers.get('Authorization') == 'Bearer local')
    async def check():
        src = source('怎么了？', '说不清，我现在想离你近点。')
        body = {'envelope': payload(src, WHEN, [relation_row(src, 'yearning', 'felt')]), 'turn': turn(src)}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=routes.routes)),
                base_url='http://local-brain') as client:
            endpoint = '/internal/persona/b1/candidate'
            assert (await client.post(endpoint, json=body)).status_code == 401
            auth = {'Authorization': 'Bearer local'}
            resp = await client.post(endpoint, json=body, headers=auth)
            assert resp.status_code == 200, resp.text
            view = resp.json()['projection']
            assert view['relation_profile'] == 'relation-dynamics-v1'
            assert view['affect']['labels'] == ['yearning']
            assert datetime.fromisoformat(view['affect']['label_until']['yearning']) == datetime.fromisoformat(when(2))
            assert view['affect']['relational_stance'] is None
            assert 'tenderness' not in view['affect']
            assert '想离你近点' not in resp.text
            clock = await client.post('/internal/persona/b1/clock', json={'scope': 'fixture',
                'occurred_at': when(3), 'policy_revision': LOCAL_REVISION, 'growth_frames': []}, headers=auth)
            assert clock.status_code == 200, clock.text
            assert clock.json()['projection']['affect']['labels'] == []
            saved = await client.get('/internal/persona/b1/numeric', params={'scope': 'fixture'}, headers=auth)
            assert saved.status_code == 200
            assert 'relational_stance' not in saved.json()['snapshot']['values']
    asyncio.run(check())


R7849_DATA = Path(__file__).resolve().parents[3] / 'staging' / 'emotion-wording-prompt-review-20261119' / '全部内态数值与波动.json'


@pytest.mark.skipif(not R7849_DATA.exists(), reason='optional local readback capture is outside the Brain checkout')
def test_service_replay_from_same_real_r7849_pre_event_state(tmp_path):
    capture = json.loads(R7849_DATA.read_text(encoding='utf-8-sig'))
    real, = (r for r in capture['brain']['points'] if r['rev'] == 7849)
    seed = {key: real['before'][key] + real['time_deltas'][key] for key in FIELDS}
    start = real['at']
    base = fixture_policy()
    base['events']['intimacy'] = {'valence': effect(.24), 'arousal': effect(.32), 'cling': effect(.12)}
    for key, factor, floor in (('longing', .45, .287), ('possessiveness', .45, .287),
            ('libido', .55, .175), ('protective_drive', .15, .273)):
        base['events']['intimacy'][key] = {**effect(relief=factor), 'floor': floor}
    base['cling_relief'] = {'threshold': .10, 'factor': .50}
    base['slow_axis_events']['intimacy'] = {'slow_security': .03}
    old_revision = 'r5-comparable-excerpt'
    old, _ = setup(tmp_path / 'old', seed, base=base, revision=old_revision, candidate=False, start=start)
    # The capture has the original numbers/receipt kind, not the chat text.
    # Nearby semantics are an explicit test fixture, not retrieved transcripts.
    old_src = source('今晚陪我。', '我们刚刚的亲密让我很满足。')
    old_ack = send(old, old_src, start, revision=old_revision, interactions=[observation(old_src, 'intimacy')])
    for key in (*NEEDS, 'cling', 'valence', 'arousal', 'slow_security'):
        assert old_ack['after'][key] == pytest.approx(real['v'][key], abs=1e-12)

    branches = {}
    for phase in ('pass', 'rise', 'satisfy'):
        adapter, _ = setup(tmp_path / phase, seed, base=base, start=start)
        advanced = move(adapter, start, 6)['b1_receipt']['receipt']
        assert advanced['after']['longing'] == pytest.approx(.71)
        src = source('今天想起我们那次聊天。', {
            'pass': '我也读到那段共同回忆了。',
            'rise': '读到这里，我突然很想再见你。',
            'satisfy': '刚才那件我惦记着想聊完的事终于说完了，我这会儿已经很满足。',
        }[phase])
        rows = [] if phase == 'pass' else [relation_row(src, 'longing', phase)]
        ack = send(adapter, src, when(6, start), rows)
        for key in NEEDS:
            if key != 'longing':
                assert ack['event_deltas'][key] == 0
        branches[phase] = adapter, ack
    assert branches['pass'][1]['after']['longing'] == pytest.approx(.71)
    assert branches['rise'][1]['after']['longing'] == pytest.approx(.73)
    buffered, ack = branches['satisfy']
    assert ack['after']['longing'] == pytest.approx(.47)
    assert move(buffered, start, 7)['affect']['longing'] == pytest.approx(.47)
    replay = buffered.settle(payload(src, when(6, start), rows), turn(src))
    assert replay['b1_receipt']['replayed']
    assert buffered.store.read('fixture')['plateaus']['longing'] == when(8, start)
    independent = source('我想说一件新事。', '听完这件事，我现在又想和你见面。', event_id='independent')
    assert send(buffered, independent, when(7.5, start), [relation_row(independent, 'longing', 'rise')])['after']['longing'] == pytest.approx(.49)
    assert move(buffered, start, 14)['affect']['longing'] == pytest.approx(.545)
