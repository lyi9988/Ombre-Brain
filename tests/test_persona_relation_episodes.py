"""Exercise the actual Brain SQLite authority and authenticated HTTP projection."""
import asyncio
from datetime import datetime
import json
import sqlite3

import httpx
import pytest
from starlette.applications import Starlette

from b1_protocol import digest
from persona_b1 import FIELDS, Conflict, PersonaB1Adapter, PersonaB1Store, policy
from persona_b1_http import register_candidate_route
from persona_engine import PersonaStateEngine
from persona_inner_state import TENDENCIES, upgrade_policy_v2
from persona_projection import safe_persona_projection
from persona_relation_episodes import CORE, KEYS, LOCAL_REVISION, PROFILE, SHORT, local_policy
from test_persona_b1_http import RouteRegistry
from test_persona_inner_state_v2 import WHEN, empathy, fixture_policy, mention, observation, source, turn
from test_persona_relation_dynamics import payload, relation_row, when
from test_stage03_memory_reactions import delivered, reaction


def prepare(tmp_path, config=None):
    store = PersonaB1Store(tmp_path / 'local.sqlite', legacy_path=tmp_path / 'unopened-legacy.sqlite')
    config = config or local_policy(fixture_policy())
    store.initialize('fixture', {key: None if key == 'security' else .61 for key in FIELDS}, WHEN, config)
    return PersonaB1Adapter(store, 'fixture', {})


def send(adapter, src, rows=None, hours=0, **options):
    env = payload(src, when(hours), rows, revision=LOCAL_REVISION, **options)
    return adapter.settle(env, turn(src))


def identity(row, src, episode_id):
    return {**row, 'episode_id': episode_id,
            'observation_id': 'rel:' + digest({'source_hash': digest(src), 'dimension': row['dimension'],
                'phase': row['phase'], 'evidence_spans': row['evidence_spans'], 'episode_id': episode_id})}


def advance(adapter, hours):
    return adapter.advance_clock(f'clock-{hours}', when(hours), LOCAL_REVISION)


def by_id(state):
    return {row['id']: row for row in state['episodes']}


def diagnostics(state):
    return state['b1_receipt']['receipt']['relation_observations']


def test_idle_seed_has_no_ghost_values_laws_or_fatigue_and_tendencies_remain_editable(tmp_path):
    adapter = prepare(tmp_path)
    store = adapter.store
    view = advance(adapter, 84)
    assert view['episodes'] == []
    assert 'fatigue' not in view
    assert set(view['affect']) & set(FIELDS) == {'valence', 'arousal', 'security'}
    assert view['affect']['security'] is None and view['affect']['relational_stance'] is None
    assert view['slow_axes']['confidence'] == .61
    assert set(view['tendencies']) == set(TENDENCIES)
    assert all(value is None for value in view['tendencies'].values())
    revision = store.read('fixture')['revision']
    store.update_tendencies('fixture', 'review-1', {'social_recovery_rate': 0.0},
        {'social_recovery_rate': {'unit': '0..1', 'source_ref': 'review-1'}},
        expected_revision=revision, occurred_at=when(84))
    numeric = store.numeric_at('fixture')
    assert set(numeric['values']) == set(CORE) | set(TENDENCIES) | {'slow_mood', 'relational_stance'}
    assert numeric['values']['social_recovery_rate'] == 0.0
    with sqlite3.connect(store.path) as db:
        raw = db.execute('SELECT state,policy FROM b1_state').fetchone()
        state, config = (json.loads(value) for value in raw)
        assert 'fatigue' not in state and set(state['values']) == set(CORE)
        assert set(config['laws']) == set(CORE) and not config['fatigue_policy']
        assert state['relation_episodes'] == {}
        assert db.execute('SELECT count(*) FROM b1_pressure_frames').fetchone()[0] == 0


def test_independent_concerns_keep_origins_across_days_restart_and_exact_resolution(tmp_path):
    adapter = prepare(tmp_path)
    first = source('近况还没定。', '我记着你复诊的事。我也惦记你一个人回夜路。', 'u1')
    new = relation_row(first, 'protective_drive', 'unresolved', '记着你复诊的事')
    dup = relation_row(first, 'protective_drive', 'rise', '我记着你复诊的事')
    other = relation_row(first, 'protective_drive', 'unresolved', '我也惦记你一个人回夜路')
    result = send(adapter, first, [new, dup, other])
    assert sorted(row['status'] for row in diagnostics(result)) == ['applied', 'applied', 'same_cause_already_settled']
    originals = {row['observation_id']: row['episode_id'] for row in diagnostics(result) if row['status'] == 'applied'}
    old_id, other_id = originals[new['observation_id']], originals[other['observation_id']]
    assert len(result['episodes']) == 2
    assert by_id(result)[old_id]['origin']['evidence_spans'] == new['evidence_spans']
    assert send(adapter, first, [new, dup, other])['b1_receipt']['replayed'] is True

    late = advance(adapter, 72)
    assert by_id(late)[old_id]['status'] == 'unresolved'
    assert by_id(late)[old_id]['intensity'] == pytest.approx(by_id(result)[old_id]['intensity'] / 2)
    reopened = PersonaB1Adapter(PersonaB1Store(adapter.store.path, legacy_path=tmp_path / 'unused.sqlite'), 'fixture', {})
    assert by_id(reopened.snapshot()) == by_id(late)

    no_link = source('检查结束了。', '复诊没事，我松了口气。', 'u2')
    refused = send(reopened, no_link, [relation_row(no_link, 'protective_drive', 'satisfy')], hours=72)
    assert diagnostics(refused)[0]['status'] == 'episode_target_required'
    assert by_id(refused) == by_id(late)
    resolved = source('报告没问题。', '你复诊的结果好，我放心了。', 'u3')
    relief = identity(relation_row(resolved, 'protective_drive', 'satisfy'), resolved, old_id)
    satisfied = send(reopened, resolved, [relief], hours=72)
    assert by_id(satisfied)[old_id]['status'] == 'sated'
    assert datetime.fromisoformat(by_id(satisfied)[old_id]['buffer_until']) == datetime.fromisoformat(when(74))
    assert by_id(satisfied)[other_id] == by_id(late)[other_id]
    assert send(reopened, resolved, [relief], hours=72)['b1_receipt']['replayed'] is True
    repeated = source('刚刚那次复诊。', '复诊的确已经放心了。', 'u4')
    again = send(reopened, repeated,
        [identity(relation_row(repeated, 'protective_drive', 'satisfy'), repeated, old_id)], hours=72)
    assert diagnostics(again)[0]['status'] == 'already_sated'
    assert by_id(again)[old_id] == by_id(satisfied)[old_id]
    new_event = source('医院请我回来复查。', '既然要复查，我又惦记起来。', 'u5')
    renewed = send(reopened, new_event,
        [identity(relation_row(new_event, 'protective_drive', 'unresolved'), new_event, old_id)], hours=72.5)
    assert by_id(renewed)[old_id]['status'] == 'unresolved'
    assert 'buffer_until' not in by_id(renewed)[old_id]
    assert by_id(renewed)[old_id]['origin'] == by_id(late)[old_id]['origin']
    assert len(by_id(renewed)[old_id]['source_refs']) == 3


def test_short_feelings_fade_and_coarse_row_does_not_create_fine_scores(tmp_path):
    adapter = prepare(tmp_path)
    src = source('你还想说什么？', '我不满。我失落。想黏你。也想被你偏爱。说不清，还想靠近。',
                 'u-feelings')
    rows = [relation_row(src, key, 'rise', word) for key, word in zip(SHORT,
            ('不满', '失落', '想黏你', '想被你偏爱'))]
    rows += [relation_row(src, 'yearning', 'felt', '还想靠近')]
    out = send(adapter, src, rows)
    assert {item['key'] for item in out['episodes']} == set(SHORT)
    assert out['affect']['labels'] == ['yearning']
    midpoint = advance(adapter, 5)
    assert 'yearning' not in midpoint['affect']['labels']
    assert len(midpoint['episodes']) < 4
    gone = advance(adapter, 50)
    assert gone['episodes'] == []
    assert {r['status'] for r in diagnostics(gone)} == {'expired'}
    assert {r['status'] for r in adapter.store.receipt('fixture', src['user_id'])['relation_observations']} == {'applied'}


def test_linked_still_unresolved_confirms_without_pulse_but_new_rise_can_raise(tmp_path):
    adapter = prepare(tmp_path)
    first = source('检查结果还没定。', '复诊结果出来前，我还惦记着。', 'u-pending')
    opened = send(adapter, first, [relation_row(first, 'protective_drive', 'unresolved')])
    episode_id = opened['episodes'][0]['id']
    quiet = by_id(advance(adapter, 24))[episode_id]

    next_turn = source('报告仍然没到。', '这件事果然还没定。', 'u-still-pending')
    still = identity(relation_row(next_turn, 'protective_drive', 'unresolved'), next_turn, episode_id)
    unchanged = send(adapter, next_turn, [still], hours=24)
    receipt, = diagnostics(unchanged)
    assert receipt['status'] == 'still_unresolved'
    assert receipt['episode_id'] == episode_id
    assert receipt['before'] == receipt['after'] == quiet['intensity']
    assert receipt['source_ref']['observation_id'] == still['observation_id']
    assert by_id(unchanged)[episode_id] == quiet
    assert adapter.store.receipt('fixture', 'u-still-pending')['relation_observations'] == [receipt]
    assert send(adapter, next_turn, [still], hours=24)['b1_receipt']['replayed']

    different = source('医院发现了新问题。', '听到这个新发现，我一下又担心起来。', 'u-new-risk')
    rise = identity(relation_row(different, 'protective_drive', 'rise'), different, episode_id)
    raised = send(adapter, different, [rise], hours=24)
    impact, = diagnostics(raised)
    assert impact['status'] == 'applied' and impact['before'] == quiet['intensity']
    assert impact['after'] > impact['before']
    after = by_id(raised)[episode_id]
    assert after['status'] == 'unresolved'
    assert after['updated_at'] != quiet['updated_at']
    assert len(after['source_refs']) == len(quiet['source_refs']) + 1


def test_cross_conversation_target_and_wrong_hash_do_not_block_a_valid_feeling(tmp_path):
    adapter = prepare(tmp_path)
    original = source('还没能见面。', '我想你，现在还见不到。', 'u-original')
    prior = send(adapter, original, [relation_row(original, 'longing', 'unresolved')])
    old_id = prior['episodes'][0]['id']
    cross = source('我来了。', '总算见到了。我还是有点紧张。', 'u-cross')
    cross['conversation_key'] = digest('conversation-2')
    wired = {**turn(cross), 'conversation_id': 'conversation-2'}
    foreign = identity(relation_row(cross, 'longing', 'satisfy', '总算见到了'), cross, old_id)
    wrong_hash = {**relation_row(cross, 'anger', 'rise', '还是有点紧张'), 'episode_id': old_id}
    valid = relation_row(cross, 'relational_stance', 'lead', '总算见到了')
    env = payload(cross, WHEN, [foreign, wrong_hash, valid], revision=LOCAL_REVISION)
    rejected = adapter.settle(env, wired)
    assert rejected['episodes'] == prior['episodes']
    assert rejected['affect']['relational_stance'] == .75
    assert {row['reason'] for row in diagnostics(rejected) if row['status']=='rejected'} == {
        'episode_target_not_bound', 'observation_identity_mismatch'}


def test_verified_help_targets_the_empathy_incident_and_old_label_remains_source_bound(tmp_path):
    config = local_policy(fixture_policy())
    config['proof_policy'] = {'gains': {'her_ok': .03, 'first': .03, 'hard': .04, 'helped_ack': .02},
        'daily_factor': .6, 'high_threshold': .75, 'high_factor': .5}
    adapter = prepare(tmp_path, config)
    ids = []
    for event_id, who in (('u-lin','小林'), ('u-wang','小王')):
        src = source('你怎么看？', f'我心疼{who}，他的事情我没放下。', event_id)
        claim = observation(src, 'empathy', actor='assistant', target='owner', subject=mention(src, 'assistant', who))
        explicit = relation_row(src, 'grieve', 'rise', f'我心疼{who}')
        details = empathy(claim, f'person-{event_id}', f'incident-{event_id}')
        ack = send(adapter, src, [explicit], interactions=[claim], empathy_details=[details])
        assert diagnostics(ack)[0]['status'] == 'same_cause_already_settled'
        ids.append(diagnostics(ack)[0]['episode_id'])

    old = source('你刚刚还好吗？', '我有点不满。', 'u-anger')
    claim = observation(old, 'conflict', actor='assistant', target='owner')
    old_label = {'observation_id': claim['observation_id'], 'label': 'angry',
                 'type': claim['type'], 'evidence_hash': claim['evidence_hash']}
    before = send(adapter, old, interactions=[claim], labels=['angry'], feeling_sources=[old_label])
    assert len(before['episodes']) == 3
    assert 'angry' not in before['affect']['labels']
    anger = next(item for item in before['episodes'] if item['key']=='anger')
    assert anger['origin']['source_hash'] == digest(old)

    evidence = {'scope': 'fixture', 'operation_id': 'verified-help-1', 'status': 'succeeded',
        'proof': 'helped_ack', 'revision': 0, 'result_hash': digest('verified success'), 'occurred_at': WHEN,
        'helped': {'subject_ref': 'person-u-lin', 'incident_ref': 'incident-u-lin', 'sub': '心疼'}}
    adapter.result_resolver = lambda _result: evidence
    after = adapter.record_operation('verified-help-1', LOCAL_REVISION)
    assert by_id(after)[ids[0]]['intensity'] < by_id(before)[ids[0]]['intensity']
    assert by_id(after)[ids[1]] == by_id(before)[ids[1]]
    assert by_id(after)[anger['id']] == anger
    assert adapter.record_operation('verified-help-1', LOCAL_REVISION)['b1_receipt']['replayed']


def test_historical_policy_and_frames_keep_their_values_without_creating_episodes(tmp_path):
    old_policy = upgrade_policy_v2(fixture_policy(), 'older-r5')
    adapter = prepare(tmp_path, old_policy)
    store = adapter.store
    source_old = source(event_id='older')
    body = payload(source_old, WHEN, revision='older-r5', interactions=[observation(source_old)])
    prior = adapter.settle(body, turn(source_old))['b1_receipt']['receipt']
    adapter.advance_clock('precutover', when(1), 'older-r5')
    new = store.upgrade_policy('fixture', local_policy(old_policy), expected_revision=2)
    assert new['migrated']
    assert adapter.snapshot()['episodes'] == []
    assert 'longing' not in store.numeric_at('fixture')['values']
    assert store.numeric_at('fixture', WHEN)['values']['longing'] == .61
    assert store.receipt('fixture', 'older') == prior
    assert adapter.settle(body, turn(source_old))['b1_receipt']['replayed']
    with pytest.raises(Conflict):
        store.upgrade_policy('fixture', old_policy, expected_revision=3)


def test_verified_memory_eases_only_its_group_and_verified_http_exposes_safe_origins(tmp_path):
    adapter = prepare(tmp_path)
    engine = PersonaStateEngine({'buckets_dir': str(tmp_path), 'persona': {'profile_id': 'fixture', 'mode': 'disabled'}},
        str(tmp_path / 'legacy.sqlite'), b1_candidate=adapter)
    routes = RouteRegistry()

    async def resolve(context, original):
        return {'state': 'available', 'eligible_fragment_ids': [context['fragments'][0]['fragment_id']],
                'delivery_receipt_id': context['delivery_receipt_id'], 'delivery_sha256': context['delivery_sha256']}

    register_candidate_route(routes, lambda: engine, lambda request: request.headers.get('Authorization') == 'Bearer local',
        memory_context_resolver=resolve)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=routes.routes)),
                base_url='http://brain-local') as client:
            ep_path = '/internal/persona/b1/candidate'
            token = {'Authorization': 'Bearer local'}
            first_ids = {}
            for i, (group, direction, effect) in enumerate((
                    ('own-group', 'increase', 'evoke'), ('other-group', 'increase', 'evoke'),
                    ('wrong-group', 'decrease', 'ease'), ('own-group', 'decrease', 'ease'))):
                src = source('我提到那段回忆。', '这次想起时，心里有了变化。', f'u-play-{i}')
                ctx, original = delivered(src, group=group)
                row = reaction(src, ctx, dimension='longing', direction=direction, effect=effect)
                body = {'envelope': payload(src, WHEN, revision=LOCAL_REVISION, memory_context=ctx,
                    memory_reactions={'schema': 'memory-reactions-v1', 'observations': [row]}), 'turn': original}
                if i == 0:
                    assert (await client.post(ep_path,json=body)).status_code == 401
                response = await client.post(ep_path, json=body, headers=token)
                assert response.status_code == 200, response.text
                assert '这次想起时' not in response.text
                projection = response.json()['projection']
                assert projection['relation_profile'] == PROFILE
                assert 'drive' not in projection and 'fatigue' not in projection
                receipt = response.json()['b1_receipt']['receipt']['memory_reactions']['observations'][0]
                if i < 2:
                    assert receipt['status'] == 'applied'
                    first_ids[group] = receipt['episode_id']
                    episode = by_id(projection)[receipt['episode_id']]
                    assert episode['origin']['source_hash'] == digest(src)
                    assert episode['source_refs'][0]['observation_id'] == row['reaction_id']
                elif i == 2:
                    assert receipt['status'] == 'episode_target_required'
                    assert len(projection['episodes']) == 2
                else:
                    assert receipt['episode_id'] == first_ids['own-group'] and receipt['after'] < receipt['before']
                    assert by_id(projection)[first_ids['other-group']]['intensity'] == .05
            snapshot = await client.get('/internal/persona/b1/numeric', params={'scope': 'fixture'}, headers=token)
            assert snapshot.status_code == 200
            assert not (set(snapshot.json()['snapshot']['values']) & (set(KEYS) | {'fatigue'}))
    asyncio.run(check())


def test_fatigue_seed_fence_rejects_r7_named_old_policy_and_fatigue_modulation():
    old = upgrade_policy_v2(fixture_policy(), LOCAL_REVISION)
    fatigue = {'initial': .7, 'maximum': 1, 'build_per_hour': .15,
        'recover_per_hour': .2, 'pressure_threshold': .3}
    old['fatigue_policy'] = fatigue
    with pytest.raises(ValueError, match='requires the episode profile'):
        policy(old)
    current = local_policy(old)
    assert policy(current)['fatigue_policy'] == {}
    current['fatigue_policy'] = fatigue
    with pytest.raises(ValueError, match='Plan B episode policy'):
        policy(current)


def test_regular_read_and_pre_reply_expose_same_unresolved_episode_without_write(tmp_path):
    adapter = prepare(tmp_path)
    engine = PersonaStateEngine({'buckets_dir': str(tmp_path), 'persona': {'profile_id': 'fixture', 'mode': 'disabled'}},
        str(tmp_path / 'legacy.sqlite'), b1_candidate=adapter)
    src = source('检查还没出结果。', '等你这份复诊报告，我还记挂着。', 'u-checkup')
    send(adapter, src, [relation_row(src, 'protective_drive', 'unresolved')])
    advance(adapter, 72)
    checkpoint = adapter.store.read('fixture')
    normal = engine.get_current_state('session-main')
    before_reply = asyncio.run(engine.build_pre_reply_guidance('session-main', '结果还没来。'))
    safe = safe_persona_projection(normal)
    assert safe['episodes'] == normal['episodes'] == before_reply['episodes']
    assert len(safe['episodes']) == 1 and safe['episodes'][0]['status'] == 'unresolved'
    assert safe['episodes'][0]['intensity'] == pytest.approx(.225)
    assert 'fatigue' not in normal and 'fatigue' not in safe
    assert adapter.store.read('fixture') == checkpoint
