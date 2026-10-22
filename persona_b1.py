"""Explicit offline Brain B1 owner; no production configuration or model calls.

Existing short affect plus distinct grieve/anger/favored/libido and transient
cling are not stable personality/relationship writes. Every numeric policy is
explicit; fixtures are not approved personal calibration.
"""
from contextlib import closing
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,math,os,sqlite3
from pathlib import Path

FIELDS=('valence','arousal','tenderness','possessiveness','longing','security','protective_drive','session_defensiveness','libido','grieve','anger','favored','cling','slow_security','slow_confidence')
TYPES=frozenset(('companionship','affection','intimacy','sharing','discovery','task_progress','reflection','conflict','loss','reconciliation','slighted','empathy','helped','intrigued'))
LABELS=frozenset(('warm_neutral','warm_playful','neutral','happy','sad','calm','curious','tired','playful','mixed-playful','longing','affectionate','tender','jealous','hurt','angry','clingy','secure','insecure','conflicted','guarded','focused','desire','sensual','possessive','intimate','yearning','attached','vulnerable','disappointed'))
class Conflict(ValueError):pass

from b1_protocol import canonical,digest
def ident(x):
    if not isinstance(x,str) or not x.strip() or len(x)>256:raise ValueError('bounded identity required')
    return x

def number(x,lo=0,hi=1):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi:raise ValueError('finite bounded number required')
    return float(x)
def at(x):
    v=x if isinstance(x,datetime) else datetime.fromisoformat(x)
    if v.tzinfo is None:raise ValueError('timezone required')
    return v.astimezone(timezone.utc).isoformat(timespec='microseconds')
def values(x):
    if not isinstance(x,dict) or set(x)!=set(FIELDS):raise ValueError('complete state with explicit unknowns required')
    return {k:None if v is None else number(v) for k,v in x.items()}
def policy(x):
    x=deepcopy(x)
    if set(x)-{'conflict_policy','empathy_policy','light_policy','presence_policy','security_grief_policy','correction_confidence_delta','growth_policy','rhythm_policy','fatigue_policy','pressure_policy','display_policy'}!={'revision','laws','events','repeat_hours','repeat_factor','couplings','subtype_policy','mixed_policy','cling_relief','sleep_multipliers','initial_sleeping','mood_policy','presentation','slow_axis_events','proof_policy'}:raise ValueError('explicit policy required')
    x.setdefault('display_policy',{})
    if x['display_policy']:
        d=x['display_policy']
        if d.get('mode')=='coexist':
            if set(d)!={'mode'}:raise ValueError('coexist needs no thresholds')
        elif d.get('mode')=='primary':
            if set(d)!={'mode','lit_min_valence','fresh_minutes','fresh_lit_priority','aged_lit_priority','cause_max_valence','phase_max_valence','phase_priority'}:raise ValueError('explicit display arbitration parameters')
            for key in ('lit_min_valence','cause_max_valence','phase_max_valence'):number(d[key],0,1)
            for key in ('fresh_lit_priority','aged_lit_priority','phase_priority'):number(d[key],0,100)
            number(d['fresh_minutes'],0,1440)
        else:raise ValueError('known display mode required')
    x.setdefault('pressure_policy',{})
    if x['pressure_policy']:
        from b1_pressure_protocol import validate_policy as validate_pressure
        validate_pressure(x['pressure_policy'])
    x.setdefault('fatigue_policy',{})
    if x['fatigue_policy']:
        r=x['fatigue_policy']
        if set(r)!={'initial','maximum','build_per_hour','recover_per_hour','pressure_threshold'}:raise ValueError('explicit fatigue seed and dynamics')
        for k in ('maximum','build_per_hour','recover_per_hour','pressure_threshold'):number(r[k])
        if r['initial'] is not None:number(r['initial'],0,r['maximum'])
    x.setdefault('rhythm_policy',{})
    if x['rhythm_policy']:
        r=x['rhythm_policy'];ordinary={'longing','possessiveness','protective_drive','libido'}
        if set(r)!={'offset_minutes','night_start','night_end','dawn_start','dawn_end','night_multipliers','dawn_freeze'}:raise ValueError('explicit rhythm boundaries required')
        number(r['offset_minutes'],-840,840)
        for k in ('night_start','night_end','dawn_start','dawn_end'):
            if isinstance(r[k],bool) or not isinstance(r[k],int) or not 0<=r[k]<1440:raise ValueError('minute of day required')
        if not isinstance(r['night_multipliers'],dict) or not set(r['night_multipliers'])<=ordinary or not isinstance(r['dawn_freeze'],list) or not set(r['dawn_freeze'])<=ordinary:raise ValueError('ordinary drive rhythm only')
        for v in r['night_multipliers'].values():number(v,0,100)
    x.setdefault('growth_policy',{})
    if x['growth_policy']:
        from b1_growth_protocol import validate_policy
        validate_policy(x['growth_policy'])
    x.setdefault('correction_confidence_delta',None)
    if x['correction_confidence_delta'] is not None:number(x['correction_confidence_delta'],-1,0)
    x.setdefault('presence_policy',{});x.setdefault('security_grief_policy',{})
    if x['presence_policy']:
        r=x['presence_policy']
        if set(r)!={'possessiveness_min','cling_min','absence_hours'}:raise ValueError('complete explicit presence policy')
        number(r['possessiveness_min']);number(r['cling_min']);number(r['absence_hours'],0,8760)
    if x['security_grief_policy']:
        r=x['security_grief_policy']
        if set(r)!={'security_below','grief_event_multiplier'}:raise ValueError('explicit security modulation')
        number(r['security_below']);number(r['grief_event_multiplier'],0,10)
    x.setdefault('empathy_policy',{});x.setdefault('light_policy',{})
    from persona_b1_social import validate_policies
    validate_policies(x,number)
    x.setdefault('conflict_policy',{})
    if x['conflict_policy']:
        c=x['conflict_policy']
        if set(c)-{'conflict_repeat_factor','security_drift_per_hour'}!={'anger_to_grief','max_fade_hours','soothed_fade_hours','clear_below','soothe_types','soothe_first','soothe_step','soothe_max','soothe_floor'}:raise ValueError('complete conflict policy')
        if 'conflict_repeat_factor' in c:number(c['conflict_repeat_factor'])
        if 'security_drift_per_hour' in c:number(c['security_drift_per_hour'],-1,0)
        for k in ('anger_to_grief','clear_below','soothe_first','soothe_step','soothe_max','soothe_floor'):number(c[k])
        for k in ('max_fade_hours','soothed_fade_hours'):number(c[k],0,8760)
        if not isinstance(c['soothe_types'],list) or not set(c['soothe_types'])<= {'affection','intimacy','empathy','companionship'}:raise ValueError('explicit soothe types')
    ident(x['revision']);number(x['repeat_hours'],0,168);number(x['repeat_factor'])
    if set(x['laws'])!=set(FIELDS):raise ValueError('all time laws explicit')
    for r in x['laws'].values():
        if set(r)-{'rest_ceiling','relax_per_hour'}!={'half_life_hours','target','growth_per_hour'}:raise ValueError('complete time law required')
        if r['half_life_hours'] is not None:number(r['half_life_hours'],.001,1e6)
        number(r['target']);number(r['growth_per_hour'],-1,1)
        if 'rest_ceiling' in r or 'relax_per_hour' in r:
            number(r['rest_ceiling']);number(r['relax_per_hour'],0,1000)
            if r['half_life_hours'] is not None or r['growth_per_hour']<0:raise ValueError('ceiling has separate explicit upper relaxation')
    if not isinstance(x['events'],dict) or not set(x['events'])<=TYPES:raise ValueError('unknown event type')
    for rules in x['events'].values():
        if not isinstance(rules,dict) or not set(rules)<=set(FIELDS):raise ValueError('unknown dimension')
        for r in rules.values():
            if set(r)!={'delta','satisfaction','floor','plateau_hours'}:raise ValueError('complete effect required')
            number(r['delta'],-1,1);number(r['satisfaction']);number(r['floor']);number(r['plateau_hours'],0,168)
    if not isinstance(x['couplings'],dict) or not set(x['couplings'])<={'valence','arousal','slow_security'}:raise ValueError('only emotion target coupling allowed')
    for coordinate,coeffs in x['couplings'].items():
        if x['laws'][coordinate]['growth_per_hour']!=0:raise ValueError('coupled coordinates cannot add natural growth')
        if not isinstance(coeffs,dict) or not set(coeffs)<={'grieve','anger','favored'}:raise ValueError('one-way drive coupling only')
        for coefficient in coeffs.values():number(coefficient,-1,1)
        for rules in x['events'].values():
            if coordinate in rules and rules[coordinate]['plateau_hours']:raise ValueError('emotion coupling is not a need plateau')
    for key in ('subtype_policy','mixed_policy','cling_relief'):
        if not isinstance(x[key],dict):raise ValueError('explicit mechanism policy required')
    if x['subtype_policy']:
        if set(x['subtype_policy'])!={'bump','visible_weight','clear_below'}:raise ValueError('complete subtype policy')
        for value in x['subtype_policy'].values():number(value)
    if x['mixed_policy']:
        if set(x['mixed_policy'])!={'negative_min','positive_min','hold_hours'}:raise ValueError('complete mixed policy')
        number(x['mixed_policy']['negative_min']);number(x['mixed_policy']['positive_min']);number(x['mixed_policy']['hold_hours'],0,168)
    if x['cling_relief']:
        if set(x['cling_relief'])!={'threshold','factor'}:raise ValueError('complete cling policy')
        number(x['cling_relief']['threshold']);number(x['cling_relief']['factor'])
    if set(x['sleep_multipliers'])!=set(FIELDS):raise ValueError('explicit sleep multipliers required')
    for v in x['sleep_multipliers'].values():number(v,.01,100)
    if x['initial_sleeping'] is not None and not isinstance(x['initial_sleeping'],bool):raise ValueError('sleep phase must be explicit or unknown')
    for key in ('mood_policy','presentation','slow_axis_events','proof_policy'):
        if not isinstance(x[key],dict):raise ValueError('explicit layer configuration required')
    if x['mood_policy']:
        p=x['mood_policy']
        if set(p)!={'window_days','offset_minutes','midpoint','span'}:raise ValueError('complete mood window')
        if isinstance(p['window_days'],bool) or not isinstance(p['window_days'],int) or not 1<=p['window_days']<=31:raise ValueError('bounded mood window')
        if isinstance(p['offset_minutes'],bool) or not isinstance(p['offset_minutes'],int) or abs(p['offset_minutes'])>840:raise ValueError('fixed offset required')
        number(p['midpoint']);number(p['span'],.001,1)
    if x['presentation']:
        p=x['presentation']
        if set(p)!={'base_thresholds','base_margin','base_hold_hours','cause_hold_hours','cause_priorities','label_rules','tones'}:raise ValueError('complete presentation policy')
        if set(p['base_thresholds'])!={'pleasant_high','active_high','pleasant_low','relaxed_valence','relaxed_arousal','tense_arousal','tired_arousal'}:raise ValueError('complete base thresholds')
        for v in p['base_thresholds'].values():number(v)
        number(p['base_margin']);number(p['base_hold_hours'],0,168);number(p['cause_hold_hours'],0,168)
        for label,priority in p['cause_priorities'].items():
            if label not in LABELS:raise ValueError('unknown cause label')
            number(priority,0,100)
        for label,r in p['label_rules'].items():
            if label not in LABELS or set(r)!={'key','minimum','maximum'} or r['key'] not in FIELDS:raise ValueError('explicit source label rule')
            number(r['minimum']);number(r['maximum'])
            if r['minimum']>r['maximum']:raise ValueError('invalid label range')
        for tone,r in p['tones'].items():
            if tone not in ('warm','playful','calm','focused','tired','guarded','conflicted') or set(r)!={'valence','arousal','weight'}:raise ValueError('tone policy')
            for v in r.values():number(v)
    for kind,rules in x['slow_axis_events'].items():
        if kind not in TYPES or not set(rules)<={'slow_security'}:raise ValueError('chat events cannot fabricate confidence achievement')
        for d in rules.values():number(d,-1,1)
    if x['proof_policy']:
        p=x['proof_policy']
        if set(p)!={'gains','daily_factor','high_threshold','high_factor'} or set(p['gains'])!={'her_ok','first','hard','helped_ack'}:raise ValueError('complete proof policy')
        for d in p['gains'].values():number(d)
        for k in ('daily_factor','high_threshold','high_factor'):number(p[k])
    if x['conflict_policy']:
        for key in ('anger','grieve'):
            r=x['laws'][key]
            if r['target']!=0 or r['growth_per_hour']!=0 or 'rest_ceiling' in r:raise ValueError('conflict transfer requires explicit zero-target decaying anger/grief')
    return x

def decision_check(d):
    if not isinstance(d,dict) or set(d)!={'accepted','rejected','source_hash','interpretation_is_not_ground_truth'}:raise ValueError('bound decision required')
    if not isinstance(d['source_hash'],str) or len(d['source_hash'])!=64 or any(c not in '0123456789abcdef' for c in d['source_hash']):raise ValueError('source hash required')
    if d['interpretation_is_not_ground_truth'] is not True:raise ValueError('interpretation uncertainty must remain explicit')
    if not isinstance(d['accepted'],list) or len(d['accepted'])>16:raise ValueError('bounded decisions required')
    seen=set()
    for r in d['accepted']:
        if set(r)!={'type','actor','target','context','assertion','certainty','role','start','end','evidence_hash'}:raise ValueError('complete evidence binding required')
        if r['type'] not in TYPES or r['type'] in seen:raise ValueError('invalid duplicate kind')
        seen.add(r['type'])
        external_empathy=r['type']=='empathy' and r['actor']=='owner' and r['target']=='third_party' and r['role']=='user' and r['context'] in ('direct','reported')
        if (r['context']!='direct' and not external_empathy) or r['assertion']!='occurred' or r['certainty']!='supported':raise ValueError('unconfirmed observation')
        if not external_empathy and ({r['actor'],r['target']}!={'owner','assistant'} or r['role'] not in ('user','assistant') or (r['role']=='user')!=(r['actor']=='owner')):raise ValueError('role mismatch')
        if r['type'] in ('discovery','task_progress','reflection','helped'):raise ValueError('operation requires separate trusted result protocol')
        if r['type']=='reconciliation' and r['actor']!='owner':raise ValueError('unconfirmed reconciliation')
        if any(isinstance(r[k],bool) or not isinstance(r[k],int) for k in ('start','end')) or not 0<=r['start']<r['end']:raise ValueError('invalid evidence span')
        if not isinstance(r['evidence_hash'],str) or len(r['evidence_hash'])!=64 or any(c not in '0123456789abcdef' for c in r['evidence_hash']):raise ValueError('evidence hash required')
    # Only machine reason codes may cross this boundary, never raw excerpts.
    for r in d['rejected']:
        if set(r)!={'type','reason'} or r['type'] not in TYPES or r['reason'] not in ('not_direct_interaction','not_asserted_occurrence','uncertain_interpretation','different_or_unknown_participants','actor_source_role_mismatch','operation_result_required_not_chat_claim','owner_confirmation_required'):raise ValueError('invalid rejected decision')
    return deepcopy(d)

def advance(v,law,h,freeze_growth=False):
    if v is None:return None
    if '_transfer' in law:
        from persona_b1_conflict import transferred_grief
        tr=law['_transfer'];return transferred_grief(v,tr['anger'],tr['anger_rate'],tr['grief_rate'],tr['ratio'],h)
    if 'rest_ceiling' in law:
        from persona_b1_ceiling import ceiling_advance
        return ceiling_advance(v,h,law['rest_ceiling'],law['growth_per_hour'],law['relax_per_hour'],freeze_growth)
    growth=law['growth_per_hour'];half=law['half_life_hours'];k=math.log(2)/half if half else 0
    derivative=growth-k*(v-law['target'])
    if freeze_growth and derivative>0:return v
    if not k:return min(1,max(0,v+growth*h))
    z=k*h;gain=-math.expm1(-z)
    return min(1,max(0,v+(law['target']-v)*gain+growth*h*(gain/z if z else 1)))

MIXED_PAIRS={'anger_closeness':('anger',('longing','possessiveness')),'grieve_closeness':('grieve',('longing','possessiveness')),'anger_protective':('anger',('protective_drive',))}
def track_mixed_segment(state,config,origin,start,end,frozen):
    if not config['mixed_policy']:return
    p=config['mixed_policy'];hours=(end-start).total_seconds()/3600
    from datetime import timedelta
    for pair,(negative,positive) in MIXED_PAIRS.items():
        if origin[negative] is None or all(origin[key] is None for key in positive):
            state['mixed'][pair]={'status':'unknown','since':None};continue
        heartache=negative=='grieve' and state['subtypes'].get('grieve') and max(state['subtypes']['grieve'],key=state['subtypes']['grieve'].get)=='心疼'
        def value(key,h):return advance(origin[key],config['laws'][key],h,key in frozen)
        thresholds={negative:p['negative_min'],**{key:p['positive_min'] for key in positive if origin[key] is not None}};points={0,hours}
        for key,threshold in thresholds.items():
            intervals=[0,hours]
            tr=config['laws'][key].get('_transfer')
            if tr:
                from persona_b1_conflict import grief_peak
                peak=grief_peak(origin[key],tr['anger'],tr['anger_rate'],tr['grief_rate'],tr['ratio'])
                if peak and 0<peak<hours:intervals.insert(1,peak)
            for lo,hi in zip(intervals,intervals[1:]):
                left=value(key,lo)-threshold;right=value(key,hi)-threshold
                if left*right<0:
                    for _ in range(60):
                        mid=(lo+hi)/2
                        if (value(key,mid)-threshold)*left>0:lo=mid
                        else:hi=mid
                    points.add((lo+hi)/2)
        points=sorted(points)
        for a,b in zip(points,points[1:]):
            active=not heartache and value(negative,(a+b)/2)>=p['negative_min'] and any(origin[key] is not None and value(key,(a+b)/2)>=p['positive_min'] for key in positive)
            if not active:
                if any(origin[key] is None for key in positive) and value(negative,(a+b)/2)>=p['negative_min'] and not heartache:
                    state['mixed'][pair]={'status':'unknown','since':None}
                else:state['mixed'].pop(pair,None)
                continue
            current=state['mixed'].get(pair,{})
            since=current.get('since') or at(start+timedelta(hours=a))
            held=(end-datetime.fromisoformat(since)).total_seconds()/3600
            state['mixed'][pair]={'status':'visible' if held>=p['hold_hours'] else 'forming','since':since}

class PersonaB1Store:
    def __init__(self,path,*,legacy_path,timeout=.02):
        self.path=Path(path);old=Path(legacy_path)
        if self.path.is_symlink() or self.path.resolve()==old.resolve() or (self.path.exists() and old.exists() and os.path.samefile(self.path,old)):raise ValueError('isolated database required')
        self.timeout=number(timeout,0,.1);self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(self._connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');v=db.execute('PRAGMA user_version').fetchone()[0]
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if v==6:
                if tables!={'b1_state','b1_events','b1_growth_frames','b1_pressure_frames','b1_numeric'}:raise ValueError('incomplete B1 schema')
            elif v!=0 or tables:raise ValueError('no implicit migration')
            else:
                db.execute('CREATE TABLE b1_state(scope TEXT PRIMARY KEY,revision INTEGER NOT NULL,at TEXT NOT NULL,state TEXT NOT NULL,policy TEXT NOT NULL,seed_hash TEXT NOT NULL)')
                db.execute('CREATE TABLE b1_events(scope TEXT NOT NULL,event_id TEXT NOT NULL,hash TEXT NOT NULL,receipt TEXT NOT NULL,PRIMARY KEY(scope,event_id))');db.execute('CREATE TABLE b1_growth_frames(scope TEXT NOT NULL,frame_id TEXT NOT NULL,frame_json TEXT NOT NULL,PRIMARY KEY(scope,frame_id))');db.execute('CREATE TABLE b1_pressure_frames(scope TEXT NOT NULL,frame_id TEXT NOT NULL,frame_json TEXT NOT NULL,PRIMARY KEY(scope,frame_id))');db.execute('CREATE TABLE b1_numeric(scope TEXT NOT NULL,revision INTEGER NOT NULL,at TEXT NOT NULL,snapshot TEXT NOT NULL,PRIMARY KEY(scope,revision))');db.execute('PRAGMA user_version=6')
    def _connect(self,readonly=False):
        db=sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro',uri=True,timeout=self.timeout) if readonly else sqlite3.connect(self.path,timeout=self.timeout)
        db.row_factory=sqlite3.Row
        version=db.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0,6):
            db.close();raise ValueError('unsupported B1 schema')
        return db
    def _write(self,fn):
        with closing(self._connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');r=fn(db);self._before_commit(db);return r
    def _before_commit(self,db):pass
    def initialize(self,scope,initial,when,config):
        scope=ident(scope);initial=values(initial);when=at(when);config=policy(config)
        seed=digest([scope,initial,when,config])
        def op(db):
            r=db.execute('SELECT * FROM b1_state WHERE scope=?',(scope,)).fetchone()
            if r:
                if r['seed_hash']!=seed:raise Conflict('different initial seed or policy')
                return self._state(r)
            state={'values':initial,'plateaus':{},'last_by_kind':{},'labels':[],'labels_updated_at':None,'causes':[],'subtypes':{},'mixed':{},'sleeping':config['initial_sleeping'],'mood_days':{},'slow_mood':{'value':None,'unit':'-1..1','covered_hours':0,'unknown_hours':0},'base_label':None,'pending_base':None,'cause_label':None,'current_labels':[],'uncertain_labels':[],'proof_days':{},'label_times':{},'grudge':None,'conflict_episode':None,'empathy_loads':{},'empathy_seen':{},'empathy_daily':{},'light_taps':[],'heart_days':{},'lit':None,'light_marks':[],'last_conversation_at':None,'growth_coverage':{},'fatigue':{'value':config['fatigue_policy'].get('initial'),'as_of':when,'status':'initial_explicit' if config['fatigue_policy'] else 'unconfigured'}}
            if config['presentation']:
                from persona_b1_layers import update_layers
                update_layers(state,config,datetime.fromisoformat(when))
            db.execute('INSERT INTO b1_state VALUES(?,?,?,?,?,?)',(scope,0,when,canonical(state),canonical(config),seed))
            self._save_numeric(db,scope,0,when,state,config)
            return self._state(db.execute('SELECT * FROM b1_state WHERE scope=?',(scope,)).fetchone())
        return self._write(op)
    def _state(self,r):
        if r is None:raise KeyError('scope not initialized')
        state=json.loads(r['state']);state['values']=values(state['values'])
        return {'scope':r['scope'],'revision':r['revision'],'at':r['at'],**state}
    def read(self,scope):
        with closing(self._connect(True)) as db:return self._state(db.execute('SELECT * FROM b1_state WHERE scope=?',(ident(scope),)).fetchone())
    def _save_numeric(self,db,scope,revision,when,state,config):
        numeric={'scope':scope,'revision':revision,'as_of':when,'policy_revision':config['revision'],'values':{**state['values'],'fatigue':state['fatigue']['value']}}
        db.execute('INSERT INTO b1_numeric VALUES(?,?,?,?)',(scope,revision,when,canonical(numeric)))
    def numeric_at(self,scope,when=None):
        # Only historical committed coordinates, never advance or interpolate.
        with closing(self._connect(True)) as db:
            if when is None:r=db.execute('SELECT snapshot FROM b1_numeric WHERE scope=? ORDER BY revision DESC LIMIT 1',(ident(scope),)).fetchone()
            else:r=db.execute('SELECT snapshot FROM b1_numeric WHERE scope=? AND at<=? ORDER BY at DESC,revision DESC LIMIT 1',(ident(scope),at(when))).fetchone()
            return json.loads(r[0]) if r else None
    def receipt(self,scope,event_id):
        with closing(self._connect(True)) as db:
            r=db.execute('SELECT receipt FROM b1_events WHERE scope=? AND event_id=?',(ident(scope),ident(event_id))).fetchone();return json.loads(r[0]) if r else None
    def settle(self,envelope,*,expected_revision):
        if set(envelope)-{'nuances','tone','phase','operation_result','empathy_details','light_details','interaction_modes','feeling_sources','conversation_observed_at','correction_details','growth_frames','pressure_frames','phase_source_kind'}!={'namespace','scope','event_id','occurred_at','policy_revision','decision','labels'} or envelope['namespace']!='b1-completed-turn-v1':raise ValueError('internal completed-turn envelope required')
        e=deepcopy(envelope);e['scope']=ident(e['scope']);e['event_id']=ident(e['event_id']);e['occurred_at']=at(e['occurred_at']);e['decision']=decision_check(e['decision'])
        if isinstance(expected_revision,bool) or not isinstance(expected_revision,int) or expected_revision<0:raise ValueError('revision required')
        modes=e.get('interaction_modes',{})
        if not isinstance(modes,dict) or not set(modes)<={r['type'] for r in e['decision']['accepted']} or any(v not in ('apply','mark_only') for v in modes.values()):raise ValueError('bound per-interaction mode required')
        applied_interactions=[r for r in e['decision']['accepted'] if modes.get(r['type'],'apply')=='apply']
        labels=e['labels']
        if not isinstance(labels,list) or len(labels)>8 or any(not isinstance(x,str) or x not in LABELS for x in labels):raise ValueError('bounded source labels required')
        if labels and not e['decision']['accepted']:raise ValueError('labels require supported source interaction')
        feeling_sources=e.get('feeling_sources',[])
        if not isinstance(feeling_sources,list) or len(feeling_sources)>32:raise ValueError('bounded feeling source bindings')
        for binding in feeling_sources:
            if not isinstance(binding,dict) or set(binding)!={'label','type','evidence_hash'} or binding['label'] not in labels or not any(row['type']==binding['type'] and row['evidence_hash']==binding['evidence_hash'] for row in e['decision']['accepted']):raise ValueError('feeling source mismatch')
        has_mark_only=any(v=='mark_only' for v in modes.values())
        applied_labels=[label for label in labels if (not has_mark_only or any(b['label']==label and modes.get(b['type'],'apply')=='apply' for b in feeling_sources))]
        corrections=e.get('correction_details',[])
        if not isinstance(corrections,list) or len(corrections)>1:raise ValueError('one bound correction per turn')
        for r in corrections:
            if not isinstance(r,dict) or set(r)!={'type','actor','target','context','assertion','certainty','user_start','user_end','assistant_start','assistant_end','evidence_hash'}:raise ValueError('complete correction evidence')
            if any(r[k]!=v for k,v in {'type':'correction','actor':'owner','target':'assistant','context':'direct','assertion':'occurred','certainty':'supported'}.items()):raise ValueError('supported direct correction required')
            for role in ('user','assistant'):
                if any(isinstance(r[role+'_'+k],bool) or not isinstance(r[role+'_'+k],int) for k in ('start','end')) or not 0<=r[role+'_start']<r[role+'_end']:raise ValueError('bounded correction spans')
        # Capture source inputs outside our write transaction. Reuse the first
        # persisted grid frame, never re-sample a previously settled interval.
        from b1_growth_protocol import grid_start,derive_frame,verify_frame
        with closing(self._connect(True)) as read_db:
            row=read_db.execute('SELECT * FROM b1_state WHERE scope=?',(e['scope'],)).fetchone()
            frame_policy=policy(json.loads(row['policy']))['growth_policy'];initial_at=datetime.fromisoformat(row['at'])
            supplied=e.pop('growth_frames',None);frames=[]
            if supplied is not None:
                if not isinstance(supplied,list) or len(supplied)>512:raise ValueError('bounded supplied frames')
                if supplied and not frame_policy:raise ValueError('growth policy not configured')
                for item in supplied:verify_frame(frame_policy,item)
            if frame_policy and datetime.fromisoformat(e['occurred_at'])>initial_at:
                from datetime import timedelta
                cursor=grid_start(initial_at,frame_policy['grid_seconds'],frame_policy.get('anchor_utc'));stop=datetime.fromisoformat(e['occurred_at'])
                while cursor<stop:
                    if len(frames)>=512:raise RuntimeError('growth frame budget exhausted; advance in chronological chunks')
                    frame_id=frame_policy['revision']+':'+cursor.isoformat()
                    saved=read_db.execute('SELECT frame_json FROM b1_growth_frames WHERE scope=? AND frame_id=?',(e['scope'],frame_id)).fetchone()
                    if saved:
                        frame=json.loads(saved[0])
                        if supplied is not None:
                            matches=[f for f in supplied if f.get('grid_start')==cursor.isoformat()]
                            if len(matches)!=1 or canonical(matches[0])!=canonical(frame):raise Conflict('persisted growth frame cannot be replaced')
                    elif supplied is not None:
                        matches=[f for f in supplied if f.get('grid_start')==cursor.isoformat()]
                        if len(matches)!=1:raise ValueError('complete unique supplied growth interval required')
                        frame=matches[0]
                    else:
                        loader=getattr(self,'growth_frame_loader',None)
                        if loader is not None:frame=loader(cursor)
                        else:
                            frame=derive_frame(frame_policy,cursor,{'scope':frame_policy['source_scope'],'policy_revision':frame_policy['source_policy_revision'],'revision':None,'as_of':None,'values':{},'availability':'unavailable'})
                    verify_frame(frame_policy,frame);frames.append(frame)
                    cursor+=timedelta(seconds=frame_policy['grid_seconds'])
            from b1_pressure_protocol import build_pressure_frame,verify_pressure_frame
            pressure_policy=policy(json.loads(row['policy']))['pressure_policy'];pressure_frames=[]
            supplied_pressure=e.pop('pressure_frames',None)
            if supplied_pressure is not None:
                if not isinstance(supplied_pressure,list) or len(supplied_pressure)>512:raise ValueError('bounded pressure frames')
                if supplied_pressure and not pressure_policy:raise ValueError('pressure policy not configured')
                for item in supplied_pressure:verify_pressure_frame(pressure_policy,item)
            if pressure_policy and datetime.fromisoformat(e['occurred_at'])>initial_at:
                from datetime import timedelta
                cursor=grid_start(initial_at,pressure_policy['grid_seconds'],pressure_policy.get('anchor_utc'));stop=datetime.fromisoformat(e['occurred_at'])
                while cursor<stop:
                    if len(pressure_frames)>=512:raise RuntimeError('pressure frame budget exhausted; advance chronological chunks')
                    fid=pressure_policy['revision']+':'+cursor.isoformat()
                    saved=read_db.execute('SELECT frame_json FROM b1_pressure_frames WHERE scope=? AND frame_id=?',(e['scope'],fid)).fetchone()
                    matches=[f for f in (supplied_pressure or []) if f.get('grid_start')==cursor.isoformat()]
                    if saved:
                        frame=json.loads(saved[0])
                        if supplied_pressure is not None and (len(matches)!=1 or canonical(matches[0])!=canonical(frame)):raise Conflict('persisted pressure frame cannot be replaced')
                    elif supplied_pressure is not None:
                        if len(matches)!=1:raise ValueError('complete unique pressure interval required')
                        frame=matches[0]
                    else:
                        loader=getattr(self,'pressure_frame_loader',None)
                        frame=loader(cursor) if loader else build_pressure_frame(pressure_policy,cursor,None,None)
                    verify_pressure_frame(pressure_policy,frame);pressure_frames.append(frame)
                    cursor+=timedelta(seconds=pressure_policy['grid_seconds'])
            # Frames are durable integration inputs, not part of an event's
            # canonical identity: retry at later checkpoint must use old receipt.
        h=digest(e)
        def op(db):
            old=db.execute('SELECT * FROM b1_events WHERE scope=? AND event_id=?',(e['scope'],e['event_id'])).fetchone()
            if old:
                if old['hash']!=h:raise Conflict('event identity reused with different source/policy')
                return {'replayed':True,'receipt':json.loads(old['receipt'])}
            row=db.execute('SELECT * FROM b1_state WHERE scope=?',(e['scope'],)).fetchone();s=self._state(row);p=policy(json.loads(row['policy']))
            if p['revision']!=e['policy_revision']:raise Conflict('policy revision mismatch')
            if s['revision']!=expected_revision:raise Conflict('stale revision')
            start=datetime.fromisoformat(s['at']);end=datetime.fromisoformat(e['occurred_at'])
            if end<start:raise Conflict('late event requires explicit reconciliation')
            if 'phase_source_kind' in e:
                if e.get('phase') not in ('sleep','wake') or e['phase_source_kind'] not in ('internal_owner_request','internal_rest_plan'):raise ValueError('explicit internal lifecycle source required')
                if end!=start:raise Conflict('phase requires coordinated checkpoint at same time')
            for frame in frames:
                fid=frame_policy['revision']+':'+frame['grid_start'];raw=canonical(frame)
                existing=db.execute('SELECT frame_json FROM b1_growth_frames WHERE scope=? AND frame_id=?',(e['scope'],fid)).fetchone()
                if existing and existing[0]!=raw:raise Conflict('growth frame identity conflict')
                if not existing:db.execute('INSERT INTO b1_growth_frames VALUES(?,?,?)',(e['scope'],fid,raw))
            for frame in pressure_frames:
                fid=pressure_policy['revision']+':'+frame['grid_start'];raw=canonical(frame)
                existing=db.execute('SELECT frame_json FROM b1_pressure_frames WHERE scope=? AND frame_id=?',(e['scope'],fid)).fetchone()
                if existing and existing[0]!=raw:raise Conflict('pressure frame identity conflict')
                if not existing:db.execute('INSERT INTO b1_pressure_frames VALUES(?,?,?)',(e['scope'],fid,raw))
            before=deepcopy(s['values']);points={start,end}
            for frame in frames+pressure_frames:
                for key in ('grid_start','grid_end'):
                    boundary=datetime.fromisoformat(frame[key])
                    if start<boundary<end:points.add(boundary)
            if p['rhythm_policy']:
                from persona_b1_layers import rhythm_boundaries
                points.update(rhythm_boundaries(start,end,p['rhythm_policy']))
            for v in s['plateaus'].values():
                t=datetime.fromisoformat(v)
                if start<t<end:points.add(t)
            if p['conflict_policy']:
                from persona_b1_conflict import expiry,retire
                retire(s,p['conflict_policy'],start)
                boundary=expiry(s,p['conflict_policy'])
                if boundary is not None and start<boundary<end:points.add(boundary)
                anger=s['values']['anger'];half=p['laws']['anger']['half_life_hours'];threshold=p['conflict_policy']['clear_below']
                if half and s['sleeping'] is True:half/=p['sleep_multipliers']['anger']
                if s.get('grudge') and anger is not None and half and anger>threshold>0:
                    from datetime import timedelta
                    boundary=start+timedelta(hours=math.log(anger/threshold)*half/math.log(2))
                    if start<boundary<end:points.add(boundary)
            # Rest-ceiling crossing is a real segment boundary, never a tick clamp.
            from datetime import timedelta
            for key,law in p['laws'].items():
                value=s['values'][key]
                if 'rest_ceiling' in law and value is not None and value<law['rest_ceiling'] and law['growth_per_hour']>0:
                    begin=max(start,datetime.fromisoformat(s['plateaus'][key])) if key in s['plateaus'] else start
                    boundary=begin+timedelta(hours=(law['rest_ceiling']-value)/law['growth_per_hour'])
                    if start<boundary<end:points.add(boundary)
            points=sorted(points)
            time_methods=[];base_budget=8192
            effective_laws=deepcopy(p['laws'])
            for key,multiplier in p['sleep_multipliers'].items():
                if end>start and s['sleeping'] is None and multiplier!=1:s['values'][key]=None
                elif s['sleeping'] is True:
                    if effective_laws[key]['half_life_hours'] is not None:effective_laws[key]['half_life_hours']/=multiplier
                    effective_laws[key]['growth_per_hour']*=multiplier
            time_policy={**p,'laws':effective_laws}
            for a,b in zip(points,points[1:]):
                if p['conflict_policy']:
                    from persona_b1_conflict import retire
                    retire(s,p['conflict_policy'],a)
                origin=dict(s['values']);hstep=(b-a).total_seconds()/3600
                if p['fatigue_policy']:
                    from persona_b1_layers import fatigue_step
                    frame=next((f for f in pressure_frames if datetime.fromisoformat(f['grid_start'])<=a<datetime.fromisoformat(f['grid_end'])),None)
                    pressure=frame['pressure']['values'] if frame else None
                    old_fatigue=s['fatigue']['value']
                    value,status=fatigue_step(old_fatigue,s['sleeping'],pressure,hstep,p['fatigue_policy'])
                    record={'value':value,'as_of':b.isoformat(),'status':status}
                    if value is None:
                        record['last_known']=s['fatigue'].get('last_known',old_fatigue);record['last_known_at']=s['fatigue'].get('last_known_at',s['fatigue']['as_of'])
                    s['fatigue']=record
                frozen={key for key in s['plateaus'] if a<datetime.fromisoformat(s['plateaus'][key])}
                segment_laws=deepcopy(effective_laws)
                if p['rhythm_policy']:
                    from persona_b1_layers import rhythm_flags
                    night,dawn=rhythm_flags(a,p['rhythm_policy'])
                    for key,law in segment_laws.items():
                        if dawn and key in p['rhythm_policy']['dawn_freeze']:
                            segment_laws[key]={'half_life_hours':None,'target':0,'growth_per_hour':0}
                        elif night:law['growth_per_hour']*=p['rhythm_policy']['night_multipliers'].get(key,1)
                if frame_policy:
                    current_frame=next((f for f in frames if datetime.fromisoformat(f['grid_start'])<=a<datetime.fromisoformat(f['grid_end'])),None)
                    if current_frame is None:raise ValueError('uncovered growth interval')
                    for key in ('longing','possessiveness','protective_drive','libido'):
                        if key not in frame_policy['targets']:continue
                        law=segment_laws[key];factor=current_frame['targets'][key]['factor']
                        needs_growth=law['growth_per_hour']>0 and key not in frozen and ('rest_ceiling' not in law or (origin[key] is not None and origin[key]<law['rest_ceiling']))
                        if needs_growth:
                            if factor is None:
                                old=s['growth_coverage'].get(key,{})
                                s['growth_coverage'][key]={'status':'unknown','since':old.get('since',a.isoformat()),'last_known':origin[key] if origin[key] is not None else old.get('last_known'),'as_of':a.isoformat() if origin[key] is not None else old.get('as_of')}
                                origin[key]=None;s['values'][key]=None
                            else:
                                law['growth_per_hour']*=factor
                                if origin[key] is not None:s['growth_coverage'][key]={'status':'known','as_of':a.isoformat()}

                if p['conflict_policy'] and s.get('grudge'):
                    ka=math.log(2)/segment_laws['anger']['half_life_hours'] if segment_laws['anger']['half_life_hours'] else 0
                    kg=math.log(2)/segment_laws['grieve']['half_life_hours'] if segment_laws['grieve']['half_life_hours'] else 0
                    segment_laws['grieve']['_transfer']={'anger':(None if origin['anger'] is None else max(0,origin['anger']-sum(x['value'] for x in s['empathy_loads'].values() if x['key']=='anger'))),'anger_rate':ka,'grief_rate':kg,'ratio':p['conflict_policy']['anger_to_grief']}
                time_policy={**p,'laws':segment_laws}
                if p['conflict_policy'] and s.get('grudge') and p['conflict_policy'].get('security_drift_per_hour'):
                    segment_laws['slow_security']['_extra_drift']=p['conflict_policy']['security_drift_per_hour']
                track_mixed_segment(s,time_policy,origin,a,b,frozen)
                def evolve(key,t):
                    from persona_b1_coupling import coupled,reflected_coordinate
                    if '_extra_drift' in segment_laws[key]:return reflected_coordinate(origin[key],segment_laws[key],t,origin,p['couplings'].get(key,{}),segment_laws,frozen,segment_laws[key]['_extra_drift'])[0]
                    if key in p['couplings']:return coupled(origin[key],segment_laws[key],t,origin,p['couplings'][key],segment_laws,frozen)[0]
                    return advance(origin[key],segment_laws[key],t,key in frozen)
                if p['presentation']:
                    from persona_b1_layers import track_base_segment
                    def sample_values(t):
                        from persona_b1_coupling import coupled
                        return {key:evolve(key,t) for key in FIELDS}
                    base_budget-=track_base_segment(s,p,a,b,sample_values,budget=base_budget)
                for key in FIELDS:
                    if '_extra_drift' in segment_laws[key]:
                        from persona_b1_coupling import reflected_coordinate
                        s['values'][key],method=reflected_coordinate(origin[key],segment_laws[key],hstep,origin,p['couplings'].get(key,{}),segment_laws,frozen,segment_laws[key]['_extra_drift']);time_methods.append({'key':key,**method})
                    elif key in p['couplings']:
                        from persona_b1_coupling import coupled
                        s['values'][key],method=coupled(origin[key],segment_laws[key],hstep,origin,p['couplings'][key],segment_laws,frozen)
                        time_methods.append({'key':key,**method})
                    else:s['values'][key]=advance(origin[key],segment_laws[key],hstep,key in frozen)
                if p['mood_policy']:
                    from persona_b1_layers import record_mood
                    def sample(t):
                        if 'valence' in p['couplings']:
                            from persona_b1_coupling import coupled
                            return coupled(origin['valence'],segment_laws['valence'],t,origin,p['couplings']['valence'],segment_laws,frozen)[0]
                        return advance(origin['valence'],segment_laws['valence'],t,'valence' in frozen)
                    record_mood(s,a,b,sample,p['mood_policy'],max_step_hours=min(1.0,min((law['half_life_hours']/math.log(2) for law in segment_laws.values() if law['half_life_hours']),default=1.0)))
                from persona_b1_social import decay_loads
                decay_loads(s,segment_laws,hstep)
                for key,weights in s['subtypes'].items():
                    half=segment_laws[key]['half_life_hours'];retention=math.exp(-math.log(2)*hstep/half) if half else 1
                    for label in weights:weights[label]*=retention
            if p['conflict_policy']:
                retire(s,p['conflict_policy'],end)
            effects=[];deferred=[]
            for item in applied_interactions:
                kind=item['type'];rules=p['events'].get(kind)
                if kind=='empathy' and e.get('empathy_details'):rules=None  # detailed share owns this event; no generic second dose
                for key,delta in p['slow_axis_events'].get(kind,{}).items():
                    if s['values'][key] is not None:s['values'][key]=min(1,max(0,s['values'][key]+delta))
                if rules is None:deferred.append(kind);continue
                last=s['last_by_kind'].get(kind);factor=p['repeat_factor'] if last and (end-datetime.fromisoformat(last)).total_seconds()<p['repeat_hours']*3600 else 1
                applied=False
                for key,r in rules.items():
                    v=s['values'][key]
                    if v is None:effects.append({'key':key,'kind':kind,'status':'unknown_not_updated'});continue
                    applied=True;relief=1
                    if key=='longing' and kind in ('affection','intimacy') and p['cling_relief'] and s['values']['cling'] is not None and s['values']['cling']>=p['cling_relief']['threshold']:
                        relief=p['cling_relief']['factor']
                    v=v-max(0,v-r['floor'])*r['satisfaction']*factor*relief
                    # Inertia: same-direction event pulses diminish near extremes.
                    delta=r['delta']*factor
                    if kind=='conflict' and delta>0 and p['conflict_policy'] and s.get('grudge'):
                        delta*=p['conflict_policy'].get('conflict_repeat_factor',1)**s['conflict_episode']['conflicts']
                    if key=='grieve' and delta>0 and p['security_grief_policy']:
                        sec=s['values']['slow_security'];mod=p['security_grief_policy']
                        if sec is None:
                            s['values'][key]=None;effects.append({'key':key,'status':'unknown_security_modulation'});continue
                        if sec<mod['security_below']:delta*=mod['grief_event_multiplier']
                    # Upstream inertia is emotion-coordinate-specific, not an
                    # unapproved attenuation of intimacy/possessiveness/libido.
                    v+=delta*((1-v) if delta>0 else v) if key in ('valence','arousal') else delta
                    s['values'][key]=min(1,max(0,v))
                    if r['satisfaction'] and r['plateau_hours'] and (key not in s['plateaus'] or datetime.fromisoformat(s['plateaus'][key])<=end):
                        from datetime import timedelta
                        s['plateaus'][key]=at(end+timedelta(hours=r['plateau_hours']))
                    effects.append({'key':key,'kind':kind,'status':'candidate_applied','repeat_factor':factor})
                if applied:s['last_by_kind'][kind]=e['occurred_at']
            from persona_b1_social import cap_loads,apply_empathy,light
            cap_loads(s)
            empathy_details=e.get('empathy_details',[]);light_details=e.get('light_details',[])
            if not isinstance(empathy_details,list) or len(empathy_details)>8 or not isinstance(light_details,list) or len(light_details)>8:raise ValueError('bounded social details')
            for detail in empathy_details:
                if modes.get(detail.get('type'),'apply')=='mark_only':continue
                effects.append({'mechanism':'empathy',**apply_empathy(s,p['empathy_policy'],detail,e['decision']['accepted'],end)})
            for detail in light_details:
                if detail.get('sub')=='骄傲' and not any(d.get('sub')=='替人高兴' and d.get('closeness')=='owner' and d.get('identity_status')=='resolved' and d.get('evidence_hash')==detail.get('evidence_hash') for d in empathy_details):raise ValueError('pride requires resolved owner success source')
                lit_result=light(s,p['light_policy'],detail,e['decision']['accepted'],end)
                if lit_result:effects.append({'mechanism':'light','status':'lit',**lit_result})
            if p['conflict_policy']:
                from persona_b1_conflict import apply_episode
                apply_episode(s,p['conflict_policy'],applied_interactions,e['event_id'],end)
            if corrections and p['correction_confidence_delta'] is not None and s['values']['slow_confidence'] is not None:
                s['values']['slow_confidence']=max(0,s['values']['slow_confidence']+p['correction_confidence_delta'])
                effects.append({'mechanism':'correction','status':'source_bound_interpretation','not_logical_ground_truth':True})
            if 'phase' in e:
                if e['phase'] not in ('sleep','wake') or e['decision']['accepted']:raise ValueError('phase is explicit lifecycle event, not chat inference')
                s['sleeping']=e['phase']=='sleep'
            if 'operation_result' in e:
                r=e['operation_result']
                if not isinstance(r,dict) or set(r)-{'helped'}!={'operation_id','proof','status','result_hash'} or r['status']!='succeeded' or r['proof'] not in ('her_ok','first','hard','helped_ack'):raise ValueError('verified successful operation required')
                ident(r['operation_id'])
                if 'helped' in r:
                    from persona_b1_social import help_share
                    help_data=r['helped']
                    if not isinstance(help_data,dict) or set(help_data)!={'subject_ref','incident_ref','sub'} or r['proof']!='helped_ack':raise ValueError('trusted corresponding helped result required')
                    effects.append({'mechanism':'helped',**help_share(s,p['empathy_policy'],**help_data)})
                if not p['proof_policy']:raise ValueError('proof policy not configured')
                if s['values']['slow_confidence'] is not None:
                    day=e['occurred_at'][:10];count=s['proof_days'].get(day,0)
                    gain=p['proof_policy']['gains'][r['proof']]*p['proof_policy']['daily_factor']**count
                    if s['values']['slow_confidence']>p['proof_policy']['high_threshold']:gain*=p['proof_policy']['high_factor']
                    s['values']['slow_confidence']=min(1,s['values']['slow_confidence']+gain);s['proof_days'][day]=count+1
            if 'operation_result' in e:
                lit_result=light(s,p['light_policy'],None,[],end,operation=True)
                if lit_result:effects.append({'mechanism':'light','status':'lit',**lit_result})
            if 'tone' in e and applied_interactions and not has_mark_only:
                if not p['presentation'] or e['tone'] not in p['presentation']['tones'] or not e['decision']['accepted']:raise ValueError('source-bound configured tone required')
                tone=p['presentation']['tones'][e['tone']]
                for key in ('valence','arousal'):
                    if s['values'][key] is not None:s['values'][key]+=(tone[key]-s['values'][key])*tone['weight']
            allowed_subtypes={'anger':{'生气','不满','不甘心','不平'},'grieve':{'失落','委屈','分别','心疼','自责'},'favored':{'吃醋','被晾着','被忘'}}
            nuances=e.get('nuances',[])
            if not isinstance(nuances,list) or len(nuances)>16:raise ValueError('bounded nuances required')
            for nuance in nuances:
                if modes.get(nuance.get('type'),'apply')=='mark_only':continue
                if not isinstance(nuance,dict) or set(nuance)!={'type','key','label','evidence_hash'}:raise ValueError('complete nuance binding required')
                key=nuance['key'];label=nuance['label']
                if key not in allowed_subtypes or label not in allowed_subtypes[key]:raise ValueError('unsupported subtype')
                if not any(r['type']==nuance['type'] and r['evidence_hash']==nuance['evidence_hash'] for r in e['decision']['accepted']):raise ValueError('nuance not bound to accepted interaction')
                if not p['subtype_policy']:raise ValueError('nuance policy not configured')
                if s['values'][key] is not None:
                    weights=s['subtypes'].setdefault(key,{})
                    weights[label]=min(1,weights.get(label,0)+p['subtype_policy']['bump'])
            if p['subtype_policy']:
                for key in list(s['subtypes']):
                    if s['values'][key] is not None and s['values'][key]<p['subtype_policy']['clear_below']:del s['subtypes'][key]
            # Mixed state does not alter either drive. It never clears real
            # feelings merely because an author-specific display cap elapsed.
            pairs=MIXED_PAIRS
            if p['mixed_policy']:
                for pair,(negative,positive) in pairs.items():
                    n=s['values'][negative];positive_values=[s['values'][key] for key in positive];known_positive=[v for v in positive_values if v is not None]
                    heartache=negative=='grieve' and s['subtypes'].get('grieve') and max(s['subtypes']['grieve'],key=s['subtypes']['grieve'].get)=='心疼'
                    if n is None or not known_positive:
                        s['mixed'][pair]={'status':'unknown','since':None};continue
                    if n>=p['mixed_policy']['negative_min'] and any(v>=p['mixed_policy']['positive_min'] for v in known_positive) and not heartache:
                        current=s['mixed'].get(pair,{})
                        since=current.get('since') or e['occurred_at']
                        held=(end-datetime.fromisoformat(since)).total_seconds()/3600
                        s['mixed'][pair]={'status':'visible' if held>=p['mixed_policy']['hold_hours'] else 'forming','since':since}
                    elif n>=p['mixed_policy']['negative_min'] and any(v is None for v in positive_values) and not heartache:
                        s['mixed'][pair]={'status':'unknown','since':None}
                    else:s['mixed'].pop(pair,None)
            # Labels are evidence-supported simultaneous feelings, never inferred
            # from missing values or forced into a single winner.
            if applied_labels and applied_interactions:
                s['labels']=list(dict.fromkeys((s['labels'] if p['presentation'] else [])+applied_labels));s['labels_updated_at']=e['occurred_at']
                for label in applied_labels:s['label_times'][label]=e['occurred_at']
            if p['presentation']:
                from persona_b1_layers import update_layers
                update_layers(s,p,end,applied_labels if applied_interactions else [],update_base=bool(applied_interactions or 'operation_result' in e or (start==end and s['base_label'] is None)))
            if e['decision']['accepted']:
                s['causes']=[{'kind':r['type'],'evidence_hash':r['evidence_hash']} for r in e['decision']['accepted']]
            if 'conversation_observed_at' in e:
                if at(e['conversation_observed_at'])!=e['occurred_at']:raise ValueError('conversation observation time mismatch')
                s['last_conversation_at']=e['occurred_at']
            stored={k:s[k] for k in ('values','plateaus','last_by_kind','labels','labels_updated_at','causes','subtypes','mixed','sleeping','mood_days','slow_mood','base_label','pending_base','cause_label','current_labels','uncertain_labels','proof_days','label_times','grudge','conflict_episode','empathy_loads','empathy_seen','empathy_daily','light_taps','heart_days','lit','light_marks','last_conversation_at','growth_coverage','fatigue')}
            receipt={'scope':e['scope'],'event_id':e['event_id'],'source_hash':e['decision']['source_hash'],'occurred_at':e['occurred_at'],'before_revision':s['revision'],'after_revision':s['revision']+1,'before':before,'after':s['values'],'effects':effects,'growth_frame_hashes':[frame['frame_hash'] for frame in frames],'pressure_frame_hashes':[frame['frame_hash'] for frame in pressure_frames],'observed_labels':labels,'applied_labels':applied_labels,'interaction_modes':{r['type']:modes.get(r['type'],'apply') for r in e['decision']['accepted']},'time_methods':time_methods,'deferred':deferred,'rejected':e['decision']['rejected'],'status':'candidate_applied','global_personality_changed':False,'production_applied':False}
            db.execute('UPDATE b1_state SET revision=?,at=?,state=? WHERE scope=?',(s['revision']+1,e['occurred_at'],canonical(stored),e['scope']))
            db.execute('INSERT INTO b1_events VALUES(?,?,?,?)',(e['scope'],e['event_id'],h,canonical(receipt)))
            self._save_numeric(db,e['scope'],s['revision']+1,e['occurred_at'],s,p)
            return {'replayed':False,'receipt':receipt}
        return self._write(op)

    def advance_clock(self,scope,event_id,when,policy_revision,growth_frames=None,pressure_frames=None):
        # Explicit scheduler input. Reads never invoke this. Zero-time clocks
        # may create one idempotency receipt but do not change numeric state.
        source={'clock_id':ident(event_id),'scope':ident(scope),'at':at(when)}
        if growth_frames is not None:source['growth_frame_hashes']=[f.get('frame_hash') for f in growth_frames]
        if pressure_frames is not None:source['pressure_frame_hashes']=[f.get('frame_hash') for f in pressure_frames]
        envelope={'namespace':'b1-completed-turn-v1','scope':scope,'event_id':'clock:'+event_id,
          'occurred_at':source['at'],'policy_revision':policy_revision,
          'decision':{'accepted':[],'rejected':[],'source_hash':digest(source),'interpretation_is_not_ground_truth':True},'labels':[]}
        if growth_frames is not None:envelope['growth_frames']=growth_frames
        if pressure_frames is not None:envelope['pressure_frames']=pressure_frames
        return self.settle(envelope,expected_revision=self.read(scope)['revision'])

    def phase_event(self,scope,event_id,when,policy_revision,phase,source_kind=None):
        if phase not in ('sleep','wake'):raise ValueError('explicit phase required')
        source={'phase_id':ident(event_id),'scope':ident(scope),'at':at(when),'phase':phase}
        if source_kind is not None:
            if source_kind not in ('internal_owner_request','internal_rest_plan'):raise ValueError('explicit internal phase source')
            source['source_kind']=source_kind
        envelope={'namespace':'b1-completed-turn-v1','scope':scope,'event_id':'phase:'+event_id,
          'occurred_at':source['at'],'policy_revision':policy_revision,
          'decision':{'accepted':[],'rejected':[],'source_hash':digest(source),'interpretation_is_not_ground_truth':True},'labels':[],'phase':phase}
        if source_kind is not None:envelope['phase_source_kind']=source_kind
        return self.settle(envelope,expected_revision=self.read(scope)['revision'])

class PersonaB1Adapter:
    """Internal dependency injection, not a public authorization endpoint."""
    def __init__(self,store,scope,global_snapshot,*,result_resolver=None,growth_frame_loader=None,pressure_frame_loader=None):
        self.store=store;self.scope=ident(scope);self.result_resolver=result_resolver;self.store.growth_frame_loader=growth_frame_loader;self.store.pressure_frame_loader=pressure_frame_loader
        self.global_snapshot={}
        for group,keys in [('relationship',('affinity','dominance','defensiveness','trust')),('personality',('openness','conscientiousness','extraversion','agreeableness','neuroticism'))]:
            data=global_snapshot.get(group,{})
            self.global_snapshot[group]={key:None if data.get(key) is None else number(data[key]) for key in keys}
        names=('exploration_affinity','social_recovery_rate','connection_initiative','creative_satisfaction','challenge_tolerance','reflection_preference')
        data=global_snapshot.get('tendencies',{})
        self.global_snapshot['tendencies']={}
        for name in names:
            value=data.get(name)
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('finite existing tendency required')
            self.global_snapshot['tendencies'][name]=value

    def snapshot(self):
        s=self.store.read(self.scope);v=s['values'];affect={k:v[k] for k in FIELDS if not k.startswith('slow_')}
        with closing(self.store._connect(True)) as db:
            p=policy(json.loads(db.execute('SELECT policy FROM b1_state WHERE scope=?',(self.scope,)).fetchone()[0]))
        affect['subtypes']=[{'key':key,'label':label,'weight':weight} for key,weights in s['subtypes'].items() for label,weight in weights.items() if p['subtype_policy'] and weight>=p['subtype_policy']['visible_weight'] and s['values'][key] is not None];affect['mixed']=deepcopy(s['mixed']);affect['labels']=s['current_labels'] if p['presentation'] else s['labels'];affect['label_times']=deepcopy(s['label_times']);affect['uncertain_labels']=s['uncertain_labels'];affect['base_label']=s['base_label'];affect['cause_label']=deepcopy(s['cause_label']);affect['labels_updated_at']=s['labels_updated_at'];affect['mood_label']=affect['labels'][0] if affect['labels'] else None
        affect['lit']=deepcopy(s['lit']) if s['lit'] and datetime.fromisoformat(s['lit']['until'])>datetime.fromisoformat(s['at']) else None
        affect['light_marks']=deepcopy(s['light_marks'])
        affect['connection_phase']=None
        if p['presence_policy']:
            r=p['presence_policy'];pv=v['possessiveness'];cv=v['cling']
            if pv is not None and cv is not None and pv>=r['possessiveness_min']:
                if cv>=r['cling_min']:affect['connection_phase']='想黏着'
                elif s['last_conversation_at'] and (datetime.fromisoformat(s['at'])-datetime.fromisoformat(s['last_conversation_at'])).total_seconds()>=r['absence_hours']*3600:affect['connection_phase']='想念'
        from persona_b1_layers import display_choice
        affect['display']=display_choice(affect,datetime.fromisoformat(s['at']),p['display_policy'])
        affect['last_conversation_at']=s['last_conversation_at'];affect['growth_coverage']=deepcopy(s['growth_coverage'])
        affect['empathy_load']={key:sum(x['value'] for x in s['empathy_loads'].values() if x['key']==key) for key in ('anger','grieve')}
        return {'profile_id':self.scope,'affect':affect,'relationship':deepcopy(self.global_snapshot.get('relationship',{})),
                'personality':deepcopy(self.global_snapshot.get('personality',{})),'updated_at':s['at'],'revision':s['revision'],
                'conflict_episode':({'active':bool(s.get('grudge')),'reconciled':s['conflict_episode']['reconciled'],'soothes':s['conflict_episode']['soothes'],'conflicts':s['conflict_episode']['conflicts'],'close_reason':s['conflict_episode']['close_reason']} if s.get('conflict_episode') else None),'slow_axes':{'security':v['slow_security'],'confidence':v['slow_confidence'],'mood':deepcopy(s['slow_mood'])},'sleeping':s['sleeping'],'tendencies':deepcopy(self.global_snapshot['tendencies']),
                'fatigue':deepcopy(s['fatigue']),'source':'b1_candidate','private_narrative_included':False}
    def advance_clock(self,event_id,when,policy_revision,growth_frames=None,pressure_frames=None):
        result=self.store.advance_clock(self.scope,event_id,when,policy_revision,growth_frames,pressure_frames)
        return {**self.snapshot(),'b1_receipt':result}
    def set_phase(self,event_id,when,policy_revision,phase,source_kind=None):
        result=self.store.phase_event(self.scope,event_id,when,policy_revision,phase,source_kind)
        return {**self.snapshot(),'b1_receipt':result}
    def record_operation(self,result_ref,policy_revision):
        if self.result_resolver is None:raise ValueError('trusted canonical result resolver required')
        ident(result_ref);raw=self.result_resolver(result_ref)
        return self._record_verified_operation(raw,policy_revision)

    def _record_verified_operation(self,raw,policy_revision):
        # Trusted canonical resolver or authenticated internal result route only.
        expected={'scope','operation_id','status','proof','revision','result_hash','occurred_at'}
        if not isinstance(raw,dict) or set(raw)-{'helped'}!=expected or raw['scope']!=self.scope or raw['status']!='succeeded':raise ValueError('canonical successful result required')
        if raw['proof'] not in ('her_ok','first','hard','helped_ack'):raise ValueError('routine/self-declared work is not confidence proof')
        ident(raw['operation_id'])
        if isinstance(raw['revision'],bool) or not isinstance(raw['revision'],int) or raw['revision']<0:raise ValueError('canonical result revision required')
        if not isinstance(raw['result_hash'],str) or len(raw['result_hash'])!=64 or any(c not in '0123456789abcdef' for c in raw['result_hash']):raise ValueError('canonical result hash required')
        when=at(raw['occurred_at'])
        envelope={'namespace':'b1-completed-turn-v1','scope':self.scope,'event_id':'result:'+raw['operation_id'],
          'occurred_at':at(when),'policy_revision':policy_revision,'decision':{'accepted':[],'rejected':[],'source_hash':digest(raw),'interpretation_is_not_ground_truth':True},
          'labels':[],'operation_result':{key:raw[key] for key in ('operation_id','proof','status','result_hash')}}
        if 'helped' in raw:envelope['operation_result']['helped']=deepcopy(raw['helped'])
        result=self.store.settle(envelope,expected_revision=self.store.read(self.scope)['revision'])
        return {**self.snapshot(),'b1_receipt':result}
    def settle(self,envelope,turn):
        if 'phase' in envelope or 'phase_source_kind' in envelope or 'operation_result' in envelope:raise ValueError('lifecycle/result cannot be claimed by chat evaluator')
        if not isinstance(turn,dict) or set(turn)!={'conversation_id','user_event','assistant_event'}:raise ValueError('original completed turn required in memory')
        u,a=turn['user_event'],turn['assistant_event']
        if not isinstance(u,dict) or not isinstance(a,dict):raise ValueError('completed events required')
        for ev,role in ((u,'user'),(a,'assistant')):
            if ev.get('role',role)!=role or not isinstance(ev.get('content'),str):raise ValueError('turn role/text mismatch')
        src={'conversation_key':digest(ident(turn['conversation_id'])),'user_id':ident(str(u.get('id') or '').strip()),'assistant_id':ident(str(a.get('id') or '').strip()),'user':u['content'],'assistant':a['content']}
        if envelope['decision']['source_hash']!=digest(src) or envelope['event_id']!=src['user_id']:raise Conflict('turn source binding mismatch')
        decision_check(envelope['decision'])
        for r in envelope['decision']['accepted']:
            text=src[r['role']]
            if not 0<=r['start']<r['end']<=len(text) or digest(text[r['start']:r['end']])!=r['evidence_hash']:raise Conflict('source span binding mismatch')
        for row in envelope.get('correction_details',[]):
            for role in ('user','assistant'):
                lo=row.get(role+'_start');hi=row.get(role+'_end')
                if isinstance(lo,bool) or isinstance(hi,bool) or not isinstance(lo,int) or not isinstance(hi,int) or not 0<=lo<hi<=len(src[role]):raise Conflict('correction span binding mismatch')
            proof={'user':src['user'][row['user_start']:row['user_end']],'assistant':src['assistant'][row['assistant_start']:row['assistant_end']]}
            if row.get('evidence_hash')!=digest(proof):raise Conflict('correction evidence mismatch')
        if envelope.get('scope')!=self.scope:raise Conflict('adapter scope mismatch')
        envelope=deepcopy(envelope);envelope['conversation_observed_at']=at(envelope['occurred_at'])
        r=self.store.settle(envelope,expected_revision=self.store.read(self.scope)['revision'])
        return {**self.snapshot(),'b1_receipt':r}
