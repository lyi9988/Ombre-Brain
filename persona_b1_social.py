"""Bound empathy shares and event lights; no name/keyword inference."""
import math
from datetime import datetime,timedelta,timezone
from persona_inner_state import bound_observation, safe_source_refs
SUBS={'心疼':'grieve','不平':'anger','想帮忙':'expression','替人高兴':None}
CLOSENESS=('owner','family','known','stranger','unknown')
LIGHTS=('心动','害羞','得意','骄傲')

def validate_policies(p,number):
    e=p.get('empathy_policy',{})
    if e:
        if set(e)!={'base','weights','caps','daily_caps','dedupe_hours','helped_relief','offset_minutes','impulses'}:raise ValueError('complete empathy policy')
        number(e['base']);number(e['helped_relief']);number(e['dedupe_hours'],0,8760);number(e['offset_minutes'],-840,840)
        for key in ('weights','caps','daily_caps'):
            if set(e[key])!=set(CLOSENESS)-{'unknown'}:raise ValueError('explicit closeness policies')
            for value in e[key].values():
                if value is not None:number(value)
        if any(v is None for v in e['weights'].values()):raise ValueError('explicit known weights')
        if set(e['impulses'])!=set(SUBS):raise ValueError('all empathy impulses explicit')
        for r in e['impulses'].values():
            if set(r)!={'valence','arousal'}:raise ValueError('two coordinates')
            for value in r.values():number(value,-1,1)
    l=p.get('light_policy',{})
    if l:
        if set(l)!={'minutes','light_needed','window_minutes','repeat_factor','min_minutes','offset_minutes','proof_confidence_min'}:raise ValueError('complete light policy')
        if set(l['minutes'])!=set(LIGHTS):raise ValueError('all light durations')
        for n in l['minutes'].values():number(n,0,10080)
        if isinstance(l['light_needed'],bool) or not isinstance(l['light_needed'],int) or not 1<=l['light_needed']<=1000:raise ValueError('integer light count')
        number(l['window_minutes'],0,1440);number(l['repeat_factor']);number(l['min_minutes'],0,10080);number(l['offset_minutes'],-840,840);number(l['proof_confidence_min'])

def bound_detail(detail,kind,accepted):
    if not isinstance(detail,dict) or detail.get('type')!=kind:raise ValueError('typed source detail')
    return bound_observation(detail,accepted,kind)

def validate_empathy_detail(detail,accepted):
    observation=bound_detail(detail,'empathy',accepted)
    if set(detail)!={'observation_id','type','sub','subject_ref','incident_ref','closeness','identity_status','evidence_hash','subject_binding'} or detail['sub'] not in SUBS or detail['closeness'] not in CLOSENESS:raise ValueError('complete empathy subject binding')
    for key in ('subject_ref','incident_ref'):
        if not isinstance(detail[key],str) or not 0<len(detail[key])<=256:raise ValueError('bounded subject/incident')
    if observation['subject_binding'] is None or detail['subject_binding']!=observation['subject_binding']:raise ValueError('empathy mention identity mismatch')
    if detail['closeness']=='owner' and observation['target']=='third_party':raise ValueError('third-party source cannot resolve to owner')
    if detail['identity_status'] not in ('resolved','unresolved'):raise ValueError('explicit identity status')
    return observation

def validate_light_detail(detail,accepted):
    if not isinstance(detail,dict) or set(detail)!={'observation_id','type','sub','strength','evidence_hash'} or detail['type'] not in ('affection','intimacy','empathy') or detail['strength'] not in ('light','heavy'):raise ValueError('complete light detail')
    bound_detail(detail,detail['type'],accepted)
    word=detail['sub']
    if word not in ('心动','害羞','骄傲') or (word=='骄傲')!=(detail['type']=='empathy'):raise ValueError('light subject mismatch')

def day(now,offset):return now.astimezone(timezone(timedelta(minutes=offset))).date().isoformat()

def decay_loads(state,laws,hours):
    for load in state['empathy_loads'].values():
        half=laws[load['key']]['half_life_hours']
        if half:load['value']*=math.exp(-math.log(2)*hours/half)

def cap_loads(state):
    # Generic relief must not leave an external share greater than its total.
    for key in ('anger','grieve'):
        total=state['values'][key]
        if total is None:continue
        rows=[x for x in state['empathy_loads'].values() if x['key']==key];amount=sum(x['value'] for x in rows)
        if amount>total and amount:
            for row in rows:row['value']*=total/amount

def apply_empathy(state,p,d,accepted,now):
    validate_empathy_detail(d,accepted)
    if d['identity_status']=='unresolved' or d['closeness']=='unknown':return {'status':'unknown_closeness','key':SUBS[d['sub']]}
    if not p:return {'status':'policy_not_configured','key':SUBS[d['sub']]}
    key=SUBS[d['sub']]
    if key=='expression':return {'status':'aiz_owned','key':key}
    from b1_protocol import digest
    identity=digest([d['subject_ref'],d['incident_ref'],d['sub']])
    seen=state['empathy_seen'].get(identity)
    if seen and (now-datetime.fromisoformat(seen)).total_seconds()<p['dedupe_hours']*3600:return {'status':'duplicate_incident','key':key}
    state['empathy_seen'][identity]=now.isoformat()
    weight=p['weights'][d['closeness']]
    add=p['base']*weight
    if key:
        value=state['values'][key]
        if value is None:return {'status':'unknown_state','key':key}
        cap=p['caps'][d['closeness']]
        relevant=sum(x['value'] for x in state['empathy_loads'].values() if x['key']==key and x['closeness']==d['closeness'])
        if cap is not None:add=min(add,max(0,cap-(value if d['closeness']=='owner' else relevant)))
        today=day(now,p['offset_minutes']);daily_key=today+':'+d['closeness'];used=state['empathy_daily'].get(daily_key,0);dailycap=p['daily_caps'][d['closeness']]
        if dailycap is not None:add=min(add,max(0,dailycap-used))
        add=min(add,1-value);state['values'][key]+=add;state['empathy_daily'][daily_key]=used+add
        if d['closeness']!='owner' and add:
            row=state['empathy_loads'].setdefault(identity,{'key':key,'value':0,'subject_ref':d['subject_ref'],'incident_ref':d['incident_ref'],'sub':d['sub'],'closeness':d['closeness']})
            row['value']+=add
        if not add:return {'status':'capped','key':key}
        if add:state['subtypes'].setdefault(key,{})[d['sub']]=min(1,state['subtypes'].get(key,{}).get(d['sub'],0)+add)
    for coordinate,delta in p['impulses'][d['sub']].items():
        v=state['values'][coordinate]
        if v is not None:state['values'][coordinate]=min(1,max(0,v+delta*weight*((1-v) if delta>0 else v)))
    return {'status':'applied' if add or key is None else 'capped','key':key,'observation_id':d['observation_id'],'subject_ref':d['subject_ref'],'incident_ref':d['incident_ref']}

def help_share(state,p,subject_ref,incident_ref,sub):
    if sub not in SUBS or not subject_ref or not incident_ref:raise ValueError('help result must identify corresponding incident')
    key=SUBS[sub]
    if key=='expression':return {'status':'aiz_owned'}
    if key is None or not p:return {'status':'no_brain_share'}
    if state['values'][key] is None:return {'status':'unknown_state'}
    dropped=0
    for row in state['empathy_loads'].values():
        if row['subject_ref']==subject_ref and row['incident_ref']==incident_ref and row['sub']==sub:
            drop=min(row['value']*p['helped_relief'],state['values'][key]);row['value']-=drop;state['values'][key]-=drop;dropped+=drop
    return {'status':'relieved' if dropped else 'no_corresponding_share','amount':dropped}

def light(state,p,d,accepted,now,*,operation=False,source_refs=None):
    if not operation:validate_light_detail(d,accepted)
    if not p:return
    if operation:
        confidence=state['values']['slow_confidence']
        if confidence is None or confidence<p['proof_confidence_min']:return
        word='得意'
    else:
        word=d['sub']
        if word!='骄傲' and d['strength']=='light' and d['type']!='intimacy' and word!='害羞':
            state['light_taps']=[t for t in state['light_taps'] if (now-datetime.fromisoformat(t)).total_seconds()<=p['window_minutes']*60]+[now.isoformat()]
            if len(state['light_taps'])<p['light_needed']:return
        state['light_taps']=[]
    minutes=p['minutes'][word]
    if word in ('心动','害羞'):
        today=day(now,p['offset_minutes']);count=state['heart_days'].get(today,0);state['heart_days'][today]=count+1
        minutes=max(p['min_minutes'],minutes*p['repeat_factor']**count)
    refs=safe_source_refs(source_refs)
    state['lit']={'word':word,'at':now.isoformat(),'until':(now+timedelta(minutes=minutes)).isoformat(),'source_refs':refs}
    state['light_marks'].append({'word':word,'at':now.isoformat(),'source_refs':refs})
    state['light_marks']=state['light_marks'][-128:]
    return dict(state['lit'])
