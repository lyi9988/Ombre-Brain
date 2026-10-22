"""B1 slow mood, source labels, base/cause layers. Policies are explicit."""
from datetime import datetime,timedelta,timezone
import math

BASE_LABELS=('joyful','secure','irritable','low','relaxed','tense','tired','calm')
def base_label(v,a,p):
    if v is None or a is None:return None
    if v>=p['pleasant_high'] and a>=p['active_high']:return 'joyful'
    if v>=p['pleasant_high']:return 'secure'
    if v<=p['pleasant_low'] and a>=p['active_high']:return 'irritable'
    if v<=p['pleasant_low']:return 'low'
    if v>=p['relaxed_valence'] and a<=p['relaxed_arousal']:return 'relaxed'
    if a>=p['tense_arousal']:return 'tense'
    if a<=p['tired_arousal']:return 'tired'
    return 'calm'

def update_layers(state,policy,now,source_labels=None,*,update_base=True):
    p=policy['presentation'];v=state['values']['valence'];a=state['values']['arousal']
    if update_base:
        candidate=base_label(v,a,p['base_thresholds']);old=state['base_label'];pending=state['pending_base']
        if candidate is None:
            state['base_label']=None;state['pending_base']=None
        else:
            margin=p['base_margin']
            deep=all(base_label(min(1,max(0,x)),min(1,max(0,y)),p['base_thresholds'])==candidate for x,y in [(v+margin,a),(v-margin,a),(v,a+margin),(v,a-margin)])
            if old is None or old==candidate or deep:
                state['base_label']=candidate;state['pending_base']=None
            elif pending and pending['label']==candidate:
                if (now-datetime.fromisoformat(pending['since'])).total_seconds()>=p['base_hold_hours']*3600:
                    state['base_label']=candidate;state['pending_base']=None
            else:state['pending_base']={'label':candidate,'since':now.isoformat()}
    # Label lifecycle is configured per label and source state, not a global
    # TTL which indiscriminately removes intimacy. Unknown axes stay unknown.
    current=[];uncertain=[]
    for label in state['labels']:
        rule=p['label_rules'].get(label)
        if rule is None:
            # Missing lifecycle configuration is not evidence that a supplied
            # feeling disappeared. Preserve source label + original timestamp.
            current.append(label);uncertain.append(label);continue
        key=rule['key'];level=state['values'].get(key)
        if level is None:uncertain.append(label)
        elif rule['minimum']<=level<=rule['maximum']:current.append(label)
    state['current_labels']=current;state['uncertain_labels']=uncertain
    # Original cause priority/hold; never replace a cause with a bare base label.
    choices=[(p['cause_priorities'][x],x) for x in (source_labels or []) if x in p['cause_priorities']]
    old=state['cause_label']
    if choices:
        priority,label=max(choices)
        still_held=old and (now-datetime.fromisoformat(old['since'])).total_seconds()<p['cause_hold_hours']*3600
        if not still_held or priority>old['priority']:
            state['cause_label']={'label':label,'priority':priority,'since':now.isoformat()}
    cause=state['cause_label']
    if cause and cause['label'] not in current:
        held=(now-datetime.fromisoformat(cause['since'])).total_seconds()<p['cause_hold_hours']*3600
        if not held:state['cause_label']=None

def integrate_mean(fn,a,b,tol=1e-8,depth=14):
    """Adaptive Simpson integral of bounded valence; unknown => uncovered."""
    evaluations=0
    def value(t):
        nonlocal evaluations
        evaluations+=1
        if evaluations>8192:raise RuntimeError('mood integration budget exhausted')
        return fn(t)
    fa=value(a);fb=value(b);m=(a+b)/2;fm=value(m)
    if None in (fa,fb,fm):return None
    def recurse(a,b,fa,fm,fb,whole,budget,d):
        m=(a+b)/2;fl=value((a+m)/2);fr=value((m+b)/2)
        if fl is None or fr is None:return None
        l=(m-a)*(fa+4*fl+fm)/6;r=(b-m)*(fm+4*fr+fb)/6
        delta=l+r-whole
        if abs(delta)<=15*budget:return l+r+delta/15
        if not d:raise RuntimeError('mood integration precision budget exhausted')
        x=recurse(a,m,fa,fl,fm,l,budget/2,d-1);y=recurse(m,b,fm,fr,fb,r,budget/2,d-1)
        return None if x is None or y is None else x+y
    return recurse(a,b,fa,fm,fb,(b-a)*(fa+4*fm+fb)/6,tol,depth)

def record_mood(state,start,end,sample,p,*,max_step_hours=1.0):
    """Time-weighted UTC-offset calendar-day observations, not request counts.
    Retain only projection window; permanent event receipts remain unchanged.
    """
    tz=timezone(timedelta(minutes=p['offset_minutes']));lastday=end.astimezone(tz).date()
    firstday=lastday-timedelta(days=p['window_days']-1)
    keepstart=datetime.combine(firstday,datetime.min.time(),tzinfo=tz)
    cursor=max(start,keepstart)
    while cursor<end:
        local=cursor.astimezone(tz);nextday=datetime.combine(local.date()+timedelta(days=1),datetime.min.time(),tzinfo=tz)
        stop=min(end,nextday);lo=(cursor-start).total_seconds()/3600;hi=(stop-start).total_seconds()/3600
        parts=max(1,math.ceil((hi-lo)/max_step_hours))
        if parts>8192:raise RuntimeError('mood trajectory budget exhausted; advance in chronological chunks')
        integral=0.0
        for i in range(parts):
            piece=integrate_mean(sample,lo+(hi-lo)*i/parts,lo+(hi-lo)*(i+1)/parts,tol=1e-8/parts)
            if piece is None:integral=None;break
            integral+=piece
        key=local.date().isoformat()
        row=state['mood_days'].setdefault(key,{'integral':0.0,'covered_hours':0.0,'unknown_hours':0.0})
        if integral is None:row['unknown_hours']+=hi-lo
        else:row['integral']+=integral;row['covered_hours']+=hi-lo
        cursor=stop
    state['mood_days']={k:v for k,v in state['mood_days'].items() if k>=firstday.isoformat()}
    hours=sum(v['covered_hours'] for v in state['mood_days'].values());unknown=sum(v['unknown_hours'] for v in state['mood_days'].values())
    mean=sum(v['integral'] for v in state['mood_days'].values())/hours if hours else None
    state['slow_mood']={'value':None if mean is None or unknown else min(1,max(-1,(mean-p['midpoint'])/p['span'])),
                        'covered_hours':hours,'unknown_hours':unknown,'window_days':p['window_days'],'unit':'-1..1'}


def track_base_segment(state,policy,start,end,sample,*,budget):
    """UTC minute grid, independent of caller refresh frequency.
    Event jumps are handled separately at their exact occurrence time.
    """
    first=(math.floor(start.timestamp()/60)+1)*60
    count=max(0,math.floor((end.timestamp()-first)/60)+1)
    if count>budget:raise RuntimeError('base label observation budget exhausted; advance in chronological chunks')
    saved=state['values']
    try:
        for i in range(count):
            now=datetime.fromtimestamp(first+i*60,tz=start.tzinfo)
            state['values']=sample((now-start).total_seconds()/3600)
            update_layers(state,policy,now)
    finally:state['values']=saved
    return count

def rhythm_boundaries(start,end,p):
    tz=timezone(timedelta(minutes=p['offset_minutes']));first=start.astimezone(tz).date();last=end.astimezone(tz).date()
    days=(last-first).days+1
    if days>2048:raise RuntimeError('rhythm boundary work budget exhausted')
    result=[]
    for i in range(days):
        day=first+timedelta(days=i)
        for minute in (p['night_start'],p['night_end'],p['dawn_start'],p['dawn_end']):
            boundary=datetime.combine(day,datetime.min.time(),tzinfo=tz)+timedelta(minutes=minute)
            if start<boundary<end:result.append(boundary)
    return result

def rhythm_flags(now,p):
    t=now.astimezone(timezone(timedelta(minutes=p['offset_minutes'])));minute=t.hour*60+t.minute+t.second/60
    def inside(a,b):return a<=minute<b if a<b else minute>=a or minute<b if a>b else False
    return inside(p['night_start'],p['night_end']),inside(p['dawn_start'],p['dawn_end'])

PRESSURE_KEYS=('possess','monitor','share','libido','curiosity','boredom','duty','reflection','grieve','anger','favored')
def fatigue_step(value,phase,pressure,hours,p):
    """Independent fatigue, never an inverse energy estimate."""
    if not hours:return value,'unchanged'
    if value is None:return None,'unknown_seed'
    if phase is True:return max(0,value-p['recover_per_hour']*hours),'sleep_recovery'
    if phase is None:return None,'unknown_sleep_phase'
    known=isinstance(pressure,dict) and set(pressure)==set(PRESSURE_KEYS) and all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in pressure.values())
    if not known:return None,'unknown_pressure'
    return (min(p['maximum'],value+p['build_per_hour']*hours),'high_pressure') if sum(pressure.values())/len(PRESSURE_KEYS)>p['pressure_threshold'] else (value,'known_pressure_below_or_equal')

def display_choice(affect,now,p):
    """Optional source-backed primary word, not deletion of coexisting feelings.

    The upstream rawShown thresholds are reference only. Callers must provide
    all threshold/priority choices; the default is honest coexisting signals.
    Existing cause_label has already passed its persistent cause-hold layer.
    """
    lit=affect.get('lit');cause=affect.get('cause_label');phase=affect.get('connection_phase')
    available=[]
    if lit:available.append(lit['word'])
    if cause:available.append(cause['label'])
    if phase:available.append(phase)
    available=list(dict.fromkeys(available))
    if not p or p['mode']=='coexist':return {'mode':'coexist','primary':None,'secondary':available,'status':'coexisting'}
    valence=affect.get('valence');candidates=[];unknown=False
    if lit:
        if valence is None:unknown=True
        elif valence>=p['lit_min_valence']:
            age=(now-datetime.fromisoformat(lit['at'])).total_seconds()/60
            candidates.append((p['fresh_lit_priority'] if age<p['fresh_minutes'] else p['aged_lit_priority'],lit['word']))
    if cause:
        if valence is None:unknown=True
        elif valence<=p['cause_max_valence']:candidates.append((cause['priority'],cause['label']))
    if phase:
        if valence is None:unknown=True
        elif valence<=p['phase_max_valence']:candidates.append((p['phase_priority'],phase))
    primary=max(candidates,key=lambda x:x[0])[1] if candidates else None
    return {'mode':'primary','primary':primary,'secondary':[v for v in available if v!=primary],'status':'partial_unknown' if unknown else 'selected' if primary else 'no_eligible_primary'}
