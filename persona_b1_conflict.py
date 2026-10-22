"""Source-bound conflict episodes; no keyword parser, no stored raw quotation."""
import math
from datetime import datetime,timedelta

def grief_peak(value,anger,ka,kg,ratio):
    if value is None or anger is None or not ka or not ratio or not anger:return None
    forcing=ratio*ka*anger
    if ka==kg:
        return (forcing-kg*value)/(kg*forcing) if kg and forcing>kg*value else None
    d=forcing/(kg-ka);c=value-d
    quotient=-kg*c/(ka*d) if ka*d else 0
    if quotient<=0:return None
    t=math.log(quotient)/(kg-ka)
    return t if t>0 else None

def transferred_grief(value,anger,anger_rate,grief_rate,ratio,hours):
    if value is None:return None
    if not ratio or not anger_rate:return value*math.exp(-grief_rate*hours)
    if anger is None:return None
    ka=anger_rate;kg=grief_rate
    def raw(v,a,t):
        if ka==kg:addition=ratio*ka*a*t*math.exp(-kg*t)
        else:
            d=kg-ka
            if abs(d*t)<1e-5:addition=ratio*ka*a*math.exp(-kg*t)*math.expm1(d*t)/d
            else:addition=ratio*ka*a*(math.exp(-ka*t)-math.exp(-kg*t))/d
        return v*math.exp(-kg*t)+addition
    peak=grief_peak(value,anger,ka,kg,ratio)
    highest=raw(value,anger,min(hours,peak)) if peak else max(value,raw(value,anger,hours))
    if highest<=1:return min(1,max(0,raw(value,anger,hours)))
    # Reflect at physical 1, never carry discarded overflow into later steps.
    release=math.log(ratio*ka*anger/kg)/ka if ka and kg else math.inf
    if hours<=release:return 1.0
    return min(1,max(0,raw(1.0,anger*math.exp(-ka*release),hours-release)))

def expiry(state,p):
    ep=state.get('conflict_episode')
    if not ep or not state.get('grudge') or ep['reconciled']:return None
    hours=p['soothed_fade_hours'] if ep['soothes'] else p['max_fade_hours']
    return datetime.fromisoformat(ep['last_at'])+timedelta(hours=hours)

def retire(state,p,now):
    end=expiry(state,p)
    anger=state['values']['anger']
    if state.get('grudge') and ((end is not None and now>=end) or (anger is not None and anger<=p['clear_below']+1e-14)):
        state['grudge']=None
        state['conflict_episode']['closed_at']=now.isoformat()
        state['conflict_episode']['close_reason']='faded'

def apply_episode(state,p,accepted,event_id,now):
    ep=state.get('conflict_episode')
    for row in accepted:
        kind=row['type']
        if kind=='conflict':
            if state.get('grudge') is None:
                state['grudge']={'event_id':event_id,'evidence_hash':row['evidence_hash'],'at':now.isoformat()}
                ep={'started_at':now.isoformat(),'last_at':now.isoformat(),'conflicts':0,'soothes':0,'reconciled':False,'reconciled_at':None,'closed_at':None,'close_reason':None}
                state['conflict_episode']=ep
            ep['conflicts']+=1;ep['last_at']=now.isoformat()
        elif ep and state.get('grudge') and not ep['reconciled']:
            if kind=='reconciliation':
                ep['reconciled']=True;ep['reconciled_at']=now.isoformat();ep['closed_at']=now.isoformat();ep['close_reason']='reconciled';state['grudge']=None
            elif kind in p['soothe_types']:
                ratio=min(p['soothe_max'],p['soothe_first']+p['soothe_step']*ep['soothes'])
                a=state['values']['anger']
                if a is not None and a>p['soothe_floor']:state['values']['anger']=max(p['soothe_floor'],a*(1-ratio))
                ep['soothes']+=1
