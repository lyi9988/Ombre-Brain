"""Shared pure B1 growth protocol. Identical source must be used by both hosts.

No clock, database, state writer, owner text, model call or implicit coefficients.
Hash binds the complete canonical frame except its own hash field. Verification
recomputes all factors from the explicit policy and typed source, not caller math.
"""
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import hashlib
import json
import math


def finite(value):
    try:return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)
    except OverflowError:return False


def identity(value,field):
    if not isinstance(value,str) or not value.strip() or len(value)>256:raise ValueError(field+': invalid identity')
    return value


def stamp(value):
    if not isinstance(value,str) or len(value)>40:return None
    try:
        parsed=datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
        return parsed.astimezone(timezone.utc).isoformat() if parsed.tzinfo is not None and parsed.utcoffset() is not None else None
    except (ValueError,OverflowError):return None


def canonical(value):
    def normalize(v):
        if v is None or isinstance(v,(str,bool)):return v
        if isinstance(v,(int,float)):
            if not finite(v):raise ValueError('finite JSON number required')
            return 0.0 if isinstance(v,float) and v==0 else v
        if isinstance(v,list):return [normalize(x) for x in v]
        if isinstance(v,dict) and all(isinstance(k,str) for k in v):return {k:normalize(x) for k,x in v.items()}
        raise ValueError('JSON values required')
    return json.dumps(normalize(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

FRAME_SCHEMA='b1-growth-frame-v2'

TARGETS={'expression','curiosity','boredom','duty','reflection','longing','possessiveness','protective_drive','libido','novelty','creativity','autonomy','competence','solitude'}
TENDENCY_SOURCES=('exploration_affinity','social_recovery_rate','connection_initiative','creative_satisfaction','challenge_tolerance','reflection_preference')
def validate_policy(raw):
    if not isinstance(raw,dict) or not {'revision','grid_seconds','source_scope','source_policy_revision','targets'}<=set(raw) or set(raw)-{'revision','grid_seconds','source_scope','source_policy_revision','targets','anchor_utc'}:raise ValueError('explicit modulation policy required')
    identity(raw['revision'],'modulation revision');identity(raw['source_scope'],'source scope');identity(raw['source_policy_revision'],'source policy')
    if 'anchor_utc' in raw:
        anchor=raw['anchor_utc']
        if not isinstance(anchor,str) or stamp(anchor)!=anchor or not anchor.endswith('+00:00'):raise ValueError('anchor_utc must be canonical UTC ISO')
    grid=raw['grid_seconds']
    if isinstance(grid,bool) or not isinstance(grid,int) or not 1<=grid<=3600:raise ValueError('bounded explicit grid required')
    if not isinstance(raw['targets'],dict) or not set(raw['targets'])<=TARGETS:raise ValueError('unknown growth target')
    for target,r in raw['targets'].items():
        if not isinstance(r,dict) or not {'emotion','drives','floor','ceiling'}<=set(r) or set(r)-{'emotion','drives','floor','ceiling','fatigue'}:raise ValueError('complete growth rule')
        if not all(finite(r[k]) for k in ('floor','ceiling')) or not 0<=r['floor']<=r['ceiling']<=10:raise ValueError('invalid explicit factor bounds')
        if not isinstance(r['drives'],dict) or not set(r['drives'])<= {'anger','grieve','favored'}:raise ValueError('bounded drive sources')
        if any(not finite(v) or abs(v)>10 for v in r['drives'].values()):raise ValueError('invalid drive coefficient')
        fatigue=r.get('fatigue')
        if fatigue is not None:
            if not isinstance(fatigue,dict) or set(fatigue)!={'slope','maximum'} or not all(finite(v) for v in fatigue.values()) or abs(fatigue['slope'])>10 or not 0<fatigue['maximum']<=1:raise ValueError('explicit fatigue transform required')
        e=r['emotion']
        if e is not None:
            if not isinstance(e,dict) or set(e)!={'valence_slope','arousal_slope','valence_center','valence_span','arousal_center','arousal_span'}:raise ValueError('complete emotion transform')
            if any(not finite(v) for v in e.values()) or any(e[k]<=0 for k in ('valence_span','arousal_span')):raise ValueError('finite nonzero emotion scale')
    return deepcopy(raw)

def grid_start(at,seconds,anchor_utc=None):
    if not isinstance(at,datetime) or at.tzinfo is None or at.utcoffset() is None:raise ValueError('aware time required')
    when=at.astimezone(timezone.utc)
    origin=datetime(1970,1,1,tzinfo=timezone.utc) if anchor_utc is None else datetime.fromisoformat(anchor_utc)
    if anchor_utc is not None and (stamp(anchor_utc)!=anchor_utc or not anchor_utc.endswith('+00:00')):raise ValueError('canonical UTC anchor required')
    steps=(when-origin)//timedelta(seconds=seconds)
    return origin+steps*timedelta(seconds=seconds)

def derive_frame(policy,at,source):
    p=validate_policy(policy);start=grid_start(at,p['grid_seconds'],p.get('anchor_utc'));end=start+timedelta(seconds=p['grid_seconds'])
    if 'anchor_utc' in p and start<datetime.fromisoformat(p['anchor_utc']):raise ValueError('frame predates initialization anchor')
    if not isinstance(source,dict) or not {'scope','policy_revision','revision','as_of','values'}<=set(source) or set(source)-{'scope','policy_revision','revision','as_of','values','availability'}:raise ValueError('typed source only')
    if source['scope']!=p['source_scope'] or source['policy_revision']!=p['source_policy_revision']:raise ValueError('wrong source authority')
    availability=source.get('availability','available')
    if availability not in ('available','unavailable','future_snapshot'):raise ValueError('invalid source availability')
    revision=source['revision'];as_of=stamp(source['as_of'])
    valid_identity=availability=='available' and isinstance(revision,int) and not isinstance(revision,bool) and finite(revision) and revision>=0 and as_of is not None
    if valid_identity and datetime.fromisoformat(as_of)>start:raise ValueError('future snapshot cannot describe grid start')
    values=source['values'] if isinstance(source['values'],dict) else {}
    out={}
    for key,r in p['targets'].items():
        needed=set(r['drives'])
        if r['emotion'] is not None:needed|={'valence','arousal'}
        if r.get('fatigue') is not None:needed.add('fatigue')
        known=valid_identity and all(finite(values.get(k)) and 0<=values[k]<=1 for k in needed)
        if not known:
            out[key]={'factor':None,'status':'unknown','depends_on':sorted(needed)};continue
        factor=1.0
        for k,slope in r['drives'].items():factor*=max(0.0,1+slope*values[k])
        if r['emotion'] is not None:
            e=r['emotion'];dv=max(-1,min(1,(values['valence']-e['valence_center'])/e['valence_span']));da=max(-1,min(1,(values['arousal']-e['arousal_center'])/e['arousal_span']))
            factor*=max(r['floor'],min(r['ceiling'],1+e['valence_slope']*dv+e['arousal_slope']*da))
        if r.get('fatigue') is not None:
            factor*=max(0.0,1+r['fatigue']['slope']*min(values['fatigue'],r['fatigue']['maximum']))
        if not finite(factor):raise ValueError('factor overflow')
        factor=max(r['floor'],min(r['ceiling'],factor))
        out[key]={'factor':factor,'status':'known','depends_on':sorted(needed)}
    clean_source={'scope':p['source_scope'],'policy_revision':p['source_policy_revision'],'revision':revision if valid_identity else None,'as_of':as_of,
        'values':{k:values[k] if finite(values.get(k)) and 0<=values[k]<=1 else None for k in ('valence','arousal','anger','grieve','favored','fatigue')}}
    clean_source['availability']=availability
    # Optional canonical source keys preserve old historical frame hashes when
    # absent; new normalized sources carry them without a second state read.
    clean_source['values'].update({k: values[k] if valid_identity and finite(values.get(k)) and 0<=values[k]<=1 else None
                                  for k in TENDENCY_SOURCES if k in values})
    frame={'schema':FRAME_SCHEMA,'policy_revision':p['revision'],'grid_start':start.isoformat(),'grid_end':end.isoformat(),
        'policy_hash':digest(p),'source':clean_source,'targets':out,'coverage_complete':all(v['status']=='known' for v in out.values())}
    frame['frame_hash']=digest(frame)
    return frame

def verify_frame(policy,frame):
    if not isinstance(frame,dict):raise ValueError('typed growth frame required')
    try:
        start=datetime.fromisoformat(frame['grid_start'])
        expected=derive_frame(policy,start,frame['source'])
    except (KeyError,TypeError) as exc:raise ValueError('incomplete growth frame') from exc
    if canonical(expected)!=canonical(frame):raise ValueError('growth frame policy/content mismatch')
    return deepcopy(expected)

