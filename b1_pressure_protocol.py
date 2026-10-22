"""Pure original-eleven pressure source protocol; no fatigue/energy inference.

Hash is integrity, not authorization. The coordinator must obtain snapshots
from the actual versioned hosts. Only bound numeric fields survive allowlisting.
"""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math

KEYS=('possess','monitor','share','libido','curiosity','boredom','duty','reflection','grieve','anger','favored')
HOSTS=('aiz','brain')

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(value):return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

def ident(value):
    if not isinstance(value,str) or not value or value!=value.strip() or len(value)>256:raise ValueError('explicit identity required')
    return value

def utc(value):
    if not isinstance(value,str):raise ValueError('UTC timestamp required')
    try:d=datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    except ValueError as exc:raise ValueError('invalid timestamp') from exc
    if d.tzinfo is None or d.utcoffset() is None:raise ValueError('aware timestamp required')
    return d.astimezone(timezone.utc)

def number(value,maximum):
    try:valid=not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=maximum
    except OverflowError:valid=False
    if not valid:raise ValueError('finite in-range pressure value required; unknown must be None')
    return value

def validate_policy(raw):
    required={'revision','grid_seconds','hosts','bindings'}
    if not isinstance(raw,dict) or not required<=set(raw) or set(raw)-required-{'anchor_utc'}:raise ValueError('explicit pressure policy required')
    ident(raw['revision']);g=raw['grid_seconds']
    if type(g) is not int or not 1<=g<=3600:raise ValueError('bounded explicit grid required')
    if 'anchor_utc' in raw and utc(raw['anchor_utc']).isoformat()!=raw['anchor_utc']:raise ValueError('canonical UTC anchor required')
    if not isinstance(raw['hosts'],dict) or set(raw['hosts'])!=set(HOSTS):raise ValueError('two explicit host identities required')
    for row in raw['hosts'].values():
        if not isinstance(row,dict) or set(row)!={'scope','policy_revision'}:raise ValueError('explicit host policy identity required')
        ident(row['scope']);ident(row['policy_revision'])
    if not isinstance(raw['bindings'],dict) or set(raw['bindings'])!=set(KEYS):raise ValueError('all original eleven logical bindings required')
    seen=set()
    for row in raw['bindings'].values():
        if not isinstance(row,dict) or set(row)!={'host','key','unit'} or row['host'] not in HOSTS or row['unit'] not in ('0..100','0..1'):raise ValueError('one typed source per logical pressure required')
        ident(row['key'])
        if row['host']=='brain' and row['unit']!='0..1':raise ValueError('Brain unit must be 0..1')
        pair=(row['host'],row['key'])
        if pair in seen:raise ValueError('one source cannot be counted as two original pressures')
        seen.add(pair)
    return deepcopy(raw)

def build_pressure_frame(policy,start,aiz_snapshot,brain_snapshot):
    p=validate_policy(policy)
    if not isinstance(start,datetime) or start.tzinfo is None or start.utcoffset() is None:raise ValueError('aware grid time required')
    at=start.astimezone(timezone.utc)
    origin=utc(p['anchor_utc']) if 'anchor_utc' in p else datetime(1970,1,1,tzinfo=timezone.utc)
    seconds=timedelta(seconds=p['grid_seconds']);begin=origin+((at-origin)//seconds)*seconds
    if 'anchor_utc' in p and begin<origin:raise ValueError('pressure predates initialization')
    end=begin+seconds;sources={};origins={};values={};snapshots={'aiz':aiz_snapshot,'brain':brain_snapshot}
    for host in HOSTS:
        source=snapshots[host];expected=p['hosts'][host]
        if source is None:
            sources[host]=None;origins[host]={'scope':expected['scope'],'revision':None,'as_of':None};continue
        if not isinstance(source,dict) or set(source)!={'scope','policy_revision','revision','as_of','values'}:raise ValueError('typed host snapshot required')
        if source['scope']!=expected['scope'] or source['policy_revision']!=expected['policy_revision']:raise ValueError('wrong host scope or policy')
        if type(source['revision']) is not int or source['revision']<0:raise ValueError('host revision required')
        observed=utc(source['as_of'])
        if observed>begin:raise ValueError('future source cannot describe pressure grid start')
        if not isinstance(source['values'],dict):raise ValueError('host values object required')
        selected={}
        for binding in p['bindings'].values():
            if binding['host']!=host:continue
            key=binding['key'];value=source['values'].get(key)
            selected[key]=None if value is None else number(value,100 if binding['unit']=='0..100' else 1)
        sources[host]={'scope':source['scope'],'policy_revision':source['policy_revision'],'revision':source['revision'],'as_of':observed.isoformat(),'values':selected}
        origins[host]={'scope':source['scope'],'revision':source['revision'],'as_of':observed.isoformat()}
    for logical in KEYS:
        binding=p['bindings'][logical];s=sources[binding['host']];v=s['values'][binding['key']] if s else None
        values[logical]=None if v is None else v/(100 if binding['unit']=='0..100' else 1)
    times=[origins[h]['as_of'] for h in HOSTS]
    as_of=min(times) if all(x is not None for x in times) else None
    frame={'grid_start':begin.isoformat(),'grid_end':end.isoformat(),'policy_hash':digest(p),'sources':sources,'pressure':{'as_of':as_of,'origins':origins,'values':values}}
    frame['frame_hash']=digest(frame)
    return frame

def verify_pressure_frame(policy,frame):
    if not isinstance(frame,dict):raise ValueError('typed pressure frame required')
    try:expected=build_pressure_frame(policy,utc(frame['grid_start']),frame['sources']['aiz'],frame['sources']['brain'])
    except (KeyError,TypeError) as exc:raise ValueError('incomplete pressure frame') from exc
    if canonical(expected)!=canonical(frame):raise ValueError('pressure frame content/policy mismatch')
    return deepcopy(expected)
