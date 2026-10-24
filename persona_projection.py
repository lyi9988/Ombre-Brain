"""Owner-safe Persona projection with no transport or storage dependencies."""
from __future__ import annotations
import math
from datetime import datetime
from persona_inner_state import STATE_SCHEMA, normalize_tendencies, safe_source_refs

def _stamp(value):
    try:
        parsed=datetime.fromisoformat(value)
        return parsed.isoformat() if parsed.tzinfo is not None else None
    except (TypeError,ValueError):return None


def _finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):return None
    return float(value)

def _unit(value):
    value=_finite(value)
    return value if value is not None and 0<=value<=1 else None

def _b1_projection(state,updated_at,profile_id):
    affect=state.get("affect",{});rel=state.get("relationship",{})
    def level(v,lo,hi):
        n=_unit(v)
        return None if n is None else projection_level(n,lo,hi)
    labels=affect.get("labels",[])
    labels=[v for v in labels if isinstance(v,str) and 0<len(v)<=60][:8] if isinstance(labels,list) else []
    raw_label_sources=affect.get('label_sources',{})
    raw_label_sources=raw_label_sources if isinstance(raw_label_sources,dict) else {}
    label_sources={label:safe_source_refs(raw_label_sources.get(label)) for label in labels}
    unattributed=affect.get('unattributed_labels',[])
    unattributed=[label for label in unattributed if isinstance(label,str) and 0<len(label)<=60] if isinstance(unattributed,list) else []
    unattributed=list(dict.fromkeys(unattributed+[label for label in labels if not label_sources[label]]))
    labels=[label for label in labels if label_sources[label]]
    label_sources={label:label_sources[label] for label in labels}
    drive_labels=[]
    if (_unit(affect.get("longing")) or 0)>=.45:drive_labels.append("connection")
    if (_unit(affect.get("protective_drive")) or 0)>=.62:drive_labels.append("protection")
    affinity=_unit(rel.get("affinity"));trust=_unit(rel.get("trust"));defense=_unit(rel.get("defensiveness"))
    weather=None
    if affinity is not None and trust is not None and affinity>=.78 and trust>=.72:weather="close_and_secure"
    elif defense is not None and defense>=.45:weather="careful"
    elif all(v is not None for v in (affinity,trust,defense)):weather="steady"
    base=affect.get('base_label')
    if base not in ('joyful','secure','irritable','low','relaxed','tense','tired','calm'):base=None
    cause=affect.get('cause_label');cause_safe=None
    source_labels=set(labels)
    if isinstance(cause,dict) and isinstance(cause.get('label'),str) and cause['label'] in source_labels:
        cause_safe={'label':cause['label'],'since':_stamp(cause.get('since'))}
    uncertain=affect.get('uncertain_labels',[])
    uncertain=[x for x in uncertain if isinstance(x,str) and len(x)<=60] if isinstance(uncertain,list) else []
    subtype_names={'anger':{'生气','不满','不甘心','不平'},'grieve':{'失落','委屈','分别','心疼','自责'},'favored':{'吃醋','被晾着','被忘'}}
    subtypes=[]
    for item in affect.get('subtypes',[]) if isinstance(affect.get('subtypes'),list) else []:
        if not isinstance(item,dict):continue
        key=item.get('key');label=item.get('label');weight=_finite(item.get('weight'))
        if key in subtype_names and label in subtype_names[key] and weight is not None and 0<=weight<=1:subtypes.append({'key':key,'label':label,'weight':weight})
    mixed={}
    raw=affect.get('mixed',{})
    for key in ('anger_closeness','grieve_closeness','anger_protective'):
        item=raw.get(key) if isinstance(raw,dict) else None
        if isinstance(item,dict) and item.get('status') in ('forming','visible','unknown'):
            mixed[key]={'status':item['status'],'since':_stamp(item.get('since'))}
    slow=state.get('slow_axes',{})
    mood=slow.get('mood',{}) if isinstance(slow,dict) else {}
    if not isinstance(mood,dict):mood={}
    mood_value=_finite(mood.get('value'))
    if mood.get('unit')=='-1..1' and state.get('state_schema')!=STATE_SCHEMA:
        mood_value=(mood_value+1)/2 if mood_value is not None and -1<=mood_value<=1 else None
    elif mood.get('unit')!='0..1':mood_value=None
    slow_safe={'security':_unit(slow.get('security')),'confidence':_unit(slow.get('confidence')),
        'mood':{'value':_unit(mood_value),'unit':'0..1','covered_hours':_finite(mood.get('covered_hours')),'unknown_hours':_finite(mood.get('unknown_hours'))}}
    tendencies,tendency_sources,_,_=normalize_tendencies(state.get('tendencies'),state.get('tendency_sources'))
    raw_lit=affect.get('lit');lit=None
    if isinstance(raw_lit,dict) and raw_lit.get('word') in ('心动','害羞','得意','骄傲') and _stamp(raw_lit.get('at')) and _stamp(raw_lit.get('until')):
        refs=safe_source_refs(raw_lit.get('source_refs'))
        if refs:lit={'word':raw_lit['word'],'at':_stamp(raw_lit['at']),'until':_stamp(raw_lit['until']),'source_refs':refs}
    marks=[]
    for row in affect.get('light_marks',[]) if isinstance(affect.get('light_marks'),list) else []:
        if isinstance(row,dict) and row.get('word') in ('心动','害羞','得意','骄傲') and _stamp(row.get('at')):marks.append({'word':row['word'],'at':_stamp(row['at']),'source_refs':safe_source_refs(row.get('source_refs'))})
    raw_load=affect.get('empathy_load',{})
    loads={key:_unit(raw_load.get(key)) if isinstance(raw_load,dict) else None for key in ('anger','grieve')}
    episode=None  # Special conflict episodes are historical migration evidence.
    raw_display=affect.get('display',{});display=None
    display_words=set(labels)
    if lit:display_words.add(lit['word'])
    if affect.get('connection_phase') in ('想黏着','想念'):display_words.add(affect['connection_phase'])
    if cause_safe:display_words.add(cause_safe['label'])
    if isinstance(raw_display,dict) and raw_display.get('mode') in ('coexist','primary') and raw_display.get('status') in ('coexisting','partial_unknown','selected','no_eligible_primary'):
        primary=raw_display.get('primary');secondary=raw_display.get('secondary',[])
        if primary is None or primary in display_words:
            display={'mode':raw_display['mode'],'primary':primary,'secondary':[v for v in secondary if isinstance(v,str) and v in display_words][:32],'status':raw_display['status']}
    coverage={};raw_coverage=affect.get('growth_coverage',{})
    if isinstance(raw_coverage,dict):
        for key in ('longing','possessiveness','protective_drive','libido'):
            row=raw_coverage.get(key)
            if isinstance(row,dict) and row.get('status') in ('known','unknown'):
                coverage[key]={'status':row['status'],'as_of':_stamp(row.get('as_of'))}
                if row['status']=='unknown':coverage[key].update(since=_stamp(row.get('since')),last_known=_unit(row.get('last_known')))
    raw_fatigue=state.get('fatigue',{});fatigue_state=None
    if isinstance(raw_fatigue,dict) and raw_fatigue.get('status') in ('initial_explicit','unconfigured','unchanged','unknown_seed','sleep_recovery','unknown_sleep_phase','unknown_pressure','high_pressure','known_pressure_below_or_equal'):
        fatigue_state={'value':_unit(raw_fatigue.get('value')),'as_of':_stamp(raw_fatigue.get('as_of')),'status':raw_fatigue['status']}
        if 'last_known' in raw_fatigue:fatigue_state.update(last_known=_unit(raw_fatigue.get('last_known')),last_known_at=_stamp(raw_fatigue.get('last_known_at')))
    return {"profile_id":str(state.get("profile_id") or profile_id),
      "affect":{"label":labels[0] if labels else None,"labels":labels,"label_sources":label_sources,"unattributed_labels":unattributed,"labels_updated_at":_stamp(affect.get("labels_updated_at")),"subtypes":subtypes,"mixed":mixed,"lit":lit,"growth_coverage":coverage,"display":display,"light_marks":marks[-128:],"empathy_load":loads,"conflict_episode":episode,"connection_phase":affect.get("connection_phase") if affect.get("connection_phase") in ("想黏着","想念") else None,"last_conversation_at":_stamp(affect.get("last_conversation_at")),"base_label":base,"cause_label":cause_safe,"uncertain_labels":uncertain,"intensity":level(affect.get("arousal"),.24,.62),"valence":_unit(affect.get("valence")),"arousal":_unit(affect.get("arousal")),"tenderness":_unit(affect.get("tenderness")),"session_defensiveness":_unit(affect.get("session_defensiveness")),"source":"brain_b1_candidate"},
      "drive":{"labels":drive_labels,"longing":_unit(affect.get("longing")),"protective":_unit(affect.get("protective_drive")),"longing_level":level(affect.get("longing"),.20,.45),"protective_level":level(affect.get("protective_drive"),.35,.62),"possessiveness":_unit(affect.get("possessiveness")),**{k:_unit(affect.get(k)) for k in ("libido","grieve","anger","favored","cling")},"source":"brain_b1_candidate"},
      "relationship":{"weather":weather,"closeness":level(affinity,.45,.78),"trust":level(trust,.45,.72),"source":"ombre_relationship_frozen"},
      "updated_at":str(updated_at or state.get("updated_at") or ""),"revision":state.get("revision"),"policy_revision":state.get("policy_revision"),"source":"b1_candidate","state_schema":STATE_SCHEMA,"slow_axes":slow_safe,"tendencies":tendencies,"tendency_sources":tendency_sources,"fatigue_state":fatigue_state,"fatigue":_unit(state.get("fatigue",{}).get("value")) if isinstance(state.get("fatigue"),dict) else None}



def projection_level(value: object, low: float, high: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number >= high:
        return "high"
    if number <= low:
        return "low"
    return "medium"


def safe_persona_projection(state: dict, updated_at: str = "", profile_id: str = "") -> dict:
    """Reduce Persona state to approved labels and exclude private narrative."""
    if isinstance(state,dict) and state.get("source")=="b1_candidate":
        return _b1_projection(state,updated_at,profile_id)
    affect = state.get("affect", {}) if isinstance(state, dict) else {}
    relationship = state.get("relationship", {}) if isinstance(state, dict) else {}
    mood_label = str(affect.get("mood_label") or "warm_neutral")[:60]
    drive_labels = []
    if float(affect.get("longing") or 0.0) >= 0.45:
        drive_labels.append("connection")
    if float(affect.get("protective_drive") or 0.0) >= 0.62:
        drive_labels.append("protection")
    if float(affect.get("arousal") or 0.0) >= 0.62:
        drive_labels.append("expression")
    if not drive_labels:
        drive_labels.append("companionship")

    affinity = float(relationship.get("affinity") or 0.0)
    trust = float(relationship.get("trust") or 0.0)
    defensiveness = float(relationship.get("defensiveness") or 0.0)
    if affinity >= 0.78 and trust >= 0.72:
        weather = "close_and_secure"
    elif defensiveness >= 0.45:
        weather = "careful"
    else:
        weather = "steady"

    return {
        "profile_id": str(state.get("profile_id") or profile_id),
        "affect": {
            "label": mood_label,
            "intensity": projection_level(affect.get("arousal"), 0.24, 0.62),
            "valence": round(float(affect.get("valence") or 0.0), 3),
            "arousal": round(float(affect.get("arousal") or 0.0), 3),
            "source": "ombre_affect",
        },
        "drive": {
            "labels": drive_labels[:4],
            "longing": projection_level(affect.get("longing"), 0.20, 0.45),
            "protective": projection_level(affect.get("protective_drive"), 0.35, 0.62),
            "source": "ombre_affect",
        },
        "relationship": {
            "weather": weather,
            "closeness": projection_level(affinity, 0.45, 0.78),
            "trust": projection_level(trust, 0.45, 0.72),
            "source": "ombre_relationship",
        },
        "updated_at": str(updated_at or ""),
    }

