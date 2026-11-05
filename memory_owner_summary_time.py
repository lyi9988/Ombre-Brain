"""Source-local time support for owner-summary drafts; never repairs their prose.

Recorded/uploaded timestamps are custody metadata. Only the actually observed
original text can support a calendar date, clock precision or timezone here.
This is a necessary evidence check, not a semantic truth/causality classifier.
"""
from __future__ import annotations

import re

from memory_l1_service import L1OwnerError

_DATE = re.compile(r'(?<!\d)(\d{4})(?:年|[-/])(\d{1,2})(?:月|[-/])(\d{1,2})(?:日|号)?(?!\d)')
_MONTH_DAY = re.compile(r'(?<!\d)(\d{1,2})月(\d{1,2})(?:日|号)')
_YEAR_MONTH = re.compile(r'(?<!\d)(\d{4})年(\d{1,2})月')
_YEAR = re.compile(r'(?<!\d)(\d{4})年')
_CLOCK = re.compile(r'(?<![\d+\-])(\d{1,2})[:：](\d{2})(?:[:：](\d{2})(?:\.(\d+))?)?(Z|[+\-]\d{2}:?\d{2})?(?!\d)')
_CHINESE_CLOCK = re.compile(r'(上午|下午|晚上|凌晨|中午)?(\d{1,2})[点时](?:(\d{1,2})分?)?(?:(\d{1,2})秒)?')
_MONTHS = {label: index for index, name in enumerate((
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"), start=1) for label in (name, name[:3])}
_MONTH_WORD = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_ENGLISH_DATE = re.compile(rf'\b({_MONTH_WORD})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b', re.I)
_DAY_MONTH = re.compile(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_WORD})\.?(?:\s+(\d{{4}}))?\b', re.I)


def tokens(text):
    result = set()
    for match in _DATE.finditer(text):
        year, month, day = map(int, match.groups())
        result.update({('date',year,month,day), ('month_day',month,day), ('year_month',year,month), ('year',year)})
    for match in _MONTH_DAY.finditer(text): result.add(('month_day', *map(int,match.groups())))
    for match in _YEAR_MONTH.finditer(text):
        year, month = map(int,match.groups()); result.update({('year_month',year,month), ('year',year)})
    for match in _YEAR.finditer(text): result.add(('year',int(match.group(1))))
    for pattern, day_first in ((_ENGLISH_DATE, False), (_DAY_MONTH, True)):
        for match in pattern.finditer(text):
            month, day, year = match.groups()
            if day_first:
                month, day = day, month
            month, day = _MONTHS[month.lower()], int(day)
            result.add(('month_day', month, day))
            if year:
                result.update({('date', int(year), month, day), ('year_month', int(year), month), ('year', int(year))})
    for match in _CLOCK.finditer(text):
        hour, minute, second, fraction, zone = match.groups()
        result.update({('clock', int(hour), None, None, None), ('clock', int(hour), int(minute), None, None)})
        if second is not None:
            result.update({('clock', int(hour), int(minute), int(second), None),
                           ('clock', int(hour), int(minute), int(second), fraction)})
        if zone: result.add(('zone','+0000' if zone=='Z' else zone.replace(':','')))
    for match in _CHINESE_CLOCK.finditer(text):
        period, hour, minute, second = match.groups()
        hour = int(hour)
        if period in ('下午','晚上') and hour < 12: hour += 12
        if period == '凌晨' and hour == 12: hour = 0
        result.add(('clock',hour,None,None,None))
        if minute is not None: result.add(('clock',hour,int(minute),None,None))
        if second is not None: result.add(('clock',hour,int(minute or 0),int(second),None))
    if re.search(r'\b(?:UTC|GMT)\b(?!\s*[+\-])',text): result.add(('zone','+0000'))
    if any(zone in text for zone in ('北京时间','上海时间','Asia/Shanghai','UTC+8','UTC+08:00')): result.add(('zone','+0800'))
    return result


def _check_text(text, evidence, path):
    if isinstance(text,str) and tokens(text) - tokens(evidence):
        raise L1OwnerError('owner_summary_time_not_supported',status=409,field_path=path)


def source_time_evidence(text):
    """Verifiable observations, with no date inferred from record metadata."""
    values=tokens(text)
    dates=sorted(t[1:] for t in values if t[0]=='date')
    months=sorted(t[1:] for t in values if t[0]=='year_month' and not any(d[:2]==t[1:] for d in dates))
    years=sorted(t[1] for t in values if t[0]=='year' and not any(d[0]==t[1] for d in dates+months))
    partial=sorted(t[1:] for t in values if t[0]=='month_day' and not any(d[1:]==t[1:] for d in dates))
    clocks=[t[1:] for t in values if t[0]=='clock']
    def refines(fine,coarse):
        return fine!=coarse and all(v is None or fine[i]==v for i,v in enumerate(coarse))
    clocks=[clock for clock in clocks if not any(refines(fine,clock) for fine in clocks)]
    return {'origin':'observed_original_text',
        'absolute_dates':[{'year':y,'month':m,'day':d,'precision':'day'} for y,m,d in dates]
            +[{'year':y,'month':m,'precision':'month'} for y,m in months]
            +[{'year':y,'precision':'year'} for y in years],
        'partial_calendar_dates':[{'month':m,'day':d,'year':None} for m,d in partial],
        'clock_times':[{'hour':h,'minute':m,'second':s,'fraction':f,
            'precision':'fraction' if f is not None else 'second' if s is not None else 'minute' if m is not None else 'hour'}
            for h,m,s,f in sorted(clocks,key=repr)],
        'timezones':[{'utc_offset':t[1][:3]+':'+t[1][3:]} for t in sorted(values,key=repr) if t[0]=='zone'],
        'metadata_event_time':{'state':'unknown'}, 'metadata_validity_time':{'state':'unknown'}}


def validate_time(document, source_texts):
    """Validate every statement against its own bindings, not the batch pool."""
    def evidence(refs):
        return '\n'.join(source_texts[ref] for ref in refs if ref in source_texts)
    def visit(node, inherited_refs, path):
        if isinstance(node,list):
            for index,item in enumerate(node): visit(item,inherited_refs,path+'['+str(index)+']')
        elif isinstance(node,dict):
            refs = node.get('evidence_refs',inherited_refs)
            original = evidence(refs)
            if node.get('state') == 'known' and isinstance(node.get('value'),str):
                value = node['value']
                _check_text(value,original,path+'.value')
                # Relative/zone boundary strings also require original support;
                # absence of a machine calendar token is not a free pass.
                if path.endswith(('.start','.end','.timezone')) and not tokens(value) and value not in original:
                    zone_aliases = {'UTC':('UTC','世界协调时','Z'), 'Asia/Shanghai':('Asia/Shanghai','北京时间','上海时间','UTC+8','UTC+08:00','+08:00')}
                    if not any(alias in original for alias in zone_aliases.get(value,())):
                        raise L1OwnerError('owner_summary_time_not_supported',status=409,field_path=path+'.value')
            for key,value in node.items():
                if key != 'evidence_refs': visit(value,refs,path+'.'+key)
        elif isinstance(node,str): _check_text(node,evidence(inherited_refs),path)
    all_refs = set()
    for index,statement in enumerate(document.get('statements',[])):
        refs = statement.get('evidence_refs',[])
        all_refs.update(refs)
        base = 'document.statements['+str(index)+']'
        _check_text(statement.get('content'),evidence(refs),base+'.content')
        for key in ('time','conditions','cognition','property','attributes'):
            if key in statement: visit(statement[key],refs,base+'.'+key)
        event = (statement.get('time') or {}).get('event') or {}
        if event.get('precision') == 'instant' and not any(t[0]=='clock' for t in tokens(evidence(refs))):
            raise L1OwnerError('owner_summary_time_precision_unsupported',status=409,field_path=base+'.time.event.precision')
    # Narrative has no per-sentence evidence field. Restrict it to this actual
    # candidate's supporting statements, never unused neighboring messages.
    for key in ('title','body'):
        _check_text((document.get('narrative') or {}).get(key),evidence(all_refs),'document.narrative.'+key)
    visit(document.get('retrieval',{}),all_refs,'document.retrieval')
    visit(document.get('attributes',[]),all_refs,'document.attributes')
    for index,scene in enumerate(document.get('scenes',[])):
        refs = scene.get('source_refs') or [ref for s in document.get('statements',[]) if s.get('statement_id') in scene.get('statement_refs',[]) for ref in s.get('evidence_refs',[])]
        _check_text(scene.get('title'),evidence(refs),'document.scenes['+str(index)+'].title')


def validate_prepared(authority, proposal):
    """Recheck owner-edited drafts immediately before their original adoption."""
    from memory_l0_sources import MemoryL0Sources
    from memory_l1_schema import join_document
    semantic = proposal['metadata']['memory_l1']
    sources = {}
    for binding in semantic['evidence_bindings']:
        snapshot = MemoryL0Sources(authority).copy(binding)
        if not snapshot:
            raise L1OwnerError('owner_summary_time_source_missing',status=409,field_path='evidence_bindings')
        sources[binding['binding_id']] = '\n'.join(snapshot['text'][anchor['start']:anchor['end']] for anchor in binding['anchors'])
    validate_time(join_document(proposal['proposed_body'],semantic),sources)
