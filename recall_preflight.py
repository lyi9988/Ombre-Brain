"""Cheap, conservative request preflight; uncertainty keeps ordinary Recall."""
import re
import unicodedata


# Whole clauses only. Bare "几点"/"什么日子" may refer to an event or anniversary.
_CLOCK_CLAUSE = re.compile(
    r"(?:请问|告诉我|可以告诉我)?(?:"
    r"(?:现在|此刻|当前)(?:是)?(?:几点(?:钟)?|什么时间|什么日期|几月几[日号])(?:了)?"
    r"|(?:现在|当前)(?:的)?时间(?:是)?多少"
    r"|今天(?:是)?(?:几[日号]|几月几[日号]|星期几|周几|什么日期)"
    r")(?:吗|呢|呀|啊)?"
    r"|what(?:'s| is) the (?:time|date)(?: now| today)?"
    r"|what time is it(?: now)?"
    r"|what (?:date|day of the week) is it today",
    re.IGNORECASE,
)


def clock_preflight(natural_input: dict, messages: list[dict], clock: dict) -> dict:
    """Never infer absence of memory needs from shortness, type, or a keyword.

    Require complete canonical current-turn coverage, plain text and a unique
    authenticated system clock actually present in this request. Inspect every
    staged message; a mixed request or missing evidence takes the normal path.
    """
    meta = natural_input.get('metadata') or {}
    result = {'policy': 'conservative-clock-v1', 'skip_dynamic_recall': False,
              'reason': 'not_pure_current_clock'}
    if (meta.get('input_source') != 'canonical_coverage'
            or meta.get('current_input_complete') is not True):
        return {**result, 'reason': 'current_coverage_unconfirmed'}
    matches = clock.get('physical_matches') or []
    if (clock.get('source') != 'aizizhu_turn_snapshot'
            or not clock.get('physical_verified')
            or len(matches) != 1 or not isinstance(matches[0], dict)
            or matches[0].get('role') != 'system'):
        return {**result, 'reason': 'clock_not_verified'}
    current = natural_input.get('current_messages') or []
    if not current or len(current) != len(meta.get('staged_user_event_ids') or []):
        return {**result, 'reason': 'staged_coverage_unconfirmed'}
    for row in current:
        index = row.get('index')
        if type(index) is not int or not 0 <= index < len(messages):
            return {**result, 'reason': 'current_coverage_unconfirmed'}
        message = messages[index]
        content = message.get('content')
        if isinstance(content, list):
            if not content or any(not isinstance(p, dict) or p.get('type') not in ('text', 'input_text')
                                  or not isinstance(p.get('text'), str) for p in content):
                return {**result, 'reason': 'nontext_current_input'}
            content = ''.join(p['text'] for p in content)
        if not isinstance(content, str) or message.get('role') != 'user':
            return {**result, 'reason': 'nontext_current_input'}
        # Do not classify a truncated/cleaned projection of a mixed message.
        if content.strip() != row.get('text', '').strip():
            return {**result, 'reason': 'current_text_transformed'}
        text = unicodedata.normalize('NFKC', content).strip()
        if len(text) > 160:
            return result
        clauses = [re.sub(r'\s+', ' ', s).strip() for s in re.split(r'[,，。.!?！？;；\n]+', text) if s.strip()]
        if not clauses or any(not _CLOCK_CLAUSE.fullmatch(s) for s in clauses):
            return result
    return {**result, 'skip_dynamic_recall': True, 'reason': 'pure_current_clock'}
