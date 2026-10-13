"""Place resolved Gateway blocks on the original request skeleton, without flattening."""
from copy import deepcopy
import hashlib


def place_context_messages(messages, blocks, coverage=None):
    wire = deepcopy(messages)
    coverage = coverage or {}
    taken, combined = set(), list(blocks)
    for row in coverage.get('prompt_blocks', []):
        if not (isinstance(row, list) and len(row) == 6 and type(row[0]) is int
                and 0 <= row[0] < len(wire) and row[0] not in taken):
            continue
        index, anchor, order, block_id, digest, depth = row
        item = wire[index]
        body = item.get('content')
        if (not isinstance(body, str) or item.get('tool_calls')
                or item.get('role') not in {'system', 'developer', 'user', 'assistant'}
                or hashlib.sha256(body.encode('utf-8')).hexdigest() != digest):
            continue
        taken.add(index)
        combined.append({'block_id':block_id, 'anchor':anchor, 'order':order,
                         'history_depth':depth, 'role':item['role'], 'content':body})
    remap, original = {}, []
    for index, item in enumerate(wire):
        if index not in taken:
            remap[index] = len(original)
            original.append(item)
    def indices(name, member):
        ranges = coverage.get(name)
        if isinstance(ranges, list):
            values = []
            for pair in ranges:
                if (isinstance(pair, list) and len(pair) == 2
                        and all(type(x) is int for x in pair)
                        and 0 <= pair[0] <= pair[1] <= len(wire)):
                    values.extend(remap[x] for x in range(*pair) if x in remap)
            return sorted(set(values))
        return sorted({remap[x['message_index']] for x in coverage.get('items', [])
                       if isinstance(x, dict) and bool(x.get('turn_member')) == member
                       and type(x.get('message_index')) is int
                       and x['message_index'] in remap})
    current = indices('current_user_message_ranges', True)
    history = indices('history_message_ranges', False)
    if not current:
        current = [i for i, x in enumerate(original) if x.get('role') == 'user'][-1:]
    before = current[0] if current else len(original)
    after = current[-1] + 1 if current else len(original)
    if not history:
        history = [i for i, x in enumerate(original[:before])
                   if x.get('role') in {'user', 'assistant'} and not x.get('tool_calls')]
    leading = 0
    while leading < len(original) and original[leading].get('role') in {'system', 'developer'}:
        leading += 1
    insertions, prefixes = {}, []
    ranks = {'instructions.start':10, 'instructions.end':20, 'gateway.after_first_system':25,
             'history.start':30, 'history.depth':40, 'current_user.before':50,
             'current_user.prefix':60, 'gateway.current_user_prefix':60, 'current_user.after':110}
    combined.sort(key=lambda x:(ranks.get(x.get('anchor'), 50), int(x.get('order') or 0), str(x.get('block_id') or '')))
    for block in combined:
        anchor = block.get('anchor')
        role = block.get('role', 'system')
        if anchor == 'gateway.current_user_prefix' and role == 'user' and current:
            prefixes.append(block['content'])
            continue
        if anchor == 'instructions.start':
            pos = 0
        elif anchor == 'instructions.end':
            pos = leading
        elif anchor == 'gateway.after_first_system':
            pos = 1 if leading else 0
        elif anchor == 'history.start':
            pos = history[0] if history else before
        elif anchor == 'history.depth':
            depth = max(0, int(block.get('history_depth') or 0))
            pos = history[max(0, len(history) - depth)] if history and depth else before
        elif anchor == 'current_user.after':
            pos = after
        else:
            pos = before
        # A call and its results form an indivisible protocol group.
        while 0 < pos < len(original) and original[pos].get('role') in {'tool', 'function'}:
            pos -= 1
        insertions.setdefault(pos, []).append({'role': role, 'content': block['content']})
    if prefixes:
        prefix = '<ombre_live_context>\n' + '\n\n'.join(prefixes) + '\n</ombre_live_context>\n\nCurrent user message:\n'
        target = original[current[-1]]
        content = target.get('content')
        if isinstance(content, list):
            target['content'] = [{'type': 'text', 'text': prefix}, *content]
        else:
            target['content'] = prefix + str(content or '')
    for pos in sorted(insertions, reverse=True):
        original[pos:pos] = insertions[pos]
    return original
