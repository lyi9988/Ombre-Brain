"""Retired Ombre impressions stay historical, outside automatic material feeds."""

RETIRED_TAGS = frozenset({'daily_impression', 'weekly_impression'})


def is_retired_impression(item):
    if not isinstance(item, dict):
        return False
    meta = item.get('metadata') if isinstance(item.get('metadata'), dict) else item
    tags = {str(tag).strip().lower() for tag in meta.get('tags', []) or []}
    return bool(tags & RETIRED_TAGS)
