"""Shared pure canonical encoding for source-bound B1 decisions. No I/O."""
import hashlib,json,math

def canonical(value):
    """Reject non-JSON/NaN/Inf recursively, including numeric booleans elsewhere."""
    def check(obj):
        if obj is None or isinstance(obj, (str, bool)):
            return
        if isinstance(obj, (int, float)):
            if not math.isfinite(obj): raise ValueError('canonical finite number required')
            return
        if isinstance(obj, list):
            for item in obj: check(item)
            return
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
            for item in obj.values(): check(item)
            return
        raise ValueError('only JSON values with string object keys are accepted')
    check(value)
    def normalize(obj):
        if isinstance(obj, float) and obj == 0: return 0.0
        if isinstance(obj, list): return [normalize(x) for x in obj]
        if isinstance(obj, dict): return {k:normalize(v) for k,v in obj.items()}
        return obj
    return json.dumps(normalize(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def identity(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f'{field}: nonempty string <=256 required')
    return value

