"""Synthetic local decoding comparison, not provider or end-to-end latency."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding_vector_cache import VectorDecodeCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vectors', type=int, default=1061)
    parser.add_argument('--dimensions', type=int, default=1024)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    assert 1 <= args.vectors <= 2048 and 1 <= args.dimensions <= 4096
    assert 1 <= args.rounds <= 10
    rng = random.Random(20261010)
    payloads = [json.dumps([rng.uniform(-1, 1) for _ in range(args.dimensions)])
                for _ in range(args.vectors)]
    cache = VectorDecodeCache()
    reference_times, reuse_times = [], []
    started = time.perf_counter()
    for payload in payloads:
        actual, valid = cache.decode(payload)
        assert valid and actual == tuple(json.loads(payload))
    warmup_ms = round((time.perf_counter() - started) * 1000, 2)
    last_debug = {}
    for _ in range(args.rounds):
        started = time.perf_counter()
        for payload in payloads:
            vector = json.loads(payload)
            assert isinstance(vector, list) and vector
            assert not any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in vector)
        reference_times.append(round((time.perf_counter() - started) * 1000, 2))
        debug = {}
        started = time.perf_counter()
        for payload in payloads:
            _vector, valid = cache.decode(payload, diagnostics=debug)
            assert valid
        reuse_times.append(round((time.perf_counter() - started) * 1000, 2))
        last_debug = debug
    print(json.dumps({
        'kind': 'synthetic_local_decode_only', 'provider_calls': 0,
        'vectors': args.vectors, 'dimensions': args.dimensions,
        'rounds': args.rounds, 'values_equal': True,
        'warmup_including_equality_check_ms': warmup_ms,
        'baseline_decode_validate_ms': reference_times,
        'cached_decode_ms': reuse_times,
        'baseline_median_ms': statistics.median(reference_times),
        'cached_median_ms': statistics.median(reuse_times),
        'last_round': last_debug, 'cache': cache.snapshot(),
    }))


if __name__ == '__main__':
    main()
