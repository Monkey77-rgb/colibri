"""Throughput measurement.

Reports the batch HISTOGRAM from the server, not just tokens/sec. A throughput
number without it cannot distinguish "the scheduler batched 8 sequences" from
"the client sent them one at a time and the server was idle" -- and those have
identical client-side timings at low concurrency.

TWO DEFECTS FIXED 2026-08-19, both of which reported MORE work than was done:

1. Throughput was computed from `c * max_tokens`, the tokens REQUESTED. A
   sequence that hits EOS stops early (scheduler.go marks it Reason="stop"), so
   the numerator exceeded the tokens actually generated and tok/s came out high.
   The ground truth was already being fetched and thrown away: /health returns
   `tokens`, and this function already takes a before/after health snapshot to
   diff the batch histogram. It now diffs `tokens` too and divides by that.

2. The prefix KV cache is ON by default in the engine. This loop re-sends the
   SAME prompts at every concurrency level, in one process, against a LIFO slot
   allocator -- so levels 2/4/8 could match a prefix that level 1 paid for and
   skip prefill entirely, making later levels look faster for a reason that has
   nothing to do with batching. `batch_histogram` cannot detect this: it proves
   DECODE batched, which is a different question. The cache is now turned OFF
   for the run by default, and the measured hit rate is reported either way, so
   the reader can see it was zero rather than trust that it was.
"""
from __future__ import annotations

import asyncio
import time
from typing import Sequence

from .client import Client


def _sum(snapshots, key) -> int:
    """Sum a scalar field across every server in the snapshot list."""
    return sum(int(s.get(key) or 0) for s in snapshots)


def benchmark(client: Client, prompts: Sequence[str], max_tokens: int = 32,
              concurrencies: Sequence[int] = (1, 2, 4, 8),
              prefix_cache: bool = False) -> list[dict]:
    """Measure throughput at each concurrency level.

    prefix_cache=False (the default) asks the server to disable KV prefix reuse
    for the run. Leave it False for any comparison ACROSS concurrency levels;
    set it True only when the cache itself is what you are measuring.
    """
    if hasattr(client, "set_prefix_cache"):
        client.set_prefix_cache(prefix_cache)

    rows = []
    for c in concurrencies:
        work = [prompts[i % len(prompts)] for i in range(c)]
        before = client.health()
        t0 = time.time()
        asyncio.run(client.complete_many(work, max_tokens=max_tokens, concurrency=c))
        wall = time.time() - t0
        after = client.health()

        # Which batch sizes the server actually ran, over this window only.
        delta: dict[str, int] = {}
        for b, a in zip(before, after):
            for k, v in (a.get("batch_histogram") or {}).items():
                delta[k] = delta.get(k, 0) + v - (b.get("batch_histogram") or {}).get(k, 0)
        delta = {k: v for k, v in sorted(delta.items(), key=lambda kv: int(kv[0])) if v > 0}

        # Tokens the server says it produced in this window. Falls back to the
        # requested count ONLY if the field is missing, and says so, because a
        # silent fallback to the old estimate is how the defect would return.
        gen = _sum(after, "tokens") - _sum(before, "tokens")
        estimated = gen <= 0
        if estimated:
            gen = c * max_tokens

        # Prefix reuse over this window. Reported even when zero: "0 of 4096"
        # is evidence the cache did not fire; a missing field is not.
        reused = _sum(after, "prefix_reused") - _sum(before, "prefix_reused")
        asked = _sum(after, "prefix_asked") - _sum(before, "prefix_asked")

        rows.append({
            "concurrency": c,
            "wall_s": round(wall, 2),
            "tokens_generated": gen,
            "tokens_requested": c * max_tokens,
            "total_tok_s": round(gen / wall, 1),
            "per_req_tok_s": round(gen / c / wall, 1),
            "tok_s_estimated": estimated,     # True => server gave no count
            "server_batch_histogram": delta,
            "prefix_reused": reused,
            "prefix_asked": asked,
            "prefix_hit_rate": round(reused / asked, 4) if asked > 0 else 0.0,
        })
    return rows
