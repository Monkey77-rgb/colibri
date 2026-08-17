"""Throughput measurement.

Reports the batch HISTOGRAM from the server, not just tokens/sec. A throughput
number without it cannot distinguish "the scheduler batched 8 sequences" from
"the client sent them one at a time and the server was idle" -- and those have
identical client-side timings at low concurrency.
"""
from __future__ import annotations

import asyncio
import time
from typing import Sequence

from .client import Client


def benchmark(client: Client, prompts: Sequence[str], max_tokens: int = 32,
              concurrencies: Sequence[int] = (1, 2, 4, 8)) -> list[dict]:
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

        rows.append({
            "concurrency": c,
            "wall_s": round(wall, 2),
            "total_tok_s": round(c * max_tokens / wall, 1),
            "per_req_tok_s": round(max_tokens / wall, 1),
            "server_batch_histogram": delta,
        })
    return rows
