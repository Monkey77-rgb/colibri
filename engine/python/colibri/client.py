"""Async client and worker pool."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, Sequence

import urllib.request
import urllib.error


@dataclass
class Worker:
    """One coli-server process.

    `slots` mirrors the server's own slot count. The pool uses it to stop sending
    a worker more concurrent work than it can batch -- past that point requests
    queue behind the scheduler and latency grows with no throughput gain, which
    looks like the server being slow rather than the client over-subscribing it.
    """
    url: str
    slots: int = 4
    inflight: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    healthy: bool = field(default=True, init=False)

    def load(self) -> float:
        return self.inflight / max(self.slots, 1)


class Pool:
    """Least-loaded routing over a set of workers.

    Least-loaded rather than round-robin because requests here have wildly
    different costs: a 2000-token prompt occupies a slot far longer than a
    20-token one, and round-robin would keep handing work to the worker already
    stuck behind it.
    """

    def __init__(self, workers: Sequence[Worker]):
        if not workers:
            raise ValueError("a pool needs at least one worker")
        self.workers = list(workers)

    def pick(self) -> Worker:
        live = [w for w in self.workers if w.healthy]
        if not live:
            # Everything is marked down. Rather than fail, retry the least-failed
            # one: a health flag is a heuristic, and refusing all traffic on a
            # heuristic is worse than trying.
            live = sorted(self.workers, key=lambda w: w.failures)[:1]
        return min(live, key=lambda w: w.load())


class Client:
    """Talks to a pool of coli-server workers.

    Nothing here is in the token path; every call is HTTP to Go.
    """

    def __init__(self, urls: str | Iterable[str] = "http://127.0.0.1:8100",
                 slots: int = 4, timeout: float = 600.0, retries: int = 2):
        if isinstance(urls, str):
            urls = [urls]
        self.pool = Pool([Worker(u.rstrip("/"), slots) for u in urls])
        self.timeout = timeout
        self.retries = retries

    # ---- blocking ----
    def complete(self, prompt: str, max_tokens: int = 128, **kw) -> str:
        body = {"prompt": prompt, "max_tokens": max_tokens, **kw}
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            w = self.pool.pick()
            w.inflight += 1
            try:
                req = urllib.request.Request(
                    f"{w.url}/v1/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                w.healthy = True
                return d["choices"][0]["text"]
            except Exception as e:                    # noqa: BLE001
                last = e
                w.failures += 1
                if w.failures >= 3:
                    w.healthy = False
            finally:
                w.inflight -= 1
        raise RuntimeError(f"all {self.retries + 1} attempts failed: {last}")

    # ---- async fan-out: the actual reason this layer exists ----
    async def complete_many(self, prompts: Sequence[str], max_tokens: int = 128,
                            concurrency: int | None = None, **kw) -> list[str]:
        """Run many prompts concurrently across the pool.

        Default concurrency is the pool's total slot count: enough to keep every
        worker's batch full, and no more. Sending more does not increase
        throughput -- it queues behind the scheduler -- so the default is the
        measured-correct number rather than an arbitrary large one.
        """
        if concurrency is None:
            concurrency = sum(w.slots for w in self.pool.workers)
        sem = asyncio.Semaphore(max(1, concurrency))
        loop = asyncio.get_running_loop()

        async def one(p: str) -> str:
            async with sem:
                return await loop.run_in_executor(
                    None, lambda: self.complete(p, max_tokens, **kw))

        return await asyncio.gather(*(one(p) for p in prompts))

    async def stream(self, prompt: str, max_tokens: int = 128, **kw) -> AsyncIterator[str]:
        """Server-sent events from one worker."""
        w = self.pool.pick()
        body = {"prompt": prompt, "max_tokens": max_tokens, "stream": True, **kw}
        req = urllib.request.Request(
            f"{w.url}/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=self.timeout))
        while True:
            line = await loop.run_in_executor(None, resp.readline)
            if not line:
                return
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)["choices"][0]["text"]
            except Exception:                          # noqa: BLE001
                continue

    def health(self) -> list[dict]:
        out = []
        for w in self.pool.workers:
            try:
                with urllib.request.urlopen(f"{w.url}/health", timeout=5) as r:
                    d = json.load(r)
                d["url"] = w.url
                w.healthy = True
                out.append(d)
            except Exception as e:                     # noqa: BLE001
                w.healthy = False
                out.append({"url": w.url, "status": "down", "error": str(e)})
        return out
