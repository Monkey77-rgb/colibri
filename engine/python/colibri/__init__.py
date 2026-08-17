"""colibri — the scale-out layer.

WHY PYTHON IS HERE, and what it is deliberately NOT.

The split across this engine is by what each language is actually good at, not
by preference:

  C++   kernels, model, forward pass. Needs intrinsics, manual memory layout and
        compile-time specialisation. Measured: the int8 GEMM dispatches on
        (ISA x batch size) because AVX-512 VNNI is 1.28x at batch>=4 and 0.83x
        at batch 1 -- that is a decision only the compute layer can make.
  Go    one process: HTTP, the continuous-batching scheduler, slot allocation,
        cross-compilation to every target. Concurrency plumbing.
  Py    MANY processes: fan-out, retries, routing, orchestration, evaluation.

Python is NOT in the token path. Every function here talks to a Go worker over
HTTP; none of them touch a tensor. That is the point -- putting Python between
the sampler and the KV cache would add interpreter latency to every token, which
is exactly the mistake the split exists to avoid.

What it buys: horizontal scale (N workers, possibly N machines), and a place for
the logic that changes weekly without recompiling anything.
"""

from .client import Client, Worker, Pool          # noqa: F401
from .bench import benchmark                       # noqa: F401

__all__ = ["Client", "Worker", "Pool", "benchmark"]
__version__ = "0.1.0"
