#!/usr/bin/env python3
"""Build a complete, deterministic COLI_MOE_PROFILE from training traces only.

COLI_MOE_TRACE rows: layer, batch size, selected expert IDs. Default: decode
rows (S==1), matching slot-hit counters. Supply model dimensions explicitly;
unseen experts must also be emitted. --reverse ranks coldest first, including
zero-count experts. A one-token prefill cannot be distinguished from decode.
Example: moe_profile.py --layers 36 --experts 128 --output hot.txt train*.txt

--model-hash (2026-09-18): writes an optional first-line header
"# coli-moe-profile model=<hash> layers=<L> experts=<E>" so the engine
(src/moe_profile.h) refuses this profile outright on a model it was not
built for, instead of silently mispinning slots. <hash> must be the loaded
GGUF's coli_gguf_meta_hash() as 16 lowercase hex digits -- read it off a
real `coli`/`coli-gpu` run's stderr ("model identity: gguf_meta_hash=...");
this tool has no GGUF reader of its own and does not recompute it. Omitting
--model-hash writes the pre-existing headerless format, which the engine
still accepts (with a printed "unvalidated legacy profile" notice) as long
as its own layer/expert counts already agree with the loaded model.
"""
import argparse
from collections import Counter
from pathlib import Path


def counts(paths, layers, experts, phase):
    result = [Counter() for _ in range(layers)]
    for path in paths:
        for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
            try:
                layer, size, *selected = map(int, line.split())
                if not (0 <= layer < layers and size > 0 and selected
                        and len(selected) == len(set(selected))
                        and all(0 <= e < experts for e in selected)):
                    raise ValueError("invalid layer, size or expert selections")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if phase == "all" or (size == 1) == (phase == "decode"):
                result[layer].update(selected)
    if any(not c for c in result):
        raise ValueError("every layer must have observations in the chosen phase")
    return result


def ranking(frequencies, experts, reverse=False):
    return [sorted(range(experts), key=lambda e: (c[e] if reverse else -c[e], e))
            for c in frequencies]


def coverage(frequencies, order, slots):
    q, rem = divmod(slots, len(order))
    rows = []
    for layer, c in enumerate(frequencies):
        total = sum(c.values())
        hit = sum(c[e] for e in order[layer][:q + (layer < rem)])
        rows.append((hit, total))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", nargs="+", type=Path)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--experts", type=int, required=True)
    p.add_argument("--phase", choices=["decode", "prefill", "all"], default="decode")
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-hash", metavar="HEX16",
                    help="loaded GGUF's coli_gguf_meta_hash, 16 hex digits (see module docstring); "
                         "adds the validated header so the engine refuses this profile on any other model")
    a = p.parse_args()
    if a.layers <= 0 or a.experts <= 0:
        p.error("dimensions must be positive")
    if a.model_hash is not None:
        stripped = a.model_hash.lower().removeprefix("0x")
        if len(stripped) != 16 or any(ch not in "0123456789abcdef" for ch in stripped):
            p.error("--model-hash must be exactly 16 hex digits, e.g. 00000000cafebabe")
        a.model_hash = stripped
    order = ranking(counts(a.traces, a.layers, a.experts, a.phase), a.experts, a.reverse)
    # Exclusive creation prevents silently replacing a measured profile.
    with a.output.open("x") as f:
        if a.model_hash is not None:
            f.write(f"# coli-moe-profile model={a.model_hash} layers={a.layers} experts={a.experts}\n")
        for layer, row in enumerate(order):
            f.write(" ".join(map(str, [layer, *row])) + "\n")


if __name__ == "__main__":
    main()
