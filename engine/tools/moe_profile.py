#!/usr/bin/env python3
"""Build a complete, deterministic COLI_MOE_PROFILE from training traces only.

COLI_MOE_TRACE rows: layer, batch size, selected expert IDs. Default: decode
rows (S==1), matching slot-hit counters. Supply model dimensions explicitly;
unseen experts must also be emitted. --reverse ranks coldest first, including
zero-count experts. A one-token prefill cannot be distinguished from decode.
Example: moe_profile.py --layers 36 --experts 128 --output hot.txt train*.txt
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
    a = p.parse_args()
    if a.layers <= 0 or a.experts <= 0:
        p.error("dimensions must be positive")
    order = ranking(counts(a.traces, a.layers, a.experts, a.phase), a.experts, a.reverse)
    # Exclusive creation prevents silently replacing a measured profile.
    with a.output.open("x") as f:
        for layer, row in enumerate(order):
            f.write(" ".join(map(str, [layer, *row])) + "\n")


if __name__ == "__main__":
    main()
