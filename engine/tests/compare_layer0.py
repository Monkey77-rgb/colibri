#!/usr/bin/env python3
"""compare_layer0.py -- diffs test_gptoss_layer0.cpp's output against
gptoss_layer0_ref.py's, both already run. Exit 1 if they disagree beyond
tolerance, or if either file is missing/malformed (a diff over data neither
side actually produced is not a comparison)."""
import sys

def read(path):
    tokens = []
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("token "):
                cur = []
                tokens.append(cur)
            else:
                cur.append(float(line))
    return tokens

def main():
    a = read("/tmp/coli_gptoss_layer0_cpp.txt")
    b = read("/tmp/coli_gptoss_layer0_py.txt")
    if len(a) != len(b):
        print("FAIL: token count differs: cpp=%d py=%d" % (len(a), len(b))); sys.exit(1)
    max_abs = 0.0
    max_rel = 0.0
    worst = None
    for t, (ta, tb) in enumerate(zip(a, b)):
        if len(ta) != len(tb):
            print("FAIL: token %d dim count differs: cpp=%d py=%d" % (t, len(ta), len(tb))); sys.exit(1)
        for i, (x, y) in enumerate(zip(ta, tb)):
            d = abs(x-y)
            if d > max_abs:
                max_abs = d; worst = (t, i, x, y)
            denom = max(abs(x), abs(y), 1e-6)
            r = d/denom
            if r > max_rel:
                max_rel = r
    n_total = sum(len(t) for t in a)
    print("compared %d values across %d tokens" % (n_total, len(a)))
    print("max abs diff = %.6g at token=%d dim=%d (cpp=%.6g py=%.6g)" % (max_abs, worst[0], worst[1], worst[2], worst[3]))
    print("max rel diff = %.6g" % max_rel)
    TOL = 5e-3  # Q8_0 dequant + f32 accumulation order differences (C sums in
                # double per dot product; Python sums left-to-right in double
                # too, but 4096/2880-wide reductions in different element
                # order are not bit-exact even in exact arithmetic terms of
                # associativity) -- this is a CORRECTNESS bound, not a
                # bit-exactness claim.
    if max_abs > TOL:
        print("FAIL: max abs diff %.6g exceeds tolerance %.6g" % (max_abs, TOL))
        sys.exit(1)
    print("PASS: within tolerance %.6g" % TOL)

if __name__ == '__main__':
    main()
