#!/usr/bin/env python3
"""
sweep.py — measure Colibri's OLMoE expert-cache behaviour instead of guessing at it.

WHY THIS EXISTS
---------------
Upstream ships `chat_olmoe.sh` with CACHE=64, which equals OLMoE's per-layer
expert count (config.json: num_experts=64). Every expert is therefore resident
after warmup, eviction never fires, and *nothing streams*. Upstream also ships
PILOT=0, with a comment that prefetch measured slower "once CACHE=64 gives a
~94% hit rate — the prefetch thread has little disk-wait left to hide".

Both statements are true and both are measured at the one setting where the
streaming path is switched off. So the project's central claim — that
router-lookahead prefetch (PILOT) hides expert I/O — has never been measured
where it could possibly matter: a cache smaller than the expert count.

This sweep measures it.

WHAT IT VARIES
--------------
  MOE_GROUP  0 = legacy, one expert_get per (token, top-K slot)
             1 = grouped, one expert_get per DISTINCT expert per tile
  CACHE      slots per layer (argv[1]). < 64 is where eviction begins.
  PILOT      0..3, router-lookahead depth
  PROMPT     prefill length. tf_nll() prefills the whole prompt in ONE step()
             call (olmoe.c: `step(m, full, np, 0)`), so this is what actually
             exercises batched expert demand. Decode is S=1 and cannot.

HOW IT MEASURES
---------------
PPL=1 (teacher-forced NLL) is the right mode: deterministic, and it deliberately
skips rt_save so a sweep cannot fold its own tokens into the persisted expert
ranking (olmoe.c: "no rt_save on purpose"). It reports NLL, tok/s, peak RSS and
the hit/miss counters in one shot.

`hit + miss` IS the expert_get call count, so the dedup effect is read directly
off it rather than inferred.

CORRECTNESS GATE
----------------
--verify asserts MOE_GROUP=0 and MOE_GROUP=1 produce an IDENTICAL TF-NLL. The
grouped path reorders *when* experts load, never the arithmetic: partial results
are accumulated per (token, slot) and summed back in kk order, and the routing
weight is applied at accumulation time so the expression stays `os[d] += w*p[d]`
— the same shape GCC contracts to an FMA at -O3 -march=native. Pre-multiplying
instead rounds the product to float first, and that 1-ULP delta compounds across
layers until a token flips. That is not theoretical: it happened, and this gate
is what caught it. Run --verify after ANY change to moe().

TOKEN IDS
---------
Synthetic, seeded, uniform over the vocab. Absolute perplexity from random ids
is meaningless and is NOT reported as a quality figure — NLL is used only as a
high-entropy fingerprint for the correctness gate, and as a fixed workload for
timing. Routing on random ids is arbitrary but *identical* across configs, which
is exactly what a controlled comparison needs.

USAGE
  ./sweep.py --verify                         # correctness gate only
  ./sweep.py --quick                          # small grid, sanity
  ./sweep.py --out results.csv                # full grid
"""

import argparse, csv, json, os, pathlib, random, re, subprocess, sys, time

HERE  = pathlib.Path(__file__).resolve().parent
ROOT  = HERE.parent
OLMOE = ROOT / "c" / "olmoe"
SNAP  = pathlib.Path(os.environ.get("SNAP", "/home/monkey/models/olmoe_merged"))

RE_NLL   = re.compile(r"TF-NLL:\s*([-\d.]+)\s*nats/token over\s*(\d+)")
RE_PPL   = re.compile(r"ppl\s*=\s*([\d.]+)")
RE_CACHE = re.compile(r"hit rate:\s*([\d.]+)%\s*\(hit=(\d+)\s+miss=(\d+)\)")
RE_SPEED = re.compile(r"Speed:\s*([\d.]+)\s*tok/s\s*\(([\d.]+)s")
RE_RSS   = re.compile(r"PEAK RSS:\s*([\d.]+)\s*GB")


def model_dims():
    cfg = json.loads((SNAP / "config.json").read_text())
    return {
        "vocab":   cfg["vocab_size"],
        "layers":  cfg["num_hidden_layers"],
        "experts": cfg["num_experts"],
        "topk":    cfg["num_experts_per_tok"],
    }


def make_ref(path, prompt_len, score_len, vocab, seed=1234):
    """Synthetic ref.json: `prompt_len` prefill tokens + `score_len` scored tokens."""
    rng = random.Random(seed)
    ids = [rng.randrange(vocab) for _ in range(prompt_len + score_len)]
    path.write_text(json.dumps({"prompt_ids": ids[:prompt_len], "full_ids": ids}))


def run(cache, pilot, group, ref, tile=128, threads=8, extra=None, timeout=1800):
    env = dict(os.environ)
    env.update({
        "SNAP": str(SNAP), "PPL": "1", "PILOT": str(pilot),
        "MOE_GROUP": str(group), "MOE_TILE": str(tile),
        "OMP_NUM_THREADS": str(threads),
    })
    env.pop("CHAT", None)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})

    t0 = time.time()
    try:
        p = subprocess.run([str(OLMOE), str(cache), "8", str(ref)],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(ROOT / "c"), env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout>{timeout}s"}
    wall = time.time() - t0
    out = p.stdout + p.stderr
    if p.returncode != 0:
        return {"ok": False, "error": f"rc={p.returncode}: {out.strip()[-200:]}"}

    r = {"ok": True, "wall_s": round(wall, 2)}
    if (m := RE_NLL.search(out)):   r["nll"], r["scored"] = m.group(1), int(m.group(2))
    if (m := RE_PPL.search(out)):   r["ppl"] = float(m.group(1))
    if (m := RE_CACHE.search(out)):
        r["hit_pct"], r["hit"], r["miss"] = float(m.group(1)), int(m.group(2)), int(m.group(3))
        r["expert_gets"] = r["hit"] + r["miss"]
    if (m := RE_SPEED.search(out)): r["tok_s"], r["gen_s"] = float(m.group(1)), float(m.group(2))
    if (m := RE_RSS.search(out)):   r["peak_rss_gb"] = float(m.group(1))
    if "nll" not in r:
        return {"ok": False, "error": "could not parse TF-NLL: " + out.strip()[-200:]}
    return r


def verify(dims, prompt_lens, caches):
    """MOE_GROUP=0 vs 1 must agree EXACTLY on TF-NLL."""
    ref = HERE / "_ref_verify.json"
    print("=== correctness gate: grouped must equal legacy, bit for bit ===")
    ok = True
    for plen in prompt_lens:
        make_ref(ref, plen, 16, dims["vocab"])
        for cache in caches:
            a = run(cache, 0, 0, ref)
            b = run(cache, 0, 1, ref)
            if not a["ok"] or not b["ok"]:
                print(f"  prompt={plen:<5} CACHE={cache:<3} ERROR {a.get('error') or b.get('error')}")
                ok = False; continue
            same = a["nll"] == b["nll"]
            ok &= same
            print(f"  prompt={plen:<5} CACHE={cache:<3} legacy={a['nll']} grouped={b['nll']} "
                  f"gets {a['expert_gets']}->{b['expert_gets']} "
                  f"({a['expert_gets']/max(b['expert_gets'],1):.1f}x fewer)  "
                  f"{'MATCH' if same else 'MISMATCH <<<'}")
    ref.unlink(missing_ok=True)
    print("PASS" if ok else "FAIL")
    return ok


def sweep(dims, prompt_lens, caches, pilots, groups, out_csv):
    ref = HERE / "_ref_sweep.json"
    rows, fields = [], ["prompt_len", "cache", "pilot", "moe_group", "expert_gets",
                        "hit", "miss", "hit_pct", "tok_s", "wall_s", "peak_rss_gb", "nll", "error"]
    total = len(prompt_lens) * len(caches) * len(pilots) * len(groups)
    n = 0
    print(f"=== sweep: {total} runs "
          f"(experts={dims['experts']} topk={dims['topk']} layers={dims['layers']}) ===")
    for plen in prompt_lens:
        make_ref(ref, plen, 16, dims["vocab"])
        for cache in caches:
            for pilot in pilots:
                for group in groups:
                    n += 1
                    r = run(cache, pilot, group, ref)
                    row = {"prompt_len": plen, "cache": cache, "pilot": pilot, "moe_group": group}
                    row.update({k: r.get(k) for k in fields if k in r})
                    row["error"] = "" if r["ok"] else r["error"]
                    rows.append(row)
                    if r["ok"]:
                        print(f"  [{n}/{total}] prompt={plen:<5} CACHE={cache:<3} PILOT={pilot} "
                              f"GROUP={group} | gets={r.get('expert_gets'):<8} "
                              f"hit={r.get('hit_pct')}% tok/s={r.get('tok_s')} "
                              f"rss={r.get('peak_rss_gb')}GB")
                    else:
                        print(f"  [{n}/{total}] prompt={plen} CACHE={cache} PILOT={pilot} "
                              f"GROUP={group} | ERROR {r['error']}")
    ref.unlink(missing_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="correctness gate only")
    ap.add_argument("--quick",  action="store_true", help="small grid")
    ap.add_argument("--out", default=str(HERE / "results.csv"))
    a = ap.parse_args()

    if not OLMOE.exists():
        sys.exit(f"missing {OLMOE} — run: make -C {ROOT}/c olmoe")
    if not (SNAP / "config.json").exists():
        sys.exit(f"missing snapshot at {SNAP} (override with SNAP=...)")

    dims = model_dims()
    E = dims["experts"]

    if a.verify:
        sys.exit(0 if verify(dims, [128, 512], [8, E]) else 1)

    if a.quick:
        sweep(dims, [512], [8, E], [0, 1], [0, 1], a.out)
    else:
        # caches below E are the whole point: that is where eviction — and any
        # benefit from PILOT — can exist at all.
        sweep(dims, [128, 512, 1024], [4, 8, 16, 32, E], [0, 1, 2, 3], [0, 1], a.out)


if __name__ == "__main__":
    main()
