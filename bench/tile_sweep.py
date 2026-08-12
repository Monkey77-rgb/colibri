"""MOE_TILE sweep — the one grouped-path parameter the 120-run grid never varied.

MOE_TILE bounds the partial-output buffer at tile*K*D floats, so it trades memory for
how many tokens get grouped per expert load. Two things must hold:
  correctness — NLL must be IDENTICAL at every tile size (tiling must not change math)
  memory      — peak RSS should rise with tile size
Neither has ever been measured.
"""
import sys, pathlib, statistics
sys.path.insert(0, str(pathlib.Path("/home/monkey/colibri/bench")))
import sweep as S

dims = S.model_dims()
ref = pathlib.Path("/tmp/ref_tile.json")
S.make_ref(ref, prompt_len=512, score_len=32, vocab=dims["vocab"])

print("  %-9s %-11s %-9s %-9s %-12s" % ("MOE_TILE","NLL","tok/s","hit%","peak_rss_gb"))
rows=[]
for tile in (1, 8, 32, 128, 512, 2048):
    r = S.run(cache=8, pilot=0, group=1, ref=ref, tile=tile, threads=8)
    if not r.get("ok"):
        print("  %-9s FAILED: %s" % (tile, r.get("error"))); continue
    rows.append((tile, r.get("nll"), r.get("tok_s"), r.get("hit_pct"), r.get("peak_rss_gb")))
    print("  %-9s %-11s %-9s %-9s %-12s" % (tile, r.get("nll"), r.get("tok_s"),
                                            r.get("hit_pct"), r.get("peak_rss_gb")))
nlls = {r[1] for r in rows if r[1] is not None}
print()
print("  distinct NLL values across all tile sizes: %d  %s" % (len(nlls), sorted(nlls)))
print("  VERDICT:", "PASS — tiling is mathematically inert" if len(nlls)==1
      else "FAIL — tiling changes the result, that is a correctness bug")
