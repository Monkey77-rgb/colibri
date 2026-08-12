#!/usr/bin/env python3
"""effective_bw.py — derive the bandwidth of the path a model is ACTUALLY executing on.

Why this exists
---------------
membw.c measures what the CPU cores can pull from DRAM. On a discrete-GPU box that is
the wrong number entirely (weights live in VRAM). On a unified-memory APU it is *still*
the wrong number, but subtly: the iGPU and the CPU cores read the same DIMMs at
different achievable rates, because a handful of CPU cores cannot keep enough loads in
flight to saturate the controller while the GPU's memory pipeline can.

So a roofline built on the CPU figure will tell you a GPU-offloaded model is running
*faster than physically possible*, which is a strong hint the model was wrong, not the
measurement. This tool closes that loop by inverting a real generation:

    effective_read_bandwidth = decode_bytes_per_token * measured_tok/s

decode_bytes_per_token comes from modelprobe's exact tensor-directory walk, not from the
file size (which over-counts the input embedding, and under-counts nothing).

That gives two things worth more than either alone:
  * the bandwidth of the real execution path, on whatever backend is in use, with no
    vendor tooling and no assumptions about the memory subsystem; and
  * once you also pass --peak, the fraction of the hardware ceiling being achieved,
    which is the only honest measure of whether a tuning change helped.

Caveats stated up front, because a number like this is easy to over-read:
  * It attributes ALL decode time to weight reads. Attention/KV traffic, sampling and
    framework overhead are folded in, so the result is a LOWER BOUND on the true
    bandwidth of the path. Real bandwidth is at least this.
  * Consequently it is most accurate at long generations with a short prompt, where
    per-token weight reads dominate. Prompt processing is compute-bound and is excluded
    (we use predicted_per_second, never the total wall time).
  * A KV cache large relative to the weights will bias it upward. Reported alongside.

Read-only: issues one ordinary completion request. It does not reconfigure the server.
"""
import argparse, json, subprocess, sys, urllib.request


def probe_model(modelprobe, path):
    """Exact per-token read bytes from the tensor directory."""
    out = subprocess.run([modelprobe, "--json", path],
                         capture_output=True, text=True, check=True).stdout
    m = json.loads(out)[0]
    if not m.get("ok"):
        sys.exit(f"modelprobe could not read {path}")
    if not m.get("decode_bytes"):
        sys.exit(f"modelprobe found no tensor directory in {path} — cannot be exact, refusing to guess")
    return m


def generate(url, n_predict, prompt):
    """One ordinary completion. Returns llama.cpp's own timings block."""
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    t = d.get("timings")
    if not t:
        sys.exit("server returned no timings block — is this llama-server?")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="llama-server base URL, e.g. http://127.0.0.1:8086")
    ap.add_argument("--model", required=True, help="path to the GGUF that server is running")
    ap.add_argument("--modelprobe", default="./modelprobe")
    ap.add_argument("-n", "--n-predict", type=int, default=192,
                    help="tokens to generate; longer amortizes startup better (default 192)")
    ap.add_argument("--repeats", type=int, default=3, help="best of N (default 3)")
    ap.add_argument("--peak", type=float, default=0.0,
                    help="theoretical peak GB/s, to report %% of hardware ceiling")
    ap.add_argument("--cpu-bw", type=float, default=0.0,
                    help="membw CPU-side read GB/s, to show the CPU-vs-actual-path gap")
    ap.add_argument("--prompt", default="Write a detailed paragraph about copper wire.")
    a = ap.parse_args()

    m = probe_model(a.modelprobe, a.model)
    per_tok = m["decode_bytes"]

    best_tps, best = 0.0, None
    for _ in range(a.repeats):
        t = generate(a.url, a.n_predict, a.prompt)
        tps = t.get("predicted_per_second") or 0.0
        if tps > best_tps:
            best_tps, best = tps, t

    bw = per_tok * best_tps / 1e9

    print(f"model            : {a.model}")
    print(f"  topology       : {'MoE' if m['moe'] else 'dense'}  "
          f"{m['layers']} layers, hidden {m['hidden']}, {m['precision']}")
    print(f"  on disk        : {m['bytes_on_disk']/1e9:.2f} GB")
    print(f"  read/token     : {per_tok/1e9:.3f} GB   (exact, tensor directory"
          f"{'; tied embedding counted' if m.get('tied_embd') else ''})")
    print()
    print(f"measured         : {best_tps:.2f} tok/s decode  "
          f"(best of {a.repeats}, {best.get('predicted_n')} tokens)")
    print(f"  prompt         : {best.get('prompt_per_second', 0):.1f} tok/s "
          f"(compute-bound, excluded from the bandwidth math)")
    print()
    print(f"EFFECTIVE READ BW: {bw:.1f} GB/s on the path this model is actually running")
    print( "                   (lower bound — all decode time charged to weight reads)")

    if a.cpu_bw > 0:
        print(f"  CPU-side membw : {a.cpu_bw:.1f} GB/s  ->  this path is {bw/a.cpu_bw:.2f}x the CPU figure")
        if bw > a.cpu_bw:
            print( "                   Above the CPU number means the weights are NOT being read")
            print( "                   by the CPU cores. Size the roofline with THIS value.")
    if a.peak > 0:
        print(f"  theoretical    : {a.peak:.1f} GB/s  ->  achieving {100*bw/a.peak:.0f}% of peak")
        print( "                   Headroom to peak is the honest cap on further tuning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
