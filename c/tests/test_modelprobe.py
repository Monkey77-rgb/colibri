#!/usr/bin/env python3
"""Known-answer test for modelprobe's GGUF tensor-directory math.

modelprobe reports bytes-per-expert two ways. For safetensors/HF it must estimate from
the precision label. For GGUF it walks the tensor directory and computes exactly -- real
dimensions, real per-tensor ggml type. This test pins the exact path.

Method: synthesize a GGUF whose expert tensors have dimensions and a type we choose, so
the correct byte count is computable here, independently of the C code. Then assert
modelprobe reproduces it exactly. A rounding error, a wrong block size in the ggml type
table, or an off-by-one in the dimension walk all fail this.

We also assert the *shape* fields, because getting the bytes right while mis-reading
top-k would still produce a wrong streaming-demand figure -- which is the number this
tool exists to produce.

Run:  python3 c/tests/test_modelprobe.py [path/to/modelprobe]
"""
import json, os, struct, subprocess, sys, tempfile

GGML_Q4_K, BLCK, BYTES = 12, 256, 144      # 256 elements packed into 144 bytes

ARCH, LAYERS, EXPERTS, TOPK, HID, FFN = "testmoe", 4, 8, 2, 64, 32


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_str(k, v):
    return _s(k) + struct.pack("<I", 8) + _s(v)


def _kv_u32(k, v):
    return _s(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def build_gguf(path):
    """Write a minimal MoE GGUF; return the ground truth we expect back."""
    kvs = [
        _kv_str("general.architecture", ARCH),
        _kv_u32("general.file_type", 15),                       # Q4_K_M label
        _kv_u32(f"{ARCH}.block_count", LAYERS),
        _kv_u32(f"{ARCH}.expert_count", EXPERTS),
        _kv_u32(f"{ARCH}.expert_used_count", TOPK),
        _kv_u32(f"{ARCH}.embedding_length", HID),
        _kv_u32(f"{ARCH}.expert_feed_forward_length", FFN),
    ]

    tensors = []
    for l in range(LAYERS):
        for nm in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            tensors.append((f"blk.{l}.{nm}.weight", (HID, FFN, EXPERTS), GGML_Q4_K))
    tensors.append(("token_embd.weight", (HID, 1000), GGML_Q4_K))   # a non-expert tensor

    out = struct.pack("<I", 0x46554747) + struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    out += b"".join(kvs)

    off = exps = total = 0
    for name, dims, t in tensors:
        out += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", t) + struct.pack("<Q", off)
        n = 1
        for d in dims:
            n *= d
        b = (n // BLCK) * BYTES
        off += b
        total += b
        if "_exps" in name:
            exps += b

    with open(path, "wb") as f:
        f.write(out + b"\x00" * 4096)      # tensor data region; contents irrelevant

    return {
        "expert_bank_exact": exps,
        "tensor_bytes_total": total,
        "exps_tensors": 3 * LAYERS,
        "bytes_per_expert": exps / (LAYERS * EXPERTS),
        "layers": LAYERS, "experts": EXPERTS, "experts_active": TOPK,
        "hidden": HID, "expert_ffn": FFN, "arch": ARCH, "moe": True,
    }


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "modelprobe")
    if not os.path.exists(binary):
        print(f"SKIP: {binary} not built (make modelprobe)")
        return 0

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fake_moe.gguf")
        want = build_gguf(path)
        got = json.loads(subprocess.run([binary, "--json", path],
                                        capture_output=True, text=True,
                                        check=True).stdout)[0]

        fails = []
        for k in ("arch", "layers", "experts", "experts_active", "hidden",
                  "expert_ffn", "moe", "expert_bank_exact", "tensor_bytes_total",
                  "exps_tensors"):
            if got.get(k) != want[k]:
                fails.append(f"  {k}: got {got.get(k)!r}, want {want[k]!r}")

        per = got["expert_bank_exact"] / (got["layers"] * got["experts"])
        if per != want["bytes_per_expert"]:
            fails.append(f"  bytes_per_expert: got {per}, want {want['bytes_per_expert']}")

        if fails:
            print("FAIL — modelprobe disagrees with independently computed ground truth:")
            print("\n".join(fails))
            return 1

    print(f"PASS — exact tensor-directory math "
          f"({want['expert_bank_exact']} expert bytes over {want['exps_tensors']} tensors, "
          f"{want['bytes_per_expert']:.0f} B/expert)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
