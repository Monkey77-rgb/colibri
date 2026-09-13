#!/usr/bin/env python3
"""gptoss_layer0_ref.py -- INDEPENDENT oracle for test_gptoss_layer0.cpp.

Pure Python, stdlib only (struct + math). No numpy, no torch, no llama.cpp,
no shared code with engine/src/loader.c or arch_ops.cpp -- this file parses
the GGUF header and dequantizes Q8_0 blocks from first principles, and
reimplements RMSNorm / YaRN RoPE / GQA attention with a sink logit from the
formulas cited below, not by importing anything from the C++ side. That is
the whole point: if this and test_gptoss_layer0.cpp agree, the agreement is
evidence the C++ implementation is right, not a restatement of it.

Sources for the formulas (all read this session, see the commit message for
what was fetched from where):
  - Q8_0 block layout: engine/c/ggml_dequant.h's PROVENANCE-2 comment
    (uint16 f16 delta + 32x int8, y[j]=qs[j]*d) -- transcribed independently
    here, not by calling that file.
  - RMSNorm / attention shape: llama.cpp src/models/openai-moe.cpp
    (fetched via local browser_service 2026-09-13).
  - YaRN NTK-by-parts frequency scaling, swiglu, sdpa-with-sink: OpenAI's own
    scratchpad/gptoss_model.py (fetched 2026-09-13), read directly for this
    script's rope()/attn() functions -- same source engine/src/arch_ops.cpp
    cites, but reimplemented from the formula, not ported from that file.

Usage: python3 gptoss_layer0_ref.py <path-to-gpt-oss-120b-MXFP4.gguf>
Writes /tmp/coli_gptoss_layer0_py.txt in the same format
test_gptoss_layer0.cpp writes /tmp/coli_gptoss_layer0_cpp.txt, for
tests/compare_layer0.py to diff.
"""
import struct, sys, math

GGUF_TYPES = {0:('u8',1),1:('i8',1),2:('u16',2),3:('i16',2),4:('u32',4),5:('i32',4),
              6:('f32',4),7:('bool',1),8:('str',0),9:('arr',0),10:('u64',8),11:('i64',8),12:('f64',8)}
FMT = {'u8':'<B','i8':'<b','u16':'<H','i16':'<h','u32':'<I','i32':'<i','f32':'<f',
       'bool':'<?','u64':'<Q','i64':'<q','f64':'<d'}

class Gguf:
    def __init__(self, path):
        self.f = open(path, 'rb')
        rd = self.f.read
        def u32(): return struct.unpack('<I', rd(4))[0]
        def u64(): return struct.unpack('<Q', rd(8))[0]
        def s():
            n = u64(); return rd(n).decode('utf-8', 'replace')
        def val(t):
            if t == 8: return s()
            if t == 9:
                et = u32(); n = u64(); return [val(et) for _ in range(n)]
            nm, sz = GGUF_TYPES[t]; b = rd(sz)
            return struct.unpack(FMT[nm], b)[0]
        assert rd(4) == b'GGUF', "not a GGUF file"
        self.version = u32()
        n_tensors = u64(); n_kv = u64()
        self.kv = {}
        for _ in range(n_kv):
            k = s(); t = u32(); self.kv[k] = val(t)
        self.tensors = {}   # name -> (dims list, ggml type, rel_offset)
        for _ in range(n_tensors):
            name = s(); nd = u32(); dims = [u64() for _ in range(nd)]
            tt = u32(); off = u64()
            self.tensors[name] = (dims, tt, off)
        align = self.kv.get('general.alignment', 32)
        pos = self.f.tell()
        self.data_start = ((pos + align - 1) // align) * align

    def kv_get(self, key, default=None):
        return self.kv.get(key, default)

    def tensor_bytes_offset(self, name):
        dims, tt, off = self.tensors[name]
        return self.data_start + off, dims, tt

    def read_f32_tensor(self, name):
        """Full dequant, for small (f32-typed or tiny) tensors only."""
        off, dims, tt = self.tensor_bytes_offset(name)
        n = 1
        for d in dims: n *= d
        self.f.seek(off)
        if tt == 6:  # f32
            raw = self.f.read(n*4)
            return list(struct.unpack('<%df' % n, raw))
        elif tt == 0:  # actually f32 in this file's histogram (ggml type 0 == GGML_TYPE_F32)
            raw = self.f.read(n*4)
            return list(struct.unpack('<%df' % n, raw))
        else:
            raise ValueError("read_f32_tensor: tensor '%s' is not f32 (ttype=%d)" % (name, tt))

    def read_q8_0_row(self, name, row, I):
        """Dequantize ONE row of a [I, O] (ne0=I contiguous, ne1=O rows) Q8_0
        tensor. Row byte offset = tensor_start + row * ceil(I/32) * 34."""
        off, dims, tt = self.tensor_bytes_offset(name)
        assert tt == 8, "tensor '%s' ttype=%d, expected 8 (Q8_0)" % (name, tt)
        nblk = (I + 31) // 32
        row_bytes = nblk * 34
        self.f.seek(off + row * row_bytes)
        raw = self.f.read(row_bytes)
        out = [0.0]*I
        p = 0
        for b in range(nblk):
            d = struct.unpack_from('<e', raw, p)[0]; p += 2
            qs = struct.unpack_from('<32b', raw, p); p += 32
            base = b*32
            n_here = min(32, I-base)
            for j in range(n_here):
                out[base+j] = qs[j]*d
        return out


def rmsnorm(x, w, eps):
    n = len(x)
    ss = sum(v*v for v in x) / n
    scale = 1.0 / math.sqrt(ss + eps)
    return [x[i]*scale*w[i] for i in range(n)]


def matvec_q8_0(gg, name, x, I, O, bias):
    """y[o] = x . dequant_row(o) + bias[o], reading rows lazily (no full
    tensor dequant -- I/O-bound but memory-tiny, and independent of
    test_gptoss_layer0.cpp's coli_gguf_load_f32 full-tensor dequant path)."""
    y = [0.0]*O
    for o in range(O):
        row = gg.read_q8_0_row(name, o, I)
        acc = bias[o] if bias else 0.0
        for i in range(I):
            acc += x[i]*row[i]
        y[o] = acc
    return y


def yarn_rope_table(pos, hd, base, factor, beta_fast, beta_slow, orig_ctx):
    half = hd//2
    if factor > 1.0:
        concentration = 0.1*math.log(factor) + 1.0
        d_half = hd/2.0
        lnbase = math.log(base)
        low = d_half*math.log(orig_ctx/(beta_fast*2*math.pi))/lnbase
        high = d_half*math.log(orig_ctx/(beta_slow*2*math.pi))/lnbase
    else:
        concentration = 1.0
    c = [0.0]*half; s = [0.0]*half
    for i in range(half):
        freq = base**((2.0*i)/hd)
        if factor > 1.0:
            interp = 1.0/(factor*freq); extrap = 1.0/freq
            ramp = (i - low)/(high - low) if high != low else 0.0
            ramp = max(0.0, min(1.0, ramp))
            mask = 1.0 - ramp
            inv_freq = interp*(1.0-mask) + extrap*mask
        else:
            inv_freq = 1.0/freq
        ang = pos*inv_freq
        c[i] = math.cos(ang)*concentration
        s[i] = math.sin(ang)*concentration
    return c, s, half


def apply_rope(v, c, s, half):
    for i in range(half):
        a, b = v[i], v[i+half]
        v[i] = a*c[i] - b*s[i]
        v[i+half] = a*s[i] + b*c[i]


def softmax_with_sink(scores, sink):
    n = len(scores)
    mx = max(sink, max(scores))
    exps = [math.exp(v-mx) for v in scores]
    denom = sum(exps) + math.exp(sink-mx)
    return [e/denom for e in exps]


def swa_masked(qpos, kpos, window):
    if kpos > qpos: return True
    if window > 0 and (qpos-kpos) >= window: return True
    return False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/monkey/Documents/Ai_Models/gptoss/gpt-oss-120b-MXFP4.gguf"
    gg = Gguf(path)
    kv = gg.kv
    D = kv['gpt-oss.embedding_length']
    H = kv['gpt-oss.attention.head_count']
    KVH = kv['gpt-oss.attention.head_count_kv']
    hd = kv['gpt-oss.attention.key_length']
    eps = kv['gpt-oss.attention.layer_norm_rms_epsilon']
    theta = kv['gpt-oss.rope.freq_base']
    factor = kv['gpt-oss.rope.scaling.factor']
    beta_fast = kv['gpt-oss.rope.scaling.yarn_beta_fast']
    beta_slow = kv['gpt-oss.rope.scaling.yarn_beta_slow']
    orig_ctx = kv['gpt-oss.rope.scaling.original_context_length']
    window = kv['gpt-oss.attention.sliding_window']
    qD = H*hd; kvD = KVH*hd; mult = H//KVH

    attn_norm = gg.read_f32_tensor('blk.0.attn_norm.weight')
    bq = gg.read_f32_tensor('blk.0.attn_q.bias')
    bk = gg.read_f32_tensor('blk.0.attn_k.bias')
    bv = gg.read_f32_tensor('blk.0.attn_v.bias')
    bo = gg.read_f32_tensor('blk.0.attn_output.bias')
    sinks = gg.read_f32_tensor('blk.0.attn_sinks.weight')

    ids = [1000, 2000, 3000, 4000]
    N = len(ids)

    x = [gg.read_q8_0_row('token_embd.weight', tid, D) for tid in ids]
    xn = [rmsnorm(xt, attn_norm, eps) for xt in x]

    Q = [matvec_q8_0(gg, 'blk.0.attn_q.weight', xnt, D, qD, bq) for xnt in xn]
    K = [matvec_q8_0(gg, 'blk.0.attn_k.weight', xnt, D, kvD, bk) for xnt in xn]
    V = [matvec_q8_0(gg, 'blk.0.attn_v.weight', xnt, D, kvD, bv) for xnt in xn]

    # Per-head RoPE: Python list slicing copies, so mutate through a temporary
    # and write the segment back rather than relying on aliasing.
    for t in range(N):
        c, s, half = yarn_rope_table(t, hd, theta, factor, beta_fast, beta_slow, orig_ctx)
        for h in range(H):
            base = h*hd
            seg = Q[t][base:base+hd]
            apply_rope(seg, c, s, half)
            Q[t][base:base+hd] = seg
        for h in range(KVH):
            base = h*hd
            seg = K[t][base:base+hd]
            apply_rope(seg, c, s, half)
            K[t][base:base+hd] = seg

    swa_active = (window > 0)  # layer 0 is SWA per set_swa_pattern(period=2, dense_first=False)
    attn_out = [[0.0]*qD for _ in range(N)]
    scale = 1.0/math.sqrt(hd)
    for h in range(H):
        kvh = h // mult
        for qp in range(N):
            scores = []
            mask = []
            for kp in range(N):
                m = swa_masked(qp, kp, window if swa_active else 0)
                mask.append(m)
                if m:
                    scores.append(-1e30)
                else:
                    qv = Q[qp][h*hd:(h+1)*hd]; kv = K[kp][kvh*hd:(kvh+1)*hd]
                    dot = sum(a*b for a,b in zip(qv,kv))
                    scores.append(dot*scale)
            w = softmax_with_sink(scores, sinks[h])
            out = attn_out[qp]
            for kp in range(N):
                vv = V[kp][kvh*hd:(kvh+1)*hd]
                wk = w[kp]
                for i in range(hd):
                    out[h*hd+i] += wk*vv[i]

    cur = [matvec_q8_0(gg, 'blk.0.attn_output.weight', attn_out[t], qD, D, bo) for t in range(N)]

    with open('/tmp/coli_gptoss_layer0_py.txt', 'w') as f:
        for t in range(N):
            f.write("token %d id=%d\n" % (t, ids[t]))
            for v in cur[t]:
                f.write("%.9g\n" % v)
    print("wrote /tmp/coli_gptoss_layer0_py.txt (%d tokens x %d dims)" % (N, D))


if __name__ == '__main__':
    main()
