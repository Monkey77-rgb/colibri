# Research notes — building an inference engine

Sources for the design decisions in this engine. Two rules for what goes in here,
both learned the hard way in this project:

1. **Every entry is something actually read**, and says where. A file:line from a
   checkout that exists on this machine, or a URL that was fetched and returned
   content. Nothing is cited from memory.
2. **A source is a hypothesis, not a result.** Published numbers describe other
   hardware, other builds, other workloads. Where we tested a claim locally, the
   local measurement is given separately and marked, including when it disagrees.

Retrieved 2026-08-17. Web sources fetched through the private `browser_service`
(`:8300`). Source checkouts read directly.

> **Reachability, stated rather than hidden:** `arxiv.org` and
> `ar5iv.labs.arxiv.org` both time out through the private browser (60–90 s,
> repeatedly). `export.arxiv.org` works and is what the arXiv entries below came
> from. So the arXiv entries are **title, authors and abstract**, not full texts —
> where a specific figure from inside a paper matters, it is marked NOT READ
> rather than filled in from recollection.

---

## 1. The primary source that matters most: llama.cpp, read locally

Checkout: `~/.unsloth/llama.cpp`, at `c96f608d9`. This is the single most useful
reference for this project, because it is a working engine for the same models on
the same hardware class, and because it can be *read* rather than inferred.

### 1.1 Batch size decides the kernel — independently arrived at

- `ggml/src/ggml-cpu/repack.cpp:3104` — *"If there are more than three rows in
  src1, use gemm; otherwise, use gemv"*, and it runs GEMM over the largest
  multiple of 4 with GEMV for the remainder.
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp:3689` — `if (n < 2) return false;`,
  commented *"only enable sgemm for prompt processing"*.

Two independent thresholds in one codebase, both in the 2–4 range. Our
`COLI_GEMM_MIN_WIDE = 4` was measured before this was read, which makes it
corroboration rather than imitation.

**Where we go further:** ggml selects its CPU variant from an ISA feature bitmask
(`ggml-cpu/arch/x86/cpu-feats.cpp`, `ggml_backend_cpu_x86_score()`), so on a Zen
4/5 box it always takes the AVX-512 build and has no way to prefer AVX2 for
decode. We measured VNNI at **0.83× at n=1** — i.e. that choice is a regression on
the path that runs for every generated token.

### 1.2 GPU quantized matmul dequantizes to float

`ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl`, `DATA_A_Q4_0`:

```glsl
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2(vui & 0xF, vui >> 4) - 8.0f);
}
```

Nibbles become floats before any multiply. That is the right engineering call
*for ggml*, which routes ~20 quantization formats through one templated matmul and
cannot specialise arithmetic per format. It is not obviously right for an engine
with exactly one weight format, which is why `shaders/gemm_i4.comp` keeps the
nibbles integer through the dot product.

**Measured, 2026-08-17, rather than argued.** `shaders/gemm_i4f.comp` is the same
shader with the one line changed — identical buffers, layout and access pattern,
so only the arithmetic differs. RTX 4070, 8192×16384, min of 5, two runs:

| n | integer nibbles | dequant to float |
|---|---|---|
| 1 | **0.31 ms** | 0.33–0.34 |
| 2 | 0.63–0.69 | 0.63–0.64 |
| 4 | **1.16–1.17** | 1.23 |
| 8 | **2.31–2.34** | 2.44–2.45 |

**Integer wins by ~5%, consistently, and ties at n=2.** Small — and the honest
reading is that it *vindicates* ggml rather than beating it: they pay about 5% on
this shape and get one kernel that serves twenty formats. If this engine ever
grows a second weight format, that 5% is the wrong thing to defend.

Accuracy is identical, and not by luck: with `|q| ≤ 8` and `|x| ≤ 127` the
products are ≤ 1016 and a block of 8 sums to ≤ 8128, all exactly representable in
float32. The float path loses nothing *at these bit widths*. It would stop being
free at int8×int8, where the sums exceed 2²⁴.

### 1.3 The whole graph is one submission, not one dispatch per operator

`ggml/src/ggml-vulkan/ggml-vulkan.cpp:14103`, `ggml_backend_vk_graph_compute()`.
The design our own backend is missing, in three concrete parts:

- Nodes are recorded into a command buffer and submitted in **batches**, with the
  comment *"Submit after enough work has accumulated, to overlap CPU cmdbuffer
  generation with GPU execution."*
- The batching threshold is **adaptive to model size**:
  `mul_mat_bytes_per_submit = std::min(uint64_t(100*1000*1000), ctx->last_total_mul_mat_bytes / 40u)`
  — roughly 100 MB of weight traffic per submit, scaled down for smaller models,
  with a floor of `nodes_per_submit = 100` for graphs without much matmul.
- The first few submits are deliberately **smaller**: `if (submit_count < 3)
  mul_mat_bytes_per_submit *= 2;` — get the GPU busy early, then amortise.

Intermediate activations stay in device buffers for the whole graph; only model
inputs and the final logits cross the bus. Our backend uploads activations and
downloads results **per GEMM call**, which is why it needed persistent buffers just
to stop being slower than the CPU. This is the reference design for the
"activations resident across layers" item.

### 1.4 MoE on GPU *does* get a GEMM — the CPU-only gap, correctly scoped

`ggml-vulkan.cpp:8626`, `ggml_vk_mul_mat_id()` dispatches to
`ggml_vk_mul_mat_vec_id_q_f16` **or** `ggml_vk_mul_mat_id_q_f16` depending on
`ggml_vk_use_mul_mat_vec_id()`. There is also a fused
`vulkan-shaders/topk_moe.comp` for expert routing.

This is a correction to our own README, which is worth recording because the
original wording was too broad: `repack.cpp`'s `forward_mul_mat_id` really does
contain a `gemv<>` call and no `gemm<>` call, so **on CPU** 64 tokens routed to one
expert become 64 GEMVs — but "llama.cpp does not do MoE GEMM" would have been
wrong by an entire backend.

### 1.5 Flash attention is a shipped shader family, not a research idea

`vulkan-shaders/` contains `flash_attn.comp`, `flash_attn_cm1.comp`,
`flash_attn_cm2.comp` (cooperative-matrix variants), `flash_attn_mask_opt.comp`
and `flash_attn_split_k_reduce.comp`, plus tuning-parameter shared-memory
feasibility checks (`ggml_vk_flash_attn_scalar_shmem_support`, ~line 8639). Our
attention is a plain CPU loop; this is what the mature version looks like.

---

## 2. Quantization — the papers behind the format choices

Abstracts read via `export.arxiv.org`; full texts NOT read.

| | what it establishes | relevance here |
|---|---|---|
| **GPTQ** — Frantar, Ashkboos, Hoefler, Alistarh, [2210.17323](https://export.arxiv.org/abs/2210.17323) (v2, Mar 2023) | one-shot post-training quantization of GPT-scale models to low bit-width | our quantizer is plain round-to-nearest per block. GPTQ's second-order weight update is the obvious accuracy upgrade for the **+3.87% NLL** int4-only costs |
| **AWQ** — Lin, Tang, Tang, Yang et al., [2306.00978](https://export.arxiv.org/abs/2306.00978) (v6, Apr 2026) | activation-aware weight quantization: which weights matter is decided by activation magnitude, not weight magnitude | directly applicable: we choose block scales from `max|w|` alone and never look at activations |
| **SmoothQuant** — Xiao, Lin, Seznec, Wu, Demouth, Han, [2211.10438](https://export.arxiv.org/abs/2211.10438) (v7) | migrate activation outliers into weights so both sides quantize well | we quantize activations per 16-element block at runtime, which handles outliers by locality instead. Untested against this |
| **LLM.int8()** — Dettmers, Lewis, Belkada, Zettlemoyer, [2208.07339](https://export.arxiv.org/abs/2208.07339) (v2) | int8 matmul at scale, with outlier features kept in higher precision | the mixed-precision outlier idea is the standard answer to what block scales approximate |

**Marlin** — IST-DASLab, `github.com/IST-DASLab/marlin`, README fetched in full.
The closest prior art to what we measured today, and it corroborates it:

> *"an extremely optimized FP16xINT4 matmul kernel ... can deliver close to ideal
> (4x) speedups up to batchsizes of 16-32 tokens (in contrast to the 1-2 tokens of
> prior work with comparable speedup)"*

Their framing is exactly ours: the *format's* benefit extends well past batch 1,
and prior kernels lost it not because int4 is bad at larger batches but because
the kernel was. Their reasoning — GPUs have FLOP:byte around 100–200, so under
25–50 MACs per 4-bit weight the 4× should hold — is the GPU version of the
bandwidth argument we made for CPU. Their techniques (async global weight loads
with immediate-eviction cache policy, double-buffered shared memory, offline
reshuffling of weights *and* group scales into the layout the tensor cores want)
are a roadmap for a serious GPU int4 kernel.

**Local corroboration, measured 2026-08-17:** hoisting the nibble unpack out of the
per-row loop took our int4 at n=4 from 12.59 ms to 5.83 ms, and n=32 from 101.96 ms
to 48.17 ms — 1.07–1.18× of int8 instead of 1.7–1.8× worse. Same shape of result as
Marlin's claim, on a CPU, arrived at before the README was read.

---

## 3. Serving architecture

**PagedAttention / vLLM** — Kwon et al., [2309.06180](https://export.arxiv.org/abs/2309.06180);
design doc `vllm/docs/design/paged_attention.md` fetched in full (20.7 KB).
KV cache is managed in fixed-size blocks like OS virtual memory pages, so
sequences need not be contiguous and blocks can be shared between sequences.

**How ours differs, and why that is a real gap:** our KV cache is
`[n_slots][n_kv_heads][max_ctx][head_dim]` — a full `max_ctx` allocation per slot,
reserved whether used or not. At 4 slots and 32k context that is the dominant
memory cost of a small model, and it is why `n_slots` is documented as a memory
decision. Paged blocks would let short sequences cost what they actually use.
Not built.

**Mixtral of Experts** — Jiang et al., [2401.04088](https://export.arxiv.org/abs/2401.04088):
sparse mixture-of-experts, router selects a subset per token.

**PowerInfer** — Song, Mi, Xie, Chen, [2312.12456](https://export.arxiv.org/abs/2312.12456):
hot/cold neuron split for serving on one consumer GPU. Relevant to the actual
production target (a 16 GiB handheld), NOT to the desktop this was developed on.

---

## 4. Claims we tested locally and did **not** confirm

Recorded because a survey that only lists supporting evidence is advocacy.

- **T-MAC's "performance improves as bit-width decreases" (LUT-based low-bit
  GEMM).** Built as `tests/exp_lut_int4.c`. Locally the LUT **lost** to a good
  SIMD unpack — 5.62 ms vs 3.46 ms at n=1. Their published baseline is llama.cpp
  **b2794 (May 2024)**, which predates every interleaved/repacked kernel. The
  mechanism is real; the generalisation does not hold at W4A8 on Zen 5. It may
  still hold at W1/W2, which is their actual regime.
- **Published Vulkan batch-scaling from an RX 6900 XT** said batch-2 runs at 0.43×
  batch-1. On gfx1103 it measured **1.85×**. Trusting it would have killed
  speculative decoding before it produced 42.4 → 105.4 tok/s.
- **Our own earlier int4 measurement.** The most expensive wrong source in this
  file was written by us: "int4 is 0.80× at n≥4, so nibble unpacking costs more
  than it saves". It was measuring one loop nest. See the withdrawal notice in
  `README.md`.

---

## 5. What this survey says to build next

Ordered by evidence, not by appeal:

1. **Graph-level GPU submission with resident activations** (§1.3). The strongest
   single design lesson available, from a working implementation we can read, and
   now the largest remaining item — the int4 GPU kernel that was blocked here has
   since been built and run (`vulkan-headers` installed, RTX 4070, correct to
   rel 9e-07 against the CPU with a control that exceeds tolerance).
2. **Activation-aware or second-order quantization** (AWQ / GPTQ, §2) to buy back
   part of the +3.87% NLL that int4-only costs. This is the highest-value
   *accuracy* work available, and it is CPU-side, so nothing blocks it.
3. **Paged KV blocks** (§3) — the memory model, not the kernel. Matters most on
   the 16 GiB production target, where per-slot `max_ctx` reservation is the
   binding constraint.
4. **Flash attention** (§1.5). Our attention is a plain loop; this is well-trodden
   and the shader family is readable next door.

**Resolved the same day:** MoE against a real model. `muse-glimmer` (27.9B) turned
out to be **dense** — no `expert_count` key — so `Qwen3-30B-A3B` was pulled
(owner-authorized). `qwen3moe` now loads and runs: 128 experts, top-8, grouped
GEMM 1.51× over per-token on real routing, and greedy output identical to
llama.cpp character-for-character. Peak RSS 18.68 GiB at int4-only, against
roughly 33 GB for int8 — which does not fit in this machine's 30 GB, so §2's
quantization work is what made this model runnable here at all.
