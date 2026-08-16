# Colibri engine

A from-scratch C inference engine for dense and MoE transformers, targeting
Linux, Windows, macOS and Android on CPU and GPU.

This directory is **the kernel and dispatch foundation**, not yet a whole
engine. What is here is built and measured; what is not here is listed at the
bottom as not-built. Nothing in this file is a projection.

## The one idea

> **The fastest kernel is a function of batch size, not just of ISA.**

Every mainstream CPU engine picks its kernel from the instruction set. That is
not sufficient, and the measurement is unambiguous.

AMD Ryzen 9800X3D, int8 GEMM, min of 9 reps after warmup, 2026-08-16:

| weights | n=1 | n=2 | n=4 | n≥8 |
|---|---|---|---|---|
| **448 MiB, streaming from RAM** | VNNI **0.83×** | **0.78×** | 1.17× | 1.18–1.26× |
| **16 MiB, resident in 96 MiB L3** | 1.28× | 1.28× | 1.28× | 1.28× |

Below n=4 in the streaming regime the loop is bound by **weight bytes** — both
kernels sit on the same **~70 GB/s** DRAM ceiling — so the wider ISA is strictly
worse. Turning AVX-512 VNNI on unconditionally is a **17–22% regression on
decode**, the path that runs for every generated token.

The same effect decides the weight *format*:

| n | int8 (448 MiB) | int4 (224 MiB) | winner |
|---|---|---|---|
| 1 | 6.76 ms | **3.35 ms** | **int4, 2.02×** |
| 2 | 7.15 ms | 6.61 ms | ~tie |
| 4 | **10.77 ms** | 13.23 ms | **int8** |
| 8 | **22.18 ms** | 27.57 ms | **int8** |

Both formats hit the same ~70 GB/s at n=1; int4 simply moves half the bytes. At
n≥4 the regime is ALU-bound and nibble unpacking costs more than it saves.

So the dispatch key is **(ISA × batch size × residency)**. `src/gemm_i8.c`
implements it; `coli_gemm_i8_kernel()` reports the choice without running it, so
a benchmark can prove the dispatch actually moved.

## Corroboration, and where we differ

Read from a local llama.cpp checkout at `b8252` (verified file:line, not recalled):

- `ggml/src/ggml-cpu/repack.cpp:3104` — *"If there are more than three rows in
  src1, use gemm; otherwise, use gemv"*. **llama.cpp independently chose 4.** It
  also runs GEMM over the largest multiple of 4 and GEMV for the remainder; we
  copy that tail handling.
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp:3689` — `if (n < 2) return false;`,
  commented *"only enable sgemm for prompt processing"*.
- MLX (`mlx/backend/metal/quantized.cpp`, `get_qmv_batch_limit`) makes the limit
  a tuned function of `(batch, K, N, GPU generation)` returning 12–33. So **4 is
  a floor for our shapes, not a universal constant** — hence `COLI_GEMM_MIN_WIDE`
  is a tunable.

Where we go further: **no engine surveyed switches weight *format* on batch
size.** They pick one on-disk format and switch kernels. Whether the dual-format
memory cost is worth it is an open question this engine is built to answer.

Two things llama.cpp leaves on the table, both source-confirmed:

- **CPU MoE is GEMV-only at every batch size.** `repack.cpp`'s
  `forward_mul_mat_id` contains a `gemv<...>` call and no `gemm<...>` call at
  all, even though a working GEMM kernel exists for the same weight type. 64
  tokens routed to one expert become 64 separate GEMVs.
- **Q5_K and Q6_K do not repack on x86** — their dispatch has only
  `ggml_cpu_has_neon()` guards. A `Q4_K_M` GGUF mixes Q4_K with Q6_K tensors, so
  part of every such model stays on the slow path on both our targets.

## Non-negotiable: no FP contraction

`-ffp-contract=off`, plus a pragma in `gemm_i8.c` for builds that forget the flag.

This is not pedantry. With contraction on, the reference implementation and every
dispatched kernel disagreed on **76% of output cells at n=1 (1,557 of 2,048)**
while their *integer* dots were identical — the compiler contracted `a*b+c` in
one function and not the other. With it off: **0 of 266,240**. An engine whose own
reference test cannot agree with its kernels cannot make a bit-exactness claim
about anything.

## Weight storage is offset-to-unsigned

AVX-512 `VPDPBUSD` is u8 × s8. Offsetting the **activations** needs a
per-weight-block sum table (+12.5% of model size). Offsetting the **weights**
makes the correction `128*sum(activation block)` — n·I values against I·O
weights, i.e. free:

```
dpbusd(u, x) = Σ(q+128)·x = Σq·x + 128·Σx
```

exact in integer arithmetic. One weight array serves both kernels; the AVX2 path
widens unsigned and subtracts 128 in int16, one extra instruction per 16 weights,
in the regime that is DRAM-bound anyway.

Holding a signed *and* an unsigned copy costs **6.45 GiB vs 3.28 GiB** on a 3B
model. Measured, after shipping exactly that mistake.

### The int16 overflow trap

ik_llama.cpp PR #141, the author's own words: *"the unsigned integers still need
to be within 0…127, else adding up two adjacent products may overflow the int16_t
range (and gets silently truncated if it does)"*. Their buggy version scored
perplexity 7.3725 against 7.3443 correct — **small enough to pass a smoke test.**
Any kernel here that accumulates in int16 must state why it cannot overflow. The
AVX2 path is safe because `_mm256_madd_epi16` accumulates into int32; the VNNI
path is safe because `dpbusd` accumulates into int32 directly.

## Testing rules

- Every dispatched kernel must be **bit-exact** with `coli_gemm_i8_ref`. Not
  close — equal. A kernel that is not bit-exact is a different algorithm and gets
  its own entry point so the difference is visible.
- **Every test ships a negative control that must fail.** `make test` builds
  `test_gemm_i8_broken` with `-DCOLI_BREAK_WIDE` and fails the build if that
  binary *passes*. A control that perturbs something both sides share — the input
  file, the model, the corpus — is inert by construction and proves nothing.
- The test refuses to report PASS if only one kernel was exercised
  (`INCONCLUSIVE`, rc=3). A comparison that silently ran the same kernel twice is
  the failure mode this prevents.

```
make test
```

## Built

| file | what |
|---|---|
| `src/cpu_features.{h,c}` | runtime ISA + cache detection, x86 and ARM. Verified against `lscpu`. Returns **per-core** L1/L2 and shared L3 — the size one thread sees, not the socket total. |
| `src/gemm_i8.{h,c}` | int8 GEMM, dispatch on (ISA × n), unsigned weight storage, reference implementation, build-time negative control |
| `tests/` | bit-exactness + dispatch-coverage tests |

## Not built

Stated so this file cannot be mistaken for a finished engine: no model loader
here yet (the validated GGUF reader, metadata reader and tokenizer live in
`../c/` and are to be moved), no attention, no KV cache, no sampler, no MoE
routing, no GPU backend, no server. `coli_cache_bytes()` is not yet consulted by
the dispatch — residency is currently inferred by the caller, not measured.

## GPU plan, and why

From the survey, the empirical answer is **three kernel families, not four**:

- **CUDA source compiled for both CUDA and HIP/ROCm.** `ggml/src/ggml-hip/`
  contains exactly one file — a `CMakeLists.txt` that globs `../ggml-cuda/*.cu`.
  Source reuse across API dialects is nearly free.
- **Metal.**
- **Vulkan** — covers Android and the Legion's Radeon 780M in one write.

**Not SYCL.** llama.cpp's `docs/backend/SYCL.md` changelog, entry 2026.02:
*"Remove support for Nvidia & AMD GPU, because the oneAPI plugin for Nvidia & AMD
GPU is unavailable"*. The cross-vendor promise died on toolchain distribution,
not performance. **Not Kompute** — removed from llama.cpp in PR #14501,
*"development for this backend has stopped"*. A portability layer you do not
control is a dependency that can strand you.

## Open questions this engine exists to settle

1. **LUT-based int4.** T-MAC (Microsoft) claims performance improves as bit-width
   *decreases* even when ALU-bound — the opposite of our n≥4 result. Different
   mechanism: a table lookup instead of unpack-then-multiply. Their baseline is
   llama.cpp **b2794 (May 2024)**, which predates all the interleaved kernels, so
   the 4–5× is against a straw man. Needs settling locally, and it is the only
   documented route to making int4 win at n≥4.
2. **Grouped-GEMM MoE on CPU.** Source-confirmed unclaimed headroom.
3. **Does the crossover move with K and N?** MLX says yes and returns 12–33. Our
   4 comes from one shape.

## Measurement caveat that applies to everything above

Every number here is from the **desktop** 9800X3D: 96 MiB L3, Zen 5. The
production target is a Legion Go S — Z1 Extreme, Zen 4, **16 MiB L3**, unified
LPDDR5, Radeon 780M. Desktop measurements **systematically over-report the
cache-resident regime**, which is exactly the regime where VNNI looked best. The
streaming numbers are the ones that transfer. **Nothing here has been run on the
Legion.**
