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

## The LUT experiment — settled, and it says no

T-MAC claims performance improves as bit-width *decreases* even when ALU-bound,
via a lookup table instead of unpack-then-multiply. That would have flipped the
int4 result above, so it was worth settling locally rather than inheriting.

Implemented in `tests/exp_lut_int4.c`: activations grouped in 4, a 16-entry table
of every subset sum, weights decomposed into 4 bit-planes **transposed so one
table serves many output rows** (the table is shared across `o`, so `o` must be
the vector dimension), one `_mm512_permutexvar_epi16` resolving 32 output rows
per (group, plane). Storage is unchanged at `I*O/2` bytes. Verified equivalent to
the unpack path: `max|diff| / max|y| = 2.8e-07`.

Streaming regime (int4 224 MiB, int8 448 MiB, L3 96 MiB):

| n | scalar-i4 | **SIMD-i4** | LUT-i4 | int8+VNNI |
|---|---|---|---|---|
| 1 | 25.42 ms | **3.46** | 5.62 | 8.03 |
| 2 | 46.30 | 6.88 | 6.39 | 7.08 |
| 4 | 96.28 | 13.75 | 13.51 | **8.89** |
| 8 | 183.89 | 32.14 | 28.01 | **16.13** |

**LUT loses to a well-written SIMD unpack.** It beats a *scalar* unpack by 7–9×,
which is the trap: my own first run used a scalar baseline and reported 8.1× for
the LUT before I checked. T-MAC's published baseline is llama.cpp **b2794 (May
2024)**, which predates every interleaved kernel — the same kind of baseline.
Their mechanism is real; the generalisation "fewer bits is always faster" is not,
at W4A8 on Zen 5. It may still hold at W1/W2, which is their actual regime.

**Design consequence: one weight format per regime, chosen by n — int4 for
decode, int8+VNNI for prefill. No LUT.**

## MoE: grouped GEMM, which llama.cpp does not do on CPU

Source-confirmed in a local b8252 checkout: `repack.cpp`'s `forward_mul_mat_id`
contains a `gemv<...>` call and **no `gemm<...>` call at all**, so 64 tokens
routed to one expert become 64 separate GEMVs even where a GEMM kernel exists for
that weight type. `ggml_compute_forward_mul_mat_id` sorts tokens by expert and
then calls `vec_dot` with `nrc=1`.

`moe_ffn()` gathers an expert's tokens into one contiguous batch and issues **one**
`coli_gemm_i8` call, so the expert's weights are read once per group instead of
once per token — and the group size crosses `COLI_GEMM_MIN_WIDE`, which is what
lets the wide kernel be selected at all.

Measured on a synthetic 8-expert top-2 model (d=1024, ffn=2048, 4 layers, 1400
tokens): grouped **1.0 s** vs per-token **1.5 s**, identical output
(TF-NLL 7.1261, ppl 1244.061 both). `COLI_MOE_UNGROUPED=1` selects the per-token
path — it perturbs the *implementation*, the only kind of control that can fail a
differential test.

## Vulkan backend — correct, and honestly slower than the CPU here

`src/vk_backend.{h,c}` + `shaders/gemm_i8.comp`. Optional: `make VULKAN=1 vk`.
Everything else builds without it, and `coli_vk_init` returning NULL is a normal
answer — a machine with no GPU is not an error state.

**Correctness**, RTX 4070, against `coli_gemm_i8_ref`:

| shape | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| relative error vs CPU | 4.3e-07 | 4.0e-07 | 4.0e-07 | 3.8e-07 |

This is the one kernel in the engine **not** held to bit-exactness, and the
exception is deliberate: the shader reduces through shared memory in a tree, the
CPU accumulates sequentially, so they round differently and equality is
impossible. So the bound is stated (2e-05 relative to max|y|) **and** the test
ships a control — perturbing the CPU result by 0.1% must exceed the bound, and
does (1.0e-03). A tolerance nothing can fail is not a test.

**Performance: the GPU loses to the CPU at every size measured here**, and the
reason is a measurement rather than a guess. The test prints the memory type the
weights actually landed in:

```
device: NVIDIA GeForce RTX 4070  (DISCRETE)
weight memory: type2 heap1 22.8 GiB [HOST_VISIBLE HOST_COHERENT]
```

No `DEVICE_LOCAL`, and heap1 is 22.8 GiB — system RAM, not the card's 12 GB of
VRAM. Every weight read crosses PCIe. That is a direct consequence of a
deliberate choice: allocations are `HOST_VISIBLE|HOST_COHERENT` because the
intended targets (Legion 780M, laptop Vega, any APU) have **unified** memory
where a device-local copy would be pure overhead. On a discrete card it is
exactly the wrong choice, so these timings say nothing about the target hardware.

Two things follow, both not-built: a device-local staging path for discrete GPUs,
and — more important — keeping activations **resident** across layers. Right now
every call uploads activations and downloads results, which is the wrong shape
for an engine and puts a ~1 ms floor under every dispatch.

**Portability choices in the shader**, and why: weights are read as `uint32` and
unpacked with `bitfieldExtract` rather than declared as 8-bit storage.
`VK_KHR_8bit_storage` + `shaderInt8` are supported on the 780M but are not core
Vulkan 1.1 and are missing on older mobile drivers. Four extra instructions per
four weights buys a shader that runs on any Vulkan 1.1 device — which is the
entire reason for choosing Vulkan over per-vendor kernels. The weights are
uploaded in the **same** offset-to-unsigned layout the CPU uses, so no repacking
happens on either side.

**Not tested on the Legion.** Its 780M is the actual target and the only place
these numbers would mean anything, but it is the production host and this is a
GPU load test. Needs authorization first.

## Built

| file | what |
|---|---|
| `src/cpu_features.{h,c}` | runtime ISA + cache detection, x86 and ARM. Verified against `lscpu`. Returns **per-core** L1/L2 and shared L3 — the size one thread sees, not the socket total. |
| `src/gemm_i8.{h,c}` | int8 GEMM, dispatch on (ISA × n), unsigned weight storage, reference implementation, build-time negative control |
| `src/model.{h,c}` | GGUF loader, GQA attention, KV cache, per-arch RoPE, dense FFN, MoE routing + grouped GEMM |
| `src/sample.c` | greedy / temp / top-k / top-p / min-p / repetition penalty, explicit seeded xoshiro256** |
| `src/main.c` | CLI: text in, text out, `--nll` for teacher-forced validation |
| `src/server.c` | HTTP server, OpenAI-compatible `/v1/completions` + `/health` |
| `src/vk_backend.{h,c}`, `shaders/` | optional Vulkan compute backend for the int8 GEMM |
| `tests/` | bit-exactness, dispatch coverage, the LUT experiment, GPU-vs-CPU |

## Validation

The engine reproduces the previously-validated `../c/dense.c` numbers **exactly**,
once inputs and build flags match:

| model | engine | dense.c | note |
|---|---|---|---|
| qwen2.5-3b, 912 tok | TF-NLL 3.4057, ppl 30.136 | 3.4057, ppl 30.136 | identical |
| llama-3.1 WRN-8B, 912 tok | TF-NLL 3.1666, ppl 23.726 | 3.1666, ppl 23.726 | identical |

`dense.c` built with contraction *on* gives 3.4052 — the whole 0.0005 discrepancy
was FMA contraction, confirmed by rebuilding it with `-ffp-contract=off`. That is
the second time in one session that contraction masqueraded as a numerical bug.

Sampler: same seed → byte-identical output across runs; different seed → different
output (the control fires).

Server, all found by tests rather than by reading:

- **same seed returned different text across requests.** The RNG was a file
  static, so the second request skipped the reseed and continued the previous
  stream. Fixed by moving the state into `coli_sampler`; three identical requests
  now give one hash, and a different seed still differs.
- **EOS was rendered into the returned text** as a literal `<|im_end|>`.
- **SIGTERM was ignored** — `signal()` installs restarting handlers on Linux, so
  `accept()` resumed instead of returning `EINTR` and the listening socket stayed
  bound after `kill`. `sigaction` with `sa_flags = 0`. Caught by checking `ss`
  after the test, not by the test itself.
- KV is reset per request; verified by issuing the same prompt before and after a
  different one and requiring identical output. Without it, request N+1 attends to
  request N's tokens — a cross-request leak, not merely a wrong answer.

## Not built

Stated so this file cannot be mistaken for a finished engine:

- **The GPU backend is CORRECT but not yet FAST, and only the GEMM is on it.**
  See the Vulkan section below.
- **The server has ONE slot.** No batching across requests, no continuous
  batching, no streaming, no auth, no TLS. Time-to-first-token therefore grows
  linearly with queue depth — the exact limitation the brief that started this
  work wrongly attributed to C++ engines in general. It is a property of a
  single-slot scheduler, not of the language, and llama.cpp does not have it.
- **`WQ=f32` is not wired** — `coli_load`'s `wq_int8` argument is ignored, so a
  new architecture cannot yet be validated in f32 before int8.
- **MoE is tested against a SYNTHETIC model only.** The routing, grouping and
  expert-tensor split are exercised; correctness against a real MoE architecture
  is NOT established, and real MoE archs (`qwen2moe`, `mixtral`) are still
  refused by the architecture check. No MoE GGUF exists on this machine to test
  with.
- `coli_cache_bytes()` is not yet consulted by the dispatch — residency is
  assumed from n, not measured.
- No int4 path in the engine yet: the 2.3×-at-n=1 result is measured in the
  benchmark, not implemented in `gemm_i8.c`.

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
