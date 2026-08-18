# Colibri engine

A from-scratch inference engine for dense and MoE transformers, targeting Linux,
Windows, macOS and Android on CPU and GPU.

## Three layers, split by what each language is actually for

| layer | language | owns | why |
|---|---|---|---|
| core | **C++20** | kernels, model, forward pass, batched decode | intrinsics, manual memory layout, compile-time specialisation. Its int8 GEMM dispatches on **(ISA × batch size)** — a decision only the compute layer can make. |
| serving | **Go** | HTTP, the continuous-batching **scheduler**, slot allocation, cross-compilation | scheduling is concurrency plumbing; goroutines and channels express it in a fraction of the code, and one toolchain targets every OS |
| scale-out | **Python** | fan-out, routing, retries, orchestration, evaluation | many processes, logic that changes weekly without recompiling |

**Python is deliberately NOT in the token path.** Every call in `python/` talks
to a Go worker over HTTP; none touch a tensor. Putting an interpreter between the
sampler and the KV cache would add latency to every token — the exact mistake the
split exists to avoid.

The boundary is `src/coli_api.h`: a flat `extern "C"` ABI. cgo and ctypes can
call C, not C++, so the engine keeps RAII and templates internally and exposes
POD-only functions outward. MLX is built the same way, for the same reason.

```
make            # libcoli.so + coli (CLI) + go/coli-server
make test       # kernel bit-exactness + a control that must fail
make windows    # cross-compile to coli.exe
```

### Fused prefill

`coli_prefill_slot` used to step the batch path once per token. Same maths,
catastrophically wrong shape: it read every weight matrix once **per token**
instead of once for the prompt. Measured on 331 tokens, qwen2.5-3b:

| | time |
|---|---|
| stepped (one forward per token) | 34.5 s |
| **fused (one forward for the prompt)** | **7.1 s** |

**4.9×**, and the Go server paid it on every request. Verified against the
reference forward path: identical greedy output.

### Measured, end to end

Continuous batching, qwen2.5-3b, identical output at every batch size (0
mismatches — batching changes the *shape* of the work, never the result):

| batch | tok/s total | per sequence |
|---|---|---|
| 1 | 9.6 | 9.6 |
| 2 | 19.5 | 9.7 |
| 4 | 30.3 | 7.6 |
| 8 | **33.5** | 4.2 |

Through the full stack (Python → Go → C++), with the server's own batch
histogram confirming it actually batched rather than a mean that would hide it:

| concurrency | tok/s | server ran batches |
|---|---|---|
| 1 | 12.8 | `{1: 20}` |
| 2 | 20.3 | `{2: 20}` |
| 4 | **27.5** | `{4: 20}` |

**This is the claim from the original brief — that C++ engines have "entirely
flat" throughput — measured and retired.** Throughput was never a property of
the language; it was a property of having one slot.

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

Both formats hit the same ~70 GB/s at n=1; int4 simply moves half the bytes.

> ⚠️ **The n≥4 half of this table was a kernel defect, not a property of int4,**
> and the conclusion originally drawn from it ("nibble unpacking costs more than
> it saves") is **withdrawn**. The int4 kernel unpacked each block once per
> activation row, so those n≥4 figures are the unpack repeated n times. With the
> unpack hoisted out of that loop, int4 at n=4 goes 12.59 → 5.83 ms and lands at
> 1.18× of int8 rather than 1.23× *worse*. The table is left standing because the
> row it got right — n=1 — is the one the design still rests on, and because a
> withdrawn measurement is more useful visible than deleted. See
> *"That open question, answered twice"* below.

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
size.** They pick one on-disk format and switch kernels.

### That open question, answered twice — the second answer reverses the first

**First attempt (`--w4 1`), and its verdict was "not like this."** int4 weights
with per-32 block scales carried *alongside* int8, dispatched by batch size. It
bought 1.42× decode for **+56% memory**, and on a 16 GiB production handheld with
~5 GiB free that is the wrong resource to spend. Recorded as rejected.

That verdict rested on a number that turned out to be measuring the wrong thing.
The int4 kernel unpacked each weight block **inside its loop over activation
rows**, so an n-row call unpacked the whole matrix n times. The tell was in the
original table and I read past it: 3.46 ms at n=1 to 13.75 ms at n=4 is 3.97× for
4× the rows, which is not "int4 is bad at prefill" — it is "the dot product has
become free next to the unpack". int4 was being blamed for a property of one
loop nest.

**Hoisting the unpack out of that loop** (a scratch row per thread, reused across
the batch) changes the kernel comparison completely. 16384×16384, int8 256 MiB
against 96 MiB of L3 so the matrix genuinely **streams**, min of 7 reps:

| | int4 narrow (before) | int4 wide (after) | int8, its own best | int4 wide ÷ int8 |
|---|---|---|---|---|
| n=1 | 3.03 ms | **1.89 ms** | 3.10 ms | **0.62×** (median of 6) |
| n=2 | 6.06 ms | **3.14 ms** | 3.29 ms | 0.95× |
| n=4 | 12.59 ms | **5.83 ms** | 4.95 ms | 1.18× |
| n=32 | 101.96 ms | **48.17 ms** | 45.04 ms | 1.07× |

**0.62× at n=1 is the number that matters** — median of six independent runs
(0.63, 0.62, 0.62, 1.20, 0.50, 0.62; the outliers are memory-bandwidth contention
from other processes, which a streaming kernel at ~84 GB/s is fully exposed to).
The int4/int8 **byte** ratio is 0.625, so decode has landed on the bandwidth
ratio and there is nothing further to win there.

⚠️ An earlier version of this line said **0.61×** and called it "exactly" the byte
ratio. That was one sample presented as a point measurement. The median supports
the conclusion; the single figure did not support the word "exactly". The old kernel managed 52 GB/s against int8's 83 GB/s — it was
ALU-bound on its own unpack *even at n=1*. So the int4 threshold is **1**, not
the 4 that is right for int8: int8's two kernels differ only in ISA width against
one DRAM ceiling, int4's differ in how many times the matrix gets unpacked.

**Second attempt (`--w4 2`): int4 only, no int8 form built at all.** qwen2.5-3b,
the frozen 681-token prose prompt in `tests/nll_prompt.txt`, decode path
(`--nll1`, one token per step so every GEMM really runs at n=1 — plain `--nll`
prefills in one call and would have silently measured the wide kernel instead):

| qwen2.5-3b | NLL | ppl | decode | prefill | peak RSS |
|---|---|---|---|---|---|
| int8 (`--w4 0`) | 2.7845 | 16.192 | 73.4 s | **14.6 s** | 3.39 GiB |
| both (`--w4 1`) | 2.8922 | 18.033 | 50.7 s | — | 5.17 GiB |
| int4 only (`--w4 2`) | 2.8922 | 18.033 | **50.1 s** | 15.8 s | **2.29 GiB** |

Repeated on **llama-3.1 WRN-8B**, the case where the memory actually binds — an
int8 8B does not fit on the 16 GiB production handheld alongside the rest of the
fleet:

| llama-3.1 WRN-8B | NLL | ppl | decode | peak RSS |
|---|---|---|---|---|
| int8 (`--w4 0`) | 2.5419 | 12.703 | 162.3 s | 8.43 GiB |
| int4 only (`--w4 2`) | 2.5984 | 13.442 | **109.3 s** | **5.81 GiB** |

**1.48× decode, −2.62 GiB, +2.22% NLL** — a *smaller* relative accuracy cost than
the 3-B model paid, on the model where the saving is worth the most.

Decode times are single runs and repeat to within ~3% (int4 measured 51.6 s and
50.1 s on two builds of identical arithmetic); the RSS and NLL figures are
deterministic.

**1.42× faster decode, 1.09 GiB smaller, 1.08× slower prefill, +3.87% NLL.**
Against the rejected dual-format build it is the same speed and *the same NLL to
four decimals* — which is the cross-check that the int4 path really is the only
one running — for **2.88 GiB less memory**.

The accuracy cost is real and is the price. It is also **not fixed**, and the
cheapest way to buy some of it back is in the quantizer rather than the kernel.

### The int4 kernel on the GPU

Measured RTX 4070, weights staged into VRAM (DEVICE_LOCAL), min of 5 dispatches,
4096×16384 — int8 64 MiB against int4 40 MiB, byte ratio 0.625:

| n | int8 GPU | int4 GPU | int4 ÷ int8 |
|---|---|---|---|
| 1 | 0.35 ms | **0.22 ms** | **0.66×** |
| 2 | 0.52 | 0.51 | 0.83× (min-of-5 across runs) |
| 4 | 1.02 | 0.85 | 0.83× |
| 8 | 2.06 | 1.66 | 0.81× |

Correct to **rel 4e-07 to 9e-07** against the CPU int4 kernel, with the
`cpu × 1.001` control exceeding tolerance as required. Note the CPU landed
*exactly* on the 0.625 byte ratio at n=1 and the GPU does not (0.66×, then ~0.82×
above n=1) — the GPU has enough spare bandwidth that the nibble unpack is no
longer free, which is the opposite of the CPU's problem.

**And the integer-vs-float question from `RESEARCH.md` is now measured, not
argued.** `shaders/gemm_i4f.comp` is the identical shader with one line changed
to dequantize to float the way ggml does. Integer wins by **~5%** and ties at
n=2 — small enough that it *vindicates* ggml's choice rather than beating it:
they pay ~5% on this shape and get one kernel that serves twenty formats.

### Activations resident on the GPU — and which half of it actually paid

`coli_vk_ffn4` runs a whole SwiGLU FFN with **one upload and one download**:
both projections, the nonlinearity, and the requantized activation the
down-projection needs all stay in device memory. What made it possible is
`shaders/silu_mul_q.comp` — the down-projection needs int8 activations with
per-block scales and sums, and until the GPU could produce those, the
intermediate had to come back to the CPU and residency was impossible by
construction.

It was built in two steps, and the split is the interesting part:

| | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| residency only, still 4 submits | 0.95× | 1.07× | 1.02× | 1.13× |
| **residency + ONE submission** | **1.51×** | **1.49×** | 1.15× | 1.31× |

**Keeping the data on the device was worth about 5%. Recording all four
dispatches into one command buffer and waiting on one fence was worth the rest.**
The first version still submitted per operator and waited on a fence each time,
which is exactly what ggml's own comment warns about —
*"Submit after enough work has accumulated, to overlap CPU cmdbuffer generation
with GPU execution"* (`ggml-vulkan.cpp:14103`). Had I stopped at the fused-memory
version I would have concluded that residency barely matters, and been wrong
about why.

Correct to **rel 2.9e-07 to 3.5e-07** against the same FFN computed entirely on
the CPU — well inside the 5e-3 the test allows, which is deliberately loose
because `silu` uses the GPU's `exp()` and the intermediate is requantized
independently on each side.

Not yet done: this is one FFN, not a graph. ggml batches ~100 nodes or ~100 MB of
matmul per submit across the *whole model*; we batch four dispatches within one
layer's FFN. Attention, RoPE and the norms are still CPU-side, so nothing above
runs inside a real forward pass yet.

### Buying back the accuracy: a least-squares scale search

`amax/7` picks the scale that makes the largest weight in a block representable.
That is not the scale that minimises error over the block — one outlier drags the
step size up and the other 31 weights pay for it. llama.cpp already solves this
for its K-quants, and the algorithm is readable:
`ggml/src/ggml-quants.c:451`, `make_qx_quants()` — take the least-squares optimal
scale after rounding, then search 18 candidates either side of the amax-derived
one (`iscale = -(nmax + 0.1f*is)/max`, `is` in −9…9) weighting each error by
`x*x`, and keep whichever maximises `sumlx²/suml2`.

Ported to our 32-weight blocks (`w4_block_scale`), same output format, same
kernels, cost at **load time only**:

| qwen2.5-3b, int4-only | NLL | ppl | vs int8 | load |
|---|---|---|---|---|
| `--w4 12` amax/7 | 2.8922 | 18.033 | +3.87% | 7.4 s |
| **`--w4 2` search (default)** | **2.8596** | **17.455** | **+2.70%** | 14.5 s |
| int8 reference | 2.7845 | 16.192 | — | 2.6 s |

| llama-3.1 WRN-8B, int4-only | NLL | ppl | vs int8 | decode | peak RSS |
|---|---|---|---|---|---|
| `--w4 12` amax/7 | 2.5984 | 13.442 | +2.22% | 109.3 s | 5.81 GiB |
| **`--w4 2` search (default)** | **2.5651** | **13.002** | **+0.91%** | 110.5 s | 5.81 GiB |
| int8 reference | 2.5419 | 12.703 | — | 162.3 s | 8.43 GiB |

**It recovers 30% of the quantization gap on the 3B and 59% on the 8B**
(0.1077 → 0.0751 and 0.0565 → 0.0232 nats) for load time alone. Decode speed and
peak RSS are unchanged to the digit — 110.5 s vs 109.3 s and 5.81 GiB both ways —
which is what it should be, since only the values of the scales differ.

> Two corrections behind those rows, both mine:
>
> - An earlier 8B row read 124.9 s and 5.55 GiB. Both were artifacts of running
>   `make windows` and `make go` concurrently. Quiet, it is 110.5 s and 5.81 GiB.
>   A timing taken while you compile something else is not a timing.
> - The amax rows were first measured under a flag spelled `--w4 21`, which the
>   digit-packed mode decoded as format **1** (both formats), not format 2. The
>   NLL was identical either way, because dual-format uses int4 at n=1 — only the
>   peak RSS gave it away, 11.56 GiB against int4-only's 5.81 GiB. The numbers
>   above are from runs that really were int4-only; the flag is now `--w4 12`, and
>   an unrecognised mode is **rejected** instead of quietly becoming int8. That
is why the search is the DEFAULT for int4 and `--w4 12` selects the old quantizer
— the flag exists so the two remain comparable on one build, not because either
is a preference.

This is the cheap end of a known research direction, and worth being honest about
which end: **AWQ** ([2306.00978](https://export.arxiv.org/abs/2306.00978)) decides
which weights matter from *activations*, and **GPTQ**
([2210.17323](https://export.arxiv.org/abs/2210.17323)) uses second-order
information. Both need calibration data. `x*x` weighting is the version that needs
nothing at all, and it captured a third of the gap.

> ⚠️ **The first version of this measurement was wrong and looked right.** The
> mode was encoded as a tens digit and decoded with `w4 -= 10`, which turned 22
> into 12 — no valid format, so it silently fell back to int8 while the label,
> computed separately with `% 10`, still printed "int4 [RMSE scale search]". It
> reported NLL 2.7845 and ppl 16.192: a *better* result, and exactly the int8
> baseline to four decimals. Two places deriving the same thing two different ways
> is the bug; a result that matches another configuration exactly is the tell.

Whether the remaining +2.70% is acceptable is a deployment decision, not an engine
one, so int4-only stays opt-in rather than the default.

Two things worth stating about how these numbers were obtained, because the
earlier round got both wrong:

- **The prompt is committed** (`tests/nll_prompt.txt`). The previous baseline's
  prompt lived in `/tmp`, did not survive a reboot, and took its NLL figure with
  it — a number nobody can reproduce is not a baseline.
- **Prose, not boilerplate.** Run against the AGPL text this same comparison
  reports ppl 1.159 → 1.654, a 3.4× ratio, purely because near-zero-entropy text
  amplifies a small absolute error. Same code, same day, wildly misleading number.

Note the scales are per-32-block, not per-row: int8 gets away with one scale per
output row, int4 has 16 levels instead of 256 and does not. `token_embd.weight`
stays int8 in every mode — it is indexed row-wise rather than multiplied, so it
never reaches the GEMM, and it contributes ~D bytes per token against the I×O a
weight matrix costs.

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

### The 127.5 tie, which flips a sign

`lrintf(127.5) = 128` — on both libcs. With weights stored offset-to-unsigned,
`128 + 128 = 256` wraps a `uint8` to `0`, which decodes as **−128**: a sign flip
on the largest weight in the row. The scale is `am/127`, so `|r·inv|` should
reach exactly 127 and never 127.5 — but "should" is doing a lot of work there,
and one compare per weight is cheaper than discovering which model triggers it.
Both quantizers now clamp to ±127. Surfaced while chasing the cross-platform
difference above; it was never the cause of that, just something the probe walked
past.

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

**Design consequence: no LUT.** The rest of that conclusion — "one weight format
per regime, chosen by n" — was superseded once the int4 kernel stopped
re-unpacking per row. int4-only across both regimes now costs 1.08× on prefill
and saves 1.09 GiB, which beats carrying two formats to avoid that 8%.

## MoE: grouped GEMM, which llama.cpp does not do **on CPU**

Source-confirmed in a local b8252 checkout: `repack.cpp`'s `forward_mul_mat_id`
contains a `gemv<...>` call and **no `gemm<...>` call at all**, so 64 tokens
routed to one expert become 64 separate GEMVs even where a GEMM kernel exists for
that weight type. `ggml_compute_forward_mul_mat_id` sorts tokens by expert and
then calls `vec_dot` with `nrc=1`.

**Scope of that claim, checked rather than assumed: it is about the CPU path
only.** Re-verified 2026-08-17 against a newer local checkout (`c96f608d9`), the
Vulkan backend *does* have an MoE GEMM —
`ggml/src/ggml-vulkan/ggml-vulkan.cpp:8626`, `ggml_vk_mul_mat_id()` picks
`ggml_vk_mul_mat_vec_id_q_f16` or `ggml_vk_mul_mat_id_q_f16` depending on
`ggml_vk_use_mul_mat_vec_id()`, and there is a fused `topk_moe.comp` for the
routing. So the gap we exploit is specific to CPU, and worth stating that way:
"llama.cpp does not do this" would have been wrong by a whole backend.

`moe_ffn()` gathers an expert's tokens into one contiguous batch and issues **one**
`coli_gemm_i8` call, so the expert's weights are read once per group instead of
once per token — and the group size crosses `COLI_GEMM_MIN_WIDE`, which is what
lets the wide kernel be selected at all.

Measured on a synthetic 8-expert top-2 model (d=1024, ffn=2048, 4 layers, 1400
tokens): grouped **1.0 s** vs per-token **1.5 s**, identical output
(TF-NLL 7.1261, ppl 1244.061 both). `COLI_MOE_UNGROUPED=1` selects the per-token
path — it perturbs the *implementation*, the only kind of control that can fail a
differential test.

### And now on a real MoE, which it had never been

`Qwen3-30B-A3B` (`qwen3moe`, **128 experts, top-8**, 48 layers, d=2048,
expert-ffn 768, GQA 32/4), 681-token prompt, `--w4 2`:

| | TF-NLL | ppl | prefill |
|---|---|---|---|
| grouped (default) | 2.5656 | 13.008 | **25.5 s** |
| `COLI_MOE_UNGROUPED=1` | 2.5656 | 13.008 | 38.4 s |

**1.51× for free, identical to four decimals** — the same ratio the synthetic
model showed, now against real routing where group sizes vary per token.

**The architecture is cross-checked, not merely plausible.** Greedy, temperature
0, same prompt, ours against llama.cpp through ollama on the same GGUF:

```
ours:      " Berlin. What is the capital of Italy?\n\nOkay, the user is asking about"
llama.cpp: " Berlin. What is the capital of Italy?\n\nOkay, the user is asking about"
```

Character-for-character identical over 16 tokens — and note the two sides
quantized the weights *differently* (our int4 block search vs their Q4_K_M), so
this is agreement on the architecture, not on the arithmetic. That matters
because a wrong architecture does not error; it produces fluent nonsense. This
project has the scar: the llama path once ran at ppl 639 against llama.cpp's 29.3
with nothing in the logs.

**Peak RSS: 18.68 GiB at int4-only.** The same model in int8 would be roughly
33 GB and does not fit in this machine's 30 GB at all — so on this desktop int4
is not an optimisation for this model, it is the difference between running and
not running.

## Portability

The brief was "runs on any OS". It did not: the loader used `pread`/`open`/
`fstat`, CPU detection used `getauxval`, and the server used BSD sockets and
`sigaction`. All POSIX — on Windows that meant **WSL2 only**, which is not the
same claim.

`src/platform.h` puts those ten calls in one place. Deliberately tiny, because
each one is a place two operating systems can disagree:

| | POSIX | Windows |
|---|---|---|
| positional read | `pread` (looped — short reads are legal) | `ReadFile` + `OVERLAPPED` offset; `_lseek`+`_read` would move the shared file pointer |
| file size | `fstat` | `GetFileSizeEx` |
| socket close | `close` | `closesocket` |
| net init | ignore `SIGPIPE` | `WSAStartup` |
| signals | `sigaction`, `sa_flags = 0` | `signal` (no restart distinction) |

The `sa_flags = 0` is not a detail: `signal()` installs **restarting** handlers on
Linux, which is exactly how the server shipped ignoring SIGTERM. Making the
non-restarting form the only way to install a handler stops it recurring.

### It actually builds and runs on Windows

Cross-compiled with mingw-w64 and executed under wine — not asserted, run:

```
PE32+ executable ... x86-64
qwen2: 36 layers, d=2048, heads=16/2, ...
 Paris. The capital of Spain is Madrid. The capital of
generated 12 tokens in 0.83s (14.5 tok/s)
```

Built `-march=x86-64-v3` (AVX2, no AVX-512), which is exactly what a Zen 3 laptop
CPU supports. Three real gaps the cross-compile found, none visible on Linux:

- `platform.h` was missing `<fcntl.h>`/`<signal.h>` on the Windows branch.
- `aligned_alloc` is **absent from the mingw CRT**, and Windows pairs
  `_aligned_malloc` with `_aligned_free` — calling plain `free()` on that pointer
  is undefined behaviour, not a leak. Hence `coli_aligned_alloc`/`coli_aligned_free`
  as a pair.
- A **dynamically** linked build starts and silently produces nothing (missing
  `libgomp-1.dll`/`libwinpthread-1.dll`). Link `-static`.

`../c/compat.h` already had a Windows layer — including a `compat_pread` built on
`ReadFile`+`OVERLAPPED`, the same approach `platform.h` arrived at independently,
with the same "never `_read`/`_lseeki64`" reasoning. It only needed
`-D_FILE_OFFSET_BITS=64`. **That duplication is on me for not checking first.**

### Determinism holds across platforms — after the cause was found

**Resolved.** Linux and Windows builds now produce byte-identical output: the
entire forward pass checksums the same at every traced stage (156 trace lines,
zero differences), and greedy generation is character-for-character equal.

**Root cause: `sinf` differs by 1 ULP between glibc and msvcrt.** At RoPE
frequency index 7, `sinf(1*fr)` is `0x3e6023da` on glibc and `0x3e6023d9` on
msvcrt. Neither is wrong — IEEE-754 requires correct rounding for `+ - * /` and
`sqrt` and explicitly does **not** for transcendentals, so every libm is free to
differ in the last place. That one bit enters through RoPE, propagates through
attention, and eventually flips a near-tie argmax.

**How it was found**, after three wrong answers:

1. Ruled out weights (bit-identical checksums), `lrintf` (identical incl. ties),
   thread count (2.9016 at 1/2/4/8/16), and the kernels.
2. A libm probe compared `expf`/`powf`/`sinf`/`cosf`/`sqrtf` at eight arbitrary
   points, found every bit equal, and **wrongly cleared libm**. It exercised the
   functions but not the *arguments the engine uses* — the same defect as a test
   that never reaches the code path it claims to cover.
3. `COLI_TRACE=1` checksums the hidden state at named stages. First divergence:
   `rope_q`, and only at `pos != 0` where the rotation is not the identity.
4. Comparing **all 64** RoPE frequency indices instead of a sample of 8 found it
   in one run.

**The fix** (`src/trig.h`): compute sin, cos and pow from `+ - * /` only —
operations IEEE-754 *does* require to be correctly rounded — via Cody-Waite range
reduction and Taylor series in double. Accuracy against libm: max abs error
**8.4e-11** for sin/cos, two orders below a float ULP.

Effect on model quality is **noise, not improvement**, and it moved in both
directions: qwen 3.4057 → 3.4068, llama 3.1666 → **3.1651**. The win here is
reproducibility, not accuracy, and claiming otherwise would be reading a rounding
change as a result.

`COLI_TRACE=1` is kept — it is the tool that found this, and the next such
divergence will need it too.

### The old finding, superseded

Same source, same flags, same ISA target: Linux and Windows builds produce
**different** output. Greedy generation diverges after a few tokens, and TF-NLL
differs from the very first forward pass (1.1744 vs 1.1732 over 3 tokens), so it
is not accumulation.

What has been **ruled out by measurement**, not by reasoning:

| candidate | result |
|---|---|
| loaded weights | **bit-identical** — FNV checksums of `qu` and `scale` match on all three tensors checked |
| `expf`/`powf`/`sinf`/`cosf`/`sqrtf` | **bit-identical** hex patterns across glibc and msvcrt |
| `lrintf` rounding (incl. ties) | **identical**, both `FE_TONEAREST`, half-to-even |
| OpenMP thread count | TF-NLL **2.9016 at 1, 2, 4, 8 and 16 threads** — the engine is thread-deterministic |
| GEMM kernel vs its reference | `bad=0` on both platforms |

So the residual is in the non-GEMM float code, and **it is not isolated.** Stated
as an open finding rather than left for someone to discover. Practical effect:
reproduce a generation on the platform that produced it. (llama.cpp has the same
property; the difference is that this is written down.)

Note the kernel cross-check above is weaker than it looks and is not evidence of
cross-platform equality: the test seeds its data with `rand()`, and glibc and
msvcrt do not generate the same sequence, so the two runs tested *different* data.

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

**Performance: the GPU now beats the CPU at every size**, after two fixes that
had nothing to do with the kernel.

| n | GPU before | **GPU after** | CPU |
|---|---|---|---|
| 1 | 3.53 ms | **0.27** | 1.95 |
| 2 | 7.27 | **0.55** | 2.10 |
| 4 | 14.08 | **1.13** | 2.47 |
| 8 | 26.87 | **2.03** | 4.41 |

*(4096×16384; correctness unchanged at 5.8e-07 to 7.1e-07, control still fires.)*

**1. Buffers are persistent.** The first version allocated four buffers, uploaded,
dispatched, downloaded and destroyed them — every call. That put a floor under
each dispatch larger than the arithmetic. They are now allocated once at the
high-water mark and reused, and the descriptor set is allocated once instead of
per call.

**2. Weights live in VRAM on discrete cards.** They were `HOST_VISIBLE`, which on
a discrete GPU means system RAM and a PCIe crossing per read — the test measured
`heap1 22.8 GiB [HOST_VISIBLE HOST_COHERENT]` against the card's 12 GiB of VRAM.
Weights are written once and read every step, so a one-time staging copy is
obviously right there and pointless on a unified-memory APU. **The device type
decides**, and the test now reports `heap0 12.0 GiB [DEVICE_LOCAL]`.

⚠ The warning the test prints is now conditional on where the weights actually
landed. It previously said "host-visible → PCIe" unconditionally, and after the
device-local path landed it was printing that directly above numbers proving the
opposite — a stale warning is worse than none.

These are still desktop numbers on a discrete card. The intended target is the
Legion's 780M, where memory is unified and no staging happens at all, and it has
**still not been run there**.

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
| `src/platform.h` | the ~10 OS calls the engine uses, POSIX + Win32 |
| `src/vk_backend.{h,c}`, `shaders/` | optional Vulkan compute backend for the int8 GEMM |
| `tests/` | bit-exactness, dispatch coverage, the LUT experiment, GPU-vs-CPU |
| `tests/nll_prompt.txt` | the frozen prompt every NLL figure is measured on — committed so the numbers stay reproducible |
| `RESEARCH.md` | sources behind these design decisions, each one actually read, with the local re-tests that contradicted some of them |

## Validation

The engine reproduces the previously-validated `../c/dense.c` numbers **exactly**,
once inputs and build flags match:

| model | current | note |
|---|---|---|
| qwen2.5-3b, 912 tok | TF-NLL **3.4068**, ppl 30.168 | was 3.4057 before portable trig |
| llama-3.1 WRN-8B, 912 tok | TF-NLL **3.1651**, ppl 23.690 | was 3.1666 |
| qwen2.5-3b, f32 weights | TF-NLL 2.8717 | the quantization-free reference |

> ⚠️ **Those three rows cannot be reproduced.** The 912-token prompt they were
> measured on lived in `/tmp` and did not survive a reboot. They are kept as a
> record of what was true, not offered as a check you can re-run. Everything
> measured from 2026-08-17 onward uses **`tests/nll_prompt.txt`**, which is
> committed for exactly this reason; on that prompt the int8 baseline is
> **2.7845 nats, ppl 16.192** (qwen2.5-3b, 681 tokens, `--nll1`). A baseline
> whose input is gone is a number, not a baseline.

Both matched `../c/dense.c` exactly before the trig change; they moved by ~0.03%
in opposite directions after it, which is rounding rather than a quality change.

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

## Do these numbers transfer to the production target?

Everything above was measured on the development desktop (Ryzen 9800X3D, RTX
4070). The deployment target is a Legion Go S handheld (Ryzen Z1 Extreme, Radeon
780M). Both hosts were probed, so this is a comparison, not an assumption:

| | desktop | Legion | |
|---|---|---|---|
| cores / threads | 8c / 16t | 8c / 16t | **identical** |
| L2 | 8 MiB | 8 MiB | **identical** |
| microarch | Zen 5 | Zen 4 | AVX-512 VNNI on both |
| **L3** | **96 MiB** | **16 MiB** | **6×** |
| RAM | 31.9 GB DDR5 | 28.5 GB LPDDR5, **unified with the GPU** | |
| GPU | RTX 4070, dedicated GDDR6X | gfx1103, GTT carved from system RAM | **nothing in common** |

**The CPU side transfers better than the L3 gap suggests, and the reason is worth
stating.** A 6× cache difference sounds decisive, and for a microbenchmark that
hammers one matrix it is — the same VNNI kernel measures 0.83× at n=1 streaming
and 1.28× resident. But in *decode* every weight matrix is read once per token,
and between two uses of the same matrix the entire rest of the model streams past:
2.3 GB for qwen-3b at int4, 5.8 GB for the 8B. That evicts everything on both
hosts. **Real decode is in the streaming regime on a 96 MiB L3 and a 16 MiB L3
alike**, which is why the thresholds here were taken at 16384×16384 (256 MiB) —
the only regime both machines actually run in.

Where the gap *would* bite is large-batch prefill and the KV cache, where
activations are reused inside one matrix's lifetime. Those are not measured here.

**The GPU numbers do not transfer at all, and should not be quoted for the
Legion.** `coli_vk_gemm4` at 0.66× and `coli_vk_ffn4` at 1.15–1.51× were measured
on a discrete card with dedicated VRAM reached over PCIe. gfx1103 is an
integrated RDNA3 part whose "VRAM" is a carve-out of the same LPDDR5 the CPU is
using, so the DEVICE_LOCAL staging that made the desktop GPU fast is pure waste
there, and CPU and GPU contend for one memory pool rather than having their own.

### ⚠️ Measured under a Legion-sized L3 — and one result did NOT survive

`resctrl` CAT was mounted and **verified to enforce** before anything was read
from it (48 MiB streaming probe, 300 passes): 96 MiB → **167.9 GB/s**, 18 MiB →
68.3, 12 MiB → 61.7, 6 MiB → 59.9. Falls out of cache exactly where capacity says.
The first attempt at this control showed a *flat* 167 GB/s across every mask —
the task assignment had silently failed because this shell is zsh and `$BASHPID`
expanded to nothing. A flat table reads as a finding and is not one.

**The 16384×16384 thresholds hold, and int4 gets *better* as cache shrinks:**

| int4/int8 | 96 MiB | 18 MiB | 12 MiB |
|---|---|---|---|
| n=1 | 0.62× | 0.66× | 0.64× |
| n=4 | 1.18× | 1.16× | **0.96×** |
| n=32 | 1.14× | 1.04× | 1.04× |

int4's prefill penalty *shrinks* with the cache, which is what the bandwidth
argument predicts: less cache means more DRAM-bound, and the format moving fewer
bytes gains.

**But at the REAL llama-8B FFN shape (4096×14336, 56 MiB) it inverts, and this is
unresolved:**

| n=1 int4/int8 | run 1 | run 2 | run 3 |
|---|---|---|---|
| 96 MiB | 0.48× | 1.26× | 0.82× |
| **12 MiB** | **1.31×** | **3.11×** | **2.60×** |

Consistently *worse* under the small mask, where 16384×16384 was consistently
better. The likely mechanism: int4's win requires actually being DRAM-bound. At
this shape the kernel measures ~19 GB/s — nowhere near the ~70–80 GB/s ceiling —
so nothing is being saved by moving fewer bytes, while the per-row unpack is
still paid. **Open question, not a conclusion**, and it must be settled before
assuming `--w4 2` speeds up decode on the Legion. Memory footprint is unaffected
either way: 5.81 GiB against 8.43 GiB is a property of the format, not the kernel.

**How to check before importing:** `Hardware/scripts/diagnostics/legion-sim.sh`
constrains a run on the desktop to the Legion's measured CPU envelope — L3 via
resctrl CAT, RAM via `MemoryMax`, physical-core pinning — and prints every limit
it could *not* apply, because a silent gap is how a desktop pass gets read as a
Legion pass. Note 16 MiB is not exactly reachable: this CPU's CAT granularity is
6 MiB per way, so the honest form is to bracket with 12 and 18 MiB and report
both.

⚠️ **You cannot validate parity by comparing generated text.** Measured
2026-08-14: the same source built `-march=znver5` and `-march=znver4` produced
only **18 of 64 identical tokens** while teacher-forced NLL agreed to **0.06%**
(3.6242 vs 3.6222). A sub-epsilon ISA difference flips a near-tie argmax and
greedy decoding amplifies it from there. Compare NLL over enough tokens, or
intermediate tensors — never the output string.

## Not built

Stated so this file cannot be mistaken for a finished engine:

- **The GPU backend is CORRECT but not yet FAST, and only the GEMM is on it.**
  See the Vulkan section below.
- ~~Prefill steps one token at a time~~ — **fixed**, see below.
- ~~**The server has ONE slot.**~~ — **fixed.** `go/scheduler.go` is a
  continuous-batching scheduler and the measured table at the top of this file
  shows the server's own histogram running `{4: 20}`. Still missing: **streaming,
  auth, TLS**. (This bullet said "one slot" long after batching landed and while
  the batching numbers sat 500 lines above it — a stale not-built entry reads as
  a limitation that is still being worked around.)
- ~~**`WQ=f32` is not wired**~~ — **it is.** `wq_int8=0` keeps f32 weights
  (`quant_rows`) and `mm()` routes them to `coli_gemm_f32`; `--f32` on the CLI
  reaches it. The f32 reference NLL of 2.8717 in the validation table could only
  have come from that path, so this bullet contradicted a measurement in the same
  file.
- ~~**MoE is tested against a SYNTHETIC model only.**~~ — **fixed.** `qwen3moe`
  loads and runs (Qwen3-30B-A3B, 128 experts, top-8), grouped GEMM is 1.51× on
  real routing, and the greedy output matches llama.cpp character-for-character.
  Still refused: `qwen2moe`, `mixtral`, and every other architecture — the gate
  now allows `qwen2`, `qwen3`, `qwen3moe` and `llama` only.
- `coli_cache_bytes()` is not yet consulted by the dispatch — residency is
  assumed from n, not measured.
- ~~No int4 path in the engine yet~~ — built. `--w4 2` (`coli_open_w4`,
  `coli.OpenW4`, `coli-server -w4 2`) is int4-only weights end to end. Still
  opt-in: it costs +3.87% NLL, which is a deployment call, not an engine default.
- ~~**The Vulkan int4 kernel has never run.**~~ — **it runs.** `vulkan-headers`
  installed, RTX 4070: correct to **rel 9.1e-07** against the CPU int4 kernel with
  the control exceeding tolerance as required, and **0.66× of the int8 GPU kernel
  at n=1**. `tests/test_shader_index.c` (the CPU transliteration that verified the
  addressing while this was blocked) stays — it catches an indexing regression
  without a GPU, on machines that have none.
- **The Vulkan backend is still not wired into the model at all** — it is reached
  only from its test, so `--w4 2` is CPU-only regardless of the above.
- **`token_embd.weight` stays int8 under `--w4 2`** by design — see the note in
  the int4 section. On qwen2.5-3b that is 2048 × 151936 = **297 MiB** of the
  2.29 GiB peak, so a model
  with a large vocabulary saves proportionally less than the 0.62× the weight
  matrices alone would suggest.

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
