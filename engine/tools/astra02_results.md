# Dispatch 2: measured expert residency (2026-09-17)

PEER_REPORT; changes are UNVERIFIED_EXTERNAL_CHANGE until the lead validates.
Engine base: `5ce7ac2`. No inference, residency-policy, or kernel changes.
Evidence directory (R):
`/home/monkey/Documents/Ai/Hardware/diagnostics/host/2026-09-17-desktop-diskio/`.
Full 12-section handoff:
`/home/monkey/Documents/Ai/ModelDev/HANDOFF_astra_2026-09-17_dispatch2.md`.

## Reproduction

`moe_profile.py --layers 36 --experts 128 --output profile.txt train*.txt`
builds a complete hottest-first order from decode rows. `--reverse` ranks
all experts coldest-first, including unobserved experts. `--phase prefill`
or `all` changes the training population explicitly. It never infers model
dimensions from observed IDs, and refuses to overwrite an existing output.
S=1 identifies decode in these runs; one-token prefill is indistinguishable.

`astra02_measure.py train|eval1|eval2|paired` records exact commands and
conditions; it requires a systemd user scope with `MemoryMax=22G` and
`MemorySwapMax=0`. Timed comparisons use three sequential alternating pairs,
discard pair zero, 8 CPU threads, `OMP_WAIT_POLICY=active`, `COLI_EXPERT_GB=12`,
`--backend auto -n 96 --temp 0 -c 512`. Trace collection is separate from timing.
Existing raw files cause a refusal, preventing accidental replacement.
All profiles remain explicit runtime inputs, not installed defaults.

R/astra02_commands.txt and the named `.sh`/`.py` files preserve exact scripts.
R/astra02_build.txt preserves the forced rebuild and binary hashes.

## A: isolated kernel oracle

R/astra02_kernel.sh/.txt: RTX 4070, 451 MiB held by other processes,
load 0.60/0.73/0.80, 22 GiB cap, 8 threads. Actual gpt-oss-120b MXFP4
gate/up/down weights at layer/expert 0/3, 17/67, 35/127; each 2880x2880,
n=1 and n=8. Deterministic synthetic activations, quantized once and shared.

All 18 comparisons pass: max absolute difference divided by max absolute
CPU output is at most **5.5758e-7** (mean normalized absolute error per case
5.0041e-9–4.3687e-8), below the prior int4/coop figure of 3.1e-4. CPU SIMD
matches scalar CPU byte-for-byte in every case. GPU error against a double
arbiter is <=1.718e-7 normalized, CPU <=4.595e-7. Multiplying activation
scales by 1.1 fails every comparison (~0.1 normalized error).

Source supports accumulation-order differences: CPU accumulates each 32-value
block sequentially; `shaders/gemm_i4_dp.comp` under `MXFP4_LUT` sums smaller
chunks and uses a subgroup reduction. The weight LUT and halved scale agree.
This test does not exercise fused SwiGLU, its internal requantization, or
router amplification. No kernel defect or justified repair found in its scope.

The lead's prior io13 NLL gate remains failed: +0.001225637 mean nats,
max token delta 0.935184, 557/680 above 0.005. No-profile is byte-identical
to io07; old broken-store control mean is 12.211498 versus 2.346491.
R/astra02_prior_oracle_audit.txt audits those retained dumps; these were not
rerun. The isolated GEMM pass does not establish end-to-end equivalence.

## B: general profile

Training: networking, photosynthesis, musical tension, orbital mechanics;
96 decoded tokens each, 384 total. Neither EVAL1 (B-tree) nor EVAL2
(sourdough) is training data. Prompts are saved as R/astra02_prompt_*.txt.
Each run reports 638/4608 slots: 18 per layer for layers 0–25, 17 for 26–35.

R/astra02_analyze.txt: on the lead's id-order evaluation traces, predicted
general-profile decode coverage is 43.12%/62.97% for EVAL1/EVAL2; prefill
27.42%/33.73%. Reversed general profile predicts 3.39%/0.90%, versus id-order
16.30%/13.95%. Full frequencies and per-layer coverage are retained there.
EVAL2 exceeds the proposed 33–54% bracket; it is not a constraint on results.
R/astra02_builder_check.txt: ranking/phase/completeness checks pass and seven
malformed or empty trace controls are rejected.

All timed runs: 22 GiB/no-swap cap, 12 GiB store, 638 slots, 451 MiB held
VRAM, same rebuilt binary, 96 generated tokens, pair zero discarded. EVAL1
has 23 prompt tokens; EVAL2 has 21. Load before kept EVAL1 runs 5.58–6.86;
EVAL2 6.52–7.28. No cache drops. R/astra02_eval{1,2}.txt and per-run raws;
R/astra02_summary.txt includes exact sector deltas and polled process reads.

| prompt / arm | decode tok/s | prefill tok/s | GPU hits | NVMe GiB/run | polled VmHWM GiB |
|---|---:|---:|---:|---:|---:|
| EVAL1 id | 7.0 / 7.0 | 4.9 / 4.8 | 16.30% | 71.181 / 72.575 | 15.413 / 15.415 |
| EVAL1 general | 8.8 / 8.8 | 5.2 / 5.2 | 44.30% | 66.819 / 66.899 | 15.574 / 15.542 |
| EVAL2 id | 8.7 / 8.7 | 5.1 / 4.9 | 13.95% | 53.986 / 54.645 | 15.423 / 15.422 |
| EVAL2 general | 10.9 / 10.9 | 5.4 / 5.4 | 63.03% | 53.633 / 53.700 | 15.484 / 15.487 |

Decode increases ~25% on both held-out prompts. Prefill improves ~6–10% in
these short-prompt cells. EVAL1 NVMe reads decrease 6.1–7.8%; EVAL2 only
0.7–1.7%, too small to claim a robust I/O benefit from two pairs. EVAL1's
no-profile decode still matches io09's 7.0; its prefill is 4.8–4.9 versus
io13 EVAL1's 5.2–5.3, a condition-specific regression worth retaining.
The general profile is slower on EVAL1 than the lead's domain-specific
9.2–9.3, but reaches higher EVAL2 coverage than the 33.1% off-domain prediction.

Prediction versus measurement differences are not proof of slot churn:
all timed Banana raws report zero fetches and evictions. Changed arithmetic
can change routing and generated continuations. No profile-arm routing trace
was collected during timing, so the exact source of each difference remains
unisolated.

Runtime reversed-general control, EVAL1, same cap/store/VRAM: **3.284% hits,
6.2 decode tok/s, 5.0 prefill tok/s, 85.637 GiB NVMe, 15.552 GiB polled
VmHWM**, load 5.44→6.36. It loses to id-order as required. R/astra02_reversed.txt
and astra02_eval1_reversed_control_raw.txt.

## D: CPU cache candidate (description only)

R/astra02_cpu_candidates.txt is an offline membership calculation, **not an
LRU replay or a RAM-cache benefit measurement**. After excluding GPU-pinned
experts, ranking remaining experts globally by TRAIN frequency and protecting
486 triples consumes at most 6,431,809,536 bytes (conservative aligned-superset
bound), leaving at least half of the 12 GiB store available for dynamic fills.
Those candidates cover 37.06%/43.90% of residual CPU selections on the old
EVAL1/EVAL2 traces; a coldest-residual control covers 3.38%/1.48%.

This justifies a controlled future experiment, not implementing protection
now. Existing LRU may already retain those experts. Do not protect the same
leading entries blindly: they already reside on the GPU. Prefill has different
demand and currently prefetches selected experts regardless of static GPU
residency; pinning must preserve space for those fills. Use physical bytes,
prefetch fills, and token time to judge a change: request hit rate includes
reads satisfied by prefetch and is not a disk-avoidance metric.
