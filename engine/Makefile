# Colibri engine — three layers, split by what each language is actually for.
#
#   C++  (src/*.cpp)   kernels, model, forward pass, batched decode
#   Go   (go/)         HTTP, continuous-batching scheduler, cross-compilation
#   Py   (python/)     fan-out, routing, orchestration -- NOT in the token path
#
# -ffp-contract=off is NOT optional: see the comment at the top of gemm_i8.cpp.
# Without it the engine's own reference test disagrees with its kernels on 76%
# of cells and every bit-exactness claim becomes unfalsifiable.

CXX     ?= c++
CC      ?= cc
CXXFLAGS?= -std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter
CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter
COMMON   = -ffp-contract=off -D_FILE_OFFSET_BITS=64 -I../c -Isrc
ARCH    ?= native
COMMON  += -march=$(ARCH)
OMP     ?= -fopenmp
LDLIBS   = -lm

# arch.cpp / arch_ops.cpp / gemm_mxfp4.cpp joined the core 2026-09-14: model.cpp now
# runs gpt-oss (arch descriptor, YaRN table, native MXFP4 experts). Previously test-only.
CPP_SRC  = src/coli_api.cpp src/model.cpp src/gemm_i8.cpp src/gemm_q4k.cpp src/expert_store.cpp src/cpu_features.cpp src/sample.cpp src/spec.cpp src/arch.cpp src/gpu_keepalive.cpp src/backend_bench.cpp src/arch_ops.cpp src/gemm_mxfp4.cpp src/backend.c
C_SRC    = src/loader.c src/q4k_shim.c  # stay C: wrap ../c unchanged, see loader.h / q4k_shim.h

.PHONY: all lib cli cli-gpu go python test vk windows clean gpu-if-built
# `all` deliberately does NOT list coli-gpu: a fresh clone must build without a
# Vulkan SDK. But leaving it out entirely made plain `make` a silent no-op for
# the GPU binary -- measured 2026-08-24: a whole interleaved A/B ran on the
# previous day's 20:52 coli-gpu and only the absence of a new control string in
# `strings coli-gpu` caught it. Fifth defect of the same family as the four
# recorded above the coli-gpu rule.
#
# gpu-if-built rebuilds coli-gpu ONLY when one is already on disk, which is
# exactly the stale case. No coli-gpu, no Vulkan needed, no behaviour change on
# a fresh clone -- and the control can fail: `rm coli-gpu && make` must not
# invoke glslc.
all: lib cli go gpu-if-built

gpu-if-built:
	@if [ -f coli-gpu ]; then $(MAKE) --no-print-directory coli-gpu; fi

# ---- Banana ---------------------------------------------------------------
# Owner-named 2026-08-19. Banana is this homelab's build of the engine; the
# upstream lineage is Colibri. These targets produce the SAME binaries under the
# preferred name -- hard links, not copies, so they cannot drift apart and a
# rebuild of one is a rebuild of both.
#
# The C ABI is still coli_* on purpose (111 symbols / 629 occurrences in src/,
# plus the Go and Python bindings). The tooling takes the name first; the ABI
# rename is a separate change with its own risk, deliberately not bundled with
# a release or a benchmark.
banana: coli coli-gpu
	ln -f coli banana
	ln -f coli-gpu banana-gpu
	@echo "banana + banana-gpu -> same inodes as coli + coli-gpu"

banana-cpu: coli
	ln -f coli banana


# ---- C++ core -------------------------------------------------------------
lib: libcoli.so
libcoli.so: $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -fPIC -c src/loader.c -o /tmp/coli_loader.o
	$(CC)  $(CFLAGS)   $(COMMON) -fPIC -c src/q4k_shim.c -o /tmp/coli_loader_q4k.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -fPIC -shared -o $@ $(CPP_SRC) /tmp/coli_loader.o /tmp/coli_loader_q4k.o $(LDLIBS)

cli: coli
coli: src/main.cpp $(CPP_SRC) $(C_SRC) src/hw_detect.c src/hw_detect.h src/backend.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_loader_s.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_loader_s_q4k.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/hw_detect.c -o /tmp/coli_hw_s.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -o $@ src/main.cpp $(CPP_SRC) /tmp/coli_loader_s.o /tmp/coli_loader_s_q4k.o /tmp/coli_hw_s.o $(LDLIBS)

# ---- Go serving layer -----------------------------------------------------
go: libcoli.so
	cd go && CGO_ENABLED=1 go build -o coli-server ./cmd/coli-server
	@echo "run: LD_LIBRARY_PATH=$(PWD) go/coli-server -m model.gguf"

# ---- Python scale-out -----------------------------------------------------
python:
	cd python && python3 -c "import sys;sys.path.insert(0,'.');import colibri;print('colibri',colibri.__version__)"

# ---- tests ----------------------------------------------------------------
tests/test_gemm_i8: tests/test_gemm_i8.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ $^ -x none -o $@ $(LDLIBS)
tests/test_gemm_i8_broken: tests/test_gemm_i8.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_WIDE -x c++ $^ -x none -o $@ $(LDLIBS)
tests/test_gemm_i4: tests/test_gemm_i4.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ $^ -x none -o $@ $(LDLIBS)
tests/test_gemm_i4_broken: tests/test_gemm_i4.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_I4 -x c++ $^ -x none -o $@ $(LDLIBS)
# Negative control for the packed-panel int4 kernel (run_panel_check() in
# test_gemm_i4.c): perturbs ONLY the panel kernel's correction term, so the
# panel-vs-COLI_I4_TILE=0 differential in that same binary must FAIL.
tests/test_gemm_i4_panel_broken: tests/test_gemm_i4.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_I4_PANEL -x c++ $^ -x none -o $@ $(LDLIBS)
tests/bench_gemm_tile: tests/bench_gemm_tile.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ $^ -x none -o $@ $(LDLIBS)
tests/test_shader_index: tests/test_shader_index.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ $^ -x none -o $@ $(LDLIBS)
# Native Q4_K kernel (phase1/native-q4k): correctness (kernel vs its own
# scalar ref vs an independent dequant->f32 dot) plus a perturbation control
# that must FAIL, same discipline as test_gemm_i4's COLI_BREAK_I4.
tests/test_gemm_q4k: tests/test_gemm_q4k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp src/q4k_shim.c src/q4k_shim.h src/gemm_q4k.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_q4kshim_test.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_gemm_q4k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_q4kshim_test.o -o $@ $(LDLIBS)
tests/test_gemm_q4k_broken: tests/test_gemm_q4k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp src/q4k_shim.c src/q4k_shim.h src/gemm_q4k.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_q4kshim_testb.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_Q4K -x c++ tests/test_gemm_q4k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_q4kshim_testb.o -o $@ $(LDLIBS)
# Native Q6_K kernel (2026-09-13): real Q6_K expert bytes from the 30B/235B files,
# decode vs gguf_dequant_q6_K (exact, with a bit-flip control), kernel vs its scalar
# reference, kernel vs dequantized truth. _broken uses the wrong sub-block scale in
# the AVX2 kernel only and must FAIL. Exit 77 = SKIP (no model on disk).
tests/test_gemm_q6k: tests/test_gemm_q6k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp src/q4k_shim.c src/q4k_shim.h src/gemm_q4k.h src/loader.c src/loader.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_q6kshim_test.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_q6kloader_test.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_gemm_q6k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_q6kshim_test.o /tmp/coli_q6kloader_test.o -o $@ $(LDLIBS)
tests/test_gemm_q6k_broken: tests/test_gemm_q6k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp src/q4k_shim.c src/q4k_shim.h src/gemm_q4k.h src/loader.c src/loader.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_q6kshim_testb.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_q6kloader_testb.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_Q6K -x c++ tests/test_gemm_q6k.c src/gemm_q4k.cpp src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_q6kshim_testb.o /tmp/coli_q6kloader_testb.o -o $@ $(LDLIBS)
# The OOM path in cache_ids_grow is unreachable under Linux overcommit, so the
# test build forces it (COLI_BREAK_CACHE_GROW) and the CONTROL build additionally
# restores the pre-fix guard (COLI_CACHE_UNSAFE_GUARD). Both carry ASan: the bug
# is a heap-buffer-overflow, and without a sanitizer the control would very
# likely "pass" by overwriting something harmless and never crashing.
# Standalone: no model, no shaders beyond the one used to init the device.
# Previously built by hand, so `make` could not reproduce a published number.
# Same omission as test_vk_floor: no rule existed, so `make` fell through to the
# builtin C rule and the link failed. Every GPU number from this harness was
# produced by a hand-typed command line nobody recorded.
tests/test_vk_attn: tests/test_vk_attn.c src/vk_backend.c shaders/attn_decode.spv
	$(CC) $(CFLAGS) $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_attn.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_vk_attn.c \
	  -x none /tmp/coli_vkb_attn.o -o $@ $(LDLIBS) -lvulkan

# split-K decode attention (2026-09-13): compares attn_decode_split.comp +
# attn_decode_merge.comp against the same two-pass CPU reference test_vk_attn
# uses, plus a perturbation control and an old-vs-new dispatch timing loop.
tests/test_attn_split: tests/test_attn_split.c src/vk_backend.c shaders/attn_decode.spv shaders/attn_decode_split.spv shaders/attn_decode_merge.spv
	$(CC) $(CFLAGS) $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_attn_split.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_attn_split.c \
	  -x none /tmp/coli_vkb_attn_split.o -o $@ $(LDLIBS) -lvulkan

tests/test_vk_rope: tests/test_vk_rope.c src/vk_backend.c shaders/rope_bias.spv shaders/kvwrite.spv shaders/attn_decode.spv
	$(CC) $(CFLAGS) $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_rope.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_vk_rope.c \
	  -x none /tmp/coli_vkb_rope.o -o $@ $(LDLIBS) -lvulkan

tests/test_slot_cache: tests/test_slot_cache.c src/moe_slot_cache.h
	$(CXX) $(CXXFLAGS) $(COMMON) -x c++ tests/test_slot_cache.c -o $@

tests/test_vk_oai: tests/test_vk_oai.c src/vk_backend.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp shaders/gemm_i4_mx_dp.spv shaders/swiglu_oai_q.spv shaders/attn_decode.spv
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_oai.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_vk_oai.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp -x none /tmp/coli_vkb_oai.o -o $@ $(LDLIBS) -lvulkan

tests/test_vk_gemm: tests/test_vk_gemm.c src/vk_backend.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_gemm.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_vk_gemm.c src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_vkb_gemm.o -o $@ $(LDLIBS) -lvulkan

tests/bench_gemv: tests/bench_gemv.c src/vk_backend.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_gemm.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/bench_gemv.c src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_vkb_gemm.o -o $@ $(LDLIBS) -lvulkan

tests/gpu_keepalive: tests/gpu_keepalive.c src/vk_backend.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_gemm.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/gpu_keepalive.c src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_vkb_gemm.o -o $@ $(LDLIBS) -lvulkan

tests/test_moe4: tests/test_moe4.c src/vk_backend.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_gemm.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_moe4.c src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_vkb_gemm.o -o $@ $(LDLIBS) -lvulkan

tests/test_vk_fill: tests/test_vk_fill.c src/vk_backend.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp shaders/gemm_i4_mx_dp.spv
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_fill.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_vk_fill.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp -x none /tmp/coli_vkb_fill.o -o $@ $(LDLIBS) -lvulkan

tests/test_vk_floor: tests/test_vk_floor.c src/vk_backend.c
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_floor.o
	$(CXX) $(CXXFLAGS) $(COMMON) -DCOLI_HAVE_VK -x c++ tests/test_vk_floor.c -x none /tmp/coli_vkb_floor.o -o $@ $(LDLIBS) -lvulkan

tests/test_cache_oom: tests/test_cache_oom.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -c src/loader.c -o /tmp/coli_loader_oom.o
	$(CC)  $(CFLAGS)   $(COMMON) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -c src/q4k_shim.c -o /tmp/coli_loader_oom_q4k.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -x c++ tests/test_cache_oom.c $(CPP_SRC) -x none /tmp/coli_loader_oom.o /tmp/coli_loader_oom_q4k.o -o $@ $(LDLIBS)
tests/test_cache_oom_unsafe: tests/test_cache_oom.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -DCOLI_CACHE_UNSAFE_GUARD -c src/loader.c -o /tmp/coli_loader_oomu.o
	$(CC)  $(CFLAGS)   $(COMMON) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -DCOLI_CACHE_UNSAFE_GUARD -c src/q4k_shim.c -o /tmp/coli_loader_oomu_q4k.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -DCOLI_CACHE_UNSAFE_GUARD -x c++ tests/test_cache_oom.c $(CPP_SRC) -x none /tmp/coli_loader_oomu.o /tmp/coli_loader_oomu_q4k.o -o $@ $(LDLIBS)

tests/test_batch: tests/test_batch.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_loader_t.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_loader_t_q4k.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_batch.c $(CPP_SRC) -x none /tmp/coli_loader_t.o /tmp/coli_loader_t_q4k.o -o $@ $(LDLIBS)

# GGUF MXFP4 (ttype 39) / Q8_0 (ttype 8) oracle -- see test_mxfp4.c's header.
# The shim is built with the PLAIN C compiler, never -x c++: it is what lets
# the test #include gguf_reader.h/gemm_i8.h/gemm_mxfp4.h under `-x c++` (for
# consistent name mangling against the C++-compiled kernels, same reason
# test_gemm_i8 uses -x c++) WITHOUT also dragging c/ggml_dequant.h's C-only
# st.h/json.h includes into a C++ translation unit, where they fail to build.
# Needs a real gpt-oss-120b-MXFP4.gguf on disk; exits 77 (SKIP) if absent.
tests/mxfp4_dequant_shim.o: tests/mxfp4_dequant_shim.c ../c/ggml_dequant.h
	$(CC) $(CFLAGS) -D_FILE_OFFSET_BITS=64 -I../c -c tests/mxfp4_dequant_shim.c -o $@
tests/test_mxfp4: tests/test_mxfp4.c src/gemm_mxfp4.cpp src/gemm_i8.cpp src/cpu_features.cpp tests/mxfp4_dequant_shim.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_mxfp4.c src/gemm_mxfp4.cpp src/gemm_i8.cpp src/cpu_features.cpp -x none tests/mxfp4_dequant_shim.o -o $@ $(LDLIBS)
# Same test with the SIMD MXFP4 kernel's LUT perturbed (-DCOLI_BREAK_MXFP4): MUST exit nonzero.
tests/test_mxfp4_broken: tests/test_mxfp4.c src/gemm_mxfp4.cpp src/gemm_i8.cpp src/cpu_features.cpp tests/mxfp4_dequant_shim.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_MXFP4 -x c++ tests/test_mxfp4.c src/gemm_mxfp4.cpp src/gemm_i8.cpp src/cpu_features.cpp -x none tests/mxfp4_dequant_shim.o -o $@ $(LDLIBS)

# arch.h/arch.cpp: GGUF-metadata-only architecture descriptor (phase 2,
# "any model" program). Reads real headers only -- coli_gguf_open() never
# touches tensor data, so this is safe against the 63 GB gpt-oss-120b file.
# Needs both qwen2.5-3b and gpt-oss-120b-MXFP4.gguf on disk at the paths
# hardcoded in test_arch.cpp; there is no SKIP path because both are expected
# to be present on this host (see the task this was written under).
tests/test_arch: tests/test_arch.cpp src/arch.cpp src/arch.h src/loader.c src/loader.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_arch_loader.o
	$(CXX) $(CXXFLAGS) $(COMMON) -x c++ tests/test_arch.cpp src/arch.cpp -x none /tmp/coli_arch_loader.o -o $@ $(LDLIBS)

# arch_ops.h/.cpp: CPU primitives gpt-oss needs and qwen/llama don't (YaRN
# rope table, swiglu_oai, biased top-k-softmax router gating, attention-sink
# softmax, sliding-window mask). Standalone -- no GGUF, no model, self-
# contained oracles + perturbation controls, see the file header.
tests/test_arch_ops: tests/test_arch_ops.cpp src/arch_ops.cpp src/arch_ops.h
	$(CXX) $(CXXFLAGS) $(COMMON) -x c++ tests/test_arch_ops.cpp src/arch_ops.cpp -o $@ $(LDLIBS)

# gpt-oss LAYER 0 dense-attention-only probe (phase 2 step 3). Loads ONLY
# blk.0's dense tensors + token_embd (~2.2 GiB dequanted, freed on exit) from
# the real 63 GB gpt-oss-120b-MXFP4.gguf -- never an expert tensor, never more
# than one layer. Oracle is tests/gptoss_layer0_ref.py, an independent
# stdlib-only Python re-implementation; `make test-gptoss-layer0` runs both
# and diffs them, then rebuilds+reruns a COLI_L0_BREAK_ROPE control build and
# asserts THAT fails the same diff (a comparison that cannot fail proves
# nothing). Needs gpt-oss-120b-MXFP4.gguf on disk; no SKIP path, see test_arch.
tests/test_gptoss_layer0: tests/test_gptoss_layer0.cpp src/arch.cpp src/arch_ops.cpp src/loader.c
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_l0_loader.o
	$(CXX) $(CXXFLAGS) $(COMMON) -x c++ tests/test_gptoss_layer0.cpp src/arch.cpp src/arch_ops.cpp -x none /tmp/coli_l0_loader.o -o $@ $(LDLIBS)
tests/test_gptoss_layer0_broken: tests/test_gptoss_layer0.cpp src/arch.cpp src/arch_ops.cpp src/loader.c
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_l0_loader_b.o
	$(CXX) $(CXXFLAGS) $(COMMON) -DCOLI_L0_BREAK_ROPE -x c++ tests/test_gptoss_layer0.cpp src/arch.cpp src/arch_ops.cpp -x none /tmp/coli_l0_loader_b.o -o $@ $(LDLIBS)
test-gptoss-layer0: tests/test_gptoss_layer0 tests/test_gptoss_layer0_broken
	./tests/test_gptoss_layer0
	python3 tests/gptoss_layer0_ref.py
	python3 tests/compare_layer0.py
	@echo "--- negative control (must FAIL) ---"
	./tests/test_gptoss_layer0_broken
	@if python3 tests/compare_layer0.py >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the comparison proves nothing"; exit 1; \
	 else echo "control correctly failed (RoPE-disabled build disagrees with the oracle)"; fi
	@echo "--- restoring the real (non-broken) output ---"
	./tests/test_gptoss_layer0
	python3 tests/compare_layer0.py

# COLI_EXPERT_STORE (phase1c/expert-store): slice reader + LRU cache vs the
# pre-existing whole-tensor loader, on REAL data. Needs coli_gguf_slice's fd
# (loader.c) and the ColiEstore vtable (expert_store.cpp); no broken/negative
# BUILD variant like test_gemm_q4k's -DCOLI_BREAK_Q4K -- its negative control
# is a runtime byte perturbation inside the same binary (see the file's own
# header), not a compile-time one.
tests/test_expert_store: tests/test_expert_store.c src/loader.c src/expert_store.cpp src/loader.h src/expert_store.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_loader_estest.o
	$(CXX) $(CXXFLAGS) $(COMMON) -x c++ tests/test_expert_store.c src/expert_store.cpp -x none /tmp/coli_loader_estest.o -o $@ $(LDLIBS)

test: tests/test_gemm_i8 tests/test_gemm_i8_broken tests/test_gemm_i4 tests/test_gemm_i4_broken tests/test_gemm_i4_panel_broken tests/test_shader_index tests/test_gemm_q4k tests/test_gemm_q4k_broken tests/test_gemm_q6k tests/test_gemm_q6k_broken tests/test_expert_store
	@./tests/test_gemm_i8 2048 2048
	@echo "--- negative control (must FAIL) ---"
	@if ./tests/test_gemm_i8_broken 2048 2048 >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the test proves nothing"; exit 1; \
	 else echo "control failed as required"; fi
	@# 16384 on purpose: 256 MiB of int8 weights against 96 MiB of L3, so the
	@# matrix STREAMS. At 2048 it is L3-resident and measures the wrong regime.
	@./tests/test_gemm_i4 16384 16384
	@echo "--- int4 negative control (must FAIL) ---"
	@if ./tests/test_gemm_i4_broken 4096 4096 >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the test proves nothing"; exit 1; \
	 else echo "control failed as required"; fi
	@echo "--- int4 packed-panel negative control (must FAIL) ---"
	@if ./tests/test_gemm_i4_panel_broken 2048 2048 >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the test proves nothing"; exit 1; \
	 else echo "control failed as required"; fi
	@echo "--- gemm_i4.comp addressing (CPU-side; NOT a GPU test) ---"
	@./tests/test_shader_index 512 256
	@echo "--- native Q4_K: correctness + expert shapes ---"
	@./tests/test_gemm_q4k 2048 768
	@./tests/test_gemm_q4k 768 2048
	@echo "--- native Q4_K negative control (must FAIL) ---"
	@if ./tests/test_gemm_q4k_broken 2048 768 >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the test proves nothing"; exit 1; \
	 else echo "control failed as required"; fi
	@echo "--- native Q6_K on real expert bytes ---"
	@./tests/test_gemm_q6k; rc=$$?; \
	 if [ $$rc -eq 77 ]; then echo "SKIPPED (no Q6_K expert model on this machine)"; \
	 elif [ $$rc -ne 0 ]; then echo "test_gemm_q6k FAILED"; exit 1; fi
	@echo "--- native Q6_K negative control (must FAIL) ---"
	@./tests/test_gemm_q6k_broken >/dev/null 2>&1; rc=$$?; \
	 if [ $$rc -eq 0 ]; then echo "CONTROL DID NOT FAIL -- the test proves nothing"; exit 1; \
	 elif [ $$rc -eq 77 ]; then echo "SKIPPED"; else echo "control failed as required"; fi
	@echo "--- COLI_EXPERT_STORE: slice reader + LRU cache vs whole-tensor reference ---"
	@./tests/test_expert_store; rc=$$?; \
	 if [ $$rc -eq 77 ]; then echo "SKIPPED (no model on this machine)"; \
	 elif [ $$rc -ne 0 ]; then echo "test_expert_store FAILED"; exit 1; \
	 else echo "test_expert_store PASS (including inline negative control)"; fi

# ---- optional Vulkan ------------------------------------------------------
VK_INC ?=
shaders/%.spv: shaders/%.comp
	glslc -O --target-env=vulkan1.1 $< -o $@

# The SAME source with the direct-store epilogue enabled. Explicit rule, so it
# wins over the pattern above (there is no gemm_i4_coop_ds.comp and there must
# not be: two copies of a 180-line kernel is how the variants drift apart).
# Both .spv ship, and COLI_VK_COOP_DS picks between them at runtime -- a
# rebuild-to-switch A/B compares two binaries, not two kernels.
# Short-row int4 GEMV (LANES=8, OUTS=8): measured 2026-09-08 on the 4070, batch 1, 768-wide rows
# (expert down_proj): 2.98 us vs 5.2-6.3 us for the default 16x4 layout; long rows prefer 16x4.
# record_gemm picks this pipeline when I <= 1024 (COLI_VK_NO_SHORT=1 disables it, the control).
shaders/gemm_i4_short.spv: shaders/gemm_i4.comp
	glslc -O --target-env=vulkan1.1 -DLANES=8 -DOUTS=8 $< -o $@
# Multi-matrix int4 GEMV (2026-09-08): one dispatch over N same-shape matrices addressed through a
# device-address table (binding 4). DP4a source only; _short = the 8x8 geometry for I <= 1024.
# coli_vk_moe4 uses it when COLI_MOE_FUSED=1; bit-identical to the per-matrix path (tests/test_moe4).
shaders/gemm_i4_moe_dp.spv: shaders/gemm_i4_moe.comp
	glslc -O --target-env=vulkan1.1 $< -o $@
shaders/gemm_i4_moe_short_dp.spv: shaders/gemm_i4_moe.comp
	glslc -O --target-env=vulkan1.1 -DLANES=8 -DOUTS=8 $< -o $@
shaders/gemm_i4_short_dp.spv: shaders/gemm_i4_dp.comp
	glslc -O --target-env=vulkan1.1 -DLANES=8 -DOUTS=8 $< -o $@

shaders/gemm_i4_coop_ds.spv: shaders/gemm_i4_coop.comp
	glslc -O --target-env=vulkan1.1 -DCOOP_DIRECT=1 $< -o $@
# gpt-oss on the GPU (2026-09-14). gemm_i4_mx*_dp: the DP4a int4 GEMV with the nibble
# decoded through the MXFP4 e2m1 table (same buffers, same activation side; see
# gemm_i4_dp.comp MXFP4_LUT). swiglu_oai_q: silu_mul_q with gate/up biases and
# ggml's swiglu_oai. All from the SAME sources as their int4/SiLU siblings.
shaders/gemm_i4_mx_dp.spv: shaders/gemm_i4_dp.comp
	glslc -O --target-env=vulkan1.1 -DMXFP4_LUT=1 $< -o $@
shaders/gemm_i4_mx_short_dp.spv: shaders/gemm_i4_dp.comp
	glslc -O --target-env=vulkan1.1 -DMXFP4_LUT=1 -DLANES=8 -DOUTS=8 $< -o $@
shaders/swiglu_oai_q.spv: shaders/silu_mul_q.comp
	glslc -O --target-env=vulkan1.1 -DSWIGLU_OAI=1 $< -o $@
# vk_backend.c stays C: Vulkan's designated initialisers and void* conversions
# are C idioms, and vk_backend.h carries extern "C" so C++ links to it.
vk: shaders/gemm_i8.spv shaders/gemm_i4.spv shaders/gemm_i4_dp.spv shaders/gemm_i4f.spv shaders/silu_mul_q.spv shaders/attn_decode.spv shaders/attn_decode_split.spv shaders/attn_decode_merge.spv shaders/rope_bias.spv shaders/kvwrite.spv shaders/quant.spv shaders/qknorm.spv shaders/gemm_i4_tile.spv shaders/gemm_i4_coop.spv shaders/gemm_i4_coop_ds.spv shaders/gemm_i4_mx_dp.spv shaders/gemm_i4_mx_short_dp.spv shaders/swiglu_oai_q.spv
	$(CC) $(CFLAGS) $(COMMON) $(if $(VK_INC),-I$(VK_INC),) -c src/vk_backend.c -o /tmp/coli_vkb.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) $(if $(VK_INC),-I$(VK_INC),) -x c++ \
	  tests/test_vk_gemm.c src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_vkb.o \
	  -o tests/test_vk_gemm $(LDLIBS) -lvulkan
	./tests/test_vk_gemm 4096 16384

# A CLI linked against Vulkan, so `--gpu` works. Kept separate from `cli` because
# the default build must not require a Vulkan SDK to compile.
# The produced file is coli-gpu, so THAT must be the target name: `banana`
# depends on coli-gpu, and while this rule was called cli-gpu make had no
# rule for it -- `make banana` exited 2 and only looked like it worked
# because stale banana/banana-gpu files already existed. Measured 2026-08-20.
cli-gpu: coli-gpu
# The C/C++ sources MUST be prerequisites. With only the shaders listed, an
# edit to model.cpp or vk_backend.c did not rebuild coli-gpu at all and the
# old binary was silently re-measured -- caught 2026-08-20 when an added
# printf never appeared in the output. Third defect of this family here,
# after `make banana` having no rule and `make clean` missing coli-gpu.
# gemm_i4f.spv added 2026-08-21: editing gemm_i4f.comp did NOT rebuild it, so a
# shader whose scale format had changed stayed on disk in the old format while
# every other shader moved. Only tests/test_vk_gemm loads it, which is exactly
# why it went unnoticed -- the stale file would have failed that test with a
# numeric mismatch and no clue as to why. Fourth defect of this family.
coli-gpu: shaders/gemm_i8.spv shaders/gemm_i4.spv shaders/gemm_i4_dp.spv shaders/gemm_i4f.spv \
          shaders/silu_mul_q.spv shaders/attn_decode.spv shaders/attn_decode_split.spv shaders/attn_decode_merge.spv shaders/rope_bias.spv shaders/kvwrite.spv shaders/quant.spv shaders/qknorm.spv shaders/gemm_i4_tile.spv shaders/gemm_i4_coop.spv shaders/gemm_i4_coop_ds.spv shaders/gemm_i4_short.spv shaders/gemm_i4_short_dp.spv shaders/gemm_i4_moe_dp.spv shaders/gemm_i4_moe_short_dp.spv \
          shaders/gemm_i4_mx_dp.spv shaders/gemm_i4_mx_short_dp.spv shaders/swiglu_oai_q.spv \
          src/main.cpp $(CPP_SRC) $(C_SRC) src/vk_backend.c src/backend_vk.c src/backend.h src/hw_detect.c src/hw_detect.h
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_loader_g.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_loader_g_q4k.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_g.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/backend_vk.c -o /tmp/coli_bevk_g.o
	$(CC)  $(CFLAGS)   $(COMMON) -DCOLI_HAVE_VK -c src/hw_detect.c -o /tmp/coli_hw_g.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -o coli-gpu src/main.cpp $(CPP_SRC) \
	  /tmp/coli_loader_g.o /tmp/coli_loader_g_q4k.o /tmp/coli_vkb_g.o /tmp/coli_bevk_g.o /tmp/coli_hw_g.o $(LDLIBS) -lvulkan

# ---- optional libtorch plugin (coverage backend, 2026-09-14) --------------
# See src/backend_torch.cpp's header for WHY this backend exists and what it
# trades away (dequantize-to-bf16, no integer kernel). NEVER linked into coli
# or coli-gpu -- dlopen'ed at runtime by src/backend.c's open_torch(), so a
# machine with no libtorch installed builds and runs every other target
# unchanged. TORCH_DIR defaults to whatever `python3 -c 'import torch'`
# resolves on THIS machine; override it for a different install.
TORCH_DIR ?= $(shell python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)
TORCH_INC  = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
TORCH_LIB  = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10
# libtorch_cuda.so present -> this install has a CUDA build of torch. Only
# then do we link -ltorch_cuda/-lc10_cuda AND reach for the CUDA toolkit's own
# headers (cuda_runtime_api.h) that ATen/cuda/CUDAContext.h needs for the
# device-name query -- measured on this machine: torch's own include/ tree
# ships ATen's declarations but not the CUDA SDK headers underneath them.
# CUDA_INC is therefore only ever added alongside -ltorch_cuda, never without it.
HAVE_TORCH_CUDA := $(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so)
ifneq ($(HAVE_TORCH_CUDA),)
TORCH_LIB += -ltorch_cuda -lc10_cuda
CUDA_INC  ?= $(firstword $(wildcard /opt/cuda/targets/x86_64-linux/include /usr/local/cuda/include /opt/cuda/include))
ifneq ($(CUDA_INC),)
TORCH_INC += -I$(CUDA_INC) -DCOLI_TORCH_HAVE_CUDA_PROPS
endif
endif

libcoli_torch.so: src/backend_torch.cpp src/backend.h
	$(CXX) -std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
	  $(TORCH_INC) -Isrc -I../c -o $@ src/backend_torch.cpp \
	  $(TORCH_LIB) -Wl,-rpath,$(TORCH_DIR)/lib

# dlopens libcoli_torch.so exactly the way the engine does (coli_backend_open,
# COLI_PLUGIN_DIR / next-to-binary / default path) -- NEVER links libtorch or
# backend_torch.cpp itself. See the file header for the five things checked.
tests/test_torch_backend: tests/test_torch_backend.cpp src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp src/backend.c src/backend.h libcoli_torch.so
	$(CC) $(CFLAGS) $(COMMON) -c src/backend.c -o /tmp/coli_backend_torchtest.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_torch_backend.cpp src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp -x none /tmp/coli_backend_torchtest.o -o $@ $(LDLIBS) -ldl

# ---- Windows cross-build --------------------------------------------------
# -static is REQUIRED: a dynamic build starts and silently produces nothing when
# libgomp-1.dll / libwinpthread-1.dll are missing.
WINCXX ?= x86_64-w64-mingw32-g++
WINCC  ?= x86_64-w64-mingw32-gcc
windows:
	$(WINCC)  -O2 -march=x86-64-v3 -ffp-contract=off -D_FILE_OFFSET_BITS=64 -I../c -Isrc -c src/loader.c -o /tmp/coli_loader_w.o
	$(WINCC)  -O2 -march=x86-64-v3 -ffp-contract=off -D_FILE_OFFSET_BITS=64 -I../c -Isrc -c src/q4k_shim.c -o /tmp/coli_loader_w_q4k.o
	$(WINCXX) -std=c++20 -O2 -march=x86-64-v3 -fopenmp -ffp-contract=off -D_FILE_OFFSET_BITS=64 \
	  -I../c -Isrc -static -o coli.exe src/main.cpp $(CPP_SRC) /tmp/coli_loader_w.o /tmp/coli_loader_w_q4k.o -lm
	@file coli.exe

clean:
	rm -f coli coli-gpu banana banana-gpu coli.exe libcoli.so go/coli-server shaders/*.spv \
	      tests/test_gemm_i8 tests/test_gemm_i8_broken tests/test_batch tests/test_vk_gemm

# The prefix cache cannot be tested from the CLI: one prompt per process means
# the cache is always cold and the reuse path never executes. This is the only
# thing that reaches it.
# The slot path (coli_prefill_slot) with GPU attention. Needs the VK build --
# a CPU-only build compiles the arm away and the test would pass having never
# reached it, which is the failure it exists to prevent.
tests/test_prefill_slot_gpu: tests/test_prefill_slot_gpu.c $(CPP_SRC) $(C_SRC) src/vk_backend.c shaders/attn_decode.spv
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_loader_psg.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_loader_psg_q4k.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_psg.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -x c++ tests/test_prefill_slot_gpu.c $(CPP_SRC) \
	  -x none /tmp/coli_loader_psg.o /tmp/coli_loader_psg_q4k.o /tmp/coli_vkb_psg.o -o $@ $(LDLIBS) -lvulkan

tests/test_prefix_cache: tests/test_prefix_cache.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_loader_pc.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_loader_pc_q4k.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_prefix_cache.c $(CPP_SRC) -x none /tmp/coli_loader_pc.o /tmp/coli_loader_pc_q4k.o -o $@ $(LDLIBS)

# Negative control: COLI_BREAK_PREFIX reuses one token more than was matched, so
# the reused KV no longer describes the prompt. `make check` requires this build
# to FAIL -- if it passes, the bit-identity assertion is not testing what it says.
tests/test_prefix_cache_broken: tests/test_prefix_cache.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -DCOLI_BREAK_PREFIX -c src/loader.c -o /tmp/coli_loader_pcb.o
	$(CC)  $(CFLAGS)   $(COMMON) -DCOLI_BREAK_PREFIX -c src/q4k_shim.c -o /tmp/coli_loader_pcb_q4k.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_PREFIX -x c++ tests/test_prefix_cache.c $(CPP_SRC) -x none /tmp/coli_loader_pcb.o /tmp/coli_loader_pcb_q4k.o -o $@ $(LDLIBS)

# ---- model-dependent checks ------------------------------------------------
# Separate from `test` because these need real weights. `test` must stay
# runnable on a machine with no model on disk.
#   usage: make check MODEL=/path/to/model.gguf
# ASAN_OPTIONS=detect_leaks=0 throughout: the gguf index is deliberately not
# freed at exit, and that pre-existing leak would otherwise drown the signal
# these tests exist to produce.
check: tests/test_prefix_cache tests/test_prefix_cache_broken tests/test_cache_oom tests/test_cache_oom_unsafe tests/test_prefill_slot_gpu
	@if [ -z "$(MODEL)" ]; then echo "usage: make check MODEL=/path/to/model.gguf"; exit 2; fi
	@echo "--- prefix cache: bit-identity ---"
	@./tests/test_prefix_cache $(MODEL) >/dev/null 2>&1 && echo "prefix cache PASS" || { echo "prefix cache FAILED"; exit 1; }
	@echo "--- prefix cache negative control (must FAIL) ---"
	@if ./tests/test_prefix_cache_broken $(MODEL) >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the bit-identity assertion proves nothing"; exit 1; \
	 else echo "control failed as required"; fi
	@echo "--- cache id record vs allocation width, forced OOM ---"
	@ASAN_OPTIONS=detect_leaks=0 ./tests/test_cache_oom $(MODEL) >/dev/null 2>&1 && echo "cache OOM PASS" || { echo "cache OOM FAILED"; exit 1; }
	@echo "--- cache OOM negative control (pre-fix guard; must FAIL under ASan) ---"
	@if ASAN_OPTIONS=detect_leaks=0 ./tests/test_cache_oom_unsafe $(MODEL) >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the overflow is not being detected"; exit 1; \
	 else echo "control failed as required"; fi
	@echo "--- coli_prefill_slot with GPU attention (SKIPs without a GPU) ---"
	@./tests/test_prefill_slot_gpu $(MODEL) 1 2>/dev/null | tail -1
	@./tests/test_prefill_slot_gpu $(MODEL) 1 >/dev/null 2>&1 || { echo "slot-path GPU attention FAILED"; exit 1; }
	@echo "all model-dependent checks passed"

# ---- hw_detect: CPU/RAM/Vulkan/CUDA/torch-plugin hardware probe + planner --
# hw_detect.c is plain C99 and never hard-links CUDA (dlopen("libcuda.so.1")
# at runtime, see its own file header) -- both variants below build it
# identically. Only the Vulkan enumeration half is compile-time optional.
# Compiled together with cpu_features.cpp AS C++ (-x c++), same trick
# test_gemm_i8 uses: cpu_features.h declares coli_cpu_features() with no
# extern "C", so it is always C++-mangled, and hw_detect.c calling it must be
# compiled by the same compiler into the same translation for the names to
# match -- a separate $(CC)-compiled hw_detect.o would link-fail here. hw_detect.h
# itself IS extern "C"-guarded for any future C++ caller elsewhere in the tree.
#
# -ldl: dlopen/dlsym. Not folded into $(LDLIBS) because nothing else in this
# engine's test suite needs it, and this project's Makefile keeps link flags
# next to the rule that actually requires them (see -lvulkan on the vk tests).
tests/test_hw_detect: tests/test_hw_detect.c src/hw_detect.c src/hw_detect.h src/cpu_features.cpp src/cpu_features.h
	$(CXX) $(CXXFLAGS) $(COMMON) -DCOLI_HAVE_VK -x c++ tests/test_hw_detect.c src/hw_detect.c src/cpu_features.cpp -o $@ $(LDLIBS) -lvulkan -ldl

# Plain variant: no -DCOLI_HAVE_VK, no -lvulkan. Proves the module builds and
# runs (n_vk forced to 0) on a tree with no Vulkan SDK at all, which is this
# project's standing requirement for everything outside the explicit `vk`/
# `coli-gpu` targets (see the `all` rule's own comment on why coli-gpu is kept
# separate). CUDA/torch detection is unaffected -- neither depends on COLI_HAVE_VK.
tests/test_hw_detect_novk: tests/test_hw_detect.c src/hw_detect.c src/hw_detect.h src/cpu_features.cpp src/cpu_features.h
	$(CXX) $(CXXFLAGS) $(COMMON) -x c++ tests/test_hw_detect.c src/hw_detect.c src/cpu_features.cpp -o $@ $(LDLIBS) -ldl

# Runs both variants; COLI_TEST_EXPECT_GPU=1 on the VK build only, since the
# plain build can never see a GPU (n_vk is compiled to always be 0 there) and
# asserting n_vk>=1 against it would be asserting against a control that
# cannot produce the opposite result.
test-hw-detect: tests/test_hw_detect tests/test_hw_detect_novk
	@echo "--- test_hw_detect (COLI_HAVE_VK, COLI_TEST_EXPECT_GPU=1) ---"
	COLI_TEST_EXPECT_GPU=1 ./tests/test_hw_detect
	@echo "--- test_hw_detect_novk (no Vulkan SDK at compile time) ---"
	./tests/test_hw_detect_novk
# ---- optional CUDA (2026-09-14) ---------------------------------------------
# A second discrete-GPU backend behind the same coli_backend seam Vulkan fills
# (src/backend.h / src/backend_cuda.cu). Appended, not interleaved with the
# Vulkan rules above, for the same reason the Vulkan section stayed out of
# `all`: a fresh clone must build without the CUDA toolkit either.
CUDA_HOME ?= /opt/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_89

/tmp/coli_becuda_g.o: src/backend_cuda.cu src/backend.h src/gemm_i8.h
# -fmad=false is an ADDITION to the flags this rule was specified with, not in
# the original list. Measured 2026-09-14: without it, tests/test_cuda_gemm's
# int8/int4 GEMV disagreed with the CPU reference by ~2e-7 relative (well
# inside the 2e-5 tolerance, but nonzero) purely from nvcc fusing `acc +=
# xs[b]*(float)s` into one FMA where the CPU (built engine-wide with
# -ffp-contract=off, see the top of gemm_i8.cpp and its 76%-of-cells story)
# does a separate multiply and add. Adding this flag alone took every int8/int4
# case in the test from a bounded relative error to EXACT (0.00e+00) equality
# -- the whole point of the block-order design in backend_cuda.cu's header.
# Without it this backend would be silently less exact than the CPU-parity
# claim its own comments make.
	$(NVCC) -O3 -arch=$(CUDA_ARCH) -Xcompiler "-fPIC" --allow-unsupported-compiler -fmad=false \
	  -Isrc -I../c -DCOLI_HAVE_CUDA -c src/backend_cuda.cu -o $@

# coli-cuda: the SAME binary shape as coli-gpu (same shader deps -- CUDA has
# no shaders of its own but the Vulkan backend stays compiled in, so both
# accelerators are present and COLI_BACKEND_ORDER (default "cuda,vulkan,torch")
# picks CUDA first on a machine that has one) plus the CUDA object and
# -DCOLI_HAVE_CUDA on the C++ compile, linked against cudart.
coli-cuda: shaders/gemm_i8.spv shaders/gemm_i4.spv shaders/gemm_i4_dp.spv shaders/gemm_i4f.spv \
          shaders/silu_mul_q.spv shaders/attn_decode.spv shaders/attn_decode_split.spv shaders/attn_decode_merge.spv shaders/rope_bias.spv shaders/kvwrite.spv shaders/quant.spv shaders/qknorm.spv shaders/gemm_i4_tile.spv shaders/gemm_i4_coop.spv shaders/gemm_i4_coop_ds.spv shaders/gemm_i4_short.spv shaders/gemm_i4_short_dp.spv shaders/gemm_i4_moe_dp.spv shaders/gemm_i4_moe_short_dp.spv \
          shaders/gemm_i4_mx_dp.spv shaders/gemm_i4_mx_short_dp.spv shaders/swiglu_oai_q.spv \
          src/main.cpp $(CPP_SRC) $(C_SRC) src/vk_backend.c src/backend_vk.c src/backend.h src/backend_cuda.cu src/hw_detect.c
	$(CC)  $(CFLAGS)   $(COMMON) -c src/loader.c -o /tmp/coli_loader_gc.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/q4k_shim.c -o /tmp/coli_loader_gc_q4k.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_g.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/backend_vk.c -o /tmp/coli_bevk_g.o
	$(CC)  $(CFLAGS)   $(COMMON) -DCOLI_HAVE_VK -c src/hw_detect.c -o /tmp/coli_hw_gc.o
	$(MAKE) --no-print-directory /tmp/coli_becuda_g.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -DCOLI_HAVE_CUDA -o coli-cuda src/main.cpp $(CPP_SRC) \
	  /tmp/coli_loader_gc.o /tmp/coli_loader_gc_q4k.o /tmp/coli_vkb_g.o /tmp/coli_bevk_g.o /tmp/coli_becuda_g.o /tmp/coli_hw_gc.o \
	  $(LDLIBS) -lvulkan -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64

tests/test_cuda_gemm: tests/test_cuda_gemm.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp src/backend_cuda.cu src/backend.c src/backend.h
	$(MAKE) --no-print-directory /tmp/coli_becuda_g.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_CUDA -x c++ tests/test_cuda_gemm.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp src/backend.c \
	  -x none /tmp/coli_becuda_g.o -o $@ $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64

tests/test_cuda_oai: tests/test_cuda_oai.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp src/backend_cuda.cu src/backend.c src/backend.h
	$(MAKE) --no-print-directory /tmp/coli_becuda_g.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_CUDA -x c++ tests/test_cuda_oai.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp src/backend.c \
	  -x none /tmp/coli_becuda_g.o -o $@ $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64

# test_cuda_fill: the async expert-slot fill (2026-09-15), mirroring
# tests/test_vk_fill's build rule shape but through the coli_backend seam
# (see tests/test_cuda_gemm / test_cuda_oai above for the same pattern).
tests/test_cuda_fill: tests/test_cuda_fill.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp src/backend_cuda.cu src/backend.c src/backend.h
	$(MAKE) --no-print-directory /tmp/coli_becuda_g.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_CUDA -x c++ tests/test_cuda_fill.c src/gemm_i8.cpp src/gemm_mxfp4.cpp src/cpu_features.cpp src/backend.c \
	  -x none /tmp/coli_becuda_g.o -o $@ $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64

# test_cuda_attn (2026-09-15): KV cache + decode attention through the
# coli_backend seam -- same link shape as test_cuda_gemm/test_cuda_oai above,
# backend.c supplies the declining-stub fill for every entry backend_cuda.cu
# leaves NULL (moe4*, rope_*, qknorm_*, attn_block) so the link has no holes.
tests/test_cuda_attn: tests/test_cuda_attn.c src/backend_cuda.cu src/backend.c src/backend.h
	$(MAKE) --no-print-directory /tmp/coli_becuda_g.o
	$(CXX) $(CXXFLAGS) $(COMMON) -DCOLI_HAVE_CUDA -x c++ tests/test_cuda_attn.c src/backend.c \
	  -x none /tmp/coli_becuda_g.o -o $@ $(LDLIBS) -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64
