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

CPP_SRC  = src/coli_api.cpp src/model.cpp src/gemm_i8.cpp src/cpu_features.cpp src/sample.cpp
C_SRC    = src/loader.c          # stays C: wraps ../c unchanged, see loader.h

.PHONY: all lib cli cli-gpu go python test vk windows clean
all: lib cli go

# ---- C++ core -------------------------------------------------------------
lib: libcoli.so
libcoli.so: $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -fPIC -c $(C_SRC) -o /tmp/coli_loader.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -fPIC -shared -o $@ $(CPP_SRC) /tmp/coli_loader.o $(LDLIBS)

cli: coli
coli: src/main.cpp $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -c $(C_SRC) -o /tmp/coli_loader_s.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -o $@ src/main.cpp $(CPP_SRC) /tmp/coli_loader_s.o $(LDLIBS)

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
tests/test_shader_index: tests/test_shader_index.c src/gemm_i8.cpp src/cpu_features.cpp
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ $^ -x none -o $@ $(LDLIBS)
# The OOM path in cache_ids_grow is unreachable under Linux overcommit, so the
# test build forces it (COLI_BREAK_CACHE_GROW) and the CONTROL build additionally
# restores the pre-fix guard (COLI_CACHE_UNSAFE_GUARD). Both carry ASan: the bug
# is a heap-buffer-overflow, and without a sanitizer the control would very
# likely "pass" by overwriting something harmless and never crashing.
# Standalone: no model, no shaders beyond the one used to init the device.
# Previously built by hand, so `make` could not reproduce a published number.
tests/test_vk_floor: tests/test_vk_floor.c src/vk_backend.c
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_floor.o
	$(CXX) $(CXXFLAGS) $(COMMON) -DCOLI_HAVE_VK -x c++ tests/test_vk_floor.c -x none /tmp/coli_vkb_floor.o -o $@ $(LDLIBS) -lvulkan

tests/test_cache_oom: tests/test_cache_oom.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -c $(C_SRC) -o /tmp/coli_loader_oom.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -x c++ tests/test_cache_oom.c $(CPP_SRC) -x none /tmp/coli_loader_oom.o -o $@ $(LDLIBS)
tests/test_cache_oom_unsafe: tests/test_cache_oom.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -DCOLI_CACHE_UNSAFE_GUARD -c $(C_SRC) -o /tmp/coli_loader_oomu.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -fsanitize=address -g -DCOLI_BREAK_CACHE_GROW -DCOLI_CACHE_UNSAFE_GUARD -x c++ tests/test_cache_oom.c $(CPP_SRC) -x none /tmp/coli_loader_oomu.o -o $@ $(LDLIBS)

tests/test_batch: tests/test_batch.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -c $(C_SRC) -o /tmp/coli_loader_t.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_batch.c $(CPP_SRC) -x none /tmp/coli_loader_t.o -o $@ $(LDLIBS)

test: tests/test_gemm_i8 tests/test_gemm_i8_broken tests/test_gemm_i4 tests/test_gemm_i4_broken tests/test_shader_index
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
	@echo "--- gemm_i4.comp addressing (CPU-side; NOT a GPU test) ---"
	@./tests/test_shader_index 512 256

# ---- optional Vulkan ------------------------------------------------------
VK_INC ?=
shaders/%.spv: shaders/%.comp
	glslc -O $< -o $@
# vk_backend.c stays C: Vulkan's designated initialisers and void* conversions
# are C idioms, and vk_backend.h carries extern "C" so C++ links to it.
vk: shaders/gemm_i8.spv shaders/gemm_i4.spv shaders/gemm_i4f.spv shaders/silu_mul_q.spv
	$(CC) $(CFLAGS) $(COMMON) $(if $(VK_INC),-I$(VK_INC),) -c src/vk_backend.c -o /tmp/coli_vkb.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) $(if $(VK_INC),-I$(VK_INC),) -x c++ \
	  tests/test_vk_gemm.c src/gemm_i8.cpp src/cpu_features.cpp -x none /tmp/coli_vkb.o \
	  -o tests/test_vk_gemm $(LDLIBS) -lvulkan
	./tests/test_vk_gemm 4096 16384

# A CLI linked against Vulkan, so `--gpu` works. Kept separate from `cli` because
# the default build must not require a Vulkan SDK to compile.
cli-gpu: shaders/gemm_i8.spv shaders/gemm_i4.spv shaders/silu_mul_q.spv
	$(CC)  $(CFLAGS)   $(COMMON) -c $(C_SRC) -o /tmp/coli_loader_g.o
	$(CC)  $(CFLAGS)   $(COMMON) -c src/vk_backend.c -o /tmp/coli_vkb_g.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_HAVE_VK -o coli-gpu src/main.cpp $(CPP_SRC) \
	  /tmp/coli_loader_g.o /tmp/coli_vkb_g.o $(LDLIBS) -lvulkan

# ---- Windows cross-build --------------------------------------------------
# -static is REQUIRED: a dynamic build starts and silently produces nothing when
# libgomp-1.dll / libwinpthread-1.dll are missing.
WINCXX ?= x86_64-w64-mingw32-g++
WINCC  ?= x86_64-w64-mingw32-gcc
windows:
	$(WINCC)  -O2 -march=x86-64-v3 -ffp-contract=off -D_FILE_OFFSET_BITS=64 -I../c -Isrc -c $(C_SRC) -o /tmp/coli_loader_w.o
	$(WINCXX) -std=c++20 -O2 -march=x86-64-v3 -fopenmp -ffp-contract=off -D_FILE_OFFSET_BITS=64 \
	  -I../c -Isrc -static -o coli.exe src/main.cpp $(CPP_SRC) /tmp/coli_loader_w.o -lm
	@file coli.exe

clean:
	rm -f coli coli.exe libcoli.so go/coli-server shaders/*.spv \
	      tests/test_gemm_i8 tests/test_gemm_i8_broken tests/test_batch tests/test_vk_gemm

# The prefix cache cannot be tested from the CLI: one prompt per process means
# the cache is always cold and the reuse path never executes. This is the only
# thing that reaches it.
tests/test_prefix_cache: tests/test_prefix_cache.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -c $(C_SRC) -o /tmp/coli_loader_pc.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -x c++ tests/test_prefix_cache.c $(CPP_SRC) -x none /tmp/coli_loader_pc.o -o $@ $(LDLIBS)

# Negative control: COLI_BREAK_PREFIX reuses one token more than was matched, so
# the reused KV no longer describes the prompt. `make check` requires this build
# to FAIL -- if it passes, the bit-identity assertion is not testing what it says.
tests/test_prefix_cache_broken: tests/test_prefix_cache.c $(CPP_SRC) $(C_SRC)
	$(CC)  $(CFLAGS)   $(COMMON) -DCOLI_BREAK_PREFIX -c $(C_SRC) -o /tmp/coli_loader_pcb.o
	$(CXX) $(CXXFLAGS) $(COMMON) $(OMP) -DCOLI_BREAK_PREFIX -x c++ tests/test_prefix_cache.c $(CPP_SRC) -x none /tmp/coli_loader_pcb.o -o $@ $(LDLIBS)

# ---- model-dependent checks ------------------------------------------------
# Separate from `test` because these need real weights. `test` must stay
# runnable on a machine with no model on disk.
#   usage: make check MODEL=/path/to/model.gguf
# ASAN_OPTIONS=detect_leaks=0 throughout: the gguf index is deliberately not
# freed at exit, and that pre-existing leak would otherwise drown the signal
# these tests exist to produce.
check: tests/test_prefix_cache tests/test_prefix_cache_broken tests/test_cache_oom tests/test_cache_oom_unsafe
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
	@echo "all model-dependent checks passed"
