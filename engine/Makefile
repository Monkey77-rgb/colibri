# Colibri engine. -ffp-contract=off is NOT optional: see the comment at the top
# of src/gemm_i8.c. Without it the engine's own reference test disagrees with
# its kernels on 76% of cells and every bit-exactness claim becomes unfalsifiable.
CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -ffp-contract=off
OMP     ?= -fopenmp
ARCH    ?= native
CFLAGS  += -march=$(ARCH)
SRC      = src/cpu_features.c src/gemm_i8.c src/model.c src/sample.c
CFLAGS  += -I../c        # gguf_reader.h, gguf_meta.h, ggml_dequant.h, tok.h, tok_gguf.h
LDLIBS   = -lm

# Vulkan is OPTIONAL. VULKAN=1 builds the GPU backend; it needs Vulkan headers
# (pacman -S vulkan-headers / apt install libvulkan-dev) and glslc. Without it
# everything else still builds -- a machine with no GPU is not an error state.
VULKAN  ?= 0
VK_INC  ?=
ifeq ($(VULKAN),1)
CFLAGS  += -DCOLI_VULKAN $(if $(VK_INC),-I$(VK_INC),)
SRC     += src/vk_backend.c
LDLIBS  += -lvulkan
endif

shaders/gemm_i8.spv: shaders/gemm_i8.comp
	glslc -O $< -o $@

tests/test_vk_gemm: tests/test_vk_gemm.c src/vk_backend.c src/gemm_i8.c src/cpu_features.c shaders/gemm_i8.spv
	$(CC) $(CFLAGS) $(OMP) $(if $(VK_INC),-I$(VK_INC),) -o $@ $(filter %.c,$^) $(LDLIBS) -lvulkan

.PHONY: all test bench vk clean

# Runs the GPU test; SKIPs cleanly when there is no Vulkan device.
vk: tests/test_vk_gemm
	./tests/test_vk_gemm 2048 2048
all: coli coli-server test

coli: src/main.c $(SRC)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(LDLIBS)

coli-server: src/server.c $(SRC)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(LDLIBS)

tests/test_cpu_features: tests/test_cpu_features.c src/cpu_features.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_gemm_i8: tests/test_gemm_i8.c $(SRC)
	$(CC) $(CFLAGS) $(OMP) -o $@ $^ $(LDLIBS)
# Negative control: the SAME test against a deliberately broken wide kernel.
# If this passes, the test proves nothing.
tests/test_gemm_i8_broken: tests/test_gemm_i8.c $(SRC)
	$(CC) $(CFLAGS) $(OMP) -DCOLI_BREAK_WIDE -o $@ $^ $(LDLIBS)

test: tests/test_cpu_features tests/test_gemm_i8 tests/test_gemm_i8_broken
	./tests/test_cpu_features
	@echo "--- kernels vs reference (must PASS) ---"
	@./tests/test_gemm_i8 2048 2048
	@echo "--- negative control (must FAIL, rc=1) ---"
	@if ./tests/test_gemm_i8_broken 2048 2048 >/dev/null 2>&1; then \
	   echo "CONTROL DID NOT FAIL -- the test proves nothing"; exit 1; \
	 else echo "control failed as required"; fi

clean:
	rm -f coli coli-server tests/test_vk_gemm shaders/gemm_i8.spv tests/test_cpu_features tests/test_gemm_i8 tests/test_gemm_i8_broken
