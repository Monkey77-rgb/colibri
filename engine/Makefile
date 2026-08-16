# Colibri engine. -ffp-contract=off is NOT optional: see the comment at the top
# of src/gemm_i8.c. Without it the engine's own reference test disagrees with
# its kernels on 76% of cells and every bit-exactness claim becomes unfalsifiable.
CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra -Wno-unused-parameter
CFLAGS  += -ffp-contract=off
OMP     ?= -fopenmp
ARCH    ?= native
CFLAGS  += -march=$(ARCH)
SRC      = src/cpu_features.c src/gemm_i8.c
LDLIBS   = -lm

.PHONY: all test bench clean
all: test

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
	rm -f tests/test_cpu_features tests/test_gemm_i8 tests/test_gemm_i8_broken
