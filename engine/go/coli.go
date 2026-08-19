// Package coli binds the C++ inference core.
//
// WHY GO OWNS THIS LAYER. The core is C++ because the kernels need intrinsics
// and manual memory layout. Everything ABOVE the kernels is scheduling: which
// request runs, in which slot, batched with which others. That is concurrency
// plumbing, and Go's goroutines and channels express it in a fraction of the
// code C++ would need -- with cross-compilation to Linux, Windows, macOS and
// ARM from one toolchain, which is the other half of why it is here.
//
// THE THREADING CONTRACT, and it is not optional. A coli_ctx owns one KV cache
// and one set of scratch buffers, so it is NOT thread-safe. Two goroutines
// stepping it concurrently is a data race that no lock inside the kernels could
// fix. This package therefore gives the context to exactly ONE scheduler
// goroutine and everything else talks to it over channels. That is not a
// workaround for the C++ side being unsafe; it is the correct shape anyway,
// because batching REQUIRES a single point that decides what runs together.
package coli

/*
#cgo CFLAGS: -I${SRCDIR}/../src
#cgo LDFLAGS: -L${SRCDIR}/.. -lcoli -lm -Wl,-rpath,${SRCDIR}/..
#include <stdlib.h>
#include "coli_api.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type Model struct {
	ctx    *C.coli_ctx
	NVocab int
	NCtx   int
	NSlots int
	BOS    int
	EOS    int
	AddBOS bool
	Arch   string
}

// Weight format, passed to OpenW4.
const (
	W4Off  = 0 // int8 weights only
	W4Both = 1 // both formats, chosen per batch: same speed as W4Only, +56% memory
	W4Only = 2 // int4 only: 0.68x peak RSS, 1.42x decode, 1.08x prefill, +3.87% NLL
)

// Open loads a model with int8 weights. nSlots is the maximum number of
// sequences that can decode together; each costs a full KV cache, so it is a
// memory decision.
func Open(path string, nCtx, nSlots int, int8Weights bool) (*Model, error) {
	return OpenW4(path, nCtx, nSlots, int8Weights, W4Off)
}

// OpenW4 is Open with the weight format chosen explicitly. It is a separate
// function rather than a wider Open for the same reason coli_open_w4 is
// separate in the C ABI: existing callers keep compiling. The format is a
// per-model property, so a server can hold one int4 model and one int8 model at
// the same time -- which is why it is an argument and not a process-wide
// setting.
func OpenW4(path string, nCtx, nSlots int, int8Weights bool, w4 int) (*Model, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	errbuf := (*C.char)(C.malloc(512))
	defer C.free(unsafe.Pointer(errbuf))
	*(*C.char)(unsafe.Pointer(errbuf)) = 0

	q := C.int(0)
	if int8Weights {
		q = 1
	}
	ctx := C.coli_open_w4(cp, C.int(nCtx), C.int(nSlots), q, C.int(w4), errbuf, 512)
	if ctx == nil {
		return nil, errors.New(C.GoString(errbuf))
	}
	m := &Model{
		ctx:    ctx,
		NVocab: int(C.coli_n_vocab(ctx)),
		NCtx:   int(C.coli_n_ctx(ctx)),
		NSlots: int(C.coli_n_slots(ctx)),
		BOS:    int(C.coli_bos(ctx)),
		EOS:    int(C.coli_eos(ctx)),
		AddBOS: C.coli_add_bos(ctx) != 0,
		Arch:   C.GoString(C.coli_arch(ctx)),
	}
	// A finalizer is a backstop, not the plan: callers should Close. Without it
	// a dropped Model leaks several GiB, which is the kind of leak nobody
	// notices until the box swaps.
	runtime.SetFinalizer(m, func(x *Model) { x.Close() })
	return m, nil
}

func (m *Model) Close() {
	if m.ctx != nil {
		C.coli_close(m.ctx)
		m.ctx = nil
		runtime.SetFinalizer(m, nil)
	}
}

// Kernel reports which CPU kernel a batch of the given size would select. Used
// by /health so an operator can see the batch-size dispatch actually moving,
// rather than trusting that it does.
func (m *Model) Kernel(batch int) string {
	return C.GoString(C.coli_kernel(m.ctx, C.int(batch)))
}

func (m *Model) Tokenize(text string) ([]int32, error) {
	ct := C.CString(text)
	defer C.free(unsafe.Pointer(ct))
	max := len(text) + 16
	buf := make([]int32, max)
	n := int(C.coli_tokenize(m.ctx, ct, (*C.int32_t)(unsafe.Pointer(&buf[0])), C.int(max)))
	if n < 0 {
		return nil, fmt.Errorf("tokenize failed (%d)", n)
	}
	return buf[:n], nil
}

func (m *Model) Detokenize(ids []int32) string {
	if len(ids) == 0 {
		return ""
	}
	// 8 bytes per token is generous for UTF-8 pieces; the C side bounds-checks
	// anyway and returns the count it actually wrote.
	buf := make([]byte, len(ids)*8+64)
	n := int(C.coli_detokenize(m.ctx, (*C.int32_t)(unsafe.Pointer(&ids[0])), C.int(len(ids)),
		(*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	if n <= 0 {
		return ""
	}
	return string(buf[:n])
}

// Prefill runs a prompt into a slot and returns logits for its last token.
func (m *Model) Prefill(slot int, ids []int32) ([]float32, error) {
	if len(ids) == 0 {
		return nil, errors.New("empty prompt")
	}
	out := make([]float32, m.NVocab)
	r := C.coli_prefill(m.ctx, C.int(slot), (*C.int32_t)(unsafe.Pointer(&ids[0])),
		C.int(len(ids)), (*C.float)(unsafe.Pointer(&out[0])))
	if r != 0 {
		return nil, fmt.Errorf("prefill failed (%d)", r)
	}
	return out, nil
}

// SetPrefixCache turns KV prefix reuse on or off.
//
// Turn it OFF for any benchmark that re-sends a prompt. With it on, the second
// and later sends match the cached prefix and skip prefill, so the run measures
// the cache rather than the work -- and the batch histogram cannot detect that,
// because it proves DECODE batched, a different question.
//
// The flag is process-wide in the engine today, not per-Model.
func (m *Model) SetPrefixCache(on bool) {
	v := C.int(0)
	if on {
		v = 1
	}
	C.coli_prefix_cache_set(m.ctx, v)
}

// PrefixStats returns cumulative prompt tokens reused from cache and asked for,
// across every Prefill on this Model. reused/asked is the hit rate.
func (m *Model) PrefixStats() (reused, asked int64) {
	var r, a C.longlong
	C.coli_prefix_stats(m.ctx, &r, &a)
	return int64(r), int64(a)
}

// DecodeBatch steps n sequences by one token each. Returns n*NVocab logits.
func (m *Model) DecodeBatch(slots, positions, tokens []int32) ([]float32, error) {
	n := len(tokens)
	if n == 0 || len(slots) != n || len(positions) != n {
		return nil, errors.New("slots, positions and tokens must be the same non-zero length")
	}
	out := make([]float32, n*m.NVocab)
	r := C.coli_decode_batch(m.ctx,
		(*C.int32_t)(unsafe.Pointer(&slots[0])),
		(*C.int32_t)(unsafe.Pointer(&positions[0])),
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int(n), (*C.float)(unsafe.Pointer(&out[0])))
	if r != 0 {
		return nil, fmt.Errorf("decode failed (%d)", r)
	}
	return out, nil
}

// Sampler carries its own RNG state, so two requests with the same seed produce
// the same tokens no matter how they interleave. A shared sampler would not.
type Sampler struct {
	Temp          float32
	TopK          int32
	TopP          float32
	MinP          float32
	RepeatPenalty float32
	RepeatLastN   int32
	Seed          uint64
	rng           [4]uint64
}

func NewSampler() Sampler {
	return Sampler{Temp: 0, TopK: 0, TopP: 1, MinP: 0, RepeatPenalty: 1}
}

func (m *Model) Sample(logits []float32, s *Sampler, prev []int32) int32 {
	p := C.coli_sample_params{
		temp: C.float(s.Temp), top_k: C.int32_t(s.TopK), top_p: C.float(s.TopP),
		min_p: C.float(s.MinP), repeat_penalty: C.float(s.RepeatPenalty),
		repeat_last_n: C.int32_t(s.RepeatLastN), seed: C.uint64_t(s.Seed),
	}
	for i := 0; i < 4; i++ {
		p.rng_state[i] = C.uint64_t(s.rng[i])
	}
	var pp *C.int32_t
	if len(prev) > 0 {
		pp = (*C.int32_t)(unsafe.Pointer(&prev[0]))
	}
	t := C.coli_sample(m.ctx, (*C.float)(unsafe.Pointer(&logits[0])), &p, pp, C.int(len(prev)))
	for i := 0; i < 4; i++ {
		s.rng[i] = uint64(p.rng_state[i])
	}
	return int32(t)
}
