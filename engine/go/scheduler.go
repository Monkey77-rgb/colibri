package coli

// Continuous batching scheduler.
//
// THE POINT, stated plainly because it is the whole reason this layer exists.
// A single-slot server runs every generated token at batch size 1. That is the
// regime we measured to be DRAM-bound, where the weights are re-read for each
// token and the wide CPU kernel is 17-22% SLOWER than the narrow one. Batching k
// sequences reads the weights ONCE per step for all k. Measured on this engine,
// qwen2.5-3b: 9.4 tok/s at B=1 against 33.5 tok/s at B=8 -- 3.6x from scheduling
// alone, with the kernels untouched.
//
// CONTINUOUS, not static. A finished sequence releases its slot immediately and
// a waiting request takes it on the very next step. A static batch would make
// every sequence wait for the slowest in its cohort, which is how "batching
// helps throughput but ruins latency" happens. Here a request never waits for an
// unrelated one to finish.
//
// The engine context is NOT thread-safe, so exactly one goroutine (run) ever
// touches it. Everything else communicates over channels. That is also why the
// scheduler owns slot allocation: the thing that decides what runs together must
// be the same thing that knows what is running.

import (
	"context"
	"errors"
	"sync"
)

type Request struct {
	Prompt    string
	MaxTokens int
	Sampler   Sampler
	Stop      []string

	Out  chan Token // streamed; closed when the sequence ends
	Err  chan error
	once sync.Once
}

type Token struct {
	Text string
	ID   int32
	Done bool
	// Reason is "stop" (EOS or a stop string) or "length".
	Reason string
}

type seqState struct {
	req     *Request
	slot    int
	pos     int32
	cur     int32
	emitted int
	history []int32
	text    string
}

type Scheduler struct {
	m       *Model
	incoming chan *Request
	free    []int
	live    map[int]*seqState

	mu    sync.Mutex
	stats Stats
}

type Stats struct {
	Active      int
	Queued      int
	StepsTotal  uint64
	TokensTotal uint64
	// BatchHist[k] counts steps that ran with k sequences. This is the honest
	// way to report batching: a mean batch size hides a server that is really
	// running at 1 almost always.
	BatchHist map[int]uint64
}

func NewScheduler(m *Model, queue int) *Scheduler {
	s := &Scheduler{
		m:        m,
		incoming: make(chan *Request, queue),
		live:     map[int]*seqState{},
	}
	for i := 0; i < m.NSlots; i++ {
		s.free = append(s.free, i)
	}
	s.stats.BatchHist = map[int]uint64{}
	return s
}

func (s *Scheduler) Submit(r *Request) error {
	r.Out = make(chan Token, 64)
	r.Err = make(chan error, 1)
	select {
	case s.incoming <- r:
		return nil
	default:
		return errors.New("queue full")
	}
}

func (s *Scheduler) Snapshot() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.stats
	c.Active = len(s.live)
	c.Queued = len(s.incoming)
	h := make(map[int]uint64, len(s.stats.BatchHist))
	for k, v := range s.stats.BatchHist {
		h[k] = v
	}
	c.BatchHist = h
	return c
}

func (s *seqState) finish(reason string) {
	s.req.once.Do(func() {
		s.req.Out <- Token{Done: true, Reason: reason}
		close(s.req.Out)
	})
}

// Run owns the model. One goroutine, for the lifetime of the process.
func (s *Scheduler) Run(ctx context.Context) {
	for {
		// Admit as many waiting requests as there are free slots. Prefill is
		// where a new sequence costs the most, so it happens here, outside the
		// decode step, rather than stalling the batch mid-flight.
		for len(s.free) > 0 {
			var r *Request
			select {
			case r = <-s.incoming:
			default:
			}
			if r == nil {
				break
			}
			ids, err := s.m.Tokenize(r.Prompt)
			if err != nil {
				r.Err <- err
				continue
			}
			if s.m.AddBOS && s.m.BOS >= 0 {
				ids = append([]int32{int32(s.m.BOS)}, ids...)
			}
			if len(ids)+r.MaxTokens > s.m.NCtx {
				r.Err <- errors.New("prompt + max_tokens exceeds context")
				continue
			}
			slot := s.free[len(s.free)-1]
			s.free = s.free[:len(s.free)-1]
			lg, err := s.m.Prefill(slot, ids)
			if err != nil {
				s.free = append(s.free, slot)
				r.Err <- err
				continue
			}
			st := &seqState{req: r, slot: slot, pos: int32(len(ids)), history: ids}
			st.cur = s.m.Sample(lg, &r.Sampler, st.history)
			s.live[slot] = st
		}

		if len(s.live) == 0 {
			// Nothing running: block rather than spin. A busy-wait here would
			// burn a core doing nothing, which on a handheld is a battery bug.
			select {
			case <-ctx.Done():
				return
			case r := <-s.incoming:
				s.incoming <- r
				continue
			}
		}

		// One batched step across every live sequence.
		n := len(s.live)
		slots := make([]int32, 0, n)
		poss := make([]int32, 0, n)
		toks := make([]int32, 0, n)
		order := make([]*seqState, 0, n)
		for _, st := range s.live {
			slots = append(slots, int32(st.slot))
			poss = append(poss, st.pos)
			toks = append(toks, st.cur)
			order = append(order, st)
		}
		logits, err := s.m.DecodeBatch(slots, poss, toks)
		if err != nil {
			for _, st := range order {
				st.req.Err <- err
				st.finish("error")
				s.free = append(s.free, st.slot)
				delete(s.live, st.slot)
			}
			continue
		}

		s.mu.Lock()
		s.stats.StepsTotal++
		s.stats.TokensTotal += uint64(n)
		s.stats.BatchHist[n]++
		s.mu.Unlock()

		for i, st := range order {
			id := st.cur
			st.history = append(st.history, id)
			st.pos++
			st.emitted++

			done, reason := false, ""
			if int(id) == s.m.EOS {
				done, reason = true, "stop"
			} else {
				piece := s.m.Detokenize([]int32{id})
				st.text += piece
				select {
				case st.req.Out <- Token{Text: piece, ID: id}:
				default: // client is not draining; drop rather than stall the batch
				}
				for _, sw := range st.req.Stop {
					if sw != "" && len(st.text) >= len(sw) && st.text[len(st.text)-len(sw):] == sw {
						done, reason = true, "stop"
					}
				}
			}
			if !done && st.emitted >= st.req.MaxTokens {
				done, reason = true, "length"
			}
			if !done && int(st.pos) >= s.m.NCtx {
				done, reason = true, "length"
			}
			if done {
				st.finish(reason)
				s.free = append(s.free, st.slot)
				delete(s.live, st.slot)
				continue
			}
			st.cur = s.m.Sample(logits[i*s.m.NVocab:(i+1)*s.m.NVocab], &st.req.Sampler, st.history)
		}

		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}
