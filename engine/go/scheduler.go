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
	"log"
	"sync"
	"time"
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
	m        *Model
	incoming chan *Request
	free     []int
	live     map[int]*seqState

	// idleTimeout is how long Run waits with no live sequence and an empty
	// queue before putting the model to sleep. 0 disables the feature
	// entirely (default; mirrors llama-server's --sleep-idle-seconds=0).
	// 2026-09-13, owner req.
	idleTimeout time.Duration

	mu              sync.Mutex
	stats           Stats
	sleeping        bool
	lastWakeSeconds float64
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
	// Sleeping and LastWakeSeconds report idle-sleep state for /health.
	// 2026-09-13.
	Sleeping        bool
	LastWakeSeconds float64
}

// NewScheduler builds a scheduler. idleTimeout <= 0 disables idle-sleep: Run
// then behaves exactly as it did before 2026-09-13 (blocks forever with no
// live sequences, never frees the model).
func NewScheduler(m *Model, queue int, idleTimeout time.Duration) *Scheduler {
	s := &Scheduler{
		m:           m,
		incoming:    make(chan *Request, queue),
		live:        map[int]*seqState{},
		idleTimeout: idleTimeout,
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
	c.Sleeping = s.sleeping
	c.LastWakeSeconds = s.lastWakeSeconds
	return c
}

// drainQueueWithError fails every request currently sitting in the incoming
// queue with err. Used only when wake() fails after an idle-sleep: those
// requests cannot be served by a model that will not reopen, and leaving them
// queued would hang their callers forever rather than surfacing the failure.
// 2026-09-13.
func (s *Scheduler) drainQueueWithError(err error) {
	for {
		select {
		case r := <-s.incoming:
			r.Err <- err
		default:
			return
		}
	}
}

func (s *seqState) finish(reason string) {
	s.req.once.Do(func() {
		s.req.Out <- Token{Done: true, Reason: reason}
		close(s.req.Out)
	})
}

// Run owns the model. One goroutine, for the lifetime of the process.
//
// IDLE-SLEEP (2026-09-13, owner req, mirrors llama-server's
// --sleep-idle-seconds). When idleTimeout > 0 and Run finds nothing live and
// the incoming queue empty, it starts a timer instead of blocking forever. If
// the timer fires before a request arrives, Run -- the sole goroutine that
// ever touches the model's ctx, per this file's own threading contract above
// -- frees the model (RAM, KV cache, any GPU allocations) via Model.sleep()
// and then blocks on the SAME incoming channel to wait for the next request.
// Because Run is that single owner, "the wake lock" is simply "only Run
// reopens the model", which also gives the required exactly-one-reload
// guarantee for free: N concurrent requests all land in the buffered
// `incoming` channel (Submit never touches ctx), Run drains exactly one of
// them to trigger wake(), reopens once, then proceeds through the normal
// admit loop below where the rest are already waiting. No request is ever
// dropped -- a request that arrives while sleep() is in flight simply sits in
// the channel until Run reaches the wake-wait select below.
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
			if s.idleTimeout <= 0 {
				// Idle-sleep disabled (default): block rather than spin. A
				// busy-wait here would burn a core doing nothing, which on a
				// handheld is a battery bug. Unchanged from before 2026-09-13.
				select {
				case <-ctx.Done():
					return
				case r := <-s.incoming:
					s.incoming <- r
					continue
				}
			}

			// Idle-sleep enabled: same wait, but bounded by idleTimeout so we
			// can act if nothing shows up. 2026-09-13.
			timer := time.NewTimer(s.idleTimeout)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case r := <-s.incoming:
				timer.Stop()
				s.incoming <- r
				continue
			case <-timer.C:
				// Timer fired with (as of the last check) nothing live and
				// nothing queued. Re-check right now, under the lock, before
				// actually freeing anything: a request landing in the
				// buffered `incoming` channel in the gap between the timer
				// firing and this line must win the race and skip sleep this
				// cycle, per the owner's requirement. len() on a channel is a
				// safe, instantaneous read; s.mu here is just to make the
				// sleeping-flag update atomic with this check for /health's
				// benefit, not because the channel read needs it.
				s.mu.Lock()
				if len(s.incoming) > 0 {
					s.mu.Unlock()
					continue
				}
				s.sleeping = true
				s.mu.Unlock()

				t0 := time.Now()
				s.m.sleep()
				log.Printf("coli: idle %s with no in-flight work -- model asleep "+
					"(weights/KV/GPU freed) at %s", s.idleTimeout, t0.Format(time.RFC3339))

				// Block for the next request. This IS the wake lock: Run is
				// the only goroutine that reopens the model, so whichever
				// request arrives here is the one and only trigger for the
				// reload, no matter how many others are queued behind it.
				var r *Request
				select {
				case <-ctx.Done():
					return
				case r = <-s.incoming:
				}

				wt0 := time.Now()
				if err := s.m.wake(); err != nil {
					// The model could not be reopened (e.g. the file moved,
					// or ran out of memory while other processes grew into
					// the freed space). Fail this request and everything
					// else currently queued rather than hang forever with
					// sleeping stuck true and no way out; the operator's log
					// gets the reason.
					log.Printf("coli: wake failed after %s idle: %v", s.idleTimeout, err)
					r.Err <- err
					s.drainQueueWithError(err)
					s.mu.Lock()
					s.sleeping = false
					s.mu.Unlock()
					continue
				}
				woke := time.Since(wt0).Seconds()
				s.mu.Lock()
				s.sleeping = false
				s.lastWakeSeconds = woke
				s.mu.Unlock()
				log.Printf("coli: woke in %.3fs at %s", woke, time.Now().Format(time.RFC3339))

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
