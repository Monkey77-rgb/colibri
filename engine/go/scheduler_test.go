package coli

// Tests for the idle-sleep plumbing added 2026-09-13 (owner req, mirrors
// llama-server's --sleep-idle-seconds). These deliberately avoid needing a
// real *Model (which needs a live GGUF and cgo) by exercising the pieces of
// Scheduler that do not touch the model's ctx: drainQueueWithError and the
// Stats fields Snapshot copies out under the lock. The timer race itself
// (idleTimer fires vs. a request landing in `incoming`) and the actual
// sleep()/wake() round trip are covered by the end-to-end protocol run in the
// task record (server RSS drop, is_sleeping, single-reload-under-concurrency)
// rather than here, since they need cgo against a real GGUF -- not available
// as a lightweight, hermetic unit-test fixture in this tree.

import (
	"errors"
	"testing"
)

// TestDrainQueueWithErrorFailsEveryQueuedRequest is the "no request is
// silently dropped" half of the wake-failure path: if wake() cannot reopen
// the model after an idle-sleep, every request already queued behind the one
// that triggered the wake attempt must see the error, not hang forever.
func TestDrainQueueWithErrorFailsEveryQueuedRequest(t *testing.T) {
	s := &Scheduler{incoming: make(chan *Request, 4)}
	want := errors.New("reopen failed: out of memory")

	reqs := make([]*Request, 3)
	for i := range reqs {
		r := &Request{Err: make(chan error, 1)}
		reqs[i] = r
		s.incoming <- r
	}

	s.drainQueueWithError(want)

	if n := len(s.incoming); n != 0 {
		t.Fatalf("queue not drained: %d requests left", n)
	}
	for i, r := range reqs {
		select {
		case got := <-r.Err:
			if got != want {
				t.Fatalf("request %d: got error %v, want %v", i, got, want)
			}
		default:
			t.Fatalf("request %d: no error delivered", i)
		}
	}
}

// TestDrainQueueWithErrorOnEmptyQueueIsANoOp guards the `default:` branch:
// calling drain when nothing is queued (the common case -- wake() usually
// succeeds) must return immediately rather than blocking on the channel.
func TestDrainQueueWithErrorOnEmptyQueueIsANoOp(t *testing.T) {
	s := &Scheduler{incoming: make(chan *Request, 1)}
	done := make(chan struct{})
	go func() {
		s.drainQueueWithError(errors.New("unused"))
		close(done)
	}()
	select {
	case <-done:
	default:
	}
	<-done // if drainQueueWithError blocked, this would hang and the test would time out
}

// TestSnapshotReportsSleepState is the read side /health depends on: Sleeping
// and LastWakeSeconds must survive the copy Snapshot makes under s.mu, since
// /health calls it from a goroutine other than the scheduler's owning one.
func TestSnapshotReportsSleepState(t *testing.T) {
	s := &Scheduler{
		incoming: make(chan *Request, 1),
		live:     map[int]*seqState{},
	}
	s.stats.BatchHist = map[int]uint64{}

	if snap := s.Snapshot(); snap.Sleeping || snap.LastWakeSeconds != 0 {
		t.Fatalf("fresh scheduler should report awake with no prior wake: %+v", snap)
	}

	s.mu.Lock()
	s.sleeping = true
	s.mu.Unlock()
	if snap := s.Snapshot(); !snap.Sleeping {
		t.Fatalf("Snapshot did not report sleeping=true")
	}

	s.mu.Lock()
	s.sleeping = false
	s.lastWakeSeconds = 2.5
	s.mu.Unlock()
	snap := s.Snapshot()
	if snap.Sleeping {
		t.Fatalf("Snapshot still reports sleeping after wake")
	}
	if snap.LastWakeSeconds != 2.5 {
		t.Fatalf("LastWakeSeconds = %v, want 2.5", snap.LastWakeSeconds)
	}
}
