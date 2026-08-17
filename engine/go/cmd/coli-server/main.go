// coli-server — the Go serving layer.
//
// Replaces the 250-line C server, and the reason is not aesthetic. That one had
// a single slot, so every generated token ran at batch size 1 -- the regime
// measured to be DRAM-bound and where the wide CPU kernel is 17-22% SLOWER.
// Continuous batching is a concurrency problem, and it is the thing Go is here
// for. The kernels did not change; the scheduling did.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	coli "github.com/Monkey77-rgb/colibri/engine/go"
)

var (
	model  = flag.String("m", "", "path to a .gguf model (required)")
	addr   = flag.String("addr", "127.0.0.1:8100", "listen address")
	nCtx   = flag.Int("c", 0, "context length (0 = the model's own)")
	nSlots = flag.Int("slots", 4, "max sequences decoding together; each costs a full KV cache")
	f32    = flag.Bool("f32", false, "full-precision weights (~4x memory, slow; for validating a new arch)")
	queue  = flag.Int("queue", 256, "max queued requests")
)

type completionReq struct {
	Prompt        string   `json:"prompt"`
	MaxTokens     int      `json:"max_tokens"`
	Temperature   float32  `json:"temperature"`
	TopK          int32    `json:"top_k"`
	TopP          float32  `json:"top_p"`
	MinP          float32  `json:"min_p"`
	RepeatPenalty float32  `json:"repeat_penalty"`
	Seed          uint64   `json:"seed"`
	Stop          []string `json:"stop"`
	Stream        bool     `json:"stream"`
}

func main() {
	flag.Parse()
	if *model == "" {
		flag.Usage()
		os.Exit(2)
	}
	m, err := coli.Open(*model, *nCtx, *nSlots, !*f32)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	defer m.Close()
	log.Printf("%s: vocab=%d ctx=%d slots=%d", m.Arch, m.NVocab, m.NCtx, m.NSlots)
	log.Printf("kernel at batch 1: %s | at batch %d: %s", m.Kernel(1), *nSlots, m.Kernel(*nSlots))

	sch := coli.NewScheduler(m, *queue)
	ctx, cancel := context.WithCancel(context.Background())
	go sch.Run(ctx)

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		st := sch.Snapshot()
		writeJSON(w, 200, map[string]any{
			"status": "ok", "arch": m.Arch, "slots": m.NSlots,
			"active": st.Active, "queued": st.Queued,
			"steps": st.StepsTotal, "tokens": st.TokensTotal,
			// The batch histogram, not a mean: a mean hides a server that is
			// really running at batch 1 almost all the time.
			"batch_histogram": st.BatchHist,
			"kernel_batch1":   m.Kernel(1),
			"kernel_full":     m.Kernel(m.NSlots),
		})
	})

	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeErr(w, 405, "POST only")
			return
		}
		var q completionReq
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&q); err != nil {
			writeErr(w, 400, "bad json: "+err.Error())
			return
		}
		if q.Prompt == "" {
			writeErr(w, 400, "missing prompt")
			return
		}
		if q.MaxTokens <= 0 {
			q.MaxTokens = 128
		}
		s := coli.NewSampler()
		s.Temp, s.TopK, s.Seed = q.Temperature, q.TopK, q.Seed
		if q.TopP > 0 {
			s.TopP = q.TopP
		}
		s.MinP = q.MinP
		if q.RepeatPenalty > 0 {
			s.RepeatPenalty = q.RepeatPenalty
			s.RepeatLastN = 64
		}
		req := &coli.Request{Prompt: q.Prompt, MaxTokens: q.MaxTokens, Sampler: s, Stop: q.Stop}
		if err := sch.Submit(req); err != nil {
			writeErr(w, 503, err.Error())
			return
		}

		if q.Stream {
			streamSSE(w, r, req)
			return
		}
		var text string
		var reason string
		for {
			select {
			case e := <-req.Err:
				writeErr(w, 500, e.Error())
				return
			case t, ok := <-req.Out:
				if !ok {
					writeJSON(w, 200, map[string]any{
						"object": "text_completion", "model": m.Arch,
						"choices": []any{map[string]any{
							"index": 0, "text": text, "finish_reason": reason}},
					})
					return
				}
				if t.Done {
					reason = t.Reason
					continue
				}
				text += t.Text
			case <-r.Context().Done():
				return
			}
		}
	})

	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		log.Printf("listening on %s (continuous batching, %d slots)", *addr, m.NSlots)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Println("shutting down")
	cancel()
	sc, c2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer c2()
	_ = srv.Shutdown(sc)
}

func streamSSE(w http.ResponseWriter, r *http.Request, req *coli.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, 500, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	for {
		select {
		case e := <-req.Err:
			fmt.Fprintf(w, "data: {\"error\":%q}\n\n", e.Error())
			fl.Flush()
			return
		case t, ok := <-req.Out:
			if !ok {
				fmt.Fprint(w, "data: [DONE]\n\n")
				fl.Flush()
				return
			}
			if t.Done {
				continue
			}
			b, _ := json.Marshal(map[string]any{
				"choices": []any{map[string]any{"index": 0, "text": t.Text}}})
			fmt.Fprintf(w, "data: %s\n\n", b)
			fl.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": map[string]any{"message": msg}})
}
