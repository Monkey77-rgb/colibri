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
	"crypto/subtle"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
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
	w4     = flag.Int("w4", 0, "weight format: 0 int8 | 1 both, per batch | 2 int4 only (0.68x RSS, 1.42x decode, +3.87% NLL)")
	queue  = flag.Int("queue", 256, "max queued requests")
	apiKey = flag.String("api-key", "", "require this key in Authorization: Bearer <key> or X-API-Key (empty = no auth)")
	tlsCrt = flag.String("tls-cert", "", "PEM certificate; with -tls-key, serve HTTPS instead of HTTP")
	tlsKey = flag.String("tls-key", "", "PEM private key")
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

type chatReq struct {
	Messages      []coli.ChatMessage `json:"messages"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   float32            `json:"temperature"`
	TopK          int32              `json:"top_k"`
	TopP          float32            `json:"top_p"`
	MinP          float32            `json:"min_p"`
	RepeatPenalty float32            `json:"repeat_penalty"`
	Seed          uint64             `json:"seed"`
	Stop          []string           `json:"stop"`
	Stream        bool               `json:"stream"`
}

// finishReason maps the scheduler's internal reason to the OpenAI vocabulary.
// The scheduler says "stop" (EOS or a stop string) or "length"; OpenAI clients
// expect exactly "stop" or "length", so anything unexpected becomes "stop"
// rather than a value a client will not recognise.
func finishReason(r string) string {
	if r == "length" {
		return "length"
	}
	return "stop"
}

// trimStops removes a trailing stop string from the accumulated text. The
// scheduler detects a stop AFTER appending the piece that completed it, so the
// terminator can be present in the final text; a chat client should never see
// "<|eot_id|>" in message.content. Streaming has the same issue per-chunk and
// is handled separately in streamChatSSE.
func trimStops(text string, stops []string) string {
	for _, sw := range stops {
		if sw != "" && strings.HasSuffix(text, sw) {
			return text[:len(text)-len(sw)]
		}
	}
	return text
}

// auth wraps the mux with a bearer/API-key check.
//
// CONSTANT-TIME COMPARISON, not ==. A byte-by-byte string compare leaks the
// length of the matching prefix through timing, which is enough to recover a key
// one character at a time over enough requests. subtle.ConstantTimeCompare is
// the whole reason this is three lines instead of one.
//
// /health is deliberately EXEMPT: it exposes no model output and a liveness probe
// that needs a credential is a liveness probe that gets disabled. Everything
// else requires the key when one is set.
//
// Empty -api-key means no authentication AT ALL, which is the default because
// the server binds 127.0.0.1 by default. Binding it elsewhere without a key is a
// choice the operator makes, and the startup log says which one they made.
func auth(next http.Handler) http.Handler {
	if *apiKey == "" {
		return next
	}
	want := []byte(*apiKey)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			next.ServeHTTP(w, r)
			return
		}
		got := r.Header.Get("X-API-Key")
		if got == "" {
			if b := r.Header.Get("Authorization"); strings.HasPrefix(b, "Bearer ") {
				got = strings.TrimPrefix(b, "Bearer ")
			}
		}
		if subtle.ConstantTimeCompare([]byte(got), want) != 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"unauthorized"}`))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	flag.Parse()
	if *model == "" {
		flag.Usage()
		os.Exit(2)
	}
	m, err := coli.OpenW4(*model, *nCtx, *nSlots, !*f32, *w4)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	defer m.Close()
	log.Printf("%s: vocab=%d ctx=%d slots=%d", m.Arch, m.NVocab, m.NCtx, m.NSlots)
	log.Printf("kernel at batch 1: %s | at batch %d: %s", m.Kernel(1), *nSlots, m.Kernel(*nSlots))

	sch := coli.NewScheduler(m, *queue)
	ctx, cancel := context.WithCancel(context.Background())
	go sch.Run(ctx)

	// Chat config is read ONCE at startup, not per request: it opens the GGUF a
	// second time to read the embedded template. A model whose template we do
	// not recognise does not crash the server -- /v1/completions still works --
	// but /v1/chat/completions then answers 501 with the reason, rather than
	// formatting every reply wrong in a way the output cannot reveal.
	chatCfg, chatErr := coli.ReadChatConfig(*model)
	if chatErr != nil {
		log.Printf("chat: /v1/chat/completions DISABLED: %v", chatErr)
	} else {
		log.Printf("chat: family=%s stops=%v", chatCfg.Family, chatCfg.Stops)
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		st := sch.Snapshot()
		// Prefix-cache reach, not just its existence. Without this a benchmark
		// that re-sends prompts cannot tell whether it measured the engine or the
		// cache, and batch_histogram cannot answer it -- that one proves DECODE
		// batched, which is a different question entirely.
		prefixReused, prefixAsked := m.PrefixStats()
		prefixRate := 0.0
		if prefixAsked > 0 {
			prefixRate = float64(prefixReused) / float64(prefixAsked)
		}
		writeJSON(w, 200, map[string]any{
			"status": "ok", "arch": m.Arch, "slots": m.NSlots,
			"active": st.Active, "queued": st.Queued,
			"steps": st.StepsTotal, "tokens": st.TokensTotal,
			// The batch histogram, not a mean: a mean hides a server that is
			// really running at batch 1 almost all the time.
			"batch_histogram": st.BatchHist,
			"kernel_batch1":   m.Kernel(1),
			"kernel_full":     m.Kernel(m.NSlots),
			// hit_rate is reused/asked over the process lifetime.
			"prefix_reused":   prefixReused,
			"prefix_asked":    prefixAsked,
			"prefix_hit_rate": prefixRate,
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

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeErr(w, 405, "POST only")
			return
		}
		if chatErr != nil {
			writeErr(w, 501, "chat unsupported for this model: "+chatErr.Error())
			return
		}
		var q chatReq
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20)).Decode(&q); err != nil {
			writeErr(w, 400, "bad json: "+err.Error())
			return
		}
		if len(q.Messages) == 0 {
			writeErr(w, 400, "missing messages")
			return
		}
		prompt, err := coli.RenderChat(chatCfg.Family, q.Messages)
		if err != nil {
			writeErr(w, 500, err.Error())
			return
		}
		if q.MaxTokens <= 0 {
			q.MaxTokens = 512
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
		// The family's own turn terminator is ALWAYS a stop, in addition to EOS
		// (the scheduler stops on EOS by id). Without it a model that emits
		// <|eot_id|> as text rather than the EOS id would run to max_tokens,
		// bleeding the start of a hallucinated next turn into the reply. User
		// stops are added on top.
		stops := append(append([]string{}, chatCfg.Stops...), q.Stop...)
		req := &coli.Request{Prompt: prompt, MaxTokens: q.MaxTokens, Sampler: s, Stop: stops}
		if err := sch.Submit(req); err != nil {
			writeErr(w, 503, err.Error())
			return
		}

		if q.Stream {
			streamChatSSE(w, r, req, m.Arch)
			return
		}
		var text, reason string
		for {
			select {
			case e := <-req.Err:
				writeErr(w, 500, e.Error())
				return
			case t, ok := <-req.Out:
				if !ok {
					text = trimStops(text, stops)
					writeJSON(w, 200, map[string]any{
						"object": "chat.completion", "model": m.Arch,
						"choices": []any{map[string]any{
							"index":         0,
							"message":       map[string]any{"role": "assistant", "content": text},
							"finish_reason": finishReason(reason),
						}},
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

	srv := &http.Server{Addr: *addr, Handler: auth(mux), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		log.Printf("listening on %s (continuous batching, %d slots)", *addr, m.NSlots)
		var err error
		if *tlsCrt != "" || *tlsKey != "" {
			if *tlsCrt == "" || *tlsKey == "" {
				log.Fatal("-tls-cert and -tls-key must be given together")
			}
			log.Printf("serving HTTPS on %s", *addr)
			err = srv.ListenAndServeTLS(*tlsCrt, *tlsKey)
		} else {
			err = srv.ListenAndServe()
		}
		if err != nil && err != http.ErrServerClosed {
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

// streamChatSSE streams an assistant reply in OpenAI chat delta format.
//
// THE HOLDBACK, and why it is not optional. A turn terminator like <|eot_id|>
// is often a DISTINCT token from the model's EOS id, so it arrives as a normal
// text piece and is caught by the stop-STRING, not the EOS-id path -- which
// means the scheduler emits it before it knows it was a stop. In non-streaming
// we trim it off the final text. In streaming we cannot un-send a chunk, so we
// hold back the last `maxStop` bytes and only release them once we know they
// are not the start of a terminator. Without this the client sees "<|eot_id|>"
// in the middle of the visible answer.
func streamChatSSE(w http.ResponseWriter, r *http.Request, req *coli.Request, model string) {
	fl, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, 500, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)

	maxStop := 0
	for _, sw := range req.Stop {
		if len(sw) > maxStop {
			maxStop = len(sw)
		}
	}

	sendDelta := func(role, content string) {
		delta := map[string]any{}
		if role != "" {
			delta["role"] = role
		}
		if content != "" {
			delta["content"] = content
		}
		b, _ := json.Marshal(map[string]any{
			"object":  "chat.completion.chunk",
			"model":   model,
			"choices": []any{map[string]any{"index": 0, "delta": delta}},
		})
		fmt.Fprintf(w, "data: %s\n\n", b)
		fl.Flush()
	}

	sendDelta("assistant", "") // OpenAI convention: first chunk carries the role
	var pending string
	for {
		select {
		case e := <-req.Err:
			fmt.Fprintf(w, "data: {\"error\":%q}\n\n", e.Error())
			fl.Flush()
			return
		case t, ok := <-req.Out:
			if !ok {
				pending = trimStops(pending, req.Stop) // flush whatever the holdback kept
				if pending != "" {
					sendDelta("", pending)
				}
				fmt.Fprint(w, "data: [DONE]\n\n")
				fl.Flush()
				return
			}
			if t.Done {
				continue
			}
			pending += t.Text
			if len(pending) > maxStop {
				emit := pending[:len(pending)-maxStop]
				pending = pending[len(pending)-maxStop:]
				sendDelta("", emit)
			}
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
