package coli

// Chat templating for the OpenAI /v1/chat/completions surface.
//
// WHY NOT A JINJA ENGINE. llama.cpp renders the GGUF's embedded jinja
// `tokenizer.chat_template` with minja precisely because a general template can
// be anything. Banana's whole fleet is two families -- Llama-3 header format and
// ChatML -- and both are simple, fixed, and well-specified. A hand-rolled jinja
// interpreter in Go would be a large correctness surface for two strings we
// already know. So we DETECT the family from the embedded template (the string
// llama.cpp would render) and format with a checked, tested function. A model
// whose template we do not recognise is REFUSED at load, not rendered wrong --
// a wrong prompt format is invisible in the output and poisons every reply.
//
// BOS IS NOT OUR JOB HERE. The scheduler already prepends the model's BOS id to
// every request when the model asks for it (add_bos). The jinja templates put
// bos_token at the very front; if we ALSO emitted a literal <|begin_of_text|>
// the sequence would carry two, which shifts every position. So the renderers
// below emit the body WITHOUT bos, exactly as the scheduler expects, and BOS is
// added once, in one place, for both the completion and the chat paths.

/*
#cgo CFLAGS: -I${SRCDIR}/../src
#include <stdlib.h>
#include "loader.h"
*/
import "C"

import (
	"fmt"
	"strings"
	"unsafe"
)

// ChatMessage is one turn. Role is "system" | "user" | "assistant".
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Family is the recognised chat format of a model.
type Family int

const (
	FamilyUnknown Family = iota
	FamilyLlama3         // <|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>
	FamilyChatML         // <|im_start|>role\n...<|im_end|>\n
)

func (f Family) String() string {
	switch f {
	case FamilyLlama3:
		return "llama3"
	case FamilyChatML:
		return "chatml"
	default:
		return "unknown"
	}
}

// ChatConfig is what a server needs to serve chat for one model: the detected
// family and the stop strings that end an assistant turn for it.
type ChatConfig struct {
	Family Family
	Stops  []string
}

// ReadChatConfig opens the GGUF's metadata (a second, cheap open that reads no
// tensors), inspects the embedded chat template, and picks the family. It does
// NOT guess from the architecture alone: "llama" architecture covers Llama-2
// (a different template) and Llama-3, and a Qwen model may ship either its own
// ChatML or something else. The embedded template is the ground truth for what
// the model was tuned to expect, so that is what we key on -- the same string
// llama.cpp would hand to minja.
func ReadChatConfig(path string) (ChatConfig, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	var errbuf [256]C.char
	g := C.coli_gguf_open(cp, &errbuf[0], C.size_t(len(errbuf)))
	if g == nil {
		return ChatConfig{}, fmt.Errorf("gguf meta open: %s", C.GoString(&errbuf[0]))
	}
	defer C.coli_gguf_close(g)

	tmpl := ggufStr(g, "tokenizer.chat_template")

	switch {
	case strings.Contains(tmpl, "<|start_header_id|>"):
		return ChatConfig{Family: FamilyLlama3, Stops: []string{"<|eot_id|>"}}, nil
	case strings.Contains(tmpl, "<|im_start|>"):
		return ChatConfig{Family: FamilyChatML, Stops: []string{"<|im_end|>"}}, nil
	case tmpl == "":
		return ChatConfig{}, fmt.Errorf("model has no tokenizer.chat_template; " +
			"cannot serve /v1/chat/completions (use /v1/completions with your own formatting)")
	default:
		// A real template we do not recognise. Refuse rather than pick a family
		// and format every reply wrong in a way the output cannot reveal.
		head := tmpl
		if len(head) > 80 {
			head = head[:80]
		}
		return ChatConfig{}, fmt.Errorf("unrecognised chat template family (starts %q); "+
			"add a renderer before serving chat for this model", head)
	}
}

// ggufStr reads one string metadata key, returning "" if absent.
//
// coli_gguf_str returns 1 on success / 0 on absence (NOT a byte length -- an
// earlier version of this function read the 1 as a length and returned the
// first byte, "{", which silently disabled chat detection for every model).
// The C side null-terminates the buffer and truncates to fit it. 16 KiB
// comfortably holds these templates, and the family signature we look for is in
// the first ~200 bytes regardless, so truncation cannot change the detection.
func ggufStr(g *C.coli_gguf, key string) string {
	ck := C.CString(key)
	defer C.free(unsafe.Pointer(ck))
	const n = 16384
	buf := make([]byte, n)
	if int(C.coli_gguf_str(g, ck, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(n))) == 0 {
		return ""
	}
	return C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
}

// RenderChat turns a message list into the exact prompt body the model was
// tuned on, WITHOUT the leading BOS (the scheduler adds it) and WITH the
// trailing "assistant" header so the model generates the reply. The output of
// these two functions is what a correctness test compares against llama.cpp's
// rendering of the same messages.
func RenderChat(f Family, msgs []ChatMessage) (string, error) {
	switch f {
	case FamilyLlama3:
		return renderLlama3(msgs), nil
	case FamilyChatML:
		return renderChatML(msgs), nil
	default:
		return "", fmt.Errorf("no renderer for family %s", f)
	}
}

// renderLlama3 matches the Llama-3.1 template:
//   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
// repeated, then an empty assistant header to prompt generation. BOS is omitted
// (added by the scheduler). Content is trimmed, matching the template's
// `content | trim`.
func renderLlama3(msgs []ChatMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString("<|start_header_id|>")
		b.WriteString(m.Role)
		b.WriteString("<|end_header_id|>\n\n")
		b.WriteString(strings.TrimSpace(m.Content))
		b.WriteString("<|eot_id|>")
	}
	b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return b.String()
}

// renderChatML matches the ChatML template used by Qwen2.5/Qwen3:
//   <|im_start|>{role}\n{content}<|im_end|>\n
// then <|im_start|>assistant\n to prompt generation. BOS omitted. ChatML does
// NOT trim content (Qwen's template emits it verbatim).
//
// KNOWN LIMITATION, verified against Qwen2.5-3B's embedded template 2026-08-28:
// when NO system message is supplied, Qwen's own template injects a default
// ("You are Qwen, created by Alibaba Cloud..."). This renderer does not — it
// emits bare ChatML. The result is still well-formed (never garbled), just
// missing that generic persona. It matters nothing for this fleet: every bot
// (Atlas, NetSec) and the harness send an EXPLICIT system message, and the
// with-system output is byte-identical to the embedded template (proven). Full
// fidelity for the no-system case needs rendering the embedded jinja itself,
// which is the documented future-work path if a bare-ChatML caller ever appears.
func renderChatML(msgs []ChatMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}
