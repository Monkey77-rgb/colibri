package coli

import "testing"

// The expected strings below are the EXACT output of rendering each model's own
// embedded jinja `tokenizer.chat_template` (verified 2026-08-28 with jinja2
// against ARIAofWebsec-v6 for Llama-3 and qwen2.5-3b-instruct for ChatML),
// MINUS the leading bos_token — which the scheduler prepends, so the renderers
// must NOT emit it. If a change to the renderers breaks byte-parity with the
// model's own template, the model sees a prompt shape it was not tuned on and
// every reply degrades in a way the output cannot reveal. That is what this
// test exists to catch.

func TestRenderLlama3(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "system", Content: "You are ARIAofWebSec."},
		{Role: "user", Content: "What is SSRF?"},
	}
	want := "<|start_header_id|>system<|end_header_id|>\n\nYou are ARIAofWebSec.<|eot_id|>" +
		"<|start_header_id|>user<|end_header_id|>\n\nWhat is SSRF?<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	if got := renderLlama3(msgs); got != want {
		t.Fatalf("llama3 render mismatch:\n got %q\nwant %q", got, want)
	}
}

// Llama-3's template applies `content | trim`; the renderer must too, or a
// prompt with trailing whitespace tokenizes differently from the reference.
func TestRenderLlama3Trims(t *testing.T) {
	msgs := []ChatMessage{{Role: "user", Content: "  hi  \n"}}
	want := "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	if got := renderLlama3(msgs); got != want {
		t.Fatalf("llama3 trim mismatch:\n got %q\nwant %q", got, want)
	}
}

func TestRenderChatML(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "system", Content: "You are Atlas."},
		{Role: "user", Content: "Hi there"},
	}
	want := "<|im_start|>system\nYou are Atlas.<|im_end|>\n" +
		"<|im_start|>user\nHi there<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got := renderChatML(msgs); got != want {
		t.Fatalf("chatml render mismatch:\n got %q\nwant %q", got, want)
	}
}

// ChatML must NOT trim (Qwen emits content verbatim) — the inverse of the
// Llama-3 case, and a real way the two families' renderers could be wrongly
// unified.
func TestRenderChatMLNoTrim(t *testing.T) {
	msgs := []ChatMessage{{Role: "user", Content: "  spaced  "}}
	want := "<|im_start|>user\n  spaced  <|im_end|>\n<|im_start|>assistant\n"
	if got := renderChatML(msgs); got != want {
		t.Fatalf("chatml no-trim mismatch:\n got %q\nwant %q", got, want)
	}
}
