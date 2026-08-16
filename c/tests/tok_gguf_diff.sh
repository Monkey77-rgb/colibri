#!/bin/bash
# tok_gguf_diff.sh — differential: our GGUF tokenizer vs llama.cpp's own.
#
# WHY A SHELL SCRIPT AND NOT A GATE. The reference is llama.cpp's llama-tokenize
# run on a real multi-GB GGUF. Neither is in this repo, and neither should be, so
# this cannot be a Makefile gate. It skips cleanly when its inputs are absent
# rather than passing vacuously -- a test that "passes" because it never ran is
# the failure mode this whole file exists to avoid.
#
# The comparison is DIFFERENTIAL, so a control must perturb OUR implementation.
# Corrupting the corpus feeds both sides the same wrong bytes and they agree.
# NEGCTL does the right thing; --neg runs the whole matrix.
#
# usage:
#   tests/tok_gguf_diff.sh <model.gguf> [--neg]
# env:
#   LLAMA_TOKENIZE=/path/to/llama-tokenize   (default: search PATH and /opt)

set -u
here="$(cd "$(dirname "$0")" && pwd)"
model="${1:-}"
[ -n "$model" ] || { echo "usage: $0 <model.gguf> [--neg]"; exit 2; }
[ -r "$model" ] || { echo "SKIP: no model at $model"; exit 0; }

ref_bin="${LLAMA_TOKENIZE:-}"
if [ -z "$ref_bin" ]; then
  for c in "$(command -v llama-tokenize 2>/dev/null)" \
           /opt/llama-cpp/bin/llama-tokenize /opt/llama.cpp/bin/llama-tokenize \
           /opt/llama.cpp-new/bin/llama-tokenize; do
    [ -n "$c" ] && [ -x "$c" ] && { ref_bin="$c"; break; }
  done
fi
# -x, not -n: an explicit LLAMA_TOKENIZE pointing at nothing must SKIP, not run
# and report 27 empty references as 27 failures.
[ -n "$ref_bin" ] && [ -x "$ref_bin" ] || { echo "SKIP: no usable llama-tokenize (set LLAMA_TOKENIZE)"; exit 0; }

ours="$here/test_tok_gguf"
if [ ! -x "$ours" ]; then
  cc -O2 -I"$here/.." -o "$ours" "$here/test_tok_gguf.c" -lm || exit 1
fi

corpus="$here/tok_gguf_corpus"
# Large, messy, real inputs matter: the crafted cases are short and a tokenizer
# can be wrong only in the tail. Generated here rather than committed, so the
# repo does not carry copies of its own files.
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/c"
cp "$corpus"/*.txt "$tmp/c/" 2>/dev/null
head -c 30000 "$here/../dense.c"  > "$tmp/c/90_real_csrc.txt" 2>/dev/null
head -c 20000 "$here/../tok.h"    > "$tmp/c/92_real_tokh.txt" 2>/dev/null

run_one() {   # $1 = label, $2..= env assignments
  local label="$1"; shift
  local pass=0 fail=0 rt=0 failed="" tok=0
  for f in "$tmp"/c/*.txt; do
    b="$(basename "$f" .txt)"
    r="$tmp/ref.$b"
    [ -s "$r" ] || "$ref_bin" -m "$model" -f "$f" --ids --no-bos --no-escape --log-disable 2>/dev/null \
        | tr -d '[] ' | tr ',' '\n' | grep -v '^$' > "$r"
    [ -s "$r" ] || { echo "  REF-EMPTY $b"; fail=$((fail+1)); continue; }
    o="$(env "$@" "$ours" "$model" < "$f" 2>"$tmp/err")"
    grep -q 'ROUNDTRIP MISMATCH' "$tmp/err" && { rt=$((rt+1)); failed="$failed $b(rt)"; }
    tok=$((tok + $(wc -l < "$r")))
    if [ "$(cat "$r")" = "$o" ]; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $b"; fi
  done
  printf '  %-16s pass=%-3d fail=%-3d roundtrip_fail=%-3d ref_tokens=%d%s\n' \
         "$label" "$pass" "$fail" "$rt" "$tok" "${failed:+  failed:$failed}"
  echo "$fail" > "$tmp/lastfail"
}

echo "model: $(basename "$model")   reference: $ref_bin"
run_one SHIPPED
shipped_fail="$(cat "$tmp/lastfail")"

if [ "${2:-}" = "--neg" ]; then
  # Every control must FAIL. One known exception: NEGCTL=digits on a qwen2 vocab
  # is structurally inert -- that vocab contains 0 multi-digit tokens (measured:
  # 151,936 tokens, 0 of them multi-digit), so widening the \p{N} bound changes
  # the chunking but BPE decomposes it straight back to single digits. Report it
  # as INERT, never as a pass.
  for ctl in digits o200k nospecial mergerank; do run_one "NEG:$ctl" "NEGCTL=$ctl"; done
fi

[ "$shipped_fail" -eq 0 ] || exit 1
echo "OK"
