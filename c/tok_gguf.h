/* tok_gguf.h — build a Tok from a GGUF file's own tokenizer metadata.
 *
 * WHY. dense.c could load weights from a GGUF and run a forward pass, but took
 * token ids in and gave ids out, because tok.h loads a HuggingFace
 * tokenizer.json that has to be fetched separately and kept in sync with the
 * weights by hand. The vocab and merges are already inside the GGUF; a model
 * file that carries its own tokenizer cannot drift from it.
 *
 * WHAT IS REUSED, AND WHAT IS NOT. Everything downstream of loading is tok.h's:
 * the GPT-2 byte<->unicode map, bpe_piece, the pre-tokenizers, tok_encode,
 * tok_decode. This header only fills the same `Tok` struct from a different
 * source. GGUF stores `tokenizer.ggml.tokens` in exactly the byte-level unicode
 * form tokenizer.json's model.vocab uses -- verified by probe, not assumed:
 * token 259 in qwen2.5-3b is the 3 bytes C4 A0 74, i.e. U+0120 'Ġ' + 't'. So
 * the BPE machinery needs no change at all.
 *
 * THE ONE REAL DIFFERENCE, and it is not cosmetic. llama.cpp's QWEN2
 * pre-tokenizer uses a bare `\p{N}` -- ONE digit per chunk -- where the cl100k
 * regex tok.h was written against uses `\p{N}{1,3}`. Getting this wrong does not
 * fail loudly: it silently emits a different token sequence for every
 * multi-digit number in the prompt, which then reads as a model quality
 * problem. Hence Tok.n_digits.
 *
 * Both fleet families are otherwise the plain cl100k shape tok.h already has
 * (`\p{L}+`, rule D tail `[\r\n]*`) -- NOT the o200k case-split shape:
 *   pre=qwen2      cl100k, n_digits=1   (llama-vocab.cpp:371 QWEN2, with
 *                                        STABLELM2 / HUNYUAN / SOLAR_OPEN)
 *   pre=llama-bpe  cl100k, n_digits=3   (llama-vocab.cpp:283 LLAMA3)
 *
 * ⚠ I got llama-bpe wrong first time round, by reading a regex string a few
 * dozen lines below the LLAMA3 case label -- it belonged to PRE_TYPE_YOUTU, and
 * it *was* the o200k shape with a bare \p{N}, so it looked like a confirmation.
 * The corpus differential against llama-tokenize caught it: 16/25, and every
 * failure was a number. Read to the case label, not to the nearest plausible
 * string, and let the differential arbitrate. All figures below are from that
 * differential, not from the source read.
 *
 * REFUSE, DO NOT GUESS. Any tokenizer.ggml.model other than "gpt2", or any
 * tokenizer.ggml.pre outside the verified table above, aborts. tok.h has no
 * implementation of, say, the plain GPT2 pre-tokenizer, and quietly falling back
 * to the nearest family would produce a tokenizer that works on prose and
 * diverges on punctuation -- the worst possible failure shape. Same discipline
 * as dense.c refusing an architecture it has not been checked against.
 *
 * MEASURED, 2026-08-16, tests/tok_gguf_diff.sh against llama.cpp's own
 * llama-tokenize (/opt/llama-cpp) on 27 inputs x 3 fleet models -- prose, code,
 * JSON, markdown, URLs, CJK, emoji, combining marks, smart quotes, control
 * chars, chat markup, and ~75 KB of real source and docs:
 *
 *   SHIPPED        81/81 exact, 49,344 reference tokens, 0 round-trip failures
 *   NEG:o200k       3 fail/model   NEG:nospecial  1-2 fail/model
 *   NEG:mergerank  11-16 fail/model
 *   NEG:digits     10 fail on llama-bpe;  INERT on qwen2 -- see below
 *
 * ⚠ NEG:digits CANNOT fail on a qwen2 vocab, and its 27/27 is NOT a pass. That
 * vocab contains 0 multi-digit tokens (measured: 151,936 tokens, 151,936
 * distinct, 0 multi-digit; llama-bpe has 1,100), so widening the \p{N} bound
 * changes the chunk boundary and BPE decomposes it straight back to single
 * digits. Same shape as nomic-embed's Q4_K control on a file with no K-quant
 * tensors: a control that is structurally incapable of firing.
 *
 * UNTRUSTED INPUT. A GGUF is untrusted. Array lengths are bounds-checked by
 * gguf_meta.h; on top of that this file compares the count each array actually
 * DELIVERED against the count it DECLARED. A truncated vocab that stops at
 * 90,000 of 151,936 tokens would otherwise load, encode, and be wrong only for
 * the missing tail.
 */
#ifndef COLI_TOK_GGUF_H
#define COLI_TOK_GGUF_H

#ifndef COLI_GGUF_META_H
#error "include gguf_reader.h and gguf_meta.h before tok_gguf.h"
#endif

/* GGUF token_type values (llama.cpp llama_token_attr / TOKEN_TYPE_*) */
#define TG_NORMAL 1
#define TG_UNKNOWN 2
#define TG_CONTROL 3
#define TG_USER_DEFINED 4
#define TG_UNUSED 5
#define TG_BYTE 6

typedef struct { Tok *T; uint64_t n_set; } tg_ctx;

static int tg_put_token(uint64_t i, const char *s, uint32_t len, void *user) {
    tg_ctx *c = (tg_ctx *)user;
    if (i >= (uint64_t)c->T->n_ids) return 0;
    char *dup = (char *)malloc((size_t)len + 1);
    if (!dup) { fprintf(stderr, "tok_gguf: OOM at token %llu\n", (unsigned long long)i); exit(1); }
    memcpy(dup, s, len); dup[len] = 0;
    /* A vocab holding the same piece twice would leave hm_get pointing at
     * whichever won. Take the FIRST id, matching how tokenizer.json's object
     * keys behave. Defensive only: measured 0 duplicates in both fleet vocabs
     * (151,936 tokens / 151,936 distinct, and 128,256 / 128,256), so this branch
     * is NOT exercised by the differential below -- do not read the 84/84 as
     * covering it. */
    if (c->T->id2str[i]) free(c->T->id2str[i]);
    c->T->id2str[i] = dup;
    if (hm_get(&c->T->vocab, dup, (int)len) < 0)
        hm_put(&c->T->vocab, dup, (int)len, (int)i);
    c->n_set++;
    return 1;
}

static int tg_put_merge(uint64_t i, const char *s, uint32_t len, void *user) {
    Tok *T = (Tok *)user;
    /* "left right". Neither side can contain a space: in byte-level form a real
     * space is U+0120, never 0x20. So a second space means this is not the
     * format assumed here, and splitting at the first one would build a subtly
     * wrong merge table -- abort instead. */
    const char *sp = (const char *)memchr(s, ' ', len);
    if (!sp || sp == s || (uint32_t)(sp - s) + 1 >= len) {
        fprintf(stderr, "tok_gguf: malformed merge %llu (no usable separator)\n",
                (unsigned long long)i); exit(1);
    }
    if (memchr(sp + 1, ' ', len - (uint32_t)(sp - s) - 1)) {
        fprintf(stderr, "tok_gguf: merge %llu has more than one space\n",
                (unsigned long long)i); exit(1);
    }
    int ll = (int)(sp - s), rl = (int)(len - (uint32_t)(sp - s) - 1);
    char *key = (char *)malloc((size_t)ll + 1 + (size_t)rl);
    if (!key) { fprintf(stderr, "tok_gguf: OOM at merge %llu\n", (unsigned long long)i); exit(1); }
    memcpy(key, s, ll); key[ll] = 0; memcpy(key + ll + 1, sp + 1, rl);
    hm_put(&T->merges, key, ll + 1 + rl, (int)i);
    return 1;
}

/* Fills T from `path`. Exits on anything it cannot verify. bos/eos are returned
 * through the out params (-1 when the file does not carry them); add_bos is 1/0,
 * defaulting to 0 when the key is absent, which is what llama.cpp does for a
 * gpt2-model vocab. */
static void tok_load_gguf(Tok *T, const char *path, int *out_bos, int *out_eos, int *out_add_bos) {
    memset(T, 0, sizeof *T);
    tk_build_bytemap(T);

    GgufMeta m; char err[256];
    if (!gguf_meta_open(path, &m, err, sizeof err)) {
        fprintf(stderr, "tok_gguf: %s: %s\n", path, err); exit(1);
    }

    char model[64] = {0}, pre[64] = {0};
    if (!gguf_meta_str(&m, "tokenizer.ggml.model", model, sizeof model)) {
        fprintf(stderr, "tok_gguf: %s has no tokenizer.ggml.model\n", path); exit(1);
    }
    if (strcmp(model, "gpt2") != 0) {
        fprintf(stderr, "tok_gguf: tokenizer.ggml.model=\"%s\"; only \"gpt2\" "
                        "(byte-level BPE) is implemented here\n", model); exit(1);
    }
    if (!gguf_meta_str(&m, "tokenizer.ggml.pre", pre, sizeof pre)) snprintf(pre, sizeof pre, "(absent)");

    if (!strcmp(pre, "qwen2")) { T->o200k = 0; T->kimi = 0; T->n_digits = 1; }
    else if (!strcmp(pre, "llama-bpe")) { T->o200k = 0; T->kimi = 0; T->n_digits = 3; }
    else {
        fprintf(stderr, "tok_gguf: tokenizer.ggml.pre=\"%s\" is not one of the "
                        "pre-tokenizers verified against llama.cpp here "
                        "(qwen2, llama-bpe). Refusing to guess a family -- the "
                        "failure mode is silent divergence, not an error.\n", pre);
        exit(1);
    }

    /* ---- vocab ---- */
    GgufArr toks;
    if (!gguf_meta_arr(&m, "tokenizer.ggml.tokens", &toks) || toks.etype != G_STR) {
        fprintf(stderr, "tok_gguf: missing/!str tokenizer.ggml.tokens\n"); exit(1);
    }
    if (toks.n == 0 || toks.n > (1u << 21)) {
        fprintf(stderr, "tok_gguf: implausible vocab size %llu\n", (unsigned long long)toks.n); exit(1);
    }
    T->n_ids = (int)toks.n;
    T->owns_str = 1;
    T->id2str    = (char **)calloc((size_t)T->n_ids, sizeof(char *));
    T->id_added  = (int *)calloc((size_t)T->n_ids, sizeof(int));
    T->id_special= (int *)calloc((size_t)T->n_ids, sizeof(int));
    if (!T->id2str || !T->id_added || !T->id_special) {
        fprintf(stderr, "tok_gguf: OOM sizing %d ids\n", T->n_ids); exit(1);
    }
    int vc = 1; while (vc < T->n_ids * 2) vc <<= 1;
    hm_init(&T->vocab, vc);

    tg_ctx ctx = { T, 0 };
    uint64_t got = gguf_meta_arr_str_foreach(&m, &toks, tg_put_token, &ctx);
    if (got != toks.n || ctx.n_set != toks.n) {
        fprintf(stderr, "tok_gguf: vocab truncated -- declared %llu, delivered %llu, stored %llu\n",
                (unsigned long long)toks.n, (unsigned long long)got, (unsigned long long)ctx.n_set);
        exit(1);
    }

    /* ---- merges ---- */
    GgufArr mg;
    if (!gguf_meta_arr(&m, "tokenizer.ggml.merges", &mg) || mg.etype != G_STR || mg.n == 0) {
        /* No merges list means tiktoken-style ranking, which tok.h supports via
         * rankbpe -- but neither fleet family is built that way, so treat it as
         * a file this loader has not been checked against. */
        fprintf(stderr, "tok_gguf: no tokenizer.ggml.merges; this loader has only "
                        "been verified on merge-list vocabs\n"); exit(1);
    }
    int mc = 1; while (mc < (int)(mg.n * 2)) mc <<= 1;
    hm_init(&T->merges, mc);
    uint64_t mgot = gguf_meta_arr_str_foreach(&m, &mg, tg_put_merge, T);
    if (mgot != mg.n) {
        fprintf(stderr, "tok_gguf: merges truncated -- declared %llu, delivered %llu\n",
                (unsigned long long)mg.n, (unsigned long long)mgot); exit(1);
    }

    /* ---- token_type -> added / special, and the atomic-match list ---- */
    GgufArr tt;
    if (gguf_meta_arr(&m, "tokenizer.ggml.token_type", &tt)) {
        int32_t *ty = (int32_t *)malloc((size_t)T->n_ids * sizeof(int32_t));
        if (!ty) { fprintf(stderr, "tok_gguf: OOM for token_type\n"); exit(1); }
        uint64_t tgot = gguf_meta_arr_i32(&m, &tt, ty, (uint64_t)T->n_ids);
        if (tgot != (uint64_t)T->n_ids || tt.n != (uint64_t)T->n_ids) {
            fprintf(stderr, "tok_gguf: token_type covers %llu of %d ids (declared %llu)\n",
                    (unsigned long long)tgot, T->n_ids, (unsigned long long)tt.n); exit(1);
        }
        int nsp = 0;
        for (int i = 0; i < T->n_ids; i++)
            if (ty[i] == TG_CONTROL || ty[i] == TG_USER_DEFINED) nsp++;
        T->sp = (Special *)calloc((size_t)(nsp ? nsp : 1), sizeof(Special));
        if (!T->sp) { fprintf(stderr, "tok_gguf: OOM for %d specials\n", nsp); exit(1); }
        T->nsp = 0;
        for (int i = 0; i < T->n_ids; i++) {
            if (ty[i] != TG_CONTROL && ty[i] != TG_USER_DEFINED) continue;
            if (!T->id2str[i]) continue;
            /* CONTROL is a control token (<|im_start|>): never legitimate model
             * output content. USER_DEFINED (<tool_call>) is real text and must
             * render -- the same split tok.h draws from added_tokens[].special. */
            T->id_added[i] = 1;
            if (ty[i] == TG_CONTROL) T->id_special[i] = 1;
            T->sp[T->nsp].str = T->id2str[i];
            T->sp[T->nsp].len = (int)strlen(T->id2str[i]);
            T->sp[T->nsp].id  = i;
            T->nsp++;
        }
        qsort(T->sp, T->nsp, sizeof(Special), cmp_sp_len);   /* longest match first */
        free(ty);
    }

    /* ---- bos / eos / add_bos ---- */
    long long v;
    if (out_bos) *out_bos = gguf_meta_i64(&m, "tokenizer.ggml.bos_token_id", &v) ? (int)v : -1;
    if (out_eos) *out_eos = gguf_meta_i64(&m, "tokenizer.ggml.eos_token_id", &v) ? (int)v : -1;
    if (out_add_bos) *out_add_bos = gguf_meta_i64(&m, "tokenizer.ggml.add_bos_token", &v) ? (int)(v != 0) : 0;

    gguf_meta_close(&m);
}

#endif /* COLI_TOK_GGUF_H */
