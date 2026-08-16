/* test_tok_gguf — encode stdin with the tokenizer read out of a GGUF file.
 *
 * Not a self-contained gate, and deliberately not given a Makefile rule (see
 * the TEST_RULES comment in the Makefile: "having a rule is the honest
 * definition of a gate"). It needs a real multi-GB model file, which the repo
 * does not and should not carry. It is the OUR-SIDE half of the differential
 * driven by tests/tok_gguf_diff.sh, whose other half is llama.cpp's own
 * llama-tokenize on the same bytes.
 *
 * Build:  cc -O2 -I. -o tests/test_tok_gguf tests/test_tok_gguf.c -lm
 * Use:    tests/test_tok_gguf model.gguf < input.txt      (ids, one per line)
 *
 * NEGCTL perturbs OUR tokenizer after load, which is the only kind of control
 * that can fail a differential test -- corrupting the input file feeds both
 * sides the same wrong bytes and they agree anyway. See the header comment in
 * tok_gguf.h for the measured matrix, including the one control (digits, on a
 * qwen vocab) that is structurally INERT and must not be counted as a pass.
 */
#include "gguf_reader.h"
#include "gguf_meta.h"
#include "tok.h"
#include "tok_gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> < text\n", argv[0]); return 2; }
    Tok T; int bos, eos, ab;
    tok_load_gguf(&T, argv[1], &bos, &eos, &ab);

    /* Empty is unset. getenv returns "" for NEGCTL= , which used to fall into the
     * unknown-control branch and exit 2 with no output -- an unperturbed run that
     * could not pass, the mirror image of a control that could not fail. */
    const char *ng = getenv("NEGCTL");
    if (ng && *ng) {
        if (!strcmp(ng, "digits"))         T.n_digits = (T.n_digits == 1) ? 3 : 1;
        else if (!strcmp(ng, "o200k"))     T.o200k = !T.o200k;
        else if (!strcmp(ng, "nospecial")) T.nsp = 0;
        else if (!strcmp(ng, "mergerank")) {       /* invert merge priority entirely */
            int mx = 0;
            for (int i = 0; i < T.merges.cap; i++)
                if (T.merges.e[i].used && T.merges.e[i].v > mx) mx = T.merges.e[i].v;
            for (int i = 0; i < T.merges.cap; i++)
                if (T.merges.e[i].used) T.merges.e[i].v = mx - T.merges.e[i].v;
        } else { fprintf(stderr, "unknown NEGCTL %s\n", ng); return 2; }
        fprintf(stderr, "# NEGCTL=%s applied\n", ng);
    }
    fprintf(stderr, "# n_ids=%d nsp=%d o200k=%d n_digits=%d bos=%d eos=%d add_bos=%d\n",
            T.n_ids, T.nsp, T.o200k, T.n_digits, bos, eos, ab);

    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf - 1, stdin);
    buf[n] = 0;

    static int ids[1 << 18];
    int k = tok_encode(&T, buf, (int)n, ids, 1 << 18);
    for (int i = 0; i < k; i++) printf("%d\n", ids[i]);
    fprintf(stderr, "# %d tokens\n", k);

    /* Round-trip must reproduce the input byte for byte. Independent of the
     * reference: it catches a decode bug that encode-only comparison cannot. */
    static char rt[1 << 20];
    int m = tok_decode(&T, ids, k, rt, sizeof rt - 1);
    if (m != (int)n || memcmp(rt, buf, n))
        fprintf(stderr, "# ROUNDTRIP MISMATCH (%d bytes out vs %zu in)\n", m, n);
    else
        fprintf(stderr, "# roundtrip OK (%d bytes)\n", m);

    tok_free(&T);
    return 0;
}
