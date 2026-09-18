/* moe_profile.h -- COLI_MOE_PROFILE parsing AND validation (2026-09-18), pure
 * bookkeeping over integers so tests/test_moe_profile.c can exercise it
 * without a model load, same split as moe_slot_cache.h.
 *
 * FORMAT. "<layer> <e_hot> ... <e_cold>" per line, one line per layer, each
 * line listing EVERY expert of that layer in rank order (hottest first) --
 * exactly what tools/moe_profile.py has always written. An OPTIONAL first
 * line "# coli-moe-profile model=<16 hex digits> layers=<L> experts=<E>"
 * pins the profile to the GGUF it was built for; model is the loaded file's
 * coli_gguf_meta_hash() (loader.c/loader.h) formatted as 16 lowercase hex
 * digits -- reused rather than inventing a second hashing scheme, since that
 * hash already exists for exactly this purpose (w4snap identity, model.cpp).
 *
 * MEASURED MOTIVATION (PEER_REPORT, astra02_results.md section D /
 * HANDOFF_astra_2026-09-17_dispatch2.md section D): COLI_MOE_PROFILE today
 * validates nothing about which model a profile names. A profile built for
 * one GGUF loaded against a different one (different layer count, expert
 * count, or simply the wrong model) produces a plausible-looking but wrong
 * slot assignment with no error -- the loop below silently drops any layer
 * index >= NL and silently truncates/ignores any expert index >= NE.
 *
 * VALIDATION, applied to every profile whether or not it carries a header:
 *   (a) every layer 0..NL-1 must appear in the file EXACTLY once
 *   (b) every layer's line must list EXACTLY NE expert indices, each in
 *       [0,NE), with no duplicate within that line
 * A header additionally requires its layers/experts fields to equal NL/NE
 * and, when model_hash is nonzero (the caller could compute one), its model
 * field to match. ANY failure leaves prof[] COMPLETELY UNTOUCHED (the caller
 * pre-seeds it, normally identity order) and returns 0 with a reason string
 * -- no crash, no silent partial pin. Headerless profiles that otherwise
 * validate are accepted with had_header=0, so the caller can print the
 * "unvalidated legacy profile" notice. */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* prof must already hold NL*NE ints (the caller's identity-order seed).
 * On success prof[l*NE+r] = the r-th hottest expert of layer l for every l.
 * *had_header reports whether the recognised header line was present
 * (valid or not -- callers only read it after a successful parse). */
static inline int coli_moe_profile_parse(const char *path, int NL, int NE,
                                          uint64_t model_hash, int *prof,
                                          int *had_header, char *reason, size_t reason_cap) {
    if (had_header) *had_header = 0;
#define PFAIL(...) do { if (reason && reason_cap) snprintf(reason, reason_cap, __VA_ARGS__); return 0; } while (0)
    if (NL <= 0 || NE <= 0) PFAIL("model has no layers/experts (NL=%d NE=%d)", NL, NE);
    FILE *f = fopen(path, "r");
    if (!f) PFAIL("cannot open %s", path);

    char line[65536];
    int *tmp = (int*)malloc((size_t)NL * (size_t)NE * sizeof(int));
    char *seen_layer = (char*)calloc((size_t)NL, 1);
    if (!tmp || !seen_layer) { fclose(f); free(tmp); free(seen_layer); PFAIL("out of memory"); }

    int header_layers = -1, header_experts = -1;
    uint64_t header_hash = 0; int header_has_hash = 0;
    int first_line = 1, ok = 1; char why[256] = "";

    while (ok && fgets(line, sizeof line, f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (first_line && p[0] == '#') {
            first_line = 0;
            if (!strncmp(p, "# coli-moe-profile", 18)) {
                if (had_header) *had_header = 1;
                const char *ml = strstr(p, "model=");
                const char *ly = strstr(p, "layers=");
                const char *ex = strstr(p, "experts=");
                if (ml) header_hash = strtoull(ml + 6, NULL, 16), header_has_hash = 1;
                if (ly) header_layers = (int)strtol(ly + 7, NULL, 10);
                if (ex) header_experts = (int)strtol(ex + 8, NULL, 10);
            }
            continue;   /* header line (recognised or not) never carries layer data */
        }
        first_line = 0;
        if (p[0] == '\n' || p[0] == '\0' || p[0] == '#') continue;   /* blank/comment: skip */
        char *end;
        long l = strtol(p, &end, 10);
        if (end == p) continue;                       /* no leading integer: not a data line */
        if (l < 0 || l >= NL) { snprintf(why, sizeof why, "layer index %ld out of range [0,%d)", l, NL); ok = 0; break; }
        if (seen_layer[l]) { snprintf(why, sizeof why, "layer %ld appears more than once", l); ok = 0; break; }
        seen_layer[l] = 1;
        p = end;
        char *row_seen = (char*)calloc((size_t)NE, 1);
        if (!row_seen) { snprintf(why, sizeof why, "out of memory"); ok = 0; break; }
        int r = 0;
        while (1) {
            long e = strtol(p, &end, 10);
            if (end == p) break;
            p = end;
            if (e < 0 || e >= NE) { snprintf(why, sizeof why, "layer %ld: expert index %ld out of range [0,%d)", l, e, NE); ok = 0; break; }
            if (row_seen[e]) { snprintf(why, sizeof why, "layer %ld: expert %ld listed more than once", l, e); ok = 0; break; }
            if (r >= NE) { snprintf(why, sizeof why, "layer %ld lists more than %d experts", l, NE); ok = 0; break; }
            row_seen[e] = 1;
            tmp[(size_t)l * NE + r] = (int)e;
            r++;
        }
        free(row_seen);
        if (!ok) break;
        if (r != NE) { snprintf(why, sizeof why, "layer %ld lists %d experts, expected %d", l, r, NE); ok = 0; break; }
    }
    fclose(f);
    if (ok) {
        for (int l = 0; l < NL; l++) if (!seen_layer[l]) { snprintf(why, sizeof why, "layer %d is missing from the profile", l); ok = 0; break; }
    }
    if (ok && header_layers >= 0 && header_layers != NL) { snprintf(why, sizeof why, "header layers=%d, loaded model has %d", header_layers, NL); ok = 0; }
    if (ok && header_experts >= 0 && header_experts != NE) { snprintf(why, sizeof why, "header experts=%d, loaded model has %d", header_experts, NE); ok = 0; }
    if (ok && header_has_hash && model_hash != 0 && header_hash != model_hash) {
        snprintf(why, sizeof why, "header model=%016llx does not match the loaded GGUF (%016llx)",
                 (unsigned long long)header_hash, (unsigned long long)model_hash);
        ok = 0;
    }
    free(seen_layer);
    if (!ok) { free(tmp); if (reason && reason_cap) snprintf(reason, reason_cap, "%s: %s", path, why); return 0; }
    memcpy(prof, tmp, (size_t)NL * (size_t)NE * sizeof(int));
    free(tmp);
    return 1;
#undef PFAIL
}
