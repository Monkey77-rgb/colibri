/* test_moe_profile -- COLI_MOE_PROFILE parser/validator (src/moe_profile.h),
 * no model load. Writes small profile files under /tmp, matching the shape
 * NL=3 NE=4 used throughout. Every property below has a companion negative
 * case (an apparatus control): a profile the parser must accept, and the
 * same profile broken one way must be refused -- so a parser that accepts
 * everything cannot pass this file silently. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "../src/moe_profile.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static const char *write_tmp(const char *name, const char *content) {
    static char path[256];
    snprintf(path, sizeof path, "/tmp/coli_moe_profile_test_%s_%d.txt", name, (int)getpid());
    FILE *f = fopen(path, "w");
    fputs(content, f);
    fclose(f);
    return path;
}

static void seed_identity(int *prof, int NL, int NE) {
    for (int l = 0; l < NL; l++) for (int r = 0; r < NE; r++) prof[l * NE + r] = r;
}

int main(void) {
    const int NL = 3, NE = 4;
    int prof[12], had_header; char reason[512];

    /* 1. legacy (headerless) profile, valid: layers 0,1,2 each a full permutation of 0..3 */
    {
        const char *p = write_tmp("legacy_ok", "0 3 2 1 0\n1 1 0 3 2\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0, prof, &had_header, reason, sizeof reason);
        CHECK(ok && !had_header, "legacy valid profile should parse: ok=%d header=%d reason=%s", ok, had_header, reason);
        CHECK(ok && prof[0*NE+0]==3 && prof[1*NE+0]==1 && prof[2*NE+0]==0, "legacy profile ranks applied correctly");
    }
    /* control: same content, one layer's line REMOVED -- must be refused ((a): layer count) */
    {
        const char *p = write_tmp("legacy_layer_removed", "0 3 2 1 0\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE); prof[1*NE+0] = 99;   /* sentinel: must NOT be touched on refusal */
        int ok = coli_moe_profile_parse(p, NL, NE, 0, prof, &had_header, reason, sizeof reason);
        CHECK(!ok, "profile with layer 1 removed must be REFUSED, got ok=%d", ok);
        CHECK(prof[1*NE+0] == 99, "prof[] must be left UNTOUCHED on refusal (no silent partial pin)");
        CHECK(!ok && strstr(reason, "missing") != NULL, "reason should mention the missing layer: %s", reason);
    }
    /* control: expert index >= NE -- must be refused ((b): expert range) */
    {
        const char *p = write_tmp("expert_oob", "0 3 2 1 0\n1 1 0 3 9\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0, prof, &had_header, reason, sizeof reason);
        CHECK(!ok, "expert index 9 >= NE(4) must be REFUSED, got ok=%d", ok);
    }
    /* control: a short row (fewer than NE experts) -- must be refused ((b): count per layer) */
    {
        const char *p = write_tmp("short_row", "0 3 2 1 0\n1 1 0\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0, prof, &had_header, reason, sizeof reason);
        CHECK(!ok, "layer with 2 experts instead of 4 must be REFUSED, got ok=%d", ok);
    }
    /* control: a duplicate expert within one row -- must be refused */
    {
        const char *p = write_tmp("dup_expert", "0 3 2 1 0\n1 1 1 3 2\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0, prof, &had_header, reason, sizeof reason);
        CHECK(!ok, "duplicate expert 1 in layer 1's row must be REFUSED, got ok=%d", ok);
    }
    /* control: cannot open the file at all -- must be refused, not crash */
    {
        seed_identity(prof, NL, NE); prof[0] = 42;
        int ok = coli_moe_profile_parse("/tmp/coli_moe_profile_test_does_not_exist", NL, NE, 0, prof, &had_header, reason, sizeof reason);
        CHECK(!ok && prof[0] == 42, "missing file must be refused with prof[] untouched, ok=%d prof[0]=%d", ok, prof[0]);
    }

    /* 2. header profile with the correct model hash/layers/experts -- accepted */
    {
        const char *p = write_tmp("header_ok",
            "# coli-moe-profile model=00000000cafebabe layers=3 experts=4\n"
            "0 3 2 1 0\n1 1 0 3 2\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0x00000000cafebabeULL, prof, &had_header, reason, sizeof reason);
        CHECK(ok && had_header, "correct-header profile should be accepted: ok=%d header=%d reason=%s", ok, had_header, reason);
    }
    /* control: header names a DIFFERENT model hash -- must be refused ((c)) */
    {
        const char *p = write_tmp("header_wrong_hash",
            "# coli-moe-profile model=1111111111111111 layers=3 experts=4\n"
            "0 3 2 1 0\n1 1 0 3 2\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0x00000000cafebabeULL, prof, &had_header, reason, sizeof reason);
        CHECK(!ok, "profile header naming a different model hash must be REFUSED, got ok=%d", ok);
    }
    /* control: header names the wrong layer count -- must be refused */
    {
        const char *p = write_tmp("header_wrong_layers",
            "# coli-moe-profile model=00000000cafebabe layers=5 experts=4\n"
            "0 3 2 1 0\n1 1 0 3 2\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0x00000000cafebabeULL, prof, &had_header, reason, sizeof reason);
        CHECK(!ok, "profile header naming layers=5 (loaded model has 3) must be REFUSED, got ok=%d", ok);
    }
    /* 3. model_hash==0 (caller could not compute one): a hash-bearing header must NOT
     * be rejected on the hash field alone -- only layers/experts still gate it. */
    {
        const char *p = write_tmp("header_hash_unavailable",
            "# coli-moe-profile model=1111111111111111 layers=3 experts=4\n"
            "0 3 2 1 0\n1 1 0 3 2\n2 0 1 2 3\n");
        seed_identity(prof, NL, NE);
        int ok = coli_moe_profile_parse(p, NL, NE, 0, prof, &had_header, reason, sizeof reason);
        CHECK(ok, "model_hash=0 (unavailable) must not block on a header hash it cannot check: ok=%d reason=%s", ok, reason);
    }

    /* apparatus control: a deliberately wrong expectation must be caught */
    { int before = fails;
      CHECK(1 == 0, "(expected failure) apparatus control");
      int caught = fails > before; fails = before;
      printf("apparatus control %s\n", caught ? "can fail" : "INERT");
      if (!caught) fails++; }

    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
