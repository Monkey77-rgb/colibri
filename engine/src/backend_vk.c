/* backend_vk.c — the Vulkan backend behind the coli_backend seam (2026-09-14).
 *
 * Every entry is a one-line adapter from `void *ctx` to `coli_vk *`; the
 * implementations stay in vk_backend.c, and the tests keep calling them
 * directly. Generated from COLI_BE_FUNCS so an entry added to the seam without
 * a Vulkan implementation is a compile error here rather than a NULL at run time.
 */
#include "backend.h"
#include "vk_backend.h"
#include <stdlib.h>
#include <string.h>

/* CTX is the placeholder the seam uses for the first argument; here it is the
 * cast to the Vulkan handle. */
#define CTX ((coli_vk*)ctx)
#define X(r, n, P, A, D) static r vk_##n P { return coli_vk_##n A; }
COLI_BE_FUNCS(X)
#undef X
#undef CTX

static void vk_close(void *ctx) { coli_vk_free((coli_vk*)ctx); }

coli_backend *coli_backend_vk_open(char *err, size_t errcap) {
    coli_vk *v = coli_vk_init("shaders/gemm_i8.spv", err, errcap);
    if (!v) return NULL;
    coli_backend *be = (coli_backend*)calloc(1, sizeof *be);
    if (!be) { coli_vk_free(v); snprintf(err, errcap, "out of memory"); return NULL; }
    be->ctx = v; be->close = vk_close;
#define X(r, n, P, A, D) be->n = vk_##n;
    COLI_BE_FUNCS(X)
#undef X
    return be;
}
