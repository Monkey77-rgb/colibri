/* backend.c — backend registry and the declining stubs (2026-09-14).
 *
 * Compiled into EVERY build, including the CPU-only `coli`, so model.cpp can
 * always call coli_backend_open() and get "no backend in this build" as an
 * answer instead of a link error. Which constructors exist is decided by the
 * build flags: COLI_HAVE_VK (backend_vk.c), COLI_HAVE_CUDA (backend_cuda.cu);
 * torch is never linked -- it is dlopen'ed from libcoli_torch.so so the engine
 * binary carries no libtorch dependency and still runs on the 780M.
 */
#define _POSIX_C_SOURCE 200809L   /* readlink under -std=c99 */
#include "backend.h"
#ifdef COLI_HAVE_VK
#include "vk_backend.h"
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

/* One stub per entry, generated from the same list as the struct. `(void)ctx`
 * keeps -Wunused quiet; the return is the DECLINE column of COLI_BE_FUNCS. */
#define X(r, n, P, A, D) static r stub_##n P { (void)ctx; return (r)(D); }
COLI_BE_FUNCS(X)
#undef X

static void fill_defaults(coli_backend *be) {
#define X(r, n, P, A, D) if (!be->n) be->n = stub_##n;
    COLI_BE_FUNCS(X)
#undef X
}

const char *coli_backend_built(void) {
    static char s[64];
    s[0] = 0;
#ifdef COLI_HAVE_VK
    strcat(s, "vulkan");
#endif
#ifdef COLI_HAVE_CUDA
    if (s[0]) strcat(s, ",");
    strcat(s, "cuda");
#endif
    if (s[0]) strcat(s, ",");
    strcat(s, "torch(plugin)");
    return s;
}

typedef coli_backend *(*open_fn)(char *, size_t);

/* The torch plugin: found next to the binary, in COLI_PLUGIN_DIR, or on the
 * default dlopen path. Kept open for the process lifetime (the table's function
 * pointers live in it). */
static coli_backend *open_torch(char *err, size_t cap) {
    const char *dir = getenv("COLI_PLUGIN_DIR");
    char path[2048]; void *h = NULL;
    if (dir) { snprintf(path, sizeof path, "%s/libcoli_torch.so", dir); h = dlopen(path, RTLD_NOW|RTLD_LOCAL); }
    if (!h) {
        char exe[1024]; ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) { exe[n] = 0; char *sl = strrchr(exe, '/'); if (sl) { *sl = 0;
            snprintf(path, sizeof path, "%s/libcoli_torch.so", exe); h = dlopen(path, RTLD_NOW|RTLD_LOCAL); } }
    }
    if (!h) h = dlopen("libcoli_torch.so", RTLD_NOW|RTLD_LOCAL);
    if (!h) { snprintf(err, cap, "torch: libcoli_torch.so not found (%s)", dlerror()); return NULL; }
    open_fn f = (open_fn)dlsym(h, "coli_backend_torch_open");
    if (!f) { snprintf(err, cap, "torch: plugin lacks coli_backend_torch_open"); dlclose(h); return NULL; }
    return f(err, cap);
}

static coli_backend *open_one(const char *name, char *err, size_t cap) {
    coli_backend *be = NULL;
    if (!strcmp(name, "vulkan")) {
#ifdef COLI_HAVE_VK
        be = coli_backend_vk_open(err, cap);
#else
        snprintf(err, cap, "vulkan: not in this build");
#endif
    } else if (!strcmp(name, "cuda")) {
#ifdef COLI_HAVE_CUDA
        be = coli_backend_cuda_open(err, cap);
#else
        snprintf(err, cap, "cuda: not in this build");
#endif
    } else if (!strcmp(name, "torch")) {
        be = open_torch(err, cap);
    } else {
        snprintf(err, cap, "unknown backend '%s'", name);
    }
    if (be) { be->name = name; fill_defaults(be); }
    return be;
}

coli_backend *coli_backend_open(const char *which, char *err, size_t errcap) {
    if (!which || !*which) which = "auto";
    if (strcmp(which, "auto") != 0) return open_one(which, err, errcap);
    const char *order = getenv("COLI_BACKEND_ORDER");
    if (!order || !*order) order = "cuda,vulkan,torch";
    char buf[128]; snprintf(buf, sizeof buf, "%s", order);
    char why[512] = {0}; size_t wl = 0;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        char e[256] = {0};
        coli_backend *be = open_one(tok, e, sizeof e);
        if (be) return be;
        wl += (size_t)snprintf(why + wl, sizeof why - wl, "%s%s", wl ? "; " : "", e);
        if (wl >= sizeof why) break;
    }
    snprintf(err, errcap, "no backend opened: %s", why);
    return NULL;
}

void coli_backend_close(coli_backend *be) {
    if (!be) return;
    if (be->close) be->close(be->ctx);
    free(be);
}

int coli_backend_probe_class(const char *name) {
#ifdef COLI_HAVE_VK
    if (!strcmp(name, "vulkan")) return coli_vk_probe_class("shaders/gemm_i8.spv");
#endif
#ifdef COLI_HAVE_CUDA
    if (!strcmp(name, "cuda")) return coli_cuda_probe_class();
#endif
    (void)name;
    return -1;
}
