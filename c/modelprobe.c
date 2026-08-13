/* modelprobe.c — format-agnostic model introspection.
 *
 * WHY THIS EXISTS
 * ---------------
 * hwprobe answers "what can this machine do?". This answers the other half: "what does
 * this model need?" -- and it has to answer it for a model in whatever form it arrives.
 * An engine that only understands the one layout its author tested is an engine that runs
 * one model. So this reads the formats models actually ship in and reports a single
 * normalized description that a planner can act on.
 *
 * SUPPORTED INPUTS
 *   - GGUF file            (llama.cpp / ollama lineage) -- binary header parsed here
 *   - safetensors file     (raw tensor container; JSON header)
 *   - HF model directory   (config.json + *.safetensors shards)
 *   - a directory holding  *.gguf
 *
 * WHAT IT NORMALIZES TO
 *   architecture, dense vs MoE, layers, experts/layer, experts activated per token,
 *   hidden size, expert FFN width, bytes per expert, total expert bytes, weights on disk.
 * From those, the streaming demand per token follows -- which is the number that decides
 * whether a model is runnable on a given machine, and it is the number nobody publishes.
 *
 * PARSING UNTRUSTED INPUT
 * A model file is untrusted input from the internet. Every length here is bounds-checked
 * against the actual file size before it is used, string lengths are capped, and counts
 * are capped. This file was written immediately after fixing a heap out-of-bounds read in
 * this project's own JSON parser reachable from a truncated model header; the lesson is
 * applied rather than merely noted.
 *
 * BUILD:  cc -O2 -std=c99 -o modelprobe modelprobe.c
 * USAGE:  ./modelprobe [--json] PATH [PATH...]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <limits.h>
#include <time.h>

#include "json.h"          /* reused deliberately: same parser the engines use, now hardened */

#ifndef NAME_MAX
#  define NAME_MAX 255
#endif

/* GGUF_MAGIC/MAX_KV/MAX_STRLEN/MAX_ARRLEN now come from gguf_reader.h (below) */
#define MAX_HDR      (256u << 20)         /* refuse absurd safetensors headers */

typedef enum { FMT_UNKNOWN, FMT_GGUF, FMT_SAFETENSORS, FMT_HFDIR, FMT_GGUFDIR } Fmt;

typedef struct {
    char      path[PATH_MAX];
    Fmt       fmt;
    char      arch[128];
    char      quant[32];
    long long layers;
    long long experts;          /* routed experts per layer; 0 => dense */
    long long experts_active;   /* routed experts used per token */
    long long shared_experts;
    long long hidden;
    long long expert_ffn;       /* moe_intermediate_size */
    long long dense_ffn;
    long long bytes_on_disk;
    long long expert_bank_exact;   /* summed from the tensor directory; 0 = not available */
    long long tensor_bytes_total;
    long long trunk_bytes;      /* always-live weights: everything that is not a routed expert */
    long long decode_bytes;     /* bytes read to produce ONE token; 0 = could not determine */
    int       exps_tensors;
    int       tied_embd;        /* embedding table doubles as the output head */
    int       ok;
    char      note[256];
} Model;

/* ------------------------------------------------------------------ io */

static long long file_size(const char *p) {
    struct stat st;
    if (stat(p, &st) != 0) return -1;
    return (long long)st.st_size;
}

static long long dir_size_of(const char *dir, const char *suffix, int *count) {
    DIR *d = opendir(dir);
    if (!d) return -1;
    long long tot = 0; int n = 0;
    struct dirent *e;
    char p[PATH_MAX];
    size_t slen = strlen(suffix);
    while ((e = readdir(d))) {
        size_t nl = strlen(e->d_name);
        if (nl < slen || strcmp(e->d_name + nl - slen, suffix) != 0) continue;
        if (snprintf(p, sizeof p, "%s/%s", dir, e->d_name) >= (int)sizeof p) continue;
        long long s = file_size(p);
        if (s > 0) { tot += s; n++; }
    }
    closedir(d);
    if (count) *count = n;
    return n ? tot : -1;
}

/* -------------------------------------------------------------- gguf
 *
 * GGUF_MAGIC/MAX_KV/MAX_STRLEN/MAX_ARRLEN, the G_* value-type enum, gguf_str(),
 * gguf_skip(), gguf_read_int(), the GgmlType table and ggml_type() moved to
 * gguf_reader.h (Stage 1 of the GGUF weight loader) so a retaining tensor index
 * (GgufIndex / gguf_index_open()) can reuse them without duplicating this already-
 * hardened bounds-checked parsing. Behaviour here is unchanged: same tables, same
 * checks, same read_at -> gguf_read_at rename (identical pread wrapper, renamed to
 * avoid colliding with a second private copy). See gguf_reader.h for the tensor-
 * offset math (GGML_PAD / general.alignment) this file does not need. */
#include "gguf_reader.h"

/* Map a GGUF general.file_type enum to a human name. Only the common ones; anything
 * else is reported by number rather than guessed at. */
static void gguf_ftype_name(long long ft, char *out, size_t n) {
    static const struct { int v; const char *s; } m[] = {
        {0,"F32"},{1,"F16"},{2,"Q4_0"},{3,"Q4_1"},{7,"Q8_0"},{8,"Q5_0"},{9,"Q5_1"},
        {10,"Q2_K"},{11,"Q3_K_S"},{12,"Q3_K_M"},{13,"Q3_K_L"},{14,"Q4_K_S"},{15,"Q4_K_M"},
        {16,"Q5_K_S"},{17,"Q5_K_M"},{18,"Q6_K"},{19,"IQ2_XXS"},{20,"IQ2_XS"},
        {23,"IQ3_XXS"},{25,"IQ4_NL"},{26,"IQ3_S"},{28,"IQ2_S"},{30,"IQ4_XS"},{-1,NULL}
    };
    for (int i = 0; m[i].s; i++) if (m[i].v == ft) { snprintf(out,n,"%s",m[i].s); return; }
    snprintf(out, n, "ftype:%lld", ft);
}

static int probe_gguf(const char *path, Model *M) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(M->note, sizeof M->note, "cannot open"); return 0; }
    long long fsz = file_size(path);

    uint32_t magic, ver; uint64_t ntensor, nkv;
    if (fsz < 24 || !gguf_read_at(fd,&magic,4,0) || magic != GGUF_MAGIC) { close(fd); return 0; }
    if (!gguf_read_at(fd,&ver,4,4) || !gguf_read_at(fd,&ntensor,8,8) || !gguf_read_at(fd,&nkv,8,16)) {
        close(fd); snprintf(M->note,sizeof M->note,"truncated GGUF header"); return 0;
    }
    if (nkv > MAX_KV) {
        close(fd); snprintf(M->note,sizeof M->note,"implausible kv count %llu",(unsigned long long)nkv);
        return 0;
    }

    M->fmt = FMT_GGUF;
    M->bytes_on_disk = fsz;
    snprintf(M->quant, sizeof M->quant, "unknown");

    long long off = 24, ftype = -1;
    int kv_ok = 1;
    char key[256];
    /* Architecture-prefixed keys are only known after we read general.architecture, and
     * it is not guaranteed to come first -- so stash the interesting suffixes as we go. */
    long long v_block=-1, v_expcnt=-1, v_expused=-1, v_embd=-1, v_ffn=-1, v_expffn=-1, v_shared=-1;

    for (uint64_t i = 0; i < nkv; i++) {
        if (!gguf_str(fd,&off,fsz,key,sizeof key)) { snprintf(M->note,sizeof M->note,"malformed kv key at %llu",(unsigned long long)i); kv_ok = 0; break; }
        uint32_t t;
        if (off + 4 > fsz || !gguf_read_at(fd,&t,4,off)) break;
        off += 4;
        long long vpos = off;

        const char *dot = strrchr(key, '.');
        const char *suf = dot ? dot + 1 : key;

        if (!strcmp(key,"general.architecture") && t == G_STR) {
            long long tmp = off;
            /* read straight into the field: gguf_str already truncates to the buffer it
             * is given, so bouncing through sval only risks a second truncation */
            gguf_str(fd, &tmp, fsz, M->arch, sizeof M->arch);
        } else if (!strcmp(key,"general.file_type")) {
            ftype = gguf_read_int(fd,vpos,t);
        } else if (!strcmp(suf,"block_count"))            v_block  = gguf_read_int(fd,vpos,t);
        else if (!strcmp(suf,"expert_count"))             v_expcnt = gguf_read_int(fd,vpos,t);
        else if (!strcmp(suf,"expert_used_count"))        v_expused= gguf_read_int(fd,vpos,t);
        else if (!strcmp(suf,"embedding_length"))         v_embd   = gguf_read_int(fd,vpos,t);
        else if (!strcmp(suf,"feed_forward_length"))      v_ffn    = gguf_read_int(fd,vpos,t);
        else if (!strcmp(suf,"expert_feed_forward_length")) v_expffn = gguf_read_int(fd,vpos,t);
        else if (!strcmp(suf,"expert_shared_count"))      v_shared = gguf_read_int(fd,vpos,t);

        if (!gguf_skip(fd,&off,fsz,t)) { snprintf(M->note,sizeof M->note,"malformed kv value at %llu",(unsigned long long)i); kv_ok = 0; break; }
    }

    /* ---- tensor directory: the exact answer -------------------------------------
     * Only reachable if the KV section parsed cleanly, because the tensor infos start
     * wherever the KV section ended. Each entry is:
     *     name (u64 len + bytes) | n_dims u32 | dims[n_dims] u64 | type u32 | offset u64
     * Expert weights are named "...ffn_{gate,up,down}_exps.weight" and are stored as one
     * 3-D tensor per layer holding all experts, so summing every "_exps" tensor gives the
     * whole expert bank exactly, with no assumption about layer naming or count. */
    if (kv_ok && ntensor <= (1u << 22)) {
        long long exps_bytes = 0, total_bytes = 0, embd_bytes = 0;
        int n_exps_tensors = 0, unknown_type = 0, has_output_head = 0;
        char tname[512];
        for (uint64_t i = 0; i < ntensor; i++) {
            if (!gguf_str(fd,&off,fsz,tname,sizeof tname)) { snprintf(M->note,sizeof M->note,"malformed tensor name at %llu",(unsigned long long)i); break; }
            uint32_t ndim;
            if (off + 4 > fsz || !gguf_read_at(fd,&ndim,4,off)) break;
            off += 4;
            if (ndim > 8) { snprintf(M->note,sizeof M->note,"implausible tensor rank %u",ndim); break; }
            long long nelem = 1; int bad = 0;
            for (uint32_t d = 0; d < ndim; d++) {
                uint64_t dim;
                if (off + 8 > fsz || !gguf_read_at(fd,&dim,8,off)) { bad = 1; break; }
                off += 8;
                if (dim == 0 || dim > (1ULL<<40)) { bad = 1; break; }
                nelem *= (long long)dim;
                if (nelem < 0) { bad = 1; break; }      /* overflow guard */
            }
            if (bad) { snprintf(M->note,sizeof M->note,"malformed tensor dims at %llu",(unsigned long long)i); break; }
            uint32_t ttype; uint64_t toff;
            if (off + 12 > fsz || !gguf_read_at(fd,&ttype,4,off) || !gguf_read_at(fd,&toff,8,off+4)) break;
            off += 12;

            const GgmlType *g = ggml_type(ttype);
            if (!g) { unknown_type++; continue; }
            long long bytes = (nelem / g->blck) * g->bytes;
            total_bytes += bytes;
            if (strstr(tname, "_exps")) { exps_bytes += bytes; n_exps_tensors++; }
            /* The input embedding is a row lookup -- one row per token, not a matmul, so
             * its bytes do NOT enter the per-token read. UNLESS the model ties embeddings,
             * in which case the same table is also the output head and IS read in full
             * every token. Presence of a separate "output.weight" is the tell. Getting
             * this wrong is worth ~12% of the read on a 3B with a 150k vocab, which is
             * more than most of the tuning knobs people reach for first. */
            if (!strncmp(tname, "token_embd", 10)) embd_bytes = bytes;
            if (!strncmp(tname, "output.weight", 13)) has_output_head = 1;
        }
        /* Record the total unconditionally -- a dense model has no _exps tensors, and
         * the total is still the useful number (and the cross-check against file size). */
        M->tensor_bytes_total = total_bytes;
        M->exps_tensors       = n_exps_tensors;
        if (exps_bytes > 0) M->expert_bank_exact = exps_bytes;

        /* ---- bytes actually read to decode ONE token -----------------------------
         * This is the number the roofline divides into. It is not the file size.
         *
         * Dense : every weight is live every token, minus the embedding lookup
         *         (when it is not doing double duty as the output head).
         * MoE   : the shared trunk is live every token, but only experts_active of
         *         experts are touched per layer. The rest of the bank sits idle --
         *         which is the entire reason a 35B MoE and a 35B dense have wildly
         *         different ceilings on the same silicon.
         * We compute the MoE case from the EXACT expert bank, so it inherits the
         * tensor-directory accuracy rather than re-estimating. */
        long long trunk = total_bytes - exps_bytes;
        if (!has_output_head && embd_bytes > 0) {
            /* tied: table is read as the head, so it stays in the per-token figure */
        } else {
            trunk -= embd_bytes;
        }
        if (trunk < 0) trunk = 0;
        M->trunk_bytes = trunk;
        if (exps_bytes > 0 && v_expcnt > 0 && v_expused > 0) {
            /* +shared experts, which are unconditionally active when present */
            double frac = (double)(v_expused + (v_shared > 0 ? v_shared : 0)) / (double)v_expcnt;
            if (frac > 1.0) frac = 1.0;
            M->decode_bytes = trunk + (long long)((double)exps_bytes * frac);
        } else {
            M->decode_bytes = trunk;
        }
        M->tied_embd = !has_output_head && embd_bytes > 0;
        if (unknown_type)
            snprintf(M->note, sizeof M->note, "%d tensor(s) of unrecognized ggml type — sizes exclude them", unknown_type);
    }
    close(fd);

    M->layers         = v_block;
    M->experts        = v_expcnt  > 0 ? v_expcnt  : 0;
    M->experts_active = v_expused > 0 ? v_expused : 0;
    M->shared_experts = v_shared  > 0 ? v_shared  : 0;
    M->hidden         = v_embd;
    M->dense_ffn      = v_ffn;
    M->expert_ffn     = v_expffn > 0 ? v_expffn : (M->experts > 0 ? v_ffn : -1);
    if (ftype >= 0) gguf_ftype_name(ftype, M->quant, sizeof M->quant);
    if (!M->arch[0]) snprintf(M->arch,sizeof M->arch,"unknown");
    M->ok = 1;
    return 1;
}

/* ------------------------------------------------- safetensors / HF config */

static long long j_int(jval *o, const char *k, long long dflt) {
    if (!o || o->t != J_OBJ) return dflt;
    for (int i = 0; i < o->len; i++)
        if (o->keys[i] && !strcmp(o->keys[i], k)) {
            jval *v = o->kids[i];
            if (v && v->t == J_NUM) return (long long)v->num;
        }
    return dflt;
}

static const char *j_str(jval *o, const char *k) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++)
        if (o->keys[i] && !strcmp(o->keys[i], k)) {
            jval *v = o->kids[i];
            if (v && v->t == J_STR) return v->str;
            if (v && v->t == J_ARR && v->len > 0 && v->kids[0] && v->kids[0]->t == J_STR)
                return v->kids[0]->str;      /* architectures: ["X"] */
        }
    return NULL;
}

/* HF config.json. Field naming differs per model family, which is exactly why a probe is
 * needed instead of a constant -- we accept every spelling we have actually seen. */
static int probe_hf_config(const char *cfg, Model *M) {
    long long sz = file_size(cfg);
    if (sz <= 0 || sz > MAX_HDR) return 0;
    FILE *f = fopen(cfg, "rb");
    if (!f) return 0;
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return 0; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;

    jparser p; memset(&p, 0, sizeof p); p.s = buf;
    jval *root = j_parse_val(&p);
    if (!root || root->t != J_OBJ) { free(buf); return 0; }

    const char *a = j_str(root, "architectures");
    if (!a) a = j_str(root, "model_type");
    snprintf(M->arch, sizeof M->arch, "%s", a ? a : "unknown");

    M->layers  = j_int(root, "num_hidden_layers", -1);
    if (M->layers < 0) M->layers = j_int(root, "n_layer", -1);
    M->hidden  = j_int(root, "hidden_size", -1);
    if (M->hidden < 0) M->hidden = j_int(root, "n_embd", -1);

    long long e = j_int(root, "num_experts", 0);
    if (!e) e = j_int(root, "num_local_experts", 0);
    if (!e) e = j_int(root, "n_routed_experts", 0);
    M->experts = e;

    long long ea = j_int(root, "num_experts_per_tok", 0);
    if (!ea) ea = j_int(root, "num_experts_per_token", 0);
    if (!ea) ea = j_int(root, "top_k", 0);
    M->experts_active = ea;

    M->shared_experts = j_int(root, "n_shared_experts", 0);

    long long mi = j_int(root, "moe_intermediate_size", 0);
    if (!mi) mi = j_int(root, "expert_intermediate_size", 0);
    M->dense_ffn  = j_int(root, "intermediate_size", -1);
    M->expert_ffn = mi ? mi : (M->experts > 0 ? M->dense_ffn : -1);

    const char *tor = j_str(root, "torch_dtype");
    snprintf(M->quant, sizeof M->quant, "%s", tor ? tor : "unknown");

    free(buf);
    M->ok = 1;
    return 1;
}

/* --------------------------------------------------------- dispatch */

static int is_dir(const char *p) { struct stat st; return stat(p,&st)==0 && S_ISDIR(st.st_mode); }

static int probe_path(const char *path, Model *M) {
    memset(M, 0, sizeof *M);
    snprintf(M->path, sizeof M->path, "%s", path);
    M->layers = M->hidden = M->expert_ffn = M->dense_ffn = -1;
    M->bytes_on_disk = -1;

    if (is_dir(path)) {
        char cfg[PATH_MAX];
        if (snprintf(cfg, sizeof cfg, "%s/config.json", path) < (int)sizeof cfg && file_size(cfg) > 0) {
            if (probe_hf_config(cfg, M)) {
                M->fmt = FMT_HFDIR;
                int n = 0;
                long long s = dir_size_of(path, ".safetensors", &n);
                if (s > 0) { M->bytes_on_disk = s; snprintf(M->note,sizeof M->note,"%d safetensors shard(s)",n); }
                return 1;
            }
        }
        int n = 0;
        long long s = dir_size_of(path, ".gguf", &n);
        if (s > 0) {  /* probe the first gguf we find for structure, report total size */
            DIR *d = opendir(path); struct dirent *e; char p[PATH_MAX]; int done = 0;
            while (d && (e = readdir(d)) && !done) {
                size_t nl = strlen(e->d_name);
                if (nl < 5 || strcmp(e->d_name+nl-5, ".gguf")) continue;
                if (snprintf(p,sizeof p,"%s/%s",path,e->d_name) >= (int)sizeof p) continue;
                done = probe_gguf(p, M);
            }
            if (d) closedir(d);
            M->fmt = FMT_GGUFDIR;
            M->bytes_on_disk = s;
            snprintf(M->note,sizeof M->note,"%d gguf file(s)",n);
            return M->ok;
        }
        snprintf(M->note,sizeof M->note,"directory has no config.json, *.safetensors or *.gguf");
        return 0;
    }

    long long fsz = file_size(path);
    if (fsz < 0) { snprintf(M->note,sizeof M->note,"no such file"); return 0; }

    if (probe_gguf(path, M)) return 1;

    /* safetensors: u64 header length, then a JSON object */
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        uint64_t hlen;
        if (fsz > 8 && gguf_read_at(fd,&hlen,8,0) && hlen > 1 && hlen <= MAX_HDR && (long long)hlen <= fsz-8) {
            char c;
            if (gguf_read_at(fd,&c,1,8) && c == '{') {
                M->fmt = FMT_SAFETENSORS;
                M->bytes_on_disk = fsz;
                snprintf(M->note,sizeof M->note,"raw tensor container; structure lives in the sibling config.json");
                snprintf(M->arch,sizeof M->arch,"unknown");
                snprintf(M->quant,sizeof M->quant,"unknown");
                M->ok = 1;
                close(fd);
                return 1;
            }
        }
        close(fd);
    }
    snprintf(M->note,sizeof M->note,"unrecognized format");
    return 0;
}

/* ----------------------------------------------------------- report */

static const char *fmt_name(Fmt f) {
    switch (f) { case FMT_GGUF: return "gguf"; case FMT_SAFETENSORS: return "safetensors";
                 case FMT_HFDIR: return "hf-dir"; case FMT_GGUFDIR: return "gguf-dir";
                 default: return "unknown"; }
}

/* Bytes per expert for a gated FFN: gate + up + down = 3 * hidden * ffn * bytes/param.
 * bytes/param is inferred from the file when we can: total_bytes / total_params is the
 * honest estimator, but we do not have a param count for every format, so for quantized
 * GGUF we use the well-known ~0.6 B/param for Q4_K_M-class and say when we are guessing. */
static double bytes_per_param(const Model *M) {
    if (!strncmp(M->quant,"Q4",2) || !strncmp(M->quant,"IQ4",3)) return 0.60;
    if (!strncmp(M->quant,"Q5",2)) return 0.70;
    if (!strncmp(M->quant,"Q6",2)) return 0.82;
    if (!strncmp(M->quant,"Q8",2)) return 1.06;
    if (!strncmp(M->quant,"Q3",2) || !strncmp(M->quant,"IQ3",3)) return 0.46;
    if (!strncmp(M->quant,"Q2",2) || !strncmp(M->quant,"IQ2",3)) return 0.33;
    if (!strcmp(M->quant,"F16")   || !strcmp(M->quant,"bfloat16") ||
        !strcmp(M->quant,"float16")) return 2.0;
    if (!strcmp(M->quant,"F32")   || !strcmp(M->quant,"float32")) return 4.0;
    return -1.0;
}

/* Measured achievable read bandwidth, GB/s. <=0 means "not supplied, skip the roofline".
 * Set from --bw, or measured in-process by --measure-bw. We never guess it: a roofline
 * built on a spec-sheet peak is a marketing number wearing an engineering hat. */
static double g_bw_gbs = 0.0;
static const char *g_bw_src = NULL;
/* Second tier of the memory hierarchy. On a unified-memory box the interesting question
 * is not "does the model fit" but "how much of the expert bank fits", because what does
 * not fit is re-read from storage on every miss. Supplying both makes the two-tier
 * roofline computable; supplying neither just omits it. */
static double g_disk_gbs  = 0.0;   /* tier 3: storage, at the engine's real queue depth */
static double g_cache_gb  = 0.0;   /* tier 1 capacity: GPU-addressable bytes */
static double g_ram_gbs   = 0.0;   /* tier 2: host RAM / page cache read rate */
static double g_ram_gb    = 0.0;   /* tier 2 capacity: host RAM available for weights */

/* Quick in-process read-bandwidth probe. Deliberately small and short -- this may run on
 * a production host. Same shape as membw.c's read loop: multi-threaded is better, but a
 * single-threaded figure understates DRAM badly, so we only use this when the caller has
 * explicitly asked, and we label the result as a floor. */
static double measure_read_bw(void) {
    size_t n = (size_t)(256u << 20) / sizeof(double);     /* 256 MiB, well past any LLC */
    double *a = malloc(n * sizeof(double));
    if (!a) return 0.0;
    for (size_t i = 0; i < n; i++) a[i] = 1.0;            /* first touch, before timing */
    double best = 0.0;
    for (int r = 0; r < 3; r++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (size_t i = 0; i + 3 < n; i += 4) {
            s0 += a[i]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3];
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec)*1e-9;
        double gbs = (double)(n * sizeof(double)) / dt / 1e9;
        if (gbs > best) best = gbs;
        if (s0 + s1 + s2 + s3 == 1e300) return 0.0;       /* keep the sum live */
    }
    free(a);
    return best;
}

/* The whole reason the structural fields above are worth extracting.
 *
 *      tok/s_ceiling = achievable_read_bandwidth / bytes_read_per_token
 *
 * during autoregressive decode, which is memory-bound on every machine that is not
 * a datacentre accelerator. Compute, kernels, quantization format and engine choice
 * all move you toward this line; none of them move the line. Stating it explicitly is
 * the difference between "optimize the engine" and "this model cannot go faster here". */
static void print_roofline(const Model *M) {
    if (g_bw_gbs <= 0 || M->decode_bytes <= 0) return;
    double per_tok_gb = (double)M->decode_bytes / 1e9;
    double ceiling    = g_bw_gbs / per_tok_gb;

    puts("  ---- roofline ----");
    printf("  read bandwidth : %.1f GB/s   (%s)\n", g_bw_gbs, g_bw_src ? g_bw_src : "supplied");
    printf("  read/token     : %.2f GB", per_tok_gb);
    if (M->experts > 0) printf("   (trunk %.2f GB + active experts %.2f GB)",
                               (double)M->trunk_bytes/1e9,
                               (double)(M->decode_bytes - M->trunk_bytes)/1e9);
    else if (M->tied_embd) printf("   (tied embedding counted — it is the output head)");
    puts("");
    printf("  CEILING        : %.1f tok/s  — upper bound for ONE token per weight read.\n", ceiling);
    puts("                   Engine, kernel and quant changes only close the gap to it.");
    /* The honest exception, which an earlier version of this tool wrongly denied.
     * The ceiling bounds tokens-per-weight-read, not tokens-per-second. Anything that
     * amortizes one read over several tokens moves the line itself:
     *   - speculative decoding: a draft model proposes k tokens, the target verifies all
     *     k in a single forward pass. Accepted tokens are nearly free in bandwidth terms.
     *   - batching: one weight read serves N sequences. Raises throughput, NOT the latency
     *     of any single stream -- a distinction worth keeping straight when a "tok/s"
     *     number is quoted at you.
     * Neither is a free lunch: speculation pays for rejected drafts and for the draft
     * model's own reads, so the gain tracks the acceptance rate and is workload-dependent.
     * It is measured, not assumed. */
    printf("                   EXCEPT speculative decoding and batching, which amortize one\n");
    printf("                   weight read over several tokens: with acceptance rate a and\n");
    printf("                   draft length k the effective ceiling scales roughly with the\n");
    printf("                   mean accepted run, so >%.0f tok/s is reachable but must be\n", ceiling);
    puts("                   measured — the gain is entirely workload-dependent.");
    if (M->bytes_on_disk > 0)
        printf("  resident need  : %.2f GB of RAM to hold the weights\n", (double)M->bytes_on_disk/1e9);

    /* ---- two-tier: what happens when the weights do NOT all fit ------------------
     * The single-tier ceiling above silently assumes every byte is one memory read
     * away. Once the bank exceeds what the engine can hold, misses are served from
     * storage, and storage is typically an order of magnitude slower than DRAM. The
     * arithmetic below is deliberately simple and stated, not hidden:
     *
     *   t_token = trunk/BW_mem + hit_bytes/BW_mem + miss_bytes/BW_disk
     *
     * assuming misses are served at full storage bandwidth with perfect overlap-free
     * accounting -- i.e. an OPTIMISTIC treatment of the slow tier. If the numbers are
     * bad here they are worse in reality.
     *
     * This is what separates "the model fits" from "the model runs", and it is the
     * question a 20+ GB model on a 10 GB device actually poses. */
    if (g_cache_gb > 0 && g_disk_gbs > 0 && M->experts > 0 && M->expert_bank_exact > 0) {
        double trunk_gb = (double)M->trunk_bytes / 1e9;
        double bank_gb  = (double)M->expert_bank_exact / 1e9;
        double act_gb   = (double)(M->decode_bytes - M->trunk_bytes) / 1e9;

        /* Fill the hierarchy greedily: trunk first (it is live on every token, so it has
         * the strongest claim on the fastest tier), then experts spill downward.
         *
         * Modelling tier 2 separately is not a refinement -- it is the difference between
         * a right and a wrong answer on a unified-memory box. Collapsing "not in the GPU's
         * address space" into "on disk" charges host-RAM traffic at storage rates. On this
         * class of hardware those differ by more than an order of magnitude, so the
         * collapsed model understates a perfectly healthy configuration into looking
         * unusable. If tier 2 is not supplied we say so rather than silently assuming it
         * away. */
        int have_t2 = (g_ram_gbs > 0 && g_ram_gb > 0);
        double t1 = g_cache_gb - trunk_gb;           /* room left in tier 1 for experts */
        if (t1 < 0) t1 = 0;
        if (t1 > bank_gb) t1 = bank_gb;
        double t2 = have_t2 ? g_ram_gb : 0;
        if (t2 > bank_gb - t1) t2 = bank_gb - t1;
        if (t2 < 0) t2 = 0;
        double t3 = bank_gb - t1 - t2;
        if (t3 < 0) t3 = 0;

        double f1 = bank_gb > 0 ? t1 / bank_gb : 0;
        double f2 = bank_gb > 0 ? t2 / bank_gb : 0;
        double f3 = bank_gb > 0 ? t3 / bank_gb : 0;

        puts("  ---- memory hierarchy (the expert bank does not fit in one tier) ----");
        printf("  tier 1  GPU    : %6.2f GB cap @ %6.2f GB/s   holds trunk %.2f + experts %.2f GB (%.0f%% of bank)\n",
               g_cache_gb, g_bw_gbs, trunk_gb, t1, 100*f1);
        if (have_t2)
            printf("  tier 2  RAM    : %6.2f GB cap @ %6.2f GB/s   holds experts %.2f GB (%.0f%% of bank)\n",
                   g_ram_gb, g_ram_gbs, t2, 100*f2);
        else
            puts("  tier 2  RAM    : NOT SUPPLIED — everything outside tier 1 is charged at storage");
        printf("  tier 3  disk   : %6s     @ %6.2f GB/s   holds experts %.2f GB (%.0f%% of bank)\n",
               "rest", g_disk_gbs, t3, 100*f3);
        if (g_cache_gb - trunk_gb <= 0)
            puts("  VERDICT        : the trunk alone does not fit tier 1. Nothing below this helps.");
        puts("");

        /* Uniform routing means the tier fractions ARE the hit fractions. Real routers are
         * skewed, and skew is the only thing that makes a partially resident bank behave
         * better than its residency suggests -- so it is swept, not asserted. Skew is
         * modelled as pulling traffic up into tier 1 first, which is what a correct
         * hot-expert cache does; it must be MEASURED per model before any row below the
         * first is treated as achievable. */
        printf("  %-22s %10s   %s\n", "tier-1 hit rate", "tok/s", "note");
        double rates[] = {f1, 0.50, 0.80, 0.90, 0.95, 0.99, 1.00};
        for (size_t i = 0; i < sizeof rates/sizeof *rates; i++) {
            double h = rates[i];
            if (h < 0) h = 0;
            if (h > 1) h = 1;
            /* traffic above the tier-1 share spills to tier 2 then tier 3, in proportion */
            double spill = 1.0 - h;
            double rest  = f2 + f3;
            double s2 = rest > 0 ? spill * (f2 / rest) : 0;
            double s3 = rest > 0 ? spill * (f3 / rest) : spill;
            double t = trunk_gb / g_bw_gbs
                     + act_gb * h  / g_bw_gbs
                     + act_gb * s2 / (have_t2 ? g_ram_gbs : g_disk_gbs)
                     + act_gb * s3 / g_disk_gbs;
            const char *n = (i == 0) ? "<- uniform routing, no skew: today's honest number"
                          : (h >= 1.0) ? "<- all experts in tier 1 = the single-tier ceiling" : "";
            printf("  %21.0f%% %10.1f   %s\n", 100*h, t > 0 ? 1.0/t : 0, n);
        }
        puts("");
        puts("  The disk bandwidth must be the one the engine ACTUALLY achieves: a rated");
        puts("  figure is a high-queue-depth number, and a synchronous demand-paged engine");
        puts("  issues one small read at a time. Measure it with expertio at the real");
        puts("  expert size and the real prefetch depth, or this table flatters the design.");
    }
    puts("");
}

static void print_model(const Model *M) {
    printf("%s\n", M->path);
    printf("  format         : %s\n", fmt_name(M->fmt));
    if (!M->ok) { printf("  status         : NOT USABLE — %s\n\n", M->note[0]?M->note:"unrecognized"); return; }
    printf("  architecture   : %s\n", M->arch);
    printf("  precision      : %s\n", M->quant);
    if (M->bytes_on_disk > 0) printf("  weights on disk: %.2f GB\n", M->bytes_on_disk/1e9);
    if (M->layers  > 0) printf("  layers         : %lld\n", M->layers);
    if (M->hidden  > 0) printf("  hidden size    : %lld\n", M->hidden);

    if (M->experts > 0) {
        printf("  topology       : MoE — %lld experts/layer", M->experts);
        if (M->experts_active > 0) printf(", top-%lld active", M->experts_active);
        if (M->shared_experts > 0) printf(", +%lld shared", M->shared_experts);
        puts("");
        if (M->expert_ffn > 0) printf("  expert ffn     : %lld\n", M->expert_ffn);

        long long total_experts = M->experts * (M->layers > 0 ? M->layers : 1);
        double per = 0.0; const char *how = NULL;

        /* Prefer the tensor directory: exact dimensions, exact per-tensor ggml type,
         * no inference from the quantization label. Fall back to the label-derived
         * estimate only when the directory is unavailable (safetensors/HF), and say
         * which one produced the number. */
        if (M->expert_bank_exact > 0 && total_experts > 0) {
            per = (double)M->expert_bank_exact / (double)total_experts;
            how = "EXACT — summed from the tensor directory";
        } else {
            double bpp = bytes_per_param(M);
            if (M->hidden > 0 && M->expert_ffn > 0 && bpp > 0) {
                per = 3.0 * (double)M->hidden * (double)M->expert_ffn * bpp;
                how = "estimated — 3 x hidden x ffn x B/param from the precision label";
            }
        }

        if (per > 0) {
            printf("  bytes/expert   : %.2f MB   (%s)\n", per/1e6, how);
            printf("  expert bank    : %.2f GB   (%lld experts total)\n",
                   per * (double)total_experts / 1e9, total_experts);
            if (M->expert_bank_exact > 0 && M->tensor_bytes_total > 0) {
                printf("  expert share   : %.1f%% of all tensor bytes (%d _exps tensors)\n",
                       100.0 * (double)M->expert_bank_exact / (double)M->tensor_bytes_total,
                       M->exps_tensors);
            }
            if (M->experts_active > 0 && M->layers > 0) {
                long long acts = (M->experts_active + M->shared_experts) * M->layers;
                double demand = per * (double)acts;
                printf("  activations/tok: %lld\n", acts);
                printf("  STREAM DEMAND  : %.2f GB per token at 0%% cache hit\n", demand/1e9);
                printf("                   -> needs %.2f GB/s sustained for 10 tok/s cold,\n", demand*10/1e9);
                printf("                      or %.2f GB/s at a 90%% hit rate.\n", demand/1e9);
                puts("                   Cross this against hwprobe's measured disk bandwidth.");
            }
        } else {
            puts("  bytes/expert   : unknown — need hidden size, expert ffn width and precision");
        }
        puts("  engine class   : MoE — expert streaming applies. This is the case Colibri exists for.");
    } else if (M->experts == 0 && M->layers > 0) {
        printf("  topology       : DENSE");
        if (M->dense_ffn > 0) printf(" — ffn %lld", M->dense_ffn);
        puts("");
        puts("  engine class   : DENSE — expert streaming does NOT apply. There are no experts");
        puts("                   to stream; the whole weight set is live every token. Use a");
        puts("                   dense engine (llama.cpp) and size RAM for the full model.");
    } else {
        puts("  topology       : unknown — no expert/layer metadata in this container");
    }
    if (M->note[0]) printf("  note           : %s\n", M->note);
    puts("");
    print_roofline(M);
}

static void print_model_json(const Model *M, int first) {
    printf("%s  {\"path\": \"%s\", \"format\": \"%s\", \"ok\": %d, \"arch\": \"%s\", "
           "\"precision\": \"%s\", \"layers\": %lld, \"hidden\": %lld, \"experts\": %lld, "
           "\"experts_active\": %lld, \"shared_experts\": %lld, \"expert_ffn\": %lld, "
           "\"bytes_on_disk\": %lld, \"moe\": %s, \"expert_bank_exact\": %lld, "
           "\"tensor_bytes_total\": %lld, \"exps_tensors\": %d, "
           "\"trunk_bytes\": %lld, \"decode_bytes\": %lld, \"tied_embd\": %s}",
           first ? "" : ",\n", M->path, fmt_name(M->fmt), M->ok, M->arch, M->quant,
           M->layers, M->hidden, M->experts, M->experts_active, M->shared_experts,
           M->expert_ffn, M->bytes_on_disk, M->experts > 0 ? "true" : "false",
           M->expert_bank_exact, M->tensor_bytes_total, M->exps_tensors,
           M->trunk_bytes, M->decode_bytes, M->tied_embd ? "true" : "false");
}

int main(int argc, char **argv) {
    int as_json = 0, first = 1, n = 0, bad = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--json")) as_json = 1;
        else if (!strcmp(argv[i],"--bw") && i + 1 < argc) {
            g_bw_gbs = atof(argv[++i]);
            g_bw_src = "supplied via --bw";
        } else if (!strcmp(argv[i],"--disk-bw") && i + 1 < argc) {
            g_disk_gbs = atof(argv[++i]);
        } else if (!strcmp(argv[i],"--cache-gb") && i + 1 < argc) {
            g_cache_gb = atof(argv[++i]);
        } else if (!strcmp(argv[i],"--ram-bw") && i + 1 < argc) {
            g_ram_gbs = atof(argv[++i]);
        } else if (!strcmp(argv[i],"--ram-gb") && i + 1 < argc) {
            g_ram_gb = atof(argv[++i]);
        } else if (!strcmp(argv[i],"--measure-bw")) {
            g_bw_gbs = measure_read_bw();
            g_bw_src = "measured in-process, single-threaded — a FLOOR; "
                       "use membw for the real aggregate";
        }
    }
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) {
            printf("usage: %s [--json] [--bw GB/s | --measure-bw]\n"
                   "       [--disk-bw GB/s --cache-gb N] PATH [PATH...]\n"
                   "  Identifies GGUF / safetensors / HF-directory models, normalizes their\n"
                   "  structure, and computes per-token expert streaming demand for MoE models.\n"
                   "  Read-only. Bounds-checks all untrusted header fields.\n", argv[0]);
            return 0;
        }

    if (as_json) printf("[\n");
    for (int i = 1; i < argc; i++) {
        /* skip the flag AND its value -- "--bw 55.4" must not treat 55.4 as a path */
        if (!strcmp(argv[i],"--bw")      || !strcmp(argv[i],"--disk-bw") ||
            !strcmp(argv[i],"--cache-gb")|| !strcmp(argv[i],"--ram-bw")  ||
            !strcmp(argv[i],"--ram-gb")) { i++; continue; }
        if (argv[i][0] == '-') continue;
        Model M;
        probe_path(argv[i], &M);
        n++;
        if (!M.ok) bad++;
        if (as_json) { print_model_json(&M, first); first = 0; }
        else print_model(&M);
    }
    if (as_json) printf("\n]\n");

    if (!n) { fprintf(stderr, "no paths given (try --help)\n"); return 2; }
    return bad == n ? 1 : 0;     /* non-zero only if nothing at all was recognized */
}
