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

#include "json.h"          /* reused deliberately: same parser the engines use, now hardened */

#ifndef NAME_MAX
#  define NAME_MAX 255
#endif

#define GGUF_MAGIC   0x46554747u          /* "GGUF" little-endian */
#define MAX_KV       (1u << 20)           /* sanity caps on untrusted counts */
#define MAX_STRLEN   (1u << 20)
#define MAX_ARRLEN   (1u << 24)
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

static int read_at(int fd, void *buf, size_t n, long long off) {
    ssize_t r = pread(fd, buf, n, (off_t)off);
    return r == (ssize_t)n;
}

/* -------------------------------------------------------------- gguf */

/* GGUF metadata value types, per the upstream spec. */
enum { G_U8, G_I8, G_U16, G_I16, G_U32, G_I32, G_F32, G_BOOL,
       G_STR, G_ARR, G_U64, G_I64, G_F64 };

static int gguf_scalar_size(uint32_t t, size_t *sz) {
    switch (t) {
        case G_U8: case G_I8: case G_BOOL: *sz = 1; return 1;
        case G_U16: case G_I16:            *sz = 2; return 1;
        case G_U32: case G_I32: case G_F32:*sz = 4; return 1;
        case G_U64: case G_I64: case G_F64:*sz = 8; return 1;
        default: return 0;                       /* string/array handled separately */
    }
}

/* Read a length-prefixed GGUF string. Advances *off. Bounds-checked against fsz. */
static int gguf_str(int fd, long long *off, long long fsz, char *out, size_t outn) {
    uint64_t len;
    if (*off + 8 > fsz || !read_at(fd, &len, 8, *off)) return 0;
    *off += 8;
    if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
    size_t take = (len < outn - 1) ? (size_t)len : outn - 1;
    if (take && !read_at(fd, out, take, *off)) return 0;
    out[take] = 0;
    *off += (long long)len;
    return 1;
}

/* Skip a value of type t. Returns 0 on malformed input. */
static int gguf_skip(int fd, long long *off, long long fsz, uint32_t t) {
    size_t sz;
    if (gguf_scalar_size(t, &sz)) {
        if (*off + (long long)sz > fsz) return 0;
        *off += (long long)sz;
        return 1;
    }
    if (t == G_STR) {
        uint64_t len;
        if (*off + 8 > fsz || !read_at(fd, &len, 8, *off)) return 0;
        *off += 8;
        if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
        *off += (long long)len;
        return 1;
    }
    if (t == G_ARR) {
        uint32_t et; uint64_t n;
        if (*off + 12 > fsz) return 0;
        if (!read_at(fd, &et, 4, *off) || !read_at(fd, &n, 8, *off + 4)) return 0;
        *off += 12;
        if (n > MAX_ARRLEN) return 0;
        if (gguf_scalar_size(et, &sz)) {
            long long need = (long long)n * (long long)sz;
            if (need < 0 || need > fsz - *off) return 0;
            *off += need;
            return 1;
        }
        if (et == G_STR) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len;
                if (*off + 8 > fsz || !read_at(fd, &len, 8, *off)) return 0;
                *off += 8;
                if (len > MAX_STRLEN || (long long)len > fsz - *off) return 0;
                *off += (long long)len;
            }
            return 1;
        }
        return 0;                              /* nested arrays: not in practice */
    }
    return 0;
}

static long long gguf_read_int(int fd, long long off, uint32_t t) {
    uint64_t u = 0; int64_t s = 0;
    switch (t) {
        case G_U8:  { uint8_t v;  if (read_at(fd,&v,1,off)) u = v; break; }
        case G_I8:  { int8_t  v;  if (read_at(fd,&v,1,off)) s = v; return s; }
        case G_U16: { uint16_t v; if (read_at(fd,&v,2,off)) u = v; break; }
        case G_I16: { int16_t v;  if (read_at(fd,&v,2,off)) s = v; return s; }
        case G_U32: { uint32_t v; if (read_at(fd,&v,4,off)) u = v; break; }
        case G_I32: { int32_t v;  if (read_at(fd,&v,4,off)) s = v; return s; }
        case G_U64: { uint64_t v; if (read_at(fd,&v,8,off)) u = v; break; }
        case G_I64: { int64_t v;  if (read_at(fd,&v,8,off)) s = v; return s; }
        case G_BOOL:{ uint8_t v;  if (read_at(fd,&v,1,off)) u = v; break; }
        default: return -1;
    }
    return (long long)u;
}

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
    if (fsz < 24 || !read_at(fd,&magic,4,0) || magic != GGUF_MAGIC) { close(fd); return 0; }
    if (!read_at(fd,&ver,4,4) || !read_at(fd,&ntensor,8,8) || !read_at(fd,&nkv,8,16)) {
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
    char key[256];
    /* Architecture-prefixed keys are only known after we read general.architecture, and
     * it is not guaranteed to come first -- so stash the interesting suffixes as we go. */
    long long v_block=-1, v_expcnt=-1, v_expused=-1, v_embd=-1, v_ffn=-1, v_expffn=-1, v_shared=-1;

    for (uint64_t i = 0; i < nkv; i++) {
        if (!gguf_str(fd,&off,fsz,key,sizeof key)) { snprintf(M->note,sizeof M->note,"malformed kv key at %llu",(unsigned long long)i); break; }
        uint32_t t;
        if (off + 4 > fsz || !read_at(fd,&t,4,off)) break;
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

        if (!gguf_skip(fd,&off,fsz,t)) { snprintf(M->note,sizeof M->note,"malformed kv value at %llu",(unsigned long long)i); break; }
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
        if (fsz > 8 && read_at(fd,&hlen,8,0) && hlen > 1 && hlen <= MAX_HDR && (long long)hlen <= fsz-8) {
            char c;
            if (read_at(fd,&c,1,8) && c == '{') {
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

        double bpp = bytes_per_param(M);
        if (M->hidden > 0 && M->expert_ffn > 0 && bpp > 0) {
            double per = 3.0 * (double)M->hidden * (double)M->expert_ffn * bpp;
            long long total_experts = M->experts * (M->layers > 0 ? M->layers : 1);
            printf("  bytes/expert   : %.2f MB   (3 x hidden x ffn x %.2f B/param)\n", per/1e6, bpp);
            printf("  expert bank    : %.2f GB   (%lld experts total)\n",
                   per * (double)total_experts / 1e9, total_experts);
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
}

static void print_model_json(const Model *M, int first) {
    printf("%s  {\"path\": \"%s\", \"format\": \"%s\", \"ok\": %d, \"arch\": \"%s\", "
           "\"precision\": \"%s\", \"layers\": %lld, \"hidden\": %lld, \"experts\": %lld, "
           "\"experts_active\": %lld, \"shared_experts\": %lld, \"expert_ffn\": %lld, "
           "\"bytes_on_disk\": %lld, \"moe\": %s}",
           first ? "" : ",\n", M->path, fmt_name(M->fmt), M->ok, M->arch, M->quant,
           M->layers, M->hidden, M->experts, M->experts_active, M->shared_experts,
           M->expert_ffn, M->bytes_on_disk, M->experts > 0 ? "true" : "false");
}

int main(int argc, char **argv) {
    int as_json = 0, first = 1, n = 0, bad = 0;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i],"--json")) as_json = 1;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) {
            printf("usage: %s [--json] PATH [PATH...]\n"
                   "  Identifies GGUF / safetensors / HF-directory models, normalizes their\n"
                   "  structure, and computes per-token expert streaming demand for MoE models.\n"
                   "  Read-only. Bounds-checks all untrusted header fields.\n", argv[0]);
            return 0;
        }

    if (as_json) printf("[\n");
    for (int i = 1; i < argc; i++) {
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
