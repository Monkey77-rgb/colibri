/* gguf_dump_io11.c — io11 oracle #1 tool: dump the FULL parsed GGUF picture
 * (every KV key/type/scalar-or-string-value-or-array-length, then every
 * tensor's name/type/dims/data_off/shard) to stdout in a deterministic,
 * diffable text format.
 *
 * Two invocations of this SAME source produce the BEFORE/AFTER comparison:
 *   BEFORE: compiled against the pre-io11 gguf_reader.h (checked out from git
 *           to a scratch copy) -- GGUF_RBUF_SZ is not defined there, so the
 *           RD_* macros below resolve to the original plain gguf_read_at-based
 *           functions (gguf_str/gguf_skip/gguf_read_int/gguf_read_at).
 *   AFTER:  compiled against this worktree's gguf_reader.h -- GGUF_RBUF_SZ IS
 *           defined, so the RD_* macros resolve to the new _buf functions
 *           (gguf_str_buf/gguf_skip_buf/gguf_read_int_buf/gguf_buf_read) that
 *           gguf_index_open_shard() itself now calls.
 * A byte-identical `cmp` between the two runs' output is therefore a direct
 * check that the buffered KV walk parses every key/value exactly as the
 * original unbuffered walk did -- not just that the tensor directory (which
 * gguf_index_open() already returns either way) came out the same.
 *
 * The KV walk here is deliberately a SEPARATE, from-scratch pass over the file
 * (open, header, then the same per-KV loop body gguf_index_open_shard() runs)
 * rather than a call into that static function, which does not retain or
 * expose individual KV records (only `alignment`/split flags survive it).
 * This pass exercises the identical primitives, in the identical order, on
 * the identical bytes -- it is not testing separate code, just observing more
 * of what the shared primitives did.
 *
 * The tensor directory dump below DOES call the real production entry point,
 * gguf_index_open(), unmodified from any other caller's perspective.
 *
 * Usage: gguf_dump_io11 <path.gguf> [max_kv_string_bytes_to_print=4096]
 * Exit 0 + dump on stdout on success; exit 1 + message on stderr on any parse
 * failure (a malformed/corrupted file must fail loudly, not print a partial
 * or wrong dump -- this is also the vehicle for the corruption negative
 * control in the oracle: run it against a good file and a one-byte-flipped
 * copy and diff, or confirm the flipped copy is refused).
 */
#include "../gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef GGUF_RBUF_SZ
typedef GgufReadBuf RdBuf;
#define RD_INIT(rb)                 do { (rb).base = -1; (rb).len = 0; } while (0)
#define RD_AT(fd,rb,buf,n,off)      gguf_buf_read((fd),&(rb),(buf),(n),(off))
#define RD_STR(fd,rb,off,fsz,o,on)  gguf_str_buf((fd),&(rb),(off),(fsz),(o),(on))
#define RD_SKIP(fd,rb,off,fsz,t)    gguf_skip_buf((fd),&(rb),(off),(fsz),(t))
#define VARIANT_NAME                "buffered (post-io11)"
#else
typedef int RdBuf;                  /* unused placeholder, no buffer before io11 */
#define RD_INIT(rb)                 do { (rb) = 0; } while (0)
#define RD_AT(fd,rb,buf,n,off)      gguf_read_at((fd),(buf),(n),(off))
#define RD_STR(fd,rb,off,fsz,o,on)  gguf_str((fd),(off),(fsz),(o),(on))
#define RD_SKIP(fd,rb,off,fsz,t)    gguf_skip((fd),(off),(fsz),(t))
#define VARIANT_NAME                "unbuffered (pre-io11)"
#endif

static void hexdump(const unsigned char *b, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
}

static void print_escaped(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\' || c == '"') { putchar('\\'); putchar(c); }
        else if (c >= 0x20 && c < 0x7f) putchar(c);
        else printf("\\x%02x", c);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <path.gguf>\n", argv[0]); return 2; }
    const char *path = argv[1];

    printf("# gguf_dump_io11 variant=%s file=%s\n", VARIANT_NAME, path);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open failed: %s\n", path); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { fprintf(stderr, "fstat failed\n"); return 1; }
    long long fsz = (long long)st.st_size;

    RdBuf rb; RD_INIT(rb);

    uint32_t magic, ver; uint64_t ntensor, nkv;
    if (fsz < 24 || !RD_AT(fd, rb, &magic, 4, 0) || magic != GGUF_MAGIC) {
        fprintf(stderr, "not a GGUF file (bad magic or too small)\n"); close(fd); return 1;
    }
    if (!RD_AT(fd, rb, &ver, 4, 4) || !RD_AT(fd, rb, &ntensor, 8, 8) || !RD_AT(fd, rb, &nkv, 8, 16)) {
        fprintf(stderr, "truncated GGUF header\n"); close(fd); return 1;
    }
    if (nkv > MAX_KV) { fprintf(stderr, "implausible kv count\n"); close(fd); return 1; }

    printf("HDR ver=%u ntensor=%llu nkv=%llu\n", ver, (unsigned long long)ntensor, (unsigned long long)nkv);

    long long off = 24;
    char key[256];
    for (uint64_t i = 0; i < nkv; i++) {
        if (!RD_STR(fd, rb, &off, fsz, key, sizeof key)) {
            fprintf(stderr, "malformed kv key at %llu\n", (unsigned long long)i); close(fd); return 1;
        }
        uint32_t t;
        if (off + 4 > fsz || !RD_AT(fd, rb, &t, 4, off)) {
            fprintf(stderr, "truncated kv type at %llu\n", (unsigned long long)i); close(fd); return 1;
        }
        off += 4;
        long long vpos = off;

        size_t sz;
        if (gguf_scalar_size(t, &sz)) {
            unsigned char b[8] = {0};
            if (vpos + (long long)sz > fsz || !RD_AT(fd, rb, b, sz, vpos)) {
                fprintf(stderr, "truncated kv scalar value at %llu\n", (unsigned long long)i); close(fd); return 1;
            }
            printf("KV %llu key=\"", (unsigned long long)i); print_escaped(key, strlen(key));
            printf("\" type=%u scalar=", t); hexdump(b, sz); printf("\n");
        } else if (t == G_STR) {
            char buf[4097];
            long long soff = vpos;   /* local copy: the real `off` still advances via RD_SKIP below */
            if (!RD_STR(fd, rb, &soff, fsz, buf, sizeof buf)) {
                fprintf(stderr, "malformed kv string value at %llu\n", (unsigned long long)i); close(fd); return 1;
            }
            printf("KV %llu key=\"", (unsigned long long)i); print_escaped(key, strlen(key));
            printf("\" type=%u str=\"", t); print_escaped(buf, strlen(buf)); printf("\"\n");
        } else if (t == G_ARR) {
            uint32_t et; uint64_t n;
            if (vpos + 12 > fsz || !RD_AT(fd, rb, &et, 4, vpos) || !RD_AT(fd, rb, &n, 8, vpos + 4)) {
                fprintf(stderr, "truncated kv array header at %llu\n", (unsigned long long)i); close(fd); return 1;
            }
            printf("KV %llu key=\"", (unsigned long long)i); print_escaped(key, strlen(key));
            printf("\" type=%u arr etype=%u n=%llu\n", t, et, (unsigned long long)n);
        } else {
            fprintf(stderr, "unknown kv type %u at %llu\n", t, (unsigned long long)i); close(fd); return 1;
        }

        if (!RD_SKIP(fd, rb, &off, fsz, t)) {
            fprintf(stderr, "malformed kv value (skip) at %llu\n", (unsigned long long)i); close(fd); return 1;
        }
    }
    close(fd);   /* the from-scratch KV pass owns this fd; gguf_index_open() below opens its own */

    GgufIndex idx; char err[256];
    if (!gguf_index_open(path, &idx, err, sizeof err)) {
        fprintf(stderr, "gguf_index_open failed: %s\n", err); return 1;
    }
    printf("IDX nshard=%zu alignment=%u data_base=%lld ntensor=%zu\n",
           idx.nshard, idx.alignment, idx.data_base, idx.n);
    for (size_t i = 0; i < idx.n; i++) {
        const GgufTensorInfo *ti = &idx.t[i];
        printf("T %zu name=\"", i); print_escaped(ti->name, strlen(ti->name));
        printf("\" ttype=%u rank=%d dims=[", ti->ttype, ti->rank);
        for (int d = 0; d < ti->rank; d++) printf("%s%llu", d ? "," : "", (unsigned long long)ti->shape[d]);
        printf("] data_off=%llu shard=%d\n", (unsigned long long)ti->data_off, ti->shard);
    }
    gguf_index_free(&idx);
    return 0;
}
