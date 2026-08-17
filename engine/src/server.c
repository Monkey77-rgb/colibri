/* server.c — minimal HTTP server, OpenAI-compatible /v1/completions.
 *
 * HONEST SCOPE. One model, one request at a time, served from a queue. There is
 * no continuous batching and no cross-request batching. That is a real
 * limitation and it is stated here rather than discovered under load: with a
 * single slot, time-to-first-token grows linearly with queue depth.
 *
 * It is ALSO the reason the batch-size dispatch in gemm_i8.c matters more than
 * it looks. A server that batched requests would spend most of its time at n>=4
 * where the wide kernel wins; a single-slot server spends every generated token
 * at n=1, where the wide kernel LOSES 17-22%. The dispatch is what makes those
 * two futures share one kernel file.
 *
 * SECURITY POSTURE, deliberately narrow:
 *   - binds 127.0.0.1 by default; --host must be given explicitly to widen it,
 *     and doing so is the caller's decision, not a default.
 *   - hard cap on request body size, so a Content-Length of 4 GB cannot make us
 *     allocate it.
 *   - the JSON parser accepts only the fields it needs and ignores the rest.
 * No auth, no TLS. Do not expose this to a network without something in front.
 */
#define _GNU_SOURCE
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#include "platform.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BODY (1<<20)      /* 1 MiB of prompt is already absurd */
#define MAX_TOK  65536

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s){ (void)s; g_stop = 1; }

/* ---- tiny JSON field extraction. Not a parser: it pulls the handful of keys
 * this endpoint understands and ignores everything else. A full parser is a
 * larger attack surface than this endpoint needs. ---- */
static const char *jfind(const char *b, const char *key) {
    char pat[64]; snprintf(pat,sizeof pat,"\"%s\"",key);
    const char *p = strstr(b, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') p++;
    return p;
}
static double jnum(const char *b, const char *key, double dflt) {
    const char *p = jfind(b,key); if (!p) return dflt;
    char *end; double v = strtod(p,&end);
    return end==p ? dflt : v;
}
/* Unescapes into `out`; returns length, or -1 if absent. */
static int jstr(const char *b, const char *key, char *out, int cap) {
    const char *p = jfind(b,key); if (!p || *p != '"') return -1;
    p++; int n=0;
    while (*p && *p != '"' && n < cap-1) {
        if (*p=='\\' && p[1]) {
            p++;
            switch (*p) {
                case 'n': out[n++]='\n'; break;  case 't': out[n++]='\t'; break;
                case 'r': out[n++]='\r'; break;  case 'b': out[n++]='\b'; break;
                case 'f': out[n++]='\f'; break;
                case 'u': {
                    if (!p[1]||!p[2]||!p[3]||!p[4]) return n;
                    char h[5]={p[1],p[2],p[3],p[4],0};
                    unsigned cp = (unsigned)strtoul(h,NULL,16);
                    /* UTF-8 encode; surrogate pairs are not handled and are
                     * emitted as U+FFFD rather than silently mangled */
                    if (cp>=0xD800 && cp<=0xDFFF) cp=0xFFFD;
                    if (cp<0x80) out[n++]=(char)cp;
                    else if (cp<0x800){ if(n+2>=cap)break; out[n++]=(char)(0xC0|(cp>>6)); out[n++]=(char)(0x80|(cp&0x3F)); }
                    else { if(n+3>=cap)break; out[n++]=(char)(0xE0|(cp>>12)); out[n++]=(char)(0x80|((cp>>6)&0x3F)); out[n++]=(char)(0x80|(cp&0x3F)); }
                    p+=4; break; }
                default: out[n++]=*p;
            }
            p++;
        } else out[n++]=*p++;
    }
    out[n]=0; return n;
}

/* JSON string escaping for the response. Without this, a model that emits a
 * quote or a newline produces malformed JSON -- which reads to the client as
 * "the server is broken", not "the model said something with a quote in it". */
static void jesc(const char *s, int n, char **buf, size_t *cap, size_t *len) {
    for (int i=0;i<n;i++) {
        unsigned char c = (unsigned char)s[i];
        char tmp[8]; int tn;
        if      (c=='"')  { memcpy(tmp,"\\\"",2); tn=2; }
        else if (c=='\\') { memcpy(tmp,"\\\\",2); tn=2; }
        else if (c=='\n') { memcpy(tmp,"\\n",2);  tn=2; }
        else if (c=='\r') { memcpy(tmp,"\\r",2);  tn=2; }
        else if (c=='\t') { memcpy(tmp,"\\t",2);  tn=2; }
        else if (c < 0x20){ tn = snprintf(tmp,sizeof tmp,"\\u%04x",c); }
        else              { tmp[0]=(char)c; tn=1; }
        if (*len + (size_t)tn + 1 > *cap) { *cap = (*cap?*cap*2:4096) + (size_t)tn; *buf = realloc(*buf,*cap); }
        memcpy(*buf + *len, tmp, (size_t)tn); *len += (size_t)tn;
    }
    if (*buf) (*buf)[*len]=0;
}

static void send_all(int fd, const char *b, size_t n) {
    while (n) { ssize_t w = send(fd,b,n,MSG_NOSIGNAL);
        if (w <= 0) return; b += w; n -= (size_t)w; }
}
static void send_err(int fd, int code, const char *msg) {
    char h[512]; int n = snprintf(h,sizeof h,
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nConnection: close\r\n"
        "Content-Length: %zu\r\n\r\n{\"error\":{\"message\":\"%s\"}}",
        code, code==400?"Bad Request":code==404?"Not Found":"Error",
        strlen(msg)+24, msg);
    send_all(fd,h,(size_t)n);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,
        "usage: %s <model.gguf> [--host H] [--port P] [-c N]\n"
        "  binds 127.0.0.1:8100 by default. No auth, no TLS -- put something in front.\n",
        argv[0]); return 2; }
    const char *path = argv[1], *host = "127.0.0.1";
    int port = 8100, ctx = 0;
    for (int i=2;i<argc;i++) {
        if (!strcmp(argv[i],"--host") && i+1<argc) host = argv[++i];
        else if (!strcmp(argv[i],"--port") && i+1<argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-c") && i+1<argc) ctx = atoi(argv[++i]);
        else { fprintf(stderr,"unknown option %s\n",argv[i]); return 2; }
    }
    /* sigaction WITHOUT SA_RESTART. signal() installs restarting handlers on
     * Linux (BSD semantics), so accept() resumed instead of returning EINTR and
     * the server ignored SIGTERM entirely -- it kept its listening socket bound
     * after `kill`. Caught by checking `ss` after the test, not by the test. */
    coli_net_init();          /* WSAStartup on Windows, SIGPIPE ignore on POSIX */
    coli_on_signal(on_sig);   /* non-restarting: see platform.h */

    char err[512];
    coli_model *m = coli_load(path,ctx,1,err,sizeof err);
    if (!m) { fprintf(stderr,"load failed: %s\n",err); return 1; }
    fprintf(stderr,"loaded %s: %d layers, ctx %d\n", m->cfg.arch, m->cfg.n_layers, m->max_ctx);

    int srv = socket(AF_INET,SOCK_STREAM,0);
    int one = 1; setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    struct sockaddr_in a; memset(&a,0,sizeof a);
    a.sin_family=AF_INET; a.sin_port=htons((uint16_t)port);
    if (inet_pton(AF_INET,host,&a.sin_addr) != 1) { fprintf(stderr,"bad host %s\n",host); return 1; }
    if (bind(srv,(struct sockaddr*)&a,sizeof a) < 0) { perror("bind"); return 1; }
    if (listen(srv,16) < 0) { perror("listen"); return 1; }
    fprintf(stderr,"listening on %s:%d  (single slot, no batching)\n",host,port);
    if (strcmp(host,"127.0.0.1") && strcmp(host,"localhost"))
        fprintf(stderr,"WARNING: bound to %s -- this server has no auth and no TLS\n",host);

    static int ids[MAX_TOK];
    while (!g_stop) {
        int fd = accept(srv,NULL,NULL);
        if (fd < 0) { if (coli_sock_would_retry()) continue; break; }
        setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one);

        char *req = malloc(MAX_BODY+8192); size_t got=0; int hdr_end=-1;
        while (got < MAX_BODY+8191) {
            ssize_t r = recv(fd,req+got,MAX_BODY+8191-got,0);
            if (r <= 0) break;
            got += (size_t)r; req[got]=0;
            char *p = strstr(req,"\r\n\r\n");
            if (p) { hdr_end = (int)(p-req)+4;
                const char *cl = strcasestr(req,"content-length:");
                long need = cl ? atol(cl+15) : 0;
                if (need > MAX_BODY) { send_err(fd,400,"body too large"); goto done; }
                if ((long)(got - (size_t)hdr_end) >= need) break; }
        }
        if (hdr_end < 0) { send_err(fd,400,"malformed request"); goto done; }

        if (!strncmp(req,"GET /health",11)) {
            const char *b = "{\"status\":\"ok\"}";
            char h[256]; int n=snprintf(h,sizeof h,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                "Content-Length: %zu\r\n\r\n%s",strlen(b),b);
            send_all(fd,h,(size_t)n); goto done;
        }
        if (strncmp(req,"POST /v1/completions",20) && strncmp(req,"POST /completion",16)) {
            send_err(fd,404,"try POST /v1/completions or GET /health"); goto done; }

        const char *body = req + hdr_end;
        static char prompt[MAX_BODY];
        int plen = jstr(body,"prompt",prompt,sizeof prompt);
        if (plen < 0) { send_err(fd,400,"missing \\\"prompt\\\""); goto done; }

        coli_sampler sp; coli_sampler_default(&sp);
        int n_max = (int)jnum(body,"max_tokens",128);
        sp.temp  = (float)jnum(body,"temperature",0.0);
        sp.top_p = (float)jnum(body,"top_p",1.0);
        sp.top_k = (int)  jnum(body,"top_k",0);
        sp.min_p = (float)jnum(body,"min_p",0.0);
        sp.repeat_penalty = (float)jnum(body,"repeat_penalty",1.0);
        if (sp.repeat_penalty != 1.f) sp.repeat_last_n = 64;
        sp.seed  = (uint64_t)jnum(body,"seed",0);
        if (n_max < 1) n_max = 1;
        if (n_max > MAX_TOK-16) n_max = MAX_TOK-16;

        /* Fresh KV per request. Without this, request N+1 attends to request N's
         * tokens -- a cross-request information leak, not merely a wrong answer. */
        m->n_past = 0;

        int nid = 0;
        if (m->cfg.add_bos && m->cfg.bos >= 0) ids[nid++] = m->cfg.bos;
        nid += coli_encode(m,prompt,ids+nid,MAX_TOK-nid-1);
        if (nid + n_max > m->max_ctx) {
            send_err(fd,400,"prompt + max_tokens exceeds context"); goto done; }

        char *outbuf=NULL; size_t ocap=0, olen=0;
        float *lg = coli_forward(m,ids,nid,0);
        if (!lg) { send_err(fd,500,"forward failed"); goto done; }
        int cur = coli_sample(&sp,lg,m->cfg.vocab,ids,nid); free(lg);
        int gen=0, stop_eos=0;
        for (int i=0;i<n_max;i++) {
            ids[nid++]=cur; gen++;
            /* EOS is a stop signal, not content. Rendering it put a literal
             * "<|im_end|>" into the returned text, which every downstream client
             * then has to strip. Emit the piece only when it is not EOS. */
            if (cur == m->cfg.eos) { stop_eos=1; break; }
            char piece[64]; int pn = coli_decode(m,&cur,1,piece,(int)sizeof piece-1);
            jesc(piece,pn,&outbuf,&ocap,&olen);
            float *l2 = coli_forward(m,&cur,1,0); if (!l2) break;
            cur = coli_sample(&sp,l2,m->cfg.vocab,ids,nid); free(l2);
        }
        { size_t need = olen + 512;
          char *resp = malloc(need);
          int n = snprintf(resp,need,
            "{\"object\":\"text_completion\",\"model\":\"%s\","
            "\"choices\":[{\"index\":0,\"text\":\"%s\",\"finish_reason\":\"%s\"}],"
            "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
            m->cfg.arch, outbuf?outbuf:"", stop_eos?"stop":"length", nid-gen, gen, nid);
          char h[256]; int hn=snprintf(h,sizeof h,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
            "Content-Length: %d\r\n\r\n",n);
          send_all(fd,h,(size_t)hn); send_all(fd,resp,(size_t)n);
          free(resp); }
        free(outbuf);
    done:
        free(req); coli_closesocket(fd);
    }
    fprintf(stderr,"shutting down\n");
    coli_closesocket(srv); coli_free(m);
    return 0;
}
