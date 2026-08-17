/* loader.h — a C ABI over the shared readers in ../c, so the C++ engine can use
 * them without touching them.
 *
 * WHY A SHIM AND NOT A PORT. ../c/gguf_reader.h, gguf_meta.h, tok.h and json.h
 * are C, and they use the implicit `void*` conversions that C permits and C++
 * rejects. Two ways out: edit them, or wrap them. Editing loses the thing that
 * makes them worth having -- gguf_reader.h's dequant path is proven bit-exact
 * over 473,956,352 elements of the live fleet models, and tok.h's tokenizer is
 * proven 81/81 against llama-tokenize. A portability convenience is not a reason
 * to touch either. So loader.c stays C, includes them unchanged, and exposes
 * this interface.
 *
 * It is also exactly the boundary cgo needs later: cgo can call C, not C++.
 */
#ifndef COLI_LOADER_H
#define COLI_LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct coli_gguf coli_gguf;

coli_gguf *coli_gguf_open(const char *path, char *err, size_t errcap);
void       coli_gguf_close(coli_gguf *g);

/* Metadata. Return 1 on success, 0 when absent -- absence is a normal answer,
 * not an error: llama and qwen2 legitimately carry different key sets. */
int coli_gguf_str(coli_gguf *g, const char *key, char *out, size_t outn);
int coli_gguf_i64(coli_gguf *g, const char *key, long long *out);
int coli_gguf_f32(coli_gguf *g, const char *key, float *out);

/* Tensors. shape() returns -1 when the tensor or dimension is absent. */
int     coli_gguf_has(coli_gguf *g, const char *tensor);
int64_t coli_gguf_shape(coli_gguf *g, const char *tensor, int dim);
/* Dequantize a whole tensor to f32. Caller frees with coli_gguf_free_f32.
 * Returns element count, 0 on failure. */
int64_t coli_gguf_load_f32(coli_gguf *g, const char *tensor, float **out);
void    coli_gguf_free_f32(float *p);

/* Tokenizer, read from the same file. Opaque so tok.h stays out of C++. */
void *coli_tok_load(const char *path, int *bos, int *eos, int *add_bos);
int   coli_tok_encode(void *t, const char *text, int *out, int max);
int   coli_tok_decode(void *t, const int *ids, int n, char *out, int max);
void  coli_tok_free_(void *t);

#ifdef __cplusplus
}
#endif
#endif
