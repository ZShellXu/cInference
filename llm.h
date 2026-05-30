#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ─────────────────────────────────────────────
//  Data types
// ─────────────────────────────────────────────

typedef enum {
    DTYPE_F32  = 0,
    DTYPE_INT8 = 1,
    DTYPE_INT4 = 2,   // packed: 2 values per byte
} DType;

// ─────────────────────────────────────────────
//  Tensor
// ─────────────────────────────────────────────

#define MAX_DIMS 4

typedef struct {
    void    *data;
    size_t   shape[MAX_DIMS];
    size_t   strides[MAX_DIMS];  // in elements
    int      ndim;
    DType    dtype;
    bool     owns_data;          // whether we free() on destroy
} Tensor;

Tensor *tensor_alloc(DType dtype, int ndim, size_t *shape);
void    tensor_free(Tensor *t);
size_t  tensor_numel(const Tensor *t);
// View into existing memory (no copy)
Tensor *tensor_view(void *data, DType dtype, int ndim, size_t *shape);

// ─────────────────────────────────────────────
//  Quantization
// ─────────────────────────────────────────────

typedef struct {
    float   scale;
    int8_t  zero_point;
} QuantParams;

// Quantize f32 tensor → INT8, returns params
QuantParams quantize_int8(const Tensor *src, Tensor *dst);
void        dequantize_int8(const Tensor *src, Tensor *dst, QuantParams p);

// INT4 packed into int8 array (2 vals/byte)
QuantParams quantize_int4(const Tensor *src, Tensor *dst);
void        dequantize_int4(const Tensor *src, Tensor *dst, QuantParams p);

// ─────────────────────────────────────────────
//  KV Cache
// ─────────────────────────────────────────────

// Page-based KV cache (like PagedAttention)
// Each page holds PAGE_SIZE tokens for one layer

#define KV_PAGE_SIZE  16   // tokens per page
#define KV_MAX_PAGES  4096

typedef struct {
    float *data;           // [PAGE_SIZE, n_heads, head_dim] x 2 (K and V)
    int    seq_id;         // which sequence owns this page (-1 = free)
    int    page_idx;       // index within sequence's page list
} KVPage;

typedef struct {
    KVPage  pages[KV_MAX_PAGES];
    int     free_stack[KV_MAX_PAGES];
    int     free_top;
    int     n_layers;
    int     n_heads;
    int     head_dim;
} KVCache;

KVCache *kvcache_create(int n_layers, int n_heads, int head_dim);
void     kvcache_destroy(KVCache *c);
int      kvcache_alloc_page(KVCache *c, int seq_id, int page_idx);
void     kvcache_free_seq(KVCache *c, int seq_id);
// Write K or V for token at position pos in sequence
void     kvcache_write(KVCache *c, int layer, int seq_id, int pos,
                       bool is_value, const float *vec);
// Read K or V for all tokens [0..len) of a sequence
void     kvcache_read_all(KVCache *c, int layer, int seq_id, int len,
                          bool is_value, float *out);

// ─────────────────────────────────────────────
//  Model config (LLaMA-style)
// ─────────────────────────────────────────────

typedef struct {
    int    vocab_size;
    int    hidden_dim;
    int    intermediate_dim;  // FFN inner dim
    int    n_layers;
    int    n_heads;
    int    n_kv_heads;        // for GQA
    int    head_dim;          // hidden_dim / n_heads
    int    max_seq_len;
    float  rms_norm_eps;
    float  rope_theta;
    DType  weight_dtype;
} ModelConfig;

// ─────────────────────────────────────────────
//  Transformer weights
// ─────────────────────────────────────────────

typedef struct {
    // Per-layer weights
    Tensor **attn_q;      // [n_layers] each [hidden, hidden]
    Tensor **attn_k;      // [n_layers] each [hidden, kv_hidden]
    Tensor **attn_v;      // [n_layers] each [hidden, kv_hidden]
    Tensor **attn_o;      // [n_layers] each [hidden, hidden]
    Tensor **ffn_gate;    // [n_layers] each [hidden, inter]
    Tensor **ffn_up;      // [n_layers] each [hidden, inter]
    Tensor **ffn_down;    // [n_layers] each [inter, hidden]
    Tensor **rms_attn;    // [n_layers] each [hidden]
    Tensor **rms_ffn;     // [n_layers] each [hidden]

    // Global
    Tensor  *embed_tokens; // [vocab, hidden]
    Tensor  *rms_final;    // [hidden]
    Tensor  *lm_head;      // [vocab, hidden]

    // Quant params per layer (if INT8/INT4)
    QuantParams *q_params; // [n_layers * 7] — one per weight matrix
} ModelWeights;

ModelWeights *weights_alloc(const ModelConfig *cfg);
void          weights_free(ModelWeights *w, const ModelConfig *cfg);
// Load raw binary weights from file
bool          weights_load(ModelWeights *w, const ModelConfig *cfg,
                           const char *path);

// ─────────────────────────────────────────────
//  Continuous Batching — request queue
// ─────────────────────────────────────────────

#define MAX_SEQ_LEN   4096
#define MAX_BATCH     64

typedef enum {
    SEQ_PREFILL  = 0,  // processing prompt
    SEQ_DECODE   = 1,  // generating tokens
    SEQ_DONE     = 2,
} SeqState;

typedef struct {
    int       seq_id;
    int       tokens[MAX_SEQ_LEN];
    int       prompt_len;
    int       gen_len;         // tokens generated so far
    int       max_gen;
    SeqState  state;
    float    *logits_out;      // caller-owned output buffer
} Sequence;

typedef struct {
    Sequence  seqs[MAX_BATCH];
    int       n_active;
    int       next_seq_id;
} Scheduler;

Scheduler *scheduler_create(void);
void       scheduler_destroy(Scheduler *s);
int        scheduler_add(Scheduler *s, const int *tokens, int len,
                         int max_gen, float *logits_out);
void       scheduler_mark_done(Scheduler *s, int seq_id);
// Returns ids of sequences ready to run this step (prefill first)
int        scheduler_next_batch(Scheduler *s, int *out_ids, int max_ids);

// ─────────────────────────────────────────────
//  Core ops (CPU, AVX2 where available)
// ─────────────────────────────────────────────

// in-place RMS norm: x = x / rms(x) * weight
void rms_norm(float *x, const float *weight, int size, float eps);

// Rotary embeddings applied in-place to q and k
void rope_apply(float *q, float *k, int head_dim, int pos, float theta);

// Matrix-vector multiply: out[M] = mat[M,K] * vec[K]
// Supports f32 and int8 (with quant params)
void matmul_f32(float *out, const float *mat, const float *vec,
                int M, int K);
void matmul_int8(float *out, const int8_t *mat, const float *vec,
                 int M, int K, QuantParams p);

// Multi-head attention (single sequence, uses KV cache)
void mha(float *out,
         const float *q, int seq_id, int pos,
         int n_heads, int n_kv_heads, int head_dim,
         KVCache *kvc, int layer);

// SwiGLU: out = silu(gate) * up
void swiglu(float *out, const float *gate, const float *up, int size);

// Softmax in-place
void softmax(float *x, int size);

// Sample next token from logits
int  sample_argmax(const float *logits, int vocab_size);
int  sample_topp(const float *logits, int vocab_size,
                 float temperature, float top_p);

// ─────────────────────────────────────────────
//  Forward pass
// ─────────────────────────────────────────────

typedef struct {
    ModelConfig  cfg;
    ModelWeights *weights;
    KVCache      *kvcache;
    Scheduler    *scheduler;

    // Scratch buffers (pre-allocated, reused each step)
    float        *buf_hidden;   // [MAX_BATCH, hidden_dim]
    float        *buf_q;
    float        *buf_k;
    float        *buf_v;
    float        *buf_attn_out;
    float        *buf_ffn_gate;
    float        *buf_ffn_up;
    float        *buf_logits;   // [MAX_BATCH, vocab_size]
} LLMEngine;

LLMEngine *engine_create(const ModelConfig *cfg, const char *weights_path);
void       engine_destroy(LLMEngine *e);

// Run one decode step for all active sequences
// Returns number of sequences that produced a token
int        engine_step(LLMEngine *e);

// High-level: add a prompt and run until done
// Calls callback(token, seq_id) for each generated token
typedef void (*TokenCallback)(int token, int seq_id, void *userdata);
void       engine_generate(LLMEngine *e,
                           const int *prompt, int prompt_len,
                           int max_gen,
                           float temperature, float top_p,
                           TokenCallback cb, void *userdata);
