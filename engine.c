#include "llm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Forward declare internal helper
Sequence *scheduler_get_seq(Scheduler *s, int seq_id);

// ─────────────────────────────────────────────
//  Weights alloc / load
// ─────────────────────────────────────────────

ModelWeights *weights_alloc(const ModelConfig *cfg) {
    ModelWeights *w = calloc(1, sizeof(ModelWeights));
    int L = cfg->n_layers;

    w->attn_q    = calloc(L, sizeof(Tensor*));
    w->attn_k    = calloc(L, sizeof(Tensor*));
    w->attn_v    = calloc(L, sizeof(Tensor*));
    w->attn_o    = calloc(L, sizeof(Tensor*));
    w->ffn_gate  = calloc(L, sizeof(Tensor*));
    w->ffn_up    = calloc(L, sizeof(Tensor*));
    w->ffn_down  = calloc(L, sizeof(Tensor*));
    w->rms_attn  = calloc(L, sizeof(Tensor*));
    w->rms_ffn   = calloc(L, sizeof(Tensor*));
    w->q_params  = calloc(L * 7, sizeof(QuantParams));

    int H  = cfg->hidden_dim;
    int KV = cfg->n_kv_heads * cfg->head_dim;
    int FF = cfg->intermediate_dim;

    for (int l = 0; l < L; l++) {
        DType dt = cfg->weight_dtype;
        size_t s_HH[2]   = {H,  H};
        size_t s_HKV[2]  = {KV, H};
        size_t s_HFF[2]  = {H,  FF};
        size_t s_FFH[2]  = {FF, H};
        size_t s_H[1]    = {H};

        w->attn_q[l]   = tensor_alloc(dt, 2, s_HH);
        w->attn_k[l]   = tensor_alloc(dt, 2, s_HKV);
        w->attn_v[l]   = tensor_alloc(dt, 2, s_HKV);
        w->attn_o[l]   = tensor_alloc(dt, 2, s_HH);
        w->ffn_gate[l] = tensor_alloc(dt, 2, s_HFF);
        w->ffn_up[l]   = tensor_alloc(dt, 2, s_HFF);
        w->ffn_down[l] = tensor_alloc(dt, 2, s_FFH);
        w->rms_attn[l] = tensor_alloc(DTYPE_F32, 1, s_H);
        w->rms_ffn[l]  = tensor_alloc(DTYPE_F32, 1, s_H);
    }

    size_t s_VH[2] = {cfg->vocab_size, H};
    size_t s_H[1]  = {H};
    w->embed_tokens = tensor_alloc(DTYPE_F32, 2, s_VH);
    w->rms_final    = tensor_alloc(DTYPE_F32, 1, s_H);
    w->lm_head      = tensor_alloc(DTYPE_F32, 2, s_VH);

    return w;
}

void weights_free(ModelWeights *w, const ModelConfig *cfg) {
    if (!w) return;
    for (int l = 0; l < cfg->n_layers; l++) {
        tensor_free(w->attn_q[l]);   tensor_free(w->attn_k[l]);
        tensor_free(w->attn_v[l]);   tensor_free(w->attn_o[l]);
        tensor_free(w->ffn_gate[l]); tensor_free(w->ffn_up[l]);
        tensor_free(w->ffn_down[l]); tensor_free(w->rms_attn[l]);
        tensor_free(w->rms_ffn[l]);
    }
    free(w->attn_q); free(w->attn_k); free(w->attn_v); free(w->attn_o);
    free(w->ffn_gate); free(w->ffn_up); free(w->ffn_down);
    free(w->rms_attn); free(w->rms_ffn); free(w->q_params);
    tensor_free(w->embed_tokens);
    tensor_free(w->rms_final);
    tensor_free(w->lm_head);
    free(w);
}

bool weights_load(ModelWeights *w, const ModelConfig *cfg, const char *path) {
    // Stub: in production, mmap() the file and point tensors at offsets
    // Format: [magic][config][weight blobs in order]
    fprintf(stderr, "[weights] load from %s — implement mmap loader here\n", path);
    return false;
}

// ─────────────────────────────────────────────
//  Engine create / destroy
// ─────────────────────────────────────────────

LLMEngine *engine_create(const ModelConfig *cfg, const char *weights_path) {
    LLMEngine *e = calloc(1, sizeof(LLMEngine));
    e->cfg = *cfg;

    e->weights   = weights_alloc(cfg);
    e->kvcache   = kvcache_create(cfg->n_layers, cfg->n_kv_heads, cfg->head_dim);
    e->scheduler = scheduler_create();

    int H  = cfg->hidden_dim;
    int V  = cfg->vocab_size;
    int FF = cfg->intermediate_dim;
    int B  = MAX_BATCH;

    e->buf_hidden    = aligned_alloc(64, B * H  * sizeof(float));
    e->buf_q         = aligned_alloc(64, B * H  * sizeof(float));
    e->buf_k         = aligned_alloc(64, B * cfg->n_kv_heads * cfg->head_dim * sizeof(float));
    e->buf_v         = aligned_alloc(64, B * cfg->n_kv_heads * cfg->head_dim * sizeof(float));
    e->buf_attn_out  = aligned_alloc(64, B * H  * sizeof(float));
    e->buf_ffn_gate  = aligned_alloc(64, B * FF * sizeof(float));
    e->buf_ffn_up    = aligned_alloc(64, B * FF * sizeof(float));
    e->buf_logits    = aligned_alloc(64, B * V  * sizeof(float));

    if (weights_path)
        weights_load(e->weights, cfg, weights_path);

    return e;
}

void engine_destroy(LLMEngine *e) {
    if (!e) return;
    weights_free(e->weights, &e->cfg);
    kvcache_destroy(e->kvcache);
    scheduler_destroy(e->scheduler);
    free(e->buf_hidden); free(e->buf_q);    free(e->buf_k);
    free(e->buf_v);      free(e->buf_attn_out);
    free(e->buf_ffn_gate); free(e->buf_ffn_up); free(e->buf_logits);
    free(e);
}

// ─────────────────────────────────────────────
//  Single token forward pass for one sequence
//  Returns logits in out_logits [vocab_size]
// ─────────────────────────────────────────────

static void forward_one(LLMEngine *e, int seq_id, int token_id,
                         int pos, float *out_logits) {
    const ModelConfig *cfg = &e->cfg;
    int H    = cfg->hidden_dim;
    int FF   = cfg->intermediate_dim;
    int V    = cfg->vocab_size;
    int KVD  = cfg->n_kv_heads * cfg->head_dim;

    float *x = e->buf_hidden;  // [H] — current hidden state

    // 1. Token embedding lookup
    float *emb = (float*)e->weights->embed_tokens->data + token_id * H;
    memcpy(x, emb, H * sizeof(float));

    // 2. Transformer layers
    for (int l = 0; l < cfg->n_layers; l++) {
        float *rms_w;

        // --- Pre-attention RMS norm ---
        rms_w = (float*)e->weights->rms_attn[l]->data;
        float *x_norm = e->buf_q;  // reuse as temp
        memcpy(x_norm, x, H * sizeof(float));
        rms_norm(x_norm, rms_w, H, cfg->rms_norm_eps);

        // --- QKV projections ---
        float *q_buf = e->buf_q;
        float *k_buf = e->buf_k;
        float *v_buf = e->buf_v;

        if (cfg->weight_dtype == DTYPE_F32) {
            matmul_f32(q_buf, (float*)e->weights->attn_q[l]->data, x_norm, H,    H);
            matmul_f32(k_buf, (float*)e->weights->attn_k[l]->data, x_norm, KVD,  H);
            matmul_f32(v_buf, (float*)e->weights->attn_v[l]->data, x_norm, KVD,  H);
        } else {
            QuantParams qp = e->weights->q_params[l * 7 + 0];
            matmul_int8(q_buf, (int8_t*)e->weights->attn_q[l]->data, x_norm, H,   H, qp);
            matmul_int8(k_buf, (int8_t*)e->weights->attn_k[l]->data, x_norm, KVD, H, qp);
            matmul_int8(v_buf, (int8_t*)e->weights->attn_v[l]->data, x_norm, KVD, H, qp);
        }

        // --- RoPE ---
        for (int h = 0; h < cfg->n_heads; h++)
            rope_apply(q_buf + h * cfg->head_dim,
                       k_buf + (h / (cfg->n_heads / cfg->n_kv_heads)) * cfg->head_dim,
                       cfg->head_dim, pos, cfg->rope_theta);

        // --- Write K,V into cache ---
        kvcache_write(e->kvcache, l, seq_id, pos, false, k_buf);
        kvcache_write(e->kvcache, l, seq_id, pos, true,  v_buf);

        // --- Multi-head attention ---
        float *attn_out = e->buf_attn_out;
        mha(attn_out, q_buf, seq_id, pos,
            cfg->n_heads, cfg->n_kv_heads, cfg->head_dim,
            e->kvcache, l);

        // --- Output projection ---
        float *tmp = e->buf_ffn_gate;  // reuse
        if (cfg->weight_dtype == DTYPE_F32)
            matmul_f32(tmp, (float*)e->weights->attn_o[l]->data, attn_out, H, H);
        else {
            QuantParams qp = e->weights->q_params[l * 7 + 3];
            matmul_int8(tmp, (int8_t*)e->weights->attn_o[l]->data, attn_out, H, H, qp);
        }

        // --- Residual ---
        for (int i = 0; i < H; i++) x[i] += tmp[i];

        // --- Pre-FFN RMS norm ---
        rms_w = (float*)e->weights->rms_ffn[l]->data;
        float *xf = e->buf_attn_out;  // reuse
        memcpy(xf, x, H * sizeof(float));
        rms_norm(xf, rms_w, H, cfg->rms_norm_eps);

        // --- FFN (SwiGLU) ---
        float *gate = e->buf_ffn_gate;
        float *up   = e->buf_ffn_up;
        float *down_out = tmp;

        if (cfg->weight_dtype == DTYPE_F32) {
            matmul_f32(gate, (float*)e->weights->ffn_gate[l]->data, xf, FF, H);
            matmul_f32(up,   (float*)e->weights->ffn_up[l]->data,   xf, FF, H);
        } else {
            QuantParams qp = e->weights->q_params[l * 7 + 4];
            matmul_int8(gate, (int8_t*)e->weights->ffn_gate[l]->data, xf, FF, H, qp);
            matmul_int8(up,   (int8_t*)e->weights->ffn_up[l]->data,   xf, FF, H, qp);
        }

        // gate = silu(gate) * up  (written back to gate)
        swiglu(gate, gate, up, FF);

        if (cfg->weight_dtype == DTYPE_F32)
            matmul_f32(down_out, (float*)e->weights->ffn_down[l]->data, gate, H, FF);
        else {
            QuantParams qp = e->weights->q_params[l * 7 + 6];
            matmul_int8(down_out, (int8_t*)e->weights->ffn_down[l]->data, gate, H, FF, qp);
        }

        // Residual
        for (int i = 0; i < H; i++) x[i] += down_out[i];
    }

    // 3. Final norm + LM head
    rms_norm(x, (float*)e->weights->rms_final->data, H, cfg->rms_norm_eps);
    matmul_f32(out_logits, (float*)e->weights->lm_head->data, x, V, H);
}

// ─────────────────────────────────────────────
//  engine_step: run one decode step for all
//  active sequences (continuous batching)
// ─────────────────────────────────────────────

int engine_step(LLMEngine *e) {
    int batch_ids[MAX_BATCH];
    int n = scheduler_next_batch(e->scheduler, batch_ids, MAX_BATCH);
    int produced = 0;

    for (int b = 0; b < n; b++) {
        int       sid = batch_ids[b];
        Sequence *seq = scheduler_get_seq(e->scheduler, sid);
        if (!seq) continue;

        int   pos;
        int   token_in;
        float logits[65536];  // heap in production; stack here for clarity

        if (seq->state == SEQ_PREFILL) {
            // Run each prompt token through (no output until last)
            for (int t = 0; t < seq->prompt_len; t++) {
                forward_one(e, sid, seq->tokens[t], t, logits);
            }
            pos      = seq->prompt_len - 1;
            token_in = seq->tokens[pos];
            seq->state = SEQ_DECODE;
        } else {
            // Decode: last generated token is input
            pos      = seq->prompt_len + seq->gen_len - 1;
            token_in = seq->tokens[seq->prompt_len + seq->gen_len - 1];
            forward_one(e, sid, token_in, pos, logits);
        }

        // Sample next token
        int next = sample_argmax(logits, e->cfg.vocab_size);
        seq->tokens[seq->prompt_len + seq->gen_len] = next;
        seq->gen_len++;
        produced++;

        // Check stopping condition (EOS = 2 in LLaMA tokenizer)
        bool done = (next == 2) || (seq->gen_len >= seq->max_gen);
        if (done) {
            scheduler_mark_done(e->scheduler, sid);
            kvcache_free_seq(e->kvcache, sid);
        }
    }
    return produced;
}

// ─────────────────────────────────────────────
//  High-level blocking generate
// ─────────────────────────────────────────────

void engine_generate(LLMEngine *e,
                     const int *prompt, int prompt_len,
                     int max_gen,
                     float temperature, float top_p,
                     TokenCallback cb, void *userdata) {
    float *logits = calloc(e->cfg.vocab_size, sizeof(float));
    int sid = scheduler_add(e->scheduler, prompt, prompt_len, max_gen, logits);
    if (sid < 0) { free(logits); return; }

    Sequence *seq = scheduler_get_seq(e->scheduler, sid);
    while (seq && seq->state != SEQ_DONE) {
        engine_step(e);
        if (seq->gen_len > 0 && cb) {
            int last = seq->tokens[seq->prompt_len + seq->gen_len - 1];
            cb(last, sid, userdata);
        }
    }
    free(logits);
}
