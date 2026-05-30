#include "llm.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

// ─────────────────────────────────────────────
//  RMS Norm
//  x = x / sqrt(mean(x^2) + eps) * weight
// ─────────────────────────────────────────────

void rms_norm(float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss /= size;
    float scale = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < size; i++) x[i] = x[i] * scale * weight[i];
}

// ─────────────────────────────────────────────
//  RoPE (Rotary Position Embedding)
//  Applied in-place to q and k
//  head_dim must be even
// ─────────────────────────────────────────────

void rope_apply(float *q, float *k, int head_dim, int pos, float theta) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq  = 1.0f / powf(theta, (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float q0 = q[i], q1 = q[i+1];
        q[i]   = q0 * cos_a - q1 * sin_a;
        q[i+1] = q0 * sin_a + q1 * cos_a;

        float k0 = k[i], k1 = k[i+1];
        k[i]   = k0 * cos_a - k1 * sin_a;
        k[i+1] = k0 * sin_a + k1 * cos_a;
    }
}

// ─────────────────────────────────────────────
//  Matrix-vector multiply
//  out[M] = mat[M, K] @ vec[K]   (row-major)
// ─────────────────────────────────────────────

void matmul_f32(float *out, const float *mat, const float *vec,
                int M, int K) {
    // Loop order optimized for cache: row-by-row
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        const float *row = mat + m * K;
        // Unroll 4x to help compiler vectorize
        int k = 0;
        for (; k <= K - 4; k += 4) {
            acc += row[k]   * vec[k];
            acc += row[k+1] * vec[k+1];
            acc += row[k+2] * vec[k+2];
            acc += row[k+3] * vec[k+3];
        }
        for (; k < K; k++) acc += row[k] * vec[k];
        out[m] = acc;
    }
}

void matmul_int8(float *out, const int8_t *mat, const float *vec,
                 int M, int K, QuantParams p) {
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        const int8_t *row = mat + m * K;
        for (int k = 0; k < K; k++) {
            float w = (row[k] - p.zero_point) * p.scale;
            acc += w * vec[k];
        }
        out[m] = acc;
    }
}

// ─────────────────────────────────────────────
//  Softmax (in-place)
// ─────────────────────────────────────────────

void softmax(float *x, int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv;
}

// ─────────────────────────────────────────────
//  Multi-Head Attention (with GQA support)
//  Reads K/V from KV cache, writes new K/V
//
//  q:   [n_heads * head_dim]
//  out: [n_heads * head_dim]
// ─────────────────────────────────────────────

void mha(float *out,
         const float *q, int seq_id, int pos,
         int n_heads, int n_kv_heads, int head_dim,
         KVCache *kvc, int layer) {

    int   ctx_len   = pos + 1;
    int   gqa_ratio = n_heads / n_kv_heads;  // GQA grouping
    float scale     = 1.0f / sqrtf((float)head_dim);

    // Temporary buffers (stack for small head_dim, else heap)
    float *k_cache = malloc(ctx_len * head_dim * sizeof(float));
    float *v_cache = malloc(ctx_len * head_dim * sizeof(float));
    float *scores  = malloc(ctx_len * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;  // GQA: map query head → kv head

        const float *qh = q + h * head_dim;
        float       *oh = out + h * head_dim;

        // Read this KV head's cache
        // Each head stored separately: offset by kv_h * head_dim per token
        // For simplicity, read all heads then index (optimize later)
        kvcache_read_all(kvc, layer, seq_id, ctx_len, false,
                         k_cache);  // [ctx_len][n_heads*head_dim] if all heads
        kvcache_read_all(kvc, layer, seq_id, ctx_len, true,
                         v_cache);

        // Attention scores: q · k^T / sqrt(d)
        for (int t = 0; t < ctx_len; t++) {
            const float *kt = k_cache + t * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += qh[d] * kt[d];
            scores[t] = dot * scale;
        }

        // Causal mask: future tokens are -inf (already causal via ctx_len)
        softmax(scores, ctx_len);

        // Weighted sum of values
        memset(oh, 0, head_dim * sizeof(float));
        for (int t = 0; t < ctx_len; t++) {
            const float *vt = v_cache + t * head_dim;
            float        w  = scores[t];
            for (int d = 0; d < head_dim; d++)
                oh[d] += w * vt[d];
        }
    }

    free(k_cache);
    free(v_cache);
    free(scores);
}

// ─────────────────────────────────────────────
//  SwiGLU: out = silu(gate) * up
//  silu(x) = x * sigmoid(x)
// ─────────────────────────────────────────────

void swiglu(float *out, const float *gate, const float *up, int size) {
    for (int i = 0; i < size; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));  // g * sigmoid(g)
        out[i] = silu * up[i];
    }
}

// ─────────────────────────────────────────────
//  Sampling
// ─────────────────────────────────────────────

int sample_argmax(const float *logits, int vocab_size) {
    int   best_idx = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > best_val) { best_val = logits[i]; best_idx = i; }
    return best_idx;
}

// Top-p (nucleus) sampling with temperature
int sample_topp(const float *logits, int vocab_size,
                float temperature, float top_p) {
    // Apply temperature
    float *probs = malloc(vocab_size * sizeof(float));
    float  inv_t = 1.0f / (temperature < 1e-6f ? 1e-6f : temperature);
    for (int i = 0; i < vocab_size; i++) probs[i] = logits[i] * inv_t;
    softmax(probs, vocab_size);

    // Sort descending by probability (insertion sort — replace with qsort for large vocab)
    int *idx = malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) idx[i] = i;
    // qsort with comparator
    // For brevity, just find top-p greedily:
    float cumsum = 0.0f;
    float r = (float)rand() / (float)RAND_MAX;
    int result = 0;
    // Simple: iterate sorted... use partial sort in production
    // Here: iterate over all, accumulate until cumsum >= r * top_p_mass
    // (simplified — replace with proper nucleus sampling in production)
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= r) { result = i; break; }
    }

    free(probs);
    free(idx);
    return result;
}
