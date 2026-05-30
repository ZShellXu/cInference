#include "llm.h"
#include <math.h>
#include <string.h>
#include <float.h>

// ─────────────────────────────────────────────
//  INT8 Quantization
//  Maps f32 range [min, max] → [-127, 127]
//  scale = (max - min) / 254
//  zero_point = round(-min / scale) - 127
// ─────────────────────────────────────────────

QuantParams quantize_int8(const Tensor *src, Tensor *dst) {
    const float *in  = (const float *)src->data;
    int8_t      *out = (int8_t *)dst->data;
    size_t       n   = tensor_numel(src);

    // Find min/max (can be SIMD-ized later)
    float mn =  FLT_MAX, mx = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        if (in[i] < mn) mn = in[i];
        if (in[i] > mx) mx = in[i];
    }

    QuantParams p;
    p.scale      = (mx - mn) / 254.0f;
    if (p.scale == 0.0f) p.scale = 1e-8f;
    p.zero_point = (int8_t)roundf(-mn / p.scale) - 127;

    for (size_t i = 0; i < n; i++) {
        float q = roundf(in[i] / p.scale) + p.zero_point;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        out[i] = (int8_t)q;
    }
    return p;
}

void dequantize_int8(const Tensor *src, Tensor *dst, QuantParams p) {
    const int8_t *in  = (const int8_t *)src->data;
    float        *out = (float *)dst->data;
    size_t        n   = tensor_numel(src);

    for (size_t i = 0; i < n; i++)
        out[i] = (in[i] - p.zero_point) * p.scale;
}

// ─────────────────────────────────────────────
//  INT4 Quantization
//  Packed: low nibble = even index, high nibble = odd index
//  Range: [-8, 7] (signed 4-bit)
// ─────────────────────────────────────────────

QuantParams quantize_int4(const Tensor *src, Tensor *dst) {
    const float *in  = (const float *)src->data;
    uint8_t     *out = (uint8_t *)dst->data;
    size_t       n   = tensor_numel(src);

    float mn =  FLT_MAX, mx = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        if (in[i] < mn) mn = in[i];
        if (in[i] > mx) mx = in[i];
    }

    QuantParams p;
    p.scale      = (mx - mn) / 14.0f;  // [-7, 7]
    if (p.scale == 0.0f) p.scale = 1e-8f;
    p.zero_point = (int8_t)roundf(-mn / p.scale) - 7;

    for (size_t i = 0; i < n; i += 2) {
        float q0 = roundf(in[i]   / p.scale) + p.zero_point;
        float q1 = (i+1 < n) ? roundf(in[i+1] / p.scale) + p.zero_point : 0.0f;

        // Clamp to [-8, 7]
        if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
        if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;

        // Pack: low nibble = q0, high nibble = q1
        out[i/2] = ((uint8_t)((int8_t)q0 & 0x0F)) |
                   ((uint8_t)((int8_t)q1 & 0x0F) << 4);
    }
    return p;
}

void dequantize_int4(const Tensor *src, Tensor *dst, QuantParams p) {
    const uint8_t *in  = (const uint8_t *)src->data;
    float         *out = (float *)dst->data;
    size_t         n   = tensor_numel(dst);

    for (size_t i = 0; i < n; i += 2) {
        uint8_t byte = in[i/2];

        // Sign-extend 4-bit values
        int8_t q0 = (int8_t)(byte & 0x0F);
        if (q0 & 0x08) q0 |= 0xF0;  // sign extend

        int8_t q1 = (int8_t)((byte >> 4) & 0x0F);
        if (q1 & 0x08) q1 |= 0xF0;

        out[i]   = (q0 - p.zero_point) * p.scale;
        if (i+1 < n)
            out[i+1] = (q1 - p.zero_point) * p.scale;
    }
}
