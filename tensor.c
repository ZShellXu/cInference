#include "llm.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static size_t dtype_size(DType dt) {
    switch (dt) {
        case DTYPE_F32:  return 4;
        case DTYPE_INT8: return 1;
        case DTYPE_INT4: return 1; // 2 vals packed per byte
        default: return 0;
    }
}

size_t tensor_numel(const Tensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

Tensor *tensor_alloc(DType dtype, int ndim, size_t *shape) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->dtype = dtype;
    t->ndim  = ndim;
    t->owns_data = true;

    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        numel *= shape[i];
    }

    // Compute row-major strides
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        t->strides[i] = t->strides[i+1] * shape[i+1];

    // INT4: 2 values per byte
    size_t nbytes = (dtype == DTYPE_INT4) ? (numel + 1) / 2 : numel * dtype_size(dtype);

    // 64-byte aligned for cache lines and AVX-512
    t->data = aligned_alloc(64, (nbytes + 63) & ~63ULL);
    memset(t->data, 0, (nbytes + 63) & ~63ULL);
    return t;
}

Tensor *tensor_view(void *data, DType dtype, int ndim, size_t *shape) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->dtype = dtype;
    t->ndim  = ndim;
    t->data  = data;
    t->owns_data = false;

    t->strides[ndim - 1] = 1;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    for (int i = ndim - 2; i >= 0; i--)
        t->strides[i] = t->strides[i+1] * shape[i+1];
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->owns_data && t->data) free(t->data);
    free(t);
}
