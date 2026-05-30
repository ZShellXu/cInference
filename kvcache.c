#include "llm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

KVCache *kvcache_create(int n_layers, int n_heads, int head_dim) {
    KVCache *c = calloc(1, sizeof(KVCache));
    c->n_layers = n_layers;
    c->n_heads  = n_heads;
    c->head_dim = head_dim;
    c->free_top = 0;

    // Each page stores KV_PAGE_SIZE tokens, for K and V, for all heads
    // Layout: [PAGE_SIZE][n_heads][head_dim] x2 (K then V)
    size_t page_floats = (size_t)KV_PAGE_SIZE * n_heads * head_dim * 2;

    for (int i = 0; i < KV_MAX_PAGES; i++) {
        // Allocate per-layer data lazily in a flat array
        c->pages[i].data   = aligned_alloc(64,
            page_floats * n_layers * sizeof(float));
        c->pages[i].seq_id = -1;
        c->pages[i].page_idx = -1;
        memset(c->pages[i].data, 0, page_floats * n_layers * sizeof(float));

        c->free_stack[c->free_top++] = i;
    }
    return c;
}

void kvcache_destroy(KVCache *c) {
    if (!c) return;
    for (int i = 0; i < KV_MAX_PAGES; i++)
        free(c->pages[i].data);
    free(c);
}

int kvcache_alloc_page(KVCache *c, int seq_id, int page_idx) {
    if (c->free_top == 0) {
        fprintf(stderr, "[kvcache] OOM: no free pages\n");
        return -1;
    }
    int pid = c->free_stack[--c->free_top];
    c->pages[pid].seq_id   = seq_id;
    c->pages[pid].page_idx = page_idx;
    return pid;
}

void kvcache_free_seq(KVCache *c, int seq_id) {
    for (int i = 0; i < KV_MAX_PAGES; i++) {
        if (c->pages[i].seq_id == seq_id) {
            c->pages[i].seq_id   = -1;
            c->pages[i].page_idx = -1;
            c->free_stack[c->free_top++] = i;
        }
    }
}

// Helper: find page for seq_id at page_idx
static int find_page(KVCache *c, int seq_id, int page_idx) {
    for (int i = 0; i < KV_MAX_PAGES; i++)
        if (c->pages[i].seq_id == seq_id &&
            c->pages[i].page_idx == page_idx)
            return i;
    return -1;
}

// ─────────────────────────────────────────────
//  Write one token's K or V vector
//  pos: absolute position in sequence
// ─────────────────────────────────────────────
void kvcache_write(KVCache *c, int layer, int seq_id, int pos,
                   bool is_value, const float *vec) {
    int page_idx  = pos / KV_PAGE_SIZE;
    int slot      = pos % KV_PAGE_SIZE;

    int pid = find_page(c, seq_id, page_idx);
    if (pid < 0) {
        pid = kvcache_alloc_page(c, seq_id, page_idx);
        if (pid < 0) return;
    }

    // Layout in page: [n_layers][2 (K=0,V=1)][PAGE_SIZE][n_heads][head_dim]
    size_t layer_stride = 2ULL * KV_PAGE_SIZE * c->n_heads * c->head_dim;
    size_t kv_stride    = (size_t)KV_PAGE_SIZE * c->n_heads * c->head_dim;
    size_t slot_stride  = (size_t)c->n_heads * c->head_dim;

    float *base = c->pages[pid].data
                + layer * layer_stride
                + (is_value ? kv_stride : 0)
                + slot * slot_stride;

    memcpy(base, vec, c->n_heads * c->head_dim * sizeof(float));
}

// ─────────────────────────────────────────────
//  Read all K or V vectors for a sequence [0..len)
//  out: [len][n_heads][head_dim]
// ─────────────────────────────────────────────
void kvcache_read_all(KVCache *c, int layer, int seq_id, int len,
                      bool is_value, float *out) {
    size_t layer_stride = 2ULL * KV_PAGE_SIZE * c->n_heads * c->head_dim;
    size_t kv_stride    = (size_t)KV_PAGE_SIZE * c->n_heads * c->head_dim;
    size_t slot_stride  = (size_t)c->n_heads * c->head_dim;

    for (int pos = 0; pos < len; pos++) {
        int page_idx = pos / KV_PAGE_SIZE;
        int slot     = pos % KV_PAGE_SIZE;
        int pid      = find_page(c, seq_id, page_idx);
        if (pid < 0) {
            memset(out + pos * slot_stride, 0, slot_stride * sizeof(float));
            continue;
        }
        float *src = c->pages[pid].data
                   + layer * layer_stride
                   + (is_value ? kv_stride : 0)
                   + slot * slot_stride;
        memcpy(out + pos * slot_stride, src, slot_stride * sizeof(float));
    }
}
