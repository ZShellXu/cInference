#include "llm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Scheduler *scheduler_create(void) {
    Scheduler *s = calloc(1, sizeof(Scheduler));
    s->next_seq_id = 1;
    return s;
}

void scheduler_destroy(Scheduler *s) { free(s); }

int scheduler_add(Scheduler *s, const int *tokens, int len,
                  int max_gen, float *logits_out) {
    if (s->n_active >= MAX_BATCH) {
        fprintf(stderr, "[scheduler] batch full\n");
        return -1;
    }

    // Find a free slot
    int slot = -1;
    for (int i = 0; i < MAX_BATCH; i++) {
        if (s->seqs[i].state == SEQ_DONE || s->seqs[i].seq_id == 0) {
            slot = i; break;
        }
    }
    if (slot < 0) return -1;

    Sequence *seq = &s->seqs[slot];
    memset(seq, 0, sizeof(Sequence));
    seq->seq_id    = s->next_seq_id++;
    seq->prompt_len = len;
    seq->gen_len   = 0;
    seq->max_gen   = max_gen;
    seq->state     = SEQ_PREFILL;
    seq->logits_out = logits_out;
    memcpy(seq->tokens, tokens, len * sizeof(int));
    s->n_active++;

    return seq->seq_id;
}

void scheduler_mark_done(Scheduler *s, int seq_id) {
    for (int i = 0; i < MAX_BATCH; i++) {
        if (s->seqs[i].seq_id == seq_id) {
            s->seqs[i].state = SEQ_DONE;
            s->n_active--;
            return;
        }
    }
}

// ─────────────────────────────────────────────
//  Continuous batching strategy:
//  1. Prefill sequences get priority (they block decode)
//  2. Decode sequences fill remaining batch slots
//  3. When a decode finishes, a new prefill can immediately
//     take its slot next step — no need to wait
// ─────────────────────────────────────────────

int scheduler_next_batch(Scheduler *s, int *out_ids, int max_ids) {
    int count = 0;

    // Prefill first
    for (int i = 0; i < MAX_BATCH && count < max_ids; i++) {
        Sequence *seq = &s->seqs[i];
        if (seq->state == SEQ_PREFILL)
            out_ids[count++] = seq->seq_id;
    }

    // Then decode
    for (int i = 0; i < MAX_BATCH && count < max_ids; i++) {
        Sequence *seq = &s->seqs[i];
        if (seq->state == SEQ_DECODE)
            out_ids[count++] = seq->seq_id;
    }

    return count;
}

// Helper used by engine
Sequence *scheduler_get_seq(Scheduler *s, int seq_id) {
    for (int i = 0; i < MAX_BATCH; i++)
        if (s->seqs[i].seq_id == seq_id) return &s->seqs[i];
    return NULL;
}
