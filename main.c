#include "llm.h"
#include <stdio.h>
#include <string.h>

// Simple token printer callback
static void on_token(int token, int seq_id, void *userdata) {
    printf("[seq %d] token %d\n", seq_id, token);
    fflush(stdout);
}

int main(void) {
    // LLaMA-7B-like config
    ModelConfig cfg = {
        .vocab_size       = 32000,
        .hidden_dim       = 4096,
        .intermediate_dim = 11008,
        .n_layers         = 32,
        .n_heads          = 32,
        .n_kv_heads       = 32,   // set to 8 for GQA (LLaMA-2 style)
        .head_dim         = 128,  // hidden_dim / n_heads
        .max_seq_len      = 4096,
        .rms_norm_eps     = 1e-5f,
        .rope_theta       = 10000.0f,
        .weight_dtype     = DTYPE_F32,  // swap to DTYPE_INT8 for quantized
    };

    printf("Creating engine (no weights loaded — stub mode)...\n");
    LLMEngine *engine = engine_create(&cfg, NULL);

    // Fake prompt tokens
    int prompt[] = {1, 15043, 29892, 825, 338, 278, 10344, 310, 301, 19600};
    int prompt_len = sizeof(prompt) / sizeof(prompt[0]);

    printf("Running generate (will output random tokens without real weights)...\n");
    engine_generate(engine, prompt, prompt_len,
                    /*max_gen=*/20,
                    /*temperature=*/0.8f, /*top_p=*/0.95f,
                    on_token, NULL);

    engine_destroy(engine);
    printf("Done.\n");
    return 0;
}
