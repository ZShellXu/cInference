# llm_inference — C LLM Inference Framework

LLaMA-7B–scale inference framework, implemented in pure C with zero dependencies (only libc + libm).

## Architecture Overview

```
llm.h           All type definitions, structs, and function declarations
tensor.c        Tensor: 64-byte aligned memory, shape/stride management
quant.c         INT8 / INT4 quantization and dequantization
kvcache.c       Page-based KV Cache (PagedAttention style)
ops.c           Core ops: RMSNorm, RoPE, MHA, SwiGLU, sampling
scheduler.c     Continuous Batching scheduler
engine.c        Forward pass + engine_generate high-level API
main.c          Example entry point
```

## Core Optimizations

### 1. Memory Alignment
All Tensor data uses `aligned_alloc(64, ...)` — aligned to cache line size so AVX-512 instructions can load directly without split-load penalties.

### 2. INT8 / INT4 Quantization
- Weights are quantized offline; at inference time `matmul_int8` performs dot products directly in int8
- INT4 packs two values into one byte, halving memory usage
- Each layer has its own QuantParams to avoid error accumulation across layers

### 3. Page-based KV Cache
- Each page stores K/V for 16 tokens
- Pages are freed when a sequence ends, avoiding fragmentation
- Supports multiple concurrent sequences, laying groundwork for Speculative Decoding

### 4. Continuous Batching
- Prefill sequences are scheduled first (avoids blocking the decode queue)
- Decode sequences fill remaining batch slots
- When one sequence finishes, the next prefill joins immediately without waiting for the whole batch to complete

### 5. matmul Inner-Loop Unrolling
```c
for (; k <= K - 4; k += 4) {
    acc += row[k]*vec[k] + row[k+1]*vec[k+1]
         + row[k+2]*vec[k+2] + row[k+3]*vec[k+3];
}
```
4× unrolling lets the compiler auto-vectorize with `-march=native` into AVX2/AVX-512.

## Build

```bash
make          # Release build, -O3 -march=native
make debug    # AddressSanitizer, for memory debugging
make profile  # gprof profiling
```

## Future Extensions

| Direction | Description |
|-----------|-------------|
| CUDA kernel | Rewrite matmul + MHA as `.cu`, use `cublas` or hand-written tiled GEMM |
| Flash Attention | Implement tiled softmax in MHA to reduce HBM reads/writes |
| mmap weight loading | Use `mmap` in `weights_load` to map files for zero-copy startup |
| Speculative Decoding | Add a small draft model; large model batch-verifies |
| Thread pool | Parallelize matmul within layers using `pthread` |
| Hand-written SIMD | Replace compiler auto-vectorization with hand-written AVX-512 matmul via `<immintrin.h>` |
