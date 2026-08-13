---
title: "Lecture 10 - Inference: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 10
companion_notes: "../lecture_notes/lecture_10_inference.md"
status: "complete"
---

# Lecture 10: Inference - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The reading path starts with the hardware model behind arithmetic intensity, then follows the two main inference bottlenecks: moving model and KV-cache bytes, and serial autoregressive decoding. The final papers show how modern serving systems schedule dynamic traffic and separate prefill from decode.

| Lecture theme | Papers |
|---|---|
| Hardware balance and IO-aware kernels | 1, 3 |
| KV-cache-efficient attention | 2, 9, 11-12 |
| Quantization and pruning | 5-7 |
| Exact and learned multi-token decoding | 8, 13 |
| Dynamic serving and KV-memory management | 4, 10, 14 |

## Curated papers

### 1. [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785) - 2009

**Connects to:** Arithmetic intensity, memory bandwidth, peak FLOPs, and the compute-bound versus memory-bound distinction.

**Very short abstract:** The Roofline model relates attainable performance to an operation's arithmetic intensity and a machine's compute and bandwidth ceilings. It gives a compact way to identify whether optimization should target data movement or arithmetic throughput.

**Why it is useful:** It supplies the conceptual model behind nearly every latency and throughput calculation in the lecture.

**Bigger picture:** LLM inference optimization is an application of a general systems lesson: FLOP counts alone do not predict wall-clock performance.

### 2. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - 2019

**Connects to:** Multi-query attention, shared key-value heads, decoder memory traffic, and KV-cache size.

**Very short abstract:** This paper replaces per-query-head key and value projections with one shared key-value head while retaining multiple query heads. The design reduces memory bandwidth and cache storage during incremental decoding with a limited quality tradeoff in the reported translation experiments.

**Why it is useful:** It is the foundational reference for treating KV-head count as an inference-efficiency design variable.

**Bigger picture:** MQA opened the path to GQA, cross-layer sharing, and latent KV compression in later LLMs.

### 3. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) - 2022

**Connects to:** Attention kernels, HBM traffic, SRAM tiling, exact attention, and IO complexity.

**Very short abstract:** FlashAttention reorganizes exact attention into tiles that keep intermediate work in fast on-chip memory and avoid materializing the full attention matrix in HBM. Its analysis makes data movement, rather than only arithmetic complexity, the central optimization target.

**Why it is useful:** It is the clearest concrete example of why an IO-aware algorithm can be faster without changing the mathematical operation.

**Bigger picture:** Kernel design and model architecture are complementary: one reduces the cost of an operation, while the other changes how much of that operation is required.

### 4. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - 2022

**Connects to:** Iteration-level scheduling, selective batching, ragged generation lengths, latency, and throughput.

**Very short abstract:** Orca schedules autoregressive requests one generation iteration at a time rather than holding a fixed batch until every request finishes. Selective batching lets compatible operations share execution while preserving flexibility as requests arrive and complete.

**Why it is useful:** It explains the systems origin of continuous batching and why static request-level batches waste accelerator capacity.

**Bigger picture:** Efficient LLM serving requires schedulers designed for multi-step generation, not merely repurposed feed-forward inference servers.

### 5. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://openreview.net/forum?id=tcbBPnfwxS) - 2023

**Connects to:** Weight-only quantization, low-bit inference, calibration data, and approximate second-order error correction.

**Very short abstract:** GPTQ quantizes a pretrained Transformer layer by layer while using approximate curvature information to compensate for errors introduced by earlier weight rounding. It makes low-bit compression practical for very large autoregressive models without full retraining.

**Why it is useful:** It provides a representative algorithm for the lecture's claim that fewer bytes per parameter can directly relieve decode bandwidth pressure.

**Bigger picture:** Post-training quantization became a standard route from a high-quality checkpoint to hardware-feasible deployment.

### 6. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) - 2023

**Connects to:** Activation outliers, weight-and-activation quantization, offline rescaling, and integer matrix multiplication.

**Very short abstract:** SmoothQuant applies a mathematically equivalent channelwise rescaling that moves quantization difficulty from activations into weights. This makes both weights and activations more amenable to efficient integer execution without retraining the model.

**Why it is useful:** It shows why activation statistics, not just parameter storage, determine whether quantization yields real kernel speedups.

**Bigger picture:** Compression methods increasingly co-design numerical transformations with the formats accelerators execute efficiently.

### 7. [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html) - 2023

**Connects to:** One-shot pruning, structured and unstructured sparsity, post-training repair, and model compression.

**Very short abstract:** SparseGPT formulates layerwise pruning as an approximate sparse-regression problem and updates retained weights to compensate for removed ones. It scales one-shot pruning to very large GPT-family models without a full retraining cycle.

**Why it is useful:** It turns the lecture's prune-and-repair description into a concrete large-model algorithm.

**Bigger picture:** Sparsity only becomes an inference win when the pruning pattern aligns with kernels and hardware that can skip the removed work.

### 8. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) - 2023

**Connects to:** Draft models, parallel verification, exact sampling, acceptance rules, and serial decode latency.

**Very short abstract:** A small model proposes several future tokens, and the target model checks those proposals in one parallel pass using a correction rule that preserves the target distribution. Accepted spans reduce the number of expensive target-model decoding steps.

**Why it is useful:** It gives the cleanest derivation of lossless speculative decoding and separates algorithmic correctness from empirical acceptance rate.

**Bigger picture:** Speculation converts otherwise idle parallel compute into fewer serial synchronization points, echoing a long-standing systems technique.

### 9. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/) - 2023

**Connects to:** Grouped-query attention, KV sharing across heads, uptraining, and the quality-speed tradeoff.

**Very short abstract:** GQA interpolates between full multi-head attention and single-head MQA by assigning groups of query heads to shared key-value heads. The paper also presents a low-cost procedure for converting and adapting an existing multi-head checkpoint.

**Why it is useful:** It is the standard reference for the architecture used by many current decoder-only LLMs to shrink the KV cache.

**Bigger picture:** GQA demonstrates a recurring design pattern: expose a continuous memory-quality tradeoff rather than choosing between two extremes.

### 10. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - 2023

**Connects to:** PagedAttention, vLLM, KV-cache fragmentation, copy-on-write sharing, and larger serving batches.

**Very short abstract:** PagedAttention stores each request's growing KV cache in noncontiguous blocks managed through a virtual-memory-like indirection layer. The vLLM system uses this organization to reduce fragmentation and share cache blocks across related decoding paths.

**Why it is useful:** It explains how memory allocation policy can limit batch size even when the attention computation itself is unchanged.

**Bigger picture:** Once KV caches dominate capacity, operating-system ideas become central components of model-serving runtimes.

### 11. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) - 2024

**Connects to:** Multi-head latent attention, compressed KV representations, inference-native architecture, and long-context serving.

**Very short abstract:** DeepSeek-V2 combines a sparse mixture-of-experts model with multi-head latent attention, which caches a compact latent representation rather than ordinary per-head keys and values. The architecture is designed to reduce both active computation and KV-cache cost while retaining expressive attention.

**Why it is useful:** It is the primary modern example of compressing information before caching instead of only sharing completed KV heads.

**Bigger picture:** MLA illustrates the shift from optimizing a fixed Transformer to designing models around deployment bottlenecks from the outset.

### 12. [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981) - 2024

**Connects to:** Cross-layer attention, KV sharing across depth, cache-quality Pareto frontiers, MQA, and GQA.

**Very short abstract:** Cross-layer attention reuses key and value heads between neighboring Transformer layers, extending KV sharing from the head dimension into depth. Controlled language-model experiments compare cache size and validation quality across different sharing configurations.

**Why it is useful:** It isolates the lecture's distinction between sharing KVs across heads and sharing them across layers.

**Bigger picture:** KV-cache compression is a multi-axis design space spanning heads, head dimension, layers, sequence positions, and numerical precision.

### 13. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) - 2024

**Connects to:** Multiple decoding heads, tree attention, parallel candidate verification, and draft-model-free acceleration.

**Very short abstract:** Medusa attaches extra heads to a language model so that it can propose several future tokens and candidate branches at once. Tree-structured attention verifies these candidates in parallel, avoiding the operational burden of maintaining a separate draft model.

**Why it is useful:** It shows a learned alternative to classical two-model speculative decoding.

**Bigger picture:** Modern decoding research increasingly moves speculative capability into the target model's own architecture and training process.

### 14. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) - 2024

**Connects to:** Prefill-decode disaggregation, time to first token, time per output token, interference, and service-level objectives.

**Very short abstract:** DistServe assigns compute-heavy prefill and bandwidth-sensitive decode to different GPU groups and selects resource and parallelism plans for each phase. It also considers placement and KV transfer costs when optimizing request capacity under separate latency objectives.

**Why it is useful:** It operationalizes the lecture's point that prefill and generation have different performance regimes and user-facing metrics.

**Bigger picture:** Production serving is evolving from a single model replica into a phase-specialized distributed pipeline.

## Suggested reading order

1. **Start here:** 1 (Roofline), 2 (MQA), 8 (speculative decoding), and 10 (PagedAttention).
2. **Understand the core tradeoffs:** 3 (FlashAttention), 4 (Orca), 5 (GPTQ), 7 (SparseGPT), and 9 (GQA).
3. **Study inference-native architectures:** 11 (DeepSeek-V2), 12 (cross-layer attention), and 13 (Medusa).
4. **Connect algorithms to production serving:** 6 (SmoothQuant) and 14 (DistServe).
