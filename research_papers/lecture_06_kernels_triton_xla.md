---
title: "Lecture 6 - Kernels, Triton, and XLA: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 6
companion_notes: "../lecture_notes/lecture_06_kernels_triton_xla.md"
status: "complete"
---

# Lecture 6: Kernels, Triton, and XLA - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The path starts with the GPU execution model and roofline reasoning, then moves through tensor languages and compiler systems, and ends with modern attention kernels that turn the lecture's ideas about tiling, fusion, occupancy, and memory traffic into production-grade Transformer primitives.

| Lecture theme | Papers |
|---|---|
| GPU execution and performance limits | 1-2 |
| Tensor languages, compilation, and fusion | 3-8 |
| IO-aware and hardware-specialized Transformer kernels | 9-12 |

## Curated papers

### 1. [CUDA: Scalable parallel programming for high-performance scientific computing](https://ieeexplore.ieee.org/document/4541126) - 2008

**Connects to:** Threads, blocks, grids, warps, synchronization, memory hierarchy, and CUDA's programming abstraction.

**Very short abstract:** This paper presents CUDA as a general-purpose parallel programming environment for NVIDIA GPUs. It explains how programmers express large numbers of lightweight threads while the hardware schedules them across a scalable collection of processor cores.

**Why it is useful:** It gives the historical and architectural foundation for the thread/block/grid model assumed throughout the lecture.

**Bigger picture:** Triton and later kernel DSLs raise the abstraction level, but they still compile onto the execution and memory principles introduced by CUDA.

### 2. [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785) - 2009

**Connects to:** Arithmetic intensity, bandwidth-bound versus compute-bound kernels, tiling, and performance diagnosis.

**Very short abstract:** Roofline relates attainable floating-point performance to an operation's arithmetic intensity, peak compute throughput, and memory bandwidth. The resulting visual model identifies whether additional reuse or additional computational efficiency can plausibly improve a kernel.

**Why it is useful:** It formalizes the lecture's repeated question of whether a kernel is limited by HBM traffic or arithmetic throughput.

**Bigger picture:** Roofline reasoning remains a compact first model for both single-GPU kernels and the communication rooflines used in distributed training.

### 3. [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730) - 2018

**Connects to:** Expressing tensor algebra, compiling custom operators, fusion, specialization, and autotuning.

**Very short abstract:** Tensor Comprehensions introduces a mathematical language for tensor computations and a just-in-time compiler that lowers them to CUDA kernels. Its compilation pipeline specializes programs for concrete shapes, fuses work, and searches for efficient schedules.

**Why it is useful:** It shows an early attempt to close the gap between high-level tensor notation and shape-specific GPU code without hand-writing every kernel.

**Bigger picture:** The system anticipates the compiler-assisted workflow later embodied by Triton, TVM, and PyTorch's compiled execution.

### 4. [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) - 2018

**Connects to:** Operator fusion, hardware-aware schedules, memory-latency hiding, and performance portability.

**Very short abstract:** TVM separates tensor computations from their hardware schedules and optimizes both graph-level and operator-level execution. A learned cost model helps search a large schedule space across CPUs, GPUs, and specialized accelerators.

**Why it is useful:** It explains why compiler systems need explicit representations of tiling, locality, and target hardware rather than relying on source-level algebra alone.

**Bigger picture:** TVM established an influential template for machine-learning compilers that combine graph rewriting, tensor lowering, and autotuning.

### 5. [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973) - 2019

**Connects to:** Triton program instances, tiles, pointer arithmetic, masks, block-level reductions, and matrix multiplication.

**Very short abstract:** Triton introduces a language and compiler in which programmers describe tiled tensor computations rather than explicitly coordinating individual CUDA threads. The compiler maps these block programs to GPU threads, warps, memory operations, and synchronization.

**Why it is useful:** This is the central paper for understanding why the lecture's Triton kernels look vectorized while still exposing tile shape and data-movement choices.

**Bigger picture:** Triton occupies a productive middle layer between eager tensor frameworks and CUDA, and it later became a major backend for generated PyTorch kernels.

### 6. [TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions](https://doi.org/10.1145/3341301.3359630) - 2019

**Connects to:** Graph optimization, eliminating intermediate tensors, operator fusion, and correctness-preserving rewrites.

**Very short abstract:** TASO automatically generates candidate substitutions between equivalent tensor computation graphs, formally verifies them, and searches for a lower-cost graph. It jointly considers graph structure and data layout instead of depending only on hand-written rewrite rules.

**Why it is useful:** It broadens the lecture's fusion example from one GELU expression to systematic optimization over whole computation graphs.

**Bigger picture:** Kernel quality and graph quality are complementary: a fast individual kernel cannot recover traffic wasted by a poor decomposition into operators.

### 7. [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/conference/osdi20/presentation/zheng) - 2020

**Connects to:** Shape-specific kernels, schedule search, autotuning, and the difficulty of choosing tile and launch configurations.

**Very short abstract:** Ansor constructs tensor programs from a hierarchical search space and combines random sampling, evolutionary refinement, and a learned cost model. A task scheduler distributes tuning effort across the subgraphs that matter to an end-to-end network.

**Why it is useful:** It demonstrates how automated search can navigate tradeoffs such as reuse, occupancy, vectorization, and parallelism that are hard to optimize analytically.

**Bigger picture:** Modern compilers increasingly combine human-designed abstractions with empirical search because the fastest schedule depends on shape and hardware generation.

### 8. [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://pytorch.org/assets/pytorch2-2.pdf) - 2024

**Connects to:** `torch.compile`, graph capture, TorchInductor, Triton code generation, and fusion of eager PyTorch operations.

**Very short abstract:** PyTorch 2 introduces TorchDynamo for extracting graphs from dynamic Python programs and TorchInductor for compiling those graphs to optimized CPU or GPU code. The design preserves an eager-style programming interface while enabling fusion and backend specialization.

**Why it is useful:** It provides the systems explanation behind the lecture's observation that compiling a naive GELU expression can turn many launches into one fused Triton kernel.

**Bigger picture:** Compiler-backed eager frameworks make custom-kernel ideas available to ordinary model code while retaining escape hatches for manually tuned kernels.

### 9. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://papers.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) - 2022

**Connects to:** HBM traffic, tiling, stable softmax, fused reductions, and avoiding materialized attention matrices.

**Very short abstract:** FlashAttention reorganizes exact attention into tiled computations that reuse data in on-chip memory and avoid writing the full score and probability matrices to HBM. Its analysis treats memory reads and writes as first-class algorithmic costs rather than optimizing FLOPs alone.

**Why it is useful:** It is the clearest large-scale application of nearly every kernel principle developed in the lecture.

**Bigger picture:** FlashAttention helped shift efficient-algorithm research toward IO complexity and real hardware behavior, not only asymptotic arithmetic counts.

### 10. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://openreview.net/forum?id=mZn2Xyh9Ec) - 2024

**Connects to:** Occupancy, block scheduling, warp partitioning, shared-memory traffic, and register-pressure tradeoffs.

**Very short abstract:** FlashAttention-2 keeps the IO-aware attention algorithm but redesigns how work is divided across thread blocks and warps. It reduces non-matrix arithmetic and avoids unnecessary communication among warps to use GPU compute units more effectively.

**Why it is useful:** It shows that an algorithm can already have good asymptotic IO behavior yet still leave substantial performance on the table through poor scheduling.

**Bigger picture:** The progression from FlashAttention to FlashAttention-2 mirrors the lecture's distinction between algorithmic correctness, data movement, and hardware-specific execution efficiency.

### 11. [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) - 2024

**Connects to:** Alternatives to Triton, warp-level tiles, asynchronous pipelines, and mapping abstractions onto the GPU hierarchy.

**Very short abstract:** ThunderKittens proposes a CUDA-embedded framework built around reusable matrix-tile abstractions at warp, block, and grid levels. It aims to make hand-optimized AI kernels shorter and more maintainable while retaining direct access to modern GPU mechanisms.

**Why it is useful:** It is a concrete answer to the lecture's closing question about DSLs whose inductive biases differ from Triton's block-program model.

**Bigger picture:** Kernel programming is evolving toward several specialized abstraction layers rather than one universal progression from high-level code to raw PTX.

### 12. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) - 2024

**Connects to:** Hopper-specific instructions, asynchronous data movement, warp specialization, operation overlap, and FP8 computation.

**Very short abstract:** FlashAttention-3 redesigns exact attention for Hopper GPUs by overlapping data movement and tensor-core work, interleaving matrix multiplication with softmax, and adding an accuracy-aware FP8 path. The algorithm is explicitly co-designed with newer asynchronous hardware features.

**Why it is useful:** It demonstrates how a mature kernel must be revised when a new GPU generation changes the balance among tensor cores, memory movement, and non-matrix operations.

**Bigger picture:** High-performance kernels are portable in mathematical purpose but often generational in implementation, which is why benchmarking and profiling must be repeated on new hardware.

## Suggested reading order

1. **Start here:** 1, 2, and 5 - CUDA, Roofline, and Triton establish the execution, performance, and programming models.
2. **Then deepen:** 4, 7, and 8 - TVM, Ansor, and PyTorch 2 explain how compilers fuse graphs and search for hardware-aware schedules.
3. **Study the canonical kernel case:** 9 and 10 - FlashAttention and FlashAttention-2 connect IO-aware algorithms to block and warp scheduling.
4. **Modern extensions:** 11 and 12 - ThunderKittens and FlashAttention-3 show newer abstractions and Hopper-specific optimization.
5. **Optional compiler context:** 3 and 6 - Tensor Comprehensions and TASO broaden the view from one kernel to tensor and graph compilation.
