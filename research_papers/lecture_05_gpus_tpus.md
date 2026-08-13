---
title: "Lecture 5 - GPUs, TPUs, and Hardware-Aware Performance: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 5
companion_notes: "../lecture_notes/lecture_05_gpus_tpus.md"
status: "complete"
---

# Lecture 5: GPUs, TPUs, and Hardware-Aware Performance - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The path begins with a machine-independent performance model and then examines the matrix engines and memory hierarchies of TPUs and GPUs. It follows the move from FP16 to FP8 and block-scaled microscaling formats, introduces compilers that express fusion and tiling, and ends with online softmax and the three generations of FlashAttention.

| Lecture theme | Papers | Main question |
|---|---:|---|
| Roofline and accelerator anatomy | 1-4 | How do throughput, bandwidth, execution units, and memory hierarchy bound a kernel? |
| Low and mixed precision | 5-7 | How can narrower formats increase speed without destabilizing training? |
| Fusion, tiling, and kernel compilers | 8-9 | How can high-level tensor programs map efficiently onto accelerator tiles? |
| Online softmax and FlashAttention | 10-13 | How can exact attention be reordered around the memory hierarchy? |

## Curated papers

### 1. [Roofline: an insightful visual performance model for multicore architectures](https://www.osti.gov/pages/biblio/1407073) - 2009

**Connects to:** Arithmetic intensity, peak throughput, memory bandwidth, and the matrix-performance puzzle.

**Very short abstract:** The Roofline model bounds an application's attainable arithmetic rate using operations per byte, memory bandwidth, and peak compute. Its ceilings reveal whether more locality, vectorization, or arithmetic hardware can improve a workload.

**Why it is useful:** It is the foundational performance model used to organize the lecture's hardware reasoning.

**Bigger picture:** Roofline analysis explains why reducing data movement can matter more than reducing nominal FLOPs.

### 2. [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760) - 2017

**Connects to:** TPUs, matrix-multiply units, software-managed on-chip memory, specialization, latency, and throughput.

**Very short abstract:** This paper analyzes an early production TPU, a domain-specific inference ASIC centered on a large matrix multiply unit and explicitly managed memory. It compares the design and deployed workloads with contemporary CPU and GPU systems.

**Why it is useful:** It gives a concrete architectural counterpoint to the GPU execution model in the lecture.

**Bigger picture:** TPUs demonstrate how regular neural-network workloads can justify replacing general control machinery with dense arithmetic and predictable dataflow.

### 3. [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826) - 2018

**Connects to:** Streaming multiprocessors, warps, caches, shared memory, registers, instruction behavior, and memory transactions.

**Very short abstract:** The authors use targeted microbenchmarks and instruction disassembly to infer details of NVIDIA's Volta GPU architecture. They characterize execution and memory behavior that is not fully exposed by public high-level specifications.

**Why it is useful:** It shows how the practical GPU mental model can be measured rather than treated as a purely conceptual diagram.

**Bigger picture:** Kernel optimization depends on concrete architectural details such as transaction width, hierarchy capacity, and scheduling behavior.

### 4. [NVIDIA Tensor Core Programmability, Performance & Precision](https://arxiv.org/abs/1803.04014) - 2018

**Connects to:** Tensor cores, matrix-multiply-accumulate primitives, tiling, mixed accumulation, and achieved versus theoretical throughput.

**Very short abstract:** This study examines several programming interfaces for Volta Tensor Cores and measures their matrix-multiplication performance and numerical behavior. It highlights both the throughput opportunity and the precision considerations of specialized mixed-precision matrix units.

**Why it is useful:** It connects the abstract statement that GPUs are matrix machines to the programming granularity of actual tensor hardware.

**Bigger picture:** Tensor-core tile shapes increasingly influence model dimensions, numerical formats, and kernel design.

### 5. [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - 2018

**Connects to:** FP16 computation, wider master weights and accumulators, loss scaling, memory savings, and numerical stability.

**Very short abstract:** Mixed-precision training stores and computes many tensors in FP16 while retaining selected FP32 state and using loss scaling to protect small gradients. The method lets neural networks exploit low-precision hardware without simply performing every operation in a narrow dtype.

**Why it is useful:** It establishes the systems-and-numerics pattern behind later BF16, FP8, and microscaling approaches.

**Bigger picture:** Low precision succeeds as a mixed numerical system, not as a global replacement of every tensor by the same format.

### 6. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) - 2022

**Connects to:** E4M3, E5M2, range-versus-resolution tradeoffs, low-precision matrix engines, and wider accumulation.

**Very short abstract:** The paper proposes two eight-bit floating-point encodings with complementary exponent and mantissa allocations and evaluates them in training and inference. It defines an interchange format intended to support different tensor roles while retaining a practical implementation path.

**Why it is useful:** It gives the formal design rationale for the FP8 formats discussed in the lecture.

**Bigger picture:** FP8 made tensor-specific format selection and scaling a standard part of accelerator-aware model training.

### 7. [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) - 2023

**Connects to:** Block scaling, MX formats, MXFP8, MXFP4, and the hidden metadata behind a nominal element bit width.

**Very short abstract:** Microscaling pairs narrow per-element floating-point or integer values with a scale shared by a small block. The paper evaluates how this representation balances hardware efficiency, model accuracy, and the effort required to adopt sub-eight-bit computation.

**Why it is useful:** It directly explains why block-scaled formats can retain useful dynamic range with extremely small element types.

**Bigger picture:** Shared scales shift numerical representation from an element property toward a tensor-layout and kernel-co-design decision.

### 8. [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) - 2018

**Connects to:** Operator fusion, tiling, memory-latency hiding, hardware-specific schedules, and autotuning.

**Very short abstract:** TVM provides graph-level and operator-level transformations for compiling deep-learning workloads across varied hardware. It uses a learned cost model to search low-level schedules including fusion, tiling, vectorization, and memory placement.

**Why it is useful:** It shows how the lecture's optimization techniques can be represented and searched systematically.

**Bigger picture:** Tensor compilers turn hardware-aware implementation from a collection of hand-written kernels into a reusable optimization stack.

### 9. [Triton: An intermediate language and compiler for tiled neural network computations](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations) - 2019

**Connects to:** Program tiles, shared-memory reuse, coalescing, fusion, alignment, and custom GPU kernels.

**Very short abstract:** Triton introduces a language and compiler in which programmers manipulate parameterized multidimensional tiles rather than individual GPU threads. Compiler passes map those tile programs to efficient GPU code for operations such as matrix multiplication and convolution.

**Why it is useful:** It expresses the lecture's tiling mental model directly as a programming abstraction.

**Bigger picture:** Triton sits between high-level tensor frameworks and CUDA, enabling research kernels to be concise without discarding hardware structure.

### 10. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) - 2018

**Connects to:** Online softmax, running maxima and normalizers, fewer memory accesses, and fused softmax operations.

**Very short abstract:** This paper derives a single-pass online update for the maximum and normalization term required by exact softmax. The recurrence reduces data passes and also supports fusing softmax with related selection work.

**Why it is useful:** It provides the mathematical primitive that lets FlashAttention process score tiles without storing an entire row.

**Bigger picture:** A numerically stable streaming reduction can change an algorithm's memory traffic even when its mathematical output is unchanged.

### 11. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) - 2022

**Connects to:** HBM traffic, shared-memory tiling, online softmax, fusion, and backward recomputation.

**Very short abstract:** FlashAttention computes exact dense attention in tiles chosen around the GPU memory hierarchy and never materializes the full score matrix in HBM. It combines tiled matrix multiplications with online normalization and an IO-complexity analysis.

**Why it is useful:** It is the central worked example of every hardware-aware optimization principle in the lecture.

**Bigger picture:** FlashAttention showed that an asymptotically unchanged algorithm can become dramatically better through a different schedule of data movement.

### 12. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - 2023

**Connects to:** Thread-block and warp partitioning, occupancy, non-matmul overhead, and wave utilization.

**Very short abstract:** FlashAttention-2 reorganizes the original algorithm to reduce non-matrix operations and distribute work more effectively across thread blocks and warps. The mathematical attention operation remains exact while the implementation uses GPU resources more efficiently.

**Why it is useful:** It shows that good IO complexity is only the beginning; work partitioning still determines achieved utilization.

**Bigger picture:** Successive kernel generations increasingly optimize scheduling details after the high-level algorithm has already been made IO-aware.

### 13. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) - 2024

**Connects to:** Hopper Tensor Cores, asynchronous data movement, warp specialization, FP8, block quantization, and accuracy-aware low precision.

**Very short abstract:** FlashAttention-3 pipelines data movement with tensor-core computation and interleaves matrix multiplication with softmax on Hopper GPUs. It also develops an FP8 path using block quantization and transformations designed to reduce low-precision error.

**Why it is useful:** It demonstrates how the same attention algorithm must evolve with each accelerator generation.

**Bigger picture:** Modern kernel design is hardware-software-numerics co-design: scheduling, memory, tensor units, and representation are optimized together.

## Suggested reading order

1. **Start here:** 1, 2, and 3 - learn the Roofline model and compare TPU and GPU execution structures.
2. **Matrix units and precision:** 4, 5, 6, and 7 - connect tensor cores to mixed, FP8, and block-scaled numerical systems.
3. **Express hardware-aware programs:** 8 and 9 - study compiler search and explicit tile-level programming.
4. **Derive exact IO-aware attention:** 10 and 11 - read online softmax immediately before FlashAttention.
5. **Modern extensions:** 12 and 13 - follow work partitioning into Hopper-era asynchronous and FP8 kernels.
