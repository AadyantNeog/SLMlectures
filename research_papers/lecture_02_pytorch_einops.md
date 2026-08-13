---
title: "Lecture 2 - PyTorch, Einops, and Resource Accounting: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 2
companion_notes: "../lecture_notes/lecture_02_pytorch_einops.md"
status: "complete"
---

# Lecture 2: PyTorch, Einops, and Resource Accounting - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The reading path starts with the programming and differentiation abstractions behind tensor code. It then builds a hardware cost model around arithmetic intensity and numerical precision, before moving to activation rematerialization, optimizer-state memory, distributed training, utilization, and compute-optimal budgeting.

| Lecture theme | Papers | Main question |
|---|---:|---|
| Tensor programs, autodiff, and optimizers | 1-3 | What state and computation does a training framework create? |
| Performance and numerical formats | 4-6 | When is an operation limited by compute, bandwidth, or precision? |
| Memory-saving training methods | 7-10 | How can activations and persistent state fit within device memory? |
| Large-model accounting and forecasting | 11-13 | How do local cost formulas guide cluster-scale training plans? |

## Curated papers

### 1. [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) - 2019

**Connects to:** Tensor storage, device placement, eager execution, and the framework used for the course assignments.

**Very short abstract:** This paper describes PyTorch's imperative, Python-oriented programming model and the runtime components that make it practical on accelerators. It explains how dynamic execution, automatic differentiation, and optimized tensor kernels coexist in one system.

**Why it is useful:** It clarifies what happens beneath familiar tensor operations and why PyTorch is both interactive and fast.

**Bigger picture:** Modern ML frameworks are the layer that translates mathematical tensor programs into memory allocation, kernels, and gradient computation.

### 2. [Automatic Differentiation in Machine Learning: a Survey](https://www.jmlr.org/papers/v18/17-468.html) - 2018

**Connects to:** Computational graphs, reverse-mode autodiff, saved activations, and backpropagation cost.

**Very short abstract:** The survey distinguishes automatic differentiation from symbolic and numerical differentiation, then develops forward- and reverse-mode methods from program traces. It also reviews implementation strategies and their relationship to machine-learning backpropagation.

**Why it is useful:** It gives a precise mental model for what PyTorch autograd records and replays.

**Bigger picture:** Reverse-mode differentiation explains both the efficiency of gradient training and much of its activation-memory footprint.

### 3. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - 2015

**Connects to:** Optimizer state, first and second moments, and persistent memory beyond model weights and gradients.

**Very short abstract:** Adam combines momentum-like first-moment estimates with adaptive second-moment scaling for stochastic optimization. Its update maintains two running tensors for every optimized parameter, together with bias correction.

**Why it is useful:** The algorithm makes the lecture's optimizer-state memory accounting concrete.

**Bigger picture:** Adaptive optimizers reduce tuning friction but add persistent state that becomes a major systems concern at language-model scale.

### 4. [Roofline: an insightful visual performance model for multicore architectures](https://www.osti.gov/pages/biblio/1407073) - 2009

**Connects to:** Arithmetic intensity, peak FLOP/s, memory bandwidth, and compute-bound versus memory-bound operations.

**Very short abstract:** The Roofline model bounds attainable performance using an application's operations per byte together with a machine's bandwidth and peak arithmetic throughput. Its visual form identifies whether locality, bandwidth, or compute is the active limitation.

**Why it is useful:** It is the foundational paper for the performance model derived in the lecture.

**Bigger picture:** Roofline reasoning connects tensor shapes and data movement to realistic wall-clock performance rather than raw FLOP counts alone.

### 5. [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - 2018

**Connects to:** FP16 storage, FP32 master weights, loss scaling, and the difference between nominal and effective precision.

**Very short abstract:** This work trains neural networks with half-precision activations, weights, and gradients while retaining selected single-precision state. Loss scaling and FP32 accumulation address the limited range and resolution of FP16.

**Why it is useful:** It explains why a mixed-precision training state cannot be accounted for using one dtype per parameter.

**Bigger picture:** Mixed precision became the template for exploiting specialized low-precision hardware without giving up stable optimization.

### 6. [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322) - 2019

**Connects to:** BF16's FP32-like exponent range, reduced mantissa precision, and wider accumulation.

**Very short abstract:** The paper evaluates BF16 training across several model families and details conversions, rounding, and mixed-precision tensor flows. Its central design argument is that keeping FP32's exponent width makes low-precision training easier to use than FP16 in many settings.

**Why it is useful:** It connects the bit layout taught in the lecture to observed training behavior and implementation choices.

**Bigger picture:** BF16 helped make reduced-precision training a default rather than a specialized optimization requiring extensive loss-scale tuning.

### 7. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) - 2016

**Connects to:** Activation checkpointing, rematerialization, and the explicit trade of extra forward work for lower peak memory.

**Very short abstract:** The authors analyze computation graphs and propose checkpoint schedules that store only selected intermediate activations. Missing values are recomputed during backward, reducing activation memory below a linear dependence on network depth.

**Why it is useful:** It is the basic research reference for the checkpointing strategy described in the lecture.

**Bigger picture:** Rematerialization turns memory capacity into an optimizable resource rather than a fixed limit on model or batch size.

### 8. [Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization](https://arxiv.org/abs/1910.02653) - 2020

**Connects to:** Hardware-aware checkpoint placement and the fact that not all activations have equal memory or recomputation cost.

**Very short abstract:** Checkmate formulates tensor rematerialization as an optimization problem over a computation graph and device-specific cost model. It searches for schedules that respect a memory budget while minimizing added execution time.

**Why it is useful:** It generalizes simple evenly spaced checkpoints into a principled schedule-selection problem.

**Bigger picture:** Compiler and runtime systems can automate compute-memory tradeoffs that are too architecture-specific for a universal hand rule.

### 9. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) - 2019

**Connects to:** Dense Transformer FLOP accounting, tensor dimensions, accelerator utilization, and models that no longer fit on one device.

**Very short abstract:** Megatron-LM partitions Transformer matrix multiplications across multiple GPUs using a small number of carefully placed collectives. The work demonstrates that model parallelism can preserve efficient large GEMMs while increasing model scale.

**Why it is useful:** It shows how tensor shapes and communication enter the resource equation once single-device accounting is insufficient.

**Bigger picture:** Tensor parallelism became one component of the multidimensional parallel strategies used for large-model training.

### 10. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - 2020

**Connects to:** Parameter, gradient, and optimizer-state memory, especially Adam's replicated moments.

**Very short abstract:** ZeRO partitions optimizer state, gradients, and eventually parameters across data-parallel workers instead of replicating every tensor on every device. It retains data-parallel computation while reducing per-device persistent memory.

**Why it is useful:** It directly operationalizes the lecture's breakdown of training memory into distinct tensor categories.

**Bigger picture:** Sharded state made data parallelism a memory-scaling technique as well as a throughput-scaling technique.

### 11. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) - 2021

**Connects to:** Achieved throughput, utilization, memory limits, and composing data, tensor, and pipeline parallelism.

**Very short abstract:** This work studies how several forms of parallelism interact on large GPU clusters and proposes an interleaved pipeline schedule. It provides quantitative guidance for choosing a parallel configuration under model-size, communication, and memory constraints.

**Why it is useful:** It extends per-operation resource accounting into an end-to-end distributed training plan.

**Bigger picture:** Cluster efficiency depends on balancing arithmetic work, communication, pipeline bubbles, and memory rather than maximizing any one metric.

### 12. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) - 2022

**Connects to:** Model FLOPs utilization, large-scale accelerator training, and the empirical gap between theoretical and realized throughput.

**Very short abstract:** PaLM reports the design and training of a large dense Transformer across thousands of TPU chips using the Pathways system. Alongside model results, the report documents architecture, data, parallel execution, and efficiency measurements.

**Why it is useful:** It is a concrete case study in turning FLOP estimates and hardware peaks into a completed frontier-scale run.

**Bigger picture:** Large-model reports connect local architectural choices to the systems efficiency and total compute budget of pretraining.

### 13. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - 2022

**Connects to:** The $6PT$ approximation, training-run forecasting, and selecting parameters and tokens under a compute budget.

**Very short abstract:** The authors fit compute-optimal relationships using many language-model training runs across sizes and token counts. The resulting allocation favors increasing both parameters and data as compute grows.

**Why it is useful:** It shows the decision that resource accounting ultimately supports: choosing a feasible and efficient training target.

**Bigger picture:** FLOP formulas measure a proposed run, while scaling laws help decide which run is worth paying for.

## Suggested reading order

1. **Start here:** 1, 2, 3, and 4 - connect tensor programs, gradients, optimizer state, and hardware limits.
2. **Numerical and memory accounting:** 5, 6, 7, and 8 - understand mixed precision and activation rematerialization.
3. **Scale beyond one device:** 9, 10, and 11 - study partitioned computation, sharded state, and composed parallelism.
4. **Close the planning loop:** 12 and 13 - compare realized utilization with compute-optimal run selection.
