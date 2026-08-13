---
title: "Lecture 8 - Parallelism, Part 2: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 8
companion_notes: "../lecture_notes/lecture_08_parallelism_part_2.md"
status: "complete"
---

# Lecture 8: Parallelism, Part 2 - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

This guide follows the lecture from pipeline and tensor-parallel foundations through sharded state, activation memory, experts, and context parallelism. It ends with two modern training reports that show how these methods are composed into real multidimensional systems. A few papers also appear in the Part 1 guide because they are foundational to both lectures; here, the annotations emphasize advanced composition and scheduling.

| Lecture theme | Papers |
|---|---|
| Pipeline, tensor, and multidimensional foundations | 1-3, 6-7 |
| Sharded model state and sparse experts | 4-5, 8, 10 |
| Activation memory, modern schedules, and context parallelism | 9, 11-12 |
| Parallelism recipes in frontier-scale training | 13-14 |

## Curated papers

### 1. [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377) - 2018

**Connects to:** Pipeline schedules, multiple in-flight microbatches, stage balancing, weight versions, bubbles, and activation memory.

**Very short abstract:** PipeDream jointly partitions a network into stages and schedules a stream of minibatches through those stages. Its asynchronous pipeline improves utilization but requires explicit handling of parameter versions so that forward and backward computations remain compatible.

**Why it is useful:** It makes clear that reducing pipeline bubbles can introduce consistency and memory complications elsewhere in the training system.

**Bigger picture:** Advanced pipeline schedules are best compared along several axes at once: utilization, memory, communication, and the exact optimization semantics they preserve.

### 2. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://papers.neurips.cc/paper_files/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html) - 2019

**Connects to:** Synchronous pipeline parallelism, microbatches, bubble size, activation recomputation, and stage partitioning.

**Very short abstract:** GPipe partitions a sequential model into accelerator-resident stages and divides each minibatch into microbatches. It uses a flush-based synchronous schedule together with activation rematerialization to train models that exceed one device's memory.

**Why it is useful:** Its simple schedule is the reference point from which interleaving, one-forward-one-backward, and zero-bubble variants can be understood.

**Bigger picture:** Pipeline research repeatedly trades a baseline's conceptual simplicity for better utilization or lower peak memory.

### 3. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) - 2019

**Connects to:** Tensor parallelism, attention-head partitioning, row- and column-parallel linear layers, and intra-node collectives.

**Very short abstract:** Megatron-LM introduces a Transformer-specific tensor-parallel scheme that shards major matrix multiplications across GPUs. The partition choices allow several local operations to be fused between synchronization points and require little change to the model definition.

**Why it is useful:** It supplies the core tensor-parallel building blocks later composed with pipeline and data parallelism in large training runs.

**Bigger picture:** Tensor parallelism is a fine-grained way to enlarge a model replica, but its frequent communication usually binds it to the fastest interconnect tier.

### 4. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap379.html) - 2020

**Connects to:** ZeRO stages 1-3, sharded optimizer state, gradients and parameters, all-gather, reduce-scatter, and memory accounting.

**Very short abstract:** ZeRO removes redundant model-state storage from data-parallel workers by partitioning optimizer states, gradients, and eventually parameters. It retains data-parallel computation while reconstructing or reducing state through collective communication when needed.

**Why it is useful:** It is the defining paper for the lecture's staged progression from optimizer sharding to fully sharded training.

**Bigger picture:** ZeRO reframes data-parallel replicas as temporary computational views of globally sharded state rather than permanently complete copies.

### 5. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) - 2020

**Connects to:** Mixture-of-experts, expert parallelism, token routing, all-to-all communication, capacity constraints, and compiler sharding.

**Very short abstract:** GShard combines sparsely activated mixture-of-experts layers with lightweight sharding annotations and an SPMD compiler. Tokens are routed among distributed experts so parameter count can grow much faster than the computation applied to any one token.

**Why it is useful:** It connects the lecture's expert-parallel communication pattern to a complete compiler-assisted training design.

**Bigger picture:** Sparse models exchange dense computation for routing and load-balancing challenges, making systems design part of the model architecture.

### 6. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) - 2021

**Connects to:** 3D parallelism, tensor plus pipeline plus data parallelism, topology-aware process groups, schedules, and throughput modeling.

**Very short abstract:** This work composes tensor, pipeline, and data parallelism into a scalable training system for large Transformer language models. It analyzes how the dimensions interact and maps communication-heavy tensor parallelism differently from wider data-parallel communication.

**Why it is useful:** It is the most direct paper for understanding the lecture's practical 3D-parallel training recipe.

**Bigger picture:** No single parallelism method dominates at large scale; successful systems assign different model dimensions to different levels of the cluster topology.

### 7. [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) - 2021

**Connects to:** SPMD execution, multidimensional device meshes, sharding propagation, automatic resharding, and compiler-generated collectives.

**Very short abstract:** GSPMD compiles a tensor computation with partial sharding annotations into a single program executed by every device. It propagates layouts through the graph and introduces collective communication where producer and consumer layouts differ.

**Why it is useful:** It shows how complex combinations of data, tensor, expert, and other parallel dimensions can be expressed without hand-writing distinct distributed model code.

**Bigger picture:** As parallelism becomes higher-dimensional, compiler support becomes essential for managing layouts and the resharding boundaries between them.

### 8. [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://www.vldb.org/pvldb/vol16/p3848-huang.pdf) - 2023

**Connects to:** Fully sharded data parallelism, FSDP units, parameter prefetch, communication-compute overlap, rate limiting, and memory management.

**Very short abstract:** This paper presents PyTorch FSDP as an implementation of fully sharded training integrated with the framework's tensor, dispatcher, allocator, and initialization machinery. It explains how parameters are gathered unit by unit and how prefetching, ordering, and memory controls address the resulting communication and allocation pressure.

**Why it is useful:** It turns the abstract ZeRO-3 memory diagram into the concrete lifecycle of parameters and gradients in a production framework.

**Bigger picture:** The practical quality of sharded training depends as much on overlap, initialization, and allocator behavior as on the asymptotic memory formula.

### 9. [Reducing Activation Recomputation in Large Transformer Models](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html) - 2023

**Connects to:** Activation memory, selective recomputation, tensor-parallel communication, sequence parallelism, and memory-throughput tradeoffs.

**Very short abstract:** The authors analyze which Transformer activations dominate memory when training with model parallelism and propose selective recomputation rather than checkpointing every layer uniformly. They also use sequence parallelism for operations that need not be replicated across the tensor-parallel group.

**Why it is useful:** It explains how activation memory can remain the bottleneck even after model states are sharded, and why recomputation should be chosen at the operation level.

**Bigger picture:** Memory optimization is a portfolio: sharded states, sequence layouts, and selective rematerialization address different terms in the per-device budget.

### 10. [Tutel: Adaptive Mixture-of-Experts at Scale](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html) - 2023

**Connects to:** Expert parallelism, all-to-all routing, load imbalance, adaptive parallel strategies, and MoE communication optimization.

**Very short abstract:** Tutel is a mixture-of-experts system that optimizes routing and expert computation while adapting its parallel strategy to the model and cluster configuration. It separates several MoE execution concerns so that expert workloads can be tuned across different scales and network conditions.

**Why it is useful:** It provides a systems-level view of the all-to-all, dispatch, and load-balancing costs behind sparse expert layers.

**Bigger picture:** MoE efficiency depends on coordinating model routing decisions with hardware topology and implementation strategy.

### 11. [Zero Bubble (Almost) Pipeline Parallelism](https://openreview.net/forum?id=tuzTN0eIO5) - 2024

**Connects to:** Pipeline bubbles, splitting backward into input-gradient and parameter-gradient work, schedule search, and memory constraints.

**Very short abstract:** This paper decomposes backward computation into pieces with different dependency constraints and schedules them separately across pipeline stages. The added flexibility enables schedules with little or no bubble under suitable memory budgets while preserving synchronous training.

**Why it is useful:** It directly develops the lecture's central insight that finer-grained scheduling can fill idle pipeline slots that ordinary forward/backward units cannot.

**Bigger picture:** Better pipeline utilization increasingly comes from exposing and exploiting dependencies within training operations, not merely increasing the number of microbatches.

### 12. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) - 2023

**Connects to:** Context parallelism, sequence sharding, ring communication, blockwise attention, and communication-compute overlap.

**Very short abstract:** Ring Attention distributes a long sequence across devices and rotates key-value blocks around a ring while each device computes blockwise attention. Communication is overlapped with local attention work, allowing the standard attention operation to span contexts that do not fit on one device.

**Why it is useful:** It is a direct research realization of the lecture's context-parallel pattern and its ring-shaped movement of key-value blocks.

**Bigger picture:** Context length is itself a scalable parallel dimension, especially once model state is already distributed by tensor, pipeline, or sharded data parallelism.

### 13. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) - 2024

**Connects to:** Large-scale dense-model training, 4D parallelism, context parallelism, network topology, reliability, and training infrastructure.

**Very short abstract:** The report describes the model design, data, training, evaluation, and safety work behind the Llama 3 family. Its systems discussion details a large GPU deployment that combines several parallel dimensions and treats faults and network behavior as routine design constraints.

**Why it is useful:** It shows how the lecture's individual techniques are assembled and operated in a real long-running frontier-scale training job.

**Bigger picture:** At this scale, parallelism, observability, data pipelines, and fault recovery form one coupled training system rather than independent components.

### 14. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - 2024

**Connects to:** Mixture-of-experts, expert and pipeline parallelism, communication-compute overlap, load balancing, and low-precision training.

**Very short abstract:** DeepSeek-V3 describes a sparse language model and a training system designed around efficient expert routing, reduced auxiliary load-balancing pressure, low-precision computation, and large-cluster execution. The report emphasizes communication scheduling and node-limited expert routing as core parts of achieving usable MoE throughput.

**Why it is useful:** It is a modern case study of how expert parallelism must be co-designed with topology, numerical formats, and pipeline execution.

**Bigger picture:** Frontier training recipes increasingly combine architectural sparsity with systems techniques that hide or constrain the communication it creates.

## Suggested reading order

1. **Start with the parallel axes:** 3, 2, and 4 - learn tensor parallelism, the pipeline baseline, and ZeRO-style sharding.
2. **See the composition:** 6 and 7 - study a hand-designed 3D system and a compiler-managed multidimensional alternative.
3. **Work through memory:** 8 and 9 - connect sharded states, activation storage, prefetching, and recomputation.
4. **Add sparse and long-context dimensions:** 5, 10, and 12 - examine expert routing and context-parallel attention.
5. **Study modern scheduling:** 1 and 11 - compare asynchronous pipelining with a near-zero-bubble synchronous design.
6. **Finish with full training recipes:** 13 and 14 - inspect how dense and sparse frontier models combine these ideas in practice.
