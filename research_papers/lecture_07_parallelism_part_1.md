---
title: "Lecture 7 - Parallelism, Part 1: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 7
companion_notes: "../lecture_notes/lecture_07_parallelism_part_1.md"
status: "complete"
---

# Lecture 7: Parallelism, Part 1 - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The collection begins with collective communication and data-parallel training, proceeds to tensor and pipeline parallelism, and closes with systems that search for or compile sharding strategies. Together, the papers connect the lecture's communication primitives to complete distributed-training designs.

| Lecture theme | Papers |
|---|---|
| Collectives, topology, and communication algorithms | 1, 3, 10-11, 13 |
| Data parallelism and useful batch-size limits | 2, 5 |
| Tensor, pipeline, and automatically planned parallelism | 4, 6-9, 12 |

## Curated papers

### 1. [Bandwidth optimal all-reduce algorithms for clusters of workstations](https://www.sciencedirect.com/science/article/pii/S0743731508001767) - 2009

**Connects to:** All-reduce, reduce-scatter plus all-gather, ring algorithms, bandwidth cost, and collective communication.

**Very short abstract:** This paper develops all-reduce algorithms designed to minimize communicated data on cluster networks. It analyzes how collective structure and message partitioning determine bandwidth use rather than treating all-reduce as a single opaque operation.

**Why it is useful:** It supplies the algorithmic foundation for understanding why ring-style collectives are bandwidth efficient and why their latency can still matter.

**Bigger picture:** Modern distributed training libraries automate collectives, but performance reasoning still begins with their communication volume and topology.

### 2. [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) - 2017

**Connects to:** Distributed data parallelism, global batch size, learning-rate scaling, warmup, and optimization stability.

**Very short abstract:** The authors show how synchronous data-parallel SGD can scale to very large minibatches while retaining model quality through a linear learning-rate rule and gradual warmup. The work makes optimization choices an explicit part of systems scaling.

**Why it is useful:** It explains why adding data-parallel workers is not purely a communication problem: the resulting global batch also changes the training dynamics.

**Bigger picture:** Efficient clusters create the option to use large batches, while statistical efficiency determines whether that option actually shortens training.

### 3. [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/abs/1802.05799) - 2018

**Connects to:** Data-parallel training, rank-local model replicas, gradient all-reduce, ring collectives, and framework integration.

**Very short abstract:** Horovod provides a small distributed-training interface built around efficient collective operations rather than parameter servers. Its design lets each worker execute familiar single-device code while synchronizing gradients across workers.

**Why it is useful:** It is a clean systems example of the lecture's standard data-parallel recipe and of why collective implementations affect end-to-end throughput.

**Bigger picture:** The work helped make synchronous all-reduce the default mental and software model for replicated deep-learning training.

### 4. [Mesh-TensorFlow: Deep Learning for Supercomputers](https://arxiv.org/abs/1811.02084) - 2018

**Connects to:** Tensor dimensions, device meshes, tensor sharding, layout annotations, and composition of parallelism dimensions.

**Very short abstract:** Mesh-TensorFlow introduces a language in which named tensor dimensions are mapped onto dimensions of a processor mesh. The mapping determines which values are sharded or replicated and which collective operations are required by each computation.

**Why it is useful:** It turns the lecture's tensor-parallel diagrams into a general declarative rule for distributing tensors over hardware.

**Bigger picture:** Device meshes and dimension-based sharding became central abstractions in later SPMD compilers and large-model frameworks.

### 5. [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - 2018

**Connects to:** Critical batch size, gradient noise scale, diminishing returns from data parallelism, and compute efficiency.

**Very short abstract:** This work builds an empirical model relating batch size, optimization progress, and the noise in stochastic gradients. It identifies a task- and training-dependent scale beyond which larger batches yield progressively less reduction in the number of optimization steps.

**Why it is useful:** It gives quantitative meaning to the lecture's warning that data parallelism eventually stops buying proportional training speedups.

**Bigger picture:** Parallelism plans must account for both hardware scaling and the optimization-limited ceiling on useful global batch size.

### 6. [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377) - 2018

**Connects to:** Pipeline stages, microbatches, bubbles, asynchronous schedules, activation storage, and weight consistency.

**Very short abstract:** PipeDream partitions a model into stages and schedules multiple minibatches through them to keep accelerators busy. It jointly considers stage placement and a pipelined execution scheme while addressing the parameter-version mismatch created by overlapping iterations.

**Why it is useful:** It exposes both the utilization benefit and the semantic complications of pipeline parallel training.

**Bigger picture:** Pipeline scheduling is a dependency-management problem as much as a model-partitioning problem.

### 7. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) - 2019

**Connects to:** Tensor parallelism, row- and column-parallel linear layers, attention heads, MLP partitioning, and communication placement.

**Very short abstract:** Megatron-LM presents an intra-layer model-parallel design for Transformer language models using a small number of communication operations. It partitions matrix multiplications so that adjacent Transformer operations can often consume distributed intermediate results without immediately gathering them.

**Why it is useful:** It is the canonical concrete derivation of the tensor-parallel Transformer patterns introduced in the lecture.

**Bigger picture:** Careful algebraic partitioning can reduce synchronization frequency, but tensor parallelism remains most attractive inside high-bandwidth hardware domains.

### 8. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://papers.neurips.cc/paper_files/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html) - 2019

**Connects to:** Pipeline parallelism, microbatching, bubble fraction, synchronous updates, and activation recomputation.

**Very short abstract:** GPipe divides a sequential network across accelerators and splits each minibatch into microbatches that flow through the stages. It preserves synchronous minibatch semantics and uses rematerialization to reduce the activation memory retained at each stage.

**Why it is useful:** It provides a particularly clear baseline for calculating pipeline utilization and understanding the memory-throughput tradeoff.

**Bigger picture:** Many later pipeline schedules can be read as attempts to reduce GPipe's bubbles or memory cost while preserving acceptable optimization semantics.

### 9. [Beyond Data and Model Parallelism for Deep Neural Networks.](https://proceedings.mlsys.org/paper_files/paper/2019/hash/b422680f3db0986ddd7f8f126baaf0fa-Abstract.html) - 2019

**Connects to:** Parallelization strategy search, operator and tensor dimensions, device topology, placement, and cost modeling.

**Very short abstract:** This work defines the SOAP search space, which can distribute samples, operators, attributes, and parameters rather than choosing only conventional data or model parallelism. FlexFlow uses simulation-guided search to find a strategy for a particular model and machine.

**Why it is useful:** It shows how many hybrid layouts are possible once placement and sharding choices are treated systematically.

**Bigger picture:** As parallel plans become multidimensional, automated search and cost models increasingly replace hand-designed strategies.

### 10. [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://vldb.org/pvldb/vol13/p3005-li.pdf) - 2020

**Connects to:** DistributedDataParallel, gradient bucketing, overlap of communication with backpropagation, process groups, and correctness.

**Very short abstract:** This paper describes the design and evolution of PyTorch's distributed data-parallel module. It focuses on reducer behavior, gradient bucketing, asynchronous collective execution, and the engineering needed to preserve eager framework semantics.

**Why it is useful:** It explains what a production DDP implementation does beyond simply inserting one all-reduce after backward propagation.

**Bigger picture:** High-level parallel APIs hide substantial scheduling work whose quality determines whether communication is exposed on the critical path.

### 11. [Blink: Fast and Generic Collectives for Distributed ML](https://proceedings.mlsys.org/paper_files/paper/2020/hash/cd3a9a55f7f3723133fa4a13628cdf03-Abstract.html) - 2020

**Connects to:** Network topology, collective synthesis, heterogeneous links, spanning trees, bandwidth, and NCCL alternatives.

**Very short abstract:** Blink generates collective communication algorithms from the measured topology of a GPU system. It packs multiple trees into the available links so that collective schedules can exploit irregular or heterogeneous connectivity.

**Why it is useful:** It makes the lecture's topology lesson concrete: the same logical all-reduce may require a different schedule on a different cluster.

**Bigger picture:** Topology-aware collective generation is one layer of the broader shift from fixed distributed recipes to machine-specific optimization.

### 12. [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) - 2021

**Connects to:** SPMD execution, tensor sharding annotations, automatic collectives, device meshes, and compiler-managed parallelism.

**Very short abstract:** GSPMD lets users describe how selected tensors should be partitioned while a compiler propagates those choices across a full computation graph. The compiler rewrites the graph into one SPMD program and inserts the communication needed to maintain semantics.

**Why it is useful:** It demonstrates how the lecture's data, tensor, and other sharding patterns can be represented and composed in a compiler rather than encoded manually.

**Bigger picture:** Compiler-generated SPMD programs are a major route to scaling model code without maintaining a separate implementation for every parallel configuration.

### 13. [TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches](https://www.usenix.org/conference/nsdi23/presentation/shah) - 2023

**Connects to:** Collective algorithms, hierarchical interconnects, topology-aware scheduling, latency-bandwidth tradeoffs, and communication synthesis.

**Very short abstract:** TACCL synthesizes collective algorithms from a cluster topology and a compact user-provided sketch of promising communication structure. This combination narrows the search space while retaining the ability to optimize schedules for specific hardware.

**Why it is useful:** It extends the lecture's collective cost models into a modern method for producing practical topology-specific algorithms.

**Bigger picture:** At large scale, communication primitives themselves become compiled artifacts tuned to the machine on which the model will run.

## Suggested reading order

1. **Start here:** 1, 3, and 10 - move from the all-reduce algorithm to complete data-parallel framework implementations.
2. **Add the optimization constraint:** 2 and 5 - understand why global batch size limits the return from additional data-parallel workers.
3. **Learn model parallelism:** 7, 8, and 6 - study tensor parallelism and two influential pipeline designs.
4. **Generalize the layout:** 4, 9, and 12 - move from device meshes to searched and compiler-generated sharding plans.
5. **Deepen communication:** 11 and 13 - see how topology-aware systems optimize the collectives beneath those plans.
