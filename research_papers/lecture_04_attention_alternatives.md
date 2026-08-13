---
title: "Lecture 4 - Attention Alternatives and Mixtures of Experts: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 4
companion_notes: "../lecture_notes/lecture_04_attention_alternatives.md"
status: "complete"
---

# Lecture 4: Attention Alternatives and Mixtures of Experts - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The first half follows efficient sequence modeling from associative linear attention to selective state-space models, gated delta updates, deployed hybrid attention, and learned sparse retrieval. The second half follows sparse conditional computation from early top-$K$ routing through stability and GPU systems, ending with the DeepSeek MoE, latent-attention, load-balancing, and multi-token-prediction designs discussed in the lecture.

| Lecture theme | Papers | Main question |
|---|---:|---|
| Linear attention and recurrent state | 1-4 | How can parallel training and fixed-state decoding describe the same sequence operator? |
| Modern hybrid and sparse attention | 5-6 | How do large models preserve selected full-attention capabilities at long context? |
| MoE routing, balance, and stability | 7-10 | How can many parameters be trained while activating only a few per token? |
| Sparse systems and DeepSeek co-design | 11-14 | How do kernels, routing, cache compression, and auxiliary objectives fit together? |

## Curated papers

### 1. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) - 2020

**Connects to:** Associative reassociation, linear attention, recurrent inference, and training-inference duality.

**Very short abstract:** The paper expresses attention through kernel feature maps so that matrix multiplication can be reassociated without forming a full token-by-token matrix. The same operator admits both a parallel sequence computation and an iterative recurrent update.

**Why it is useful:** It is the clearest foundational derivation of the linear/recurrent equivalence used in the lecture.

**Bigger picture:** It opened a design space in which attention-like models trade explicit retrieval over all prior tokens for a compressed recurrent state.

### 2. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - 2023

**Connects to:** Input-dependent forgetting, selective recurrence, hardware-aware scans, and linear-time sequence models.

**Very short abstract:** Mamba makes state-space parameters depend on the current input so the model can selectively retain or discard information. It couples this mechanism with a hardware-aware parallel algorithm and an architecture that does not require attention layers.

**Why it is useful:** It explains why selectivity is crucial when a fixed-size state must process discrete language.

**Bigger picture:** Mamba renewed interest in recurrent foundation models by aligning expressivity, linear sequence cost, and accelerator implementation.

### 3. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) - 2024

**Connects to:** Mamba-2, gated linear attention, semiseparable matrices, and dual parallel/recurrent algorithms.

**Very short abstract:** This work develops a structured state-space duality that places selective SSMs and several attention variants in one matrix framework. It uses the framework to derive Mamba-2 and more efficient algorithms for its core layer.

**Why it is useful:** It supplies the formal bridge behind the lecture's view of Mamba-2 as gated linear attention.

**Bigger picture:** The paper suggests that attention and state-space models are related factorizations rather than isolated architecture families.

### 4. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) - 2024

**Connects to:** Gated DeltaNet, controlled writes, selective erasure, and recurrent state as an associative memory.

**Very short abstract:** Gated Delta Networks combine Mamba-2-style forgetting with a delta-rule update that corrects the memory along the current key direction before writing. The design is evaluated as a language-modeling alternative to standard attention and earlier linear recurrent layers.

**Why it is useful:** It gives a concrete mechanism for distinguishing forgetting the whole state from replacing one keyed association.

**Bigger picture:** Delta updates make recurrent state behave more like an editable memory, narrowing part of the retrieval gap with softmax attention.

### 5. [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) - 2025

**Connects to:** Hybrid linear/full attention, million-token contexts, MoE, and computation-communication overlap.

**Very short abstract:** MiniMax-01 scales a foundation-model family built around Lightning Attention interleaved with other model components and sparse experts. The report treats long-context algorithms, MoE parallelism, and cluster execution as one coordinated design.

**Why it is useful:** It is a deployed-scale example of the lecture's claim that efficient recurrent attention usually appears in hybrids.

**Bigger picture:** Hybrid models preserve periodic high-capacity interaction while moving most sequence processing to cheaper mechanisms.

### 6. [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) - 2025

**Connects to:** DeepSeek Sparse Attention, a learned indexer, top-$K$ token selection, and long-context efficiency.

**Very short abstract:** DeepSeek-V3.2 introduces a learned sparse-attention mechanism that identifies a bounded subset of historical tokens for the main attention computation. The full model combines this architectural change with large-scale reasoning and agentic post-training.

**Why it is useful:** It documents the modern sparse-selection mechanism analyzed directly in the lecture.

**Bigger picture:** Learned token retrieval offers a middle ground between fixed-state recurrence and quadratic dense attention, while retaining an indexing cost of its own.

### 7. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) - 2017

**Connects to:** Top-$K$ token routing, sparse expert activation, capacity, load balance, and expert parallelism.

**Very short abstract:** This paper introduces a sparsely gated MoE layer that routes each example to a small subset of a much larger expert pool. It develops gating and balancing mechanisms that let total parameter capacity grow without proportional per-example computation.

**Why it is useful:** It is the foundational large-scale neural MoE reference for nearly every routing issue in the lecture.

**Bigger picture:** Conditional computation separates total model capacity from active FLOPs, but replaces dense simplicity with routing and communication challenges.

### 8. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) - 2021

**Connects to:** Top-1 routing, expert collapse, auxiliary load-balancing loss, capacity factors, and selective precision.

**Very short abstract:** Switch Transformer simplifies sparse MoE routing by sending each token to one expert and studies the resulting training and systems behavior at large scale. It emphasizes balancing, capacity management, and numerical choices needed to make sparse routing reliable.

**Why it is useful:** Its router and auxiliary-loss formulation closely matches the standard MoE baseline derived in the lecture.

**Bigger picture:** Simplified top-1 routing made MoE scaling more practical and established a reference point for later fine-grained, dropless, and auxiliary-loss-free designs.

### 9. [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368) - 2022

**Connects to:** Expert-choice versus token-choice routing, capacity balance, and variable experts per token.

**Very short abstract:** Expert Choice reverses the usual assignment direction: each expert selects its highest-scoring tokens up to a fixed capacity. This guarantees balanced expert loads while allowing individual tokens to receive different amounts of expert computation.

**Why it is useful:** It exposes the modeling tradeoffs hidden by the apparently simple phrase "top-$K$ routing."

**Bigger picture:** Routing is a constrained assignment problem, and changing who chooses whom can move complexity between load balance, token coverage, and distributed execution.

### 10. [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) - 2022

**Connects to:** Router z-loss, float32 routing, training instability, token dropping, and fine-tuning sparse models.

**Very short abstract:** ST-MoE systematically studies failure modes in large sparse expert models and proposes a stable design recipe for pretraining and transfer. It addresses router-logit growth, numerical precision, balancing, and downstream adaptation.

**Why it is useful:** It is a practical guide to the softmax and optimization problems unique to sparse discrete routing.

**Bigger picture:** MoE quality gains only matter when the router remains numerically stable and the pretrained specialization survives adaptation.

### 11. [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) - 2022

**Connects to:** Dropless execution, block-sparse kernels, padding waste, and hardware-aware expert computation.

**Very short abstract:** MegaBlocks reformulates dynamic MoE computation as block-sparse matrix operations and implements specialized GPU kernels. The system processes every routed token without requiring either fixed-capacity token dropping or extensive padding.

**Why it is useful:** It shows why an algorithmically appealing router needs a matching sparse-kernel implementation.

**Bigger picture:** Better systems can relax modeling constraints that were originally imposed only because dense hardware primitives handled irregular workloads poorly.

### 12. [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) - 2024

**Connects to:** Fine-grained experts, shared experts, specialization, and active-parameter efficiency.

**Very short abstract:** DeepSeekMoE divides conventional experts into smaller routed units and reserves shared experts for common knowledge. This design aims to increase the number of possible expert combinations and reduce redundant knowledge across routed experts.

**Why it is useful:** It directly motivates the fine-grained and shared-expert design axes emphasized in the lecture.

**Bigger picture:** Modern MoEs improve not only the router but also the granularity and division of labor inside the expert pool.

### 13. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) - 2024

**Connects to:** DeepSeekMoE at scale, multi-head latent attention, KV-cache compression, and economical inference.

**Very short abstract:** DeepSeek-V2 combines fine-grained sparse experts with a latent attention representation that compresses keys and values for decoding. The report treats training cost, active parameters, long context, and serving throughput as joint architectural objectives.

**Why it is useful:** It connects the lecture's MoE discussion to MLA rather than treating expert sparsity and attention-cache cost separately.

**Bigger picture:** Efficient foundation models increasingly combine conditional MLP computation with compressed attention state.

### 14. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - 2024

**Connects to:** Auxiliary-loss-free load balancing, topology-aware expert placement, multi-token prediction, MLA, and large-scale MoE stability.

**Very short abstract:** DeepSeek-V3 scales the V2 architecture and introduces a feedback-controlled balancing strategy that avoids relying on a conventional auxiliary balance loss. It also adds multi-token prediction and documents an extensive training and systems stack for a large sparse model.

**Why it is useful:** It is the culminating case study for the architecture-systems co-design traced throughout the lecture.

**Bigger picture:** The report shows MoE evolution moving from isolated routing tricks toward coordinated control, communication, cache, precision, and training-objective design.

## Suggested reading order

1. **Start with efficient sequence models:** 1, 2, and 3 - derive linear recurrence, then connect Mamba and Mamba-2 through state-space duality.
2. **Deepen the memory mechanism:** 4, 5, and 6 - compare delta updates, deployed hybrids, and learned sparse retrieval.
3. **Build the MoE foundation:** 7, 8, 9, and 10 - study sparse gating, balance, alternative assignment, and stability.
4. **Systems and modern co-design:** 11, 12, 13, and 14 - move from dropless kernels to the DeepSeek expert, cache, balance, and prediction stack.
