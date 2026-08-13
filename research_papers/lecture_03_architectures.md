---
title: "Lecture 3 - Architectures: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 3
companion_notes: "../lecture_notes/lecture_03_architectures.md"
status: "complete"
---

# Lecture 3: Architectures - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The papers move from the original Transformer to the normalization, feed-forward, positional, and residual choices on which modern dense models converged. They then cover safeguards for stable large-scale training and attention variants designed around KV-cache and long-context costs.

| Lecture theme | Papers | Main question |
|---|---:|---|
| Transformer baseline and modern building blocks | 1-7 | Which components define a strong contemporary dense Transformer? |
| Stability at scale | 8-9 | How can architecture keep softmax logits and deep training well behaved? |
| KV-cache-efficient attention | 10-11 | How many distinct key/value heads are needed for good decoding? |
| Local-global and practical model recipes | 12-14 | How can long context and serving efficiency shape the full architecture? |

## Curated papers

### 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 2017

**Connects to:** The canonical residual, attention, feed-forward, normalization, and positional-encoding design that later work modifies.

**Very short abstract:** This paper introduces the Transformer as an attention-only sequence architecture with multi-head attention and position-wise feed-forward layers. Its original post-normalized encoder-decoder layout provides the reference point for most subsequent design changes.

**Why it is useful:** Every architectural choice in the lecture is easiest to understand as a modification of this baseline.

**Bigger picture:** The field kept the Transformer's overall skeleton while replacing several internal defaults for stability, quality, and inference efficiency.

### 2. [Layer Normalization](https://arxiv.org/abs/1607.06450) - 2016

**Connects to:** Normalizing hidden features independently for each example and the role of gain and bias parameters.

**Very short abstract:** Layer normalization computes normalization statistics across the features of one example rather than across a minibatch. This makes the transformation consistent between training and inference and suitable for sequence models.

**Why it is useful:** It provides the original definition behind both Transformer LayerNorm and later simplifications.

**Bigger picture:** Where normalization is placed in a residual block became as consequential as the normalization formula itself.

### 3. [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - 2020

**Connects to:** Pre-norm versus post-norm residual blocks, gradient flow, warmup, and training stability.

**Very short abstract:** The authors analyze how LayerNorm placement changes gradient scale at initialization in deep Transformers. Their theory and experiments explain why pre-normalized blocks can train more robustly than the original post-normalized layout.

**Why it is useful:** It turns a common architectural convention into an explicit optimization argument.

**Bigger picture:** Pre-norm residual paths became a standard foundation for scaling deep language models, with newer variants revisiting the quality-stability tradeoff.

### 4. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - 2019

**Connects to:** RMSNorm, removal of mean centering, and reducing low-intensity elementwise work.

**Very short abstract:** RMSNorm normalizes activations using their root-mean-square magnitude while omitting the mean-subtraction step of LayerNorm. The paper argues that rescaling invariance supplies much of the benefit at lower computational complexity.

**Why it is useful:** It gives the precise simplification now used by many language-model families.

**Bigger picture:** RMSNorm illustrates how small per-token operations matter when repeated across every layer and token at scale.

### 5. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - 2020

**Connects to:** SwiGLU, GeGLU, gated feed-forward networks, and parameter-matched intermediate widths.

**Very short abstract:** This study replaces the Transformer's ordinary feed-forward activation with gated linear-unit variants using several nonlinearities. It shows that gating can improve sequence-model quality under comparable parameter and compute budgets.

**Why it is useful:** It is the compact empirical source for the MLP design that modern language models commonly adopt.

**Bigger picture:** The gated MLP became one of the clearest architectural improvements retained across otherwise different model families.

### 6. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - 2021

**Connects to:** Rotary position embeddings, query/key rotations, and relative displacement in attention scores.

**Very short abstract:** RoFormer encodes position by rotating paired query and key coordinates with position-dependent angles. Their inner product then carries a structured dependence on relative position while preserving a simple attention implementation.

**Why it is useful:** It provides both the derivation and the original empirical motivation for RoPE.

**Bigger picture:** RoPE became the dominant positional mechanism for decoder-only language models and a foundation for later context-extension methods.

### 7. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) - 2022

**Connects to:** Parallel attention/MLP blocks, SwiGLU, bias-free projections, RoPE, architecture defaults, and stable large-scale training.

**Very short abstract:** PaLM documents the architecture and distributed training of a large dense Transformer on TPU pods. Its recipe combines several now-familiar design choices and reports how they behave in a completed high-scale run.

**Why it is useful:** It shows individual architectural components functioning together as a coherent production-scale recipe.

**Bigger picture:** Model reports such as PaLM helped turn scattered architectural experiments into reusable defaults for later open and closed models.

### 8. [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) - 2022

**Connects to:** Softmax-logit stability, z-loss, low-precision hazards, and interventions that protect expensive training runs.

**Very short abstract:** ST-MoE studies instability and transfer behavior in large sparsely activated Transformers and presents a set of stabilizing design practices. Among them, a z-loss penalizes large softmax normalizers without forcing individual logits toward a specific target.

**Why it is useful:** Its analysis gives a practical origin for the z-loss principle discussed in the lecture.

**Bigger picture:** Stability regularizers developed around MoE routing also expose general lessons about softmaxes, numerical range, and irreversible large-run failures.

### 9. [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442) - 2023

**Connects to:** QK normalization, attention-logit growth, parallel blocks, and architecture changes motivated by scale.

**Very short abstract:** This work develops a recipe for stably training a very large Vision Transformer and studies its scaling behavior. A central architectural change normalizes queries and keys before their dot product to control attention logits.

**Why it is useful:** It provides a clean large-scale demonstration of QK normalization even though the application is vision.

**Bigger picture:** Stability techniques often transfer across modalities because they address the shared numerical structure of Transformer layers.

### 10. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - 2019

**Connects to:** Multi-query attention, autoregressive decode, memory bandwidth, and KV-cache size.

**Very short abstract:** Multi-query attention retains separate query heads but shares one set of keys and values across them. This sharply reduces the state repeatedly loaded during incremental decoding, with a possible quality tradeoff.

**Why it is useful:** It isolates the architectural change that most directly attacks KV-cache bandwidth.

**Bigger picture:** MQA made attention-head design a serving decision, not only a representational one.

### 11. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - 2023

**Connects to:** Grouped-query attention as the compromise between full multi-head attention and MQA.

**Very short abstract:** GQA uses fewer key/value heads than query heads, interpolating between multi-head and multi-query attention. The paper also proposes converting existing multi-head checkpoints through a short uptraining stage.

**Why it is useful:** It explains why grouped-query attention is now a practical default for efficient decoders.

**Bigger picture:** GQA preserves much of multi-head expressivity while capturing most of MQA's cache and bandwidth savings.

### 12. [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - 2020

**Connects to:** Sliding-window attention, sparse interaction patterns, and mixing local with selected global connections.

**Very short abstract:** Longformer replaces dense attention with local sliding windows plus task-selected global tokens, making attention scale linearly with sequence length. It adapts pretrained Transformer encoders to substantially longer documents.

**Why it is useful:** It provides the foundational local-global attention pattern used by many later long-context systems.

**Bigger picture:** Fixed sparse patterns preceded today's layerwise hybrids of local, global, and recurrent processing.

### 13. [Mistral 7B](https://arxiv.org/abs/2310.06825) - 2023

**Connects to:** Grouped-query attention, sliding-window attention, KV-cache reduction, and hardware-aware architecture selection.

**Very short abstract:** Mistral 7B combines grouped-query attention with a sliding-window mechanism in an openly released decoder-only model. The report emphasizes strong capability per parameter together with faster and cheaper inference.

**Why it is useful:** It is a compact real-world case study of two attention modifications covered in the lecture.

**Bigger picture:** Mistral helped establish that efficient attention variants can be default architectural ingredients rather than after-the-fact serving patches.

### 14. [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) - 2024

**Connects to:** Interleaved local-global attention, grouped-query attention, logit soft-capping, and architecture as a complete recipe.

**Very short abstract:** Gemma 2 presents a family of open language models that combines local-global attention, grouped-query attention, distillation, and several training refinements. The architecture also uses soft-capping to constrain attention and output logits.

**Why it is useful:** It shows the lecture's stability and inference ideas assembled in a modern model family.

**Bigger picture:** Contemporary architectures increasingly combine conservative dense-Transformer defaults with targeted constraints for stable training and economical serving.

## Suggested reading order

1. **Start here:** 1, 2, and 3 - establish the original Transformer and why residual-normalization layout matters.
2. **Modern building blocks:** 4, 5, 6, and 7 - study RMSNorm, gated MLPs, RoPE, and a full large-model recipe.
3. **Stability:** 8 and 9 - understand z-loss and QK normalization as protections against softmax-logit growth.
4. **Inference-aware attention:** 10 and 11 - follow MQA into the GQA compromise.
5. **Modern local-global recipes:** 12, 13, and 14 - trace sparse windows into practical contemporary language models.
