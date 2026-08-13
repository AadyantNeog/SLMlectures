---
title: "Lecture 1 - Overview and Tokenization: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 1
companion_notes: "../lecture_notes/lecture_01_overview_tokenization.md"
status: "complete"
---

# Lecture 1: Overview and Tokenization - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The path begins with the Transformer and the empirical laws that make full-stack resource planning possible. It then follows tokenization from deterministic subword vocabularies to stochastic segmentation and modern byte-level models that learn their own larger units.

| Lecture theme | Papers | Main question |
|---|---:|---|
| Language-model foundations and scaling | 1-3 | What architecture and empirical laws underlie modern pretraining? |
| Learned subword tokenization | 4-7 | How can a finite vocabulary cover open-ended text robustly? |
| Bytes, characters, and learned latent patches | 8-12 | Can models remove or internalize the tokenizer without becoming prohibitively expensive? |

## Curated papers

### 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 2017

**Connects to:** The modern language-model stack and the course's emphasis on architecture, training, and systems co-design.

**Very short abstract:** This paper introduces the Transformer, replacing recurrent sequence processing with attention and position-wise feed-forward layers. Its parallel training structure became the base architecture for most contemporary language models.

**Why it is useful:** It supplies the architectural vocabulary assumed throughout the course.

**Bigger picture:** Tokenization defines the sequence presented to the Transformer, while later architecture and systems work makes that sequence economical to process.

### 2. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - 2020

**Connects to:** Predictable scaling, compute budgeting, and the value of algorithms that continue to work at larger scale.

**Very short abstract:** The authors fit empirical power laws relating language-model loss to model size, data, and training compute. They use those relationships to reason about how a fixed compute budget should be allocated.

**Why it is useful:** It shows why napkin-math forecasts can guide expensive model-development decisions.

**Bigger picture:** This work made scaling an empirical engineering discipline and set up later revisions to compute-optimal model/data allocation.

### 3. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - 2022

**Connects to:** The lecture's rough compute-optimal token rule and warning that scale requires the right recipe, not merely a larger model.

**Very short abstract:** This study trains many models across parameter and token budgets to re-estimate compute-optimal scaling. It finds that parameters and training data should grow together more aggressively than earlier common practice implied.

**Why it is useful:** It turns scaling-law intuition into a practical model-size and data-size planning rule.

**Bigger picture:** The result shifted frontier pretraining toward smaller models trained on more tokens and made dataset construction even more central.

### 4. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) - 2016

**Connects to:** Byte pair encoding, open vocabularies, and the tradeoff between whole words and individual characters.

**Very short abstract:** The paper adapts byte pair encoding to learn frequent character sequences as subword units for neural machine translation. The resulting vocabulary can represent rare and unseen words compositionally while keeping sequences shorter than character-level text.

**Why it is useful:** It is the foundational reference for the BPE procedure derived in the lecture.

**Bigger picture:** BPE became the standard compromise between vocabulary coverage and sequence length in generative language models.

### 5. [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/) - 2018

**Connects to:** Lossless round trips, pre-tokenization choices, whitespace handling, and practical tokenizer implementation.

**Very short abstract:** SentencePiece trains subword models directly from raw text instead of assuming an external word tokenizer. It provides a language-independent encoding and decoding system supporting BPE and unigram vocabularies.

**Why it is useful:** It demonstrates how tokenizer algorithms become a robust, reversible production interface.

**Bigger picture:** It helped move NLP pipelines away from language-specific preprocessing and toward self-contained model artifacts.

### 6. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://aclanthology.org/P18-1007/) - 2018

**Connects to:** The fact that a string can have several valid segmentations and that tokenization is a modeling choice rather than a natural truth.

**Very short abstract:** This paper samples alternative subword segmentations during training instead of always using one deterministic split. It also introduces a unigram language-model tokenizer that assigns probabilities to candidate segmentations.

**Why it is useful:** It explains how segmentation uncertainty can act as data augmentation and improve robustness.

**Bigger picture:** Stochastic tokenization reframes the tokenizer from a fixed preprocessing rule into part of the model's regularization strategy.

### 7. [BPE-Dropout: Simple and Effective Subword Regularization](https://aclanthology.org/2020.acl-main.170/) - 2020

**Connects to:** BPE merge ordering, deterministic encoding, and robustness to uncommon word forms.

**Very short abstract:** BPE-Dropout randomly skips some merge operations during training, exposing a model to multiple granularities while retaining the same BPE vocabulary. At inference time, ordinary deterministic BPE can still be used.

**Why it is useful:** It is a minimal modification that makes the mechanics of BPE regularization especially easy to understand.

**Bigger picture:** It bridges classic merge-based tokenizers and later approaches that learn or vary segmentation inside the model.

### 8. [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) - 2021

**Connects to:** Universal byte coverage, longer byte sequences, and the possibility of removing explicit subword tokenization.

**Very short abstract:** ByT5 replaces text subwords with UTF-8 bytes in a pretrained encoder-decoder model and reallocates capacity to deeper processing. The study examines when direct byte modeling gains robustness and where its longer sequences cost efficiency.

**Why it is useful:** It provides a strong controlled comparison between subword and byte-level modeling.

**Bigger picture:** It established that explicit token vocabularies are optional, while making clear that efficient abstraction over bytes remains necessary.

### 9. [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874) - 2022

**Connects to:** Character-level coverage, multilingual text, and learned downsampling of fine-grained inputs.

**Very short abstract:** CANINE processes Unicode characters without a fixed subword vocabulary and reduces sequence length before applying a deep Transformer stack. Its pretraining setup can operate entirely at the character level or use subwords only as a soft training signal.

**Why it is useful:** It shows a concrete architecture for separating universal input coverage from expensive contextual computation.

**Bigger picture:** Learned downsampling is one route by which tokenization can migrate from preprocessing into the network itself.

### 10. [Charformer: Fast Character Transformers via Gradient-based Subword Tokenization](https://arxiv.org/abs/2106.12672) - 2022

**Connects to:** Learned abstractions, differentiable tokenization, and the efficiency gap between bytes and subwords.

**Very short abstract:** Charformer introduces a gradient-based module that scores candidate byte blocks and constructs latent subword representations end to end. The resulting model retains byte-level input while shortening the sequence passed to deeper layers.

**Why it is useful:** It makes the lecture's idea of learned abstraction over low-level units architecturally explicit.

**Bigger picture:** It is an intermediate step between fixed tokenizers and models that dynamically choose their own computation units.

### 11. [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185) - 2023

**Connects to:** Byte-level autoregressive modeling, sequence-length cost, and hierarchical computation.

**Very short abstract:** MEGABYTE divides byte sequences into patches, using a global model across patches and a local model inside each patch. This multiscale decoder makes long byte sequences more tractable while remaining end-to-end differentiable.

**Why it is useful:** It shows how architecture can recover efficiency without reintroducing a conventional learned vocabulary.

**Bigger picture:** Hierarchical byte modeling broadens language-model interfaces to raw text and other serialized modalities.

### 12. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) - 2024

**Connects to:** Dynamic patches, tokenizer compression ratio, and the lecture's closing claim that abstraction will remain even if explicit tokens disappear.

**Very short abstract:** The Byte Latent Transformer groups bytes into variable-sized patches according to local predictability, then alternates patch-level global modeling with byte-level local modules. The design allocates more computation to difficult byte regions instead of enforcing one fixed segmentation everywhere.

**Why it is useful:** It is a modern example of replacing a static tokenizer with learned, content-dependent compute allocation.

**Bigger picture:** BLT turns tokenization into an architectural and scaling decision, linking raw-byte robustness to adaptive computation.

## Suggested reading order

1. **Start here:** 1, 4, and 5 - learn the Transformer interface, the BPE compromise, and a practical tokenizer implementation.
2. **Understand scale and uncertainty:** 2, 3, 6, and 7 - connect compute planning with the fact that segmentation is not unique.
3. **Explore token-free models:** 8, 9, and 10 - compare direct bytes, characters, downsampling, and learned latent subwords.
4. **Modern extensions:** 11 and 12 - study multiscale and dynamic-patching approaches that internalize tokenization.
