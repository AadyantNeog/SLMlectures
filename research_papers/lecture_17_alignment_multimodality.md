---
title: "Lecture 17 - Alignment and Multimodality: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 17
companion_notes: "../lecture_notes/lecture_17_alignment_multimodality.md"
status: "complete"
---

# Lecture 17: Alignment and Multimodality - Research Paper Guide

> The summaries below are original, compact descriptions of the papers rather than copied abstracts. Links point to primary paper or proceedings pages.

## Reading map

| Theme | Papers to prioritize |
|---|---|
| Visual encoders and image-text alignment | Vision Transformer; CLIP; OpenCLIP scaling; SigLIP |
| Connecting vision encoders to language models | Flamingo; BLIP-2; LLaVA |
| General-purpose vision-language assistants | LLaVA-OneVision; Qwen-VL; Qwen2-VL; Qwen3-VL |
| Unified multimodal generation | VQ-VAE; Chameleon; CM3Leon |

## Curated papers

### 1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - 2021

**Connects to:** image patches, visual token sequences, Transformer encoders, and pretrained vision backbones.

**Very short abstract:** Vision Transformer divides an image into fixed-size patches, linearly embeds them, and processes the resulting sequence with a standard Transformer encoder. With sufficiently large-scale pretraining, this simple architecture transfers strongly to image-recognition benchmarks.

**Why it is useful:** It provides the visual-encoder design that underlies many later vision-language systems.

**Bigger picture:** Treating image regions as tokens made it much easier to reuse language-model architectures and scaling methods across modalities.

### 2. [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html) - 2021

**Connects to:** CLIP, contrastive learning, paired image-text data, and zero-shot classification.

**Very short abstract:** CLIP jointly trains an image encoder and a text encoder to assign high similarity to matching image-caption pairs and lower similarity to mismatches in a large batch. Natural-language prompts then turn the learned shared space into a zero-shot visual classifier.

**Why it is useful:** It is the foundational reference for the contrastive alignment objective discussed in the lecture.

**Bigger picture:** CLIP showed that web-scale language supervision can produce general visual representations and a reusable bridge between vision and text.

### 3. [Reproducible Scaling Laws for Contrastive Language-Image Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Cherti_Reproducible_Scaling_Laws_for_Contrastive_Language-Image_Learning_CVPR_2023_paper.html) - 2023

**Connects to:** OpenCLIP, dataset scale, compute allocation, model size, and reproducible multimodal training.

**Very short abstract:** The authors train a large collection of open contrastive image-text models and measure how performance changes with compute, data, model size, and sample exposure. They provide empirical scaling laws and strong open checkpoints trained on large public datasets.

**Why it is useful:** It turns CLIP from a single result into a systematic study of what scales and how to allocate multimodal training resources.

**Bigger picture:** Open reproductions and scaling curves help separate architectural gains from gains caused mainly by more data or compute.

### 4. [Sigmoid Loss for Language Image Pre-Training](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html) - 2023

**Connects to:** SigLIP, pairwise sigmoid losses, contrastive objectives, and large-batch communication.

**Very short abstract:** SigLIP replaces the globally normalized softmax contrastive loss with independent sigmoid losses over image-text pairs. This simpler objective performs competitively while reducing the need for global normalization across all examples in a batch.

**Why it is useful:** It provides a modern alternative to the original CLIP loss and makes the systems consequences of objective design concrete.

**Bigger picture:** Small changes to the alignment loss can materially affect how efficiently multimodal pretraining scales across devices.

### 5. [Flamingo: a Visual Language Model for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) - 2022

**Connects to:** frozen language models, cross-attention adapters, interleaved images and text, and few-shot multimodal prompting.

**Very short abstract:** Flamingo connects a pretrained visual encoder to a pretrained language model through a resampler and gated cross-attention layers. It is trained on interleaved image-text sequences and can perform varied visual tasks from a few examples placed in the prompt.

**Why it is useful:** It is a canonical example of adding visual perception to a capable language model without retraining every component from scratch.

**Bigger picture:** Adapter-style architectures established a practical middle ground between separate encoders and fully unified multimodal token models.

### 6. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://proceedings.mlr.press/v202/li23q.html) - 2023

**Connects to:** Q-Former, frozen backbones, representation bottlenecks, and staged image-language alignment.

**Very short abstract:** BLIP-2 introduces a lightweight Querying Transformer between a frozen image encoder and a frozen language model. A two-stage training procedure first learns useful visual-language representations and then teaches those representations to condition text generation.

**Why it is useful:** It clearly isolates the connector problem: how to translate visual features into a compact form an existing language model can use.

**Bigger picture:** Efficient connector training made multimodal assistants accessible without the cost of jointly pretraining all components end to end.

### 7. [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) - 2023

**Connects to:** LLaVA, projection layers, synthetic visual instructions, and multimodal supervised fine-tuning.

**Very short abstract:** LLaVA connects a CLIP visual encoder to a language model with a learned projection and fine-tunes the system on image-grounded instruction-following data. The training conversations are generated with the help of a strong language model from image descriptions and metadata.

**Why it is useful:** It presents a minimal and influential recipe for turning pretrained vision and language components into a conversational assistant.

**Bigger picture:** The paper helped move the field from task-specific vision-language models toward general instruction-following multimodal interfaces.

### 8. [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/abs/2408.03326) - 2024

**Connects to:** single-image, multi-image, and video understanding; task transfer; and shared multimodal representations.

**Very short abstract:** LLaVA-OneVision builds a unified vision-language model for single images, multiple images, and video using a common architecture and training recipe. It studies how capabilities learned from one visual setting transfer to others and reports strong performance across a broad evaluation suite.

**Why it is useful:** It shows what is required to extend the basic LLaVA connector recipe beyond isolated still-image conversations.

**Bigger picture:** Modern multimodal assistants increasingly need one model to handle multiple visual formats and temporal contexts without bespoke task heads.

### 9. [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966) - 2023

**Connects to:** visual grounding, OCR, multilingual vision-language modeling, and coordinate-based interaction.

**Very short abstract:** Qwen-VL combines a visual encoder, an adapter, and a large language model, then trains the system in stages on broad image-text and instruction data. Beyond captioning and question answering, it supports text reading and referring to image regions through coordinates.

**Why it is useful:** It expands the notion of multimodal alignment from describing an image to grounding language in precise spatial locations.

**Bigger picture:** Useful visual assistants must connect words not only to whole images but also to documents, text, objects, and regions within them.

### 10. [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) - 2024

**Connects to:** variable-resolution inputs, multimodal rotary position embeddings, video timing, and efficient visual tokenization.

**Very short abstract:** Qwen2-VL introduces dynamic visual resolution so that images produce token counts appropriate to their size and detail. It also develops multimodal rotary position embeddings that encode spatial and temporal structure for images and videos.

**Why it is useful:** It directly addresses the mismatch between fixed image grids and real-world inputs with widely varying resolution and aspect ratio.

**Bigger picture:** Controlling visual token budgets is central to balancing perception quality against the rapidly growing context and compute cost of multimodal models.

### 11. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) - 2025

**Connects to:** multimodal reasoning, long-context visual understanding, agents, spatial perception, and generation-grounded interaction.

**Very short abstract:** Qwen3-VL extends the Qwen vision-language line with stronger visual reasoning, document and video understanding, spatial grounding, and interaction-oriented capabilities. The report describes architectural and training changes intended to make a multimodal model operate across perception, reasoning, and agentic tasks.

**Why it is useful:** It provides a recent view of how a production-scale model family combines many capabilities that earlier papers studied separately.

**Bigger picture:** The frontier is shifting from answering questions about images toward multimodal agents that perceive interfaces and act through grounded representations.

### 12. [Neural Discrete Representation Learning](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) - 2017

**Connects to:** VQ-VAE, discrete image tokens, codebooks, and autoregressive generation.

**Very short abstract:** VQ-VAE learns a discrete latent codebook and maps continuous inputs into sequences of code indices, while a decoder reconstructs the original data. A separate autoregressive prior can then model these discrete representations.

**Why it is useful:** It supplies the basic mechanism for converting images into token-like units that a generative sequence model can predict.

**Bigger picture:** Discrete visual tokenizers enable architectures that generate images and text in one autoregressive vocabulary, rather than merely conditioning text on continuous visual features.

### 13. [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818) - 2024

**Connects to:** early fusion, mixed image-text sequences, unified token vocabularies, and multimodal generation.

**Very short abstract:** Chameleon tokenizes images and text into a shared sequence format and trains an autoregressive Transformer on arbitrarily interleaved modalities. The work develops stabilization and scaling techniques needed to train this early-fusion design and supports both multimodal understanding and generation.

**Why it is useful:** It is a direct example of the lecture's unified-token approach, where images are not handled only by a separate encoder.

**Bigger picture:** Early fusion aims for a single model that can reason over and generate any mixture of modalities, at the cost of a harder joint modeling problem.

### 14. [Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning](https://arxiv.org/abs/2309.02591) - 2023

**Connects to:** CM3Leon, text-to-image generation, multimodal retrieval, instruction tuning, and autoregressive token models.

**Very short abstract:** CM3Leon trains a retrieval-augmented autoregressive model over text and discretized images, then applies multimodal instruction tuning. One model can perform text-to-image generation, editing, captioning, visual question answering, and other mixed-modal tasks.

**Why it is useful:** It demonstrates that a token-based autoregressive model can unify high-quality visual generation and language-conditioned understanding.

**Bigger picture:** CM3Leon and Chameleon represent the generative branch of multimodality, complementing encoder-plus-language-model systems such as LLaVA and Flamingo.

## Suggested reading order

1. Read Vision Transformer, CLIP, and SigLIP to understand visual tokens and contrastive alignment.
2. Use the OpenCLIP scaling study to see how data, model size, and compute shape those representations.
3. Compare Flamingo, BLIP-2, and LLaVA as three ways to connect vision to a pretrained language model.
4. Follow LLaVA-OneVision and the Qwen-VL series to track the expansion toward resolution, video, grounding, and agents.
5. Finish with VQ-VAE, CM3Leon, and Chameleon to understand unified multimodal generation.
