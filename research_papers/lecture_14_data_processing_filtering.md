---
title: "Lecture 14 - Data Processing and Filtering: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 14
companion_notes: "../lecture_notes/lecture_14_data_processing_filtering.md"
status: "complete"
---

# Lecture 14: Data Processing and Filtering - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

This lecture moves from raw documents to a usable training mixture. The papers below follow the same path: extraction, quality filtering, duplicate removal, mixture design, and synthetic post-training data.

| Lecture theme | Papers |
|---|---|
| HTML and PDF extraction | 1-2 |
| Web-corpus auditing and quality filtering | 3-6 |
| Exact and approximate deduplication | 7-8 |
| Repetition and mixture optimization | 9-13 |
| Synthetic reasoning and software-agent data | 14-15 |

## Curated papers

### 1. [Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction](https://aclanthology.org/2021.acl-demo.15/) - 2021

**Connects to:** HTML-to-text conversion, main-content extraction, metadata preservation, and heuristic lossiness.

**Very short abstract:** Trafilatura is an open-source pipeline for discovering and extracting the main text, comments, and metadata from web pages. The paper benchmarks it against other extraction tools and exposes the practical decisions hidden between raw HTML and corpus text.

**Why it is useful:** It turns the lecture's HTML-cleaning discussion into a concrete, inspectable implementation.

**Bigger picture:** Web-scale language modeling begins with document-reconstruction choices that already shape what the model can learn.

### 2. [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418) - 2023

**Connects to:** PDF retrieval, OCR, layout recovery, equations, tables, and scientific-document cleanup.

**Very short abstract:** Nougat trains a vision encoder-decoder to translate rendered scientific pages into structured markup. It targets the equations and layout conventions that ordinary PDF text extraction and generic OCR often lose.

**Why it is useful:** It is a modern reference for treating PDF extraction as document understanding rather than byte-to-text conversion.

**Bigger picture:** Better multimodal parsers expand the high-value technical data that can enter language-model training corpora.

### 3. [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://aclanthology.org/2021.emnlp-main.98/) - 2021

**Connects to:** C4-style rules, language identification, blocklists, provenance, toxicity filtering, and unintended exclusions.

**Very short abstract:** This audit traces the sources and contents of C4 and studies what its filters remove. It shows that apparently simple cleaning rules can introduce demographic skews, admit machine-generated material, and contaminate evaluations.

**Why it is useful:** It demonstrates why every filter needs both a quality ablation and an exclusion audit.

**Bigger picture:** Dataset documentation is part of model evaluation because corpus decisions become latent model behavior.

### 4. [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) - 2024

**Connects to:** Large-scale web processing, heuristic filtering, deduplication, educational-quality classifiers, and controlled ablations.

**Very short abstract:** FineWeb constructs a large open web corpus from many Common Crawl snapshots and reports ablations of its filtering and deduplication choices. FineWeb-Edu adds a learned educational-quality filter to produce a more targeted subset.

**Why it is useful:** It provides an unusually transparent end-to-end recipe for modern web-data curation.

**Bigger picture:** Open datasets increasingly compete through measured curation quality rather than raw token count alone.

### 5. [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794) - 2024

**Connects to:** Model-based quality filtering, standardized data experiments, token-budget tradeoffs, and DCLM.

**Very short abstract:** DataComp-LM proposes a controlled benchmark for comparing language-model data-curation strategies under fixed training budgets. Its baseline experiments highlight model-based filtering as a strong component of high-quality web-corpus construction.

**Why it is useful:** It separates improvements caused by data from those caused by changing models or training recipes.

**Bigger picture:** Data curation is becoming an empirical discipline with common pools, budgets, and downstream evaluations.

### 6. [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) - 2023

**Connects to:** phi-1, educational filtering, synthetic textbooks, code data, and quality-versus-quantity tradeoffs.

**Very short abstract:** The paper trains a small code model on a curated mixture of textbook-like material and synthetic exercises. It argues that carefully selected and generated data can produce strong capability without relying only on larger indiscriminate corpora.

**Why it is useful:** It is a clear case study of data quality functioning as an architectural-scale multiplier.

**Bigger picture:** Modern corpus design increasingly includes generated pedagogical data, not just filtered web documents.

### 7. [Deduplicating Training Data Makes Language Models Better](https://aclanthology.org/2022.acl-long.577/) - 2022

**Connects to:** Exact and approximate duplication, memorization, train-test overlap, repetitive substrings, and evaluation contamination.

**Very short abstract:** The authors identify extensive duplication in common language-model corpora and build tools for document- and substring-level removal. Deduplication reduces memorized emission and produces cleaner validation estimates while preserving or improving training efficiency.

**Why it is useful:** It gives direct empirical motivation for making deduplication a standard pipeline stage.

**Bigger picture:** Deduplication links data engineering to privacy, generalization, and trustworthy evaluation.

### 8. [On the Resemblance and Containment of Documents](https://doi.org/10.1109/SEQUEN.1997.666900) - 1997

**Connects to:** Shingles, Jaccard similarity, MinHash, near-duplicate discovery, and scalable set comparison.

**Very short abstract:** Broder formalizes document resemblance and containment through sets of fingerprints and shows how random sampling can estimate them compactly. This work supplies the mathematical basis for MinHash-style near-duplicate detection.

**Why it is useful:** It explains why the minimum-hash collision probability estimates Jaccard similarity.

**Bigger picture:** A foundational web-search technique became essential infrastructure for trillion-token dataset construction.

### 9. [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) - 2023

**Connects to:** Epoch counts, repeated data, finite high-quality sources, and UniMax-style caps.

**Very short abstract:** This study measures how repeated training behaves when unique data is limited. It shows that repetition can remain useful for several epochs but eventually yields diminishing returns, and it proposes scaling-law adjustments for data-constrained regimes.

**Why it is useful:** It replaces the vague rule "avoid too many epochs" with experimentally grounded tradeoffs.

**Bigger picture:** Mixture weights must account for both source value and how often finite examples will be replayed.

### 10. [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) - 2023

**Connects to:** Learned domain weights, small proxy models, robust optimization, and source balancing.

**Very short abstract:** DoReMi trains a small proxy model with group distributionally robust optimization to infer domain weights, then uses those weights for a larger run. The method aims to improve weak domains without requiring a hand-tuned downstream-task mixture.

**Why it is useful:** It is a principled alternative to choosing mixture percentages by intuition alone.

**Bigger picture:** Cheap proxy experiments can guide expensive pretraining decisions when the transfer assumptions are monitored.

### 11. [UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining](https://arxiv.org/abs/2304.09151) - 2023

**Connects to:** Epoch caps, head-versus-tail sources, multilingual mixtures, and avoiding repeated exposure to small corpora.

**Very short abstract:** UniMax constructs a sampling distribution that gives broad coverage to large language corpora while explicitly limiting how many times smaller corpora may repeat. Across model scales, it improves on common temperature-based multilingual sampling strategies.

**Why it is useful:** It is the direct source for the lecture's hard epoch-cap mixture rule and gives that simple rule an empirical motivation.

**Bigger picture:** Mixture design is partly a resource-allocation problem: a valuable source should receive enough weight to matter, but not so much that limited examples are memorized through excessive epochs.

### 12. [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952) - 2024

**Connects to:** Small-scale mixture sweeps, surrogate prediction, scale transfer, and continual-training mixtures.

**Very short abstract:** The paper fits functional relationships between source proportions and language-model performance, then combines them with size and step scaling laws. This allows unseen mixtures to be ranked from a limited set of exploratory runs.

**Why it is useful:** It makes explicit the assumptions behind transferring small mixture experiments to larger budgets.

**Bigger picture:** Data composition is becoming another predictable scaling dimension alongside parameters, tokens, and compute.

### 13. [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492) - 2024

**Connects to:** Regression-based mixing, many inexpensive small runs, downstream predictors, and scale transfer.

**Very short abstract:** RegMix samples many candidate mixtures, trains small models, and learns regressors that predict performance as a function of mixture weights. It then selects promising recipes for larger pretraining runs.

**Why it is useful:** It is the direct research reference for the lecture's regression-surrogate mixture strategy.

**Bigger picture:** Data search can be framed as experimental design plus response-surface modeling rather than a single manual guess.

### 14. [OpenThoughts: Data Recipes for Reasoning Models](https://arxiv.org/abs/2506.04178) - 2025

**Connects to:** Teacher-generated reasoning traces, source selection, multiple answers, filtering, and open post-training data.

**Very short abstract:** OpenThoughts systematically studies a pipeline for generating and filtering math, code, and science reasoning traces from teacher models. It releases the recipes, datasets, and trained models so data-generation choices can be reproduced and ablated.

**Why it is useful:** It shows that synthetic reasoning data quality depends on the whole generation pipeline, not only the teacher's headline capability.

**Bigger picture:** Post-training datasets are increasingly engineered experimental products with their own scaling laws.

### 15. [SWE-smith: Scaling Data for Software Engineering Agents](https://arxiv.org/abs/2504.21798) - 2025

**Connects to:** Synthetic repository tasks, executable verification, software trajectories, and scaling agent environments.

**Very short abstract:** SWE-smith constructs software-engineering tasks by modifying real repositories and validating the resulting issue-resolution environments. The generated tasks support training coding agents with trajectories whose outcomes can be checked by tests.

**Why it is useful:** It explains how data generation changes when an example must be an interactive, executable environment rather than a text pair.

**Bigger picture:** Agent training extends curation from documents to verifiable worlds in which models can act and receive feedback.

## Suggested reading order

1. **Start with the pipeline:** 1, 3, 4, and 7.
2. **Understand the algorithms:** 2, 8, and 9.
3. **Study mixture optimization:** 10, 11, 12, and 13.
4. **See modern data generation:** 5, 6, 14, and 15.
