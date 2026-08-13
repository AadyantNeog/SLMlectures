---
title: "Lecture 13 - Data I: Sources and Datasets - Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 13
companion_notes: "../lecture_notes/lecture_13_data_sources_datasets.md"
status: "complete"
---

# Lecture 13: Data I - Sources and Datasets - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The reading path follows the historical move from opaque mixtures assembled for model training to open, controlled, and provenance-aware corpus research. It also includes legal and licensing scholarship because access, copyright, attribution, and downstream model behavior are inseparable from data-source selection.

The legal papers explain analytical frameworks, not case-specific legal advice. Copyright rules and litigation outcomes vary by jurisdiction and continue to change.

| Lecture theme | Papers |
|---|---|
| Early large-model mixtures and Common Crawl processing | 1-5 |
| Web-only and openly documented corpora | 6, 9-11 |
| Provenance, licensing, and fair use | 7-8, 14 |
| Code and software-history data | 12 |
| Long-horizon and synthetic web transformation | 13 |

## Curated papers

### 1. [Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) - 2020

**Connects to:** GPT-3's pretraining mixture, Common Crawl, WebText, books, Wikipedia, deduplication, and source weighting.

**Very short abstract:** GPT-3 scales autoregressive language modeling and studies in-context task performance without gradient updates. Its training-data section documents a weighted mixture of filtered web crawl, web links, books, and Wikipedia, together with quality filtering and overlap removal.

**Why it is useful:** It is a foundational example of how corpus composition became a central but only partly transparent component of a frontier model recipe.

**Bigger picture:** Later open-dataset projects were partly responses to the difficulty of reproducing or auditing mixtures described only at a high level.

### 2. [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://aclanthology.org/2020.lrec-1.494/) - 2020

**Connects to:** Common Crawl, language identification, document deduplication, perplexity filtering, and multilingual web data.

**Very short abstract:** CCNet presents a pipeline that turns Common Crawl snapshots into document-preserving monolingual corpora. It combines language identification and deduplication with an optional language-model score that ranks documents by similarity to a curated reference such as Wikipedia.

**Why it is useful:** It is the primary reference for the lecture's contrast between learned quality filtering and hand-written C4 rules.

**Bigger picture:** CCNet helped establish raw web archives as inputs to reproducible processing pipelines rather than ready-made text datasets.

### 3. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) - 2020

**Connects to:** C4, rule-based web filtering, Common Crawl snapshots, data ablations, and the T5 training recipe.

**Very short abstract:** T5 casts many language tasks into a common text-to-text format and systematically compares objectives, architectures, data, and transfer methods. The work introduces C4, a large English corpus produced from Common Crawl through explicit heuristic filtering and deduplication rules.

**Why it is useful:** It ties a highly influential model study to one of the most reused early open web corpora.

**Bigger picture:** C4 showed the scientific value of releasing a corpus recipe while later audits revealed how simple filters encode strong inclusion choices.

### 4. [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) - 2020

**Connects to:** Multi-domain mixtures, explicit component weights, books, code, papers, forums, and open corpus construction.

**Very short abstract:** The Pile assembles a large English training corpus from 22 named components spanning web text and academic, professional, conversational, and technical sources. The paper documents construction choices, evaluates component difficulty, and audits potentially concerning content.

**Why it is useful:** It is the clearest foundational example of replacing one anonymous web pool with a legible, weighted mixture of domains.

**Bigger picture:** The Pile made corpus composition itself a reusable research artifact and influenced many open language-model projects.

### 5. [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) - 2021

**Connects to:** MassiveText, web, books, news, code, Wikipedia, data cleaning, mixture design, toxicity, and model behavior.

**Very short abstract:** The Gopher report describes training models across a wide scale range and analyzes performance, bias, and toxicity. Its data section details a multi-source corpus and a substantial filtering and deduplication pipeline, while leaving the final dataset unreleased.

**Why it is useful:** It provides an unusually detailed industrial data recipe from the period between The Pile and modern open trillion-token corpora.

**Bigger picture:** Gopher illustrates the difference between methodological transparency and actual access to the resulting training data.

### 6. [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/abs/2306.01116) - 2023

**Connects to:** RefinedWeb, Common Crawl, large-scale deduplication, heuristic filtering, and web-only pretraining.

**Very short abstract:** RefinedWeb applies extensive filtering and deduplication to Common Crawl and evaluates whether web data alone can support strong language models. The work releases a large extract and models trained on it, enabling inspection of the recipe and its downstream effects.

**Why it is useful:** It directly tests the assumption that a competitive corpus must include many separately curated premium sources.

**Bigger picture:** The paper shifted attention from collecting ever more named sources to extracting more value from a well-processed open web archive.

### 7. [The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI](https://arxiv.org/abs/2310.16787) - 2023

**Connects to:** Dataset lineage, license metadata, attribution, commercial openness, and responsible source selection.

**Very short abstract:** A multidisciplinary team traces the sources, creators, licenses, and downstream relationships of a large collection of text datasets. The audit identifies missing and inconsistent licensing information and releases tools for exploring dataset lineage.

**Why it is useful:** It turns the lecture's warning that "downloadable" does not mean "permitted" into a systematic empirical research program.

**Bigger picture:** Provenance records are becoming necessary infrastructure for both legal review and scientific reproducibility.

### 8. [Foundation Models and Fair Use](https://jmlr.org/papers/v24/23-0569.html) - 2023

**Connects to:** Copyright, fair use, training inputs, substantially similar outputs, market effects, and technical mitigations.

**Very short abstract:** This interdisciplinary study surveys U.S. fair-use doctrine as it may apply to foundation-model development and deployment, connects the analysis to relevant case law, and tests model reproduction behavior. It also discusses technical mitigations and policy mechanisms while emphasizing that fair use is not automatic.

**Why it is useful:** It provides the legal and technical nuance behind the lecture's rejection of simple overlap-based answers to copyright questions.

**Bigger picture:** Model training, output behavior, market substitution, and mitigation practices may matter at different stages of one legal analysis.

### 9. [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159) - 2024

**Connects to:** Dolma, mixed-domain data, open documentation, Common Crawl, code, papers, books, social media, and ablations.

**Very short abstract:** Dolma releases a large English corpus assembled from web pages, code, scientific papers, public-domain books, social media, and encyclopedic sources. The paper documents construction decisions, analyzes intermediate corpus versions, and releases the processing toolkit.

**Why it is useful:** It is a strong reference for treating the dataset, code, documentation, and ablation evidence as one research artifact.

**Bigger picture:** Open model science depends on reconstructing the data pipeline, not merely downloading a final token stream.

### 10. [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794) - 2024

**Connects to:** DCLM, controlled data experiments, fixed training budgets, filtering, deduplication, mixing, and model-based quality scoring.

**Very short abstract:** DataComp-LM supplies a common Common Crawl pool, standardized training recipes, several model scales, and a broad evaluation suite for comparing curation strategies. Its baseline experiments identify model-based filtering as an effective component of web-dataset construction.

**Why it is useful:** It controls model and compute variables so that changes in downstream performance can be attributed more credibly to data choices.

**Bigger picture:** Dataset design is moving toward benchmarked experimental competition rather than incomparable one-off corpus releases.

### 11. [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) - 2024

**Connects to:** FineWeb, FineWeb-Edu, multi-snapshot Common Crawl, filtering ablations, deduplication, and educational quality.

**Very short abstract:** FineWeb builds a large English web corpus from many Common Crawl snapshots and reports controlled tests of extraction, filtering, and duplicate-removal choices. FineWeb-Edu adds a learned educational-quality selection stage and releases the supporting code and ablation models.

**Why it is useful:** It is one of the most transparent modern accounts of how many small pipeline choices accumulate into a strong web dataset.

**Bigger picture:** Open corpus work increasingly treats downstream model quality as the test of a filter rather than relying only on textual cleanliness heuristics.

### 12. [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/abs/2402.19173) - 2024

**Connects to:** Software Heritage, GitHub development events, code documentation, persistent identifiers, licensing, and code-language-model data.

**Very short abstract:** The Stack v2 builds a multilingual code corpus around the Software Heritage archive and supplements repository files with pull requests, notebooks, and documentation. The paper links the data recipe to StarCoder2 training and releases persistent source identifiers for greater traceability.

**Why it is useful:** It explains why a software archive contributes richer provenance and development history than a simple snapshot of GitHub files.

**Bigger picture:** Code corpora are evolving from bags of files into traceable records of software artifacts and engineering activity.

### 13. [Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](https://aclanthology.org/2025.acl-long.123/) - 2025

**Connects to:** Nemotron-CC, classifier ensembles, synthetic rephrasing, quality-quantity tradeoffs, and long token horizons.

**Very short abstract:** Nemotron-CC combines several quality classifiers with synthetic rewriting and less aggressive heuristic rejection to retain more useful Common Crawl material. The paper evaluates both high-quality subsets and a larger collection intended for training over long token horizons.

**Why it is useful:** It is the primary modern example of transforming lower-quality documents instead of making filtering a purely keep-or-discard decision.

**Bigger picture:** Synthetic data generation is becoming a corpus-recovery tool as well as a source of new post-training examples.

### 14. [The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text](https://arxiv.org/abs/2506.05209) - 2025

**Connects to:** Public-domain and openly licensed data, Common Pile, provenance constraints, mixture construction, and open model validation.

**Very short abstract:** The Common Pile aggregates many public-domain and openly licensed sources across code, research, books, reference works, education, government, and other domains. The project releases the corpus, construction code, mixture, models, and checkpoints to test whether a provenance-constrained dataset can support useful pretraining.

**Why it is useful:** It directly investigates the lecture's deliberately conservative question about training from clearly reusable material.

**Bigger picture:** Provenance-first datasets trade source breadth and collection ease for stronger transparency, creating a measurable alternative to unlicensed web-scale training.

## Suggested reading order

1. **Start with the historical recipe:** 1 (GPT-3), 2 (CCNet), 3 (T5/C4), and 4 (The Pile).
2. **See the transition to modern open corpora:** 5 (Gopher), 6 (RefinedWeb), 9 (Dolma), and 11 (FineWeb).
3. **Study controlled and specialized data work:** 10 (DataComp-LM), 12 (The Stack v2), and 13 (Nemotron-CC).
4. **Understand provenance and legal constraints:** 7 (Data Provenance), 8 (fair use), and 14 (Common Pile).
