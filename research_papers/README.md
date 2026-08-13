# Stanford CS336 Spring 2026 - Research Paper Guides

This folder is a curated research-paper companion to the 17 lecture notes for *Language Modeling from Scratch*.

Each lecture guide includes:

- a short reading map tied to the lecture's main themes;
- a mix of foundational, influential, practical, and modern papers;
- a primary-source link for every paper;
- a very short original summary of the contribution;
- why the paper is useful for studying the lecture; and
- how it fits into the larger development of language-model research.

The collection contains **232 lecture-paper entries** across **205 unique linked primary records**. Repeated papers are intentional when one work is foundational to more than one lecture. Publication years span **1992-2026**.

Course playlist: [Stanford CS336 - Language Modeling from Scratch](https://youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&si=nVODEdvz8lvNku3N)

## Lecture index

| Lecture | Topic | Papers | Guide | Companion notes |
|---:|---|---:|---|---|
| 1 | Course overview and tokenization | 12 | [Research guide](lecture_01_overview_tokenization.md) | [Lecture notes](../lecture_notes/lecture_01_overview_tokenization.md) |
| 2 | PyTorch, tensors, and einops | 13 | [Research guide](lecture_02_pytorch_einops.md) | [Lecture notes](../lecture_notes/lecture_02_pytorch_einops.md) |
| 3 | Language-model architectures | 14 | [Research guide](lecture_03_architectures.md) | [Lecture notes](../lecture_notes/lecture_03_architectures.md) |
| 4 | Attention and alternative architectures | 14 | [Research guide](lecture_04_attention_alternatives.md) | [Lecture notes](../lecture_notes/lecture_04_attention_alternatives.md) |
| 5 | GPUs and TPUs | 13 | [Research guide](lecture_05_gpus_tpus.md) | [Lecture notes](../lecture_notes/lecture_05_gpus_tpus.md) |
| 6 | Kernels, Triton, and XLA | 12 | [Research guide](lecture_06_kernels_triton_xla.md) | [Lecture notes](../lecture_notes/lecture_06_kernels_triton_xla.md) |
| 7 | Parallelism, part 1 | 13 | [Research guide](lecture_07_parallelism_part_1.md) | [Lecture notes](../lecture_notes/lecture_07_parallelism_part_1.md) |
| 8 | Parallelism, part 2 | 14 | [Research guide](lecture_08_parallelism_part_2.md) | [Lecture notes](../lecture_notes/lecture_08_parallelism_part_2.md) |
| 9 | Scaling laws: foundations | 14 | [Research guide](lecture_09_scaling_laws_basics.md) | [Lecture notes](../lecture_notes/lecture_09_scaling_laws_basics.md) |
| 10 | Inference | 14 | [Research guide](lecture_10_inference.md) | [Lecture notes](../lecture_notes/lecture_10_inference.md) |
| 11 | Advanced scaling laws | 14 | [Research guide](lecture_11_scaling_laws_advanced.md) | [Lecture notes](../lecture_notes/lecture_11_scaling_laws_advanced.md) |
| 12 | Evaluation | 14 | [Research guide](lecture_12_evaluation.md) | [Lecture notes](../lecture_notes/lecture_12_evaluation.md) |
| 13 | Data sources and datasets | 14 | [Research guide](lecture_13_data_sources_datasets.md) | [Lecture notes](../lecture_notes/lecture_13_data_sources_datasets.md) |
| 14 | Data processing and filtering | 15 | [Research guide](lecture_14_data_processing_filtering.md) | [Lecture notes](../lecture_notes/lecture_14_data_processing_filtering.md) |
| 15 | Mid-training and post-training | 14 | [Research guide](lecture_15_mid_post_training.md) | [Lecture notes](../lecture_notes/lecture_15_mid_post_training.md) |
| 16 | Reinforcement learning from verifiable rewards | 14 | [Research guide](lecture_16_post_training_rlvr.md) | [Lecture notes](../lecture_notes/lecture_16_post_training_rlvr.md) |
| 17 | Alignment and multimodal models | 14 | [Research guide](lecture_17_alignment_multimodality.md) | [Lecture notes](../lecture_notes/lecture_17_alignment_multimodality.md) |

## Suggested use

1. Read the companion lecture note before opening its paper guide.
2. Use the reading map to choose the theme most relevant to your goal.
3. Read the foundational paper first, then a modern systems or model paper that applies the idea at scale.
4. Treat each compact summary as orientation, not a substitute for the paper's methods, experiments, limitations, and appendices.
5. Recheck the primary page for revised arXiv versions or later proceedings publications.

## Curation and verification

- Links favor arXiv abstract pages and official proceedings, journal, conference, or institutional records.
- Titles and publication years were checked against the linked primary records.
- Summaries are original paraphrases and deliberately brief; they are not copied abstracts.
- Every guide uses the same four lenses: **Connects to**, **Very short abstract**, **Why it is useful**, and **Bigger picture**.
- The list is selective rather than exhaustive. It emphasizes papers that clarify the lecture, establish an important idea, or show a consequential modern extension.

The detailed authoring contract is available in [`_meta/collection_guide.md`](_meta/collection_guide.md).
