---
title: "Lecture 12 - Evaluation: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 12
companion_notes: "../lecture_notes/lecture_12_evaluation.md"
status: "complete"
---

# Lecture 12: Evaluation - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The collection moves from held-out language modeling to tests of knowledge and reasoning, then to human and model preferences, real-world agent tasks, safety, and benchmark auditing. Read it as a progression from scoring a model output to validating a claim about an entire deployed system.

| Lecture theme | Papers |
|---|---|
| Perplexity, zero-shot transfer, and intelligence constructs | 1-2 |
| Academic knowledge and holistic benchmark suites | 3-6 |
| Human preferences and LLM judges | 7-10 |
| Action-based, fresh, and safety evaluation | 11-13 |
| Benchmark auditing and repair | 14 |

## Curated papers

### 1. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - 2019

**Connects to:** Perplexity, in-distribution testing, zero-shot task transfer, continuation scoring, and GPT-2.

**Very short abstract:** GPT-2 is trained with ordinary next-token prediction on a broad web-derived corpus and evaluated both as a language model and as a zero-shot task solver. The report shows how one probabilistic model can be tested through held-out likelihood and task-specific prompt formulations.

**Why it is useful:** It is the historical bridge from perplexity-centered evaluation to modern zero-shot benchmark suites.

**Bigger picture:** The work helped establish the idea that pretraining loss and downstream capability are related but distinct evaluation targets.

### 2. [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547) - 2019

**Connects to:** Construct validity, ARC, skill versus intelligence, prior knowledge, and generalization efficiency.

**Very short abstract:** Chollet argues that task skill can be purchased through data and priors and therefore should not be equated with general intelligence. The paper proposes measuring skill-acquisition efficiency and introduces the Abstraction and Reasoning Corpus as an evaluation built around that view.

**Why it is useful:** It gives the theoretical motivation for the lecture's attempt to separate reasoning from memorized knowledge.

**Bigger picture:** A benchmark is meaningful only relative to an explicit account of the construct it is meant to measure.

### 3. [Measuring Massive Multitask Language Understanding](https://openreview.net/forum?id=d7KBjmI3GmQ) - 2021

**Connects to:** MMLU, multiple-choice continuation scoring, broad subject coverage, expert knowledge, and calibration.

**Very short abstract:** MMLU evaluates language models on questions drawn from many academic and professional subjects at varied difficulty levels. Its zero- and few-shot setup probes the breadth of knowledge and problem solving acquired during pretraining.

**Why it is useful:** It is the canonical reference for the exam-style aggregate that became a standard model-report headline.

**Bigger picture:** MMLU's saturation motivated harder expert benchmarks while also exposing the limits of one averaged multiple-choice score.

### 4. [Holistic Evaluation of Language Models](https://openreview.net/forum?id=iO4LZibEqW) - 2023

**Connects to:** Multi-metric evaluation, scenarios, robustness, fairness, toxicity, efficiency, and transparent reporting.

**Very short abstract:** HELM defines a taxonomy of use scenarios and desiderata, then evaluates many language models under standardized prompting and measurement conditions. It reports accuracy alongside calibration, robustness, fairness, bias, toxicity, and efficiency rather than reducing comparison to one leaderboard number.

**Why it is useful:** It provides a systematic framework for choosing evaluations around a decision rather than around benchmark popularity.

**Bigger picture:** Holistic evaluation treats missing coverage and metric tradeoffs as first-class findings, not inconveniences to average away.

### 5. [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://openreview.net/forum?id=Ti67584b98) - 2024

**Connects to:** Expert-written science questions, non-expert validation, source contamination, and hard multiple choice.

**Very short abstract:** GPQA contains graduate-level questions in physics, chemistry, and biology written and checked by domain experts. The construction process compares expert and skilled non-expert performance to select items that resist ordinary web search while retaining answerable scientific content.

**Why it is useful:** It shows how expert authorship and validation can raise benchmark difficulty without abandoning automated scoring.

**Bigger picture:** Even carefully designed expert questions can inherit contamination from textbooks, papers, and source material used during pretraining.

### 6. [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) - 2025

**Connects to:** Frontier academic evaluation, multimodal and short-answer items, expert contribution, private splits, and benchmark saturation.

**Very short abstract:** Humanity's Last Exam assembles difficult, broad-domain questions contributed and reviewed by subject-matter experts. It combines multiple-choice and short-answer formats, includes multimodal items, and reserves material to slow immediate benchmark exposure.

**Why it is useful:** It represents the current endpoint of escalating closed-ended academic benchmark difficulty.

**Bigger picture:** Harder questions postpone saturation but do not resolve whether exams predict useful open-ended or agentic behavior.

### 7. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) - 2023

**Connects to:** MT-Bench, LLM judges, pairwise comparison, position bias, verbosity bias, and human agreement.

**Very short abstract:** This paper studies strong language models as scalable judges of open-ended assistant responses and introduces MT-Bench and an early Chatbot Arena. It measures agreement with human preferences while documenting systematic judge biases and mitigation strategies.

**Why it is useful:** It is the foundational source for both the appeal and the failure modes of automatic preference evaluation.

**Bigger picture:** Using a model as a metric shifts evaluation cost downward but makes the judge's behavior part of the validity argument.

### 8. [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://proceedings.mlr.press/v235/chiang24b.html) - 2024

**Connects to:** Blind pairwise battles, crowdsourced prompts, Bradley-Terry or Elo-style rankings, sampling, and ecological validity.

**Very short abstract:** Chatbot Arena gathers anonymous user prompts and pairwise votes on responses from hidden model identities. The paper analyzes the diversity and discriminative value of the data and uses statistical ranking methods to compare systems from noisy preferences.

**Why it is useful:** It provides the primary methodology behind one of the most influential open-ended model leaderboards.

**Bigger picture:** Arena evaluation trades laboratory control for realism, making user population, traffic allocation, and ranking uncertainty central design choices.

### 9. [Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators](https://arxiv.org/abs/2404.04475) - 2024

**Connects to:** AlpacaEval, length bias, confounding, automatic preferences, and counterfactual win rates.

**Very short abstract:** The paper models an automatic judge's preference as a function of response length and other features, then estimates what the preference would be if candidate lengths matched. This produces a length-controlled comparison rather than rewarding verbosity indirectly.

**Why it is useful:** It is a concrete statistical correction for a bias highlighted repeatedly in the lecture.

**Bigger picture:** Evaluation metrics can be gamed by properties correlated with quality unless those properties are measured and audited explicitly.

### 10. [WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild](https://openreview.net/forum?id=MKEHCx25xp) - 2025

**Connects to:** Real user queries, task-specific rubrics, pairwise and pointwise LLM judging, and length-aware evaluation.

**Very short abstract:** WildBench selects difficult prompts from real conversation logs and creates task-specific checklists for structured automatic judging. It supports both pairwise preference and individual response scoring while exposing the rationale used by the judge.

**Why it is useful:** It combines more realistic prompts with rubrics that are more specific than a generic "which answer is better" instruction.

**Bigger picture:** Modern open-ended evaluation is moving toward natural traffic plus explicit per-task criteria rather than fixed synthetic prompts alone.

### 11. [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://openreview.net/forum?id=VTF8yNQM66) - 2024

**Connects to:** Agent evaluation, repository context, executable graders, software scaffolds, and real-world actions.

**Very short abstract:** SWE-bench asks systems to modify real software repositories in response to historical GitHub issues and evaluates patches with repository tests. Solving an item requires navigation, code understanding, editing, and execution rather than selecting or generating an isolated answer.

**Why it is useful:** It makes the lecture's move from evaluating text to evaluating actions concrete.

**Bigger picture:** Agent benchmarks measure the model together with its tools, context policy, search process, and runtime environment.

### 12. [LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code](https://arxiv.org/abs/2403.07974) - 2024

**Connects to:** Fresh evaluation, timestamped tasks, code generation, self-repair, execution, and contamination control.

**Very short abstract:** LiveCodeBench continuously collects newly released programming-contest problems and evaluates several code-related capabilities beyond generation alone. The time-indexed collection lets evaluators compare model release and task dates while retaining executable scoring.

**Why it is useful:** It is a practical implementation of the lecture's recommendation to use fresh, renewable test material.

**Bigger picture:** Dynamic benchmarks reduce static test-set exposure, though timestamps cannot prove that every underlying concept is novel.

### 13. [JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models](https://openreview.net/forum?id=j5lgypLMsl) - 2024

**Connects to:** Safety evaluation, threat models, jailbreak artifacts, refusal behavior, attack cost, and reproducibility.

**Very short abstract:** JailbreakBench standardizes harmful-behavior prompts, system and chat templates, attack artifacts, scoring, and reporting for jailbreak evaluations. Its design aims to make attack success and cost comparable across methods and model defenses.

**Why it is useful:** It demonstrates why safety scores are uninterpretable without a declared adversary, policy, context, and grading procedure.

**Bigger picture:** Safety evaluation is an evolving security process rather than a timeless dataset with one stable accuracy number.

### 14. [HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam](https://arxiv.org/abs/2602.13964) - 2026

**Connects to:** Benchmark auditing, noisy questions, error taxonomies, answer verification, and revised evaluation sets.

**Very short abstract:** HLE-Verified re-audits Humanity's Last Exam through a documented verification procedure and classifies problems in the original items. It releases corrected and structured revisions intended to support more reliable comparison on the benchmark.

**Why it is useful:** It is a direct modern example of the lecture's warning that even prominent executable or closed-ended datasets require qualitative inspection.

**Bigger picture:** Benchmark maintenance and repair should be treated as continuing research, not as an admission that the original evaluation was useless.

## Suggested reading order

1. **Start here:** 1 (GPT-2), 2 (measure of intelligence), 3 (MMLU), and 4 (HELM).
2. **Understand current frontier exams:** 5 (GPQA), 6 (Humanity's Last Exam), and 14 (HLE-Verified).
3. **Study open-ended judging:** 7 (LLM-as-a-judge), 8 (Chatbot Arena), 9 (length control), and 10 (WildBench).
4. **Move to actions, freshness, and safety:** 11 (SWE-bench), 12 (LiveCodeBench), and 13 (JailbreakBench).
