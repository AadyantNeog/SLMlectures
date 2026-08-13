---
title: "Lecture 9 - Scaling Laws: Basics: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 9
companion_notes: "../lecture_notes/lecture_09_scaling_laws_basics.md"
status: "complete"
---

# Lecture 9: Scaling Laws - Basics - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The papers trace scaling laws from early learning-curve observations to modern language-model compute allocation. They then add the complications emphasized in the lecture: batch-size effects, sparse models, finite or repeated data, data selection and mixtures, hyperparameter transfer, and the uncertainty involved in fitting and reproducing compute-optimal laws.

| Lecture theme | Papers |
|---|---|
| Historical learning curves and predictable scale trends | 1-3, 5-6 |
| Batch size, hyperparameters, and compute-optimal allocation | 4, 7-8 |
| Sparse models and data-aware scaling | 9-11, 14 |
| Auditing and reconciling compute-optimal estimates | 12-13 |

## Curated papers

### 1. [Learning Curves: Asymptotic Values and Rate of Convergence](https://papers.nips.cc/paper_files/paper/1993/hash/1aa48fc4880bb0c9b8a3bf979d3b917e-Abstract.html) - 1993

**Connects to:** Learning curves, asymptotic error, convergence rates, extrapolation, and fitting performance as data grows.

**Very short abstract:** This early study analyzes how prediction error approaches an asymptote as the size of a training set increases. It asks what features of a learning curve can be estimated from smaller experiments and how quickly additional data improves performance.

**Why it is useful:** It places modern neural scaling laws in the longer tradition of modeling performance as a smooth function of available data.

**Bigger picture:** Today's power-law fits use vastly larger models and datasets, but they inherit the same ambition: extrapolate expensive outcomes from cheaper observations.

### 2. [Scaling to Very Very Large Corpora for Natural Language Disambiguation](https://aclanthology.org/P01-1005/) - 2001

**Connects to:** Data scaling, language tasks, log-log trends, diminishing error, and empirical extrapolation.

**Very short abstract:** The paper studies language disambiguation systems over corpus sizes spanning several orders of magnitude. It observes regular improvements as more training data becomes available and uses those trends to reason about performance at still larger scales.

**Why it is useful:** It is a foundational NLP example of the empirical regularity that later language-model scaling work made central.

**Bigger picture:** Scaling behavior was visible in statistical language systems well before large Transformers made scale a primary design variable.

### 3. [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409) - 2017

**Connects to:** Power laws, irreducible error, model capacity, dataset size, compute, and extrapolation across scales.

**Very short abstract:** This work fits simple empirical relationships between deep-learning error and the amount of data, model capacity, and computation. It argues that controlled smaller runs can reveal trends that predict performance in larger regimes.

**Why it is useful:** It gives a broad pre-language-model statement of the methodology used throughout the lecture: run a scale sweep, fit a functional form, and extrapolate cautiously.

**Bigger picture:** Scaling laws became useful planning instruments once researchers showed that multiple resource axes exhibit stable, measurable regularities.

### 4. [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - 2018

**Connects to:** Critical batch size, gradient noise scale, steps versus examples, compute efficiency, and batch scaling.

**Very short abstract:** The authors model the tradeoff between the number of optimization steps and the number of training examples required at different batch sizes. A gradient-noise statistic predicts the transition from a small-batch regime to diminishing returns from further parallel batch growth.

**Why it is useful:** It clarifies why batch size belongs in scaling analysis even though it does not directly enlarge the model or dataset.

**Bigger picture:** Compute-optimal training requires allocating computation across examples, parameters, and optimizer steps rather than treating total FLOPs as a sufficient description.

### 5. [A Constructive Prediction of the Generalization Error Across Scales](https://openreview.net/forum?id=ryenvpEKDr) - 2020

**Connects to:** Functional forms for scaling, finite model and data effects, joint prediction, and generalization error.

**Very short abstract:** This paper develops a model of generalization error that combines finite-data and finite-model limitations in one expression. The proposed form is designed to interpolate between limiting regimes and predict held-out error across changes in both resources.

**Why it is useful:** It offers a principled alternative to fitting isolated one-dimensional power laws while holding every other variable fixed.

**Bigger picture:** Joint scaling surfaces are necessary when practitioners must choose model size and data size together under a budget.

### 6. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - 2020

**Connects to:** Language-model loss, parameter and data power laws, compute-optimal frontiers, Kaplan scaling, and large-model early stopping.

**Very short abstract:** The authors measure cross-entropy loss across broad variations in Transformer size, dataset size, and training compute. They fit power-law relationships and derive a compute-efficient allocation that favors relatively large models trained for fewer tokens.

**Why it is useful:** It is the foundational language-model scaling paper behind the lecture's discussion of separate resource laws and the original Kaplan compute-optimal prescription.

**Bigger picture:** Its conclusions shaped early large-model planning and created the reference point against which Chinchilla-style revisions are compared.

### 7. [Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://proceedings.neurips.cc/paper/2021/hash/8df7c2e3c3c3be098ef7b382bd2c37ba-Abstract.html) - 2021

**Connects to:** muP, width scaling, parameterization, learning-rate transfer, and controlling hyperparameters in scale sweeps.

**Very short abstract:** The paper introduces the maximal update parameterization, under which selected optimal hyperparameters remain stable as network width changes. Hyperparameters can therefore be tuned on smaller proxy models and transferred to larger models without a new search at every scale.

**Why it is useful:** It addresses a major confound in scaling experiments: apparent scale trends can reflect a parameterization whose optimization behavior changes with width.

**Bigger picture:** Reliable scaling laws depend on experimental protocols that keep training quality comparable, not only on fitting a curve after the runs finish.

### 8. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - 2022

**Connects to:** Chinchilla scaling, IsoFLOP curves, compute-optimal model and token counts, undertraining, and fixed-budget experiments.

**Very short abstract:** This work studies how model size and training-token count should be balanced at a fixed compute budget using several fitting approaches, including IsoFLOP sweeps. It concludes that the compute-efficient model and dataset should grow at roughly comparable rates, leading to smaller models trained on substantially more data than earlier prescriptions suggested.

**Why it is useful:** It is the central paper for the lecture's Chinchilla derivation and its contrast with Kaplan-style allocation.

**Bigger picture:** The result redirected model development toward greater token exposure and made data supply a first-class constraint in compute planning.

### 9. [Unified Scaling Laws for Routed Language Models](https://proceedings.mlr.press/v162/clark22a.html) - 2022

**Connects to:** Mixture-of-experts, sparse versus dense parameters, routing, expert count, and architecture-aware scaling laws.

**Very short abstract:** The authors develop empirical scaling relationships for routed language models that account for both total parameters and the parameters activated for each token. The resulting formulation compares dense and sparse models within a shared predictive framework.

**Why it is useful:** It shows why a dense-model law cannot be applied unchanged when parameter count and per-token computation become decoupled.

**Bigger picture:** Scaling laws must evolve with architecture; otherwise a single quantity such as total parameters can cease to represent the relevant resource.

### 10. [Beyond neural scaling laws: beating power law scaling via data pruning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d077df6-Abstract-Conference.html) - 2022

**Connects to:** Data quality, pruning, informative examples, effective dataset size, and departures from naive power-law scaling.

**Very short abstract:** This paper studies how selecting training examples changes learning curves compared with uniformly using all available data. It finds regimes in which principled pruning improves data efficiency and alters the apparent scaling behavior.

**Why it is useful:** It supports the lecture's point that token count alone is incomplete because filtering and selection determine how much useful signal those tokens contain.

**Bigger picture:** Scaling can improve through better resource quality and allocation, not only through increasing the raw amount of compute or data.

### 11. [Scaling Data-Constrained Language Models](https://papers.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html) - 2023

**Connects to:** Finite data, repeated epochs, token repetition, data-constrained optima, and model/data allocation.

**Very short abstract:** The paper measures language-model scaling when the supply of unique data is capped and training must reuse tokens. It models the diminishing value of additional epochs and studies how compute should be allocated when new data is unavailable.

**Why it is useful:** It directly extends the idealized one-pass scaling picture to the finite-corpus setting emphasized in the lecture.

**Bigger picture:** As training datasets approach the amount of usable text, compute-optimal decisions increasingly depend on repetition, augmentation, and data quality.

### 12. [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/abs/2406.19146) - 2024

**Connects to:** Kaplan versus Chinchilla, fitting procedures, optimization quality, learning-rate schedules, IsoFLOP analysis, and uncertainty.

**Very short abstract:** This work investigates why influential studies infer different compute-optimal allocations from seemingly similar power-law behavior. It re-examines experimental and statistical choices and shows how training setup and fitting methodology can shift the resulting prescription.

**Why it is useful:** It cautions against treating one fitted exponent or token-to-parameter ratio as a universal constant.

**Bigger picture:** Scaling laws are measurements produced by a protocol, so reproducible planning requires reporting the sweep design and estimator as carefully as the fitted curve.

### 13. [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) - 2024

**Connects to:** Reproduction, digitized data, parametric fits, uncertainty, IsoFLOP minima, and compute-optimal estimates.

**Very short abstract:** The authors attempt to reproduce Chinchilla's scaling analysis from the information and figures available in the original publication. They compare fitting approaches and examine how limited observations and modeling choices affect the recovered compute-optimal law.

**Why it is useful:** It is a practical case study in how difficult it can be to recover a headline scaling result without the original measurements and analysis pipeline.

**Bigger picture:** Scaling-law evidence is far more reusable when raw runs, definitions, and fitting code are available for independent audit.

### 14. [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html) - 2025

**Connects to:** Domain mixtures, mixture weights, data allocation, validation loss prediction, and data-specific scaling.

**Very short abstract:** This work models language-model performance as a function of how training tokens are allocated among multiple data domains. It uses smaller mixture experiments to predict promising compositions and reduce the cost of choosing a training distribution.

**Why it is useful:** It develops the lecture's mixture discussion into a practical method for optimizing what kind of data, rather than only how much data, a model sees.

**Bigger picture:** Modern scaling analysis is moving from scalar budgets toward structured allocation problems over sources, quality levels, architectures, and deployment goals.

## Suggested reading order

1. **Start with the core lineage:** 3, 6, and 8 - follow predictable scaling from general deep learning through Kaplan and Chinchilla.
2. **Build the mathematical intuition:** 1, 2, and 5 - examine earlier learning curves and a joint finite-model/finite-data formulation.
3. **Control the experiment:** 4 and 7 - understand batch effects and hyperparameter transfer before interpreting scale sweeps.
4. **Add architecture and data constraints:** 9, 11, and 14 - extend the basic laws to sparse models, repeated data, and domain mixtures.
5. **Study data efficiency:** 10 - see how selection can change the relationship between raw dataset size and performance.
6. **Finish with the audit:** 12 and 13 - assess why compute-optimal conclusions can vary and what reproducible evidence requires.
