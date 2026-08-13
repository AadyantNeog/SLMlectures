---
title: "Lecture 11 - Advanced Scaling Laws: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 11
companion_notes: "../lecture_notes/lecture_11_scaling_laws_advanced.md"
status: "complete"
---

# Lecture 11: Advanced Scaling Laws - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

These papers follow the lecture's two strategies for expensive hyperparameters. One line measures how optima drift with compute, data, and loss; the other changes parameterization so tuned values transfer across widths. The final papers connect those ideas to modern optimizers and large public training runs.

| Lecture theme | Papers |
|---|---|
| Optimizer and critical-batch foundations | 1-2 |
| Loss and compute-optimal scaling | 3-4, 7 |
| Maximum-update parameterization and transfer | 5-6, 8, 11-12 |
| Public large-model scaling recipes | 9-10, 14 |
| Scale-dependent optimizer design | 13 |

## Curated papers

### 1. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - 2015

**Connects to:** Adam and AdamW-style updates, per-coordinate normalization, optimizer-specific tuning, and learning-rate scale.

**Very short abstract:** Adam combines momentum-like first-moment estimates with adaptive second-moment normalization to produce coordinatewise parameter updates. Bias correction and a small set of global hyperparameters make it practical for large stochastic objectives.

**Why it is useful:** It establishes the baseline optimizer whose learning-rate scaling and update geometry the lecture compares with Muon and muP rules.

**Bigger picture:** Advanced scaling laws must describe an optimizer-model pair; changing the update rule can change which hyperparameters transfer.

### 2. [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - 2018

**Connects to:** Critical batch size, gradient noise scale, lower target loss, compute efficiency, and step efficiency.

**Very short abstract:** The paper proposes gradient noise scale as a measurable predictor of the largest batch that still provides useful parallel speedup. Experiments across several domains show how this scale changes through training and clarify the tradeoff between examples consumed and optimization steps.

**Why it is useful:** It is the conceptual foundation for the lecture's loss-dependent critical-batch fits.

**Bigger picture:** Batch size is not merely a hardware choice; it is a scale-dependent property of the optimization problem.

### 3. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - 2020

**Connects to:** Power-law loss curves, model size, dataset size, compute, early stopping, and extrapolation.

**Very short abstract:** This study fits empirical power laws relating autoregressive language-model loss to parameters, data, and training compute. It uses those fits to analyze overfitting, training efficiency, and compute allocation across model and token axes.

**Why it is useful:** It supplies the basic empirical language and experimental workflow that every later scaling study modifies.

**Bigger picture:** Scaling laws changed model development from choosing one large configuration to fitting trends across a controlled family of smaller runs.

### 4. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - 2022

**Connects to:** Chinchilla, IsoFLOPs curves, compute-optimal model-data allocation, and held-out prediction.

**Very short abstract:** The authors train a broad grid of model sizes and token budgets, then estimate the parameter and data combination that minimizes loss for a fixed compute budget. A target-scale Chinchilla run tests the resulting allocation against substantially larger, more lightly trained models.

**Why it is useful:** It is the canonical example of converting fitted small-run trends into a costly full-scale training decision.

**Bigger picture:** Chinchilla shifted practice toward smaller models trained on more tokens and made data supply a central scaling constraint.

### 5. [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) - 2022

**Connects to:** Maximum update parameterization, muTransfer, width scaling, initialization, multipliers, and base learning rates.

**Very short abstract:** The paper develops a parameterization in which many optimal hyperparameters remain stable as network width changes. It then tunes small proxy networks and transfers those settings to much larger Transformer and ResNet targets without target-scale sweeps.

**Why it is useful:** It is the primary source for the lecture's "remove the drift" alternative to fitting a hyperparameter scaling law.

**Bigger picture:** muP treats parameterization as experimental infrastructure for making cheap models informative about expensive ones.

### 6. [Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster](https://arxiv.org/abs/2304.03208) - 2023

**Connects to:** Open scaling ladders, Chinchilla training, muP comparisons, predictable loss, and public checkpoints.

**Very short abstract:** Cerebras-GPT trains a family of language models across a wide parameter range using a common compute-optimal recipe. The report compares standard scaling with a muP variant and releases models and code to make the scaling analysis inspectable.

**Why it is useful:** It provides public evidence for the lecture's discussion of muP residuals and large-model prediction.

**Bigger picture:** Transparent model families let researchers test scaling claims that would otherwise remain internal to frontier laboratories.

### 7. [Scaling Data-Constrained Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html) - 2023

**Connects to:** Repeated data, token scarcity, excess parameters, compute-optimal allocation, and the limits of Chinchilla assumptions.

**Very short abstract:** This paper studies language-model scaling when unique data is finite and training therefore repeats examples. It fits a modified compute-allocation law that accounts for diminishing returns from repeated tokens and investigates practical responses to data scarcity.

**Why it is useful:** It shows that a scaling law can fail because its resource assumptions change, even if the model family stays similar.

**Bigger picture:** As high-quality data becomes a bottleneck, scaling theory must model data reuse rather than treating every training token as equally novel.

### 8. [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813) - 2023

**Connects to:** Stable activations, nonvanishing feature updates, spectral norms, fan-in/fan-out scaling, and an accessible muP derivation.

**Very short abstract:** The authors state a spectral scaling condition for weight matrices and their updates that preserves nontrivial feature learning at large width. The same condition yields an elementary route to maximal-update parameterization rules.

**Why it is useful:** It directly explains the spectral-norm and activation-update derivation used in the lecture.

**Bigger picture:** It reframes muP as an invariance design problem rather than a collection of implementation-specific multipliers.

### 9. [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954) - 2024

**Connects to:** Learning-rate and batch-size grids, fitted optimum drift, WSD-like schedules, IsoFLOPs experiments, and target-run validation.

**Very short abstract:** DeepSeek LLM reports scaling experiments used to choose model-data allocation and sensitive optimization hyperparameters for released dense models. Rather than force invariance, the study fits how preferred learning rate and batch size move with training compute.

**Why it is useful:** It is the lecture's strongest public example of predicting hyperparameter drift and testing the prediction on full models.

**Bigger picture:** Release reports increasingly use small-run response surfaces as engineering inputs, not just as retrospective scientific plots.

### 10. [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395) - 2024

**Connects to:** Scaling ladders, muP, warmup-stable-decay schedules, reusable horizons, critical batch size, and compute allocation.

**Very short abstract:** MiniCPM builds small language models using model wind-tunnel experiments, a muP-style parameterization, and a warmup-stable-decay learning-rate schedule. The schedule supports continued training and efficient branching into multiple annealed endpoints for scaling studies.

**Why it is useful:** It brings several lecture concepts together in one documented training recipe.

**Bigger picture:** WSD turns a single long trajectory into reusable experimental infrastructure for data-axis and continual-training decisions.

### 11. [An Empirical Study of μP Learning Rate Transfer](https://arxiv.org/abs/2404.05728) - 2024

**Connects to:** Large-scale muTransfer stress tests, Transformer widths, learning-rate optima, RMSNorm gains, Lion, and weight decay.

**Very short abstract:** This study tests whether muP learning rates selected on small Transformers remain near-optimal for much larger models and training budgets. A broad set of ablations identifies both robust transfer cases and components that disrupt it.

**Why it is useful:** It supplies the independent positive and negative evidence discussed at the end of the lecture's muP section.

**Bigger picture:** Stress tests reveal the boundary of a scaling rule more clearly than another successful in-family demonstration.

### 12. [u-μP: The Unit-Scaled Maximal Update Parametrization](https://arxiv.org/abs/2407.17465) - 2024

**Connects to:** muP variants, unit scaling, activation and gradient magnitudes, simpler defaults, and low-precision training.

**Very short abstract:** u-muP combines maximal-update scaling with unit-scaled forward and backward quantities. The resulting scheme aims to preserve hyperparameter transfer while simplifying defaults and improving numerical behavior in low precision.

**Why it is useful:** It shows how the muP program is evolving in response to practical complexity and numerical constraints.

**Bigger picture:** Parameterization, optimizer scaling, and numerical precision are becoming a single co-designed training system.

### 13. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) - 2025

**Connects to:** Muon, spectral orthogonalization, update scale, weight decay, distributed implementation, and AdamW comparisons.

**Very short abstract:** The paper adapts the Muon optimizer for large language-model training by controlling update magnitudes and adding weight decay. It studies compute scaling and demonstrates the optimizer in a multi-trillion-token mixture-of-experts training run.

**Why it is useful:** It is the primary empirical source for the lecture's discussion of Muon at meaningful LLM scale.

**Bigger picture:** Optimizer claims increasingly need both carefully tuned small-scale comparisons and proof that the method survives distributed frontier-scale systems.

### 14. [Predictable Scale: Part I, Step Law -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining](https://arxiv.org/abs/2503.04715) - 2025

**Connects to:** Step Law, high-resolution learning-rate and batch-size grids, model and data dependence, optimal plateaus, and transfer across architectures.

**Very short abstract:** This work searches a large grid of models, data budgets, learning rates, and batch sizes to fit empirical laws for preferred hyperparameters. It emphasizes a near-optimal plateau and tests the fitted relationships across dense and mixture-of-experts configurations and different data distributions.

**Why it is useful:** It is the modern large-scale extension of the lecture's "fit the drift" approach.

**Bigger picture:** The study illustrates both the promise and fragility of turning expensive hyperparameter searches into reusable predictive tools.

## Suggested reading order

1. **Start here:** 3 (Kaplan scaling), 4 (Chinchilla), 5 (Tensor Programs V), and 9 (DeepSeek LLM).
2. **Understand sensitive hyperparameters:** 1 (Adam), 2 (critical batch size), 8 (spectral feature learning), and 10 (MiniCPM).
3. **Test transfer claims:** 6 (Cerebras-GPT), 7 (data-constrained scaling), and 11 (empirical muP study).
4. **Modern extensions:** 12 (u-muP), 13 (Muon), and 14 (Step Law).
