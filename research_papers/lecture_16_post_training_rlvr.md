---
title: "Lecture 16 - Post-Training and RLVR: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 16
companion_notes: "../lecture_notes/lecture_16_post_training_rlvr.md"
status: "complete"
---

# Lecture 16: Post-Training and RLVR - Research Paper Guide

> The summaries below are original, compact descriptions of the papers rather than copied abstracts. Links point to primary paper or proceedings pages.

## Reading map

| Theme | Papers to prioritize |
|---|---|
| Policy-gradient foundations | REINFORCE; GAE; PPO |
| Verifiers and reasoning supervision | Let's Verify Step by Step; STaR |
| GRPO and RL with verifiable rewards | DeepSeekMath; DeepSeek-R1; Understanding R1-Zero-Like Training; DAPO |
| Scaling reasoning and coding agents | Kimi k1.5; Qwen3; Qwen3-Coder-Next; SWE-RL |
| RL training infrastructure | HybridFlow |

## Curated papers

### 1. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) - 1992

**Connects to:** policy gradients, score-function estimators, and sequence-level rewards.

**Very short abstract:** Williams derives the REINFORCE family of stochastic gradient estimators, showing how a parameterized policy can be improved directly from sampled actions and scalar rewards. The estimator is unbiased but can have high variance.

**Why it is useful:** It supplies the mathematical ancestor of the policy-gradient updates used in modern language-model reinforcement learning.

**Bigger picture:** RLVR changes the model, reward, and sampling scale, but its core learning signal still follows this log-probability-times-return pattern.

### 2. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - 2015

**Connects to:** advantages, baselines, bias-variance trade-offs, and critic-based training.

**Very short abstract:** The paper introduces generalized advantage estimation (GAE), which combines multi-step temporal-difference residuals using a tunable decay factor. This gives practitioners a practical way to trade estimator bias against variance.

**Why it is useful:** It makes the role of an advantage estimate precise and explains why subtracting a suitable baseline stabilizes policy-gradient learning.

**Bigger picture:** GRPO and related language-model algorithms can be understood partly as alternative ways to construct normalized advantages without relying on a conventional per-state value model.

### 3. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - 2017

**Connects to:** clipped objectives, importance ratios, multiple updates per rollout, and trust-region-like control.

**Very short abstract:** PPO proposes simple surrogate objectives that prevent a new policy from moving too far from the policy that generated the data. Its clipped probability-ratio objective became a widely used compromise between stability, simplicity, and data reuse.

**Why it is useful:** The paper is the clearest foundation for understanding the clipping and old-policy machinery inherited by many RLHF and RLVR systems.

**Bigger picture:** Later methods such as GRPO modify the advantage and critic components, but remain closely related to PPO's on-policy surrogate objective.

### 4. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - 2024

**Connects to:** Group Relative Policy Optimization (GRPO), mathematical verifiers, and reasoning post-training.

**Very short abstract:** DeepSeekMath combines a mathematics-focused pretraining corpus with supervised fine-tuning and reinforcement learning. It introduces GRPO, which estimates relative advantages from groups of sampled answers and removes the separate critic used by PPO.

**Why it is useful:** It is the primary source for the algorithm that anchors much of the lecture's discussion of contemporary RLVR.

**Bigger picture:** The work helped establish verifiable mathematics as a practical test bed for scaling reinforcement learning in language models.

### 5. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - 2023

**Connects to:** process reward models, outcome reward models, step-level labels, and best-of-N selection.

**Very short abstract:** The authors compare supervising only the final answer with supervising each intermediate reasoning step on mathematical problems. They find process supervision substantially improves reward-model reliability and release the PRM800K step-level dataset.

**Why it is useful:** It clarifies that a verifier can judge either an entire trajectory or local reasoning steps, with different data and robustness implications.

**Bigger picture:** RLVR often exploits cheap final-answer checks, while process supervision points toward denser and potentially safer feedback for difficult reasoning tasks.

### 6. [STaR: Bootstrapping Reasoning With Reasoning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) - 2022

**Connects to:** rejection sampling, self-generated rationales, supervised iteration, and reasoning-data bootstrapping.

**Very short abstract:** STaR repeatedly asks a language model to generate rationales, retains solutions that lead to correct answers, and fine-tunes on the successful traces. When a direct attempt fails, a supplied answer can help the model rationalize a usable solution for the next round.

**Why it is useful:** It provides a clean supervised-learning comparison to online policy optimization: both use model-generated trajectories and a correctness signal, but update the model differently.

**Bigger picture:** Many modern reasoning pipelines combine rejection-sampled fine-tuning with reinforcement learning rather than treating them as mutually exclusive approaches.

### 7. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - 2025

**Connects to:** R1-Zero, pure RL, cold-start data, emergent long-form reasoning, and distillation.

**Very short abstract:** DeepSeek-R1-Zero applies large-scale reinforcement learning without an initial supervised reasoning stage and develops behaviors such as self-verification and extended deliberation. DeepSeek-R1 adds cold-start data and a multi-stage pipeline to improve readability and general capability, then distills reasoning into smaller models.

**Why it is useful:** It is a central empirical demonstration that verifiable rewards can elicit strong reasoning behavior at scale.

**Bigger picture:** The paper shifted attention from merely imitating written chains of thought toward optimizing reasoning trajectories through interaction and objective feedback.

### 8. [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) - 2025

**Connects to:** GRPO failure modes, response-length bias, difficulty bias, and Dr. GRPO.

**Very short abstract:** This analysis studies why R1-Zero-style training appears to work and identifies biases introduced by common normalization and aggregation choices. It proposes Dr. GRPO, a simplified variant designed to avoid incentives that can artificially favor longer responses or skew optimization across problems.

**Why it is useful:** It turns implementation details in the RL objective into concrete, testable explanations for surprising training behavior.

**Bigger picture:** As RLVR matures, careful objective auditing is as important as scaling compute, because apparent reasoning gains can partly reflect estimator artifacts.

### 9. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) - 2025

**Connects to:** decoupled clipping, dynamic sampling, token-level losses, overlong-response control, and large-scale RL recipes.

**Very short abstract:** DAPO presents a reproducible reinforcement-learning system and a collection of modifications aimed at stable mathematical-reasoning training. Its recipe addresses issues such as asymmetric policy updates, batches with no useful reward variation, length effects, and reward shaping for overlong outputs.

**Why it is useful:** It shows how several seemingly small algorithmic and data-pipeline decisions jointly determine whether large-scale RLVR succeeds.

**Bigger picture:** The work represents the move from a single named algorithm toward complete, engineered training recipes that expose and repair failure modes.

### 10. [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599) - 2025

**Connects to:** long-context policy optimization, curriculum design, long versus short chain of thought, and multimodal reasoning.

**Very short abstract:** Kimi k1.5 develops an RL training recipe that scales long-context reasoning using outcome rewards, length-aware sampling, curriculum design, and improved policy optimization. It also explores transferring long-chain reasoning ability into shorter responses and applies the recipe across text and vision tasks.

**Why it is useful:** It broadens the lecture's RLVR picture beyond one model family and highlights context length and sampling strategy as core scaling variables.

**Bigger picture:** Reasoning performance depends not only on the reward but also on the trajectory budget, data mixture, and distribution of problems encountered during training.

### 11. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - 2025

**Connects to:** hybrid thinking and non-thinking modes, thinking budgets, staged post-training, and distillation.

**Very short abstract:** Qwen3 trains a single model family to support both deliberate reasoning and fast direct responses. Its post-training pipeline combines long-chain-of-thought cold starts, reasoning RL, mode fusion, general RL, and transfer from larger teacher models.

**Why it is useful:** It demonstrates how RLVR-style reasoning can be integrated with ordinary instruction following rather than shipped as a separate specialist model.

**Bigger picture:** Deployable reasoning systems increasingly expose computation as a controllable resource, letting users trade latency and tokens for harder-task performance.

### 12. [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729) - 2026

**Connects to:** executable coding environments, verifiable software tasks, synthetic task generation, and agentic reinforcement learning.

**Very short abstract:** Qwen3-Coder-Next trains a sparse code model with large-scale executable tasks and environment interaction, combining continued training with reinforcement learning. The report emphasizes verifiable outcomes from code execution and evaluation in agentic software-engineering settings.

**Why it is useful:** Coding supplies richer and more diverse verification signals than exact-match mathematics while still allowing objective execution-based feedback.

**Bigger picture:** RLVR is expanding from static questions toward agents that take multi-step actions in tools and must leave an environment in a correct state.

### 13. [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449) - 2025

**Connects to:** software repositories as environments, issue resolution, code verification, and domain-specific RL data.

**Very short abstract:** SWE-RL constructs reinforcement-learning tasks from the history of open-source software projects and trains models to reason over realistic code changes. Repository evolution provides a scalable source of problems whose proposed patches can be checked with programmatic signals.

**Why it is useful:** It offers a concrete route from mathematical RLVR to software-engineering agents grounded in real codebases.

**Bigger picture:** The work illustrates a general strategy for creating RL environments from naturally occurring digital traces with automatically testable outcomes.

### 14. [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256) - 2024

**Connects to:** rollout generation, actor and critic placement, distributed dataflow, and RLHF/RLVR systems engineering.

**Very short abstract:** HybridFlow separates reinforcement-learning dataflow from distributed computation primitives so that different worker arrangements and model roles can be composed efficiently. It introduces a hybrid execution design aimed at reducing communication and memory overhead in large-model post-training.

**Why it is useful:** It explains the systems layer beneath an RL algorithm: multiple models must alternate between high-throughput generation and memory-intensive training across accelerators.

**Bigger picture:** At frontier scale, algorithm and infrastructure co-design determines how many useful trajectories can be collected and learned from per unit of compute.

## Suggested reading order

1. Start with REINFORCE, GAE, and PPO for the mathematical vocabulary.
2. Read DeepSeekMath to see how GRPO adapts that vocabulary to language-model reasoning.
3. Compare process supervision in *Let's Verify Step by Step* with STaR's rejection-sampling loop.
4. Read DeepSeek-R1, followed immediately by the critical R1-Zero analysis and DAPO.
5. Use Kimi k1.5 and Qwen3 to compare complete modern reasoning recipes.
6. Finish with Qwen3-Coder-Next, SWE-RL, and HybridFlow to connect RLVR to agents and distributed systems.
