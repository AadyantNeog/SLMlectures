---
title: "Lecture 15 - Mid- and Post-Training: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 15
companion_notes: "../lecture_notes/lecture_15_mid_post_training.md"
status: "complete"
---

# Lecture 15: Mid- and Post-Training - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

The papers trace the modern post-training stack from instruction data to preference learning. They also show why data provenance, annotator judgments, model feedback, optimization constraints, and reward overoptimization cannot be treated as separate concerns.

| Lecture theme | Papers |
|---|---|
| Instruction tuning and synthetic demonstrations | 1-2 |
| Open conversations, real-user prompts, and open recipes | 3-5 |
| Human-preference learning and language-model RLHF | 6-8 |
| AI feedback and scalable supervision | 9-10 |
| Trust regions, PPO, DPO, and overoptimization | 11-14 |

## Curated papers

### 1. [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) - 2022

**Connects to:** FLAN, task mixtures, natural-language templates, and zero-shot instruction generalization.

**Very short abstract:** FLAN instruction-tunes a pretrained model on many datasets expressed through natural-language task templates. The resulting model transfers more effectively to held-out task types without task-specific examples.

**Why it is useful:** It is the clean foundational demonstration that diverse instruction tuning can change the interface of a pretrained model.

**Bigger picture:** Post-training evolved from narrow task fine-tuning toward broad behavioral adaptation through mixtures of instructions.

### 2. [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/) - 2023

**Connects to:** Synthetic instructions, teacher generation, filtering, bootstrapping, and Alpaca-style data.

**Very short abstract:** Self-Instruct prompts a language model to generate new instructions and corresponding instances, filters invalid or redundant examples, and fine-tunes on the surviving data. It substantially reduces dependence on a large hand-authored instruction collection.

**Why it is useful:** It provides the conceptual and operational foundation for many later synthetic SFT pipelines.

**Bigger picture:** Capable models became data producers, moving the bottleneck from writing every example to designing and auditing generation loops.

### 3. [OpenAssistant Conversations - Democratizing Large Language Model Alignment](https://proceedings.neurips.cc/paper_files/paper/2023/hash/949f0f8f32267d297c2d4e3ee10a2e7e-Abstract-Datasets_and_Benchmarks.html) - 2023

**Connects to:** Conversation trees, crowd annotation, multilingual SFT, preference ratings, and open alignment data.

**Very short abstract:** OpenAssistant releases a large crowd-created collection of assistant conversations organized as branching trees with quality annotations. The paper documents the collection process and trains open assistant and preference models from the corpus.

**Why it is useful:** It exposes what human-authored chat data and preference metadata look like beyond a flat prompt-response table.

**Bigger picture:** Open post-training research requires not only model weights but also inspectable interaction data and annotation protocols.

### 4. [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470) - 2024

**Connects to:** Real-user prompts, failure mining, multilingual and toxic content, long-tail behavior, and WildChat.

**Very short abstract:** WildChat collects a large set of naturally occurring user-assistant conversations and analyzes how they differ from cleaner benchmark or crowdsourced prompts. The data contains varied languages, use cases, risk categories, and interaction patterns.

**Why it is useful:** It shows why production-aligned training and evaluation need prompts drawn from actual usage distributions.

**Bigger picture:** Real-world logs close the loop between deployment failures and the next generation of post-training data.

### 5. [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) - 2024

**Connects to:** SFT mixtures, DPO, RLVR, decontamination, safety, evaluation, and open recipes.

**Very short abstract:** Tulu 3 presents a fully open post-training pipeline combining curated SFT, preference optimization, and verifiable-reward training. It releases data, code, models, and an evaluation framework designed to distinguish development gains from unseen-task transfer.

**Why it is useful:** It is one of the clearest end-to-end references for reproducing a modern open post-training stack.

**Bigger picture:** Post-training is best understood as a staged system whose data, objectives, and evaluations must be co-designed.

### 6. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) - 2022

**Connects to:** InstructGPT, demonstrations, ranking data, reward models, PPO, annotator guidelines, and alignment tax.

**Very short abstract:** InstructGPT first trains on labeler demonstrations, learns a reward model from ranked outputs, and then optimizes the policy with RL while constraining drift from the SFT model. Human evaluation shows that post-training can make a much smaller base model more useful than a larger untuned one on the target prompt distribution.

**Why it is useful:** It is the canonical language-model RLHF pipeline against which later simplifications are compared.

**Bigger picture:** Model usefulness depends not only on pretraining capability but on an explicit behavioral objective and the people defining it.

### 7. [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) - 2017

**Connects to:** Pairwise preferences, learned rewards, scalable feedback, and separating goal learning from policy learning.

**Very short abstract:** The paper learns a reward function from human comparisons between trajectory segments and uses that reward to train deep RL policies. It demonstrates that sparse preference judgments can specify behaviors that are difficult to encode with a hand-written reward.

**Why it is useful:** It supplies the general preference-learning template later adapted to language-model outputs.

**Bigger picture:** RLHF is an instance of a broader attempt to learn objectives from human judgments rather than assume the reward is given.

### 8. [Learning to Summarize with Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) - 2020

**Connects to:** Sequence-level preferences, reward modeling, KL-regularized policy optimization, and capability-versus-preference distinctions.

**Very short abstract:** This work collects comparisons between model summaries, trains a reward model, and optimizes a summarization policy against that learned reward. It shows how human preference can replace a convenient but incomplete automatic metric such as ROUGE.

**Why it is useful:** It bridges generic preference-based RL and the later general-purpose assistant pipeline.

**Bigger picture:** Post-training is most valuable when the desired property is holistic and cannot be expressed as token-level imitation alone.

### 9. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - 2022

**Connects to:** Safety tuning, self-critique, revision, refusal behavior, AI preferences, and constitutions.

**Very short abstract:** Constitutional AI uses a written set of principles to guide model-generated critiques, revisions, and preference labels. It combines a supervised self-improvement stage with reinforcement learning from AI feedback to reduce dependence on direct harmfulness labels from humans.

**Why it is useful:** It demonstrates how behavioral specifications can be made more explicit than a collection of opaque preference choices.

**Bigger picture:** Scalable oversight shifts part of human labor from labeling every output to defining principles and auditing model-mediated judgments.

### 10. [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) - 2023

**Connects to:** Model-generated feedback, reward-model distillation, direct scoring, cost, and judge bias.

**Very short abstract:** This study compares human and language-model preference labels for summarization and dialogue alignment. It also explores directly using an external model's scores during RL instead of first distilling them into a separate reward model.

**Why it is useful:** It provides a concrete empirical comparison between human and AI feedback pipelines.

**Bigger picture:** AI feedback can scale annotation, but it also imports the evaluator model's blind spots and preferences into the trained policy.

### 11. [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html) - 2015

**Connects to:** KL constraints, stable policy updates, monotonic-improvement intuition, and the ancestry of PPO.

**Very short abstract:** TRPO derives a practical policy-update method that approximately constrains each new policy to remain near the previous one. The trust region is designed to avoid destructive large steps while still permitting iterative improvement.

**Why it is useful:** It clarifies why RLHF objectives penalize policy drift rather than maximize learned reward without restraint.

**Bigger picture:** PPO and KL-regularized language-model RL are engineering descendants of trust-region policy optimization.

### 12. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - 2017

**Connects to:** PPO clipping, multiple minibatch updates, policy ratios, and practical RLHF optimization.

**Very short abstract:** PPO replaces the more involved trust-region machinery with surrogate objectives that limit harmful policy-ratio changes. It supports several epochs of minibatch optimization while retaining a practical notion of conservative policy improvement.

**Why it is useful:** It is the algorithmic reference needed to understand the most common classical RLHF optimizer.

**Bigger picture:** PPO made preference-based language-model training tractable, but its many implementation details motivated simpler offline alternatives.

### 13. [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) - 2023

**Connects to:** DPO, Bradley-Terry preferences, the KL-regularized RLHF optimum, reference policies, and offline learning.

**Very short abstract:** DPO reparameterizes a class of KL-regularized preference objectives so the policy can be trained directly on preferred and rejected responses. It avoids fitting an explicit reward model and avoids an online RL sampling loop.

**Why it is useful:** It makes the lecture's DPO derivation and its simplifying assumptions concrete.

**Bigger picture:** Direct alignment trades operational simplicity for stronger dependence on the coverage and quality of a fixed preference dataset.

### 14. [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) - 2022

**Connects to:** Goodhart's law, proxy rewards, best-of-n, KL budgets, calibration, and reward hacking.

**Very short abstract:** The paper studies how a gold reward changes as a policy or sampler increasingly optimizes a learned proxy reward. It finds predictable overoptimization curves and measures how they vary with reward-model scale, data, and optimization method.

**Why it is useful:** It provides a quantitative model of why reward scores can rise after true answer quality has begun to fall.

**Bigger picture:** Every post-training method needs an independent evaluation channel because optimizing the evaluator eventually changes the distribution it must judge.

## Suggested reading order

1. **Start with instruction tuning:** 1, 2, and 6.
2. **Understand preference learning:** 7, 8, 11, and 12.
3. **Move to modern alternatives:** 13, 9, and 10.
4. **Study open practice and failure modes:** 3, 4, 5, and 14.
