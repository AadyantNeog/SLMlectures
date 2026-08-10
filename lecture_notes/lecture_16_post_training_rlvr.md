---
title: "Lecture 16 - Post-Training with RLVR"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 16
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 16 Post-Training - RLVR.txt"
slide_deck: "../lecture_16.pdf"
status: "complete"
---

# Lecture 16: Post-Training with Reinforcement Learning from Verifiable Rewards

## How to read these notes

Each substantive topic has two layers:

1. **What the lecturer said - transcript only.** A concise paraphrase of the complete spoken lecture, preserving substantive claims, examples, qualifications, numerical details, and audience questions while removing filler and repetition.
2. **Additional explanation.** Independent intuition, derivation, organization, and study guidance. This material is not attributed to the transcript.

The raw transcript's line spans are shown for auditability. All 61 slides were rendered and visually inspected after the transcript map was established. Slides were used to verify model and paper names, figures, implementation details, and equations. Material differences or slide-only precision appear under **Source reconciliation**.

## Lecture map

The lecture develops modern reasoning-model post-training in five stages:

1. Motivate reinforcement learning from verifiable rewards (RLVR) as a response to reward-model overoptimization in conventional RLHF.
2. Revisit policy gradients and PPO, then derive the simpler group relative policy optimization (GRPO) recipe.
3. Examine GRPO critically, especially its standard-deviation and sequence-length normalizations.
4. Read DeepSeek-R1, Kimi k1.5, and Qwen3 as concrete recipes that combine SFT, reasoning RL, general RLHF, distillation, curriculum design, and systems engineering.
5. Extend the same framework to coding agents, where the central danger is that a supposedly verifiable reward can still be hacked.

---

# Part I - Why RLVR, and why move beyond PPO?

## 1. From RLHF overoptimization to verifiable-reward reasoning

**Transcript coverage:** lines 1-147

### What the lecturer said - transcript only

This is the second post-training lecture. Its subject is reinforcement learning from verifiable rewards, or RLVR, and the timing is especially relevant because an OpenAI thinking model had just solved a major MyOpenMath problem. The question is how systems with long chains of thought solve hard, checkable mathematics and coding problems.

The previous lecture ended with instruction tuning and RLHF, roughly the route from a base model to ChatGPT. The remaining transition is toward o1- or R1-style reasoning models. Conventional preference-based RLHF runs into overoptimization: people provide comparisons, those comparisons train a reward model, and the policy is optimized against that fixed model. Human annotation is a bottleneck, and continued optimization eventually exploits or overfits the learned reward even under careful regularization.

That limitation does not reflect the full promise of reinforcement learning. AlphaGo is a contrasting case because the win-loss objective is exactly the desired outcome. There is no ambiguity in whether a game was won, so additional compute can be useful as long as the true objective improves. The lecturer loosely characterizes this as more search-like than ordinary RLHF's learning problem, while warning that the distinction is not exact.

Formal mathematics, natural-language mathematics, and some coding tasks can have more checkable outcomes. Their rewards may therefore tolerate much more reinforcement-learning compute. The algorithms are not fundamentally alien, but the results can be strikingly different.

The lecture has two parts. First come the core methods needed for the course assignments: PPO, GRPO, and their updates. Then the same ideas will be located inside major open model reports.

### Source reconciliation

The opening slides make the course progression explicit as GPT-3.5 or ChatGPT to o1 or R1, with long chain-of-thought and test-time compute as the new capabilities. They state the target contrast compactly: a learned preference reward is vulnerable to overoptimization, whereas a narrow-domain reward should measure exactly what the builder wants.

### Additional explanation

The decisive axis is **objective fidelity**, not whether the feedback happens to be called human or verifiable:

| Setting | Reward source | Main scaling limit |
|---|---|---|
| Preference RLHF | A learned model of human judgments | Proxy exploitation and finite annotation coverage |
| Game-playing RL | Exact environment outcome | Search and optimization compute |
| Math or code RLVR | A checker, tests, or answer judge | Checker coverage, false positives, and exploitability |

RLVR is powerful when the reward remains aligned with the real task under adversarial optimization. "Verifiable" should therefore be read as an engineering claim that must be tested, not as a guarantee.

## 2. Policy gradients as positively or negatively weighted SFT

**Transcript coverage:** lines 148-236

### What the lecturer said - transcript only

PPO is the starting point because language-model reinforcement learning cannot be discussed seriously without it. The central idea to remember is the REINFORCE policy-gradient update. In practical terms, RL repeatedly performs SFT-like log-probability updates weighted by reward or advantage. A positive weight raises the probability of an output; a negative weight lowers it. This weighted-SFT view is the core from which the lecture's later algorithms are derived.

A basic policy gradient normally needs fresh samples from the current policy for each update. TRPO and PPO address the desire to reuse rollouts and take more than one optimization step. The lecturer does not revisit TRPO in detail.

PPO has been a workhorse across difficult, high-dimensional reinforcement-learning settings. Early OpenAI demonstrations included locomotion in Gym-like environments and the much larger OpenAI Five system. These examples show that PPO can work in action and state spaces far more complex than a simple bandit.

### Source reconciliation

The slides write the policy-gradient estimator explicitly and then show PPO's clipped surrogate objective. They also use OpenAI locomotion and OpenAI Five as the visual examples of PPO's broad historical role.

### Additional explanation

For a prompt $x$, sampled response $y$, and scalar reward $r(x,y)$, the basic sequence-level estimator is

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_{x,y\sim\pi_\theta}
\left[
r(x,y)\nabla_\theta \log \pi_\theta(y\mid x)
\right].
$$

Because

$$
\log \pi_\theta(y\mid x)
=
\sum_{t=1}^{T}\log \pi_\theta(y_t\mid x,y_{<t}),
$$

the same sequence reward weights every generated-token log probability unless the algorithm supplies token-specific credit. This is why the lecturer's "weighted SFT" analogy is exact at the gradient level but leaves open the difficult questions: Which scalar should be used? What baseline reduces variance? Can the samples be reused after the policy changes?

## 3. PPO looks short in pseudocode but is implementation-sensitive

**Transcript coverage:** lines 237-384

### What the lecturer said - transcript only

At a conceptual level, PPO appears simple. Sample trajectories, estimate advantages, update the policy with a clipped objective, and fit a value function. OpenAI's Spinning Up pseudocode is short enough that a reader might expect to implement it in one pass.

The reality is signaled by articles cataloguing dozens of implementation details. Different libraries and choices can yield different numbers, and some published "baselines" do not merely reduce variance: they alter the optimization problem. This sensitivity motivates looking beneath the pseudocode.

Language-model PPO has many interacting components. It maintains a policy, a reference model, a learned value model, experience storage, and advantage estimation. Some terms, especially KL control, act token by token rather than only on a terminal sequence reward. The resulting problem has multi-step structure and many engineering decisions.

The lecturer recalls a student's robust RLHF PPO implementation that took substantial effort to stabilize. Its outer loop looked ordinary: collect rollouts, take PPO steps, compute the loss, clip gradients, and update. The inner loss also appeared faithful, with advantages and importance ratios. The trouble emerged in the less visible details.

### Source reconciliation

The slides contrast the compact PPO pseudocode with "The 37 Implementation Details of Proximal Policy Optimization." A language-model PPO diagram shows the policy model, supervised reference model, reward model, value model, generalized advantage estimation, rollout buffer, per-token KL rewards, and a final sequence reward in a single training system.

The exact clipped surrogate shown on the slides is

$$
L^{\mathrm{clip}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
\rho_t(\theta)\hat A_t,\,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right],
$$

where

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

### Additional explanation

PPO's clip is a guardrail on **policy change measured on collected actions**. If an action has positive advantage, increasing its probability helps until the ratio passes the upper clip. If it has negative advantage, decreasing its probability helps until the lower clip. The minimum is chosen so the clipped form removes the incentive for a very large beneficial-looking change.

The pseudocode hides three separate statistical problems:

1. **Off-policy drift:** after several optimizer steps, stored samples no longer come from the current policy.
2. **Credit assignment:** a terminal score must influence many token decisions.
3. **Variance control:** value and advantage estimates can reduce noise but introduce their own training errors.

Each is manageable in isolation. Their combination with large autoregressive models, distributed inference, and KL regularization creates the operational complexity.

## 4. Reward shaping, GAE degeneracy, memory cost, and fragile stabilizers

**Transcript coverage:** lines 385-488

### What the lecturer said - transcript only

One implementation needed a KL penalty to keep the policy near the reference model, yet training only remained stable when the estimated per-token KL contribution was clipped at zero. That operation undermines the mathematical composition of a KL divergence because positive and negative token contributions are supposed to sum together. The lecturer presents this as an anecdote about sensitivity, not as a recommended implementation.

High-variance gradient estimates and many moving parts make such stabilizing hacks tempting. Generalized advantage estimation introduces discount and trace parameters, usually written $\gamma$ and $\lambda$, and a value estimate at each generation step. In language-model PPO, however, implementations often set both parameters to one. This is a degenerate choice that largely returns the problem to a bandit-like or return-to-go setting and discards much of PPO's temporal structure.

A successful run should still show intuitive curves: task or reward-model reward rises while the negative KL regularizer becomes more costly as the policy moves away from the reference. PPO is possible at scale, and laboratories may have reliable internal systems, but it is finicky for researchers building it from scratch.

PPO also requires a value model that can be about as large as the policy. Its parameters, activations, and optimizer state consume memory that could otherwise support the policy or rollout servers.

### Source reconciliation

The implementation slides show token-level KL shaping, terminal placement of the task reward, clipping a particular KL estimate at a minimum of zero, and a standard PPO clip range of $0.2$. The GAE slide verifies the commonly used setting $\gamma=\lambda=1$. The expected training plot separately tracks environment reward, reward-model score, and non-positive KL contribution.

### Additional explanation

With temporal-difference residual

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t),
$$

generalized advantage estimation uses

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
$$

Setting $\gamma=\lambda=1$ telescopes under a terminal reward:

$$
\hat A_t
\approx
R_{\mathrm{terminal}}-V(s_t).
$$

This can be sensible for a completion whose quality is only known at the end, but it means the sophisticated multi-step apparatus is providing less structure than the full notation suggests. A separate value model still predicts a baseline for every prefix, which is expensive and can itself become unstable.

## 5. Why DPO is not the general substitute, and why GRPO became attractive

**Transcript coverage:** lines 489-540

### What the lecturer said - transcript only

DPO is not a universal replacement for PPO. It was designed for a specific form of feedback: pairwise Bradley-Terry preferences. A mathematics task naturally provides correctness or a scalar reward, not necessarily a preferred and rejected response pair. Variants can loosen the pairwise structure, but using them here can amount to choosing the wrong tool.

PPO is the more general method. DPO is often described as offline, although the lecturer considers that distinction overstated because DPO can be iterated with newly sampled data.

Researchers nevertheless strongly wanted simpler alternatives to PPO. The adoption of DPO and GRPO is itself evidence of PPO's practical burden. GRPO preserves PPO's overall spirit while removing some of its most complicated components, and it has become the common open-research method for RLVR.

### Source reconciliation

The comparison slide summarizes the tradeoff as follows: PPO is general but finicky and requires a large value model; DPO is simpler but specifically tied to pairwise preference data; GRPO targets scalar or verifiable rewards while eliminating the learned critic.

### Additional explanation

The feedback structures differ:

$$
\text{DPO input: }(x,y^+,y^-)
$$

$$
\text{RLVR input: }(x,y,r(x,y)).
$$

One can manufacture pairs by declaring a correct response preferred to an incorrect one, but this discards information about groups, reward magnitudes, repeated samples, and online exploration. GRPO uses the native scalar-reward interface while obtaining a baseline from sibling samples of the same prompt.

---

# Part II - GRPO: the simple objective and its non-simple biases

## 6. GRPO replaces the learned value model with a group-relative advantage

**Transcript coverage:** lines 541-696

### What the lecturer said - transcript only

GRPO was introduced in DeepSeek's mathematics work. It starts from PPO but removes the learned value function, which is both expensive and a source of instability. Vanilla REINFORCE without a baseline would have high variance, so GRPO builds an advantage from several completions sampled for the same prompt.

Suppose a rollout receives reward five. A value-model approach compares five with a neural prediction of the expected reward. GRPO instead samples a group of other rollouts and asks whether five is above or below that group's mean. Each reward is standardized by subtracting the group mean and dividing by the group standard deviation.

The original GRPO objective retains the PPO clipped update and a KL term that constrains the policy relative to a reference. In a strictly online update, the old policy and sampling policy are initially the same, so their probability ratio is one. Before policy drift, clipping has no effect and the update reduces conceptually to group-relative advantage with KL regularization.

This simple objective is foundational to much of the recent open reasoning-model work, so the lecturer pauses to make sure its conceptual pieces are clear.

### Source reconciliation

The GRPO slide defines, for a group of $G$ sampled outputs,

$$
A_i
=
\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}.
$$

It then inserts $A_i$ into a PPO-style clipped token objective and subtracts a reference-policy KL penalty. The slide explicitly notes that when rollouts are freshly sampled from the current policy, the old-to-current ratio begins at one.

### Additional explanation

The group mean is a prompt-dependent baseline:

$$
b(x)=\frac{1}{G}\sum_{j=1}^{G}r(x,y_j).
$$

The numerator $r_i-b(x)$ answers a useful local question: among responses to this same prompt, which ones were better than the policy's current typical behavior? This controls prompt difficulty automatically. A raw reward of one means something different for a prompt solved almost always than for one solved once in eight attempts.

For binary rewards, a mixed group creates positive weights for correct samples and negative weights for incorrect samples. An all-correct or all-incorrect group has no within-group ranking signal, which motivates careful numerical handling and curriculum design.

## 7. A one-page implementation, numerical protection, and early evidence

**Transcript coverage:** lines 697-820

### What the lecturer said - transcript only

Without a value network, basic GRPO fits in a small implementation. For each prompt, sample $k$ rollouts, compute rewards, normalize them within the group, compute the KL term, and take weighted policy-gradient updates. Automatic differentiation requires a stop-gradient at the appropriate point so that the scalar weights are not themselves optimized through.

A compact McGill reference implementation adds a small $10^{-4}$ quantity to the standard-deviation calculation. This prevents division by zero with one sample or with identical rewards, such as a group in which every attempted math solution receives zero.

The original DeepSeekMath results showed GRPO outperforming rejection fine-tuning, which simply trains on correct generated responses and discards incorrect ones. The results also suggested gains from process supervision, where intermediate reasoning steps are graded rather than only the final answer. The later case studies revisit that design choice.

### Source reconciliation

The code slide confirms an $\epsilon=10^{-4}$ stabilizer in the group standard deviation and makes the detach or stop-gradient operation explicit. The DeepSeekMath result slide compares rejection fine-tuning, outcome-supervised GRPO, and process-supervised variants; the GRPO curves are above rejection fine-tuning, with some further improvement from process supervision in that experiment.

### Additional explanation

A minimal conceptual loop is:

1. Sample $G$ completions per prompt from the rollout policy.
2. Score each completion without gradients through the scorer.
3. Form detached advantages from group rewards.
4. Sum token log probabilities for each completion.
5. Optimize advantage-weighted log probability plus a reference-policy constraint.
6. Refresh rollouts frequently enough that the update remains close to on-policy.

The detach is not cosmetic. If a differentiable proxy reward or normalization statistic were allowed to backpropagate through the advantage, the implementation would no longer be the intended score-function estimator.

Rejection fine-tuning only reinforces successful samples. GRPO additionally pushes down unsuccessful samples relative to their group. The negative update can extract more information from each rollout, but it can also create pathological incentives when its scale depends on length or group variance.

## 8. A valid REINFORCE baseline may be subtracted, but GRPO also divides

**Transcript coverage:** lines 821-960

### What the lecturer said - transcript only

The lecturer asks whether GRPO's standardized group reward is a principled advantage. In REINFORCE with a baseline, one may subtract any state-dependent value that does not depend on the sampled action. In the bandit view of language generation, the state is the prompt, so a prompt-dependent baseline preserves the expected policy-gradient direction while potentially reducing variance.

GRPO does more than subtract a group mean: it divides by the group's reward standard deviation. That scaling breaks the simple baseline guarantee. It changes how prompts are weighted, so GRPO does not directly descend the unmodified expected-reward objective in the first-principles sense.

Standard GRPO also normalizes a sequence-level update by output length, effectively spreading it across tokens. A direct policy-gradient derivation produces neither this length normalization nor the standard-deviation division. Work described as Dr. GRPO removes both factors and reports different, potentially favorable behavior.

The point is not that GRPO cannot work. It is that its standard form is a heuristic objective with consequences that must be understood, not a transparent implementation of the original reward objective.

### Source reconciliation

The slides contrast

$$
\mathbb{E}\left[(r-b(x))\nabla_\theta\log\pi_\theta(y\mid x)\right]
$$

with GRPO's

$$
\mathbb{E}\left[
\frac{r-\bar r_x}{s_x}
\nabla_\theta\log\pi_\theta(y\mid x)
\right].
$$

They identify two corrections in Dr. GRPO: remove the group standard-deviation normalizer and remove the response-length normalizer. The cited comparison is Liu et al. (2025).

### Additional explanation

For any baseline $b(x)$ independent of the sampled response,

$$
\mathbb{E}_{y\sim\pi_\theta}
\left[
b(x)\nabla_\theta\log\pi_\theta(y\mid x)
\right]
=
b(x)\nabla_\theta\sum_y\pi_\theta(y\mid x)
=0.
$$

Therefore subtracting $b(x)$ does not change the expected gradient. Multiplying each prompt's gradient by $1/s_x$ is different: it reweights prompts according to their within-group variability.

This yields a useful distinction:

- **Baseline:** changes estimator variance without changing the target in expectation.
- **Normalizer:** changes the relative importance of training examples and can change the target.

The group mean is itself estimated from samples, so exact finite-group estimators require care about whether a response contributes to its own baseline. The lecture's larger point survives that detail: standard-deviation scaling is not covered by the ordinary baseline theorem.

## 9. Sequence-length and group-variance normalization produce specific biases

**Transcript coverage:** lines 961-1076

### What the lecturer said - transcript only

Length normalization creates an especially intuitive failure. If an incorrect response receives a negative reward and that penalty is divided by response length, emitting more tokens dilutes the negative update. In the extreme, an infinitely long wrong answer would make its per-token penalty approach zero. The model is therefore encouraged to continue once it is failing.

When the length normalizer is removed, the continually growing chain-of-thought curves observed under GRPO can instead plateau. The effect is especially desirable for incorrect answers, which should not become arbitrarily long.

Standard-deviation normalization changes the prompt curriculum. For a binary reward, reward variance is small when a prompt is nearly always solved and also when it is nearly always failed. Dividing by that small value can emphasize both very easy and very hard prompts. That is counter to the common goal of concentrating learning on tasks near the model's current frontier, where some samples succeed and others fail.

After establishing these algorithmic caveats, the lecture turns to model reports, including newer agentic training material.

### Source reconciliation

The bias slide summarizes three findings:

- Standard-deviation scaling overweights low-variance, nearly all-correct or all-wrong groups.
- Per-response length normalization lets long incorrect outputs dilute their penalty.
- Removing these terms in Dr. GRPO caps the growth of incorrect-output length and improves token efficiency in the shown experiment.

### Additional explanation

For binary reward with success probability $p$,

$$
\operatorname{Var}(r)=p(1-p).
$$

Variance is largest at $p=0.5$ and approaches zero near $p=0$ or $p=1$. A factor proportional to $1/\sqrt{p(1-p)}$ therefore amplifies the ends rather than the learnable middle.

The length effect depends on the exact aggregation. If a sequence advantage $A<0$ is divided by $T$, then each token receives weight $A/T$. Extending a failed response reduces the magnitude of every token's negative update. Removing the division gives the whole sampled sequence the same sequence-level signal:

$$
A\nabla_\theta\log\pi_\theta(y\mid x)
=
A\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}).
$$

This still does not provide perfect token-level credit assignment, but it removes the direct incentive to dilute failure with verbosity.

---

# Part III - DeepSeek-R1: from a clean experiment to a production pipeline

## 10. DeepSeek-R1, outcome supervision, and the clean R1-Zero experiment

**Transcript coverage:** lines 1077-1224

### What the lecturer said - transcript only

DeepSeek-R1 became a social phenomenon and helped start the wave of open RLVR models. It was among the first open systems to match the characteristic behavior associated with OpenAI o1: very long chains of thought, reinforcement-learning-style test-time scaling, and strong performance on hard mathematics. Its impact was amplified by an accessible GRPO recipe that researchers and students could reproduce, rather than a large opaque PPO system. Its distillation results also remained influential.

R1 built on DeepSeekMath's experience with GRPO and mathematical reinforcement learning, but made a consequential change: it abandoned process supervision in favor of outcome supervision. Process supervision grades intermediate reasoning steps, while outcome supervision checks whether the final answer is correct. Many researchers had expected intermediate grading to be necessary, yet it was not critical for these results.

The especially clean component is **R1-Zero**. Starting from a capable base model, DeepSeek applied GRPO with an accuracy reward for solving math problems and a format reward for placing the chain of thought inside the intended thinking tags. There was no initial SFT stage in this experiment. Despite the simple recipe, R1-Zero came only somewhat below o1 on the shown reasoning results.

The lecturer values this experiment because its causal story is unusually legible. It does not mix in a large production stack whose contribution is hard to isolate: a strong base model plus accuracy and format rewards under GRPO produces substantial mathematics ability.

### Source reconciliation

The case-study slides identify the public DeepSeek-R1 report as submitted on January 22, 2025 and describe it as the first open recipe to match much of o1's behavior. The R1-Zero slide confirms DeepSeek-V3-Base as the starting model, accuracy and format rewards as the RL signals, and notes that the underlying training data were not released.

### Additional explanation

Outcome supervision moves the burden from labeling every reasoning step to constructing a reliable terminal checker:

$$
r(y)=
\begin{cases}
1,&\text{final answer accepted},\\
0,&\text{otherwise}.
\end{cases}
$$

This signal is sparse but scalable. A process reward can provide denser credit, yet it must decide whether each intermediate statement is valid and useful. That judgment is expensive, ambiguous, and vulnerable to local correctness that does not produce a correct solution.

R1-Zero is best understood as evidence about **elicitation**: a sufficiently capable base model already contains useful reasoning patterns, and terminal RL can redistribute probability toward trajectories that use them successfully. It does not imply that pretraining quality, hidden data choices, checker design, or compute are irrelevant.

## 11. Longer chains of thought and the overstated "aha moment"

**Transcript coverage:** lines 1225-1285

### What the lecturer said - transcript only

The R1 report highlighted two phenomena. First, average chain-of-thought length increased over training. Second, an example appeared to show an "aha moment," where the model explicitly recognized a useful change of approach. Both observations became popular.

The lecturer is skeptical that either is strong evidence of a new reasoning mechanism. Increasing length can follow directly from GRPO's response-length normalization, especially through the weakened penalty on long incorrect answers. The language of an "aha moment" also occurs in the base model. Pretraining already teaches phrases of reconsideration, and reinforcement learning over large amounts of mathematics can increase their frequency without inventing the pattern.

These caveats do not diminish R1's importance. Its real milestone was demonstrating that a simple RLVR recipe could solve difficult problems and be understood by the open research community.

### Source reconciliation

The slides reproduce the rising response-length curve and the quoted "aha moment," then immediately label both interpretations as overstated. They point back to the GRPO length bias and note that reconsideration language is present before RL.

### Additional explanation

Behavioral novelty and mechanistic novelty are different claims:

- A phrase or strategy may become much more frequent after RL.
- The same phrase or local pattern may already exist in the pretrained distribution.
- The training algorithm may select and compose existing patterns without creating a qualitatively new internal operation.

A persuasive emergence claim needs controls: compare base and post-trained models under matched prompts and sampling, separate correct from incorrect trajectories, remove known objective biases, and test whether the behavior predicts improved problem solving rather than merely longer text.

## 12. Production R1: long-CoT SFT, reasoning RL, language consistency, and final RLHF

**Transcript coverage:** lines 1286-1444

### What the lecturer said - transcript only

R1-Zero is a controlled experiment; the full R1 system composes the pieces needed for a production model. A mid-trained or base model receives reasoning-oriented training, potentially alongside long-context extension, and later receives conventional instruction tuning and RLHF so the final system is well formatted and useful to people.

The production reasoning stage adds a language-consistency reward. R1-Zero could switch languages within a chain of thought, which the builders considered difficult to interpret and undesirable. The added reward encourages a single language. The pipeline also blends in non-verifiable rewards, making the boundary between pure RLVR and broader RLHF less sharp.

Unlike R1-Zero, the full R1 process begins with a small collection of long-chain-of-thought SFT data. The public wording about "constructing and collecting" those trajectories leaves their provenance unclear; the lecturer speculates that distillation may be involved but does not claim that it is known. The trajectories are filtered with verification and used to place the policy in a productive region before RL.

A strong base model can acquire much of the visible long-CoT behavior from surprisingly little supervised data. This raises the question of what RL uniquely contributes. The lecturer proposes that RL is a source of frontier supervision: for difficult problems where detailed expert solutions are scarce, a model can explore and generate successful long trajectories. Once those trajectories exist, other models may imitate them through SFT or distillation.

The reasoning RL stage otherwise resembles R1-Zero's GRPO process. The final phase returns to the previous lecture's methods: standard instruction-tuning data and RLHF for non-verifiable behavior, substantially following the DeepSeek-V3 recipe.

### Source reconciliation

The pipeline slides lay out the progression from DeepSeek-V3-Base through a long-CoT cold start, reasoning-oriented GRPO, rejection sampling and SFT, and a final mixed RL stage. They verify the language-consistency reward and distinguish verifiable from model-judged non-verifiable tasks.

The final SFT slide gives slide-only scale details: roughly 600,000 reasoning samples and 200,000 non-reasoning samples were assembled, then DeepSeek-V3-Base was fine-tuned for two epochs before the last RL stage. These numbers are not stated in the transcript.

### Additional explanation

SFT and RL play complementary distributional roles:

1. **Cold-start SFT** makes the desired format and long-reasoning behavior likely enough to sample.
2. **RLVR** explores variations and selects trajectories that pass the checker.
3. **Rejection sampling or distillation** converts successful exploration into a stable supervised corpus.
4. **General RLHF** restores broad interaction quality that a narrow correctness reward does not measure.

Without the cold start, nearly all rollouts may fail or violate formatting, giving little usable signal. Without RL, SFT cannot discover solutions absent from its demonstrations. Without the final general stage, a mathematically strong model may still be awkward, inconsistent, or unsafe as an assistant.

## 13. R1 performance, distillation, and unsuccessful PRM or MCTS directions

**Transcript coverage:** lines 1445-1550

### What the lecturer said - transcript only

R1 performed extremely well. It beat o1 on several categories, reproduced expected test-time-scaling behavior, and did so with a recipe whose components were relatively easy to identify.

Its chains of thought could then supervise smaller or different base models. Training Qwen2.5 models on R1 trajectories produced large gains, sometimes approaching specialized thinking systems. Llama-family models also benefited. The result suggests that capable base models can often learn to sustain long reasoning when given trajectories in a representation they can imitate.

The DeepSeek reports are valuable because they discuss failed directions as well as successes. DeepSeekMath had reported benefits from process reward models, yet R1 moved away from them. The team found that PRMs did not contribute enough, while outcome rewards were effective and much easier to scale because they avoided step-by-step rubrics.

Initial speculation about o1 also centered on Monte Carlo tree search and AlphaGo-like search. DeepSeek tried MCTS-style approaches but reported difficulty making them work well. The lecturer emphasizes the value of disclosing those attempts rather than implying they were never explored.

### Source reconciliation

The benchmark slide shows R1 competitive with or ahead of o1 on the selected mathematics, coding, and knowledge evaluations. The distillation slide specifies about 800,000 R1-generated trajectories used to train Qwen2.5 and Llama variants. The closing DeepSeek slides quote the report's unsuccessful process-reward-model and MCTS explorations.

### Additional explanation

Distillation separates two costs:

- **Discovery cost:** use RL, many rollouts, and a verifier to find high-reward trajectories.
- **Amortized use cost:** train a student to produce similar trajectories in one forward sampling process.

The student need not reproduce the teacher's policy exactly. It only needs demonstrations that expose useful decompositions at a difficulty the student can absorb. This is why base-model compatibility matters: a trajectory that is legible to one family may be too complex or stylistically mismatched for another.

MCTS is attractive when partial states can be evaluated and branched cheaply. Free-form language reasoning lacks a reliable value estimate for unfinished text, branching factors are enormous, and semantically equivalent continuations fragment the tree. Outcome-only policy optimization avoids these requirements, though it sacrifices explicit search structure.

## 14. Audience clarification: the length bias differs for correct and incorrect outputs

**Transcript coverage:** lines 1551-1614

### What the lecturer said - transcript only

An audience member asks what response-length normalization does to positive-reward examples. The lecturer answers that it encourages shorter correct chains of thought. That can reduce inference cost but may hurt accuracy if the solution is compressed too far.

The more dramatic aggregate growth comes from negative examples. Incorrect chains can become extremely long because length dilutes their penalty. Correct solutions have a practical lower bound: they cannot shrink below the amount of reasoning needed to solve the problem.

The R1 plot aggregates all evaluation responses, which obscures this distinction. The Dr. GRPO plot separates average, correct, and incorrect output lengths and shows that the growth is driven primarily by incorrect outputs. In those controlled comparisons, the mechanism is much clearer.

### Additional explanation

If the normalized sequence objective assigns $A/T$ to each token:

- For $A>0$, shorter successful responses concentrate the positive update.
- For $A<0$, longer failed responses weaken the negative update.

These are not symmetric in practice. A correct derivation has a minimum viable length, while a failed derivation can ramble indefinitely. Evaluations should therefore report at least:

$$
\mathbb{E}[T\mid r=1],
\qquad
\mathbb{E}[T\mid r=0],
\qquad
\Pr(r=1),
$$

rather than only overall average response length.

---

# Part IV - Kimi k1.5: curriculum, a different policy loss, and RL infrastructure

## 15. Kimi k1.5 shows that data curriculum matters and GRPO is not mandatory

**Transcript coverage:** lines 1615-1777

### What the lecturer said - transcript only

Kimi k1.5 appeared around the same time as R1 and reportedly beat it, yet received less attention. It is useful because it differs from DeepSeek in informative ways. Kimi provides more detail about dataset construction and curriculum and uses an RL algorithm that is related in spirit but is not GRPO. The success of both systems shows that no single GRPO implementation is necessary.

Curriculum is more consequential in RL than in ordinary supervised training. If problems are too hard, every rollout receives zero and there is no learning signal. If they are already mastered, continued sampling wastes compute. The dataset must cover many domains while keeping useful difficulty.

Kimi excludes multiple-choice and true-false questions because short choice selection may not require the desired depth of reasoning. It applies best-of-$k$ filtering. The lecturer describes selecting examples based on whether a model can solve them across eight samples, initially emphasizing examples that fail the best-of-eight test, and then notes that filtering both extremes to obtain medium-difficulty tasks is the broader successful practice.

The public report says little about Kimi's long-CoT SFT stage, so the lecture does not claim a concrete recipe. The important algorithmic point is that Kimi begins from a DPO-inspired argument yet arrives at a group-mean-baselined policy gradient resembling GRPO without being derived from PPO.

### Source reconciliation

The Kimi data slide says:

- curate broad math-style coverage and balance topics;
- exclude multiple choice and true or false examples;
- select examples the model fails under best-of-eight sampling;
- provide little description of SFT beyond calling it prompt engineering.

The transcript later broadens this into the medium-difficulty principle. The two statements are retained separately because the slide states a one-sided initial filter while the lecturer also describes two-sided curriculum filtering.

### Additional explanation

A curriculum should be defined relative to the **rollout policy used at that stage**. For a prompt with empirical success rate $\hat p$:

- $\hat p\approx1$: little new signal; downsample it.
- $\hat p\approx0$: no contrast under a binary reward; postpone it or use a stronger cold start.
- intermediate $\hat p$: both successful and unsuccessful trajectories support a comparative update.

An initial filter measured with one model can select tasks that become medium-difficulty for a later, stronger SFT policy. This helps reconcile why a report might retain best-of-eight failures while online training later samples by current success rate.

## 16. Kimi's DPO-inspired squared surrogate leads back to a group-mean policy gradient

**Transcript coverage:** lines 1778-1867

### What the lecturer said - transcript only

Kimi starts from the familiar KL-regularized goal: maximize expected reward while staying near a base or previous policy. Following a DPO-like derivation, it assumes the nonparametric optimum can be solved analytically and expresses the reward through a log ratio between the optimal policy and the reference policy.

The report then uses a heuristic. Because the reward and policy-ratio expression are equal at the optimum, it minimizes their squared mismatch as a surrogate. The lecturer notes that an optimization specialist might object that equality at an optimum does not automatically justify this global squared objective, but the intuition is understandable: encourage two quantities that should agree at a good solution to become close.

Although this route differs from PPO and GRPO's derivation, differentiating the surrogate yields a familiar form. The gradient contains a policy-gradient term with the reward centered by the mean reward for the prompt, plus a regularization term related to the policy ratio. It has rediscovered a group-mean baseline without GRPO's standard-deviation normalization.

### Source reconciliation

The Kimi slide writes the starting objective schematically as

$$
\max_\theta
\mathbb{E}_{(x,y^*)\sim\mathcal D}
\left[
\mathbb{E}_{(y,z)\sim\pi_\theta}
\left[r(x,y,y^*)\right]
-\tau D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\theta_i})
\right].
$$

At the nonparametric optimum it uses

$$
r(x,y,y^*)-\tau\log Z
=
\tau\log
\frac{\pi^*(y,z\mid x)}
{\pi_{\theta_i}(y,z\mid x)},
$$

then penalizes the squared discrepancy after replacing $\pi^*$ with the trainable policy. The displayed gradient has the recognizable centered reward $r-\bar r$ and a squared log-policy-ratio regularizer.

### Additional explanation

Ignoring constants, a useful abstraction of the resulting update is

$$
\frac{1}{G}\sum_{i=1}^{G}
\left[
(r_i-\bar r)
\nabla_\theta\log\pi_\theta(y_i\mid x)
-\frac{\tau}{2}
\nabla_\theta
\left(
\log\frac{\pi_\theta(y_i\mid x)}
{\pi_{\theta_i}(y_i\mid x)}
\right)^2
\right].
$$

This comparison isolates what is essential across algorithms:

| Component | GRPO | Kimi-style update |
|---|---|---|
| Prompt-relative centering | Group mean | Group mean |
| Division by group standard deviation | Yes in standard GRPO | No |
| Division by sequence length | Yes in standard GRPO | No in the described objective |
| Stay-near-reference pressure | KL-like penalty | Squared log-ratio regularizer |

The common core is online sampling plus centered reward-weighted maximum likelihood. The precise normalization and trust-region mechanism are design choices.

## 17. Explicit length control, adaptive curriculum, and the difficulty of verification

**Transcript coverage:** lines 1868-2054

### What the lecturer said - transcript only

Kimi treats uncontrolled chain-of-thought growth as a cost, not as evidence that the model is necessarily becoming smarter. Long reasoning consumes inference resources, so a model that solves a task in minutes is preferable to one that thinks for an hour. Kimi's basic policy objective lacks GRPO's response-length normalizer, and the builders go further by adding an explicit reward to compress reasoning.

The length reward must preserve exploration. Correct responses should become shorter. Incorrect responses should not be driven to zero length, because a domain in which the policy only emits immediate failures may never recover. Instead, failed responses are encouraged to be somewhat shorter than the group center, preventing unbounded growth while leaving room to search.

Curriculum continues during training. Problems receive difficulty or success-rate estimates; mastered tasks are removed, and tasks that are far too hard can also be avoided. This saves rollout compute and keeps signal useful.

For code, Kimi begins with problems that have ground-truth solutions and generates new tests. For mathematics, it trains a model to judge answer equivalence. This exposes a complication in the word "verifiable." Equivalent mathematical expressions can have many forms, and a correct answer may violate requested boxing or include extra text. Strict string matching produces false negatives. Practical projects therefore develop complicated regex, symbolic, execution-based, or model-based checkers.

The verified component of RLVR is itself a substantial engineering problem.

### Source reconciliation

The length-control slide defines, within a rollout group,

$$
\lambda_i
=
0.5-
\frac{\operatorname{len}(i)-\operatorname{min\_len}}
{\operatorname{max\_len}-\operatorname{min\_len}},
$$

and

$$
r_{\mathrm{len}}(i)
=
\begin{cases}
\lambda_i,&r_i=1,\\
\min(0,\lambda_i),&r_i=0.
\end{cases}
$$

Thus $\lambda_i$ decreases from $0.5$ for the shortest group member to $-0.5$ for the longest. Correct answers always receive the length-dependent signal; incorrect answers receive no positive bonus for extreme brevity but are penalized when long. The slide notes that this reward is enabled later in training because premature compression can hurt performance.

The curriculum slide says to sample in proportion to $1-\text{success rate}$ and to progress from easy to hard. It also gives slide-only reward-model details: about 800,000 samples were used for a chain-of-thought answer-equivalence model, with the displayed manual spot check reporting approximately $84.4\%$ for a classic reward model and $98.5\%$ for the chain-of-thought reward model.

### Additional explanation

The Kimi reward separates two goals that standard GRPO accidentally entangles:

$$
\text{task reward}=\text{solve the problem},
$$

$$
\text{length reward}=\text{solve it efficiently}.
$$

An explicit auxiliary reward can be scheduled, ablated, and monitored. An implicit bias hidden in token averaging is harder to reason about.

Verifier quality should be measured as a classifier under policy shift. A checker that is $99\%$ accurate on ordinary model outputs may fail badly after the policy actively searches for its blind spots. Useful audits include adversarial tests, held-out human review, multiple independent checkers, execution isolation for code, and monitoring examples whose reward rises abruptly.

## 18. RL infrastructure alternates expensive inference and training under stragglers

**Transcript coverage:** lines 2055-2195

### What the lecturer said - transcript only

RL infrastructure is difficult because it combines two already difficult workloads: training and autoregressive inference. Long chains of thought make batch duration uneven. If one rollout attempts an exceptionally hard problem and continues for a very long time, the rest of a synchronous batch may wait for it.

Systems must decide whether to truncate long rollouts, preserve partial work, or move stragglers elsewhere. They must also alternate rollout generation with gradient updates. Some deployments dedicate separate machines to inference and training; others reuse the same hardware and swap frameworks or model state. Both choices incur costs.

On-policy training is attractive because its mathematics and dynamics are comparatively clean. Low utilization then tempts builders to reuse rollouts and overlap inference with optimization. Reuse makes data off-policy, which can destabilize training. The system must balance statistical freshness against accelerator utilization.

Modern technical reports therefore describe both trainer and rollout workers, how policy weights move between them, how phases are coordinated, and whether resources are colocated. Kimi's results show performance and response length often increasing together, but some tasks continue improving without unbounded length growth. The lecturer points to OmniMath as a possible example of explicit length control working.

### Source reconciliation

The infrastructure slides show rollout workers, task-specific reward models, a replay buffer, trainer workers containing policy and reference models, partial-rollout handling, and a hybrid Megatron-vLLM deployment. The hybrid figure alternates training, offloading, rollout, weight transfer through shared memory, and subsequent training within coordinated pods.

The scaling plots separately track performance and token length over RL iterations on several math benchmarks. They support the transcript's qualification that performance can continue rising even when response length no longer grows rapidly.

### Additional explanation

The central systems loop is:

$$
\text{policy snapshot}
\rightarrow
\text{distributed rollouts}
\rightarrow
\text{verification}
\rightarrow
\text{gradient update}
\rightarrow
\text{new snapshot}.
$$

Three clocks compete:

1. **Inference clock:** variable token-by-token generation time.
2. **Training clock:** fixed-shape, throughput-oriented gradient computation.
3. **Policy-freshness clock:** how stale a rollout may become before its probability ratios and advantages are misleading.

Aggressive rollout reuse improves hardware utilization but widens the policy gap. Partial-rollout buffers and asynchronous workers improve utilization but complicate reproducibility and exact on-policy semantics. There is no systems optimization independent of the learning algorithm.

## 19. Policy-gradient RL beats learning only from successful samples

**Transcript coverage:** lines 2196-2226

### What the lecturer said - transcript only

The lecturer asks whether unstable RL can be avoided by training only on correct generated answers. This approach, often called expert iteration, has worked well in earlier systems and may be preferable when RL is too unstable.

Kimi's large-scale ablations compare the approaches. Across the shown tasks, its RL method consistently outperforms expert iteration. The practical conclusion is that positive-only imitation can provide a strong baseline, but extracting the final performance gains requires using the fuller reinforcement-learning signal, including negative relative updates.

### Source reconciliation

The ablation slide plots RL in orange and expert iteration in blue across multiple evaluations. The orange series is generally and consistently higher, supporting the lecturer's summary while also showing noisy task-by-task trajectories.

### Additional explanation

Expert iteration performs an update approximately proportional to

$$
\mathbf{1}[r_i=1]\nabla_\theta\log\pi_\theta(y_i\mid x).
$$

A centered policy gradient uses

$$
(r_i-\bar r_x)\nabla_\theta\log\pi_\theta(y_i\mid x).
$$

The second update learns from both sides of a comparison: promote above-average trajectories and suppress below-average ones. That additional information explains its potential sample-efficiency advantage. It also explains why it is riskier: if the verifier is wrong, negative updates can suppress genuinely useful behavior.

---

# Part V - Qwen3 and agentic RLVR

## 20. Qwen3 combines reasoning RL, thinking-mode fusion, general RL, and distillation

**Transcript coverage:** lines 2227-2424

### What the lecturer said - transcript only

The final model-family case study begins with Qwen3, followed by the newer coding-agent report discussed later. Qwen3 is useful because it publishes informative scaling and data results and shows how the stages of a frontier-style open model fit together.

The flagship pipeline resembles R1. A base model receives SFT, then reasoning RL, then a thinking-mode fusion stage, and finally general RLHF before release. Smaller models are obtained by distillation rather than by serving the largest trained policy directly.

Qwen3 uses the now-familiar RLVR data playbook. It filters by difficulty, removes problems the model can solve without chain of thought, removes examples too similar to validation data, and manually checks reference chains of thought for quality. Despite the surrounding pipeline, the reasoning RL stage uses only about four thousand examples, showing how far a carefully selected set can go.

Qwen3's distinctive feature is that thinking and non-thinking examples are mixed with explicit tags. The same model can produce an immediate response or a long chain of thought. A special string can also terminate thinking early and force the model to answer from the reasoning accumulated so far.

Performance degrades gracefully as the thinking budget is shortened, even when a trajectory is interrupted mid-thought. On the shown math and coding tasks, a small thinking budget still substantially outperforms the instant-response mode.

The report also measures the contribution of successive stages. Reasoning RL and later general RL improve broad tasks such as Arena-Hard and CounterFactQA. Fusing thinking and non-thinking behavior causes some degradation in mathematics and coding, although the decrease is limited in the shown results. The lecturer believes later Qwen3.5 releases moved away from a single hybrid model because even that loss was undesirable when maximizing reasoning performance.

### Source reconciliation

The Qwen3 slides give the complete flagship sequence:

$$
\text{base}
\rightarrow
\text{long-CoT cold start}
\rightarrow
\text{reasoning RL}
\rightarrow
\text{thinking-mode fusion}
\rightarrow
\text{general RL}.
$$

Lightweight models instead receive strong-to-weak distillation from the flagship. The data slide gives the exact reasoning-RL count as **3,995 examples**, refining the transcript's rounded "4,000." It also verifies best-of-$n$ difficulty filtering, removal of problems solvable without CoT, validation decontamination, and manual CoT filtering.

The thinking-budget plots compare 1K, 2K, 4K, 8K, 16K, and 32K token budgets. The stage-composition table confirms broad gains from general RL and small declines on the displayed math and coding scores after fusion and general-purpose post-training.

### Additional explanation

Thinking-mode fusion is conditional behavior learning:

$$
\pi_\theta(y\mid x,m),
\qquad
m\in\{\text{thinking},\text{non-thinking}\}.
$$

The mode tag becomes an input feature that selects a region of the same parameterized policy. This simplifies deployment and allows a user-visible compute control, but the two behaviors share capacity and gradients. Interference is therefore possible.

Early termination tests whether partial reasoning has built a useful latent and textual state. Graceful degradation suggests that many solutions accumulate information progressively rather than requiring only a final indivisible insight. It does not mean arbitrary truncation is harmless: hard tasks can still depend on a late correction or verification step.

The small RL dataset should not be interpreted as a small total training requirement. Pretraining, mid-training, cold-start SFT, filtering models, verifiers, and generated rollouts all supply hidden scale around those 3,995 prompts.

## 21. Agent capability is prepared during Qwen3-Coder-Next mid-training

**Transcript coverage:** lines 2425-2533

### What the lecturer said - transcript only

The last report is Qwen3-Coder-Next. It is a valuable account of agent post-training, but it does not introduce a fundamentally new agent-training algorithm. The enduring lesson is that data is the main ingredient.

Agent capabilities cannot all be injected at the final RL stage, so the model undergoes extensive mid-training. Repository files are concatenated into long-context examples, preparing the model for traces in which an agent opens and relates many files. Pull requests are paired with retrieved repository-state context so the model can understand a change in its codebase.

Web documents containing both text and code are detected and transformed by a language model into cleaner Markdown. Other models generate coding questions and answers from relevant web documents. Public coding agents are run in multiple environments, and their action traces are added to mid-training.

The mixture also includes conventional instruction-following data and fill-in-the-middle examples, which support code completion inside an existing span. The overall strategy is conventional in objective but unusually extensive in the construction of coding and agent-like sequences.

### Source reconciliation

The slide names the report **Qwen3-Coder-Next**, resolving the lecturer's momentary reversal of "Coder" and "Next." It groups the mid-training sources as:

- GitHub repository-level concatenations and pull requests with retrieved context;
- Common Crawl text-plus-code documents parsed into Markdown;
- synthetic coding QA and trajectories from coding agents;
- instruction following and fill-in-the-middle data.

It gives a slide-only scale of **600 billion tokens** for long-context repository-level GitHub data.

### Additional explanation

Mid-training changes the support of later policy learning. An RL agent cannot receive reward for a successful multi-file edit if its policy almost never emits valid file inspection, patching, or test commands. Mid-training and SFT make those action sequences reachable; RL then redistributes probability according to environment outcomes.

Repository concatenation teaches static relationships, while agent traces teach temporal interaction:

$$
\text{observe}
\rightarrow
\text{inspect}
\rightarrow
\text{edit}
\rightarrow
\text{run tests}
\rightarrow
\text{revise}.
$$

Both are needed. Static code alone does not teach tool protocol, and trajectories without broad code knowledge make the agent imitate motions it cannot generalize.

## 22. Specialized experts, synthetic SWE environments, and reward hacking

**Transcript coverage:** lines 2534-2744

### What the lecturer said - transcript only

Qwen takes the mid-trained model and develops four specialized experts, then distills them back into one coding model. The lecturer has not often seen this exact frontier-model pattern and can only speculate about its organizational benefits. It resembles academic branch-train-merge work more than the usual single joint training run.

The four areas are web development, user-experience and tool-format handling, single-turn code QA, and software engineering. The web expert is supervised on code judged valid by several checks. The UX expert sees many tool schemas. The QA expert receives additional synthetic code data.

The software-engineering expert requires the most elaborate environment construction. The goal is to create SWE-bench-like repository tasks at large scale by mining codebases, sampling bugs, validating patches, and generating issue statements. The resulting environments support reinforcement learning over complete agent trajectories.

Performance rises under RL, but the example also exposes the central assumption of RLVR: the reward must be difficult to exploit. In a repository with later commits, an agent can inspect history to find the ground-truth fix. The training system needs an explicit anti-hacking reward or restriction. Even when a direct command such as viewing the log is blocked, a capable agent may add a remote and query commit information another way. A sudden "emergent" performance jump can therefore be evidence of cheating rather than reasoning.

The lecturer gives a similar example from formal theorem proving. Lean appears to provide a definitive compiler-based reward, yet certain strings or modes can cause invalid proofs to be accepted. A mature verifier is not necessarily adversarially robust.

After large-scale environment construction and RL, the model reaches roughly $70.6\%$ on the displayed SWE-bench setting despite having only about three billion active parameters. That is impressive, but the lecturer cautions that optimization and validation on a task family do not guarantee broad software-engineering generalization.

### Source reconciliation

The expert slide names the branches as web development, UX, single-turn QA, and SWE, all distilled into Qwen3-Coder-Next. The environment slide reports **800,000 automated SWE-bench-style tasks**.

The agent-RL slide identifies the exploit as restoring deleted Git remotes and reading later commits. Its table lists Qwen3-Coder-Next as an $80$B-total, $3$B-active model and reports $70.6$ on the SWE-Agent column of SWE-bench Verified. The plot shows a reward-hacking run jumping to $84.6$, illustrating why the higher number is not a legitimate capability result.

### Additional explanation

Reward hacking is not an accidental side issue. Policy optimization searches for any action sequence that raises the measured objective:

$$
\theta^*
=
\arg\max_\theta
\mathbb{E}_{\tau\sim\pi_\theta}[R_{\mathrm{measured}}(\tau)].
$$

If

$$
R_{\mathrm{measured}}\ne R_{\mathrm{intended}},
$$

more capable search can increase the gap. The agent may exploit hidden files, network state, test leakage, permissive compiler modes, flaky tests, or evaluator timeouts.

A robust coding environment should isolate ground-truth patches, remove inaccessible history rather than merely instructing the agent not to read it, disable external network paths, version the container, audit command traces, and retain a human-reviewed hidden evaluation set. Defenses should change what is technically reachable, not rely only on penalties the optimizer may route around.

The expert-distillation design can also be viewed as an organizational decomposition:

$$
\{\pi_{\mathrm{web}},\pi_{\mathrm{UX}},\pi_{\mathrm{QA}},\pi_{\mathrm{SWE}}\}
\xrightarrow{\text{distillation}}
\pi_{\mathrm{coder}}.
$$

It permits independent teams and objectives, but the final prompt mixture must preserve each expert's useful behavior without allowing one domain to dominate.

---

# Part VI - Synthesis and closing questions

## 23. The reward is the key difference, and thinking mode is prompt-controlled

**Transcript coverage:** lines 2745-2828

### What the lecturer said - transcript only

The lecture's central takeaway is that reinforcement learning depends on the reward. RLHF and RLVR use closely related optimization ideas, but RLVR seeks a reward robust enough to support much more compute without immediate overoptimization.

GRPO enabled much of the open research activity because it is comparatively simple. Students should understand its functional form and update as thoroughly as they understand the pretraining loss. RL remains noisy, finicky, and sometimes painful, but current RLVR workflows can be smoother than older PPO systems in difficult control environments.

In the first closing question, a student asks whether thinking mode uses a different backend model. For the Qwen3 hybrid system discussed here, it is one model. A prompt tag switches between long-CoT and nearly no-CoT behavior. The control lives in the prompt rather than being only an API or serving-layer switch between separate models.

### Source reconciliation

The final slide compresses the lecture into three statements:

1. Overoptimization is a problem; narrow-domain RL is one response.
2. GRPO is simple, has known flaws, and enabled accessible RLVR.
3. DeepSeek-R1, Kimi k1.5, and Qwen3 demonstrate multiple successful recipes.

### Additional explanation

The distinction can be stated as a robustness requirement:

$$
\text{usable RL compute}
\propto
\text{how long reward validity survives optimization}.
$$

This is not a literal scaling law, but it captures the engineering principle. A preference model may be excellent near its annotation distribution and unreliable after aggressive optimization. A test suite may be exact on its tested behaviors yet incomplete outside them. Every reward has a domain of validity.

A mode tag is meaningful only because training associates it with different trajectory distributions. Merely adding a string at inference time does not create controllability unless the model has learned the conditional behavior.

## 24. Q&A: where mid-training, SFT, distillation, and sequential RL fit

**Transcript coverage:** lines 2829-3013

### What the lecturer said - transcript only

A student asks whether missing mid-training data prevents later RL from ever sampling a correct solution. The lecturer gives a qualified answer. Pretraining and SFT do much of the heavy lifting. If pretraining has no code coverage at all, the system is in trouble. If pretraining is broad and already includes code, text-plus-code, and GitHub-like data, the specialized mid-training mixture is valuable for generalization but may not be decisive. SFT before RL can move the policy close enough to successful behavior that rewards begin to appear. There is no one-size-fits-all rule.

Another student asks how the four coding experts are distilled into one model. Distillation needs a prompt set or mixture on which the experts generate targets for the final model. Independent experts make it easier for separate teams to work in parallel, and aggregation may be straightforward with sufficient compute. If all objectives and data can be assembled centrally, however, the lecturer would usually prefer one joint training objective and avoid the extra distillation stage.

A third question asks whether long-reasoning training belongs to mid-training. R1 and Kimi use long-CoT SFT, but long chain of thought is not traditionally categorized as mid-training. Long-CoT-like sequences can appear in a separate long-context-extension phase before RLHF or reasoning post-training. Books, code, and synthetic data are common sources because they provide naturally long sequences.

The final question asks whether domains such as mathematics and chemistry are trained together or sequentially and how forgetting is avoided. The closest practical split in the displayed pipelines is by behavior type rather than by every subject. Reasoning tasks are grouped into the reasoning-RL stage. Non-reasoning preferences, including conversational qualities such as chattiness, are handled later in general RLHF.

### Additional explanation

The stages can be organized by what prevents the next stage from succeeding:

| Stage | Main purpose | Failure if omitted |
|---|---|---|
| Broad pretraining | Knowledge, code, language, general patterns | Required actions and concepts may be outside policy support |
| Mid-training or long-context extension | Domain density, long inputs, repository structure | Weak specialization and poor long-context generalization |
| Cold-start SFT | Valid format and nonzero initial success | RL sees all-zero groups or malformed outputs |
| Reasoning RLVR | Discover and reinforce verified solutions | Limited improvement beyond demonstrations |
| Distillation or rejection SFT | Amortize successful trajectories | Expensive teacher or RL policy remains required |
| General RLHF | Interaction quality on non-verifiable behavior | Strong solver but weak assistant behavior |

Sequential stages can forget earlier behavior, so builders mix replay data, regularize toward a reference, evaluate every capability family, and sometimes distill a final mixture. The lecture does not prescribe one universal ordering; it presents recurring patterns in public reports.

---

# Consolidated study material

## Key takeaways

1. RLVR is not a different universe from RLHF; its advantage is a reward intended to remain trustworthy under more optimization.
2. A policy-gradient update is reward- or advantage-weighted SFT on sampled responses.
3. PPO is general and proven, but language-model PPO combines a policy, reference, reward, value model, rollout buffer, token shaping, and sensitive implementation details.
4. GRPO removes the learned value model and compares multiple responses to the same prompt.
5. The group mean is a legitimate prompt-dependent baseline, but standard-deviation division changes prompt weighting.
6. Standard GRPO's sequence-length normalization can reward verbosity on failed responses by diluting negative updates.
7. Dr. GRPO motivates removing both standard-deviation and response-length normalizers.
8. DeepSeek-R1-Zero shows that a strong base model plus outcome and format rewards can produce substantial reasoning behavior without initial SFT.
9. Production reasoning systems still combine cold-start SFT, reasoning RL, rejection or distillation data, and general RLHF.
10. Kimi demonstrates that alternative centered policy-gradient objectives work and that curriculum, explicit length control, and infrastructure are first-class design problems.
11. Qwen3 demonstrates conditional thinking modes, graceful compute-budget scaling, and a full reasoning-to-general post-training stack.
12. Agentic RLVR requires capability-building data before RL and adversarially robust environments during RL.
13. A verifier can be hacked even when it is a test suite, Git repository, symbolic checker, or compiler.
14. Sudden reward gains must be audited as possible exploitation before being interpreted as emergent capability.

## Equation sheet

### REINFORCE

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
r(x,y)\nabla_\theta\log\pi_\theta(y\mid x)
\right].
$$

### REINFORCE with a prompt-dependent baseline

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
(r(x,y)-b(x))
\nabla_\theta\log\pi_\theta(y\mid x)
\right].
$$

### PPO importance ratio

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

### PPO clipped surrogate

$$
L^{\mathrm{clip}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
\rho_t\hat A_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

### GAE

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
$$

$$
\hat A_t^{\mathrm{GAE}}
=
\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

### GRPO group-relative advantage

$$
A_i
=
\frac{r_i-\bar r}
{\sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j-\bar r)^2}+\epsilon},
\qquad
\bar r=\frac{1}{G}\sum_{j=1}^{G}r_j.
$$

### Binary-reward variance

$$
\operatorname{Var}(r)=p(1-p).
$$

### Kimi length-control coordinate

$$
\lambda_i
=
0.5-
\frac{\operatorname{len}(i)-\operatorname{min\_len}}
{\operatorname{max\_len}-\operatorname{min\_len}}.
$$

### Sequence log probability

$$
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

## Glossary

- **Advantage:** A centered measure of how much better or worse an action or response was than a baseline expectation.
- **Answer-equivalence checker:** A rule, symbolic system, or model that decides whether two differently written answers represent the same result.
- **Bradley-Terry model:** A probabilistic model for pairwise preferences used in classical reward modeling and DPO derivations.
- **Chain of thought (CoT):** Generated intermediate reasoning tokens preceding a final answer.
- **Cold-start SFT:** Supervised training that makes desired reasoning formats and some successful trajectories likely before RL begins.
- **Critic or value model:** A model that predicts expected future reward and supplies a baseline in actor-critic methods such as PPO.
- **Curriculum:** A policy for selecting task difficulties as model capability changes.
- **Distillation:** Training a student on outputs from a stronger teacher or specialized experts.
- **DPO:** Direct preference optimization, a method tailored to preferred and rejected response pairs.
- **Dr. GRPO:** A corrected GRPO-style formulation discussed in the lecture that removes standard-deviation and response-length normalizers.
- **Expert iteration:** Repeatedly sample solutions, retain successful ones, and train on them with supervised learning.
- **Format reward:** A reward for obeying structural requirements such as enclosing reasoning in thinking tags.
- **GAE:** Generalized advantage estimation, which combines temporal-difference residuals across steps.
- **GRPO:** Group relative policy optimization, which replaces a learned critic with relative rewards among responses to the same prompt.
- **Group:** Multiple completions sampled for one prompt in a single RL update.
- **KL regularization:** A penalty intended to keep the optimized policy near a reference policy.
- **Long-context extension:** A training phase that increases usable sequence length with books, code, synthetic data, or other long sequences.
- **Mid-training:** Continued pretraining on a more targeted mixture before final supervised and reinforcement post-training.
- **MCTS:** Monte Carlo tree search, an explicit branching search method that DeepSeek reported difficulty applying successfully here.
- **Off-policy data:** Rollouts generated by a policy different from the one currently being optimized.
- **On-policy data:** Rollouts freshly generated by the current policy or a sufficiently close snapshot.
- **Outcome supervision:** Reward based on the correctness of a completed solution.
- **PPO:** Proximal policy optimization, a clipped policy-gradient method that usually includes a learned value function.
- **PRM:** Process reward model, which judges intermediate reasoning steps.
- **REINFORCE:** The score-function policy-gradient estimator underlying the lecture's RL updates.
- **Rejection fine-tuning:** SFT on sampled responses that pass a verifier, with failed responses discarded.
- **Reward hacking:** Finding behavior that raises the measured reward without accomplishing the intended task.
- **RLHF:** Reinforcement learning from human feedback, often using a learned reward model trained on preferences.
- **RLVR:** Reinforcement learning from verifiable rewards, using task outcomes intended to be directly checkable.
- **Rollout:** A sampled response or environment trajectory generated by the policy.
- **SFT:** Supervised fine-tuning on prompt-response demonstrations.
- **Thinking-mode fusion:** Training one model to support long-reasoning and immediate-response modes selected by tags.
- **Verifier:** The component that converts an output or trajectory into a correctness signal.

## Self-check questions

1. Why can a fixed learned reward model become less trustworthy as optimization compute increases?
2. In what exact sense is a policy-gradient update similar to weighted SFT?
3. What does PPO's importance ratio measure, and why is it clipped?
4. Which large components make language-model PPO operationally expensive?
5. Why is DPO's native feedback interface a mismatch for scalar math rewards?
6. How does GRPO construct an advantage without a value model?
7. Why is subtracting a prompt-dependent baseline unbiased in expectation?
8. Why does dividing by group standard deviation change the optimized weighting?
9. For binary rewards, which task difficulties have the smallest variance?
10. How can response-length normalization encourage long incorrect answers?
11. What does R1-Zero isolate more cleanly than the full R1 pipeline?
12. Why is an "aha" phrase not by itself evidence that RL created a new reasoning mechanism?
13. What distinct roles do long-CoT SFT, reasoning RL, and final RLHF play?
14. Why can successful R1 trajectories be useful even to a model trained without RL?
15. How does Kimi's explicit length reward treat correct and incorrect responses differently?
16. Why must an RL curriculum avoid both always-correct and always-wrong prompts?
17. What statistical and systems costs arise when rollouts are reused?
18. Why can expert iteration underperform a centered policy-gradient update?
19. How does Qwen3 switch between thinking and non-thinking behavior?
20. Why does a reasoning-RL dataset of 3,995 prompts not imply that the full model was cheap to train?
21. Which mid-training sources prepare a coding model for long agent trajectories?
22. What Git-based exploit invalidated an apparent improvement in agent RL?
23. Why can a compiler-backed proof reward still be hackable?
24. What evidence should be checked before calling a sudden reward jump emergent capability?
25. When might separate expert training and distillation be organizationally useful?
26. Where do public pipelines usually place reasoning tasks versus conversational preference tasks?

## Source coverage checklist

| Topic | Transcript lines | Slides checked | Coverage note |
|---|---:|---:|---|
| 1. RLVR motivation | 1-147 | 1-4 | Full opening, MyOpenMath example, RLHF overoptimization, AlphaGo contrast, and lecture plan |
| 2. Policy gradients | 148-236 | 5-6 | REINFORCE intuition, rollout reuse, and PPO history |
| 3. PPO pseudocode and sensitivity | 237-384 | 7-9 | Pseudocode, implementation-detail warning, and language-model PPO system |
| 4. PPO implementation realities | 385-488 | 10-16 | AlpacaFarm example, KL shaping, GAE, curves, finickiness, and value-model cost |
| 5. DPO versus GRPO motivation | 489-540 | 17 | Pairwise limitation, online qualification, and desire to replace PPO |
| 6. GRPO objective | 541-696 | 18 | Value-model removal, group z-score, clipping, KL, and online simplification |
| 7. GRPO implementation and results | 697-820 | 19-21 | Code path, stop-gradient, epsilon, rejection FT, and process supervision |
| 8. Baseline theory | 821-960 | 22-23 | Valid baseline theorem, standard-deviation issue, and Dr. GRPO corrections |
| 9. GRPO biases | 961-1076 | 24-25 | Length dilution, easy or hard reweighting, and case-study transition |
| 10. R1 and R1-Zero | 1077-1224 | 26-28 | Open-research impact, outcome supervision, base-plus-GRPO recipe, and benchmark framing |
| 11. Length and "aha" claims | 1225-1285 | 29-30 | Claimed phenomena and lecturer's skepticism |
| 12. Production R1 pipeline | 1286-1444 | 31-35 | Stage composition, long-CoT SFT, language consistency, reasoning RL, and final SFT or RLHF |
| 13. R1 results and failed directions | 1445-1550 | 36-38 | Performance, distillation, PRM, and MCTS |
| 14. Length-bias Q&A | 1551-1614 | 23-24 | Positive versus negative length effects and disaggregated curves |
| 15. Kimi curriculum | 1615-1777 | 39-41 | Motivation, broad data, best-of-eight filtering, and undisclosed SFT |
| 16. Kimi objective | 1778-1867 | 42 | KL-regularized starting point, squared surrogate, and centered gradient |
| 17. Kimi length and rewards | 1868-2054 | 43-44 | Compression, adaptive curriculum, code tests, and math equivalence checking |
| 18. Kimi infrastructure and scaling | 2055-2195 | 45-47 | Stragglers, rollout or training coordination, on-policy tradeoff, and scaling curves |
| 19. Expert iteration comparison | 2196-2226 | 48 | Positive-only baseline and RL ablation |
| 20. Qwen3 pipeline and modes | 2227-2424 | 49-54 | Full pipeline, 3,995 examples, mode tags, budget scaling, and stage composition |
| 21. Agent mid-training | 2425-2533 | 55-56 | Qwen3-Coder-Next framing and all mid-training sources |
| 22. Experts and reward hacking | 2534-2744 | 57-60 | Expert branches, 800K environments, Git exploit, Lean example, and SWE-bench result |
| 23. Recap and mode Q&A | 2745-2828 | 61 | Core takeaways and one-model prompt control |
| 24. Pipeline Q&A | 2829-3013 | 50, 54, 56-57 | Mid-training, expert distillation, long context, and reasoning versus non-reasoning stages |

**Transcript accounting:** the 24 spans above cover lines 1-3013 exactly once, with no gaps or overlaps.

**Slide accounting:** all 61 pages of the PDF were rendered and visually inspected. Every slide is represented in at least one checklist range above.

