---
title: "Lecture 11 - Advanced Scaling Laws"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 11
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 11  Scaling Laws.txt"
slide_deck: "../lecture_11.pdf"
status: "complete"
---

# Lecture 11: Advanced Scaling Laws

## How to read these notes

Every substantive topic has two deliberately separate layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves claims, examples, qualifications, numerical details, warnings, and substantive questions while removing filler and repetition.
2. **Additional explanation.** This adds independent intuition, derivations, connections, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are included for auditability. The full 3,233-line transcript was mapped before the 58-slide deck was rendered and inspected page by page. Slides were then used to verify structure, names, equations, fitted constants, plotted comparisons, and algorithms. Material that is visible on a slide but not stated in the transcript is kept in a separate **Source reconciliation** block.

## Lecture map

The lecture develops two complementary responses to scale-sensitive training choices:

1. Study recent public scaling recipes, especially MiniCPM and DeepSeek, to see how scaling laws are used in real model releases.
2. Fit explicit laws for sensitive hyperparameters such as learning rate and batch size, as DeepSeek and StepFun do.
3. Examine whether optimizer improvements survive changes in compute and the token-to-parameter ratio, using Muon as the main case study.
4. Reparameterize a network with maximum update parameterization (muP) so that useful activation and update scales, and ideally optimal hyperparameters, remain stable as width changes.

---

# Part I - Scaling recipes used in public model development

## 1. Why advanced scaling is a practical problem

**Transcript coverage:** lines 1-150

### What the lecturer said - transcript only

The lecture continues beyond the classical scaling-law canon of Hestness, Kaplan, and Chinchilla. Its subject is not merely whether loss follows a power law, but how one actually scales an open language model: whether Chinchilla-style analysis still works at realistic model sizes, which optimization details matter, and how scale should affect initialization, learning rate, and batch size.

The public scaling literature changed after roughly 2022. Fewer frontier builders now publish detailed scaling studies, and many of the useful public reports have come from the Chinese open-source ecosystem. The first part of the lecture therefore curates several of these reports, ranging from early MiniCPM and DeepSeek work to the more recent Kimi K2 release, to show what open-frontier builders measure and worry about.

The second part turns to optimizers and initialization. Batch size, learning rate, and optimizer behavior are scale-sensitive, so experiments at one size do not automatically specify the right settings at another. Understanding how these choices change can also help in ordinary, smaller language-model experiments when scaling a run either up or down.

MiniCPM and the original DeepSeek LLM paper receive the most attention. They are not simply the newest papers, but serious studies from groups that built competitive models. More importantly, they represent different approaches to sensitive hyperparameters: MiniCPM tries to make them invariant through parameterization, while DeepSeek measures their drift and fits it with scaling laws.

### Additional explanation

Classical scaling laws answer a resource-allocation question: given compute $C$, how should model size $N$ and data size $D$ change? This lecture adds a second layer:

$$
(C, N, D) \longrightarrow (\text{batch size},\ \text{learning rate},\ \text{schedule},\ \text{initialization},\ \text{optimizer}).
$$

That second mapping is operationally crucial. A predicted model/data allocation is not useful if the large run diverges or is trained with a badly scaled optimizer. There are two broad strategies:

- **Predict the drift:** run small sweeps and fit how the optimum moves with scale.
- **Remove the drift:** change parameterization so one base hyperparameter remains appropriate across widths.

Neither strategy eliminates empirical risk. A fitted law can extrapolate poorly, while an invariance argument can omit a real architectural or optimizer interaction.

## 2. MiniCPM: a scaling ladder built around muP

**Transcript coverage:** lines 151-323

### What the lecturer said - transcript only

MiniCPM was a 2024 effort to build a high-performance language model in roughly the 1B-to-2B-parameter range. It was competitive for that size bracket by 2024 standards, although later small models such as Gemma became stronger. The paper remains useful because it documents details that later reports increasingly take for granted.

Its first important technique is **maximum update parameterization**, or muP. The objective is to keep the optimal learning rate approximately fixed as the model is made wider. MiniCPM changes several related quantities: it scales the embedding output, scales residual contributions with the square root of depth, changes the initialization of matrix-shaped tensors using the fan-in/fan-out relationship, assigns scale-dependent learning rates to tensors, and rescales the language-model head. Per-parameter learning-rate scaling may look unusual, but it is central to the construction.

MiniCPM did not brute-force hundreds of target-size models. It trained a ladder of much smaller models, with about a fivefold gap between the largest ladder model and the final released model, and attempted to determine the sensitive choices before the expensive run. The main targets were optimal learning rate, optimal batch size, and the token-to-model-size allocation associated with Chinchilla. The Chinchilla replication is less novel, but its learning-rate schedule makes the experiment reusable in an important way discussed later.

### Source reconciliation

The slides make the MiniCPM implementation concrete. For width ratio $d_m/d_{base}$, the displayed configuration uses `scale_emb = 12`, `scale_depth = 1.4`, `init_std = 0.1`, and an overall learning rate of `0.01`. It applies the following operations:

- multiply the embedding output by `scale_emb`;
- multiply each residual increment by $\text{scale\_depth}/\sqrt{\text{num\_layers}}$;
- initialize a two-dimensional tensor with standard deviation $\text{init\_std}/\sqrt{d_m/d_{base}}$, while the other parameters use standard deviation $0.1$;
- multiply the learning rate of each two-dimensional tensor by $1/(d_m/d_{base})$, while other tensors use the overall learning rate;
- multiply the language-model-head logits by $1/(d_m/d_{base})$.

The slide's scaling ladder runs from roughly 9M parameters through 0.5B parameters, while the final target is about five times larger. The ladder keeps a fixed architectural aspect ratio while sweeping batch size, learning rate, and the token-to-model-size allocation.

### Additional explanation

The changes form a coupled parameterization, not a bag of independent tricks. If initialization changes the magnitude of a matrix but its optimizer step is not adjusted, the relative update $\|\Delta W\|/\|W\|$ may still drift with width. Likewise, rescaling the hidden state without compensating at the output can change the logit scale. A muP recipe must therefore specify initialization, multipliers, and optimizer learning rates together.

A **scaling ladder** is an experimental bridge rather than a miniature version of the final run. Each rung should preserve the architectural family and training conditions closely enough that observed optima transfer. The gap between the last rung and target is the extrapolation risk the study accepts in exchange for avoiding target-scale sweeps.

## 3. MiniCPM's stable optimal learning rate

**Transcript coverage:** lines 324-359

### What the lecturer said - transcript only

The point of MiniCPM's nonstandard initialization is to stabilize the optimal learning rate. In its experiments, this works unusually cleanly. When learning rate is swept for several model sizes, almost every minimum lies near $10^{-2}$; the smallest model shifts slightly, but remains close. This is a successful instance of muP rather than proof that muP is the only possible solution.

If a parameterization produces this behavior, learning rate no longer needs to be retuned merely because width changes. That removes one expensive dimension from the scaling ladder.

### Additional explanation

The relevant notion of transfer is not that every loss curve becomes identical. Larger models can have lower attainable losses, and curvature around the optimum can differ. What should remain stable is the **base-learning-rate coordinate of the optimum** under the prescribed per-tensor multipliers.

This distinction matters when reading a sweep. A broad valley also makes transfer practically forgiving: even if the exact minimizer shifts a little, a fixed value can remain near-optimal. A narrow valley would make a visually small shift much more consequential.

## 4. Critical batch size still drifts with the target loss

**Transcript coverage:** lines 360-425

### What the lecturer said - transcript only

Stabilizing learning rate does not make optimal batch size constant. MiniCPM finds that batch size depends on the amount of data processed and the model scale. For a fixed batch size, a vertical column of points traces a training run as more tokens are processed. By fitting equal-loss curves across many such runs, the authors estimate the batch size that best reaches each target loss.

The result resembles Kaplan's critical-batch-size analysis. Lower target loss calls for a larger batch, and the relationship has a clean power-law form. If a separate loss scaling law predicts the loss that the intended run will reach, the corresponding fit can be used to select a batch size with a favorable tradeoff between optimization steps and examples processed. Together with learning rate, batch size is one of the most sensitive quantities worth scaling carefully.

### Source reconciliation

The slide reports the fitted relationship

$$
\log(B) = -6.24\log(L) + 20.91,
$$

where $B$ is the batch size used in the fit and $L$ is the target loss. The negative coefficient captures the spoken point: lower loss corresponds to a larger fitted batch.

### Additional explanation

The term **critical batch size** describes the point beyond which increasing the batch produces diminishing reductions in the number of optimizer steps. Below it, a larger batch can reduce gradient noise and speed optimization. Far above it, each step consumes more examples without a proportional reduction in steps.

The fitted equation is empirical and its units matter. Changing whether batch is measured in sequences, examples, or tokens changes the intercept. Changing the definition of loss can change both the intercept and apparent exponent. The portable lesson is the trend and fitting procedure, not the literal constants in isolation.

## 5. Warmup-stable-decay makes training horizons reusable

**Transcript coverage:** lines 426-604

### What the lecturer said - transcript only

Chinchilla-style experiments require models trained for many different data budgets. A cosine learning-rate schedule makes this expensive because the entire curve depends on the intended terminal horizon. A run scheduled for eight million sequences cannot simply continue from the endpoint of a four-million-sequence cosine run; its earlier learning rates would have been different. Repeating this over growing horizons creates a roughly quadratic accumulation of work.

MiniCPM uses **warmup-stable-decay (WSD)**, visually a trapezoid. Warmup is usually a fixed number of steps rather than a fixed fraction of the final horizon. A long stable phase holds the maximum learning rate constant. A comparatively rapid decay then anneals the model near the desired endpoint. The lecturer describes a typical decay phase as roughly 10% to 20% of the run and says the final rate is often about 10% of the maximum rather than necessarily literally zero.

WSD makes a long run reusable. To create a longer data-budget experiment, restore the most recent checkpoint in the stable phase, continue at the stable rate, and attach a new decay. Every target still pays for its own annealing segment, but that is much cheaper than replaying all preceding training.

During the stable phase, a WSD curve can appear substantially worse than a cosine curve. Much of the difference is recovered rapidly during decay, after which WSD often matches and sometimes exceeds cosine. The lecturer's anecdotal assessment is that cosine can be slightly better in some cases, whereas WSD is close enough in many cases to be a versatile default. The sharp improvement during decay also demonstrates how important final learning-rate annealing is.

### Source reconciliation

The schedule diagram labels three consecutive regions: warmup, a long constant stable phase, and decay. The comparison plots show the apparently lagging WSD checkpoints converging toward the cosine endpoint only after their individual decay branches. This visually confirms that an undecayed stable checkpoint and a fully annealed checkpoint should not be compared as if they were at the same training stage.

### Additional explanation

WSD separates two functions that cosine entangles:

- **Representation learning:** the long stable phase continues acquiring features and fitting data.
- **Endpoint optimization:** the decay phase converts the current state into a low-loss checkpoint at a chosen stopping point.

The rewind-and-decay procedure is especially useful for data-axis sweeps because one stable trajectory can generate many terminal checkpoints. It does not make those endpoints statistically independent, and an endpoint created by decay cannot simply resume at its tiny final learning rate. Continuation requires restoring an earlier stable checkpoint.

## 6. MiniCPM's Chinchilla replication and its limits

**Transcript coverage:** lines 605-680

### What the lecturer said - transcript only

With WSD, a data sweep can be produced by one long stable run plus repeated rewind-and-decay branches. Combining those branches with a model-size sweep supports Chinchilla-style analyses at much lower cost.

MiniCPM applies Chinchilla methods 1 and 3, which the lecturer considers the least reliable of the original methods. Method 1 nevertheless produces reasonably linear-looking lower envelopes, and method 3 produces smooth fits across compute, non-embedding parameters, and data. The broad Chinchilla pattern appears to replicate.

The joint-fit exponents differ substantially from those in Chinchilla. MiniCPM interprets them as evidence for training on far more text, but the lecturer is unsure whether this is a real change in optimal allocation or an artifact of an unusual fit. The main takeaways are therefore narrower: muP can stabilize learning-rate transfer, and WSD is a common, useful schedule that makes variable-horizon experiments practical.

### Source reconciliation

The slides write the joint loss model as

$$
L(N,D) = C_NN^{-\alpha} + C_DD^{-\beta} + L_0,
$$

and display a compute-optimal relation of the form

$$
\frac{N_{opt}}{D_{opt}} = K^2\left(\frac{C}{6}\right)^\eta.
$$

The numerical MiniCPM fit shown is approximately

$$
L(N,D) = \frac{7.54\times 10^{-2}}{N^{0.30}} + \frac{2.92\times 10^{-1}}{D^{0.30}} + 0.25.
$$

The slide also shows a data-to-model figure of 95.60 at $C=10^{21}$ under its conventions. These numerical details are visible on the slide; the spoken lecture only emphasizes that the exponents imply a much more data-heavy allocation and that the lecturer doubts the strength of that conclusion.

### Additional explanation

When $\alpha$ and $\beta$ are similar, the compute-optimal allocation under $C\approx6ND$ often gives comparable power-law growth in $N$ and $D$, but the coefficients can still imply a very different numerical token-to-parameter ratio. Small errors in a fitted exponent become consequential under long extrapolation because the error is itself exponentiated.

This is why a smooth in-range curve is not sufficient evidence. A scaling fit should also be checked for sensitivity to the chosen loss range, smallest models, optimization quality, parameter-count convention, and irreducible-loss term $L_0$.

---

# Part II - Predicting hyperparameter drift instead of removing it

## 7. DeepSeek fits optimal batch size and learning rate directly

**Transcript coverage:** lines 681-841

### What the lecturer said - transcript only

The original DeepSeek LLM report predates the group's better-known mixture-of-experts and reasoning work. The lecturer regards it as one of the best executed public scaling studies and says its experimental choices already showed that the group took scaling seriously.

DeepSeek uses a strategy different from MiniCPM's muP approach. Instead of changing the parameterization to make an optimum invariant, it assumes that optimal batch size and learning rate may move with scale but that their movement can be fit predictably. The authors run learning-rate and batch-size grids at several compute scales, hold the token budget for each comparison fixed, and identify the lowest terminal loss on each grid. They then plot those optima against non-embedding training FLOPs and fit lines in log space.

The fitted trend says that larger runs should use larger batches and smaller learning rates. The batch-size points look convincingly linear. The learning-rate points look much less clean to the lecturer, although the resulting models do train successfully. A coarse grid can create quantization error: if the true minimum falls between tried values, the selected grid point can move abruptly and make a weak trend look like an arbitrary straight line.

In response to a question about why runs at similar FLOP values can have a broad range of optimal learning rates, the lecturer attributes the variation to changes in model size and related configuration details rather than compute alone.

### Source reconciliation

The slides provide several details that are not stated precisely in the transcript:

- DeepSeek LLM released 7B- and 67B-parameter models.
- The example grids are labeled $10^{17}$ FLOPs at 177M FLOPs per token and $10^{20}$ FLOPs at 2.94B FLOPs per token.
- Candidate configurations within 0.25% of the minimum are treated as near-optimal in the learning-rate analysis.
- Slide 21 displays

$$
\eta_{\mathrm{opt}} = 0.3118C^{-0.1250},
\qquad
B_{\mathrm{opt}} = 0.2920C^{0.3271}.
$$

  The later summary table on slide 33 prints `0.3188` rather than `0.3118` for the learning-rate coefficient. The deck is internally inconsistent on this coefficient, so these notes preserve the discrepancy rather than selecting one silently. The exponent and the batch-size formula agree.

### Additional explanation

This method treats an optimum as a response surface. At scale $C_i$, one observes a noisy surface

$$
L_i(\eta,B),
$$

estimates its minimizer $(\eta_i^*,B_i^*)$, and only then fits how those minimizers change with $C_i$. There are therefore two error sources: error in locating each local minimum and error in extrapolating the fitted law. A high-resolution grid reduces the first but can be expensive; a coarse grid makes the apparent optimum discrete.

A useful practical safeguard is to fit a local quadratic or smooth surrogate near each minimum and carry uncertainty intervals into the scaling fit. A broad valley is also important: if many learning rates are nearly tied, predicting a robust near-optimal region can be more valuable than claiming a precise minimizer.

## 8. DeepSeek's WSD variant, IsoFLOPs replication, and final prediction

**Transcript coverage:** lines 842-941

### What the lecturer said - transcript only

DeepSeek also reproduces Chinchilla-style compute allocation, as many open builders did in 2024. It uses a WSD-like learning-rate schedule, although with two decay stages rather than MiniCPM's single final decay. The lecturer is unsure why this variation was chosen and notes that it did not become a widespread convention.

The schedule still provides the practical benefit needed for data-axis scaling. DeepSeek performs clean IsoFLOPs sweeps, finds the loss-minimizing model size at each fixed compute budget, and derives model-data tradeoffs similar to Chinchilla. The lecturer treats this as useful evidence that Chinchilla's basic analysis remains sound at the larger compute ranges used by a serious open-model builder.

The ultimate test is prediction. DeepSeek fits a power law to its smaller runs and compares the extrapolation with the two full models it actually trains. Their losses fall fairly close to the predicted curve. The agreement is not exact, but it is strong enough to demonstrate the intended workflow: curate small runs, fit the trend, and forecast a costly release-scale run before executing it.

The lecturer summarizes DeepSeek as the **fit-the-drift** alternative to muP. One can either try to stabilize sensitive hyperparameters or explicitly estimate how their optima change and follow that scaling law.

### Source reconciliation

The deck spells out DeepSeek's multi-step schedule:

- warm up for 2,000 steps to the maximum learning rate;
- remain at that rate until 80% of training tokens have been processed;
- reduce the rate to 31.6% of its maximum;
- at 90% of the tokens, reduce it again to 10% of the maximum;
- use gradient clipping of 1.0.

The comparison plot labels this as an `80% + 10% + 10%` schedule and shows that it is broadly competitive with cosine. The IsoFLOPs slide marks the 67B target near $4.5\times10^{23}$ training FLOPs and about $1.04\times10^{12}$ tokens under the paper's accounting. The final prediction plot identifies the held-out stars as DeepSeek LLM 7B and 67B, each trained on 2T tokens, and uses bits per byte on the validation set as its loss metric.

### Additional explanation

DeepSeek's schedule is WSD-like because most of the run is horizon-independent, but it is not the simple one-branch trapezoid used in the MiniCPM explanation. Its first drop by a factor of approximately $\sqrt{10}$ and final drop to one tenth form a piecewise schedule whose exact terminal behavior is part of the recipe.

The final stars test more than whether the loss curve is smooth. They test the whole chain: small-run optimization, compute accounting, model-size selection, data allocation, schedule transfer, and power-law extrapolation. Agreement can fail if any one of those components changes at target scale.

## 9. What newer release reports use scaling laws to decide

**Transcript coverage:** lines 942-1223

### What the lecturer said - transcript only

Recent model reports often describe less of the basic machinery because Chinchilla analysis and learning-rate or batch-size scaling have become standard internal practice.

**Qwen 2.5 and Qwen 3.** Qwen 2.5 says it performs scaling experiments to estimate optimal learning rates and batch sizes for both dense and mixture-of-experts models. Qwen 3 largely says it reuses the Qwen 2.5 method. This resembles DeepSeek's approach and suggests that fitted hyperparameter drift has become part of a normal pretraining recipe.

**Kimi K2.** As builders move fully to mixture-of-experts models, sparsity becomes a new axis that must be scaled. Kimi varies training FLOPs and sparsity, measures validation loss, and finds that greater sparsity improves loss at fixed FLOPs. A quantitative curve lets the group choose a sparsity ratio of 48 at the point where further sparsity shows diminishing returns. The purpose is not merely to confirm that sparsity helps, but to choose how much sparsity is worth its architectural and systems costs.

**Hunyuan.** Hunyuan carries out a related MoE IsoFLOPs analysis while holding its sparsity convention fixed. It obtains an allocation of roughly 96 data tokens per active parameter. The lecturer presents this as another example of every new MoE builder having to characterize the relation among active parameters, sparsity, compute, and loss.

**Llama 3.** Llama 3 reports ordinary IsoFLOPs curves and a token-to-model ratio somewhat different from, but broadly similar to, other studies. Its more interesting analysis connects intrinsic loss to benchmark accuracy. As compute grows, normalized negative log-likelihood improves; benchmark accuracy can then be modeled as a sigmoid of that loss. The plotted points deviate systematically from one universal curve, so the lecturer does not treat the sigmoid as exact. It nevertheless shows that better log loss is often tightly coupled to downstream performance.

**MiniMax-01.** MiniMax uses scaling to compare full softmax attention, a linear `lightning attention` mechanism, and a hybrid. The loss-compute and compute-optimal-size trends are broadly comparable across the three. That evidence supports choosing the hybrid for the deployed system rather than assuming that an architectural advantage observed at one small scale will persist.

The overall trend is that basic scaling procedures are less visible in reports because they are now assumed knowledge. Public detail is more likely to focus on a new axis, such as MoE sparsity or architecture choice.

### Source reconciliation

The slides add the following concrete scope and measurements:

- Qwen 2.5's quoted study spans dense models from 44M to 14B parameters, MoE models with 44M to 1B active parameters, and training datasets from 0.8B to 600B tokens. It also uses scaling to compare MoEs with dense counterparts.
- Kimi defines sparsity as total experts divided by activated experts. At a validation loss of 1.5, the report says sparsity 48 uses 1.69x, 1.39x, and 1.15x fewer FLOPs than sparsities 8, 16, and 32. The final model activates 8 of 384 experts per forward pass.
- Hunyuan's displayed extrapolation marks 58.1B activated parameters and labels its optimum as a 96:1 data-to-active-parameter ratio.
- Llama 3's slide labels its IsoFLOPs allocation as 39:1 and shows experiments from $6\times10^{18}$ to $10^{22}$ FLOPs, followed by an extrapolation to the 405B model.
- MiniMax's slide says the scaling runs cover models from 70M to 7B parameters.
- The summary slide describes the DeepSeek recipe as fitting batch size, learning rate, and model size; the MiniCPM recipe as using muP plus a piecewise-linear schedule; Kimi as MoE scaling; Llama 3 and Hunyuan as primarily IsoFLOPs; and MiniMax as architecture-decision scaling.

### Additional explanation

An MoE introduces at least three sizes:

- total parameters, which affect memory and model capacity;
- active parameters per token, which dominate per-token arithmetic;
- the expert count and routing pattern, which affect communication, load balance, and statistical specialization.

Consequently, `tokens per parameter` is ambiguous unless the denominator is stated. A data-to-active-parameter ratio is useful for compute allocation, while total parameters remain relevant to memory and inference deployment.

The Llama 3 mapping also illustrates a two-stage forecast:

$$
C \longrightarrow L(C) \longrightarrow A(L),
$$

where $L$ is intrinsic loss and $A$ is downstream accuracy. Uncertainty compounds across the two mappings. A sigmoid can describe saturation and threshold behavior, but benchmark format, prompting, contamination, and capability mix can all cause deviations from a one-dimensional loss predictor.

## 10. Why post-training does not yet fit cleanly into the recipe

**Transcript coverage:** lines 1224-1296

### What the lecturer said - transcript only

An audience member asks what changes when post-training is included in model development. The lecturer says this remains a major open question. There is no strong, fully integrated scaling procedure that accounts for post-training, partly because post-training can change what kind of pretraining is desirable.

The closest related work studies whether pretraining coverage or diversity can predict later post-training gains, but the lecturer considers that work nascent. There is not yet a satisfactory account of synergy between the stages.

The lecture then divides the remaining material according to the two earlier case studies. The DeepSeek-like route fits scaling laws for learning rate, batch size, and optimizer behavior. The MiniCPM-like route changes parameterization to make those choices invariant, especially as width changes.

### Additional explanation

Pretraining loss is comparatively smooth and cheap to measure on every checkpoint. Post-training objectives can depend on prompts, sampling temperature, verifier quality, reward hacking, environment success, and the strength of the starting model. Those dependencies make it harder to define one stable response variable analogous to validation cross-entropy.

An integrated law would need to allocate compute across stages, not merely within pretraining:

$$
C_{\mathrm{total}}
= C_{\mathrm{pre}} + C_{\mathrm{mid}} + C_{\mathrm{post}} + C_{\mathrm{inference/eval}},
$$

while predicting final behavior rather than only base-model loss. The optimum may change when a pretraining feature is easy to elicit through post-training but hard to create from scratch.

## 11. StepFun's high-resolution search over learning rate and batch size

**Transcript coverage:** lines 1297-1465

### What the lecturer said - transcript only

The lecture uses a recent StepFun preprint as the most substantial current attempt to estimate hyperparameter scaling. The lecturer does not regard any learning-rate study as fully reliable, but StepFun builds credible large models and spends considerable compute on a dense grid of experiments.

Prior proposals disagree even about which independent variables belong in the law. Kaplan's critical batch size is expressed as a function of terminal loss. DeepSeek uses training compute. StepFun proposes a dependence on data size, and other papers use model size or combinations of model and data. The lecturer warns against treating the newest row of such a comparison table as settled truth. The fitted constants and functional forms remain brittle, even when the empirical patterns are instructive.

StepFun's design resembles DeepSeek's. It sweeps learning rate and batch size across multiple model sizes and token budgets, maps the loss surface, and locates a minimum. An example slice at approximately 1B parameters and 100B tokens has a smooth, nearly convex valley along either coordinate. This makes local optimization plausible and provides some confidence that the observed minimum is not simply a jagged artifact of training noise.

### Source reconciliation

The slide identifies the paper as *Predictable Scale: Part I, Step Law - Optimal Hyperparameter Scaling Law in Large Language Model Pre-training*. Its comparison table reports several incompatible formulas and gives Step Law the form

$$
\eta_{\mathrm{opt}} = 1.79N^{-0.713}D^{0.307},
\qquad
B_{\mathrm{opt}} = 0.58D^{0.571}.
$$

The same table reports a relative error of 0.94% for Step Law, compared with larger errors for the listed OpenAI, Microsoft, DeepSeek, and Porian formulas. Those percentages are results under the paper's evaluation setup, not a general ranking across all model-training regimes.

### Additional explanation

Smooth one-dimensional slices do not prove that the complete surface is globally convex. They do, however, support local interpolation between grid points. A practical search can exploit this by:

1. sampling a coarse grid in log learning rate and log batch size;
2. fitting a smooth response surface near the best region;
3. adding points where uncertainty around the minimum is largest;
4. reporting a near-optimal region rather than only one coordinate pair.

Because batch size is discrete and constrained by sequence length, parallelism, and hardware memory, the statistically fitted optimum must ultimately be rounded to a realizable systems configuration.

## 12. StepFun's empirical trends, transfer, and data dependence

**Transcript coverage:** lines 1466-1629

### What the lecturer said - transcript only

StepFun's strongest empirical observation is that optimal batch size appears to depend primarily on the total amount of training data. When optimal batches for several differently sized models are plotted against data size on log-log axes, the colored model families approximately collapse onto one line.

Learning rate behaves differently. Larger models prefer smaller learning rates, while larger token budgets at fixed model size prefer larger learning rates. The latter direction is counterintuitive and may be fragile; the lecturer notes that other work argues for the reverse dependence. In these experiments, however, both trends are visible.

Learning-rate choices are relatively forgiving in ordinary language-model training, and practitioners often know a sensible order-of-magnitude range before running a sweep. StepFun's fitted optima also transfer reasonably to mixture-of-experts models when comparisons control for active parameters. Predicted optima and directly measured MoE optima are close, though not identical.

Changing the training-data distribution shifts both learning-rate and batch-size optima. The numerical coefficients are therefore contingent on the data recipe, even if the qualitative form transfers.

At a coarse level, the batch law is close to a square-root dependence on data. The learning rate grows with data and falls with model size. Under Chinchilla-like joint scaling, model size and data size both grow with compute, and the model-size effect dominates, so learning rate decreases with compute while batch size increases. This recovers the same directions as DeepSeek, though with different exponents.

### Source reconciliation

The exact displayed Step Law uses exponent $0.571$, not exactly $1/2$, for batch size. Slide 36 explicitly warns that the positive data exponent in the learning-rate fit may be fragile under a different schedule and points to an InternLM scaling study as an example of contrary evidence.

The transfer slide tests MoEs at low, medium, and high sparsity and separately tests a bilingual corpus, a code-integrated mixture, and a code-dominant mixture. The MoE optima are closer to the predictions than the cross-dataset optima, visually supporting the spoken claim that architecture transfer is better than data-mixture transfer in this experiment.

### Additional explanation

If a simplified Chinchilla path has $N\propto C^{1/2}$ and $D\propto C^{1/2}$, substituting the slide's Step Law gives

$$
\eta_{\mathrm{opt}}
\propto C^{-0.713/2}C^{0.307/2}
= C^{-0.203},
$$

and

$$
B_{\mathrm{opt}}
\propto C^{0.571/2}
= C^{0.2855}.
$$

This derivation explains how a law that increases learning rate with $D$ at fixed $N$ can still predict a decreasing learning rate along the joint compute-optimal path. It is only an illustration: actual Chinchilla exponents and the definition of $N$ may differ.

Data dependence is not a nuisance variable to be averaged away. It can change gradient noise, sequence difficulty, repetition, token entropy, and the curvature of the objective. A transferred formula is best treated as a prior for a smaller local sweep.

## 13. When to reuse a published law and when to rerun the sweep

**Transcript coverage:** lines 1630-1725

### What the lecturer said - transcript only

An audience member asks whether practitioners should simply apply published scaling laws or run their own grid. The answer depends on compute, the desired precision, and how closely the new setting matches the study.

Near StepFun's measured regime, the lecturer would trust its law as a better default than an arbitrary guess. A major pretraining run is different: architecture, weight decay, data, schedule, or another seemingly minor choice may change the optimum or its scaling exponent. This is why model builders repeatedly redo Chinchilla and hyperparameter studies even when a published answer already exists. They want to verify that it remains first-order correct for their own recipe.

Scaling laws look highly scientific because they involve straight lines and extrapolation. In practice, deciding whether someone else's experiment is similar enough to transfer still involves judgment - what the lecturer calls `vibes`. No finite report can rule out every small but consequential mismatch.

### Additional explanation

A useful hierarchy is:

- **Default:** use a published law when the architecture, optimizer, schedule, data, and scale lie inside or near its measured range.
- **Calibration:** run a small local sweep to estimate an intercept correction while retaining the published exponent.
- **Refit:** estimate both coefficient and exponent when a sensitive component changes.
- **Reformulate:** abandon the old independent variable when residuals reveal systematic dependence on a missing factor.

The cost of calibration should be compared with the value at risk. Spending a small percentage of the target budget on validation is rational when an unstable or substantially suboptimal run could waste the remaining budget.

---

# Part III - Optimizers as scale-dependent algorithms

## 14. Fair optimizer comparisons require optimizer-specific tuning

**Transcript coverage:** lines 1726-1866

### What the lecturer said - transcript only

Recent optimizer work is exciting because alternatives to Adam have produced large gains on small language-model speedruns. On the NanoGPT speedrun that inspired Assignment 1, Muon substantially improves time-to-target loss over Adam without an obviously prohibitive per-step slowdown. The central question is whether that gain survives at realistic scale.

Before studying scale, each optimizer must be tuned fairly. Different optimizers can require different learning rates and may even have different hyperparameter scaling exponents. A poorly tuned Adam run can look far worse than several alternatives; changing Adam's learning rate can erase much of the apparent advantage. The same issue applies to weight decay. Reusing one small weight decay for an optimizer whose optimum is much larger can manufacture a misleading comparison.

The lecturer presents this not only as a scaling issue but as a basic empirical-machine-learning rule: compare algorithms near their respective optima, rather than giving all of them one shared configuration.

### Source reconciliation

The slide's learning-rate example compares AdamW at $6\times10^{-4}$ with AdamW at $8\times10^{-3}$ on a 130M model and labels the latter as roughly a 2x speedup. Its weight-decay plot marks approximately 0.6 as optimal for Lion in that experiment. These are examples of optimizer-specific tuning, not recommended universal defaults.

The NanoGPT plot reports approximate step times of 139 ms for Adam, 142 ms for Muon, 154 or 179 ms for two Distributed Shampoo settings, and 301 ms for the then-current SOAP implementation on 8 H100 GPUs. The relevant comparison is wall-clock time to a loss target, which combines optimization efficiency with per-step systems cost.

### Additional explanation

An optimizer comparison can be biased along at least four dimensions:

- number of optimizer steps;
- examples or tokens processed;
- total training FLOPs;
- wall-clock time on a specified implementation and device.

An algorithm that needs fewer steps can still lose in wall-clock time if each step is expensive. Conversely, a fused or matrix-multiply-heavy implementation may be fast on a GPU even when its mathematical update appears more elaborate. A credible result should state which budget is fixed and tune every optimizer under that budget.

## 15. The two scaling axes: compute and tokens per parameter

**Transcript coverage:** lines 1867-1969

### What the lecturer said - transcript only

The optimizer study exposes two axes that should be checked whenever an algorithm is claimed to scale.

The first is total compute. If model size and data size are increased together at a fixed ratio, the horizontal axis effectively represents a compute ladder. Relative to Adam, Muon has a large speedup at small model size, but the measured advantage shrinks as scale increases. That is not the trend one would hope to extrapolate from the smallest run.

The second is the Chinchilla ratio: training tokens divided by model parameters. This ratio distinguishes a relatively overparameterized regime from a data-rich regime. An algorithm might look good with few tokens per parameter because it regularizes implicitly or because its statistical inefficiency is hidden. Another algorithm might excel at large ratios because it packs more learned information into each parameter. Even otherwise careful model-size studies often lack the compute to vary this axis.

For the optimizer comparison shown here, relative gains are fairly consistent across the tested Chinchilla ratios. The lecturer stresses that this is not guaranteed in general and that both axes should always be examined.

### Source reconciliation

The slides show optimizer speedups on 130M, 300M, 520M, and 1.2B models trained at an 8x Chinchilla setting. Muon and SOAP fall from roughly 1.4x the AdamW baseline at 130M to about 1.1x at 1.2B. A separate 520M plot varies the Chinchilla ratio from 1 to 8 and shows matrix-oriented optimizers as solid lines and scalar-oriented optimizers as dashed lines; the solid curves remain modestly better throughout that tested range.

### Additional explanation

These axes should be varied independently when possible. If a study only follows one path $D=kN$, then compute scale and data richness are confounded. A two-dimensional design asks whether an observed speedup is approximately

$$
S(N,D) \approx S_0,
$$

or whether it changes systematically with $N$, $D/N$, or both. Even a sparse factorial design can reveal interactions that a single diagonal scaling ladder hides.

The ratio $D/N$ also affects deployment interpretation. Training far beyond the compute-optimal ratio may deliberately produce a smaller model with lower serving cost. An optimizer that retains gains in that data-rich region can therefore be valuable even if its Chinchilla-optimal speedup is modest.

## 16. A clean scaling line can still bend and fail

**Transcript coverage:** lines 1970-2051

### What the lecturer said - transcript only

Establishing that a new algorithm scales is expensive and nontrivial. The lecturer uses public experiments from the Marin project, associated with Will Held, as an unusually informative example because the team publishes failed runs rather than only a successful final plot.

The recipe combines a Cautious AdamC variant, square-root batch-size scaling, and other apparently standard choices. Its IsoFLOPs and held-out scaling curves look excellent across several compute levels. Beyond a vertical extrapolation boundary, however, the next runs become somewhat worse, then much worse, and finally diverge.

The team repairs the behavior with more careful muP-like parameterization and optimizer changes, after which scaling remains well behaved across a larger range. The lesson is not that one particular fix always works. It is that a trend can look linear over several orders of magnitude and still break abruptly. Real scaling work often produces confusing plots before it produces a trustworthy recipe.

### Source reconciliation

The held-out plot labels the first two extrapolation misses as approximately 0.8% and 2.5% worse than predicted, followed by a diverged run near $10^{23}$ FLOPs. The slide describes the repaired recipe as `Cautious AdamC + sqrt batch-size scaling of learning rates` together with more careful parameterization, scaling, and optimizer changes.

### Additional explanation

Power laws are local empirical regularities until validated out of range. A sudden bend can indicate:

- numerical instability that only appears at a larger width or batch;
- an optimizer state or update whose scale changes with dimension;
- a schedule transition occurring at a different effective time;
- data or evaluation saturation;
- a systems change that alters numerical precision or synchronization;
- selection bias from fitting only successful small runs.

Holding out the largest affordable rung is therefore valuable. Fit the law without it, preregister the prediction, and treat the rung as an extrapolation test before committing to the final scale.

## 17. Muon: momentum followed by spectral orthogonalization

**Transcript coverage:** lines 2052-2201

### What the lecturer said - transcript only

Muon merits a closer look because it has moved from a small speedrun into a large model training run. In ordinary momentum, a gradient is accumulated into a momentum buffer $B_t$, and the parameters are updated in that direction.

Muon begins from the observation that neural-network parameters have different structures. RMSNorm gains and similar tensors are vectors, while attention and MLP weights are matrices. A matrix has singular values, so its update can be treated spectrally rather than coordinate by coordinate.

Conceptually, write the matrix momentum update as

$$
B_t = U\Sigma V^\top.
$$

Muon replaces all singular values by one and applies $UV^\top$ as the update direction. Very large singular directions shrink and very small ones expand. The lecturer compares this with AdaGrad or Adam, which normalize coordinatewise magnitudes; Muon instead normalizes directions in a matrix's spectral geometry.

This operation only makes sense for matrix-valued parameters. Vector-valued parameters can still use AdamW. Muon does not compute a full SVD in training. It uses a fixed number of Newton-Schulz matrix-multiplication iterations to approximate the orthogonalized factor, making the operation practical on accelerators even though it is not an exact SVD.

### Source reconciliation

The slide spells the method `NewtonSchulz5` and gives the following pseudocode:

$$
\begin{aligned}
G_t &\leftarrow \nabla_\theta \mathcal{L}_t(\theta_{t-1}),\\
B_t &\leftarrow \mu B_{t-1} + G_t,\\
O_t &\leftarrow \operatorname{NewtonSchulz5}(B_t),\\
\theta_t &\leftarrow \theta_{t-1} - \eta O_t.
\end{aligned}
$$

The transcript repeatedly renders the name as `NewtonSchultz`; the slide's `NewtonSchulz` is used for the standard spelling. The deck also makes explicit that the idealized map is

$$
B_t=U\Sigma V^\top \longmapsto UV^\top.
$$

### Additional explanation

$UV^\top$ is the orthogonal, or for a rectangular matrix semi-orthogonal, factor in the polar decomposition. It has singular values equal to one on the supported subspace. This controls the spectral norm of the update rather than its elementwise magnitude.

Newton-Schulz iterations approximate a matrix inverse square root through repeated matrix multiplications. That makes the method attractive on GPUs, where large dense matrix multiplications are highly optimized. The iteration still requires scaling and stabilization; an approximation that is fast but outside its convergence region can become numerically unstable.

## 18. Muon works at scale, but superiority over Adam remains unproven

**Transcript coverage:** lines 2202-2282

### What the lecturer said - transcript only

Muon is extremely effective on the NanoGPT speedrun, yet controlled scaling studies report that its advantage over Adam diminishes as model size grows. That initially made the lecturer think the idea might never be used in a full pretraining run.

Kimi K2 changed that conclusion by training the entire large model with Muon, plus several safeguards introduced after the team encountered instabilities. Kimi K2 is a strong model and its training curve looks reasonable. This demonstrates that Muon is a workable large-scale optimizer.

It does **not** demonstrate that Muon is better than AdamW at the same scale, because the Kimi report provides no target-scale Adam ablation. The evidence supports feasibility, not a comparative causal claim.

The broader lesson is that judging scale transfer is extremely hard. Small experiments such as NanoGPT speedruns remain scientifically useful because they generate ideas cheaply. Some of those ideas eventually reach large models, but definitive large-scale ablations may remain unavailable. Whether Muon beats AdamW at frontier scale is still an open question.

### Source reconciliation

The slide juxtaposes the NanoGPT wall-clock result, the diminishing speedup from 130M through 1.2B, and Kimi K2's smooth loss curve over roughly 15T training tokens. Its caption is deliberately narrow: scaling gains are hard to measure, but Muon clearly `works` at scale.

### Additional explanation

Three claims should be kept distinct:

1. **Correctness:** the optimizer can train a model without pathological failure.
2. **Effectiveness:** it reaches a strong absolute result under one recipe.
3. **Superiority:** it beats a well-tuned alternative under a controlled budget.

Kimi K2 supports the first two. Only a matched ablation with comparable data, model, compute, schedule, tuning effort, and implementation quality would support the third.

## 19. Questions on Muon cost, per-layer optimizers, and interactions

**Transcript coverage:** lines 2283-2405

### What the lecturer said - transcript only

An audience question asks whether SVD is fast on GPUs. The lecturer says full SVD is not fast, but Muon does not run it. Newton-Schulz uses a finite sequence of matrix multiplications to approximate orthogonalization. The idea therefore has three separable pieces: recognizing matrix structure, choosing spectral orthogonalization, and implementing it with accelerator-friendly operations.

Another question asks about nontrivial hyperparameters. The lecturer confirms that optimizer hyperparameters differ. He cites a line of thought associated with Jeremy Bernstein in which each layer might have its own learning rate or even its own optimizer. The limiting view is that every Transformer parameter type is structurally different. That may motivate specialized updates, but it would also create an undesirable number of hyperparameters to tune.

A final question asks whether learning rate and batch size interact with all the other hyperparameters omitted from a two-dimensional grid. The lecturer agrees. A complete grid grows exponentially and is infeasible. Builders therefore sweep the most sensitive choices, especially learning rate and batch size, then may perform local one-dimensional sweeps for choices such as weight decay.

### Additional explanation

This is a structured-search compromise. If $k$ hyperparameters each have $m$ candidate values, a full grid costs $m^k$ runs. Domain knowledge is used to choose a small coupled core, optimize it jointly, and condition later sweeps on that core.

The danger is a strong interaction outside the core. Fractional factorial designs, Bayesian optimization, or sequential ablations can detect some interactions with fewer runs, but no search method removes the need to decide which dimensions and ranges are plausible.

---

# Part IV - Maximum update parameterization in depth

## 20. The muP objective, empirical evidence, and conceptual sources

**Transcript coverage:** lines 2406-2538

### What the lecturer said - transcript only

The lecturer describes muP as somewhat mysterious. Multiple papers and implementations do not always agree on the mathematical account or exact implementation, yet they share a core program and the method often works.

The desired outcome is simple: as network width grows, the base learning rate that minimizes loss should remain fixed. Achieving that invariance permits more unusual knobs than standard practice. Initialization may change by layer, learning rate may change by parameter type, and residual contributions may be rescaled with model size.

Cerebras provides evidence at meaningful scale. The company, whose language-model effort is associated with scaling-law researcher Joel Hestness, trained CerebrasGPT models from roughly 0.1B to 13B parameters with a Chinchilla recipe and also evaluated a muP variant. With muP, the extrapolated scaling trend lands close to the actual large-model results. The corresponding standard-parameterization predictions fluctuate more. MiniCPM and other models provide further evidence that muP variants can be used in real training.

The lecturer then turns from examples to foundations. Greg Yang initiated the tensor-programs research line, although the lecturer finds those papers difficult to penetrate. He recommends a paper involving Jeremy Bernstein that reframes the idea through a spectral condition for feature learning and regards it as the clearest accessible treatment. Other physicist-oriented accounts and Cerebras work provide related explanations.

### Source reconciliation

The width-scaling table on slide 44 defines a target model $M'$ whose widths are $r$ times those of a base model $M$. For `matrix-like` parameters, it shows AdamW learning rate $l\mapsto l/r$ and initialization variance $\sigma\mapsto\sigma/r$. For other parameters, those quantities remain unchanged. The output multiplier scales as $\tau\mapsto\tau/r$, while other multipliers remain fixed.

The Cerebras slide says that hyperparameters are tuned on a 40M-parameter muP model and transferred along the muP law to a 2.7B-parameter model. It also plots the larger 0.1B-to-13B CerebrasGPT family and emphasizes that the muP residuals stay much closer to the predicted scaling curve.

### Additional explanation

muP is a coupled reparameterization, not simply a special random initialization. The forward scale, backward scale, optimizer update, and output scale must agree. Changing only one entry from the slide's table can destroy the invariant the other entries were designed to preserve.

Hyperparameter transfer also has a specific direction. One tunes a small **base model** in muP coordinates, expands selected widths by a multiplier $r$, and transforms matrix-like learning rates, variances, and multipliers according to the rule. The target model does not literally use identical per-tensor learning rates; it uses the same base coordinate after applying the width-dependent multipliers.

## 21. The two muP invariants: stable activations and feature learning

**Transcript coverage:** lines 2539-2621

### What the lecturer said - transcript only

For the width-scaling argument, muP begins with two assertions.

1. At initialization, individual activations should remain order one as width changes. They should neither diverge nor vanish merely because the network is wider.
2. After one gradient step, the change in an activation should also remain order one. The network should continue to change its learned features by a meaningful amount at large width.

The second condition is called **feature learning**. The lecturer contrasts it with a neural-tangent-kernel regime, in which changes in hidden activations vanish as width grows. That limiting behavior does not satisfy the desired condition because the internal representation becomes effectively frozen.

The discussion alternates between coordinate scale and vector norm. If each of the $n_l$ coordinates in layer $l$ is order one, the Euclidean norm of the activation vector is order $\sqrt{n_l}$. The same conversion applies to an order-one coordinate change.

### Source reconciliation

The slide states the conditions with asymptotically tight notation:

$$
\text{A1: } h_{l,i}\text{ at initialization is }\Theta(1),
$$

$$
\text{A2: } \Delta h_{l,i}\text{ after one step is }\Theta(1).
$$

It then notes that coordinatewise $\Theta(1)$ implies

$$
\|h_l\|_2=\Theta(\sqrt{n_l}),
\qquad
\|\Delta h_l\|_2=\Theta(\sqrt{n_l}).
$$

The transcript sometimes says `O(1)` while also requiring a non-vanishing change. The slide's $\Theta(1)$ better captures both an upper and a lower order requirement.

### Additional explanation

Boundedness alone is insufficient for feature learning. A change of size $1/n_l$ is technically $O(1)$ but vanishes with width. The intended condition is that a typical coordinate change remains bounded above and below at constant order, hence the more precise $\Theta(1)$ notation.

The two invariants constrain different stages:

- A1 constrains initialization so signals propagate at a useful magnitude.
- A2 constrains the optimizer and learning-rate scaling so training remains nontrivial.

A parameterization can satisfy A1 while failing A2. Standard fan-in initialization often keeps forward activations stable, yet the effective feature update can still shrink or grow with width.

## 22. Deriving the initialization scale from condition A1

**Transcript coverage:** lines 2622-2745

### What the lecturer said - transcript only

The lecturer sketches the derivation for a deep linear network. At layer $l$,

$$
h_l=W_lh_{l-1},
$$

and the entries of $W_l$ are Gaussian with a width-dependent standard deviation $\sigma_l$. Standard random-matrix concentration gives the spectral norm of a Gaussian matrix as proportional to $\sigma_l$ times the sum of the square roots of its fan-in and fan-out.

In a suitable high-dimensional regime, the output norm is approximated by the input norm times this operator norm. It is always an upper bound and is only approximately attained under the assumptions being used.

Assume inductively that $\|h_{l-1}\|_2$ has order $\sqrt{n_{l-1}}$. Choose $\sigma_l$ so that multiplying by $W_l$ changes this to order $\sqrt{n_l}$. Substituting that choice makes the operator norm scale like $\sqrt{n_l/n_{l-1}}$, and the output norm becomes $\sqrt{n_l}$ up to lower-order terms. Repeating the argument across layers preserves order-one coordinates at initialization.

The lecturer acknowledges that the choice initially appears to be pulled from a hat. Its justification is that, when substituted into the induction, it satisfies the desired invariant under the stated approximations.

### Source reconciliation

The slides make the derivation explicit. With

$$
W_l\sim\mathcal{N}\!\left(0,\sigma_l^2I_{n_l\times n_{l-1}}\right),
$$

the operator norm is approximated by

$$
\|W_l\|_*
\longrightarrow
\sigma_l\left(\sqrt{n_{l-1}}+\sqrt{n_l}\right).
$$

The selected standard deviation is

$$
\sigma_l
=
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\left(\sqrt{n_l}+\sqrt{n_{l-1}}\right)^{-1}
=
\Theta\!\left[
\frac{1}{\sqrt{n_{l-1}}}
\min\!\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)
\right].
$$

Under the induction hypothesis $\|h_{l-1}\|_2=\Theta(\sqrt{n_{l-1}})$, this gives

$$
\|W_l\|_*\longrightarrow\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}},
\qquad
\|h_l\|_2=\sqrt{n_l}+o(\sqrt{n_l}).
$$

The slide explicitly calls this a kind of worst-case derivation because the approximate equality based on the operator norm begins from an upper bound.

### Additional explanation

The `min` term only changes the usual fan-in scale when fan-out is narrower than fan-in:

- If $n_l\ge n_{l-1}$, then $\min(1,\sqrt{n_l/n_{l-1}})=1$, so $\sigma_l=\Theta(1/\sqrt{n_{l-1}})$.
- If $n_l<n_{l-1}$, then $\sigma_l=\Theta(\sqrt{n_l}/n_{l-1})$, which is smaller than standard fan-in initialization.

This helps explain why muP changes some projection and output layers while leaving square hidden-to-hidden matrices closer to familiar initialization. It also shows why the derivation should not be read as an exact distributional identity: it controls asymptotic scale under a deliberately conservative norm argument.

## 23. Deriving the update scale and layerwise learning rate from condition A2

**Transcript coverage:** lines 2746-2961

### What the lecturer said - transcript only

The second muP condition concerns changes after one optimizer step. The lecturer restricts the derivation to one SGD example in a deep linear network. For a linear layer, the weight update is a rank-one outer product between the loss gradient at the layer and the preceding activation:

$$
\Delta W_l=-\eta_l\nabla_{h_l}\ell\,h_{l-1}^{\top}.
$$

Expanding the updated activation produces three contributions:

$$
\Delta h_l
=
W_l\Delta h_{l-1}
+\Delta W_lh_{l-1}
+\Delta W_l\Delta h_{l-1}.
$$

The first is the propagated change from the previous layer, the second is the direct effect of the changed weights on the old activation, and the third is their interaction. The order-of-magnitude argument asks each leading contribution to have norm $\Theta(\sqrt{n_l})$, matching a coordinatewise order-one activation change. It assumes that leading terms do not cancel.

The induction hypothesis and the A1 forward-scale argument supply the right scale for $W_l\Delta h_{l-1}$. The other terms require

$$
\|\Delta W_l\|_*
=
\Theta\!\left(
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\right).
$$

To connect that desired matrix-update norm to a learning rate, the lecturer makes another assumption: one step changes the loss by order one across widths, so training makes a comparable nontrivial amount of progress. A first-order Taylor approximation writes the loss change as an inner product between the weight update and the weight gradient. Because the update is rank one, the Frobenius and operator-norm scales can be related cleanly.

Combining the order-one loss-change assumption with the desired update norm implies a gradient norm of the reciprocal scale. Solving for the SGD learning rate gives

$$
\eta_l=\Theta\!\left(\frac{n_l}{n_{l-1}}\right),
$$

the fan-out to fan-in ratio. The lecturer notes that the analogous Adam argument instead yields a scale proportional to $1/n_{l-1}$.

The derivation is deliberately an order-tracking sketch rather than rigorous mathematics. It assumes no cancellations, approximately tight norm bounds, a single-example rank-one update, and comparable loss progress across widths. Its value is to show how invariance conditions constrain initialization and optimizer scaling.

### Source reconciliation

The slides display the update and expansion as

$$
\Delta W_l=-\eta_l\nabla_{h_l}\ell\,h_{l-1}^{\top},
$$

$$
\Delta h_l=W_l\Delta h_{l-1}
+\Delta W_l(h_{l-1}+\Delta h_{l-1}).
$$

They then impose

$$
\|\Delta W_l\|_*\sqrt{n_{l-1}}
=
\Theta(\sqrt{n_l}).
$$

With the additional assumption

$$
\Delta\ell
\approx
\Theta\!\left(
\langle\Delta W_l,\nabla_{W_l}\ell\rangle
\right)
=\Theta(1),
$$

the slide derives

$$
\|\nabla_{W_l}\ell\|_*
=
\Theta\!\left(
\frac{\sqrt{n_{l-1}}}{\sqrt{n_l}}
\right)
$$

and finally $\eta_l=\Theta(n_l/n_{l-1})$ for SGD. The slide separately flags the Adam result without deriving it.

### Additional explanation

The A2 derivation follows a constraint-solving pattern:

1. Specify the desired representation change, $\|\Delta h_l\|_2=\Theta(\sqrt{n_l})$.
2. Translate it into a desired matrix-update scale.
3. Assume a scale for the optimization progress.
4. Solve for the learning rate that makes both statements compatible.

For square hidden matrices, $n_l=n_{l-1}$, so the SGD rule is width independent. Projection matrices with unequal fan-in and fan-out receive different rates. Adam's coordinate normalization changes the algebra, making its layerwise rate decrease with fan-in even for square matrices.

The order-one loss-change assumption is the most contestable step. It is not a theorem that two widths should improve by the same amount after one update. Rather, it encodes the transfer behavior muP is trying to construct. Empirical validation remains necessary.

## 24. The muP mini-recipe and the broader invariance method

**Transcript coverage:** lines 2962-3060

### What the lecturer said - transcript only

The detailed derivation changes two practical choices relative to a standard parameterization. The initialization standard deviation becomes the usual inverse-square-root fan-in scale multiplied by a fan-out correction:

$$
\sigma_l
=
\Theta\!\left[
\frac{1}{\sqrt{n_{l-1}}}
\min\!\left(
1,\sqrt{\frac{n_l}{n_{l-1}}}
\right)
\right].
$$

When fan-out is not narrower than fan-in, this reduces to the familiar initialization. When fan-out is smaller, the extra factor matters.

Under the simplified SGD argument, the learning rate is proportional to $n_l/n_{l-1}$ rather than held constant across every matrix. Under Adam, the layer-specific rate is proportional to $1/n_{l-1}$, so layers with larger fan-in receive smaller rates. The lecturer does not derive the Adam result in class.

The high-level contribution is not only this table of formulas. muP demonstrates a general way to design scale-aware algorithms: take a width limit, state the network quantities that should remain invariant, add explicit approximations, and solve the resulting constraints on hyperparameters. This style resembles physicists' order-of-magnitude reasoning more than a conventional exact machine-learning derivation.

### Source reconciliation

The recap slide contrasts the two parameterizations:

| Quantity | Simplified muP scale | Standard scale |
|---|---|---|
| Initialization standard deviation | $\Theta\!\left[\frac{1}{\sqrt{n_{l-1}}}\min\left(1,\sqrt{n_l/n_{l-1}}\right)\right]$ | $1/\sqrt{n_{l-1}}$ |
| SGD learning rate | $n_l/n_{l-1}$ | $\Theta(1)$ |
| Adam learning rate | $1/n_{l-1}$ | typically one global base rate |

The slide emphasizes that the initialization differs when fan-out is smaller than fan-in and that Adam's layerwise learning-rate rule is the more conspicuous practical difference.

### Additional explanation

The mini-recipe is pedagogical rather than a complete Transformer implementation. Real muP also distinguishes embeddings, attention projections, MLP matrices, normalization parameters, biases, residual multipliers, and the output head. The correct transfer rule depends on which dimensions are designated as width dimensions.

The general invariance workflow is reusable:

$$
\text{desired scale-invariant behavior}
\rightarrow
\text{asymptotic constraints}
\rightarrow
\text{parameter and optimizer rules}
\rightarrow
\text{empirical stress test}.
$$

It can suggest new parameterizations, but it cannot prove that omitted nonlinearities, normalization, data distributions, schedules, or optimizers preserve the same invariants.

## 25. muP stress tests: what transfers and what breaks

**Transcript coverage:** lines 3061-3183

### What the lecturer said - transcript only

The lecturer closes the muP section with empirical stress tests from an independent researcher. muP can be viewed as a hyperparameter-transfer procedure: different Transformer parameter classes receive different initialization and Adam learning-rate scalings as width changes.

In controlled language-model experiments, baseline muP reproduces the central claim. The learning rate that is optimal for the smallest model remains optimal, or directly predicts the optimum, for models that are successively much wider. A variant that tracks projection biases also transfers.

Real language models contain many elements outside the clean theory, including SwiGLU or squared-ReLU activations, varying batch sizes, initialization variants, RMSNorm, exotic optimizers, and regularizers. Most tested deviations remain compatible enough that transfer still works.

There are important failures. Learning the RMSNorm gain breaks learning-rate transfer, although the gain can often be removed with little performance loss. Lion, a sign-based optimizer that is spiritually related to other structured update methods, also breaks the tested muP transfer. Large decoupled weight decay produces a particularly significant failure.

Overall, muP is useful when the goal is to stabilize the optimal learning rate across width. Standard parameterization can exhibit a large, predictable shift in the optimum, while muP makes it much more stable. The program remains an open research area rather than a finished universal solution. Directly fitting hyperparameter scaling laws remains a viable competing tool.

### Source reconciliation

The slides identify the stress-test paper as Lucas Dax Lingle's **A Large-Scale Exploration of mu-Transfer**. Its baseline table scales width from 128 to 512 to 2048, with each model four times wider and sixteen times larger, and shows the smallest model's best base learning rate transferring to the larger models. Projection-bias tracking also transfers.

The tested deviations include SwiGLU and squared ReLU, batch-size changes, zero-attention and other initializations, RMSNorm gains, Lion, and regularizers. The failure slides distinguish trainable vector or scalar RMSNorm gains, Lion, and strong decoupled weight decay. The weight-decay example uses $0.1$ and is labeled the most significant observed muP failure.

The final comparison shows standard parameterization's best learning rate moving strongly with width, while a large muP run retains a stable base optimum through models shown up to roughly 10B parameters.

### Additional explanation

Robustness outside a theory's assumptions can have several explanations:

- the omitted component changes only constants, not width exponents;
- normalization suppresses the scale mismatch;
- the tested width range is not large enough for the violation to dominate;
- the hyperparameter grid is too coarse to reveal a smaller drift.

A failed transfer is therefore more informative than a single successful transfer. Trainable normalization gains introduce new width-sensitive degrees of freedom. Sign-based optimizers discard gradient magnitude, changing the scaling argument used for Adam. Decoupled weight decay adds an update proportional to parameter magnitude rather than the loss gradient. Each interferes with a different part of the coupled invariant.

## 26. Scaling in the wild remains an art without a silver bullet

**Transcript coverage:** lines 3184-3233

### What the lecturer said - transcript only

The lecture ends by warning that real scaling is far messier than the clean presentation of a fitted line. Scaling laws are used to choose architectures, optimizers, learning rates, batch sizes, and other hyperparameters, but no one knows that an empirical relation will extrapolate forever.

Builders can improve their odds by using muP, searching learning rate and batch size at small scale, predicting their drift, or adopting reusable schedules. These tools control parts of the risk but do not eliminate it. There is no silver bullet yet, although a future course might be able to report a more complete solution.

### Source reconciliation

The recap slide lists three practical challenges:

1. choosing architecture hyperparameters such as width;
2. choosing optimizer hyperparameters such as learning rate and batch size;
3. paying the compute cost of a large Chinchilla-style sweep.

It lists corresponding partial solutions: assume stability or use muP, search sensitive hyperparameters at small scale and either hold or extrapolate them, and use reusable WSD-like schedules.

### Additional explanation

Scaling practice combines three kinds of evidence:

$$
\text{empirical fits}
\;+\;
\text{invariance arguments}
\;+\;
\text{large-run monitoring}.
$$

Fits summarize observed drift. Invariance arguments propose why some drift should disappear. Monitoring catches the ways both can fail outside their experimental range. A responsible large-run plan keeps contingency compute, validates intermediate checkpoints against predictions, and treats disagreement as evidence rather than forcing the run to match the line.

---

# Consolidated study material

## Key takeaways

1. Advanced scaling is about transferring an entire training recipe, not merely predicting loss from parameters and tokens.
2. MiniCPM uses muP to make sensitive hyperparameters, especially the base learning rate, more stable across width.
3. Critical batch size can still depend on the target loss, so not every optimum becomes invariant.
4. Warmup-stable-decay schedules make partial runs reusable for multiple token budgets and scaling experiments.
5. Chinchilla-style fits remain useful, but their answer depends on model family, data, schedule, and the range being fit.
6. DeepSeek and StepFun explicitly sweep learning rate and batch size, then fit how the optima drift with scale.
7. Published scaling exponents are priors, not universal constants; rerun the sweep when architecture, data, or optimizer changes materially.
8. Optimizer comparisons require optimizer-specific tuning and more than one scaling axis.
9. A straight scaling line over a narrow compute range can bend outside that range.
10. Muon orthogonalizes matrix momentum spectrally with Newton-Schulz iterations and can work in large-scale training.
11. Kimi K2 establishes Muon's feasibility at scale, not superiority to a matched AdamW baseline.
12. muP targets stable order-one activations and order-one feature updates as width changes.
13. The simplified muP derivation couples initialization, per-layer learning rates, output scaling, and parameter type.
14. Baseline subtraction and asymptotic norm arguments are only as reliable as their assumptions; transfer must be tested empirically.
15. muP often transfers through practical deviations, but trainable RMSNorm gains, Lion, and strong decoupled weight decay can break it.
16. No current method removes all scaling risk; empirical laws and scale-aware parameterization are complementary tools.

## Key equations

### Generic loss scaling law

$$
L(N,D)
\approx
E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

### Compute proxy

$$
C\propto ND.
$$

### DeepSeek-style fitted optimum

$$
B_{\mathrm{opt}}(L)
\propto
L^{-a},
\qquad
\eta_{\mathrm{opt}}(N)
\propto
N^{-b},
$$

where the constants are empirical and recipe dependent.

### Muon idealized spectral update

$$
B_t=U\Sigma V^\top
\quad\longmapsto\quad
O_t=UV^\top.
$$

### muP activation conditions

$$
h_{l,i}=\Theta(1),
\qquad
\Delta h_{l,i}=\Theta(1).
$$

### muP initialization scale

$$
\sigma_l
=
\Theta\!\left[
\frac{1}{\sqrt{n_{l-1}}}
\min\!\left(
1,\sqrt{\frac{n_l}{n_{l-1}}}
\right)
\right].
$$

### Desired muP update norm

$$
\|\Delta W_l\|_*
=
\Theta\!\left(
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\right).
$$

### Simplified per-layer learning-rate scales

$$
\eta_l^{\mathrm{SGD}}
=
\Theta\!\left(\frac{n_l}{n_{l-1}}\right),
\qquad
\eta_l^{\mathrm{Adam}}
=
\Theta\!\left(\frac{1}{n_{l-1}}\right).
$$

## Glossary

- **Aspect ratio:** The relationship among width, depth, head count, and other architectural dimensions as model size changes.
- **Chinchilla scaling:** Compute-optimal allocation of model parameters and training tokens under a fitted loss law.
- **Critical batch size:** The batch size beyond which additional examples yield diminishing optimization-speed benefits.
- **Decoupled weight decay:** Parameter shrinkage applied separately from the loss gradient, as in AdamW.
- **Feature learning:** Non-vanishing change in hidden representations during training, contrasted with a frozen-kernel limit.
- **Fan-in:** Number of input coordinates to a linear layer.
- **Fan-out:** Number of output coordinates from a linear layer.
- **Hyperparameter drift:** Movement of an optimal learning rate, batch size, or other choice as scale changes.
- **IsoFLOPs curve:** Loss measurements across model and data allocations at a fixed compute budget.
- **Learning-rate transfer:** Reusing a tuned base learning rate, after prescribed scaling transformations, at a larger width.
- **Lion:** A sign-based optimizer used in the muP stress tests.
- **muP:** Maximum update parameterization, a coupled set of scaling rules intended to preserve feature-learning behavior across width.
- **mu-transfer:** Tuning hyperparameters on a small muP base model and transferring them to a wider target.
- **Muon:** An optimizer that applies momentum and approximate spectral orthogonalization to matrix parameters.
- **Newton-Schulz iteration:** A matrix-multiplication procedure used to approximate Muon's orthogonalized update.
- **NTK regime:** A width limit in which hidden features change negligibly and training resembles kernel regression.
- **Operator norm:** The largest amount by which a matrix can stretch a unit vector; for the Euclidean norm it is the largest singular value.
- **Scaling ladder:** A sequence of smaller models used to predict choices or outcomes for a target model.
- **Spectral orthogonalization:** Replacing a matrix update's nonzero singular values by a common scale.
- **Standard parameterization:** Conventional initialization and global-learning-rate choices used as the comparison to muP.
- **Token-to-parameter ratio:** Training tokens divided by model parameters, an independent scaling axis from total compute.
- **WSD:** Warmup-stable-decay, a schedule with a reusable stable phase and an attached final decay.

## Self-check questions

1. Why is predicting loss insufficient to specify a safe large-model training run?
2. Which MiniCPM parameter and optimizer choices are coupled under muP?
3. Why can the optimal learning rate transfer while critical batch size still drifts?
4. How does WSD make a partially trained run reusable at several horizons?
5. What assumptions make a Chinchilla-style compute-optimal fit recipe dependent?
6. Why do DeepSeek and StepFun fit learning rate and batch size rather than assume them constant?
7. When is a published hyperparameter exponent a useful prior, and when should it be re-estimated?
8. Why must AdamW and Muon each receive their own tuning budget in a fair comparison?
9. What are the two independent axes of optimizer scaling discussed in the lecture?
10. How can a clean power-law line bend outside the measured compute range?
11. What geometric operation does Muon apply to a matrix momentum update?
12. Why are Newton-Schulz iterations better suited to accelerators than a full SVD?
13. Which claim about Muon does Kimi K2 establish, and which claim remains unsupported?
14. What are muP's A1 and A2 invariants?
15. Why does coordinatewise $\Theta(1)$ correspond to vector norm $\Theta(\sqrt{n_l})$?
16. How is the A1 initialization scale derived from fan-in and fan-out?
17. What three terms appear when the post-update activation is expanded?
18. Which assumptions in the A2 derivation are heuristic?
19. Why does the simplified SGD rule use fan-out divided by fan-in?
20. Why does Adam receive a different layerwise scaling?
21. What does it mean to transfer a base learning rate in muP coordinates?
22. Which modern Transformer components were included in the muP stress tests?
23. Which tested choices broke learning-rate transfer?
24. Why might decoupled weight decay violate a coupled scale invariant?
25. How do fitted scaling laws and muP complement one another?
26. What safeguards would you include before trusting a target-scale extrapolation?

## Source coverage checklist

| Topic | Transcript lines | Coverage note |
|---|---:|---|
| 1. Practical advanced scaling | 1-150 | Motivation, public-report landscape, and two approaches to sensitive hyperparameters |
| 2. MiniCPM and muP | 151-323 | Model context, coupled muP choices, scaling ladder, and study targets |
| 3. Stable learning rate | 324-359 | Learning-rate invariance evidence |
| 4. Critical batch size | 360-425 | Target-loss dependence and residual drift |
| 5. WSD schedules | 426-604 | Reusable horizons, decay attachment, and comparisons |
| 6. MiniCPM Chinchilla fit | 605-680 | Replication method and limitations |
| 7. DeepSeek hyperparameter laws | 681-841 | Batch and learning-rate sweeps and fitted drift |
| 8. DeepSeek WSD and IsoFLOPs | 842-941 | Schedule variant, allocation fit, and final prediction |
| 9. Newer report decisions | 942-1223 | Architecture, data, optimizer, and hyperparameter uses |
| 10. Post-training scaling | 1224-1296 | Why the clean pretraining recipe does not transfer directly |
| 11. StepFun search | 1297-1465 | High-resolution learning-rate and batch-size grid |
| 12. StepFun trends | 1466-1629 | Empirical drift, transfer, and data dependence |
| 13. Reuse or rerun | 1630-1725 | Conditions for borrowing a published scaling relation |
| 14. Optimizer-specific tuning | 1726-1866 | Fair comparisons and tuning budgets |
| 15. Two scaling axes | 1867-1969 | Compute scale and tokens per parameter |
| 16. Bending lines | 1970-2051 | Extrapolation failure and regime change |
| 17. Muon mechanism | 2052-2201 | Matrix structure, momentum, spectral update, and Newton-Schulz |
| 18. Muon evidence | 2202-2282 | NanoGPT trend, Kimi K2 feasibility, and missing Adam ablation |
| 19. Muon Q&A | 2283-2405 | SVD cost, layerwise methods, and hyperparameter interactions |
| 20. muP objective and evidence | 2406-2538 | Learning-rate invariance, CerebrasGPT, and conceptual sources |
| 21. muP invariants | 2539-2621 | Stable activations and feature learning |
| 22. A1 derivation | 2622-2745 | Gaussian matrix norm and initialization scale |
| 23. A2 derivation | 2746-2961 | Rank-one update, activation expansion, and learning-rate scale |
| 24. muP recap | 2962-3060 | Practical scales and the invariance-design method |
| 25. muP stress tests | 3061-3183 | Successful transfer, real-model deviations, and failures |
| 26. Scaling in the wild | 3184-3233 | Practical uncertainty and partial solutions |

**Transcript accounting:** the 26 spans cover lines 1-3233 exactly once, with no gaps or overlaps.

**Slide accounting:** all 58 slides were rendered and visually inspected. Slides were used only for verification or were explicitly separated under Source reconciliation.
