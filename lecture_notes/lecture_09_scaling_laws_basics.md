---
title: "Lecture 9 - Scaling Laws: Basics"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 9
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 9 Scaling Laws.txt"
slide_deck: "../lecture_09.pdf"
status: "complete"
---

# Lecture 9: Scaling Laws - Basics

## How to read these notes

Each substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a concise paraphrase of the spoken lecture. It preserves substantive claims, examples, qualifications, numerical values, and questions while removing filler and repetition.
2. **Additional explanation.** This adds independent intuition, derivations, or study guidance. It is not presented as part of the lecture transcript.

The raw transcript's physical line spans are shown for auditability. The complete 57-slide deck was inspected after the transcript map was established. Slides were used to verify paper names, plots, equations, and notation. Where the speech and slides differ materially, a source-reconciliation note records the difference.

## Lecture map

The lecture develops scaling laws as an engineering method for making expensive decisions using cheaper experiments:

1. Connect modern neural scaling to classical learning curves and explain why power laws appear naturally in estimation.
2. Study data scaling, including mixture selection, repetition, finite data, filtering, and the limitations of extrapolation.
3. Use scaling experiments to choose architectures, optimizers, model shapes, mixture-of-experts sparsity, batch sizes, and learning rates.
4. Derive the model-data compute tradeoff, compare Kaplan with Chinchilla, and use their disagreement to show why reliable scaling laws must be engineered carefully.

The following lecture temporarily returns to systems and inference. A later lecture resumes more advanced scaling topics, including parameterizations and optimizers.

---

# Part I - What scaling laws are and where they came from

## 1. The high-stakes engineering problem

**Transcript coverage:** lines 1-361

### What the lecturer said - transcript only

Imagine receiving 10,000 B200 GPUs for one month and being asked to build a strong open language model. Assume that the infrastructure team and pretraining data already exist. The remaining problem is choosing what to train: architecture, shape, optimizer, learning rate, batch size, data amount, and many other hyperparameters. A frontier run can cost millions of dollars, so tuning these choices directly at full scale is prohibitively wasteful.

One option is to copy established practice. That is reasonable when the goal is merely to reproduce a competitive recipe. It is insufficient when trying to improve on the frontier, because copying existing decisions cannot by itself discover something better.

A **scaling law** is a simple predictive relationship that connects performance or behavior at small scale to performance at large scale. The desired workflow is to optimize with many inexpensive small runs, identify a robust regularity, and extrapolate it to the one expensive run. In large laboratories, this can become more than a formula: it is a paradigm or working philosophy built around taking scale and prediction seriously.

The lecture adopts an engineering view. A useful scaling law is not merely a descriptive curve. It should allow model builders to choose a design at small scale and have justified confidence in its large-scale consequence.

### Additional explanation

The method replaces one unaffordable experiment with a family of cheaper experiments plus an extrapolation assumption:

```text
candidate recipes
    -> controlled small runs across several scales
    -> fit or identify a stable relationship
    -> estimate uncertainty and test held-out scales
    -> choose the large-run recipe
```

The extrapolation assumption is the risky step. More points do not help if all points lie in the wrong regime, use inconsistent hyperparameters, or measure a target that does not transfer to the real goal.

## 2. Classical roots: sample complexity and learning curves

**Transcript coverage:** lines 367-940

### What the lecturer said - transcript only

Scaling laws are closely related to classical machine-learning questions. Generalization theory asks how a learned model's population error differs from its training error, often with a bound that depends on sample size and model complexity. Such bounds describe an upper limit rather than the realized error, but they already ask how performance changes as data grows.

Empirical learning curves have a long history. A 1993 Bell Labs paper by Corinna Cortes, L. D. Jackel, Sara Solla, Vladimir Vapnik, and John Denker proposed training classifiers on small samples, fitting how error decays, and predicting the performance of an expensive full-data run. This is effectively a data scaling law.

Later NLP work argued that additional corpus data could be more valuable than further algorithm engineering when systems had not approached an asymptote. Banko and Brill showed predictable improvements over orders of magnitude of data. Kolachina and collaborators compared several curve families for machine translation and found power-law forms useful.

The lecturer highlights Hestness et al. (2017) as an especially forward-looking origin of modern neural scaling. Across machine translation, language modeling, and speech, it found regular power-law regions and discussed several consequences now associated with large models:

- apparent capability emergence can arise because accuracy is more discontinuous than loss;
- predictable data scaling makes compute a central resource;
- if computation converts into model quality, systems speed converts into accuracy.

These ideas predate the best-known OpenAI scaling papers. The lecturer uses this history to show that the modern language-model era did not invent learning curves or the idea of forecasting performance from smaller runs.

### Source reconciliation

The automatic transcript garbles several names. The slides verify **Banko and Brill (2001)**, **Kolachina et al. (2012)**, and **Hestness et al. (2017)**.

### Additional explanation

Theory and empirical scaling answer related but different questions:

- A generalization bound often says that, with high probability, an error is no worse than some expression.
- An empirical scaling law estimates the error actually observed for a specific data distribution, model family, optimizer, and training recipe.

The empirical law can be much tighter and more useful for engineering, but it has a narrower domain of validity. Changing the recipe or distribution can invalidate it.

## 3. Are power laws theoretically justified?

**Transcript coverage:** lines 943-1042

### What the lecturer said - transcript only

An audience member asked whether the historical scaling curves were only empirical fits or had a deeper justification. The lecturer answered that the published curves are fundamentally curve-fitting exercises. There is no golden rule requiring one particular family.

Theory nevertheless provides plausible functional forms because it studies how errors decay with data, model size, or approximation resolution. Physics also supplies useful habits: examine limits and look for simple asymptotic relationships. Power laws are natural candidates, but they must still be validated rather than assumed.

### Additional explanation

A power law has the form

$$
y(x)=A x^{-\alpha}+E,
$$

where $E$ is an asymptotic floor. When $A x^{-\alpha}\gg E$,

$$
\log(y-E)=\log A-\alpha\log x,
$$

so the relationship is linear on log-log axes. This algebra explains how to recognize a power law, not why a particular learning system must obey one.

## 4. The broad landscape of neural scaling relationships

**Transcript coverage:** lines 1045-1243

### What the lecturer said - transcript only

Power-law-like relationships appear for a surprising variety of quantities. Common pretraining plots place the logarithm of compute, dataset size, or non-embedding parameter count on the horizontal axis and loss on the vertical axis, producing an approximately linear trend in the relevant regime.

More exotic measurements can also look regular. Downstream accuracy often follows a sigmoid-like curve as compute grows because an accuracy metric has lower and upper bounds. Forecasting work sometimes places calendar date on the horizontal axis and fits the upper envelope of model capabilities. These relationships are not guaranteed, but language-model performance often looks much more regular as a function of resources than one might expect.

The lecture next starts with data because it is the simplest univariate case, then moves to more difficult engineering quantities such as architecture and hyperparameters.

### Additional explanation

Different axes imply different models:

- linear loss against log resource is not the same as log loss against log resource;
- accuracy cannot continue as an unbounded power law because it saturates;
- an upper envelope over calendar time combines many hidden changes in compute, algorithms, data, and measurement.

Before fitting, write the equation represented by the axes. Visual straightness alone is not a model specification.

---

# Part II - Data scaling laws

## 5. The basic data scaling experiment

**Transcript coverage:** lines 1249-1448

### What the lecturer said - transcript only

Fix the model family and training procedure, choose a model large enough not to be the limiting factor, and increase the amount of training data. With good hyperparameter tuning, error should improve approximately monotonically. The full curve begins near random guessing in a very small-data regime, enters a broad power-law region, and eventually approaches an irreducible or model-class error floor.

In the power-law region, language-model test loss plotted against dataset size on log-log axes forms a remarkably straight line. The lecturer uses the Kaplan et al. data-scaling plot as the main example. A straight log-log relationship implies polynomial decay and usually indicates that the measurements are still far from the asymptote; close to the floor, the curve must bend.

Students will construct a similar plot in Assignment 3.

### Additional explanation

A practical model is

$$
L(D)=L_\infty+A D^{-\alpha},
$$

where:

- $D$ is the number of training examples or tokens;
- $L_\infty$ is the irreducible or model-limited floor;
- $A$ sets the vertical offset;
- $\alpha$ determines the rate of improvement.

Fitting a straight line to $\log L$ without accounting for $L_\infty$ is valid only when the floor is negligible. Otherwise the estimated slope changes as the curve approaches saturation.

## 6. Why polynomial error decay is natural: mean estimation

**Transcript coverage:** lines 1450-1617

### What the lecturer said - transcript only

To show how a scaling law can arise from elementary statistics, the lecturer temporarily replaces language modeling with Gaussian mean estimation. Let

$$
x_1,\ldots,x_n\sim\mathcal{N}(\mu,\sigma^2),
\qquad
\hat\mu=\frac{1}{n}\sum_i x_i.
$$

The expected squared error is

$$
\mathbb{E}\left[(\hat\mu-\mu)^2\right]=\frac{\sigma^2}{n}.
$$

Taking logs gives

$$
\log \operatorname{Error}=-\log n+2\log\sigma.
$$

This is a scaling law with slope $-1$ on log-log axes. More generally, any error proportional to $n^{-\alpha}$ becomes linear after taking logarithms. Classical parametric estimators often have a $1/n$ rate, or a dimension-dependent constant such as $d/n$.

### Additional explanation

The example separates two properties:

- the **slope** $-1$ is the statistical convergence rate;
- the **intercept** $2\log\sigma$ reflects how noisy the problem is.

Changing data quality can improve the intercept without changing the asymptotic exponent. This distinction later motivates the claim that many interventions shift a scaling curve vertically but leave its slope nearly unchanged.

## 7. Why neural exponents are much smaller

**Transcript coverage:** lines 1618-1885

### What the lecturer said - transcript only

Observed neural scaling exponents are often around $-0.1$ to $-0.3$, much slower than the $-1$ rate of simple parametric estimation. One possible intuition comes from nonparametric regression.

Suppose the task is to estimate an arbitrary smooth function in a unit box. In a two-dimensional toy estimator, divide the space into boxes of side length $n^{-1/4}$. There are roughly $\sqrt n$ boxes and roughly $\sqrt n$ samples per box, yielding an error on the order of $n^{-1/2}$ plus smoothness terms. In a simplified $d$-dimensional picture, the rate can look like

$$
\operatorname{Error}\approx n^{-1/d}.
$$

This suggests a mental model in which a neural network with exponent near $-0.1$ behaves, in rate terms, like a flexible nonparametric estimator in roughly ten dimensions.

Bahri and other scaling-theory researchers have argued more strongly that neural exponents reveal an intrinsic dimension and nonparametric smoothing behavior. The lecturer treats this as interesting but not airtight. Estimates of intrinsic dimension can be unreliable, so he does not fully endorse the literal interpretation.

### Additional explanation

The simple $n^{-1/d}$ expression illustrates the curse of dimensionality but is not a universal nonparametric rate. Actual minimax rates depend on smoothness, dimension, loss, noise, and estimator class. The safe lesson is qualitative: flexible function classes can improve much more slowly with samples than fixed-dimensional parametric models.

An exponent is therefore descriptive evidence about the observed rate. Turning it into a claim about intrinsic dimension requires additional assumptions that the curve alone does not establish.

## 8. Staying in the power-law regime

**Transcript coverage:** lines 1891-2008

### What the lecturer said - transcript only

An audience member asked what it means for the model to be "larger than the dataset." The lecturer did not mean a strict one-to-one inequality between parameters and tokens. The practical requirement is that the model should not become the bottleneck over the studied range.

If a model is small relative to the data, it eventually fits as well as that model class allows. Additional data then produces little improvement, and the curve enters a model-limited or irreducible-error regime. A simple straight power law no longer applies.

Scaling studies can either remain in the clean power-law region or fit the asymptotic floor explicitly. As a rough experimental choice, making the model about ten times larger than the amount needed for the studied data can help keep the data experiment away from model limitation, but the relevant criterion is the observed regime rather than a universal ratio.

### Additional explanation

Confounding is the central danger. If dataset size and model limitation change together, the measured "data exponent" is not purely a data exponent. A pilot should therefore include model-size checks: increasing capacity at the same data size should not materially lower the loss in the range used for the data fit.

## 9. Data mixtures and distribution shift

**Transcript coverage:** lines 2011-2341

### What the lecturer said - transcript only

A one-source data scaling law mainly tells how quickly a model learns. Engineering also requires deciding which data to use: news versus Wikipedia, high quality versus broad coverage, or one domain versus another.

In many classical models, changing the data distribution affects the offset of a learning curve more than its slope. The model class largely determines the exponent, while the mixture determines the intercept. A toy linear-regression example shows that either source alone can be poor and a diverse mixture can minimize the intercept.

In principle, one can train small models over multiple mixture proportions and scales, fit how mixture affects loss, and extrapolate the mixture optimum to the full run. This is the idea behind data-mixing-law work.

In practice, the measurements are much noisier than the ideal picture. The lecturer says that experienced practitioners often train many small models, choose the best small-scale mixture, and scale it directly without fitting a formal scaling law. The DataDecide study found this simple procedure worked well. That outcome is consistent with equal slopes: if mixtures differ only by intercept, their ordering does not change with scale.

### Additional explanation

Let mixture choice be $q$ and suppose

$$
L(D,q)=E+A(q)D^{-\alpha}.
$$

If $\alpha$ is constant across $q$, minimizing $A(q)$ at small scale also minimizes loss at large scale. If exponents differ, curves can cross and the best mixture can change.

Real mixture studies face additional complications: one mixture may alter the tokenizer distribution, gradient variance, domain transfer, or effective number of unique examples. These effects can make the surface noisier than a one-dimensional power law.

## 10. Repeating finite data

**Transcript coverage:** lines 2347-2422

### What the lecturer said - transcript only

Compute is increasing faster than the supply of unique high-quality data, so repeated epochs matter. The study *Scaling Data-Constrained Language Models* found that, with standard recipes, repetition up to roughly four epochs was almost as useful as fresh data. Beyond that point, returns diminished rapidly.

With substantial repetition, the realized loss curve becomes worse than the curve predicted under unlimited fresh data. A modified scaling-law form can describe the effective contribution of repeated examples.

### Source reconciliation

The slide supplies the explicit effective-data formula that the speech only describes:

$$
D'=U_D+U_D R_D^*\left(1-e^{-R_D/R_D^*}\right),
$$

where $U_D$ is unique data, $R_D$ is repetition, and $R_D^*$ is a fitted saturation constant. This equation belongs to the slide source, not to the transcript-only paraphrase.

### Additional explanation

The formula treats repeated tokens as having diminishing marginal value. For small repetition, the exponential can be approximated linearly, so another pass behaves somewhat like new data. At large repetition, the extra effective data saturates near $U_D R_D^*$ even though raw tokens processed continue growing.

The result is recipe-dependent. Stronger regularization, data augmentation, changed ordering, or altered optimization could change the repetition curve.

## 11. Infinite compute, ensembling, and mostly unchanged slopes

**Transcript coverage:** lines 2428-2535

### What the lecturer said - transcript only

The lecturer describes recent work asking what can be extracted from a fixed dataset with effectively unlimited compute. Simply increasing epochs or increasing model size has diminishing returns. To improve further, the system may use regularization, multiple independently trained models, or ensembles.

These interventions improve attainable loss, but their scaling curves often have surprisingly similar slopes. Once students fit their own curves, they will repeatedly see this pattern: interventions frequently change the intercept but not the exponent.

Scaling laws should therefore be read as lower bounds on what one particular recipe achieves. If a new method improves the recipe, performance can move below the old curve even if its slope remains similar.

### Additional explanation

Calling the old curve a lower bound is informal because lower loss is better. More precisely, it is a forecast for a specified procedure, not a fundamental impossibility result. An algorithmic improvement can lower the curve; it need not violate anything.

Similar slopes imply a roughly constant multiplicative resource advantage. If

$$
L_1=A_1D^{-\alpha},\qquad L_2=A_2D^{-\alpha},
$$

then the data required to reach the same loss differs by a constant factor $(A_1/A_2)^{1/\alpha}$.

## 12. Data filtering must adapt to scale

**Transcript coverage:** lines 2536-2692

### What the lecturer said - transcript only

Optimal data filtering depends on the compute budget. A small research run cannot consume the whole internet, so it should retain only the highest-quality subset. A very large run exhausts that subset and would otherwise repeat it excessively. At larger scale, the filter should become less aggressive and admit broader, lower-quality pools.

Data quality and filtering are therefore not fixed properties independent of scale. The optimal pool changes with the number of training samples the system can afford and with the declining value of repetition.

The lecturer closes the data section by emphasizing its relative simplicity: more data produces predictable improvement over a broad power-law region. The existence of that regularity is uncontroversial; the unexpectedly small exponent is the more intriguing part.

### Additional explanation

Filtering solves a quality-quantity allocation problem. If pool $A$ has higher value per first exposure than pool $B$, the optimum may still include $B$ after repeated passes over $A$ become less valuable than a first pass over $B$.

This means a fixed classifier threshold is generally not compute-optimal. A data pipeline should consider each bucket's quality, size, duplication, domain contribution, and expected number of epochs at the target budget.

## 13. A question about axes and functional-form ambiguity

**Transcript coverage:** lines 2695-2809

### What the lecturer said - transcript only

An audience member asked why one infinite-compute plot looked linear rather than log-log. The apparent linear axis was misleading: the dataset values doubled, so the horizontal spacing represented a log scale even though the labeling did not make that obvious. The loss range was so narrow that linear and logarithmic vertical axes looked nearly identical. The fitted functions were still power laws.

The lecturer uses this to warn that a narrow experimental range cannot distinguish a polynomial from an exponential or another smooth family. A Taylor approximation makes almost every smooth curve look linear when zoomed in far enough. Functional-form claims therefore require wide scale ranges and skepticism.

### Additional explanation

A useful fit should report more than $R^2$ on the observed points. Check:

- held-out extrapolation at a larger scale;
- residual structure rather than only aggregate error;
- sensitivity to subtracting an asymptotic floor;
- alternative plausible curve families;
- uncertainty in the exponent under different fit ranges.

Extrapolation quality, not in-range visual straightness, is the relevant standard.

---

# Part III - Scaling laws for model engineering

## 14. Comparing architectures without training frontier models

**Transcript coverage:** lines 2815-3232

### What the lecturer said - transcript only

Model scaling turns architectural questions into controlled small-scale comparisons. To ask whether Transformers are better than LSTMs, the brute-force approach would train a GPT-3-scale LSTM. The scaling approach trains both families at several smaller compute or parameter scales and compares their curves.

In the displayed comparison, LSTMs have a worse intercept and possibly a worse slope than Transformers. A worse slope is especially concerning because the gap grows with scale or eventually reverses an apparent small-scale advantage. This supports scaling the Transformer rather than the LSTM.

Modern architecture papers use this pattern. Mamba, Gated DeltaNet, and related work compare a proposed architecture against a Transformer baseline across scales. A convincing intervention should remain better or scale at least as favorably.

The lecturer highlights Tay et al.'s broad study of T5-style architectures. Although its experiments were far smaller than current frontier runs, its scaling trends anticipated several modern choices. Performer-style efficient attention did not scale favorably; gated linear units did; Switch Transformers generally had good trends despite an anomalous largest run; mixture of softmax also looked promising even though it is not standard today.

This historical correspondence helps explain the strong laboratory belief that an intervention that does not appear in a scaling curve is unlikely to be valuable at the frontier.

### Additional explanation

Architecture comparisons must hold the resource definition constant. Matching parameter count can favor a computationally expensive architecture; matching FLOPs can favor one with unusually high memory traffic; matching wall time can depend on implementation quality. Report at least loss versus compute and realized hardware cost when systems behavior differs materially.

A favorable intercept with an unfavorable slope creates a crossover point. If

$$
L_A=aC^{-\alpha},\qquad L_B=bC^{-\beta},
$$

the preferred model depends on the target compute whenever $\alpha\neq\beta$.

## 15. Optimizer choice, depth, and scale-invariant shape

**Transcript coverage:** lines 3238-3484

### What the lecturer said - transcript only

The same method can compare optimizers. In a Hestness experiment on recurrent highway networks, Adam and SGD change the intercept substantially but have very similar slopes. The lecturer finds this repeated slope stability surprising given how different the optimizers feel in practice.

Depth can also be studied across scale. One-layer models have dramatically worse scaling. Moving beyond one layer provides a large gain; additional depth is competitive and, in the displayed experiments, more layers help at each compute level, though with diminishing differences.

Raw layer count is not scale-invariant because larger models generally need more layers. An **aspect ratio**, such as model width per layer, is a more plausible invariant. Kaplan's sweeps show a broad optimum near roughly 100 units of model width per layer, with some shift toward smaller ratios for deeper models. Feed-forward ratio and attention-head dimension can be analyzed similarly.

These plots justify a scaling strategy in which an approximately optimal shape is held fixed while overall depth and width grow.

### Additional explanation

The broad minima matter as much as their exact locations. If many shapes are within a few percent of the best loss, systems efficiency, parallelism, memory, and inference latency can decide among them. Reporting one precise optimum can create false confidence when the loss surface is nearly flat.

## 16. Parameter counting is part of the model

**Transcript coverage:** lines 3487-3631

### What the lecturer said - transcript only

Not all parameters behave alike, so a scaling plot depends on what counts as a parameter. Kaplan's depth curves looked irregular when embedding parameters were included. The authors removed embedding parameters and plotted only non-embedding parameters, which produced cleaner regularity. They justified this by treating the remaining parameters as the ones performing repeated computation.

This choice later has nontrivial consequences for compute-optimal conclusions. The example supports a broader point attributed to Percy: predictable scaling is **engineered**. Researchers must choose meaningful axes, tune hyperparameters consistently, and locate the correct regime. Scaling laws do not automatically emerge from arbitrary measurements.

### Additional explanation

Parameter count is a proxy. Two parameters can differ in:

- how often they are used per token;
- how many FLOPs they induce;
- whether they are active for every example;
- how much memory bandwidth they require;
- whether they scale with vocabulary rather than model depth.

The proxy should match the decision. Training compute may favor active parameters; memory capacity may require total parameters; serving cost may depend on both plus activation traffic.

## 17. Scaling mixture-of-experts sparsity

**Transcript coverage:** lines 3634-3808

### What the lecturer said - transcript only

Mixture-of-experts models make parameter accounting more complicated because total parameters and active parameters are decoupled. The lecture cites a study from Apple and MIT that fits loss as a function of both quantities and sparsity.

At fixed compute, increasing total parameters while holding active parameters roughly fixed can improve loss. Parameters belonging to experts that are inactive for one token still help the overall model because different tokens can use them. As model and compute budgets grow, the compute-optimal design in the displayed surfaces becomes increasingly sparse.

IsoFLOP surfaces allow a designer to ask how much compute to spend on active capacity and how much total conditional capacity to add. These choices show predictable regularities rather than requiring one giant run for every configuration.

### Additional explanation

For dense models, total and active parameter counts are nearly the same. For an MoE,

$$
N_{\text{active}}\ll N_{\text{total}}.
$$

Training FLOPs correlate more closely with active parameters, while memory, checkpoint size, and sometimes communication depend on total parameters. A two-axis scaling surface is therefore more informative than forcing MoEs onto a dense-model one-axis law.

## 18. Critical batch size

**Transcript coverage:** lines 3814-4468

### What the lecturer said - transcript only

Batch size and learning rate are two choices that every new large run must revisit. Large batches are desirable for data parallelism, but beyond a certain point they process more examples without proportional optimization progress.

Below the **critical batch size**, optimization is noise-limited. Adding examples reduces gradient variance, and the larger batch produces nearly perfect returns. Above it, the optimization becomes bias-limited by the local descent direction and objective geometry. Further variance reduction no longer fixes the mismatch between a local gradient direction and the path toward the broader optimum, so returns diminish.

The critical batch is a convenient crossover rather than an abrupt physical boundary. It aims to make the batch as large as possible for parallelism without accepting severe sample inefficiency.

To estimate it:

1. Choose a target loss.
2. Train with several batch sizes.
3. For each run, record steps $S$ and examples $E$ required to reach the target. They satisfy $E=SB$.
4. Fit the empirical tradeoff using minimum possible steps $S_{\min}$ and minimum possible examples $E_{\min}$:

$$
\frac{S}{S_{\min}}-1
=
\left(\frac{E}{E_{\min}}-1\right)^{-1}.
$$

5. Choose

$$
B_{\mathrm{crit}}=\frac{E_{\min}}{S_{\min}}.
$$

This balances the step and example terms, using somewhat more of each than its separate optimum. A related estimate uses the ratio of the trace of gradient covariance to the squared gradient norm.

Critical batch size grows predictably as target loss decreases, approximately as an inverse power law. Better models can therefore use larger batches efficiently. This is favorable for large-scale systems and intuitively sensible: near a fine optimum, gradient noise matters more.

### Additional explanation

There are two efficiencies:

- **step efficiency:** fewer sequential optimizer updates;
- **sample efficiency:** fewer examples processed.

Large batches improve the first and harm the second after saturation. Hardware throughput introduces a third objective: a larger batch may execute each example more efficiently even when it is statistically less efficient. Production batch selection should minimize wall-clock time or cost to a target loss, not one abstract quantity alone.

## 19. Learning-rate scaling and muP

**Transcript coverage:** lines 4471-4687

### What the lecturer said - transcript only

Under standard width scaling of a neural network, the optimal learning rate generally decreases as the model becomes wider. A common rule of thumb is inverse-width scaling: changing more parameters at once calls for a smaller global step.

One strategy is to sweep learning rates at several model sizes, fit how the optimum shifts, and extrapolate that optimum to the large model. The shifts are regular enough that this can work.

Another strategy changes initialization and optimizer scaling for different parameter tensors so the optimal learning rate remains stable across width. The lecture calls this the muP family of parameterizations. Some groups report strong success and others less. Both philosophies have been used in large training runs, though the lecturer's anecdotal impression is that more teams may currently favor direct scaling-law prediction.

The advanced scaling lecture will cover both approaches in greater detail.

### Source reconciliation

The transcript's character encoding corrupts the Greek letter in **muP**. The slides verify that the intended term is the maximal-update parameterization, normally written $\mu$P.

### Additional explanation

These approaches solve different versions of the same transfer problem:

- **fit the movement:** accept that the optimum changes and predict its location;
- **remove the movement:** reparameterize the family so one learning rate transfers.

muP still depends on applying the correct tensor-specific scaling rules. A partial or inconsistent implementation can destroy the intended invariance.

## 20. Upstream loss does not guarantee downstream quality

**Transcript coverage:** lines 4690-4852

### What the lecturer said - transcript only

Pretraining perplexity can scale cleanly while downstream performance remains irregular. In a Tay et al. architecture study, model parameter count was strongly correlated with negative log-perplexity, but the architecture with the best perplexity was not the one with the best SuperGLUE accuracy. A model that looked substantially worse upstream could win downstream.

This is an unusually poor example of upstream-to-downstream correlation, but the general warning is important. Scaling laws are cleanest for loss and perplexity. Transfer from that loss to user-facing tasks is less certain.

The lecturer mentions former students working on post-training who complain that pretraining teams hand over a model with good perplexity and treat every later problem as post-training's responsibility. Some problems originate in pretraining and will not be revealed by aggregate perplexity alone. Model builders must study transfer rather than optimize only the clean metric.

### Additional explanation

Perplexity averages token-level log loss over a distribution. A downstream benchmark may depend on rare skills, prompting format, long-range behavior, calibration, or post-training compatibility. Two models with similar average loss can allocate capability differently.

A sensible workflow uses low-variance loss for the main extrapolation and a smaller panel of downstream or diagnostic evaluations to verify that the candidate recipe preserves the needed transfer relationship.

## 21. The scaling-law design procedure and measurement questions

**Transcript coverage:** lines 4855-5212

### What the lecturer said - transcript only

Before a full run, the team should have a numerical prediction of what will happen, not merely confidence that training will not crash. If choosing Adam over SGD is expected to help, the expected loss difference should be estimated.

The basic procedure is:

1. Train several smaller models.
2. Establish a regular scaling relationship for the alternatives.
3. If the fit and extrapolation are credible, choose the predicted large-scale winner.

An audience member asked how many replicates are used for each point. Most perplexity scaling plots use one run per point because variance is very low: training data is large and homogeneous, evaluations are large, and rerun differences may appear only in the second decimal place. Learning-rate and critical-batch experiments can be much noisier. Variance reduction is used less often than it arguably should be.

Another question asked why not fit downstream metrics directly. This is possible, but their noisy or jagged curves create much more uncertainty about functional form. The lecturer's preferred philosophy is to establish at least one low-variance scaling regularity, usually loss, then separately justify or measure transfer to downstream behavior.

A final question asked when to use training versus test loss. Test loss is better scientific practice. However, most large pretraining experiments use effectively one pass over enormous data, so their generalization gap is tiny and train and validation loss are nearly identical. Some codebases omit validation loss for this reason. Repetition and infinite-compute regimes are exceptions where the distinction matters.

### Additional explanation

Low observed variance does not remove all uncertainty. A singleton curve can still have:

- correlated errors from one shared dataset;
- systematic bias from the implementation;
- hyperparameter misspecification at one scale;
- uncertainty in functional form;
- an invalid transfer assumption.

Replicates address random seed variance, not these other errors. Held-out scales and recipe audits may be more valuable than repeating every point, but noisy sweeps should include replication or uncertainty-aware fits.

---

# Part IV - Joint model-data scaling and the Chinchilla case study

## 22. The compute-allocation question

**Transcript coverage:** lines 5224-5455

### What the lecturer said - transcript only

Given a fixed amount of compute, should it be spent on a larger model or on more training tokens? A tiny model trained on enormous data eventually becomes model-limited, wasting later tokens. A very large model trained on too little data is also inefficient.

Joint scaling laws describe loss as a function of model size and data size. Rosenfeld uses a sum of two inverse power laws plus an irreducible constant. Kaplan uses a related coupled form. The limits provide a sanity check:

- with infinite data, the data term vanishes and loss is limited by model size;
- with an infinite model, the model term vanishes and loss is limited by data.

Rosenfeld and others fit these surfaces using small models and small datasets, then accurately predicted points with much larger models and data. Once the surface is known, compute-optimal allocation is a constrained nonlinear optimization: minimize predicted loss subject to the available FLOPs.

### Source reconciliation

The slides verify the illustrative forms as

$$
L(n,m)=n^{-\alpha}+m^{-\beta}+C
$$

for the Rosenfeld-style presentation, and a coupled Kaplan-style expression. The transcript describes the limits but does not read every symbol clearly. These exact displayed formulas are slide-verified rather than reconstructed from the automatic transcript.

### Additional explanation

Using the rough dense-training cost

$$
C_{\text{train}}\approx 6ND,
$$

where $N$ is parameters and $D$ is tokens, the optimization is

$$
\min_{N,D} L(N,D)
\quad\text{subject to}\quad
6ND=C_0.
$$

Substitute $D=C_0/(6N)$ to reduce the problem to one variable. The fitted exponents determine how the optimal $N$ and $D$ grow with compute.

## 23. Kaplan's compute-optimal prescription and the move to Chinchilla

**Transcript coverage:** lines 5458-5734

### What the lecturer said - transcript only

Kaplan's fit assigns very different compute exponents to parameters and data. It implies that, as compute grows, parameter count should grow much faster than token count, so tokens per parameter should decrease. This result helped motivate the era of enormous dense models, including hundreds-of-billions and even proposed trillion-parameter systems.

In 2022, Hoffmann et al.'s Chinchilla work argued that these models were far too large for their training token budgets. It found that a much smaller model trained on substantially more data could achieve better loss at the same compute. Its familiar rule of thumb is roughly 20 training tokens per parameter.

The disagreement is not merely historical trivia. It shows that a scaling law is not a mechanical crank that automatically returns truth. The result is sensitive to experimental design, parameter accounting, training schedules, optimizer settings, and fitting choices.

### Source reconciliation

While speaking, the lecturer momentarily swaps the symbols and then corrects himself. The slides state Kaplan's intended relationship unambiguously:

$$
N_{\mathrm{opt}}\propto C^{0.73},
\qquad
D_{\mathrm{opt}}\propto C^{0.27}.
$$

Here $N$ is model parameters and $D$ is training data. This makes tokens per parameter decrease with compute.

### Additional explanation

If

$$
N_{\mathrm{opt}}\propto C^a,
\qquad
D_{\mathrm{opt}}\propto C^b,
$$

and compute is proportional to $ND$, then $a+b\approx1$. The token-to-parameter ratio scales as

$$
\frac{D}{N}\propto C^{b-a}.
$$

Kaplan has $b-a=-0.46$, so the ratio falls quickly. Chinchilla methods 1 and 2 have $a\approx b\approx0.5$, so the ratio is approximately constant.

## 24. Chinchilla method 1: the training-curve envelope

**Transcript coverage:** lines 5737-5923

### What the lecturer said - transcript only

Chinchilla estimates the compute-optimal model-data tradeoff in three ways. The first takes the lower envelope over many training curves. For any FLOP budget, the envelope identifies the lowest loss reached by any model's curve. Each envelope point belongs to a model with a known parameter count and number of processed tokens.

Plotting the model sizes and token counts selected by those envelope points against compute reveals power-law trends. Extrapolating them to the target Gopher-sized compute budget predicts a model of roughly 67 billion parameters.

This method is conceptually simple but identifying a reliable lower envelope is delicate. Sparse runs, noisy curves, or schedule differences can change which point appears optimal.

### Additional explanation

The envelope uses every intermediate checkpoint, not only terminal losses. It asks, "At compute $C$, which observed training trajectory has achieved the lowest loss so far?" This can be data-efficient because one long run contributes points at many budgets, but it assumes checkpoint costs and schedules are comparable.

## 25. Chinchilla method 2: IsoFLOP profiles

**Transcript coverage:** lines 5926-6041

### What the lecturer said - transcript only

The second method fixes several FLOP budgets. At each budget, it sweeps model size and adjusts the number of tokens inversely so total compute remains fixed. Terminal losses trace a convex, valley-like curve: models that are too small waste data, while models that are too large receive too few tokens.

For each compute budget, fit a quadratic or otherwise locate the minimum. The sequence of minima gives the compute-optimal parameter and token counts as functions of FLOPs. Extrapolation to the target budget predicts about 63 billion parameters, close to method 1.

The lecturer calls IsoFLOP profiling his preferred method because it is simple, robust, and widely reusable.

### Additional explanation

An IsoFLOP experiment converts a two-variable allocation problem into repeated one-dimensional sweeps. Its quality depends on bracketing the valley: if all tested models lie on one side of the minimum, a quadratic fit can produce a fictitious optimum outside the data.

Use geometrically spaced budgets and include enough model sizes at each budget to observe both rising sides of the profile.

## 26. Chinchilla method 3: joint parametric fit

**Transcript coverage:** lines 6043-6137

### What the lecturer said - transcript only

The third method hypothesizes a joint loss function over model size and data, trains models across a grid, and fits all coefficients by regression. It is the most direct implementation of the Rosenfeld/Kaplan idea.

It is also more fragile. Fitting a multivariable surface introduces choices about objective, weighting, optimization, initialization, outliers, and parameter identifiability. Unlike the simple IsoFLOP minima, the result can be quite sensitive to curve-fitting details.

Chinchilla methods 1 and 2 estimate almost equal compute exponents for model size and data: about 0.50/0.50 and 0.49/0.51. Method 3 reports about 0.46/0.54, a difference that becomes important asymptotically.

### Additional explanation

A small exponent difference can look negligible over the observed range yet dominate a long extrapolation. With $a=0.46$ and $b=0.54$,

$$
\frac{D}{N}\propto C^{0.08},
$$

so the token-to-parameter ratio grows rather than staying fixed. Always translate fitted exponents into the downstream quantity of interest before deciding that two fits "mostly agree."

## 27. Why Kaplan and Chinchilla disagreed

**Transcript coverage:** lines 6139-6502

### What the lecturer said - transcript only

Later work reproduced Kaplan's result under Kaplan-like settings and then changed a sequence of seemingly small choices until the estimate matched Chinchilla.

First, parameter counting mattered. Kaplan excluded embedding parameters because they distorted depth scaling, but also excluded the final vocabulary projection because it has a shape dual to the input embedding. Excluding this output computation materially changed the scaling axis.

Second, some Kaplan models were so small that their learning-rate warmup lasted too much of training. They had not properly converged by the end of the relevant regime, so their learning rates were suboptimal.

Third, Kaplan used one large fixed batch size. That batch was inefficient for the smallest models. Tuning batch size across scales shifted the estimate further toward Chinchilla.

Each decision looks minor, yet together they produced a major difference in the compute-optimal frontier. A scaling law predicts what happens if its recipe is continued. Scaling a poorly tuned recipe gives a clean prediction of a poor recipe. Small runs should resemble the intended full run as closely as possible.

### Source reconciliation

The slide title identifies the replication as *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*. The slide summarizes the stages as counting last-layer FLOPs, correcting warmup, and optimizer or schedule tuning. The transcript additionally emphasizes batch-size tuning. The detailed causal attribution across later papers is nuanced; these notes preserve the lecturer's account rather than treating one slide arrow as the only explanation.

### Additional explanation

This is a failure of **scale consistency**. A nominally fixed recipe can change meaning with scale:

- a fixed warmup length can consume most of a tiny run but little of a large one;
- a fixed batch can be far above critical batch for a small model;
- excluded parameter classes can be a large fraction of a small model and a smaller fraction of a large one.

Every hyperparameter should be expressed in a scale-aware unit: fraction of total tokens, multiple of critical batch, ratio to model dimension, or another invariant justified by the experiment.

## 28. A second reconciliation: compute range and parameter definitions

**Transcript coverage:** lines 6505-6634

### What the lecturer said - transcript only

Pearce and Song offered another analysis without training new models. They started from the functional form implied by Chinchilla, generated synthetic training curves for the much smaller scales used by Kaplan, and compared total-parameter with non-embedding-parameter frontiers.

They found two interacting effects. Kaplan operated at a much lower compute range, where local nonlinearities strongly affect a fitted slope. Ignoring embedding-related parameters changed the axis enough to amplify that local effect. The resulting apparent Kaplan exponent can therefore emerge even from Chinchilla-like underlying curves.

The lesson is that reliable scaling requires finding a regime broad and stable enough for the intended extrapolation. Axis definitions and compute range cannot be treated as cosmetic.

### Additional explanation

A power-law exponent is a local slope on a log-log plot. If the true curve bends slightly, different fit windows produce different exponents. Extrapolating a local tangent far beyond its window is dangerous even when in-window residuals are tiny.

## 29. The Chinchilla method-3 refit

**Transcript coverage:** lines 6637-6837

### What the lecturer said - transcript only

One mystery remained inside the Chinchilla paper: method 3 did not agree exactly with methods 1 and 2. Methods 1 and 2 implied a fixed token-to-parameter ratio, while method 3's unequal exponents implied that the ratio should grow with compute.

Researchers at Epoch AI could not obtain the raw data or code, so they extracted data points from the paper's plots and refit the method-3 surface. They found that the original joint model had underfit the displayed data. Their refitted surface achieved smaller residuals and recovered a policy close to the familiar 20-token-per-parameter rule.

The lecturer finds the conclusion amusing: Chinchilla's authors were more right than their own method-3 fit suggested. The only internal disagreement appears to have been a fitting error rather than evidence against methods 1 and 2.

### Source reconciliation

The slide attributes this data-forensics refit to **Besiroglu et al. (2024)**. The automatic transcript does not preserve the name reliably.

### Additional explanation

Digitizing points from a plot is imperfect, but it can reveal gross fitting failures. The episode also highlights reproducibility: releasing raw measurements, fit code, loss functions, and initialization would have made the discrepancy much easier to diagnose.

## 30. Train-optimal is not deployment-optimal

**Transcript coverage:** lines 6841-7003

### What the lecturer said - transcript only

The Chinchilla ratio minimizes loss for a fixed **training** compute budget. A production model has a different objective. Much organizational compute goes to research and development or repeated inference serving rather than the final pretraining run. Serving strongly rewards a smaller capable model.

It can therefore be rational to spend more pretraining tokens on a smaller model than Chinchilla's train-optimal allocation recommends. Such a model is conventionally called "overtrained," but the lecturer argues that this is the right amount of training for its lifecycle objective.

GPT-3 was severely undertrained by modern standards, at roughly a few tokens per parameter. Chinchilla moved the field to about 20. As serving became a major cost, later dense models used much larger token-to-parameter ratios, and MoEs introduced further serving tradeoffs.

Chinchilla matters not because 20 is a universal golden ratio, but because the paper teaches how to fit and audit scaling laws. For a short research run that will not be served extensively, train-optimal allocation may still be appropriate.

### Source reconciliation

The speech says GPT-3 used roughly 3 tokens per parameter, while the slide lists 2. The slide also lists illustrative ratios for LLaMA 65B, Llama 2 70B, Mistral 7B, and Llama 3 70B that were not read aloud. The stable transcript claim is qualitative: GPT-3 used only a few tokens per parameter, Chinchilla used about 20, and later deployment-oriented models used substantially more.

### Additional explanation

The correct lifecycle objective might be

$$
\text{total cost}
=
\text{R\&D cost}
+
\text{training cost}
+
Q\times\text{cost per served token},
$$

where $Q$ is expected usage. As $Q$ grows, spending more once to reduce model size and inference cost becomes increasingly valuable.

## 31. IsoFLOPs as a general research default

**Transcript coverage:** lines 7009-7077

### What the lecturer said - transcript only

Among the Chinchilla methods, IsoFLOP experiments have endured as a general research tool. Fix a compute budget and sweep the remaining degrees of freedom to reveal the loss surface. Repeat at several budgets to see how the optimum moves.

The same design has been used for autoregressive versus diffusion models and for MoE sparsity versus active and total parameters. When facing an unfamiliar resource-allocation tradeoff, the lecturer recommends IsoFLOPs as a strong default.

### Additional explanation

IsoFLOPs are useful because they align every candidate with the actual constraint. They do not require committing to a global joint functional form before seeing the geometry. The method still needs careful FLOP accounting, comparable optimization schedules, and enough sweep points to locate the optimum.

## 32. Final perspective

**Transcript coverage:** lines 7081-7168

### What the lecturer said - transcript only

Language-model performance often has a log-linear regularity with resources. This extends beyond data to model parameters, compute, MoE sparsity, and important hyperparameters. It turns choices that might otherwise be copied or guessed into evidence-driven engineering decisions.

Data scaling has the cleanest conceptual connection to classical learning, although its small exponent is surprising. Model scaling can reduce the cost of architecture and hyperparameter design. More generally, scaling laws provide robust predictors that make engineering at a scale larger than the available experiments possible.

The next lecture covers inference. The course then returns to more advanced scaling topics.

### Additional explanation

The lecture's strongest message is methodological:

1. Choose a low-variance quantity and a resource definition that match the decision.
2. Keep the recipe scale-consistent.
3. Sample a wide enough range to reveal curvature and crossovers.
4. Prefer held-out extrapolation over in-range goodness of fit.
5. Audit downstream transfer and lifecycle cost before deploying the predicted optimum.

---

# Consolidated takeaways

1. Scaling laws use controlled small experiments to predict expensive large-scale outcomes.
2. Empirical learning curves predate neural language models and connect to classical sample-complexity questions.
3. A straight line on log-log axes corresponds to polynomial error decay only after the appropriate asymptotic floor is handled.
4. Neural data exponents around $0.1$ to $0.3$ imply much slower improvement than simple parametric estimation.
5. Data mixture, regularization, and optimizer changes often shift curve intercepts while leaving slopes similar, though this is an empirical pattern rather than a theorem.
6. Repeated data has diminishing value, so filtering and data-pool composition should adapt to the compute budget.
7. Architecture, optimizer, shape, MoE sparsity, batch size, and learning rate can all be studied with scaling experiments.
8. Critical batch size balances sequential-step efficiency against the number of examples required to reach a target loss.
9. Clean pretraining-loss scaling does not guarantee clean downstream capability scaling.
10. Kaplan and Chinchilla differed because small details in parameter accounting, warmup, batch size, compute range, and fitting can materially alter extrapolation.
11. Chinchilla's 20-token-per-parameter rule is a training-compute optimum for one regime, not a universal lifecycle optimum.
12. IsoFLOP sweeps are a robust default for resource-allocation questions because they expose the optimum at fixed cost without requiring one fragile global fit.

# Key equations

## Generic power law with a floor

$$
L(x)=L_\infty+A x^{-\alpha}.
$$

## Mean-estimation scaling

$$
\mathbb{E}\left[(\hat\mu-\mu)^2\right]=\frac{\sigma^2}{n},
\qquad
\log \operatorname{Error}=-\log n+2\log\sigma.
$$

## Simplified nonparametric rate

$$
\operatorname{Error}\approx n^{-1/d}.
$$

## Critical-batch fit

$$
\frac{S}{S_{\min}}-1
=
\left(\frac{E}{E_{\min}}-1\right)^{-1},
\qquad
B_{\mathrm{crit}}=\frac{E_{\min}}{S_{\min}}.
$$

## Dense training compute estimate

$$
C\approx 6ND.
$$

## Illustrative joint scaling law

$$
L(N,D)=L_\infty+A N^{-\alpha}+B D^{-\beta}.
$$

## Kaplan compute-optimal exponents

$$
N_{\mathrm{opt}}\propto C^{0.73},
\qquad
D_{\mathrm{opt}}\propto C^{0.27}.
$$

## Chinchilla methods 1 and 2, approximately

$$
N_{\mathrm{opt}}\propto C^{0.5},
\qquad
D_{\mathrm{opt}}\propto C^{0.5},
$$

which makes $D/N$ approximately constant over the fitted regime.

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Asymptote | The irreducible or model-limited floor approached as a resource becomes large. |
| Bias-limited regime | Large-batch regime where reducing stochastic gradient noise no longer yields proportional progress. |
| Compute-optimal | Minimizing predicted loss under a fixed training-compute constraint. |
| Critical batch size | Crossover batch balancing step efficiency and example efficiency for a target loss. |
| Data scaling law | A predictive relationship between dataset size and model error or loss. |
| Effective data | A discounted count that treats repeated examples as less valuable than fresh examples. |
| Exponent | The power $\alpha$ controlling the slope of a power law on log-log axes. |
| Extrapolation | Predicting behavior outside the measured scale range. |
| Intercept | Vertical offset of a log-linear scaling curve; often changed by data quality or algorithm choice. |
| Intrinsic dimension | A hypothesized effective dimension of the learned problem; its connection to neural scaling exponents is not settled. |
| IsoFLOP profile | A sweep over model or recipe choices while holding total training FLOPs fixed. |
| Learning curve | Performance plotted as a function of data, compute, model size, or another resource. |
| Lower envelope | Best loss achieved by any training curve at each compute budget. |
| muP | Maximal-update parameterization designed to transfer training behavior and learning-rate optima across width. |
| Noise-limited regime | Batch-size regime where more examples substantially reduce gradient variance and improve progress. |
| Non-embedding parameters | Parameters excluding token embeddings and, depending on the convention, possibly the output projection. |
| Nonparametric learning | Flexible estimation whose convergence rate can depend strongly on dimension and smoothness. |
| Overtraining | Training beyond a train-compute optimum to obtain a smaller, more capable, cheaper-to-serve model. |
| Power law | A relationship proportional to a resource raised to a fixed exponent. |
| Scale-invariant quantity | A hyperparameter or ratio whose optimum remains approximately stable as overall model scale grows. |
| Scaling recipe | The full set of architecture, data, optimizer, schedule, batch, and accounting choices being extrapolated. |
| Train-optimal | Optimal under pretraining cost alone, without accounting for R&D or inference. |

# Self-check questions

1. Why is tuning a frontier model directly at frontier scale economically unacceptable?
2. How are empirical scaling laws related to, but different from, generalization bounds?
3. What does a straight line on a log-log plot imply mathematically?
4. Why must a data-scaling experiment avoid the model-limited regime?
5. Derive the $1/n$ mean-estimation scaling law.
6. Why does the nonparametric example produce a dimension-dependent exponent?
7. What evidence would be needed to turn a scaling exponent into a credible intrinsic-dimension claim?
8. Under what condition will the best data mixture at small scale remain best at large scale?
9. Why do repeated examples eventually contribute less than fresh examples?
10. Why should a data filter become less aggressive at a larger compute budget?
11. Why can a narrow scale range fail to distinguish a power law from an exponential?
12. What do the slope and intercept each imply when comparing architectures?
13. Why is parameter count ambiguous for embeddings and MoEs?
14. Explain the noise-limited and bias-limited sides of critical batch size.
15. Why can the critical batch grow as target loss improves?
16. Compare learning-rate extrapolation with the muP strategy.
17. Why is pretraining loss a more regular scaling target than downstream accuracy?
18. Why do singleton loss measurements not eliminate systematic scaling uncertainty?
19. Use the limits of a joint model-data law to explain its two resource-limited regimes.
20. How does an IsoFLOP profile locate a compute-optimal model size?
21. Why did Kaplan's exponents imply falling tokens per parameter?
22. Which recipe and accounting choices helped explain the Kaplan-Chinchilla discrepancy?
23. Why was Chinchilla method 3's small exponent disagreement actually consequential?
24. Why can a production model rationally be trained well beyond the Chinchilla ratio?
25. What checks should precede extrapolating a small-run scaling curve to a frontier run?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-361 | Course transition, 10,000-B200 scenario, motivation, and engineering definition | Yes |
| 367-940 | Generalization bounds, 1993 learning curves, NLP history, and Hestness | Yes |
| 943-1042 | Question about empirical versus theoretical origins | Yes |
| 1045-1243 | Data, compute, parameter, downstream, and calendar-time scaling | Yes |
| 1249-1448 | Data-scaling setup, regimes, and Kaplan log-log observation | Yes |
| 1450-1617 | Gaussian mean-estimation derivation and parametric rates | Yes |
| 1618-1885 | Neural exponents, nonparametric example, and intrinsic-dimension caveat | Yes |
| 1891-2008 | Model-limited regime and audience question about model versus data size | Yes |
| 2011-2341 | Data composition, mixture-law proposal, DataDecide, and equal-slope interpretation | Yes |
| 2347-2422 | Data repetition and diminishing effective data | Yes |
| 2428-2535 | Infinite-compute setting, regularization, ensembling, and stable slopes | Yes |
| 2536-2692 | Scale-adaptive data filtering and data-section recap | Yes |
| 2695-2809 | Audience question about axes and functional-form ambiguity | Yes |
| 2815-3232 | Model engineering, Transformers versus LSTMs, and broad architecture studies | Yes |
| 3238-3484 | Adam versus SGD, depth, aspect ratio, and Transformer hyperparameters | Yes |
| 3487-3631 | Parameter-count definitions and engineered predictability | Yes |
| 3634-3808 | MoE active/total parameters, sparsity, and IsoFLOP surfaces | Yes |
| 3814-4468 | Batch-size motivation, critical-batch derivation, scaling, and interpretation | Yes |
| 4471-4687 | Learning-rate scaling, muP, and two transfer philosophies | Yes |
| 4690-4852 | Upstream perplexity versus downstream performance | Yes |
| 4855-4947 | Practical scaling-law design procedure | Yes |
| 4948-5212 | Questions on replication variance, downstream fits, and train versus test loss | Yes |
| 5224-5455 | Joint model-data laws, limiting cases, extrapolation, and optimization | Yes |
| 5458-5734 | Kaplan allocation, giant-model era, Chinchilla correction, and 20:1 rule | Yes |
| 5737-5923 | Chinchilla method 1: training-curve envelope | Yes |
| 5926-6041 | Chinchilla method 2: IsoFLOP profiles | Yes |
| 6043-6137 | Chinchilla method 3: joint parametric fit | Yes |
| 6139-6502 | Kaplan-Chinchilla discrepancy: counts, warmup, optimizer, and batch | Yes |
| 6505-6634 | Compute-range and parameter-definition reconciliation | Yes |
| 6637-6837 | Chinchilla method-3 disagreement and refit | Yes |
| 6841-7003 | Train-optimal versus serving-optimal allocation and overtraining | Yes |
| 7009-7077 | IsoFLOPs as a general research method | Yes |
| 7081-7168 | Final recap and transition to inference | Yes |
