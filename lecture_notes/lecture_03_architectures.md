---
title: "Lecture 3 - Architectures"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 3
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 3 Architectures.txt"
slide_deck: "../lecture_03.pdf"
status: "complete"
---

# Lecture 3: Architectures

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves the lecturer's claims, examples, qualifications, numerical details, and substantive questions and answers while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, derivation, intuition, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are shown so that the paraphrase can be audited. The slide deck was checked only after the complete transcript map had been established. It was used to verify model names, mathematical notation, diagrams, tables, and figures. Material differences or verbal slips are identified under **Source reconciliation** rather than silently merged into the transcript-grounded account.

## Lecture map

The lecture approaches architecture design as a study of accumulated empirical experience. It has five main parts:

1. Identify the core Transformer choices on which modern dense language models have largely converged.
2. Explain the remaining architectural variation, especially normalization, gated feed-forward networks, and positional encoding with RoPE.
3. Extract practical default hyperparameters for feed-forward width, attention heads, model aspect ratio, vocabulary size, and regularization.
4. Study interventions that make expensive training runs more stable: z-loss, QK normalization, and logit soft-capping.
5. Connect architecture to inference cost through KV caching, multi-query and grouped-query attention, and interleaved local and global attention.

---

# Part I - How to reason about architecture choices

## 1. A survey of collective architectural experience

**Transcript coverage:** lines 1-678

### What the lecturer said - transcript only

Architecture design can feel inscrutable because it is governed by many low-level empirical choices rather than a small set of clean theoretical principles. The best way to acquire architectural judgment is to train models and compare alternatives oneself. That hands-on exploration is part of the course philosophy, but no class has enough time or compute to search the full design space. The second-best method is therefore to learn from the experience encoded in many model releases and technical reports.

The lecture takes that survey approach. It asks which choices remain fixed across successful models and which choices vary without destroying performance. The starting point is the original Transformer: sinusoidal positional encodings, ReLU feed-forward layers, and normalization on the residual path. Assignment 1 instead asks students to implement a modernized variant with normalization before each non-residual computation, rotary positional embeddings, SwiGLU feed-forward layers, and no biases in the linear or normalization layers. Llama helped popularize this bundle, but the same choices appear far beyond the Llama family.

The number of relevant releases makes the comparison difficult but also informative. The previous version of the lecture found more than 19 new dense models, including families such as Qwen, Gemma, InternLM, Nemotron, and others. The current landscape contains fewer purely dense releases but many mixture-of-experts systems. The lecture mentions Qwen 3, Gemma 4, OLMo 3, the Marin 8B model, and many additional open releases. Mixture-of-experts architectures are deferred to the next lecture. Across the dense autoregressive models, a comparison table can track choices such as vocabulary size, normalization, serial versus parallel blocks, positional encoding, activation, head configuration, aspect ratio, regularization, and stability tricks.

The lecture will first cover common architectural building blocks, then descend to ordinary hyperparameters such as feed-forward dimension, vocabulary size, head count, depth, and width, and finally discuss training-stability interventions. These categories interact because an architecture must satisfy several requirements simultaneously:

- learn from data and generalize;
- execute efficiently on GPUs;
- remain numerically and optimizationally stable throughout training.

The historical pattern is described in broad phases. From the original Transformer through roughly GPT-3, groups experimented with many variants and there was no single standard. Llama 2 then became an influential template, leading many groups to train close variants. More recent work has reopened the design space: one period emphasized stable training, while the newest changes increasingly target long-context modeling. The resulting architectures are messy because empirical quality, systems performance, and stability are all baked into the design.

### Additional explanation

This is a form of **empirical design archaeology**. A technical report does not prove that every disclosed choice is optimal. However, recurrence across independent, successful training runs is evidence that a choice is at least robust in a broad operating regime. Conversely, a rare choice may be a genuine innovation, a response to one organization's hardware, an inherited historical artifact, or simply an under-tested decision.

It helps to classify each architectural choice along three axes:

| Axis | Main question | Typical evidence |
|---|---|---|
| Statistical quality | Does it reduce loss or improve downstream behavior? | Controlled ablations, scaling curves, final evaluations |
| Systems efficiency | Does it improve throughput, latency, memory, or communication? | Profiling, kernel benchmarks, serving measurements |
| Stability | Does it reduce divergence, spikes, or sensitivity to scale? | Gradient/logit traces, failed-run rates, learning-rate tolerance |

A change can improve one axis while harming another. Parallel Transformer blocks, for example, can expose more fusion opportunities while reducing sequential depth. Logit caps can eliminate extreme values while also limiting how confident attention may become. The useful question is therefore not whether a component is universally "better," but which trade-off it implements under a particular budget.

---

# Part II - Modern Transformer building blocks

## 2. Move normalization out of the residual stream

**Transcript coverage:** lines 679-1102

### What the lecturer said - transcript only

The most widely shared criticism of the original Transformer is its placement of LayerNorm. A Transformer carries a residual stream $x$ through the network. Attention and the feed-forward network compute updates that are added back to this stream. In the original post-LayerNorm design, normalization occurs after the residual addition and therefore lies directly on the residual path.

Modern models instead place normalization outside that path. In the common **pre-norm** form, the input is normalized before attention or the feed-forward computation, and the resulting update is added to the untouched residual stream. Nearly every modern language model uses some form of non-residual normalization. The humorous exception highlighted in the lecture is OPT-350M, whose post-norm placement differs even from other OPT sizes.

Early work on pre-norm was motivated partly by the desire to eliminate learning-rate warm-up. Experiments showed that the original post-norm design converged worse without warm-up, whereas pre-norm behaved much better. Warm-up is still commonly used, but the more durable reason for the change is signal propagation in deep networks.

Architecture practitioners often summarize the principle as **keep the residual stream clean**. With pre-norm, an activation can travel through the residual path from the bottom to the top without being repeatedly transformed. The backward pass likewise contains a simple route for gradients. With normalization on the residual path, each layer's normalization changes gradient magnitudes as they propagate backward. The lecture points to evidence that pre-norm reduces gradient attenuation at initialization and lowers the size and frequency of gradient spikes. Those properties improve stability and make deeper models easier to train, which explains the near-universal adoption of non-residual normalization.

### Additional explanation

For a residual sublayer $F_l$, the two idealized forms are:

**Pre-norm**

$$
x_{l+1} = x_l + F_l(\operatorname{Norm}(x_l)).
$$

**Residual post-norm**

$$
x_{l+1} = \operatorname{Norm}(x_l + F_l(x_l)).
$$

In the pre-norm form, differentiating $x_{l+1}$ with respect to $x_l$ produces an identity contribution:

$$
\frac{\partial x_{l+1}}{\partial x_l}
= I + \frac{\partial F_l(\operatorname{Norm}(x_l))}{\partial x_l}.
$$

Across many layers, that identity route gives the gradient a direct path even when the learned branch is temporarily poorly conditioned. This does not prove that pre-norm is always superior, nor does it remove all stability problems, but it makes the practical intuition behind a "clean" residual stream precise.

Warm-up and norm placement address related but distinct problems. Warm-up limits the size of early parameter updates while optimizer moments and representations are immature. Pre-norm improves the network's internal signal path. A robust recipe may use both.

## 3. Non-residual post-norm and double normalization

**Transcript coverage:** lines 1105-1201

### What the lecturer said - transcript only

If the important rule is merely to keep normalization off the residual path, then the norm does not have to appear before the learned computation. It can also appear after attention or the feed-forward network but before that branch is added back to the residual stream. Recent models such as Grok, Gemma 2, and OLMo 2 have used this **non-residual post-norm** structure.

Some architectures place norms on both sides of a computation. The lecturer calls this "double norm" and returns to the same empirical lesson later: when a model has stability problems, adding normalization at additional danger points often helps. The rule sounds inelegant, but repeated experience has supported it, including the later practice of normalizing queries and keys inside attention.

### Additional explanation

The three layouts should not be conflated:

| Layout | Schematic branch update | Is the skip path normalized? |
|---|---|:---:|
| Residual post-norm | $x_{l+1}=\operatorname{Norm}(x_l+F_l(x_l))$ | Yes |
| Pre-norm | $x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l))$ | No |
| Non-residual post-norm | $x_{l+1}=x_l+\operatorname{Norm}(F_l(x_l))$ | No |
| Double norm | $x_{l+1}=x_l+\operatorname{Norm}(F_l(\operatorname{Norm}(x_l)))$ | No |

The last two forms constrain the scale of the update branch while preserving the identity skip path. Adding norms is not free: it adds memory traffic, parameters, and extra transformations. The engineering question is whether the added stability permits a larger learning rate, avoids failed runs, or improves final quality enough to repay that cost.

## 4. LayerNorm, RMSNorm, and the removal of bias terms

**Transcript coverage:** lines 1204-1723

### What the lecturer said - transcript only

LayerNorm mean-centers an activation, divides by its standard deviation, and then applies a learned scale and bias. Many older successful models trained with it, so it is not intrinsically wrong. Most modern language models, however, use RMSNorm. RMSNorm omits mean subtraction and usually omits the additive bias, retaining only rescaling by root-mean-square magnitude and a learned multiplicative scale.

LayerNorm is representationally more expressive, but experiments indicate that RMSNorm performs just as well for ordinary language modeling while being faster. This is where architecture and systems design meet. GPUs should spend their time on high-arithmetic-intensity work such as matrix multiplication rather than on small reductions and elementwise operations that repeatedly move activations through memory.

The lecture emphasizes that FLOPs are not runtime. In the cited operator breakdown, statistical normalization accounts for only about $0.17\%$ of FLOPs but can consume roughly $25\%$ of runtime in an extreme small-model workload. Tensor contractions spend far more FLOPs per byte moved, whereas normalization is dominated by reduction and memory traffic. An audience question asks why normalization moves a disproportionate amount of data compared with contraction. The answer is that matrix multiplication reuses data for many multiplications, while normalization performs relatively little arithmetic for the activation bytes it reads and writes. The displayed percentage is especially extreme for tiny matrices that are unlike many modern large-model workloads, but it illustrates why deleting unnecessary normalization work can be a genuine optimization.

A 2020 comparison shown in the lecture uses a roughly 223M-parameter Transformer. Replacing LayerNorm with RMSNorm increases throughput from about 3.50 to 3.68 steps per second and also improves several reported quality measures. The lecturer treats the quality gain as a welcome but non-guaranteed bonus; the systems gain is the more dependable argument.

The same principle extends to linear-layer biases. The original Transformer includes bias terms, but most modern implementations omit them. Bias addition performs little computation relative to its memory movement, seems to provide little additional expressive value in this setting, and can in some cases contribute to instability. Removing biases therefore simplifies the system. This conclusion is empirical rather than derivable in advance: accumulated model-training experience suggests that biases in Transformer linear layers and RMSNorm can be dropped safely for typical language-model workloads.

### Additional explanation

For a vector $x \in \mathbb{R}^d$, a simplified LayerNorm is:

$$
\operatorname{LayerNorm}(x)
= \gamma \odot
\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}
+\beta,
$$

where

$$
\mu(x)=\frac{1}{d}\sum_{i=1}^{d}x_i,
\qquad
\sigma^2(x)=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu(x))^2.
$$

A common RMSNorm form is:

$$
\operatorname{RMSNorm}(x)
= \gamma \odot
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}.
$$

RMSNorm controls magnitude but not mean. Its advantage is not that a few scalar operations are expensive in isolation; the reduction over $d$, synchronization needed to obtain a common statistic, and reads and writes of the activation all have low arithmetic intensity. Kernel fusion can mitigate some of this cost, but deleting unnecessary work is even better.

Bias removal should be interpreted locally. A bias-free sequence of linear maps would be restricted if nothing else introduced offsets or nonlinear effects. A Transformer contains embeddings, nonlinear gates, normalization scales, residual additions, and many stacked layers, so the marginal value of each individual bias can be small. The empirical claim is specific to these architectures, not a theorem that biases never matter in neural networks.

## 5. From ReLU and GeLU to gated feed-forward networks

**Transcript coverage:** lines 1726-2254

### What the lecturer said - transcript only

There is a large vocabulary of activation functions: ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU, and others. A language model can work with an ordinary non-gated activation. ReLU was used by models such as the original Transformer, Gopher, and Chinchilla; GeLU was used by the GPT family. The small negative "divot" of GeLU mainly changes behavior and gradients around zero rather than radically changing the positive branch.

The dominant modern choice is not one particular scalar nonlinearity but **gating**. A standard feed-forward network first projects $x$ with $W_1$, applies a nonlinearity, and projects back with $W_2$. A gated linear unit adds a second projection $xV$ and multiplies it elementwise with the activated branch before the output projection. ReLU produces ReGLU, GeLU produces GeGLU, and Swish produces SwiGLU. Google-associated models have often used GeGLU, while PaLM, Llama, and many Llama-descended systems use SwiGLU. SwiGLU is probably the most common, but the distinction among credible gated variants appears less important than the decision to gate at all.

Gating adds a third matrix. If the same feed-forward dimension were retained, a gated MLP would have more parameters than a two-matrix MLP. A common parameter-matched comparison therefore reduces the feed-forward dimension by a factor of $2/3$. The lecturer presents this as a rule of thumb, not an iron law.

Noam Shazeer's original comparison reported small but consistent gains for gated variants and used multiple replicas and error bars. Because the models were parameter matched using the $2/3$ adjustment, the result is close to a free quality improvement at fixed parameter count. A separate 2020 architecture study from Narang and colleagues also found gated variants to be stronger across loss and downstream metrics, although that controlled study used a T5-style architecture rather than an autoregressive language model. The accumulated evidence therefore points to consistent gains from gating at little additional computational cost.

Gating is not necessary for a functional model. GPT-3 used GeLU, and Nemotron-4 340B used the unusual choice of squared ReLU. Both are capable models. Nevertheless, a modern model without a gated feed-forward network is now an outlier.

### Additional explanation

Ignoring biases, a conventional feed-forward network is:

$$
\operatorname{FFN}(x)=\phi(xW_1)W_2.
$$

The common activations include:

$$
\operatorname{ReLU}(z)=\max(0,z),
$$

$$
\operatorname{GeLU}(z)=z\Phi(z),
$$

$$
\operatorname{Swish}(z)=z\sigma(z).
$$

A gated version is:

$$
\operatorname{GLU}_{\phi}(x)
= \bigl(\phi(xW)\odot xV\bigr)W_2.
$$

Thus:

$$
\operatorname{GeGLU}(x)
= \bigl(\operatorname{GeLU}(xW)\odot xV\bigr)W_2,
$$

$$
\operatorname{SwiGLU}(x)
= \bigl(\operatorname{Swish}(xW)\odot xV\bigr)W_2.
$$

The gate lets one learned feature modulate another. One branch can be read as proposing content and the other as controlling how strongly that content passes. This multiplicative interaction cannot be reproduced by merely changing the scalar activation of one projection without using additional depth or width.

If the input and output dimensions are $d$ and the intermediate width is $m$, a conventional MLP has approximately $2dm$ parameters. A gated MLP has approximately $3dm$. Parameter matching requires:

$$
3dm_{\mathrm{gated}} \approx 2dm_{\mathrm{plain}},
\qquad
m_{\mathrm{gated}} \approx \frac{2}{3}m_{\mathrm{plain}}.
$$

That calculation explains the $2/3$ correction and later feed-forward ratios near $8/3$ rather than $4$.

## 6. Serial versus parallel Transformer blocks

**Transcript coverage:** lines 2257-2575 and 3343-3436

### What the lecturer said - transcript only

A conventional Transformer block computes attention and then the MLP serially. GPT-J introduced, and PaLM helped popularize, a parallel alternative in which attention and the MLP receive the same normalized input and both updates are added to the residual stream. If implemented carefully, the branches can share a normalization and some matrix multiplications can be fused, creating systems-level speedups. GPT-NeoX and some Cohere models also used related parallel layouts.

Parallel blocks have become less popular over the last two years. Optimization of the serial form has improved enough that the systems benefit may no longer justify the apparent reduction in representational power. One way to view the parallel layout is that attention can no longer feed into the MLP inside the same block, so the network loses part of its sequential computational depth.

In the later Q&A, a student asks how large the quality difference is. The lecturer says the evidence is mixed. The PaLM report presents the design confidently, claiming approximately a $15\%$ systems-utilization or training-speed improvement and little or no quality loss at large scale. Later Google models stopped using the pattern, which is an indirect sign that some cost may exist. The lecturer is not aware of a clean, comprehensive modern controlled comparison that settles the issue.

The broader conclusion is that the original Transformer remains recognizable. Norm placement, biases, and MLP gating have changed, but dense attention models still use a largely similar serial residual structure. Most current models use RMSNorm, non-residual normalization, serial attention-then-MLP blocks, and a gated MLP. More radical attention alternatives are reserved for the next lecture.

### Source reconciliation

Slide 28 quotes the PaLM report more specifically: it reports roughly $15\%$ faster large-scale training, a small quality degradation at 8B parameters, and no degradation at 62B, from which the authors extrapolated quality neutrality at 540B. The spoken Q&A gives the higher-level and more skeptical retrospective assessment. These are compatible as a report of PaLM's original evidence versus the lecturer's view that later, comprehensive comparisons remain limited.

### Additional explanation

With pre-norm, a serial block can be written schematically as:

$$
h = x + \operatorname{Attention}(\operatorname{Norm}(x)),
$$

$$
y = h + \operatorname{MLP}(\operatorname{Norm}(h)).
$$

The parallel form is:

$$
y = x
+ \operatorname{Attention}(\operatorname{Norm}(x))
+ \operatorname{MLP}(\operatorname{Norm}(x)).
$$

The second form exposes both branches concurrently, but the MLP cannot condition on the attention update until the next layer. Whether that hurts depends on how much the model benefits from the within-block composition and whether extra layers or width compensate for it.

## 7. Why attention needs positional information

**Transcript coverage:** lines 2578-2698

### What the lecturer said - transcript only

The largest remaining architectural variation lies in how models represent position and modify attention. Without positional information, attention is built from inner products and is insensitive to a permutation of the input positions.

The original Transformer adds deterministic sine and cosine vectors to token embeddings. Several later models use learned absolute embeddings, assigning a distinct vector to each position. Google models such as T5 and Chinchilla have used relative schemes that alter the attention computation according to the distance between positions rather than adding a vector to the token embedding.

Rotary position embeddings, or RoPE, have become dominant: most models released after 2024 use some RoPE variant. The lecturer describes this as remarkable because the method began outside the most prominent model-development lineage and was adopted through GPT-J before spreading widely.

### Additional explanation

Self-attention without positional information is **permutation equivariant**: permuting the input positions permutes the outputs in the same way. That is useful for sets but insufficient for language, where "dog bites man" and "man bites dog" contain the same items in different orders.

The main families make different commitments:

| Scheme | Where position enters | Typical limitation |
|---|---|---|
| Sinusoidal additive | Added to token representation | Semantic-position cross terms; not purely relative |
| Learned absolute | Added position-specific vector | Tied to absolute indices and trained range |
| Relative bias/vector | Added inside attention score computation | Not necessarily expressible as an inner product of two independently encoded tokens |
| RoPE | Rotates queries and keys before their inner product | Frequency and extrapolation behavior must be chosen carefully |

## 8. RoPE as position-dependent rotation

**Transcript coverage:** lines 2701-3226

### What the lecturer said - transcript only

RoPE begins from an opinionated objective: the attention interaction between tokens should depend on their identities and relative displacement, not on absolute positions. If $f(x,i)$ embeds token $x$ at position $i$, the desired inner product has the form

$$
\langle f(x,i),f(y,j)\rangle = g(x,y,i-j).
$$

Earlier embeddings do not satisfy this exact factorized goal. Additive sine embeddings produce cross terms involving token and position vectors; learned absolute embeddings are explicitly absolute; and traditional relative attention changes the score directly rather than producing two embeddings whose inner product has the desired property.

The geometric idea behind RoPE is rotation. Start with position-independent semantic vectors and rotate each one by an amount determined by its position. A common rotation applied to both vectors preserves their inner product, so two tokens separated by one position retain the same relative angle even if both absolute positions shift. In the example, `we` and `know` at positions 0 and 1 have the same relative rotation as the same words at positions 2 and 3 in `of course we know`.

There are infinitely many rotations in high dimensions, so RoPE uses the simplest construction: split a $d$-dimensional query or key into pairs of coordinates and rotate each two-dimensional pair. Different pairs use different angular frequencies. Slowly rotating pairs can represent long-distance relationships, while rapidly rotating pairs are sensitive to short distances. Gemma 4 is mentioned as using a partial or proportional RoPE variant that rotates only a subset of coordinates - described in the lecture as just the first pair for that setting.

In implementation, the model computes sines and cosines from position IDs and applies them multiplicatively as rotations to both queries and keys. This is done at every attention layer, rather than once at the bottom of the model, to impose the relative structure whenever attention scores are computed. Multiplication rather than addition is essential: it avoids the semantic-position cross terms of additive sinusoidal embeddings.

### Additional explanation

For one coordinate pair, the rotation at position $m$ and frequency $\theta$ is:

$$
R_{m,\theta}
=
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}.
$$

The full RoPE matrix is block diagonal, with one such $2\times2$ block per coordinate pair. Let $q_i=R_iq$ and $k_j=R_jk$. Then:

$$
q_i^\top k_j
=q^\top R_i^\top R_jk
=q^\top R_{j-i}k.
$$

The last equality shows that the positional part of the score depends on $j-i$, not on $i$ and $j$ separately. The same-position rotation does not simply disappear from each vector; rather, the pair of rotations combines into one relative rotation inside the inner product.

In code, the block-diagonal multiply is usually expressed with elementwise sine/cosine operations and a coordinate-pair rearrangement rather than materializing $R_i$. A conceptual form is:

```python
q = q * cos + rotate_pairs(q) * sin
k = k * cos + rotate_pairs(k) * sin
```

Queries and keys are rotated because their dot product determines attention weights. Values need not be rotated for the standard RoPE score construction.

## 9. Architecture and positional-encoding Q&A

**Transcript coverage:** lines 3229-3607

### What the lecturer said - transcript only

Several questions refine the preceding architecture discussion:

- **Are there higher-dimensional rotation methods?** The lecturer has not seen a prominent alternative beyond variants of two-dimensional rotations within the larger space. In principle, one could design other closed-loop manifolds or rotation paths, but he cannot point to established work using them.
- **How should this kind of knowledge be distilled from papers?** Read broadly enough to see patterns across releases, then reproduce choices at a smaller scale to form an intuition and tentative theory. A single paper is especially difficult to trust because current language-model reports rarely disclose every relevant detail.
- **What is the difference between partial/proportional RoPE and ordinary RoPE?** It changes which coordinate pairs are rotated. The stated motivation is that very low-frequency pairs rotate little, so a small model with little hidden capacity may reserve some coordinates for semantic information and rotate only a subset.
- **Why reject attention-level relative embeddings merely because they are not an inner product?** The factorization requirement is partly an aesthetic or modeling constraint. If the goal specifically requires independent embeddings $f(x,i)$ and $f(y,j)$ whose inner product is relative, direct attention biases are outside that class. They can still work; ALiBi and other methods modify attention scores directly, though they did not become the dominant choice in the surveyed models.
- **Why are additive sine embeddings not purely relative?** Their attention scores contain cross terms between semantic vectors and absolute positional vectors, allowing absolute-position information to remain recoverable. Once one accepts both relative dependence and the factorized inner-product goal, RoPE emerges naturally.

One answer about older relative embeddings is verbally ambiguous: the lecturer says they apply to "keys and values," then immediately emphasizes unwanted cross terms and the factorized score objective.

### Source reconciliation

The slide-deck equation for the relative scheme modifies the key-side attention computation with a relative term $a_{ij}^{K}$; it does not show a value-side term. The spoken phrase "keys and values" in lines 3502-3508 is therefore preserved as an ambiguity rather than treated as a reliable implementation specification.

### Additional explanation

The methodological answer is as important as the positional one. A useful evidence ladder for architecture choices is:

1. Observe the choice in one report.
2. Check whether independent model families repeat it.
3. Find controlled, parameter- and compute-matched ablations.
4. Reproduce the comparison at an affordable scale.
5. Test whether the result transfers as scale, data, and hardware change.

RoPE's factorization has practical value beyond aesthetics: position can be injected into queries and keys while retaining ordinary dot-product attention kernels. Direct relative-bias methods may also be efficient, but they express a different inductive bias and can require special handling for score construction or extrapolation.

---

# Part III - Hyperparameters that usually have forgiving defaults

## 10. Why implementation forces concrete hyperparameter choices

**Transcript coverage:** lines 3613-3700

### What the lecturer said - transcript only

Hyperparameters become unavoidable when an abstract model must be instantiated. A builder has to choose the feed-forward width, number and size of attention heads, vocabulary size, weight decay, dropout, and the depth-versus-width shape of the model. With no prior knowledge, this is a daunting high-dimensional search problem.

The empirical space used by successful language models is much smaller than the mathematical space of all possible settings. By surveying what others use, one can identify robust defaults and reserve experiments for the few dimensions on which a project has a specific thesis. The lecture therefore treats recurring values not as laws but as strong starting points that reduce an otherwise intractable search.

### Additional explanation

A default has two kinds of value:

1. It is likely to produce a competent baseline.
2. It makes later ablations interpretable because only the hypothesized change differs from a conventional recipe.

Searching every architectural and optimization parameter simultaneously creates an attribution problem: even if one run wins, it is difficult to know why. Mature training efforts usually fix most choices, test a small number of targeted variations, and use scaling studies to decide whether the result is likely to persist.

## 11. Feed-forward width relative to model width

**Transcript coverage:** lines 3703-4110

### What the lecturer said - transcript only

The first consensus hyperparameter is the ratio between feed-forward dimension $d_{ff}$ and model dimension $d_{model}$. For a conventional two-matrix MLP, the remarkably robust rule is:

$$
d_{ff} \approx 4d_{model}.
$$

Gated MLPs contain three large matrices rather than two. Applying the $2/3$ parameter-matching correction gives:

$$
d_{ff} \approx \frac{2}{3}(4d_{model})
= \frac{8}{3}d_{model}
\approx 2.67d_{model}.
$$

Many gated models use a ratio around $2.5$ to $2.67$. Llama 2 and related designs apply an additional factor of roughly $1.33$ and use a larger ratio around $3.5$, emphasizing the MLP somewhat more. The lecturer describes this as an architecture choice associated with the use of more inference-efficient attention heads, rather than a universal derivation. Across surveyed models, the ordinary clusters are therefore near $4$ for non-gated MLPs and roughly $2.6$ or $3.5$ for gated MLPs.

T5 is the dramatic exception. Its 11B configuration uses a feed-forward width about $64$ times its model dimension. The stated systems argument is that very large dense matrix multiplications can utilize accelerator hardware efficiently. Gemma 2 also pushed the ratio higher, though not nearly as far. T5 proves that an extreme ratio can train a valid model, but does not prove that it is compute-optimal.

A sweep shown from Kaplan and colleagues uses a small language model and finds a broad, flat basin: ratios from roughly 1 to 10 remain near optimal, while much larger values eventually make loss rise sharply, described in the lecture as roughly quadratic growth in the bad regime. Conventional values between about $2.6$ and $4$ all lie in this forgiving region. The punchline is that T5 v1.1, presented as an improved follow-up, returned to a conventional GeGLU multiplier near $2.5$. The lecturer reads this as indirect evidence that the $64\times$ setting was not the best default.

### Additional explanation

For a plain MLP, ignoring biases:

$$
N_{\mathrm{MLP}} \approx d_{model}d_{ff}+d_{ff}d_{model}
=2d_{model}d_{ff}.
$$

For a gated MLP:

$$
N_{\mathrm{gated}} \approx 3d_{model}d_{ff}.
$$

Thus a comparison at equal $d_{ff}$ is not parameter matched. The choice of ratio also changes more than parameter count:

- MLP FLOPs and activation memory grow with $d_{ff}$.
- Larger matrices can improve hardware utilization up to the point where memory or capacity becomes limiting.
- More MLP capacity changes the balance between token-wise transformation and cross-token attention.

The broad empirical basin means that systems considerations can legitimately choose among several statistically similar ratios. A value should nevertheless be compared at equal total compute or parameter budget if the goal is to claim a quality improvement.

## 12. Number of heads and head dimension

**Transcript coverage:** lines 4111-4257

### What the lecturer said - transcript only

The second consensus rule concerns multi-head attention. If there are $h$ heads, most models choose head dimension $d_{head}$ so that their concatenated width equals the model width:

$$
h\,d_{head} \approx d_{model}.
$$

This equality is a convention, not a mathematical requirement. The total attention projection width can be larger or smaller than $d_{model}$. Nevertheless, models from GPT-3 and Llama through recent Qwen systems usually have a ratio near one. T5 and LaMDA are notable Google-associated exceptions.

Available ablations suggest another forgiving basin around the standard value. The lecturer therefore treats this parameter as useful to understand but not normally the first place to search for a major quality gain.

### Additional explanation

Define the projection ratio:

$$
r_{head}=\frac{h\,d_{head}}{d_{model}}.
$$

The standard setting has $r_{head}=1$. Increasing it expands the total query/key/value projection width and attention-score capacity, but also increases parameters, projection FLOPs, and often KV-cache size unless the number of key/value heads is changed separately. This last qualification becomes important for grouped-query attention: the number of query heads and the number of key/value heads need not match.

The equality also makes implementation convenient. A $d_{model}$-wide projection can be reshaped into $h$ heads without changing total width, and the head axis behaves like an additional batch axis for batched matrix multiplication.

## 13. Aspect ratio: depth versus width

**Transcript coverage:** lines 4260-4570

### What the lecturer said - transcript only

When models are scaled up or down, builders often hold a model **aspect ratio** roughly fixed and increase both depth and width. The lecture defines the relevant ratio as:

$$
r_{aspect}=\frac{d_{model}}{n_{layers}}.
$$

Despite substantial variation, many modern models fall near a value of about 100. GPT-3, Llama, and several other families occupy this broad region. Models do not normally become arbitrarily deep or arbitrarily wide.

The trade-off combines expressiveness and systems constraints. More depth supplies more sequential transformations and may increase representational power. Extremely deep models, however, are difficult to parallelize: different layers depend on one another. Splitting layers across devices introduces pipeline parallelism, which creates scheduling and utilization problems that practitioners prefer to avoid when possible. Width is easier to partition with tensor parallelism, making wide models more attractive from a systems perspective. The observed aspect ratio can be read as a compromise between the expressive appeal of depth and the execution advantages of width.

The Kaplan sweep shown in the lecture finds a similar optimum across several model sizes, near the same broad aspect-ratio region. Work by Tay and colleagues likewise suggests that, across a wide range of shapes, total FLOPs explains much more of performance than exact depth-to-width allocation. The practical conclusion is that there is a forgiving band of good shapes. Once inside it, hardware utilization and parallelization may matter more than speculative claims about expressiveness.

### Additional explanation

Depth and width affect latency differently even when FLOPs are similar:

- Layers are sequential for a single example, so greater depth raises the minimum number of dependent kernel stages.
- Width creates larger matrix multiplications, which often use accelerator units more efficiently and can be sharded across devices.
- Very wide layers consume more activation and parameter memory per layer.
- Very deep pipelines can regain throughput with microbatching, but may create pipeline "bubbles" and more complex schedules.

The ratio near 100 is not dimensionless wisdom independent of scale and hardware. It is an empirical neighborhood in the surveyed dense Transformer regimes. A small teaching model may need a different shape to avoid tiny inefficient matrices, while a latency-sensitive serving model may favor fewer layers even if a deeper model is statistically comparable.

## 14. Vocabulary size, multilinguality, and comparable likelihoods

**Transcript coverage:** lines 4573-4906

### What the lecturer said - transcript only

Vocabulary size separates two broad eras and use cases. Earlier open models often targeted English and used vocabularies around 30,000 tokens. Modern multilingual or production systems more often use roughly 100,000 to 200,000 tokens. Google models are frequently at the high end, while Llama-derived models are often near 100,000. Large monolingual models have become uncommon.

Multilingual coverage is one reason for the increase: more writing systems and recurring byte sequences must be represented efficiently. Model scale also matters. Scaling-law studies suggest that larger models can support larger vocabularies, and newer multilingual systems are generally larger than the earlier monolingual ones. Vocabulary growth is therefore driven by both data diversity and overall model capacity.

In response to a question about multimodal models, the lecturer says the required vocabulary depends on representation. If images are discretized into tokens, an image tokenizer may have its own large vocabulary in addition to the text vocabulary.

Another question asks whether bits per byte can be compared across tokenizers. Language modeling assigns a probability to a fixed raw sequence. Comparisons are valid when both tokenizers model the same complete sequence distribution. Older word tokenizers sometimes dropped or replaced material, which changes the modeled sequence and invalidates a direct comparison. Modern lossless subword tokenizers can represent every sequence. Bits per byte then uses the same raw byte count for length normalization, so it remains comparable across tokenizations.

A follow-up question tries to connect token-level perplexity and bits per byte. The lecturer says that perplexity and byte-normalized measures are related, but the exchange remains unresolved because he is not sure he has understood the student's exact claim. The class defers that point rather than manufacturing a conclusion.

### Additional explanation

Token-level average negative log-likelihood is:

$$
\mathcal{L}_{token}
=-\frac{1}{T}\sum_{t=1}^{T}\log p(z_t\mid z_{<t}),
$$

and token perplexity is:

$$
\operatorname{PPL}=e^{\mathcal{L}_{token}}.
$$

If two tokenizers produce different token counts $T$, their token perplexities use different denominators and are not directly comparable. A byte-normalized quantity instead divides total negative log-likelihood by the same number of raw bytes $B$:

$$
\operatorname{bits/byte}
=-\frac{1}{B\ln 2}
\sum_{t=1}^{T}\log p(z_t\mid z_{<t}).
$$

This is comparable only if tokenization and detokenization preserve the same byte sequence and the probability model accounts for all possible inputs being evaluated.

A larger vocabulary shortens token sequences but enlarges input embeddings and the output classifier. It may also create many rare tokens. Vocabulary choice is therefore another whole-system trade-off among attention length, softmax cost, parameter count, coverage, and statistical sharing.

## 15. Dropout, weight decay, and the difference between regularization and optimization

**Transcript coverage:** lines 4909-5383

### What the lecturer said - transcript only

Standard machine-learning intuition says regularization prevents overfitting, but compute-constrained language-model pretraining is unusual. There is generally more data than the run can process, so the model often makes one pass through the corpus and may never see the same example twice. Single-pass stochastic optimization is unlikely to memorize the full dataset. Some practitioners are sufficiently confident in this regime that they monitor training loss without relying heavily on a separate validation-overfitting gap.

This makes dropout and weight decay seem unnecessary. Recent technical reports often omit the details, and dropout in particular has largely fallen out of favor. Yet weight decay remains common in high-performing modern models. This is surprising only if weight decay is assumed to act solely as a statistical regularizer.

The cited evidence argues that weight decay can instead improve **optimization dynamics**. Across weight-decay settings, training and validation loss remain close to the $x=y$ relationship, indicating that overfitting was not the issue. When weight decay is combined with a decaying learning-rate schedule, stronger-decay runs may begin more slowly but eventually reach a better training minimum. The same pattern is not necessarily present under a constant learning rate. Weight decay, learning rate, and schedule therefore interact; an intervention named for regularization can improve the optimization path even when there is no train-validation gap to fix.

The hyperparameter summary is consequently nuanced:

- use the standard feed-forward ratios near $4$ for plain MLPs or approximately $8/3$ for parameter-matched gated MLPs;
- keep total query-head width near the model dimension as a default;
- choose an aspect ratio in the broad neighborhood of 100;
- do not assume that regularization settings can be copied independently of the optimizer and learning-rate schedule.

During Q&A, the lecturer says large text diffusion models are not yet numerous enough for him to identify a distinct optimal architecture. Many existing diffusion systems retrofit a Llama-like architecture, so their visible architecture may reflect inheritance rather than a from-scratch optimum. Asked why weight decay helps, he suggests that shrinkage toward zero may permit a higher learning rate or interact favorably with decay. Dropout does not appear to have the same useful interaction and is no longer common in pretraining.

### Additional explanation

For AdamW-style decoupled weight decay, a schematic update is:

$$
\theta_{t+1}
= (1-\eta_t\lambda)\theta_t
-\eta_t\,\widehat{u}_t,
$$

where $\eta_t$ is the learning rate, $\lambda$ is weight decay, and $\widehat{u}_t$ is the optimizer's adaptive update. The cumulative shrinkage depends on both $\lambda$ and the full learning-rate schedule. Changing either one changes the effective trajectory.

This explains why "weight decay = 0.1" is not a self-contained recipe. Batch size, optimizer details, learning-rate peak, warm-up, decay duration, parameter exclusions, and total training steps all affect its meaning. It also explains why an intervention can improve both training and validation loss: it has guided the optimizer to a better region rather than merely widened a generalization gap.

Dropout is different. It injects stochastic masks into activations, which can reduce effective capacity and complicate optimized kernels. In a one-pass, underfit, compute-limited regime, that noise may cost more optimization progress than it provides in generalization benefit.

---

# Part IV - Architecture as protection against unstable training

## 16. Why stability becomes a first-class objective

**Transcript coverage:** lines 5386-5568

### What the lecturer said - transcript only

Recent architectural work increasingly prioritizes stable training rather than small improvements in final performance. Many ordinary architecture hyperparameters are forgiving, so changing them within a conventional range may have little effect. Instability is different. A run can develop large loss and gradient spikes, lose model quality, or become unrecoverable after millions of dollars of computation have already been spent.

When a neural network is unstable, softmax operations are among the first places to inspect. Softmax contains exponentials, which grow rapidly, and a division by a normalizing sum, which can be numerically dangerous if values overflow, underflow, or become badly scaled. A language model has two prominent softmax locations:

1. the output distribution over vocabulary items;
2. the normalization of attention scores.

The lecture treats both as danger zones and introduces separate interventions for each.

### Additional explanation

In exact real arithmetic, a softmax normalizer is positive. In finite precision, the intermediate exponentials can overflow to infinity, underflow to zero, or lose useful resolution. A numerically stable implementation subtracts the maximum logit before exponentiation:

$$
\operatorname{softmax}(u)_i
=\frac{e^{u_i-m}}{\sum_j e^{u_j-m}},
\qquad m=\max_j u_j.
$$

That protects the operation itself, but it does not prevent the model from learning extremely large logit norms or nearly one-hot attention. Architectural stabilization constrains the model's trajectory, not merely the arithmetic implementation.

The expected cost of instability grows superlinearly with run size. Large runs are longer, harder to restart, and more expensive per failed step. A small quality improvement may be dominated by a modest increase in failure probability, so conservative architecture choices can be economically optimal even if their mean small-scale loss is unchanged.

## 17. Output-softmax stability with z-loss

**Transcript coverage:** lines 5572-5767

### What the lecturer said - transcript only

For an output logit vector $U(x)$, the log probability of target $r$ can be written as:

$$
\log P(r\mid x)=U_r(x)-\log Z(x),
$$

where

$$
Z(x)=\sum_{r'=1}^{|V|}e^{U_{r'}(x)}.
$$

The direct target logit may remain well behaved while the log normalizer $\log Z$ grows or shrinks to a numerically dangerous scale. Softmax is overparameterized with respect to a shared additive shift: adding the same constant to all logits changes $Z$ but leaves the normalized probabilities unchanged because the shift cancels.

This redundancy allows an auxiliary **z-loss** that penalizes the distance of $\log Z$ from zero. In minimization form, the additional term is proportional to:

$$
\mathcal{L}_{z}=\alpha\bigl(\log Z(x)\bigr)^2.
$$

Keeping $\log Z$ near zero makes the output-softmax calculation better scaled without changing which normalized distribution the model can represent. The trick originates in work associated with Jacob Devlin in 2014 and later reappeared in large models. The lecturer names Baichuan as an early open model using it and points to DCLM, OLMo, and other open models as later adopters. It is described as surprisingly effective for stabilizing the output layer.

### Additional explanation

The invariance is:

$$
\operatorname{softmax}(u+c\mathbf{1})
=\operatorname{softmax}(u).
$$

However,

$$
\log\sum_j e^{u_j+c}=c+\log\sum_j e^{u_j}.
$$

Ordinary cross-entropy cannot identify the shared offset $c$, so the logits may drift along that otherwise irrelevant direction. Z-loss chooses a preferred gauge by anchoring the log-partition function. It is called a loss term, but its purpose here is numerical and optimization stability rather than conventional generalization.

The coefficient $\alpha$ must remain small enough that the model's primary objective is still predictive likelihood. The slide example shows a coefficient of $10^{-4}$ for one large-model recipe, but that number should be treated as recipe-specific rather than universal.

## 18. Attention-softmax stability with QK normalization

**Transcript coverage:** lines 5770-5968

### What the lecturer said - transcript only

Attention contains the other major softmax. Standard attention projects normalized hidden states into queries, keys, and values, multiplies queries by keys to form logits, applies softmax, and uses the resulting weights to average values.

**QK norm** inserts a normalization on the projected queries and keys immediately before their dot product. Their magnitudes are therefore controlled before they enter the attention softmax. The lecture describes the scale as remaining roughly one after RMS normalization, preventing query and key norms from driving logits to extreme magnitudes.

The intervention originated in vision or multimodal work, with examples such as Idefics and Chameleon, and was then adopted by open language models. It is now common in large models including DCLM, OLMo, Gemma, and Qwen releases. Across many runs it appears not to reduce quality, while it does prevent attention degeneracies. In some comparisons it performs slightly better because the stabilized model can tolerate a larger learning rate.

QK norm reinforces the lecture's recurring empirical pattern: begin with pre-norm outside the residual stream, add norms after branches when needed, and normalize queries and keys at the immediate source of attention instability.

### Additional explanation

Ordinary scaled dot-product attention uses:

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{head}}}\right).
$$

With QK normalization:

$$
A=\operatorname{softmax}\left(
\frac{\operatorname{Norm}(Q)\operatorname{Norm}(K)^\top}
{\sqrt{d_{head}}}
\right).
$$

The usual $1/\sqrt{d_{head}}$ factor controls variance under an initialization assumption. QK norm controls the realized vector magnitudes throughout training. The two mechanisms therefore address related but different sources of scale.

Normalizing Q and K does not force every attention score to be equal. Their directions still vary, and the learned normalization scales can retain flexibility. The intervention mainly prevents unbounded norm growth from becoming an easy route to sharper attention.

## 19. Logit soft-capping as a stronger constraint

**Transcript coverage:** lines 5971-6112

### What the lecturer said - transcript only

Logit soft-capping is a stronger, less widespread, and more Google-associated stability intervention. QK norm controls the inputs to the attention-logit calculation and hopes that the resulting scores remain well behaved. Soft-capping directly bounds the logits sent to softmax by passing them through a scaled hyperbolic tangent:

$$
\operatorname{softcap}_c(a)
=c\tanh\left(\frac{a}{c}\right).
$$

Because the output lies between $-c$ and $c$, the attention softmax can never receive an arbitrarily large positive or negative score. Gemma 2, Gemma 3, and Gemma 4 are cited as using this technique.

The safety comes with a possible quality cost. A systematic NVIDIA comparison finds QK norm slightly stronger than the baseline, partly because it permits a larger learning rate, while soft-capping alone worsens perplexity. The cap is a hard expressive restriction in practice: beyond its threshold, the model cannot communicate increasing confidence through larger logits. It is therefore a conservative way to stabilize attention, but not a free intervention.

### Additional explanation

Near zero, $\tanh(a/c)\approx a/c$, so the soft-cap behaves approximately like the identity:

$$
c\tanh(a/c)\approx a.
$$

For large $|a|$, it saturates at $\pm c$. This makes it smoother than hard clipping but still causes gradients to shrink in the saturated region. That is the core trade-off:

- guaranteed bounded logits and reduced catastrophic risk;
- weaker ability to represent extremely sharp score differences and smaller gradients once saturated.

QK norm is a softer inductive bias because it controls vector scale while leaving angular similarity and learned scaling more flexible. Soft-capping is attractive when avoiding a failed run is worth a small expected quality penalty.

---

# Part V - Attention architectures shaped by inference and long context

## 20. The attention changes covered in this lecture

**Transcript coverage:** lines 6115-6198

### What the lecturer said - transcript only

The final part of the lecture considers modifications that retain dot-product attention. State-space models and linear-time attention are postponed to the next lecture. Two current interventions are emphasized:

1. multi-query or grouped-query attention, which reduces inference cost by using fewer key/value heads;
2. sparse or sliding-window attention, which reduces the cost of long contexts by restricting the attention pattern and interleaving cheaper local layers with occasional full-attention layers.

These changes illustrate architecture-system co-design. They are not introduced primarily to alter the abstract function class; they modify what must be stored, moved, and recomputed during serving.

### Additional explanation

Attention architecture now has at least three independently adjustable dimensions:

- **score pattern:** which positions may interact;
- **head sharing:** how many distinct query, key, and value projections exist;
- **position mechanism:** what positional information accompanies local and global interactions.

Changing one dimension can alter the best setting of another. For example, full-attention layers may omit positional embeddings while local layers use RoPE, or local attention may retain many query heads while sharing a small number of KV heads.

## 21. Prefill, autoregressive decode, and the KV-cache bottleneck

**Transcript coverage:** lines 6199-6568

### What the lecturer said - transcript only

Serving a trained model consumes both arithmetic and memory bandwidth. Training and prompt prefill process many tokens together, exposing large matrix multiplications. In the lecture's order-of-magnitude accounting, with batch size $b$, sequence length $n$, hidden dimension $d$, number of heads $h$, and head dimension $k=d/h$, the dominant prefill arithmetic and memory terms are summarized as:

$$
\text{arithmetic} \sim bnd^2,
$$

$$
\text{memory access} \sim bnd+bhn^2+d^2.
$$

This produces high arithmetic intensity when the head dimension and the product of batch size and sequence length are sufficiently large:

$$
I_{prefill}
=O\left(\left(\frac{1}{k}+\frac{1}{bn}\right)^{-1}\right).
$$

Autoregressive generation is different because tokens must be produced one at a time. Each new token conditions on the previous output, so generation cannot be parallelized across future time steps. Efficient decoding stores the past projected keys and values in a **KV cache**. At the next step, the model computes only the new query, key, and value and combines the new query with cached keys and values rather than recomputing every earlier projection.

Caching saves arithmetic but creates repeated memory traffic. Aggregated over generation, the lecture gives:

$$
\text{arithmetic} \sim bnd^2,
$$

$$
\text{memory access} \sim bn^2d+nd^2,
$$

and therefore:

$$
I_{decode}
=O\left(\left(\frac{n}{d}+\frac{1}{b}\right)^{-1}\right).
$$

Efficient decode therefore benefits from large batches, short sequences relative to model width, or very large hidden dimensions. The $n/d$ term is difficult to reduce for small models serving long contexts, which motivates attention designs with smaller caches.

### Source reconciliation

At lines 6415-6427, the transcript momentarily calls the state a cache of past "keys and queries" and refers to "Q, K." Slides 59-61, the term **KV cache**, and the subsequent MQA/GQA explanation consistently show that cached states are keys and values. This is a verbal slip; queries are generated for the current decoding step, while past keys and values are reused.

The raw transcript also contains garbled speech-to-text around the displayed complexity expressions. The equations above follow the notation visibly shown on slides 58 and 60 and match the surrounding spoken interpretation.

### Additional explanation

For one layer, a standard KV cache has approximate element count:

$$
N_{KV}=2bnh_{kv}d_{head},
$$

where $h_{kv}$ is the number of key/value heads. The factor of two is for keys and values. Across $L$ layers and $s$ bytes per stored element:

$$
M_{KV}\approx 2Lbnh_{kv}d_{head}s.
$$

This grows linearly with batch size, context length, layer count, and KV-head width. During decode, those cached values are repeatedly read, so shrinking $h_{kv}$ improves both capacity and bandwidth.

The lecture's cost formulas retain dominant terms and suppress constants and some lower-order operations. They should be used to identify scaling bottlenecks, not as exact kernel-runtime predictions. Real serving also depends on cache layout, quantization, paging, batching, and memory hierarchy.

## 22. Multi-query attention: one set of keys and values

**Transcript coverage:** lines 6571-6691

### What the lecturer said - transcript only

Standard multi-head attention gives every head distinct queries, keys, and values. **Multi-query attention (MQA)** retains many query heads but shares one key head and one value head across them. The KV cache therefore becomes much smaller, and far fewer key/value elements must move through memory during decode.

The reduction in memory traffic improves arithmetic intensity, especially when there are many query heads. The lecture verbally corrects itself after momentarily saying that both memory access and arithmetic intensity are reduced: the intended benefit is reduced memory access and therefore **increased** arithmetic intensity.

MQA imposes a significant expressiveness cost because all query heads must attend through the same key and value representations. It therefore exposes a direct trade-off between serving efficiency and model quality.

### Additional explanation

Let $h_q$ be the number of query heads and $h_{kv}$ the number of key/value heads. Then:

- multi-head attention: $h_{kv}=h_q$;
- multi-query attention: $h_{kv}=1$.

Relative to ordinary MHA, the key/value-cache component shrinks by approximately:

$$
\frac{h_q}{h_{kv}}=h_q
$$

for MQA. Query projections are not cached across past tokens for standard decode, so sharing them would not provide the same memory benefit.

MQA must be part of the trained architecture. Simply merging a trained model's KV heads at inference changes its function and generally loses quality. A model can sometimes be adapted or uptrained into a shared-head form, but that is a separate conversion procedure, not the basic MQA design discussed here.

## 23. Grouped-query attention as the practical compromise

**Transcript coverage:** lines 6691-6859

### What the lecturer said - transcript only

**Grouped-query attention (GQA)** chooses an intermediate number of key/value heads. It keeps all query heads but divides them into groups, with each group sharing one key and one value head. The number of KV groups becomes a knob that interpolates between multi-head attention and MQA:

$$
1 < h_{kv} < h_q.
$$

This provides a direct way to trade expressiveness for inference efficiency. The lecture briefly notes that DeepSeek V2's multi-head latent attention uses a different factorization and will be discussed later.

Empirically, the compromise is favorable. Full multi-head attention has the strongest quality but highest time per sample. MQA is cheap but shows a clearer performance loss. GQA substantially lowers serving cost while retaining nearly the quality of full multi-head attention. Reducing the number of KV heads modestly captures much of the efficiency gain before most of the expressive cost appears. This is why GQA has become common in current models.

### Additional explanation

The group size is:

$$
g=\frac{h_q}{h_{kv}}.
$$

Each KV head serves $g$ query heads. At $g=1$, the system is ordinary MHA; at $g=h_q$, it is MQA. Intermediate powers of two are convenient for layouts and sharding, though the mathematical idea does not require them.

GQA is an example of a common systems pattern: most of the cost reduction can occur before most of the quality loss. This produces a "knee" in the trade-off curve. Finding that knee is more valuable than comparing only the two endpoints.

## 24. Q&A: how much should a model builder still search?

**Transcript coverage:** lines 6862-7023

### What the lecturer said - transcript only

A student asks whether strong rules of thumb eliminate hyperparameter search. The lecturer answers that practical runs mix exploitation and exploration. Each training effort normally has a thesis about one or a few choices that might change, while keeping most hyperparameters conventional. Technical reports frequently modify one architectural element at a time; it is rare to redesign everything simultaneously. Google is described as unusually willing to make many bold changes. Gemma 4 is mentioned as introducing separate embeddings at every layer to navigate a memory-versus-FLOP trade-off.

Asked whether parameters change dynamically during training, the lecturer identifies weight decay as a notable example because it may change together with learning rate. Most architectural hyperparameters cannot change mid-run without making the current parameters structurally incompatible, so they remain fixed.

Finally, a student confirms that MQA is not merely an inference-time patch. The lecturer agrees: the model is trained with the chosen number of key/value heads.

### Additional explanation

A disciplined search can be organized as follows:

1. Fix a proven baseline architecture and optimizer.
2. State one hypothesis, including the metric it should improve.
3. Compare at matched compute, tokens, and parameter budget where possible.
4. Test at more than one scale.
5. Profile systems behavior as well as validation loss.
6. Retain the change only if its benefit exceeds added complexity and failure risk.

Dynamic architectural changes are possible in research systems - for example, progressive depth or expert growth - but they require explicit parameter-mapping and optimizer-state rules. They are not ordinary drop-in hyperparameter schedules.

## 25. Sliding-window and hybrid local-global attention

**Transcript coverage:** lines 7024-7306

### What the lecturer said - transcript only

Sliding-window attention is an old idea. GPT-3 alternated full causal attention, in which a position can attend to all prior positions, with a banded pattern restricted to a fixed local window. The approach has recently become popular again because it offers a practical trade-off between long-context performance and inference cost without requiring a state-space model or another exotic replacement for attention.

Cohere Command A is presented as a prominent recent example. In a repeating group of four layers, three use sliding-window attention and the fourth uses full attention. As layers accumulate, local representations incorporate information from progressively larger neighborhoods, while periodic global layers permit direct long-range communication. The pattern substantially reduces the number of all-pairs attention operations.

Position encoding can also differ between the two layer types. Command A uses RoPE in the short-range sliding-window layers and no positional embedding, or NoPE, in the full-attention layers. The global layer therefore treats distant content more like an unordered set, while local layers retain precise order. Other recent models, including Llama 4 and Gemma-family releases, interleave sliding-window and full attention while retaining RoPE more broadly.

Qwen 3.5 uses the same general hybrid rhythm but replaces the cheap sliding-window layer with Gated DeltaNet, a state-space or linear-attention-style layer, and inserts one full-attention layer approximately every four layers. That mechanism is postponed to the next lecture.

Long-context architecture remains an active area. The emerging theme is neither full attention everywhere nor a cheap mechanism everywhere, but a hybrid: inexpensive local or recurrent layers carry most of the computation, and periodic global attention restores broad communication.

### Additional explanation

With a local window of width $w$, attention-score work changes from approximately:

$$
O(n^2d_{head})
$$

to:

$$
O(nwd_{head}),
$$

when $w\ll n$. Interleaving one full layer every $r$ layers gives a rough attention-score cost proportional to:

$$
\frac{1}{r}O(n^2d_{head})
+\frac{r-1}{r}O(nwd_{head}).
$$

This is not purely a compute trade-off. A local layer cannot retrieve an arbitrary distant token directly. Information must propagate through overlapping windows or pass through a global layer. The model's effective communication graph therefore depends on window size, layer ordering, and the frequency of full attention.

NoPE in global layers is an intentional inductive bias, not an absence of all order information in the network. The input to a global layer has already passed through position-aware local layers, so it may carry order features in its hidden states even if the global score computation adds no new explicit position signal.

## 26. Closing perspective

**Transcript coverage:** lines 7309-7357

### What the lecturer said - transcript only

Looking across many dense language models reveals both commonality and genuine variation. The common patterns provide good defaults for assignments and real model building. Non-residual normalization, RMSNorm, gated MLPs, conventional feed-forward and aspect ratios, and GQA have accumulated broad support.

The major differences increasingly concern context, positional encoding, tokenization, and the way full attention is combined with cheaper mechanisms. Those remain active design areas rather than settled conventions. The intended outcome is not memorization of a table but an informed sense of which choices are safe defaults, which choices embody a systems trade-off, and which choices deserve experiments.

### Additional explanation

The lecture's deepest lesson is methodological: architecture is a negotiated interface among statistics, optimization, and hardware. A component is not fully understood by its mathematical definition alone. One must ask what it does to gradients, memory traffic, matrix shapes, cache size, parallel schedules, and failure probability.

This also explains why modern Transformers look conservative and experimental at the same time. Their residual skeleton is remarkably stable, while the pressure points created by scale - stable training, serving bandwidth, and long context - continue to generate new local modifications.

---

# Consolidated takeaways

1. The best architecture knowledge comes from controlled experiments; a broad survey of successful models is the next-best source when direct exploration is unaffordable.
2. Architecture choices jointly optimize statistical quality, hardware efficiency, and training stability, so apparently inelegant components can be rational responses to real constraints.
3. Modern models keep normalization outside the residual stream, preserving a direct path for activations and gradients.
4. RMSNorm and bias-free linear layers remove low-arithmetic-intensity work that appears to provide little benefit in standard language-model regimes.
5. The decisive feed-forward innovation is gating. SwiGLU and GeGLU usually outperform comparable non-gated activations, with intermediate width reduced by roughly $2/3$ for parameter matching.
6. Serial attention-then-MLP blocks remain the norm. Parallel blocks expose fusion opportunities but sacrifice within-block sequential composition.
7. RoPE rotates query/key coordinate pairs so their inner products depend on relative displacement. It has become the dominant positional mechanism, although partial and hybrid variants remain active research areas.
8. Conventional feed-forward ratios, attention projection ratios, and depth-to-width aspect ratios lie in broad, forgiving basins. Systems constraints often decide among statistically similar settings.
9. Vocabulary size is task- and data-dependent: multilingual and multimodal systems require broader coverage, while byte-normalized likelihood is needed for fair comparison across lossless tokenizers.
10. In one-pass compute-limited pretraining, weight decay can improve optimization even when overfitting is absent. Its effect cannot be separated from the learning-rate schedule.
11. Softmaxes are stability danger zones. Z-loss anchors the output log-partition function, QK norm controls attention-logit scale, and soft-capping provides a stronger but potentially quality-reducing bound.
12. Autoregressive decode is constrained by repeated memory access to parameters and the KV cache, not merely by total FLOPs.
13. MQA minimizes KV-cache size but can hurt quality; GQA usually captures most of the serving gain with much less expressive loss.
14. Long-context models increasingly interleave cheap local or recurrent layers with periodic global attention instead of using full attention in every layer.
15. Good model design fixes robust defaults and spends experimental budget on a small number of explicit hypotheses rather than changing the entire stack at once.

# Key equations

## Residual normalization layouts

Pre-norm:

$$
x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l)).
$$

Residual post-norm:

$$
x_{l+1}=\operatorname{Norm}(x_l+F_l(x_l)).
$$

Non-residual post-norm:

$$
x_{l+1}=x_l+\operatorname{Norm}(F_l(x_l)).
$$

## LayerNorm and RMSNorm

$$
\operatorname{LayerNorm}(x)
=\gamma\odot\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}+\beta,
$$

$$
\operatorname{RMSNorm}(x)
=\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}.
$$

## Gated feed-forward network

$$
\operatorname{GLU}_{\phi}(x)
=\bigl(\phi(xW)\odot xV\bigr)W_2.
$$

Parameter matching:

$$
d_{ff,\mathrm{gated}}
\approx\frac{2}{3}d_{ff,\mathrm{plain}}.
$$

## RoPE relative-position identity

$$
q_i^\top k_j
=q^\top R_i^\top R_jk
=q^\top R_{j-i}k.
$$

## Common shape defaults

$$
d_{ff}\approx4d_{model}
\quad\text{for a plain MLP},
$$

$$
d_{ff}\approx\frac{8}{3}d_{model}
\quad\text{for a parameter-matched gated MLP},
$$

$$
h\,d_{head}\approx d_{model},
$$

$$
\frac{d_{model}}{n_{layers}}\approx100
\quad\text{as a broad empirical neighborhood}.
$$

## Output z-loss

$$
Z(x)=\sum_{r'=1}^{|V|}e^{U_{r'}(x)},
$$

$$
\mathcal{L}_z=\alpha\bigl(\log Z(x)\bigr)^2.
$$

## QK-normalized attention

$$
A=\operatorname{softmax}\left(
\frac{\operatorname{Norm}(Q)\operatorname{Norm}(K)^\top}
{\sqrt{d_{head}}}
\right).
$$

## Logit soft-cap

$$
\operatorname{softcap}_c(a)
=c\tanh\left(\frac{a}{c}\right).
$$

## KV-cache size

$$
M_{KV}\approx2Lbnh_{kv}d_{head}s.
$$

- $L$: number of layers
- $b$: batch size
- $n$: cached sequence length
- $h_{kv}$: number of key/value heads
- $d_{head}$: head dimension
- $s$: bytes per stored element

## Prefill and decode arithmetic intensity

$$
I_{prefill}
=O\left(\left(\frac{1}{d_{head}}+\frac{1}{bn}\right)^{-1}\right),
$$

$$
I_{decode}
=O\left(\left(\frac{n}{d_{model}}+\frac{1}{b}\right)^{-1}\right).
$$

These are order-of-magnitude expressions from the lecture, not exact runtime models.

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Arithmetic intensity | Useful arithmetic performed per byte moved. High intensity makes it easier to use accelerator compute fully. |
| Aspect ratio | Here, $d_{model}/n_{layers}$, summarizing the width-versus-depth shape of a model. |
| Bias term | Learned additive constant in a linear or normalization layer; commonly omitted in modern dense Transformers. |
| Double norm | Normalization both before and after a learned branch while keeping the residual path unnormalized. |
| Feed-forward dimension | Intermediate width $d_{ff}$ inside a Transformer MLP. |
| Gated linear unit | Feed-forward structure that multiplies an activated projection by a second learned projection. |
| GeGLU | A gated MLP whose activated branch uses GeLU. |
| GQA | Grouped-query attention, with more query heads than key/value heads and one KV head shared within each query group. |
| Head dimension | Width of one attention head. |
| KV cache | Stored keys and values from previous tokens, reused during autoregressive decoding. |
| LayerNorm | Normalization that removes mean, scales by standard deviation, and usually applies learned scale and bias. |
| Logit soft-capping | Bounding logits smoothly with a scaled $\tanh$ before softmax. |
| MHA | Multi-head attention with a separate key/value head for every query head. |
| MQA | Multi-query attention, which uses many query heads but only one shared key head and value head. |
| NoPE | A layer with no explicit positional encoding in its attention computation. |
| Non-residual normalization | Any layout in which the identity skip path is not passed through a norm. |
| Parallel block | Transformer block where attention and MLP consume the same input and their outputs are added together. |
| Pre-norm | Normalizing a branch input before attention or the MLP, outside the residual path. |
| QK norm | Normalizing projected queries and keys immediately before their attention dot product. |
| RMSNorm | Magnitude normalization without LayerNorm's mean subtraction or additive bias. |
| RoPE | Rotary position embedding, which rotates query/key coordinate pairs as a function of position. |
| Serial block | Transformer block that computes attention and then passes the updated state through the MLP. |
| Sliding-window attention | Attention restricted to a fixed local neighborhood rather than the full context. |
| SwiGLU | A gated MLP whose activated branch uses Swish. |
| Weight decay | Parameter shrinkage applied during optimization; in this lecture its main value may be optimization rather than overfitting control. |
| Z-loss | Auxiliary penalty on the squared log softmax normalizer, used to control output-logit scale. |

# Self-check questions

1. Why does placing a norm outside the residual stream improve gradient propagation?
2. How are pre-norm, residual post-norm, and non-residual post-norm different?
3. Why can an operation representing only $0.17\%$ of FLOPs consume a much larger fraction of runtime?
4. What statistical and systems arguments support RMSNorm over LayerNorm?
5. Why does parameter matching require reducing the intermediate width of a gated MLP by roughly $2/3$?
6. What is the common structural property shared by ReGLU, GeGLU, and SwiGLU?
7. What systems opportunity does a parallel Transformer block expose, and what sequential computation does it lose?
8. Why are ordinary additive sine embeddings not purely relative inside an attention inner product?
9. Derive why $R_i^\top R_j$ in RoPE depends only on $j-i$.
10. Why is a feed-forward ratio between roughly 1 and 10 described as a forgiving basin rather than an exact optimum?
11. What does the rule $h\,d_{head}\approx d_{model}$ standardize, and why is it not mandatory?
12. Why can extremely deep models be harder to run efficiently than comparably sized wide models?
13. Why is token-level perplexity not directly comparable across different tokenizers, while bits per byte can be?
14. How can weight decay improve both training and validation loss when overfitting is absent?
15. What redundant degree of freedom in softmax makes z-loss possible?
16. How do QK norm and logit soft-capping differ in the strength and location of their constraints?
17. What is cached during autoregressive attention, and why are past queries normally not cached?
18. How do MHA, GQA, and MQA differ in the number of key/value heads?
19. Why must the number of KV heads be chosen during training rather than treated as a free serving switch?
20. How can a stack of local-attention layers transmit information beyond one window?
21. Why might a hybrid model use RoPE in local layers but NoPE in occasional global layers?
22. Which architecture decisions in the lecture appear settled, and which remain active long-context research questions?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-678 | Survey method, model landscape, lecture plan, architectural trade-offs, historical trends | Yes |
| 679-1201 | Residual versus non-residual normalization, warm-up, gradient propagation, double norm | Yes |
| 1204-1723 | LayerNorm, RMSNorm, runtime versus FLOPs, audience question, bias removal, empirical limits | Yes |
| 1726-2254 | Activation zoo, GLU construction, parameter matching, evidence, non-gated exceptions | Yes |
| 2257-2575 | Serial and parallel blocks, systems benefit, loss of depth, modern architecture summary | Yes |
| 2578-3226 | Positional-encoding families, RoPE objective, geometry, frequencies, math, implementation | Yes |
| 3229-3612 | Q&A on rotations, learning from reports, parallel-layer evidence, partial RoPE, factorization, transition | Yes |
| 3613-4259 | Hyperparameter-search motivation, feed-forward ratio, T5 exception, head dimensions | Yes |
| 4260-4570 | Depth-width aspect ratio, pipeline versus tensor parallelism, empirical forgiving range | Yes |
| 4573-4906 | Vocabulary-size trends, multilingual and multimodal vocabularies, bits-per-byte Q&A | Yes |
| 4909-5383 | Dropout, weight decay, single-pass training, optimizer interaction, diffusion and regularization Q&A | Yes |
| 5386-6112 | Stability motivation, softmax danger, z-loss, QK norm, logit soft-capping | Yes |
| 6115-6198 | Scope of attention interventions and transition from stability to inference | Yes |
| 6199-6568 | Prefill/decode accounting, autoregressive generation, KV cache, arithmetic intensity | Yes |
| 6571-6859 | MQA, expressiveness trade-off, GQA, empirical efficiency-quality compromise, MLA mention | Yes |
| 6862-7023 | Q&A on exploiting defaults, targeted search, dynamic hyperparameters, training with MQA | Yes |
| 7024-7357 | Sliding-window attention, Command A, RoPE/NoPE, hybrid models, Qwen 3.5, conclusion | Yes |
