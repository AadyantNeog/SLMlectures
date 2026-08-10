---
title: "Lecture 4 - Attention Alternatives and Mixtures of Experts"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 4
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 4 Attention Alternatives.txt"
slide_deck: "../lecture_04.pdf"
status: "complete"
---

# Lecture 4: Attention Alternatives and Mixtures of Experts

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves the lecturer's claims, examples, qualifications, numerical details, and substantive questions and answers while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, intuition, derivation, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are shown so the paraphrase can be audited. The complete transcript was mapped before the slide deck was inspected. All 60 slides were then rendered and checked to verify names, notation, equations, plots, and architectural diagrams. Material differences between the spoken lecture and the slides are called out explicitly.

## Lecture map

The lecture has two major parts:

1. **Attention alternatives:** why long contexts make full attention expensive; how associativity produces linear attention and its recurrent form; how Mamba-2 and Gated DeltaNet make that recurrence more expressive; why deployed systems remain hybrids; and how DeepSeek Sparse Attention uses cheap retrieval followed by full attention.
2. **Mixtures of experts:** how sparse expert activation increases total parameters without proportional FLOPs; how tokens are routed; why routing and load balancing are difficult; how expert parallelism interacts with hardware; and how the DeepSeek MoE designs combine modeling and systems choices.

---

# Part I - Why attention needs alternatives

## 1. Lecture scope and the pressure for longer context

**Transcript coverage:** lines 1-238

### What the lecturer said - transcript only

The previous lecture developed the basic Transformer and the modifications that turn it into a modern language model. This lecture moves to more advanced architectural changes in two places:

1. Modify the attention block so models can handle much longer contexts, ideally with cost that grows linearly rather than quadratically with sequence length.
2. Modify the MLP block with mixtures of experts (MoEs), increasing the number of parameters available to a model relative to the computation spent on each token.

Demand for long context is rising because users want to place more knowledge, documents, state, or agent history into a model's context. Across major model providers, advertised context windows have grown rapidly.

This growth changes the compute balance inside a Transformer. Feed-forward computation grows linearly with sequence length. Full attention creates all-to-all interactions between positions and grows quadratically. At shorter contexts and in large models, the feed-forward network may dominate total cost. As context becomes longer, attention catches up and eventually becomes the dominant problem.

### Additional explanation

For a sequence of length $n$, a dense attention score matrix contains $n^2$ entries. Doubling the context therefore produces roughly four times as many pairwise scores, before considering memory traffic or the value aggregation. By contrast, applying the same MLP independently to each position requires $n$ MLP evaluations, so its cost doubles.

Long context is useful only if the model can use it and the system can afford it. Architectural work therefore has at least three objectives: reduce asymptotic cost, reduce constant factors, and preserve the content-addressable retrieval ability that makes full attention powerful.

## 2. The existing toolkit: hybrid patterns and systems engineering

**Transcript coverage:** lines 241-523

### What the lecturer said - transcript only

There are already two important ways to control attention cost.

First, models can combine local and global attention. A model might use inexpensive local attention in most layers and full global attention only once every several layers. This preserves occasional global communication while sharply lowering the total cost.

Second, systems engineering can deliver very large constant-factor gains. The lecturer warns against considering only big-O complexity. FlashAttention rearranges the attention computation to minimize transfers between accelerator memory and compute units. It does not eliminate quadratic arithmetic, but it can approximately double achieved throughput in the examples shown. It also avoids materializing the entire attention matrix, allowing some sequence lengths to run that otherwise would not fit in memory.

Constant factors are therefore powerful, but local attention and optimized kernels may still be insufficient for contexts of five or ten million tokens. At that scale, a more radical question becomes important: can attention have linear dependence on sequence length?

Researchers tried many approaches over the years, with several false starts. The lecturer says that recipes developed over roughly the last two years have now been tested at meaningful scale and in production. The common starting point needed to understand them is simply the associativity of matrix multiplication.

### Additional explanation

The two toolkits solve different bottlenecks:

- A local/global pattern changes which token pairs are allowed to interact.
- FlashAttention computes essentially the same dense attention function with a better schedule for memory movement.
- Linear or recurrent alternatives change the function class so the model does not explicitly maintain every pairwise interaction.

These approaches can be combined. A production model may use optimized kernels for its remaining full-attention layers, local windows in other layers, and recurrent layers elsewhere. Big-O analysis identifies how a method behaves as $n$ grows; hardware-aware implementation determines whether that theoretical advantage appears at practical sizes.

---

# Part II - Linear attention and recurrent state

## 3. Associativity turns quadratic attention into linear attention

**Transcript coverage:** lines 526-717

### What the lecturer said - transcript only

Let

$$
Q \in \mathbb{R}^{n \times d_k}, \qquad
K \in \mathbb{R}^{n \times d_k}, \qquad
V \in \mathbb{R}^{n \times d_v}.
$$

The usual attention operation can be written as

$$
\operatorname{Attn}(Q,K,V) = \rho(QK^\top)V,
$$

where $\rho$ is the row-wise normalization, normally softmax. Computing $QK^\top$ creates interactions between all $n$ positions and costs on the order of $n^2d_k$; multiplying the resulting matrix by $V$ adds a term on the order of $n^2d_v$.

Temporarily remove the softmax by taking $\rho$ to be the identity. Matrix multiplication is associative, so

$$
(QK^\top)V = Q(K^\top V).
$$

The original parenthesization creates an $n \times n$ matrix. The reordered computation first creates a $d_k \times d_v$ summary. Its approximate cost is

$$
2nd_kd_v,
$$

which is linear in $n$. The dependence moves from the square of sequence length to the product of the key and value dimensions. Those dimensions can be large, but they are typically thousands or tens of thousands rather than millions. When context length is extremely large, this is much more favorable.

### Additional explanation

The algebraic identity is exact only after removing the nonlinear normalization. Softmax prevents the parentheses from moving through it:

$$
\operatorname{softmax}(QK^\top)V
\ne
Q\operatorname{softmax}(K^\top V).
$$

That is the central bargain of basic linear attention: it gains a compact sufficient statistic but gives up the exact content-dependent normalization of standard attention.

The state matrix $K^\top V$ can be interpreted as a collection of key-value associations. Each outer product $k_t v_t^\top$ writes information into that matrix. A query then reads from the accumulated matrix. This view leads directly to the recurrent formulation.

## 4. The recurrent form and training-inference duality

**Transcript coverage:** lines 724-879

### What the lecturer said - transcript only

For causal self-attention, the reordered linear operation can be written incrementally:

$$
S_t = S_{t-1} + k_t v_t^\top,
$$

$$
y_t = q_t^\top S_t.
$$

The state $S_t$ accumulates key-value outer products while the model moves from left to right. This looks like an RNN. At inference time, the model need only carry a fixed-size state forward rather than retain a key-value entry for every prior position.

The useful feature is a duality between two computational forms. Training can use a parallel or dense formulation that maps well to accelerator operations, while autoregressive inference can use the serial recurrence and its fixed-size state. This offers the training parallelism associated with attention and the compact inference state associated with an RNN.

The lecturer cautions that the ungated operation is still a simple linear recurrence and is not expressive enough by itself. It is the starting point for more capable mechanisms.

### Additional explanation

The state has shape $d_k \times d_v$, independent of the number of past tokens. During decoding, each new token performs an outer-product update and a matrix-vector read. Standard attention instead grows a KV cache with context length and rereads that cache at each step.

Fixed state is also a compression bottleneck. Two different histories that map to the same $S_t$ become indistinguishable to later computation. Full attention preserves a separately addressable representation for every earlier token; linear attention compresses all earlier tokens into one matrix. Later gates and erase operations improve this memory, but do not remove the finite-state tradeoff.

The word "duality" here means two equivalent implementations of the same recurrence, not equivalence to softmax attention. The loss of equivalence happens when softmax is removed.

## 5. MiniMax M1: basic linear attention used as a hybrid

**Transcript coverage:** lines 889-987

### What the lecturer said - transcript only

MiniMax M1 is offered as evidence that even fairly basic linear attention can work in a large, strong open model. It uses a 7:1 hybrid: seven linear-attention layers for every one full softmax-attention layer. Its benchmark performance is generally competitive with strong contemporary models, and its compute grows much more gently with context length than a model built entirely from full attention.

Because the model still includes periodic full-attention layers, it is not fully linear in context length. The lecturer says that no fully linear-time attention system has yet been convincingly demonstrated at scale. Every large-scale example in this part of the lecture is a hybrid.

### Additional explanation

Hybridization lets the two layer types compensate for each other:

- Recurrent layers cheaply propagate and transform a compressed summary.
- Full-attention layers allow precise token-to-token retrieval and can repair information that would otherwise be lost in the recurrent bottleneck.

A 7:1 ratio does not imply an eightfold end-to-end speedup. Actual gains depend on the fraction of cost attributable to attention, the dimensions of the recurrent state, kernel quality, sequence length, and the cost of the remaining full-attention layers. It does, however, reduce the number of layers paying the full quadratic and KV-cache costs.

## 6. Mamba-2 as gated linear attention

**Transcript coverage:** lines 991-1250

### What the lecturer said - transcript only

Mamba, Mamba-2, and related state-space models were originally motivated through state-space theory, but their mechanics can also be understood as a small elaboration of linear attention.

The main weakness of the basic recurrence is that it always carries all of its previous state forward. Drawing on the lesson of LSTMs, Mamba-2 adds an input-dependent forget gate:

$$
S_t = \gamma_t S_{t-1} + k_t v_t^\top,
\qquad \gamma_t = f(x_t).
$$

$\gamma_t$ depends only on the current input, not on the previous state. It controls how much old state survives. Because this gate is input-dependent rather than state-dependent, it can be computed in parallel and the same training-inference duality remains available.

For completeness, the full readout shown in the lecture also contains a direct value path:

$$
y_t = q_t^\top S_t + v_t^\top D.
$$

The lecturer describes $v_t^\top D$ as an architectural residual-like term rather than the core state-update idea. It passes a gated amount of the current value directly to the output.

Nemotron 3 combines Mamba-2 as its lightweight layer with occasional full softmax attention. Its performance is competitive with similarly sized open models, and its throughput remains favorable at long context. The lecturer calls these small frontier models rather than the largest frontier systems.

### Additional explanation

If $\gamma_t$ is near zero, old state is largely forgotten; if it is near one, it is retained. Repeated gates create an input-dependent decay:

$$
S_t
=
\sum_{i \le t}
\left(\prod_{j=i+1}^{t}\gamma_j\right)
k_i v_i^\top.
$$

This makes the memory selective while preserving a scan structure. During training, prefix products and weighted sums can be evaluated with parallel algorithms. During decoding, only the current state and gate are needed.

The direct $D$ path serves a different purpose: it gives the layer immediate access to the current token representation without requiring that information to be written into and then read out of the recurrent state.

## 7. Gated DeltaNet: gated writing and selective erasure

**Transcript coverage:** lines 1255-1602

### What the lecturer said - transcript only

The recurrence can be elaborated further while retaining the parallel/serial duality, provided its gates depend on the current input rather than on the recurrent state. Gated DeltaNet is a widely used example and appears in scaled open models.

Relative to Mamba-2, it adds a second gate $\beta_t$. The lecturer interprets this as a "no input operation" gate: when $\beta_t=0$, the current token does not write new information into the state.

Its update is

$$
S_t
=
\gamma_t\left(I-\beta_t k_tk_t^\top\right)S_{t-1}
+
\beta_t k_t v_t^\top,
$$

$$
y_t=q_t^\top S_t,
\qquad
\gamma_t=f(x_t),
\qquad
\beta_t=f(x_t).
$$

The factor $I-\beta_tk_tk_t^\top$ approximately projects the previous state away from the current key direction before writing the new key-value association. Thus an update can erase old information associated with the key and replace it, rather than merely add another contribution. The lecturer notes that calling it an exact projector is only an intuition because the displayed expression does not include all conditions, such as unit normalization, required for an exact orthogonal projection.

Closely related updates have been rediscovered through other motivations, including meta-learning least-squares problems, fast-weight programming, and test-time training.

### Additional explanation

Without erasure, repeatedly writing different values under similar keys causes interference because the state accumulates all of them. The delta rule instead resembles an online correction:

1. Read what the state currently predicts in direction $k_t$.
2. Remove part of that prediction.
3. Write the new target value $v_t$.

When $k_t$ is normalized and $\beta_t=1$, $I-k_tk_t^\top$ removes the component aligned with $k_t$. With fractional $\beta_t$, the overwrite is partial. This is why the update connects naturally to online least-squares and fast-weight interpretations.

## 8. Evidence and limits of hybrid recurrent architectures

**Transcript coverage:** lines 1606-1816

### What the lecturer said - transcript only

Qwen 3.5 and its Qwen Next predecessors use a 3:1 Gated DeltaNet/full-attention hybrid. They are among the strongest open models available at lecture time. Their evaluation performance is close to strong closed and previous-generation models, while decode throughput improves substantially over Qwen 3 as context length grows. The hybrid therefore appears to preserve most model quality while improving inference characteristics.

The lecturer also discusses one of the few controlled studies of hybrid ratios, from ByteDance Seed and UC Santa Cruz. The results are somewhat messy, but the overall pattern is consistent:

- At low ratios of recurrent to full-attention layers, the strongest recurrent designs show little or no quality loss.
- As the fraction of recurrent layers increases, long-context retrieval and question-answering performance decline.
- Pure recurrent models show clear degradation across the tested architectures.
- Some single-key retrieval results may be unusually favorable because long-context architectures are often explicitly optimized for that task.

The evidence therefore supports hybrids, not a claim that recurrent layers are universally equivalent to full attention.

### Additional explanation

Hybrid ratio is a resource-quality control knob. A low recurrent fraction saves less compute but asks the recurrent state to carry less of the model's global communication burden. A high recurrent fraction saves more but increases the chance that information is compressed away before a full-attention layer can recover it.

Retrieval benchmarks also probe different failure modes. Finding one distinctive key can be easier for a specialized memory update than integrating many dispersed facts, resolving contradictions, or answering a question requiring several pieces of evidence. This explains why a mechanism can look strong on a needle-style task yet lose ground on QA.

## 9. Questions on equivalence and the direct value path

**Transcript coverage:** lines 1822-2011

### What the lecturer said - transcript only

An audience question asks whether the recurrent and parallel forms should be equivalent and therefore avoid a performance loss. The lecturer distinguishes two transformations:

1. Removing the softmax normalization from full attention to obtain linear attention is lossy and changes the function.
2. Rewriting that linear attention as a parallel matrix operation or as a recurrent update is exact.

Thus any quality gap relative to full attention begins with dropping softmax, not with using the recurrent implementation.

A second question asks about the direct $v_t^\top D$ term in Mamba-2. The lecturer explains that ordinary linear attention accumulates key-value information and reads it with the query. The extra term lets information from the current token's value flow directly into the output, much like a residual connection. $D$ controls the amount of that direct pass-through.

The lecturer closes the section by observing that many successful recent methods have converged on simple linear-attention recurrences with LSTM-like gates, despite being derived from different viewpoints.

### Additional explanation

It is useful to keep three comparisons separate:

- **Full attention vs. linear attention:** different functions because normalization and explicit token memory differ.
- **Parallel linear attention vs. recurrent linear attention:** the same function computed in different orders.
- **Ungated vs. gated recurrence:** different functions, with the latter adding input-dependent memory control.

Confusing these levels can produce contradictory statements such as "the recurrence is exact" and "the recurrence loses quality." The recurrent implementation is exact relative to its parallel linear counterpart, while the linear mechanism remains an approximation or alternative to softmax attention.

---

# Part III - Sparse selection as an attention alternative

## 10. DeepSeek Sparse Attention

**Transcript coverage:** lines 2023-2398

### What the lecturer said - transcript only

DeepSeek Sparse Attention (DSA), used in DeepSeek V3.2, takes a different route from recurrent linear attention. Rather than compressing the entire past into a state, it first uses a lightweight indexer to select a small subset of earlier tokens and then performs ordinary full attention over that subset.

The indexer starts from query-key interactions, applies a ReLU, combines a small number of indexer-head scores using weights derived from the query token, and takes the top $K$ positions. In the notation verified from the slide,

$$
I_{t,s}
=
\sum_{j=1}^{H'} w_{t,j}^{I}
\operatorname{ReLU}\left((q_{t,j}^{I})^\top k_s^{I}\right).
$$

Only the highest-scoring positions are admitted to the expensive attention computation. If the indexer is small, low-dimensional, or low-precision, its all-position scan is much cheaper than full attention over the same context.

The model need not be pretrained with DSA from the beginning. DeepSeek and GLM results show a practical recipe: pretrain a normal shorter-context Transformer, insert the indexer during a long-context extension stage, and train the model to adapt to sparse selection. This avoids paying the added routing complexity throughout initial pretraining.

DeepSeek V3.2 remains competitive with frontier models while showing much more favorable prefill and decode cost as token position grows. GLM-5 provides an independent adoption of the same approach and reports small losses relative to full attention, including on difficult long-context retrieval tasks where recurrent architectures can struggle.

DSA is not linear-time. Its indexer still evaluates query-key interactions across the available context. Its advantage comes from making that quadratic-looking selection operation cheap and restricting expensive value attention to top-$K$ positions.

### Additional explanation

DSA separates **retrieval** from **high-fidelity computation**:

1. A cheap representation answers, "Which locations might matter?"
2. Full-precision attention answers, "How should the selected values be combined?"

This resembles an information-retrieval system with a lightweight first-stage retriever and a more expensive reranker. It preserves separately addressable token memories, unlike a fixed recurrent state, but its success depends on recall: a token omitted by the indexer cannot contribute to that attention output.

Bounding $K$ makes the expensive attention phase scale with a fixed selected set rather than with the full context. The indexer's theoretical sequence dependence remains important at extreme lengths, but its small constant can make it practical over the ranges being targeted.

## 11. Questions on why a quadratic indexer can still help

**Transcript coverage:** lines 2404-2626

### What the lecturer said - transcript only

An audience member asks for the indexer's time complexity. The lecturer confirms that it is quadratic: to decide which tokens to select, it performs brute-force inner products and has no recurrent state-transition shortcut.

The follow-up asks how the method helps if both stages involve attention-like computation. The lecturer gives two constant-factor reasons:

- The indexer can use lower precision and lower-dimensional query and key projections, so its all-pairs computation is much cheaper.
- The second stage operates only on a controllable top-$K$ subset. It may use full attention, but on a much shorter effective context.

This is another warning not to treat "quadratic" as a complete performance description. One quadratic operation can be cheap, while another with larger dimensions and value aggregation is expensive.

Further questions clarify the training schedule. Long-context models are generally not trained from scratch at their maximum context. A typical order is short-context pretraining, long-context extension, and then post-training. DSA can be inserted during the long-context extension that would happen anyway. Although top-$K$ selection is non-differentiable and initially looks risky, the later MoE discussion shows that similar discrete selection can work in large models.

$K$ is chosen closer to a short-context length and is bounded independently of the full input length.

### Additional explanation

Suppose the indexer uses dimension $d_I$ while full attention uses head dimension $d$. Its score work is roughly proportional to $n^2d_I$, but the expensive selected attention is proportional to $nKd$. If $d_I \ll d$ and $K \ll n$, both terms can be much smaller than full $n^2d$ attention even though the first is still quadratic in $n$.

The staged recipe also reduces optimization risk. The base model first learns general language modeling with dense connectivity. Long-context extension then teaches the indexer to imitate or support useful long-range behavior without forcing every earlier phase of training through a discrete bottleneck.

## 12. Questions on stability, future designs, precision, and expressivity

**Transcript coverage:** lines 2629-3052

### What the lecturer said - transcript only

Several audience questions close the attention section.

**Does removing softmax create training-instability problems?** The lecturer is not aware of documented evidence that linear attention without softmax causes major instability. If anything, softmax itself often creates numerical problems, so removing it may improve stability.

**What will future attention mechanisms look like?** The lecturer does not offer one predicted winner. A likely direction is to combine several successful tricks, as modern architectures already do. Above the architecture, post-training can teach a model to manage its own context through compaction, retrieval, and related operations. Future work may therefore focus on integrating architectural and context-management layers. The recurrent side has already converged substantially toward LSTM-like or linear-attention mechanisms, and the lecturer does not expect an immediate radical departure.

**Can attention run in FP4?** Very low-precision attention is possible, and the DSA indexer is partly motivated by this opportunity. Full softmax attention is harder because small underflow, overflow, or rounding errors can significantly change probabilities. A sensible compromise is low-precision selection followed by higher-precision attention, so value vectors are combined with more accurate weights. The lecturer does not know of a generally established recipe for complete FP4 attention.

**What is the downside of state-space models relative to Transformers?** Full softmax attention provides a powerful, easy-to-train all-to-all connection. Earlier RNNs also lacked efficient parallel training, but linear-attention duality now permits dense matrix operations during training and a recurrence during inference, substantially addressing that hardware-efficiency problem. The remaining drawback is representational: a finite state must compress the whole context. A state as large as the context could preserve everything, but would forfeit the cost advantage. A very small state cannot currently retain all information from an arbitrarily large context without tradeoffs.

### Additional explanation

The answers identify three distinct bottlenecks:

- **Numerical:** exponentials and normalization are sensitive in low precision.
- **Computational:** all-to-all access is expensive at long context.
- **Information-theoretic:** a bounded state cannot injectively represent every possible unbounded history.

Retrieval, recurrent state, local windows, and full attention occupy different points in this design space. Retrieval keeps explicit memories but may fail to select them. Recurrence guarantees constant-size state but must compress. Full attention preserves direct access but pays in time and memory. Hybrid systems use each mechanism where its tradeoff is most acceptable.

---

# Part IV - Why mixtures of experts are attractive

## 13. An MoE is a sparsely activated replacement for an MLP

**Transcript coverage:** lines 3055-3291

### What the lecturer said - transcript only

Conceptually, an MoE does not change the entire Transformer. It can be viewed as a more efficient MLP. It is important because most large open models now use MoEs and because its primitives - routing, sparse selection, and conditional computation - recur in other neural architectures.

Start with a Transformer's ordinary feed-forward network. Replace it with several feed-forward networks, called experts, and add a mechanism that chooses which expert processes each token. If there are four experts, each as large as the original MLP, the layer contains approximately four times as many MLP parameters. If only one expert is activated for a token, that token still pays roughly one MLP's computation in the forward and backward pass.

The central mental model is therefore:

> Increase total parameters without increasing per-token FLOPs by the same factor.

This was a major motivation in early MoE work: if more parameters are useful but dense computation is unaffordable, conditionally activate only a subset.

### Additional explanation

MoE introduces a distinction that is absent in a dense model:

- **Total parameters:** all weights stored across every expert.
- **Activated parameters:** weights used for one token's computation.

Activated parameters are a useful proxy for arithmetic, but total parameters still affect memory capacity, checkpoint size, communication, and deployment complexity. MoE is therefore not "free parameters." It trades dense arithmetic for sparse memory and routing infrastructure.

Conditional computation also changes the function class. Different tokens can follow different parameter paths, allowing the model to allocate representational capacity nonuniformly rather than forcing one MLP to serve every token.

## 14. Empirical gains at fixed active compute

**Transcript coverage:** lines 3292-3540

### What the lecturer said - transcript only

MoEs have become popular because increasing sparse parameters while keeping active compute approximately fixed usually improves the model. Results from the Switch Transformer show language-model test loss falling as expert count grows while active parameters remain comparable. At the same training compute, models with more experts also reach better performance.

The OLMoE study reports the same general pattern in training loss, validation loss, and downstream benchmarks. In the spoken comparison, its MoE trains roughly twice as fast as a dense counterpart at comparable quality.

Released models provide another form of evidence. DeepSeek's earlier MoEs achieved competitive or better MMLU performance than dense models while activating far fewer parameters. The lecturer treats this as evidence that total sparse capacity remains valuable even when only a fraction is active for any token.

The claim is not that all parameters are simultaneously useful on every input. Rather, conditional access to a larger parameter pool improves the accuracy available for a given amount of active computation.

### Additional explanation

One way to interpret the gain is parameter reuse. A dense MLP must represent every transformation in one set of weights. An MoE can distribute incompatible or infrequently needed transformations across different experts, while a router selects a small combination for each token.

Comparisons must hold the right quantity fixed. A sparse model can look inexpensive when measured by FLOPs yet require much more accelerator memory or network capacity. Fair evaluation should report at least total parameters, activated parameters, training tokens, FLOPs, wall-clock time, hardware, and serving conditions.

## 15. Expert parallelism and the model ecosystem

**Transcript coverage:** lines 3541-3741

### What the lecturer said - transcript only

MoEs add another axis of parallelism. Each expert is a natural chunk that can be placed on a different device. Activations are then routed to the devices hosting the selected experts. This expert-parallel layout can complement data, tensor, and pipeline parallelism when a model is too large for one device.

In the Western open-model ecosystem, Llama 4 and GPT-OSS are cited as strong large MoE releases, although the lecturer says major Western open releases have slowed. Much of the architecture and training work has come from Chinese groups. Qwen, DeepSeek, MiniCPM, and others produced early strong MoEs and continued to develop them. Qwen 1.5 MoE, with about 2.7 billion activated parameters, outperformed many dense 7B models of its period. The lecturer argues that evidence from Qwen and DeepSeek helped persuade the open community that MoE was the right direction for larger models.

### Additional explanation

Expert parallelism aligns naturally with conditional computation but introduces an all-to-all communication pattern:

1. Each device begins with a local batch of token activations.
2. The router assigns tokens to experts, potentially on other devices.
3. Devices exchange activations.
4. Experts process their assigned tokens.
5. Outputs are exchanged back to their original sequence locations.

The approach is attractive when aggregate accelerator compute and memory outweigh the communication cost. Its efficiency depends strongly on topology: communication within a tightly connected node can be much cheaper than communication across nodes.

## 16. Questions about communication, sparse training, and routing granularity

**Transcript coverage:** lines 3742-4000

### What the lecturer said - transcript only

An audience member asks whether expert parallelism creates a communication bottleneck. The lecturer agrees: activations must be shipped to expert devices. The trade is more aggregate FLOPs and lower per-device memory pressure in exchange for communication. Whether it is a net win depends on network topology and implementation.

Another question asks whether training activates all experts. It does not. Training is sparse too; otherwise it would pay the full expert FLOPs and lose the point of MoE. This creates the central learning difficulty: the model observes outcomes only for the selected experts, yet must learn which experts should have been selected. The lecturer compares this to a reinforcement-learning or bandit problem, though practical systems solve it mostly with heuristics rather than RL.

Routing happens at token granularity. Each token selects experts independently. Routers are usually extremely simple, often a single learned matrix multiplication between the token representation and one vector per expert. They do not perform high-level reasoning such as recognizing a complete medical question. They may instead react to local features, for example that a token resembles Japanese text.

There is also an upper limit to expert parallelism. As a model is sharded across more devices, communication eventually dominates. Efficient sharding therefore depends on the available networking topology.

### Additional explanation

Sparse training creates partial feedback. If expert 7 is chosen and produces a good update, ordinary backpropagation can reinforce expert 7 and its gate. It provides no direct evidence about how expert 12 would have performed on the same token. This is why routing resembles a contextual bandit even though the production solution is usually not a bandit algorithm.

Token-level routing is computationally convenient because it applies the same small router independently at every position. It also explains why learned experts often correlate with surface categories such as punctuation or script rather than with broad document-level occupations.

## 17. Why MoEs were slow to become standard

**Transcript coverage:** lines 4003-4264

### What the lecturer said - transcript only

The lecturer recommends early DeepSeek MoE papers for careful architecture science. They compare dense layers, hash routing, Switch-style routers, and other choices through ablations. DeepSeek V3, V3.2, GLM, and related systems later scaled these ideas into strong models. The lecturer expects large models to remain MoE-based for at least the next several years.

Despite promising work by 2022, widespread adoption accelerated only around 2024. The delay reflects practical complexity:

- infrastructure and expert parallelism are difficult to implement efficiently;
- the large total parameter set is hard to fit on a single device;
- routing and load imbalance complicate training;
- MoE runs can become numerically unstable or collapse.

There are also proposals to apply MoE routing to attention heads, but these are much less common and have been harder to control. The production norm is to replace the feed-forward or MLP sublayer, which is the focus of the lecture.

### Additional explanation

An architecture can be compute-efficient on paper yet operationally unattractive. MoE requires coordinated support from model code, sparse kernels, distributed collectives, checkpointing, serving schedulers, and fine-tuning tools. Mature dense-Transformer infrastructure lowered risk for many teams even when its FLOPs were higher.

The MLP is a particularly convenient target because it processes positions independently. Routing tokens among independent expert MLPs does not alter the attention graph. Routing attention heads changes how tokens communicate and therefore couples sparsity with the model's global information flow, making failures harder to isolate.

---

# Part V - The MoE design space and routing

## 18. Three axes of MoE design

**Transcript coverage:** lines 4267-4342

### What the lecturer said - transcript only

The lecturer organizes MoE design around three axes:

1. **Routing function:** how tokens and experts are matched.
2. **Expert size and count:** for a fixed budget, whether to use a smaller number of large experts or a larger number of fine-grained experts.
3. **Training objective:** how to learn useful routes and prevent expert collapse despite sparse, discrete selection.

All practical MoEs preserve sparse activation. If every expert were evaluated, choosing among them would be easy, but the layer would pay the full dense compute cost. The design problem is therefore not merely to create multiple MLPs; it is to learn conditional use of them under a hard compute constraint.

### Additional explanation

These axes interact. More experts give the router finer choices but make load balancing and device placement harder. Larger top-$K$ improves redundancy and gradient coverage but increases activated FLOPs and communication. Stronger balancing makes hardware use predictable but may prevent the router from assigning naturally popular tokens to the same expert.

It is useful to treat an MoE configuration as a joint modeling-systems object rather than choosing each hyperparameter independently.

## 19. Token choice, expert choice, and global assignment

**Transcript coverage:** lines 4345-4477

### What the lecturer said - transcript only

Sparse routing can assign tokens and experts in three broad ways:

- **Token choice:** each token selects its top-$K$ experts.
- **Expert choice:** each expert selects its preferred tokens.
- **Global assignment:** an optimizer considers all token-expert scores together and chooses a joint assignment.

Almost all current large MoEs use token-choice top-$K$ routing. In the OLMoE comparison shown, token choice achieves lower validation loss and better downstream scores than expert choice. Expert-choice models can train successfully, but token choice has been easier to make work and is the standard in deployed architectures. The lecturer mentions a possible Llama 4 expert-choice design but immediately qualifies that recollection and does not treat it as strong evidence.

### Additional explanation

The routing direction determines which constraint is easiest to satisfy:

- Token choice guarantees that every token receives the requested number of experts, but expert loads can be highly uneven.
- Expert choice gives every expert a bounded token set, but some tokens may receive too many experts and others too few.
- Global assignment can enforce both token and expert capacities, but solving a matching problem for every batch is expensive.

Token choice prioritizes per-token model quality, then uses auxiliary mechanisms to repair load imbalance. That ordering is consistent with the empirical preference reported in the lecture.

## 20. Router families: learned scores, hashes, RL, and matching

**Transcript coverage:** lines 4495-4771

### What the lecturer said - transcript only

The most common router is a learned linear projection. Each expert has a vector, and a token receives one score per expert from an inner product with those vectors. The top-scoring experts are selected. Variants of this design appear in Switch Transformer, GShard, Grok, Mixtral, Qwen, DBRX, DeepSeek, and other systems, with different choices of $K$.

Surprisingly, learned routing is not strictly necessary to obtain an MoE gain. A token representation can be hashed to an expert, and hash routing often improves over a dense baseline. It generally does not match learned top-$K$ routing and is used more as a research baseline than as a production choice.

RL is a conceptually natural option. Selecting one or several experts without observing the unselected alternatives resembles a bandit problem, so the router can be treated as a policy. Early work, including Bengio's 2013 work, explored this route. It is uncommon now because RL adds stochasticity, gradient variance, and algorithmic overhead. Simple top-$K$ routing plus heuristics works well enough that the extra machinery is not justified.

A global linear-assignment problem offers the most principled assignment: compute every token-expert compatibility score and solve for the best joint matching. It can work, but its compute cost has prevented broad use at scale.

### Additional explanation

These alternatives trade optimization sophistication for throughput:

| Router | Learns specialization | Balances load directly | Main cost or weakness |
|---|:---:|:---:|---|
| Hash | No | Only through hash design | Cannot adapt routes to task loss |
| Linear top-$K$ | Yes | No | Needs balancing heuristics |
| RL policy | Yes | Can encode it in reward | High variance and complexity |
| Global matching | Yes | Yes | Expensive batch-level optimization |

Top-$K$ wins because its router is tiny, easy to parallelize, and differentiable for the selected scores almost everywhere, even though the selection boundary itself is discrete.

## 21. Top-$K$ routing equations

**Transcript coverage:** lines 4777-4896

### What the lecturer said - transcript only

For token representation $u_t^\ell$ in layer $\ell$, a standard MoE residual update is

$$
h_t^\ell
=
u_t^\ell
+
\sum_{i=1}^{N} g_{i,t}\operatorname{FFN}_i(u_t^\ell).
$$

The router computes a score for each expert using an inner product between the token and an expert vector, followed by normalization:

$$
s_{i,t}
=
\operatorname{softmax}_i\left((u_t^\ell)^\top e_i^\ell\right).
$$

Only the top-$K$ scores remain active:

$$
g_{i,t}
=
\begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{TopK}(\{s_{j,t}\}_{j=1}^{N},K),\\
0, & \text{otherwise}.
\end{cases}
$$

Thus the router is only a lightweight learned scoring operation. The selected experts' outputs are weighted by the gate and added to the residual stream. The lecturer emphasizes that the same top-$K$ pattern appeared earlier in DSA and also appears in architectures such as H-Net.

### Additional explanation

Implementations vary in whether normalization happens before or after top-$K$. If softmax is applied over all experts first, dropped experts still affect the selected experts' unrenormalized mass. If top-$K$ is applied first and softmax only over selected logits, active weights sum to one within the selected set. The lecture later notes another DeepSeek V3 variation that uses sigmoid scores and then normalizes selected gates.

The expert-vector matrix has shape approximately $d_{model} \times N$, usually tiny compared with the expert MLP weights. Routing overhead comes less from computing scores than from reorganizing and communicating the selected token activations.

## 22. Fine-grained and shared experts

**Transcript coverage:** lines 4900-5209

### What the lecturer said - transcript only

The lecturer attributes two influential design choices to DeepSeek MoE.

First, split a few large experts into a larger number of smaller, fine-grained experts. A token can then choose a more precise combination while keeping activated computation similar.

Second, designate one or more **shared experts** that process every token and bypass the router. Common computation can live in the shared expert, leaving routed experts freer to specialize. DeepSeek's ablations show gains from both finer segmentation and shared-expert isolation, with especially visible improvements on TriviaQA and Natural Questions.

OLMoE provides a partially conflicting result. Its controlled study finds gains from increasing expert count but little benefit from converting a routed expert into a shared one. The lecturer therefore treats fine-grained experts as strongly supported while acknowledging disagreement about the marginal value of shared experts.

Despite that disagreement, the combination of fine-grained routed experts and always-on shared experts became a common modern design in DeepSeek, Qwen, GLM, and related models. The lecturer compares its influence on MoEs to the way Llama's dense architecture became a common template.

### Source reconciliation

The spoken lecture says that DeepSeek MoE pioneered the shared-expert idea. Slide 32 labels the design as used by DeepSeek and Qwen but says that it originated in DeepSpeed MoE. The notes preserve the spoken attribution here and record the slide's different attribution rather than silently resolving it.

### Additional explanation

Fine-grained experts increase combinatorial capacity. Selecting 4 of 32 small experts can express many different paths while activating a controlled amount of MLP width. A shared expert provides a dense baseline path, reducing pressure on routed experts to duplicate universally useful operations.

The OLMoE result shows why an architectural intuition is not enough. A shared expert can help when it removes duplicated common features, but can be neutral when routed experts already learn those features efficiently or when the comparison reallocates capacity unfavorably.

## 23. Question on shared experts and parallelism

**Transcript coverage:** lines 5215-5254

### What the lecturer said - transcript only

An audience member asks how shared experts affect parallelization. Shared experts do not receive sparse-routing savings: every activation must pass through them. One systems option is to replicate a shared expert on multiple devices. Replication consumes more memory but avoids communicating every token to a single shared-expert device.

### Additional explanation

This is a standard memory-communication exchange. Sharding one shared expert minimizes duplicated weights but creates a communication hotspot. Replicating it makes local computation possible but stores the same weights repeatedly. The best choice depends on expert size, device memory, batch size, and interconnect bandwidth.

---

# Part VI - Training sparse, discrete routers

## 24. Why sparse MoE training is difficult

**Transcript coverage:** lines 5260-5404

### What the lecturer said - transcript only

Training must remain sparse; activating all experts would pay the FLOPs of the full parameter set. Sparse top-$K$ decisions, however, are not differentiable at the selection boundary, and the model does not observe counterfactual outputs from experts it did not choose.

Three broad solutions are possible:

1. Optimize routing as an RL policy.
2. Add stochastic perturbations to explore nearby expert choices.
3. Ignore much of the formal discrete-optimization problem and use heuristic balancing losses.

Production MoEs overwhelmingly use the third option. The surprising lesson is that a collection of simple deep-learning heuristics is sufficient to train large sparse models robustly.

### Additional explanation

Backpropagation still works through an active expert's computation and through its nonzero gate weight. What it cannot do directly is differentiate the identity of the selected set at a top-$K$ boundary. Practical training uses the gradient available on the active path and adds objectives that shape the routing distribution globally.

This is an example of a broader deep-learning pattern: a formally awkward discrete choice can work when continuous scores, large batches, noisy optimization, and auxiliary objectives provide enough useful signal.

## 25. RL and stochastic routing approximations

**Transcript coverage:** lines 5407-5686

### What the lecturer said - transcript only

REINFORCE-style routing does work. In the Clark 2020 comparison, it is a viable baseline, but its gradient variance and complexity keep it from being the best or most widely used method.

An early alternative from Shazeer and colleagues adds Gaussian noise to router logits before top-$K$ selection:

$$
H(x)_i
=
(xW_g)_i
+
\epsilon_i\operatorname{softplus}((xW_{noise})_i),
\qquad
\epsilon_i \sim \mathcal{N}(0,1).
$$

The router keeps the top $K$ values of $H(x)$ and applies softmax. When two experts have similar scores, noise lets either one be selected. This supports limited exploration and breaks ties; backpropagation can then strengthen experts that prove useful. Soft weights among the selected experts also let the model learn their relative ranking rather than making an unweighted hard choice.

A related Switch-era method applies uniform multiplicative jitter to router inputs or logits to make experts less brittle. Later Google ablations removed the perturbation and found that no stochastic jitter could improve both stability and final quality. The lecturer concludes that explicit exploration noise is not generally necessary.

### Additional explanation

Noise is helpful only if exploration reveals useful alternatives faster than it disrupts specialization. Early in training, expert scores are close and perturbations can distribute learning signal. Later, persistent noise can route tokens to inferior experts and add variance to an already unstable distributed system.

Load balancing, discussed next, offers a deterministic source of pressure toward underused experts. It often supplies enough diversity without randomizing individual routes.

## 26. Expert collapse and the Switch load-balancing loss

**Transcript coverage:** lines 5689-5932

### What the lecturer said - transcript only

Naive gradient descent produces a rich-get-richer dynamic. An expert selected early receives gradient signal, becomes more compatible with similar tokens, and is selected even more often. Other experts receive little training and starve. Eventually a small number of experts can absorb most tokens, wasting the rest of the parameter capacity.

The Switch Transformer adds an auxiliary load-balancing loss. For $N$ experts and $T$ tokens,

$$
\mathcal{L}_{bal}
=
\alpha N \sum_{i=1}^{N} f_i P_i,
$$

where

$$
f_i
=
\frac{1}{T}
\sum_{x \in \mathcal{B}}
\mathbf{1}\{\arg\max p(x)=i\}
$$

is the hard fraction of tokens dispatched to expert $i$, and

$$
P_i
=
\frac{1}{T}
\sum_{x \in \mathcal{B}} p_i(x)
$$

is the average router probability assigned to that expert.

The lecturer says the loss is easier to understand through its gradient than through a first-principles derivation. Differentiating with respect to an expert's probability mass yields a factor proportional to its dispatched-token fraction. Popular experts therefore receive stronger downward pressure, reducing their future probability mass.

### Additional explanation

If routing is uniform, both $f_i$ and $P_i$ are near $1/N$. If one expert dominates, both terms become large for that expert and increase the auxiliary loss. Multiplying hard traffic by soft probability connects the nondifferentiable assignment count to a differentiable quantity.

$\alpha$ must be tuned. Too little balancing permits collapse. Too much can force uniformity even when specialization would naturally make some experts more useful, degrading the main language-model objective.

## 27. DeepSeek's expert and device balancing

**Transcript coverage:** lines 5935-6097

### What the lecturer said - transcript only

DeepSeek V1 and V2 train with ordinary gradients through the selected experts and add balancing objectives rather than explicitly solving the exploration problem.

They use a per-expert loss closely related to the Switch loss. DeepSeek also adds per-device balancing. If one device hosts four experts and another device hosts four others, aggregate traffic to the two devices should be balanced so both remain utilized. The per-device objective has the same form as expert balancing but aggregates expert fractions and score mass by device.

DeepSeek V3 introduces a learnable or online-adjusted bias for each expert. Underused experts receive a bias that makes them easier to select, while overused experts are discouraged. This reduces reliance on auxiliary losses and is described as auxiliary-loss-free balancing. It does not remove auxiliary balancing completely: a complementary loss remains to prevent extreme imbalance within a sequence.

The lecturer presents this as progress toward cleaner routing, not a complete solution that eliminates every heuristic.

### Additional explanation

Per-expert and per-device balance optimize related but different goals:

- Per-expert balance protects model capacity and gives every expert training signal.
- Per-device balance protects throughput, because a step waits for the slowest or most overloaded device.

An online bias acts like a feedback controller. It observes routing load rather than relying only on gradients from the modeling loss. Because the bias changes selection without directly changing expert output weights, it can correct traffic while reducing interference with representation learning.

## 28. What happens without balancing, and what the heuristics mean

**Transcript coverage:** lines 6103-6337

### What the lecturer said - transcript only

OLMoE ablations show that load balancing is not optional. With the auxiliary loss, training and validation curves are normal and tokens are spread across experts. Without it, loss is materially worse and almost all tokens collapse onto roughly two experts. The unused experts contribute almost nothing for most of training, so a large fraction of parameters is effectively discarded.

The lecturer finds the result striking. Top-$K$ is discrete, counterfactual experts are unseen, and the object looks difficult to optimize. Yet ordinary gradients through active paths plus a simple balancing loss are enough for the model to train well. The positive reinforcement that strengthens useful experts and the negative balancing pressure on popular experts counteract each other.

The same broad pattern - top-$K$ selection plus auxiliary objectives that keep discrete choices usable - appears in DSA, H-Net's tokenizer-removal work, and other architectural ideas. The lecturer expects it to remain a reusable ingredient.

### Additional explanation

The two forces create a productive equilibrium:

1. The task loss encourages specialization because experts that help certain tokens receive stronger compatibility and better weights.
2. The balancing loss prevents early random advantages from monopolizing all future data.

The goal is not perfectly uniform expertise. It is enough utilization for every expert to learn while still allowing routes to reflect useful differences among tokens.

## 29. Questions on expert semantics and device-level loss

**Transcript coverage:** lines 6340-6466

### What the lecturer said - transcript only

An audience member asks whether the experts become interpretable domain specialists. Visualizations do show different token categories routing to different experts. Punctuation or other symbols may favor one expert, while characters from a non-English script may favor another. The routers are too simple, however, to yield high-level specialists such as a medical, legal, or Wall Street Journal expert. Their specialization does not usually have that clear semantic interpretation.

Another question asks why both per-expert and per-device losses are needed. Perfectly uniform expert use would imply uniform device use if experts are evenly placed. In practice, enforcing expert uniformity too strongly can harm training. Device balance is important enough for systems utilization that a separate loss can emphasize it without requiring the per-expert coefficient to force exact uniformity.

### Additional explanation

Expert identity is not a human-assigned ontology. The router and MLPs jointly discover whatever partition lowers loss under the balancing constraint. A single expert may participate in several unrelated token patterns that happen to require similar transformations.

The device objective also operates at a coarser scale. Experts can remain uneven within each device while total work across devices is balanced, preserving more routing flexibility than exact expert uniformity.

---

# Part VII - Systems, stability, and adaptation

## 30. Expert parallelism and hardware-aware sparse computation

**Transcript coverage:** lines 6469-6772

### What the lecturer said - transcript only

Data and model parallelism each have natural limits. Data parallelism cannot grow beyond the useful global batch size. Model parallelism is constrained by the available ways to partition layers and tensors. Expert parallelism adds another independent axis by placing experts on different devices and routing activations among them.

Expert computation should not be implemented as many tiny, independent matrix multiplications if that prevents efficient hardware use. Experts can instead be represented through block-diagonal or more general structured sparse matrix multiplication. Modern accelerators and libraries support these patterns, allowing larger grouped operations, cache reuse, and efficient handling of variable expert loads. The lecturer describes this as hardware-architecture co-design.

Communication remains a major cost because routed activations travel between devices. A recent Nemotron 3 idea reduces it by down-projecting the routed activation to a lower-dimensional latent vector before the all-to-all exchange, then up-projecting after expert computation. The shared expert can remain at the full residual dimension because it is local, while only the conditional path pays the compressed communication. This controls communication cost without shrinking the entire model representation.

### Additional explanation

An MoE step usually alternates between two bottlenecks:

- **All-to-all dispatch/combine:** network bandwidth and latency dominate.
- **Expert matrix multiplication:** accelerator arithmetic and local memory dominate.

Compression lowers bytes per routed token but adds projection FLOPs. It is beneficial when communication is the bottleneck and the projection preserves enough information. Grouped sparse kernels similarly exchange some scheduling complexity for larger, more efficient matrix operations.

The slide names the down-projected design **LatentMoE**. That name is slide-only; the spoken explanation focuses on the communication tradeoff rather than the label.

## 31. Capacity limits, token dropping, and dropless execution

**Transcript coverage:** lines 6775-6907

### What the lecturer said - transcript only

Older MoE infrastructure introduced an unusual source of stochasticity. If one expert became more popular than its allocated capacity, its queue could overflow. The system would drop excess token-expert assignments, return a zero for those paths, and continue.

Because capacity was determined at a batch level, another user's requests could change whether a token in the current request fit in an expert queue. Identical inputs could therefore receive different results depending on unrelated concurrent traffic.

Modern dropless architectures and libraries, including MegaBlocks, avoid this behavior. The lecturer treats token dropping as a historically interesting problem that has largely been solved in current open-source MoE systems.

### Additional explanation

Capacity factors were originally attractive because fixed-size expert buffers simplify static accelerator programs. Dropless sparse kernels support variable numbers of tokens per expert without silently discarding work. They improve determinism and accuracy but still must manage imbalance efficiently; an overloaded expert can become a straggler even when no tokens are dropped.

This example shows why numerical equivalence is not solely a property of model weights. Batching and serving infrastructure can alter the executed computation unless the implementation guarantees every routed path is honored.

## 32. Router numerical stability and z-loss

**Transcript coverage:** lines 6910-7048

### What the lecturer said - transcript only

Mixtures of experts introduce another softmax in the router. The previous architecture lecture emphasized that exponentials and divisions are numerical danger zones. Early Google work by Barrett Zoph and collaborators identified the MoE router softmax as a significant stability risk.

A common safeguard is to compute the router in float32 even when the rest of the model uses a lower-precision type. Systems may also add a router z-loss:

$$
\mathcal{L}_z(x)
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(
\log \sum_{j=1}^{N} e^{x_j^{(i)}}
\right)^2.
$$

The OLMoE ablation shows much spikier training and validation behavior without z-loss. The lecturer treats both higher-precision routing and z-loss as useful stabilization measures for this especially sensitive subcomputation.

### Additional explanation

The z-loss penalizes a large log-partition value. Softmax probabilities are unchanged if the same constant is added to every logit, so the main task loss may permit logits to drift to large magnitudes. z-loss removes that freedom and keeps the exponentials in a safer range.

Computing only the small router in float32 is inexpensive relative to running every expert in float32. It is a targeted precision allocation: spend numerical range where rank changes can alter a discrete top-$K$ decision.

## 33. Fine-tuning sparse models

**Transcript coverage:** lines 7051-7177

### What the lecturer said - transcript only

MoEs can be difficult to fine-tune on small datasets because their very large total parameter count encourages overfitting. In the comparison shown, the sparse model develops a much larger training-validation gap than the dense model on a downstream benchmark.

Possible responses include updating only non-MoE feed-forward layers, updating attention, or otherwise freezing the expert parameters. Attention-only adaptation appears frequently in recent MoE work. If the model contains some dense layers, those can also be fine-tuned.

The brute-force alternative is more data. The lecturer cites DeepSeek's use of about 1.4 million supervised fine-tuning examples. With enough varied data, a team can update much more of the MoE without the same generalization gap.

### Additional explanation

Parameter-efficient fine-tuning is especially natural for MoEs. Freezing expert MLPs and adapting attention, normalization, routers, or low-rank adapters reduces trainable capacity while preserving the pretrained conditional features.

Router updates deserve care. Changing routes can send downstream data to experts that were not trained for it, while freezing routes may limit adaptation. The best policy depends on data size and whether the target domain requires new specialization or merely new output behavior.

## 34. Upcycling a dense model into an MoE

**Transcript coverage:** lines 7180-7381

### What the lecturer said - transcript only

Upcycling initializes an MoE from an existing dense model. Copy the dense Transformer's weights, duplicate each dense MLP into several experts, randomly initialize a router, and continue training. Different tokens begin reaching different copies, so initially identical experts diverge and specialize.

Early studies showed that continued training of an upcycled MoE could improve faster than continued training of the original dense model. MiniCPM upcycled a 2.4B dense model into an MoE described in the speech as 13.4B parameters and obtained broad benchmark gains. Qwen initialized its Qwen 1.5 MoE from a 1.8B dense model and produced the Qwen1.5-MoE-A2.7B model, an early large-scale upcycling success.

The method has become less common. Teams planning a large modern run generally train an MoE from the beginning rather than first spending substantial compute on a dense model. The lecturer still includes upcycling as an important option in the MoE design space.

### Source reconciliation

The transcript states that MiniCPM was upcycled from 2.4B to 13.4B parameters. Slide 52 labels the resulting MiniCPM-MoE as 13.6B. The notes retain the spoken number and record the slide's different table value.

### Additional explanation

At initialization, copied experts implement the same function, so the new MoE can begin near the dense model's behavior. Router randomness and minibatch variation then break symmetry. This makes upcycling a lower-risk path when a valuable dense checkpoint already exists.

It is less attractive when the final architecture is known in advance. Training sparse from the beginning allows expert specialization over the full data curriculum and avoids paying for a separate dense pretraining run.

---

# Part VIII - DeepSeek as an architecture case study

## 35. The evolution from DeepSeek MoE V1 to V3

**Transcript coverage:** lines 7384-7528

### What the lecturer said - transcript only

The lecturer recommends the DeepSeek papers as clear, detailed examples of architecture design and systems-aware experimentation.

**DeepSeek MoE V1** establishes the template: fine-grained routed experts, shared experts, standard token-choice top-$K$ routing, and auxiliary expert balancing. The lecturer calls it close to a Platonic ideal of the basic modern MoE.

**DeepSeek MoE V2** scales the architecture and adds systems-aware routing. It uses two shared experts, many more fine-grained experts, device-limited routing, and communication-balancing objectives. These losses optimize not only prediction quality but also where computation and network traffic occur. The lecturer uses V2 to emphasize that successful model training must respect the hardware system.

**DeepSeek MoE V3** keeps the shared/fine-grained structure but changes routing and balance. It uses per-expert bias feedback to make balancing mostly auxiliary-loss-free, retains a sequence-wise auxiliary safeguard, and changes expert weighting through sigmoid scores followed by normalization over selected experts. The basic conditional-computation design remains recognizable.

### Source reconciliation

Slide 56 is titled "DeepSeek MoE V3" and its architecture content matches V3, but its size line begins "V2 (671B - 37 active)." This appears to be a slide-label typo. The spoken lecture consistently identifies this stage as V3.

### Additional explanation

The progression illustrates three levels of maturity:

1. Prove that sparse expert capacity improves quality at fixed active compute.
2. Co-design routing with device placement and communication.
3. Reduce the optimization side effects of auxiliary balancing while retaining traffic control.

The architecture is not evolving toward fewer systems constraints. It is making those constraints explicit in routing, objectives, and model shape.

## 36. Multi-head latent attention and KV-cache compression

**Transcript coverage:** lines 7531-7636

### What the lecturer said - transcript only

The final DeepSeek V3 walk-through includes multi-head latent attention (MLA). Instead of projecting the hidden state directly into full keys and values and caching all of them, MLA first creates a lower-dimensional latent activation $c_t^{KV}$. Keys and values are reconstructed as functions of that latent:

$$
c_t^{KV}=W^{DKV}h_t,
$$

$$
k_t^C=W^{UK}c_t^{KV},
\qquad
v_t^C=W^{UV}c_t^{KV}.
$$

At inference, the system can cache the compact $c_t^{KV}$ rather than every expanded key and value. Queries can also use a compressed latent during training.

Rotary positional embeddings complicate this factorization. Without RoPE, projection matrices can be algebraically absorbed or rearranged around the latent. Position-dependent rotations do not generally commute with those projections. DeepSeek handles this by keeping a small set of non-latent key dimensions that carry the rotary position information while the remaining content dimensions use latent caching.

### Additional explanation

MLA targets a major decode bottleneck. Standard attention stores a key and value vector for every layer and previous token. If the latent dimension is much smaller than the combined key-value representation, cache memory and memory bandwidth fall substantially.

The RoPE issue is a useful example of why factorization is not purely a rank question. A learned linear map can be merged with another learned linear map, but a different token-position rotation is inserted at every position. Separating positional and content subspaces preserves position information while retaining most of the compression benefit.

## 37. Multi-token prediction as training signal and draft model

**Transcript coverage:** lines 7639-7687

### What the lecturer said - transcript only

DeepSeek V3 also uses multi-token prediction (MTP). Rather than train only to predict the next token, auxiliary modules predict tokens farther into the future.

The lecturer gives two motivations:

- **Statistical:** predicting several future steps may encourage a representation that anticipates longer-range consequences rather than only the immediate next token.
- **Systems:** the auxiliary prediction path can serve as a built-in draft mechanism for speculative decoding. It proposes future tokens that the main model can verify, potentially increasing decoding speed.

The inference lecture will develop speculative decoding in more detail.

### Additional explanation

Multi-token objectives expose more supervision per sequence position, but farther targets are also harder and more uncertain. Auxiliary heads can share the main representation while keeping their extra compute limited.

For speculative decoding, draft quality and draft cost must be considered together. A cheap head that proposes moderately accurate tokens may be more useful than an expensive auxiliary model with slightly higher acceptance. MTP is attractive because the training-time auxiliary path can be reused for this purpose.

## 38. Closing perspective

**Transcript coverage:** lines 7690-7741

### What the lecturer said - transcript only

Mixtures of experts exploit sparsity so a model receives the benefit of a much larger parameter pool without paying the compute cost of every parameter on every token. The routing problem initially appears difficult because it is discrete and only selected experts receive feedback. In practice, simple top-$K$ routers, ordinary gradients through active paths, and balancing heuristics work at large scale.

The accumulated empirical evidence shows that MoEs are cost-effective and are likely to remain a standard part of large language models. Understanding their mechanics is therefore part of understanding modern language-model architecture.

### Additional explanation

The attention and MoE halves of the lecture share one design pattern: **spend expensive computation selectively**.

- Linear and sparse attention avoid treating every past token interaction equally.
- MoEs avoid applying every MLP parameter to every token.
- Both depend on cheap control mechanisms - gates, recurrences, or top-$K$ selectors - whose errors determine what expensive computation is available.

Efficiency comes from conditional structure, while much of the engineering difficulty comes from making that structure trainable, balanced, numerically stable, and hardware-friendly.

---

# Consolidated takeaways

1. Full attention becomes increasingly dominant as context grows because token-to-token interactions scale quadratically while MLP work scales linearly.
2. Local/global hybrids and FlashAttention deliver major savings, but million-token contexts motivate architectural changes with milder sequence dependence.
3. Removing softmax permits the exact reassociation $(QK^\top)V=Q(K^\top V)$, replacing an $n \times n$ attention matrix with a fixed-size key-value summary.
4. Causal linear attention has an exact recurrent form, enabling parallel training and fixed-state autoregressive inference.
5. Mamba-2 adds input-dependent forgetting; Gated DeltaNet also controls writes and approximately erases an existing key direction before replacement.
6. The parallel and recurrent forms of a linear mechanism are exact equivalents, but linear attention itself is not equivalent to softmax attention.
7. Large-scale recurrent attention systems remain hybrids. Low recurrent ratios can preserve quality, while pure recurrent models show clearer long-context degradation.
8. DSA uses a cheap all-context indexer and expensive full attention only on a bounded top-$K$ subset. It is not asymptotically linear, but favorable constants make it useful.
9. An MoE increases total parameters while activating only a few experts per token, improving quality per active FLOP at the cost of memory, routing, and communication complexity.
10. Token-choice top-$K$ with a learned linear router is the dominant routing design; RL and global matching are more principled but more expensive or unstable.
11. Fine-grained experts are strongly supported empirically. Shared experts are common, although controlled studies disagree on their marginal benefit.
12. Sparse training creates rich-get-richer expert collapse. Auxiliary load-balancing losses push probability mass away from overloaded experts and are essential in standard training.
13. Device balance is a separate systems goal: evenly trained experts do not automatically produce an efficient distributed step under practical loss weights and placements.
14. Expert parallelism adds useful scaling capacity but introduces all-to-all communication, sparse-kernel requirements, stragglers, and potential serving nondeterminism.
15. Float32 routing and z-loss protect the softmax/top-$K$ boundary from low-precision numerical failures.
16. DeepSeek's V1-to-V3 evolution shows modeling and systems co-design: shared and fine-grained experts, topology-aware routing, communication losses, and feedback-controlled balance.
17. MLA compresses the KV cache through a latent representation, while MTP supplies both a richer prediction objective and a possible speculative-decoding draft path.

# Key equations

## Standard and linear attention

$$
\operatorname{Attn}(Q,K,V)=\rho(QK^\top)V.
$$

When $\rho$ is the identity,

$$
(QK^\top)V=Q(K^\top V).
$$

The approximate multiplication cost changes from

$$
n^2d_k+n^2d_v
$$

to

$$
2nd_kd_v.
$$

## Recurrent linear attention

$$
S_t=S_{t-1}+k_tv_t^\top,
\qquad
y_t=q_t^\top S_t.
$$

## Mamba-2 view

$$
S_t=\gamma_tS_{t-1}+k_tv_t^\top,
\qquad
y_t=q_t^\top S_t+v_t^\top D,
\qquad
\gamma_t=f(x_t).
$$

## Gated DeltaNet

$$
S_t
=
\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}
+
\beta_tk_tv_t^\top,
$$

$$
y_t=q_t^\top S_t,
\qquad
\gamma_t=f(x_t),
\qquad
\beta_t=f(x_t).
$$

## DSA index score

$$
I_{t,s}
=
\sum_{j=1}^{H'}w_{t,j}^{I}
\operatorname{ReLU}\left((q_{t,j}^{I})^\top k_s^{I}\right).
$$

## Top-$K$ MoE layer

$$
h_t^\ell
=
u_t^\ell
+
\sum_{i=1}^{N}g_{i,t}\operatorname{FFN}_i(u_t^\ell),
$$

$$
s_{i,t}=\operatorname{softmax}_i((u_t^\ell)^\top e_i^\ell),
$$

$$
g_{i,t}
=
\begin{cases}
s_{i,t}, & i \text{ is selected by top-}K,\\
0, & \text{otherwise}.
\end{cases}
$$

## Switch-style load balancing

$$
\mathcal{L}_{bal}=\alpha N\sum_{i=1}^{N}f_iP_i,
$$

$$
f_i=\frac{1}{T}\sum_{x\in\mathcal{B}}\mathbf{1}\{\arg\max p(x)=i\},
\qquad
P_i=\frac{1}{T}\sum_{x\in\mathcal{B}}p_i(x).
$$

## Router z-loss

$$
\mathcal{L}_z(x)
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(\log\sum_{j=1}^{N}e^{x_j^{(i)}}\right)^2.
$$

## MLA latent cache

$$
c_t^{KV}=W^{DKV}h_t,
\qquad
k_t^C=W^{UK}c_t^{KV},
\qquad
v_t^C=W^{UV}c_t^{KV}.
$$

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Activated parameters | Parameters used for one token's forward pass; smaller than total parameters in a sparse MoE. |
| Auxiliary loss | A secondary training objective used here for load balance or numerical stability. |
| DSA | DeepSeek Sparse Attention, which selects a top-$K$ token subset with a lightweight indexer before full attention. |
| Dropless MoE | An implementation that processes all routed token-expert assignments rather than discarding assignments beyond a fixed expert capacity. |
| Expert choice | Routing in which each expert selects tokens. |
| Expert collapse | A failure mode in which a few experts receive almost all tokens while the rest starve. |
| Expert parallelism | Placing different experts on different devices and routing token activations among them. |
| Fine-grained expert | A smaller expert used as part of a larger expert pool, allowing more precise combinations at similar active compute. |
| FlashAttention | An exact attention implementation organized to reduce memory traffic and avoid materializing the full score matrix. |
| Gated DeltaNet | A recurrent linear-attention mechanism with input-dependent retention, write gating, and selective erasure. |
| Global assignment | Batch-level optimization that jointly matches tokens and experts. |
| Hybrid attention | An architecture combining recurrent, local, sparse, or linear layers with periodic full-attention layers. |
| Linear attention | An attention alternative whose reordered computation is linear in sequence length after replacing the usual softmax operation. |
| Load-balancing loss | An auxiliary loss that discourages excessive routing to already popular experts or devices. |
| Mamba-2 | A state-space model that can be interpreted as gated linear attention with an input-dependent retention gate. |
| MLA | Multi-head latent attention, which represents keys and values through a lower-dimensional latent to reduce KV-cache cost. |
| MoE | Mixture of experts, a layer containing multiple subnetworks of which only a sparse subset is activated per token. |
| MTP | Multi-token prediction, auxiliary prediction of more distant future tokens. |
| Recurrent state | A fixed-size summary carried from one position to the next in linear attention or a state-space model. |
| Router | The lightweight function that scores experts for a token. |
| Shared expert | An always-active expert that bypasses sparse routing. |
| Token choice | Routing in which each token selects its experts; the dominant modern MoE design. |
| Top-$K$ routing | Retaining only the $K$ highest-scoring experts or memory positions. |
| Upcycling | Initializing an MoE by copying the weights of a pretrained dense model into multiple experts. |
| z-loss | A penalty on the squared log-partition of router logits, used to control logit magnitude and softmax instability. |

# Self-check questions

1. Why does attention eventually dominate an MLP as context length increases?
2. What does FlashAttention improve, and what asymptotic problem does it leave unchanged?
3. Which change makes $(QK^\top)V=Q(K^\top V)$ usable as linear attention, and what expressivity is lost?
4. Why are the parallel and recurrent forms of linear attention exactly equivalent while neither is equivalent to full softmax attention?
5. What information must the fixed state $S_t$ compress?
6. What roles do $\gamma_t$ and $\beta_t$ play in Mamba-2 and Gated DeltaNet?
7. Why does $I-\beta_tk_tk_t^\top$ behave like an erase operation, and when would it be an exact projector?
8. Why do successful large recurrent-attention models retain some full-attention layers?
9. How can DSA improve real cost even though its indexer remains quadratic in sequence length?
10. What is gained and lost by inserting DSA only during long-context extension?
11. What is the difference between total and activated parameters in an MoE?
12. Why does token-choice routing need explicit load balancing?
13. Why is RL a conceptually natural but practically uncommon way to train a router?
14. How does the Switch balancing loss connect a nondifferentiable dispatch count to differentiable router probabilities?
15. Why can exact per-expert uniformity be undesirable even when per-device uniformity is operationally important?
16. What tradeoff is created by replicating an always-on shared expert?
17. How do structured sparse matrix multiplications make expert computation more hardware-friendly?
18. Why could identical requests historically receive different MoE outputs under token dropping?
19. Why are router logits often computed in float32, and what does z-loss control?
20. Why can an MoE overfit more severely than a dense model during small-data fine-tuning?
21. What benefit does upcycling provide, and why is it less common when training a new large model from scratch?
22. How do DeepSeek V2's routing objectives reflect network topology?
23. Why does RoPE complicate MLA's projection and cache factorization?
24. How can MTP serve both as a learning objective and as part of speculative decoding?
25. What common conditional-computation idea connects DSA and MoE routing?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-126 | Opening logistics; lecture scope: attention alternatives and MoEs | Yes |
| 127-238 | Long-context demand and the changing attention/MLP cost balance | Yes |
| 239-523 | Local/global hybrids, FlashAttention, constant factors, and motivation for linear time | Yes |
| 524-717 | Standard attention notation, removing softmax, reassociation, and complexity | Yes |
| 718-879 | Recurrent linear attention, fixed state, and parallel/serial duality | Yes |
| 880-987 | MiniMax M1, 7:1 hybrid, scale evidence, and the absence of proven fully linear systems | Yes |
| 988-1250 | Mamba-2 gate, direct value path, duality, and Nemotron 3 | Yes |
| 1251-1602 | Gated DeltaNet gates, erase update, and links to fast weights and test-time training | Yes |
| 1603-1816 | Qwen 3.5/Qwen Next and controlled hybrid-ratio evidence | Yes |
| 1817-2011 | Q&A on exact equivalence and the Mamba-2 direct value term; recurrent-method convergence | Yes |
| 2012-2398 | DSA mechanics, post-hoc long-context adaptation, DeepSeek V3.2, GLM-5, and complexity caveat | Yes |
| 2399-2626 | Q&A on quadratic indexing, constant factors, extension stages, and bounded $K$ | Yes |
| 2627-3052 | Q&A on stability, future designs, FP4, expressivity, and finite-state compression | Yes |
| 3053-3160 | MoE motivation, ubiquity, and reusable sparse-routing primitives | Yes |
| 3161-3291 | Sparse FFN construction and the total-parameter/active-FLOP mental model | Yes |
| 3292-3540 | Switch, OLMoE, and released-model evidence for sparse capacity | Yes |
| 3541-3741 | Expert parallelism and Western and Chinese MoE examples | Yes |
| 3742-4105 | Q&A on communication, sparse training, token routing, parallel limits; DeepSeek architecture science | Yes |
| 4106-4264 | Slow adoption, infrastructure and training difficulty, and uncommon attention MoEs | Yes |
| 4265-4477 | Design axes; token choice, expert choice, and global assignment | Yes |
| 4478-4771 | Learned, hash, RL, and linear-assignment routers | Yes |
| 4772-5209 | Top-$K$ equations, shared and fine-grained experts, DeepSeek and OLMoE ablations | Yes |
| 5210-5254 | Q&A on shared-expert replication and the memory/communication tradeoff | Yes |
| 5255-5686 | Sparse-training problem, RL routing, Gaussian noise, and input jitter | Yes |
| 5687-6097 | Expert collapse, Switch balance, DeepSeek expert/device losses, and V3 bias balancing | Yes |
| 6098-6466 | No-balancing ablation, top-$K$ as a general pattern, expert semantics, and device-loss Q&A | Yes |
| 6467-6907 | Expert parallel systems, sparse matmuls, latent communication, token dropping, and dropless systems | Yes |
| 6908-7177 | Router stability, float32, z-loss, and fine-tuning overfit | Yes |
| 7178-7381 | Upcycling mechanics, MiniCPM and Qwen examples, and declining use | Yes |
| 7382-7687 | DeepSeek V1-V3, topology-aware routing, MLA, RoPE complication, and MTP | Yes |
| 7688-7741 | MoE summary and conclusion | Yes |
