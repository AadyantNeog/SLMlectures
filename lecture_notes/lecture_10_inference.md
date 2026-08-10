---
title: "Lecture 10 - Inference"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 10
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 10 Inference.txt"
slide_deck: "../lecture_10.pdf"
status: "complete"
---

# Lecture 10: Inference

## How to read these notes

Every substantive topic is divided into two source layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves claims, examples, qualifications, numerical details, corrections, and substantive audience questions while removing filler and repetition.
2. **Additional explanation.** This contains independent intuition, derivation, examples, and study guidance. It is not presented as something the lecturer said.

The raw transcript line spans make the paraphrase auditable. The complete 5,587-line transcript was mapped before the deck was opened. All 42 slides were then rendered and visually inspected to verify tensor notation, formulas, numeric configurations, plots, diagrams, and algorithm details. Whenever the deck materially supplies or corrects information that cannot be recovered from the transcript alone, it appears under **Source reconciliation**.

## Lecture map

The lecture builds an account of inference from workload to serving system:

1. Explain why inference is a large, repeated, and increasingly agent-driven cost.
2. Define time to first token, per-request latency, and aggregate throughput.
3. Show why sequential autoregressive generation has a different arithmetic profile from training.
4. Derive the arithmetic intensity of Transformer MLP and attention layers during prefill and generation.
5. Connect memory traffic to latency, throughput, batch-size tradeoffs, and the KV-cache capacity limit.
6. Reduce inference cost with GQA, MLA, cross-layer attention, local or linear attention, quantization, pruning, and distillation.
7. Recover exact target-model sampling with speculative decoding.
8. Handle live, ragged workloads with continuous batching, selective batching, and PagedAttention.

---

# Part I - The inference workload and its metrics

## 1. Why inference is a first-class systems problem

**Transcript coverage:** lines 1-327

### What the lecturer said - transcript only

Inference begins after training: given a trained model and a prompt, produce a response as accurately and quickly as possible. Although inference receives only one lecture, the lecturer argues that it is increasingly important because it appears throughout the model lifecycle:

- actual use, including assistants, code completion, agents, and batch data processing;
- evaluations that require generation;
- reinforcement learning, where a model generates rollouts, the rollouts are scored, and the weights are updated from those scores.

Training is an expensive but primarily one-time cost. Inference is paid repeatedly after deployment. As perspective, the lecturer cites an estimate that OpenAI produces about 8.6 trillion tokens per day and compares it with DeepSeek V4's stated 32-trillion-token training corpus. On those figures, fewer than four days of OpenAI output contains as many tokens as that entire training run, even though frontier models may of course train on more tokens.

The growth of agents makes this comparison more important. A conventional chatbot produces text that a person reads, so human reading speed eventually limits the value of faster token generation. An agent can generate internal reasoning, invoke tools, inspect intermediate results, and continue working before it emits a relatively small human-facing answer. Most of its generated tokens may therefore never be read. In that setting, token count is better viewed as compute spent, and an ambitious task can consume more useful inference compute without running into a human reading-speed ceiling.

Inference is performed by both commercial providers and open-source systems. Closed-model API companies must serve their own models, while other companies host open-weight models. The lecturer names several open-source options:

- **vLLM**, described as a popular default;
- **SGLang**, described as especially suitable for agentic workloads but not yet as popular;
- NVIDIA's **TensorRT**, described as fast but narrower;
- **llama.cpp**, a common option for CPU and local inference.

The practical conclusion is that even a 10 percent improvement matters at this scale, and a 2x improvement would be enormous.

### Source reconciliation

Slide 4 spells the model name as **DeepSeek V4** and lists the systems as **vLLM**, **SGLang**, **TensorRT-LLM**, and **llama.cpp**. The automatic transcript renders several of these names phonetically, so the slide spellings are used here. The 8.6-trillion-tokens-per-day and 32-trillion-training-token figures are preserved as the lecturer's cited perspective, not independently verified measurements.

### Additional explanation

Inference cost is not just the cost of the final visible answer. A useful accounting identity is

$$
\text{daily inference work}
\approx
\sum_{r \in \text{requests}}
\left(
\text{prefill work}_r + \text{generated-token work}_r
\right).
$$

Agents increase both the number of requests and the number of generated tokens within each request. Tool results may also lengthen the context, making later generation steps more expensive because more cached state must be read.

The repeated-cost framing changes optimization priorities. A training optimization pays off once per run. A serving optimization is amortized over every future token and may also reduce the number of accelerators, power, or replicas needed to satisfy a service-level target.

## 2. Three different meanings of "fast"

**Transcript coverage:** lines 328-481

### What the lecturer said - transcript only

The lecturer distinguishes three metrics because no single number captures inference speed.

1. **Time to first token (TTFT)** is the delay between submitting a query and receiving the first generated token. It matters in interactive applications because the user otherwise sees no progress. Once tokens begin streaming, generation need not always outrun human reading by a large margin.
2. **Latency** is the speed experienced by one query, expressed in the lecture as how quickly tokens stream for that request. The slides state the unit as seconds per token.
3. **Throughput** measures aggregate work, usually tokens per second across many queries. It matters when a large dataset must be processed and the completion time of the whole job is more important than the order in which individual requests finish.

Latency, its reciprocal token rate for one request, and aggregate throughput are related. Many optimizations improve both latency and throughput, but they are not identical objectives. Later, batch size will produce a direct tradeoff: a larger batch can raise total tokens per second while making each participant wait longer.

### Additional explanation

For a request with prefill time $t_{\mathrm{prefill}}$, $G$ generated tokens, and generation-step times $t_1,\ldots,t_G$,

$$
\mathrm{TTFT} \approx t_{\mathrm{prefill}} + t_1,
$$

while an inter-token latency statistic can be formed from the later $t_i$. Aggregate throughput over a measurement interval is

$$
\mathrm{throughput}
=
\frac{\text{all tokens completed in the interval}}
{\text{interval duration}}.
$$

Two systems can therefore rank differently. One may deliver a first token quickly and then decode slowly; another may wait to assemble a large batch and then process far more total tokens per second. A deployment must select the metric that matches its workload rather than reporting "tokens per second" without saying whose tokens and over what batching policy.

## 3. Why inference is unlike training

**Transcript coverage:** lines 482-609

### What the lecturer said - transcript only

The lecture's central workload distinction is sequence parallelism. In supervised training, all sequence positions are already known. Attention and MLP operations can treat sequence length as another tensor dimension and process all token positions together.

Autoregressive inference cannot expose all future tokens at once. Each next token depends on tokens generated before it, so output tokens must be produced sequentially. The sequence dimension is no longer freely parallelizable during generation. This makes it harder to attain high arithmetic intensity and fully use accelerator compute, even though the model architecture and weights are the same as in training.

The planned analysis is:

1. derive arithmetic intensity, latency, and throughput for a Transformer;
2. reduce cost through smaller KV caches, quantization, and pruning;
3. discuss speculative decoding;
4. finish with practical serving concerns.

### Additional explanation

The dependency chain is the key constraint:

$$
x_{t+1} \sim p(\cdot \mid x_{1:t}).
$$

The probability distribution for $x_{t+2}$ cannot be evaluated until a value for $x_{t+1}$ has been selected. Parallel hardware can accelerate the work *inside* one decoding step and can batch independent requests, but ordinary autoregressive decoding cannot parallelize an unknown future along the time axis.

This is why a kernel or architecture optimized for training may be poorly balanced for serving. Training feeds large matrices to the accelerator. Single-request decoding often degenerates into matrix-vector-like operations plus reads of request-specific cached state.

## 4. Transformer notation and the corrected GQA convention

**Transcript coverage:** lines 610-975 and 1261-1354

### What the lecturer said - transcript only

The lecturer adopts the tensor-shape notation used in a Google scaling book, recommends that resource for its clear treatment of Transformers and inference, and notes that several lecture figures come from it. Dimension symbols denote both an axis and its length. The initial examples use:

- $B$: batch dimension, also the number of sequences;
- $T$: sequence or query dimension;
- $D$: model or embedding dimension;
- $H$: per-head dimension.

In the diagram's Einstein-style notation, a red dimension appears in both operands and disappears from the result, so it is contracted. A black dimension appears in one operand and remains in the output. A blue dimension appears in both operands and remains, so it is a batching dimension. For example, multiplying a $B \times T \times D$ tensor by a $D \times H$ matrix contracts $D$ and produces $B \times T \times H$.

The full Transformer-block diagram makes data dependencies and tensor shapes explicit. Activations $X$ are projected into queries, keys, and values, attention is computed, and the result passes through an output projection and residual path. The MLP uses a gated up-projection and a down-projection. The lecture uses these conventions:

- $F = 4D$ as a simplifying MLP-width convention;
- $D = NH$, where $N$ is the number of query heads;
- $S$ for the number of input or key-value tokens and $T$ for the number of output or query tokens;
- $S=T$ in training or prefill, while generation specializes to $T=1$.

During a student question, the lecturer rechecks the grouped-query-attention notation and corrects the spoken assignment of the group variables. The corrected meaning is that $K$ is the number of key-value groups or heads and $G$ is the number of query heads served by each key-value head.

### Source reconciliation

Slide 6 provides the complete, corrected symbol table:

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $L$ | number of layers |
| $T$ | query sequence length |
| $S$ | key-value sequence length |
| $V$ | vocabulary size |
| $D$ | model or embedding dimension |
| $F$ | MLP hidden dimension |
| $H$ | attention-head dimension |
| $N$ | number of query heads |
| $K$ | number of key-value heads |
| $G$ | query heads per key-value head, $G=N/K$ |

The slide also states $N=KG$. This table resolves the presenter's initial verbal swap and is consistent with the correction made during Q&A.

### Additional explanation

The distinction between $S$ and $T$ is what makes one set of formulas describe both phases:

- **prefill:** a block of $S$ prompt tokens supplies queries against $S$ keys, so $T=S$;
- **decode:** one newly generated position supplies one query against the existing $S$ cached positions, so $T=1$.

The blue batch dimension in the attention contraction will later explain why increasing $B$ does not improve attention's generation intensity: each batch element has a different KV cache. By contrast, every batch element multiplies by the same MLP weights, so batching can reuse those weights.

## 5. Arithmetic intensity warm-up: matrix multiplication

**Transcript coverage:** lines 976-1260

### What the lecturer said - transcript only

To review arithmetic intensity, the lecturer considers multiplying

$$
X \in \mathbb{R}^{B \times D}
\quad\text{by}\quad
W \in \mathbb{R}^{D \times F}.
$$

$B$ can be read as batch size, $D$ as model dimension, and $F$ as an MLP projection dimension. With BF16 storage, the operation reads $X$ and $W$, performs the matrix multiplication, and writes the result. The matrix multiplication performs $2BDF$ FLOPs. Arithmetic intensity is FLOPs divided by bytes transferred, and high intensity is desirable because it means more useful arithmetic is performed for each byte moved.

When $B$ is much smaller than $D$ and $F$, the expression approaches $B$ FLOPs per byte. The hardware comparison is the accelerator's peak FLOP/s divided by its memory bandwidth. For the H100 values used in the lecture, this balance point is approximately 295 FLOPs per byte. A workload above that intensity is compute-bound; one below it is memory-bound.

The extreme inference-like case is $B=1$. It reads a large $D \times F$ matrix but uses it for only one vector, giving intensity near 1. The operation is therefore strongly memory-bound. Inference commonly presents these thin matrix or tensor shapes rather than the large matrix-matrix products found in training.

### Source reconciliation

Slides 7-8 provide the byte expression that the transcript only describes:

$$
F_{\mathrm{matmul}} = 2BDF,
$$

$$
Q_{\mathrm{matmul}}
= 2BD + 2DF + 2BF
=2(BD+DF+BF)\ \text{bytes},
$$

and therefore

$$
I_{\mathrm{matmul}}
=\frac{BDF}{BD+DF+BF}
\longrightarrow B
\quad\text{when }B \ll D,F.
$$

The deck uses H100 peak BF16 throughput $989\times 10^{12}$ FLOP/s and HBM bandwidth $3.35\times10^{12}$ bytes/s, whose ratio rounds to 295 FLOPs per byte.

### Additional explanation

The limit $I\to B$ can be understood as weight reuse. The dominant cost is loading the $D \times F$ weight matrix once. A batch of $B$ vectors performs about $B$ times as much arithmetic with that same matrix, so the intensity grows linearly with $B$ until activation traffic becomes important.

The compute-bound threshold is an idealized roofline threshold. Real kernels can fall below both memory and compute ceilings because of launch overhead, imperfect tiling, synchronization, non-coalesced accesses, or unsupported low-precision paths. The calculation is still valuable because it identifies the fundamental resource pressure before implementation overhead is considered.

---

# Part II - KV caching and the source of the decode bottleneck

## 6. Naive autoregressive decoding and the KV cache

**Transcript coverage:** lines 1357-1665

### What the lecturer said - transcript only

The naive decoding procedure repeatedly feeds the entire known sequence into the Transformer, samples one token from the output logits, appends that token, and runs the full expanded prefix again. The lecture illustrates this by repeatedly extending the prompt "never gonna give you" with sampled words such as "up" and "never." A single full-sequence causal-attention pass takes quadratic time in the current sequence length. Repeating it while generating $T$ tokens therefore produces a cubic total attention cost.

Much of that work is redundant. In a causal Transformer, appending a token does not change the key and value representations already computed for earlier positions. The same prefix state is recomputed whether the next sample becomes one word or another. This would not hold for a bidirectional model, where appending content can affect representations on both sides.

The solution is to keep a **KV cache** in high-bandwidth memory. Inference then has two stages:

1. **Prefill:** process the prompt in parallel and populate key and value vectors for every prompt position.
2. **Generation or decode:** process one new token at a time, read the existing cache, produce next-token logits and the new token's key and value vectors, append those vectors to the cache, and repeat.

For every sequence, token, layer, and key-value head, the cache stores a head-dimensional vector for the key and another for the value. Prefill retains the parallelism of training over a known prompt. Generation remains sequential, but it no longer recomputes the old prefix's keys and values.

### Source reconciliation

Slide 9 states the naive total cost as $O(T^3)$ because one full feed-forward pass has an $O(T^2)$ attention term. Slide 10 depicts prefill as compute-bound and generation as memory-bandwidth-bound, which the following derivation justifies rather than assumes.

### Additional explanation

The cubic claim is the sum of growing quadratic prefixes:

$$
\sum_{t=1}^{T} O(t^2)=O(T^3).
$$

Caching changes the attention work at step $t$ from rebuilding all prefix-to-prefix interactions to computing the new query against the stored prefix, an $O(t)$ operation per layer. Summed across generation, the attention arithmetic becomes $O(T^2)$, although the model must still read an ever-growing cache.

For BF16 keys and values, an approximate cache size is

$$
M_{\mathrm{KV}}
=B\,S\,L\,K\,H
\times 2_{\{K,V\}}
\times 2_{\mathrm{bytes}}
=4BSLKH\ \text{bytes}.
$$

This cache is the central persistent state of autoregressive serving. It grows linearly with batch size, context length, layer count, number of KV heads, and head width.

## 7. MLP arithmetic intensity in prefill and generation

**Transcript coverage:** lines 1666-1927

### What the lecturer said - transcript only

The lecturer next counts FLOPs and memory traffic for the Transformer MLP, focusing on matrix multiplications because they dominate arithmetic and because smaller pointwise operations can potentially be fused.

The gated MLP reads its input and three weight matrices, computes an up projection and a gate projection, writes their intermediate outputs, combines them through the nonlinearity, computes the down projection, and writes the final output. Under the assumption that $BT$ is much smaller than the model and feed-forward dimensions, the resulting arithmetic intensity approaches $BT$.

This resembles the matrix-multiplication warm-up because MLP work is independent across both batch elements and token positions. Prefill can make $BT$ large using bigger batches or longer known sequences, so the MLP can be compute-bound.

Generation fixes $T=1$, reducing MLP intensity to approximately $B$. Here $B$ is the number of concurrent requests, which a batch data-processing job may control but an interactive service receives unpredictably as users arrive and leave. Large request batches can still make the generation MLP efficient; the later serving section addresses how to maintain such batches dynamically.

### Source reconciliation

Slides 11-12 give the exact leading counts for the lecture's gated MLP under BF16 storage:

$$
F_{\mathrm{MLP}} = 6BTDF,
$$

$$
Q_{\mathrm{MLP}}
=4BTD+4BTF+6DF\ \text{bytes}.
$$

Thus

$$
I_{\mathrm{MLP}}
=\frac{6BTDF}{4BTD+4BTF+6DF}
\longrightarrow BT
\quad\text{when }BT\ll D,F.
$$

### Additional explanation

The factor of six FLOPs comes from three dense products, each costing approximately $2BTDF$:

1. up projection;
2. gate projection;
3. down projection.

The important reuse is of shared weights. Once a tile of an MLP weight matrix reaches fast on-chip memory, many requests and token positions can multiply by it. Increasing $B$ or $T$ creates useful reuse instead of proportionally increasing the amount of weight data that must be fetched.

The asymptotic intensity omits normalization, activation functions, residual additions, and kernel overhead. Those operations matter in a real implementation, but they do not change the lecture's key contrast between shared MLP weights and request-specific attention state.

## 8. Attention arithmetic intensity and the fundamental decode limit

**Transcript coverage:** lines 1928-2256

### What the lecturer said - transcript only

For attention, $S$ is the number of previous key-value positions and $T$ is the number of query positions for which logits are being produced. The leading work reads queries, keys, and values, computes query-key attention scores, applies the softmax, multiplies by values, and writes the result. The softmax does not dominate the FLOP count used in the analysis.

The resulting arithmetic intensity is

$$
\frac{ST}{S+T}.
$$

During prefill, $T=S$, so attention intensity is $S/2$. A sufficiently long sequence can therefore give useful intensity. During generation, $T=1$, so the intensity is

$$
\frac{S}{S+1}<1.
$$

This is far below the H100's approximate balance point of 295 FLOPs per byte and is the central inference bottleneck.

Batching cannot repair it. All sequences use the same MLP weights, so loading a weight tile can serve many batch elements. In attention, each sequence owns a different KV cache. Increasing $B$ adds another independent cache and proportional memory traffic rather than increasing reuse of one cache. In the Transformer contraction diagram, this is represented by $B$ being a retained batching dimension rather than a contracted dimension. The operation behaves more like a batch of dot products, which has poor arithmetic intensity, than one reusable matrix multiplication.

The lecturer summarizes the phase split as follows:

- prefill is generally compute-bound;
- generation is memory-bound;
- prefill MLP intensity is approximately $BS$;
- prefill attention intensity is $S/2$;
- generation MLP intensity is approximately $B$ and can improve with concurrent requests;
- generation attention intensity is below 1 and cannot be raised by ordinary batching while retaining the same Transformer attention structure.

This derivation is why people describe inference as memory-bound.

### Source reconciliation

Slides 12-13 provide the exact leading counts used to obtain the spoken expression:

$$
F_{\mathrm{attn}}=4BSTD,
$$

$$
Q_{\mathrm{attn}}=4BSD+4BTD=4BD(S+T)\ \text{bytes},
$$

$$
I_{\mathrm{attn}}
=\frac{4BSTD}{4BD(S+T)}
=\frac{ST}{S+T}.
$$

The slide labels generation attention intensity as "< 1 (impossible to improve)." The intended scope is this arithmetic-intensity calculation for conventional cached Transformer attention; architecture changes discussed later can change the cache and workload.

### Additional explanation

The batch dimension cancels algebraically because both work and bytes scale with $B$:

$$
I_{\mathrm{attn}}
=\frac{B\cdot\text{work per request}}
{B\cdot\text{bytes per request}}
=I_{\text{one request}}.
$$

By contrast, the dominant MLP weight bytes do not grow with $B$ in the ideal reuse model. This distinction explains why a server can have a compute-efficient MLP and a memory-bound attention layer in the same decoding step.

The result also says that faster peak matrix-multiplication hardware alone may not speed up decoding. Bandwidth, cache compression, and avoiding unnecessary reads are more directly connected to the bottleneck.

## 9. A memory-traffic model for latency and throughput

**Transcript coverage:** lines 2257-2637

### What the lecturer said - transcript only

Once generation is known to be memory-bound, runtime can be approximated from the bytes that must move. Assuming communication and computation overlap, memory traffic is the bottleneck even though this leaves much of the accelerator's arithmetic capacity idle.

The lecturer applies this model to Llama 2 13B on one H100. The configuration specifies sequence length, model dimension, MLP width, numbers of query and KV heads, head dimension, layer count, vocabulary size, and H100 memory bandwidth.

Memory contains two main components:

1. model parameters, stored in BF16;
2. the KV cache, whose per-sequence size scales with sequence length, KV-head count, head dimension, layer count, separate key and value vectors, and two bytes per BF16 value.

For batch size $B$, total memory is parameter bytes plus $B$ times the per-sequence KV-cache bytes. Under the simplified model, one token step reads that state, so latency is total memory divided by memory bandwidth. Throughput is the $B$ tokens completed in parallel divided by that latency.

For this configuration, the parameter-count sanity check gives about 13 billion parameters. The per-sequence KV cache is about 838 million bytes, so total memory and latency are affine functions of $B$. Throughput has the form $B/(a+bB)$: it improves with batch size by amortizing parameter reads, but approaches an asymptote rather than growing without bound.

### Source reconciliation

Slides 14-15 make the simplified model explicit:

$$
P
=2VD+3DFL+\left(2DNH+2DKH\right)L,
$$

$$
M_{\mathrm{params}}=2P,
$$

$$
M_{\mathrm{KV,seq}}=4SKHL,
$$

$$
M(B)=M_{\mathrm{params}}+B M_{\mathrm{KV,seq}},
$$

$$
\ell(B)=\frac{M(B)}{W_{\mathrm{HBM}}},
\qquad
\tau(B)=\frac{B}{\ell(B)}.
$$

The deck's Llama 2 13B values are $S=1024$, $D=5120$, $F=13824$, $N=K=40$, $H=128$, $L=40$, $V=32000$, and $W_{\mathrm{HBM}}=3.35\times10^{12}$ bytes/s.

### Additional explanation

Substitution gives approximately

$$
P=13{,}015{,}449{,}600,
\qquad
M_{\mathrm{params}}=26{,}030{,}899{,}200\ \text{bytes},
$$

and

$$
M_{\mathrm{KV,seq}}=838{,}860{,}800\ \text{bytes}.
$$

The model treats every byte in parameters and KV state as if it crosses the bandwidth boundary once per generation step. Real implementations add allocator, metadata, kernel, collective, and scheduling costs; they may also retain some data in cache. The point of the model is not cycle-accurate prediction but a transparent explanation of the trends.

Writing

$$
\ell(B)=\frac{M_{\mathrm{params}}+BM_{\mathrm{KV,seq}}}{W}
$$

makes the competing effects visible. Parameter traffic is shared across the batch, but cache traffic grows per request. Consequently,

$$
\lim_{B\to\infty}\tau(B)
=\frac{W}{M_{\mathrm{KV,seq}}},
$$

so the KV cache fixes the idealized throughput ceiling.

## 10. Batch size trades latency for throughput and is limited by capacity

**Transcript coverage:** lines 2638-2922

### What the lecturer said - transcript only

At batch size one, the lecturer reports a latency of 0.08 seconds per token and throughput of 124 tokens per second. Increasing batch size raises latency because the server must read more KV-cache state before the batched step completes. At the same time, it raises throughput because the shared parameter-read cost is amortized across more requests.

The lecturer compares this with a bus. An individual passenger waits for the bus and for the batch to move together, so the trip has greater latency. The bus nevertheless transports many passengers at once, so aggregate throughput is high.

The tradeoff eventually runs into accelerator memory. A batch of 256 does not fit in the 80-GB H100 example under full multi-head attention, and even a larger-memory accelerator eventually reaches a capacity limit. Throughput gains also diminish as the curve approaches its asymptote.

The operational rule is:

- smaller batches give better per-request latency but lower throughput;
- larger batches give higher throughput but worse per-request latency.

Parallelism provides another axis. Running $M$ independent model replicas leaves the latency of each replica approximately unchanged while multiplying aggregate throughput by $M$. Sharding one model and its KV cache across devices is more complex and is not developed in the lecture.

TTFT is mainly determined by prefill time. Smaller prefill batches improve TTFT, while larger generation batches improve aggregate decoding throughput. An inference system may therefore use different batching policies for the two phases.

### Source reconciliation

The two spoken batch-one numbers are inconsistent: 124 tokens/s implies about $1/124=0.0081$ seconds per token, not 0.08. The slide formula and stated configuration also give approximately 0.0080 seconds per token and 124.7 tokens/s. These notes preserve the spoken 0.08 above but use 0.008 as the internally consistent result in calculations.

The simplified model gives:

| Batch $B$ | Memory | Latency per step | Throughput |
|---:|---:|---:|---:|
| 1 | 26.87 GB | 0.0080 s/token | 124.7 tokens/s |
| 64 | 79.72 GB | 0.0238 s/token | 2,689 tokens/s |
| 256 | 240.78 GB | 0.0719 s/token | 3,562 tokens/s |

The last row exceeds the example H100's 80-GB capacity.

### Additional explanation

The table explains why latency and throughput are not reciprocals once $B>1$. The reciprocal of 0.0238 seconds is only about 42 *batch steps* per second, but each step completes 64 tokens, yielding about 2,689 total tokens per second.

The capacity constraint is

$$
B
\le
\left\lfloor
\frac{M_{\mathrm{device}}-M_{\mathrm{params}}}
{M_{\mathrm{KV,seq}}}
\right\rfloor.
$$

This is why reducing KV bytes can improve both metrics and unlock a larger batch. It lowers the cache traffic at a fixed batch, and it increases the maximum batch that fits when throughput is the objective.

---

# Part III - Reducing the KV cache without losing too much accuracy

## 11. The optimization target: less memory, similar model quality

**Transcript coverage:** lines 2923-3000

### What the lecturer said - transcript only

The arithmetic analysis supplies a framework for making inference faster. The available methods cut across architecture and systems design. The most direct conclusion is that memory is the generation bottleneck and the KV cache can become as large as, or larger than, the parameter storage at sufficiently large batch size.

The next group of methods therefore attempts to shrink the KV cache. Every reduction is potentially lossy, so speed must be evaluated together with model accuracy rather than in isolation.

### Additional explanation

The design problem is a Pareto optimization:

$$
\text{minimize memory and runtime}
\quad\text{subject to an acceptable quality loss}.
$$

There is no universal best point. Interactive chat may prioritize latency and long-context capacity; offline data processing may prefer maximum throughput; a high-stakes application may reject an architecture change whose average benchmark score looks acceptable but whose worst-case behavior degrades.

## 12. Grouped-query attention shares KVs across query heads

**Transcript coverage:** lines 3001-3291

### What the lecturer said - transcript only

In ordinary multi-head attention, every query head has its own key and value head. **Grouped-query attention (GQA)** keeps $N$ query heads but reduces the number of KV heads to $K$. Several query heads share each KV head.

The endpoints are:

- multi-head attention (MHA): $K=N$, so there is no KV-head reduction;
- multi-query attention (MQA): $K=1$, which the lecturer says is very fast but performs poorly and is not commonly used;
- GQA: an intermediate $1<K<N$ chosen to balance quality and speed.

Reducing KV heads shrinks the cache by a factor of $N/K$. Because generation is memory-bound, this improves both latency and throughput at a fixed batch size. In the Llama example, moving from $K=40$ to $K=8$ reduces memory, improves the fixed-batch metrics, and permits batch size 256 to fit where full MHA did not. Increasing the batch again sacrifices some latency but gains more throughput.

Accuracy still has to be checked. The original GQA paper reports a favorable quality-speed tradeoff on its tested models. The lecturer warns against treating one paper's result as universal: later DeepSeek results suggest that GQA can hurt on harder evaluations. Empirical quality claims depend on the model and experiment, unlike the cache-size arithmetic.

### Source reconciliation

Slides 18-20 verify that $K$ is the KV-head count and state the cache reduction as $N/K$. Their concrete example uses $N=40$, $K=8$, and $B=64$, followed by $B=256$. The accuracy table shows similar averages for its MHA, MQA, and GQA variants, while the later DeepSeek table on slide 22 shows MHA ahead of GQA and MQA on several hard benchmarks. The slides therefore support the lecturer's caution that the empirical conclusion is not architecture-independent.

### Additional explanation

With $G=N/K$ query heads sharing one KV head, the cache becomes

$$
M_{\mathrm{KV,GQA}}
=4BSLKH
=\frac{K}{N}M_{\mathrm{KV,MHA}}.
$$

Using the deck's simplified Llama configuration, $K=8$ reduces per-sequence cache size from about 838.9 MB to 167.8 MB. Including the slightly smaller K/V projection weights, the same model gives these idealized values:

| KV heads $K$ | Batch $B$ | Total memory | Latency | Throughput |
|---:|---:|---:|---:|---:|
| 40 | 64 | 79.72 GB | 0.0238 s | 2,689 tokens/s |
| 8 | 64 | 33.41 GB | 0.0100 s | 6,417 tokens/s |
| 8 | 256 | 65.63 GB | 0.0196 s | 13,068 tokens/s |

These numbers are consequences of the lecture's ideal bandwidth model, not measured vLLM performance. They isolate the benefit of fewer cache bytes while omitting kernel and scheduling overhead.

## 13. Multi-head latent attention compresses before caching

**Transcript coverage:** lines 3292-3548

### What the lecturer said - transcript only

**Multi-head latent attention (MLA)**, associated with DeepSeek, retains a rich set of keys and values for computation but stores a much lower-dimensional latent representation in the cache.

Ordinary attention projects a token activation directly into key and value vectors whose total width is comparable to the model dimension. MLA first projects the activation down to a latent vector of width $C$. DeepSeek V2 is described as reducing a roughly 16,000-dimensional representation to 512 dimensions. When attention needs keys and values, it materializes them from this compressed latent. The cache stores the small latent rather than every full key and value vector.

There is a positional-encoding complication: this compression is not directly compatible with applying RoPE to the full keys and values. The architecture adds separate dimensions to carry the RoPE information. Despite that addition, the total cached representation is still far smaller.

As with GQA, reducing cache bytes improves latency and throughput almost linearly until another bottleneck takes over. The empirical comparison cited by the lecturer argues that GQA loses quality relative to MHA on hard benchmarks, while MLA is roughly as good as MHA and in some reported results slightly better. The lecturer treats these as experimental rather than mathematical conclusions.

A student asks why not simply reduce the model dimension. The lecturer says the displayed ablations do not answer that comparison directly. His guess is that shrinking $D$ damages everything indiscriminately, whereas these methods try to find a particular part of the model that can be compressed. Which location tolerates compression cannot be known with certainty in advance; it requires experiments.

### Source reconciliation

Slide 21 writes the construction as a compressed latent $c=W_c h$ of width $C$, from which full keys and values are projected when needed. It gives the concrete DeepSeek V2 reduction

$$
NH=16{,}384
\quad\longrightarrow\quad
C=512,
$$

and specifies 64 additional RoPE dimensions, for 576 cached dimensions in total. The transcript mentions the extra dimensions but not their exact number.

Slides 22-23 show the empirical claims more precisely: MHA substantially outperforms GQA and MQA in one dense 7B comparison, while MLA reduces KV elements per token sharply and is competitive with or better than MHA for most listed MoE benchmarks. These are results from the displayed DeepSeek study, not a general guarantee.

### Additional explanation

MLA is a learned bottleneck. Instead of committing to a fixed rule that several heads share identical KVs, it learns which information should survive in a compact latent and learns how to reconstruct the representations used by attention.

The compression factor suggested by the slide is

$$
\frac{16{,}384}{576}\approx 28.4,
$$

before considering other architecture details. The added RoPE subspace illustrates a recurring pattern in efficient architectures: preserve a small component that must obey a special operation and compress the remainder more aggressively.

## 14. Cross-layer attention shares KVs across depth

**Transcript coverage:** lines 3549-3627

### What the lecturer said - transcript only

Ordinary Transformers compute and store distinct keys and values at every layer. **Cross-layer attention (CLA)** computes KVs only in a subset of layers and lets another layer reuse a preceding layer's KV cache.

This is sharing across depth. GQA shares KVs across heads; CLA shares them across layers. The cited paper sweeps cache size by varying head count and head dimension and reports that adding CLA improves the Pareto frontier between validation quality and KV-cache size.

### Source reconciliation

Slides 24-25 depict a pair of Transformer layers in which the upper layer reuses the lower layer's projected keys and values. The plotted 1B-model results place several CLA configurations on a better validation-perplexity versus KV-bytes frontier than comparable non-CLA configurations. The plot is evidence for that study, not a proof that every model benefits.

### Additional explanation

If only one of every $r$ layers stores its own KVs, the cache's layer factor can ideally fall from $L$ to roughly $L/r$, though the exact saving depends on the sharing schedule and any layers that retain full attention.

The risk differs from GQA. GQA reduces diversity across heads at one depth; CLA reduces the layer-specific transformation of memory across depth. A hybrid architecture can vary both axes, which is why cache width alone does not fully describe representational capacity.

## 15. Local, linear, and hybrid attention trade memory for access to history

**Transcript coverage:** lines 3628-4015

### What the lecturer said - transcript only

**Local or sliding-window attention** lets each new token attend only to the most recent fixed number of positions rather than the full prefix. The cache retained for a local layer is therefore independent of total sequence length once the window is full, which is especially attractive for long contexts.

The effective receptive field can exceed one layer's window because information propagates across layers. Variants can dilate the selected positions or combine a fixed set of global positions with a local window. The cost is reduced expressivity and, empirically, possible accuracy loss. A common remedy is a hybrid model that interleaves local-attention layers with full-attention layers.

A student asks how this compares with linear attention. The lecturer does not develop the full linear-attention literature but describes its broad idea: replace an ever-growing KV cache with a compressed recurrent representation of the history. A naive version can sum key-value contributions into one state; more sophisticated examples include gated mechanisms, DeltaNet-style methods, and Mamba. Full attention, sliding-window attention, and linear or state-space layers can be combined because they capture different information:

- sliding windows preserve high-resolution recent details;
- recurrent compression can retain a broader summary of the past;
- occasional full attention preserves direct long-range access.

For very long contexts, compression is not a free lunch. In a needle-in-a-haystack task, squeezing the whole history into a small state can discard the one detail that must later be retrieved. The student presses on whether Mamba or DeltaNet layers are preferable to repeated sliding windows. The lecturer cautiously says those recurrent mechanisms are more powerful and can likely represent some sliding-window behavior through their recurrence, whereas a fixed sliding window has no additional memory mechanism beyond its cutoff. He presents this as an intuition, not a settled universal ranking.

### Source reconciliation

Slide 26 shows four patterns: full $n^2$ attention, a dense local window, a dilated window, and global-plus-local attention. It states that the effective context scales linearly with layer count and that hybrid global layers are used because purely local attention can hurt accuracy.

### Additional explanation

For a local window of width $W$, a layer's KV storage changes from

$$
O(SLKH)
\quad\text{to approximately}\quad
O(WLKH),
$$

once $S>W$. Its decode attention reads $W$ positions instead of $S$ positions. The tradeoff is structural: no optimizer or kernel can recover a token that the attention mask makes inaccessible at that layer.

A rough taxonomy is useful:

| Mechanism | Persistent history | Strongest access pattern | Main failure risk |
|---|---|---|---|
| Full attention | all token KVs | exact content-based lookup anywhere | memory grows with context |
| Sliding window | most recent $W$ token KVs | exact recent lookup | old details disappear locally |
| Linear/state-space | fixed-size recurrent state | compressed global summary | collisions or forgotten details |
| Hybrid | mixture of the above | combines local fidelity and periodic global access | greater architectural complexity |

The lecturer's "more powerful" comment should be read as a representational observation about recurrence, not as a guarantee of better accuracy, speed, or hardware utilization for every implementation.

## 16. Compressed and sparse attention select a smaller useful history

**Transcript coverage:** lines 4016-4159

### What the lecturer said - transcript only

The lecturer briefly surveys DeepSeek's newer attention mechanisms, noting that the names and acronyms are easy to confuse. The diagram communicates the main idea.

**Compressed attention** combines every $M$ KV tokens into a smaller number of compressed entries. **DeepSeek sparse attention** then keeps only a selected subset. A lightweight set of queries and keys performs a cheaper attention-like scoring pass to produce indices, after which the expensive attention computation uses the selected entries. A further heavily compressed variant reduces the representation again.

The section's unifying goal is to reduce KV-cache size because cached Transformer generation is memory-bound. Candidate reductions operate across:

- heads, as in GQA;
- latent or head dimensions, as in MLA;
- layers, as in CLA;
- positions, as in local or sparse attention;
- all history through linear or state-space compression.

Every method must be checked for accuracy loss. The lecturer also points to diffusion language models as a non-autoregressive direction that may generate faster than ordinary one-token-at-a-time decoding.

### Source reconciliation

Slide 27 labels the three stages as **Compressed Sparse Attention (CSA)**, which compresses every $m$ tokens into one entry; **DeepSeek Sparse Attention (DSA)**, whose lightning indexer selects top-$k$ entries; and **Heavily Compressed Attention (HCA)**, which compresses further. Slide 26 additionally states that the displayed DeepSeek V4 architecture supports a one-million-token context. That context-length number appears only on the slide, not in the spoken transcript.

### Additional explanation

Sparse selection separates two jobs:

1. estimate cheaply which history locations matter;
2. spend full attention compute only on those locations.

If the selector examines a compact representation of all history and returns $k\ll S$ positions, the expensive stage can scale with $k$ rather than $S$. The selector itself must remain cheap and must have high recall: a missed relevant token cannot be recovered by the later exact computation.

This is related to information retrieval inside the model. Compression constructs a smaller indexable memory; the lightweight indexer proposes candidates; the full attention layer reranks or consumes them.

---

# Part IV - Lower precision, smaller models, and exact shortcuts

## 17. Quantization reduces the bytes per value

**Transcript coverage:** lines 4160-4365

### What the lecturer said - transcript only

Quantization is presented as a systems-oriented way to make inference state smaller: store numbers at lower precision. Fewer bytes reduce memory traffic and thereby improve latency and throughput in a memory-bound workload, provided accuracy remains acceptable. The design space ranges from BF16 down to roughly four-bit integer representations.

**Quantization-aware training (QAT)** prepares the model during training. The forward pass quantizes and dequantizes values so the model experiences the resulting approximation errors while its weights can still adapt. This generally improves robustness to the final low-precision representation, but it requires expensive large-scale training.

**Post-training quantization (PTQ)** modifies an already trained model and is therefore much cheaper. A basic method chooses a scale and zero point for each tensor or layer and maps the values into the available integer range. Naive tensorwise quantization often loses too much accuracy.

More sophisticated PTQ methods exploit structure:

- **GPTQ** uses Hessian-related information, quantizes layer by layer, tracks the induced errors, and pushes corrections into weights that have not yet been quantized.
- **Activation-aware weight quantization (AWQ)** observes that some activation channels are much larger than others. Weights interacting with those channels matter disproportionately, so the method treats them more carefully rather than assigning every weight identical precision.

The lecture's picture-based intuition is to retain or allocate higher precision to a small number of important channels while heavily quantizing the rest.

### Source reconciliation

Slides 28-29 add the scalar affine-quantization mechanics:

$$
x_q=\operatorname{round}(x/s)+z,
\qquad
\hat{x}=(x_q-z)s,
$$

where $s$ is a scale and $z$ is a zero point. They list FP32 as 4 bytes, BF16 as 2 bytes, FP8 and INT8 as 1 byte, and INT4 as 0.5 byte per value before metadata and packing overhead.

Slide 30 refines the spoken AWQ description. Keeping the top 1 percent of salient weights in FP16 can preserve perplexity but creates a mixed-precision path with poor hardware efficiency. The hardware-friendly AWQ diagram instead rescales salient channels before quantizing the entire weight matrix to INT3. Slide 29 reports the cited experiment's FP16-to-INT3 result as 4x lower weight memory and 3.2x speedup. These are slide-specific results, not universal conversion factors for end-to-end serving.

### Additional explanation

For symmetric $b$-bit quantization one often chooses a scale from a calibration range and clips to approximately $[-2^{b-1},2^{b-1}-1]$. Asymmetric quantization adds a zero point so an off-center range can use the integer codes more efficiently.

The difficult cases are outliers. If one large value sets a tensorwide scale, most ordinary values occupy only a few integer levels. Per-channel or groupwise scales reduce this interference but add scale metadata and can complicate kernels.

Quantization can target different state:

- **weight-only quantization** reduces parameter reads;
- **KV-cache quantization** reduces request-specific cache traffic and capacity;
- **activation quantization** can reduce intermediate traffic but must handle input-dependent ranges.

The nominal bit width is therefore not a complete performance description. Kernel support, packing, dequantization, scale reads, accumulator width, and any high-precision exceptions determine realized speed.

## 18. Model pruning removes structure and repairs the result

**Transcript coverage:** lines 4366-4647

### What the lecturer said - transcript only

Model pruning starts with an expensive model, removes selected components, and then repairs the damaged model with more training. The cited NVIDIA work first estimates the importance of structures such as hidden units and entire layers, retains the most important ones, and trims the rest. The immediately pruned network is not expected to be good, so it is post-trained or distilled on relevant data and tasks.

The lecturer describes a result that reduced a 15-billion-parameter model to about 8 billion parameters with relatively little accuracy loss and much less training than a new model would require.

This fits a broader pair of recipes for reducing inference complexity:

1. define a faster architecture and train it from scratch;
2. define a faster architecture, initialize whatever pieces can be inherited from a larger model, and repair the hybrid with distillation.

The target may be fewer parameters, a smaller KV cache, or both. The goal remains lower inference cost without an unacceptable accuracy loss.

A student asks how important and unimportant layers are distinguished. The lecturer says a calibration set is passed through the model and activation magnitudes are inspected. Dead or nearly zero units are natural removal candidates; consistently large channels are more likely to matter.

The student challenges magnitude as a complete importance measure: a neuron might always equal 100 without conveying varying information. The lecturer agrees that it cannot simply be removed because downstream computations may depend on the offset. Mean and variance can both be examined; a high-mean, low-variance component might sometimes be absorbed as a bias rather than retained as a full dynamic unit. The whole approach relies on the empirical fact that trained models exhibit exploitable differences across components. If all channels were equally important, these methods would not work.

### Source reconciliation

Slide 31 presents an iterative pipeline: estimate importance, rank components, trim them, and distill the pruned model. It specifies a small calibration set of 1,024 samples, a number not given in the spoken answer. Slide 32 plots a Mintron 8B model derived from a 15B starting model and highlights much lower training cost; the transcript rounds this to the 15B-to-8B example and describes the quality loss qualitatively.

### Additional explanation

Structured pruning removes units that map to dense hardware dimensions - complete heads, channels, or layers - and can therefore yield a genuinely smaller dense model. Unstructured pruning sets individual weights to zero; it saves work only when the serving hardware and kernels exploit sparsity efficiently.

Magnitude is a heuristic, not causal proof of importance. More informative scores can combine activation statistics, weight magnitude, gradients, or estimated loss change. The repair phase matters because remaining components can adapt to assume functions previously distributed across the removed structure.

Pruning and distillation also create a natural draft model for speculative decoding: if the smaller repaired model is accurate enough, serve it directly; if not, use it to propose work that a larger model verifies.

## 19. Speculative decoding uses a cheap model to draft and the target to verify

**Transcript coverage:** lines 4648-4989

### What the lecturer said - transcript only

The preceding cache, quantization, and pruning methods are lossy because they may alter the served model's predictions. **Speculative sampling** or **speculative decoding** offers an elegant distribution-preserving shortcut.

The method exploits an asymmetry between the two inference phases. Given a sequence, a target model can evaluate several known positions in parallel during a prefill-like pass. Generating those same positions autoregressively one by one is memory-bound and slower. In short, checking a proposed sequence is cheaper than producing it token by token with the large model.

A small draft model $p$ first generates several candidate tokens autoregressively - four in the lecturer's example. The expensive target model $q$ then evaluates the whole candidate block in parallel. A proposed token is accepted with probability

$$
\min\left(1,\frac{q(x)}{p(x)}\right).
$$

If a token is rejected, the algorithm samples from a residual distribution and exits the current verification loop. Unlike ordinary rejection sampling that may return no candidate, this procedure always advances and is guaranteed to produce an exact sample from the target model's distribution. The lecturer skips the proof and says it follows the same probability accounting as rejection sampling.

There is a lookahead tradeoff. Too few draft tokens fail to exploit parallel target-model checking. Too many increase draft work and the chance that an early rejection wastes later proposals. In the displayed experiment, the sweet spot is around three or four draft tokens.

The draft is normally much smaller than the target but should imitate it closely so that acceptance is high. Distillation is therefore useful. The lossy compression methods from the earlier sections can produce a candidate draft: if its quality is already sufficient, serve it; otherwise let the original model correct it through speculative decoding. The lecturer notes that a large later literature extends the original algorithms but does not cover it in detail.

### Source reconciliation

Slide 34 supplies algorithm details that the transcript references but does not spell out. The draft proposes $K$ tokens; the target computes $K+1$ conditional distributions in parallel. At the first rejection, replacement is sampled from the normalized positive residual

$$
r(x)\propto\left(q(x)-p(x)\right)_+.
$$

If all $K$ proposals are accepted, one additional token is sampled from the target distribution. Slide 35 contains a two-token proof example, but the lecturer explicitly says the spoken lecture will skip the proof. The slide's reported study shows roughly 1.92x to 2.46x speedups on its listed XSum and HumanEval settings; these results are study-specific.

Slide 36 illustrates the lookahead tradeoff and gives practical draft-target pairs such as 8B drafting for 70B and 1B drafting for 8B. Slides 36-37 also name Medusa and EAGLE as draft-model extensions; the transcript only notes that many extensions exist.

### Additional explanation

For one proposed token $x$, the accepted mass is

$$
p(x)\min\left(1,\frac{q(x)}{p(x)}\right)
=\min(p(x),q(x)).
$$

Where $p$ undersamples the target, the remaining mass is $(q-p)_+$. Sampling a replacement from that positive residual exactly fills the target distribution. This is why the procedure is lossless *in distribution* even though any particular run can accept or reject different tokens.

Speed depends on more than draft parameter count. A good draft must satisfy both:

- low cost per proposed token;
- high agreement with the target.

A tiny but inaccurate draft can lose to a somewhat larger distilled draft because frequent early rejection wastes the candidate suffix. The optimal lookahead also depends on target batch efficiency, prompt length, hardware, and sampling temperature.

---

# Part V - Dynamic serving workloads

## 20. Continuous and selective batching keep ragged traffic useful

**Transcript coverage:** lines 4990-5175

### What the lecturer said - transcript only

Live serving is more irregular than training. Requests arrive at different times, may share different prefixes, and request different output lengths. Instead of a dense rectangular batch of equal-length sequences, the server sees a changing ragged workload.

The Orca system introduced **continuous batching**. The scheduler decodes one token for every active sequence at each iteration. A sequence that finishes is immediately removed, and a newly arrived request can enter the batch without waiting for every older request to complete. The batch is therefore continuously updated rather than fixed for an entire generation.

Different prefix lengths create a tensor-shape problem. Attention for lengths 3, 9, and 5 performs different-sized request-specific computations, so these attention operations cannot simply be represented as one dense equal-length tensor without padding or special handling. **Selective batching** treats attention per sequence but concatenates all tokens for the non-attention computation. The MLP, for example, can process the three sequences as one 17-token "mega-sequence" because its operation is independent at each token and uses shared weights.

### Source reconciliation

Slides 37-38 call continuous batching **iteration-level scheduling**. Their selective-batching example processes attention on separate $[3,H]$, $[9,H]$, and $[5,H]$ inputs, then concatenates non-attention work into a $[3+9+5,H]=[17,H]$ input. This makes the transcript's verbal example explicit.

### Additional explanation

Continuous batching raises occupancy without imposing a long fixed-batch barrier. Its state transition at each decoding iteration is roughly:

1. form a batch from currently admitted requests;
2. generate one token for each;
3. retire completed requests;
4. admit waiting requests if memory permits;
5. repeat.

Selective batching preserves the advantage identified in the arithmetic analysis. Request-specific attention remains ragged, while the shared MLP weight matrices see as many token rows as possible. Modern implementations use metadata and specialized kernels to avoid padding the ragged attention to the longest request.

Scheduling policy still matters. A server may prioritize short jobs, cap long contexts, reserve capacity for prefill, or separate TTFT-sensitive and throughput-oriented queues. Continuous batching supplies the mechanism, not a single fairness policy.

## 21. PagedAttention manages KV memory like virtual memory

**Transcript coverage:** lines 5176-5472

### What the lecturer said - transcript only

PagedAttention, introduced with vLLM, addresses how dynamic KV caches are placed in memory. A naive server reserves a contiguous region up to each request's maximum possible output length - for example, a 1,024-token maximum even when the response stops much earlier. If a request stops early, most of that region is unused; this is internal fragmentation. Gaps between separately allocated regions can also be too small to reuse effectively; this is external fragmentation.

The systems solution borrows paging from operating systems. Divide a logical sequence's KV cache into fixed-size blocks and place those blocks non-contiguously in physical memory. The lecture's example chunks "Four score and seven years ago, our fathers brought forth" into four-token blocks. A block table records the mapping, so logical order does not require physical contiguity.

Blocks also enable cache sharing across requests. Common system prompts can be encoded once and referenced by many queries. Applications that sample several completions from one prompt can likewise share the prompt blocks while their generated suffixes remain unique.

The lecturer explains **copy-on-write** with multiple samples from the same prefix. The samples initially reference the same physical blocks. If their next tokens remain identical, sharing can continue. Once they diverge, the affected block is copied and each branch writes its own token, while earlier prefix blocks remain shared.

The lecture omits additional kernel optimizations for lack of time. Its broader message is that operating-system ideas can organize the dynamic state of an inference server.

### Source reconciliation

Slides 39-42 distinguish internal and external fragmentation visually, show logical-to-physical block tables, and illustrate block-level copy-on-write with a reference count changing from 2 to 1. Slide 41 gives two explicit sharing cases: a common few-shot translation or system prompt, and multiple sampled responses per prompt such as program synthesis.

Slide 42 lists further vLLM optimizations that were not developed in speech: fusing block reads with attention, using recent FlashAttention and FlashDecoding kernels, and using CUDA graphs to reduce launch overhead.

### Additional explanation

Paging separates a sequence's **logical address space** from physical placement. Appending tokens requires finding or allocating the next block, not reserving the full maximum sequence in advance. This reduces wasted capacity and makes admission of new requests more flexible.

If the block size is too large, the last block of many sequences wastes space. If it is too small, block-table metadata and indirect addressing overhead increase. Block size is therefore another systems tradeoff.

Prefix sharing can reduce both compute and memory:

- prefill for a cached system prompt need not be repeated;
- its physical KV blocks can be referenced rather than copied;
- copy-on-write preserves correctness when sampled continuations diverge.

The idea does not reduce the mathematical KV state required by one unique sequence. It reduces fragmentation, duplication, and allocation overhead across a changing collection of sequences.

## 22. Final synthesis and the case for inference-native architectures

**Transcript coverage:** lines 5473-5587

### What the lecturer said - transcript only

The lecture closes by reiterating that inference is both important and qualitatively different from training. The same trained model is asked to operate sequentially on a dynamic live workload, and cached generation becomes strongly memory-bound.

The techniques surveyed attack that cost from several directions:

- new attention or sequence architectures reduce cached state;
- quantization stores state with fewer bits;
- pruning and distillation create smaller models;
- speculative sampling lets a cheap draft propose work while preserving the target model's output distribution;
- paging and related serving mechanisms manage dynamic memory and requests.

The recurring principle is to reduce parameters or KV-cache traffic without hurting accuracy too much. Systems ideas such as paging and speculative execution can be repurposed for inference servers.

The lecturer emphasizes one direction that time did not permit him to explore fully: architectures designed for inference. Conventional attention and its growing KV cache make the Transformer fundamentally unfriendly to sequential serving. State-space models, linear attention, and diffusion-style generation may offer larger improvements if they are designed around inference rather than inheriting a training-oriented architecture.

The lecture ends with a transition back to Tatsu Hashimoto for the second scaling-laws lecture.

### Additional explanation

The methods operate at different layers and can be composed:

| Layer of the stack | Representative intervention | Resource changed |
|---|---|---|
| Architecture | GQA, MLA, CLA, local or linear attention | KV bytes and attention work |
| Representation | quantization | bytes per parameter or cache value |
| Model construction | pruning and distillation | parameter count and often cache width/depth |
| Decoding algorithm | speculative decoding | number of expensive sequential target steps |
| Scheduler | continuous/selective batching | realized reuse and occupancy |
| Memory manager | PagedAttention and prefix sharing | fragmentation and duplicate KV state |

The most useful diagnostic question is not simply "Is inference memory-bound?" It is "Which bytes are moving in this phase, which of them are shared, and which can be removed, compressed, or reused without changing the required behavior?"

---

# Consolidated takeaways

1. Inference is a repeated cost used in products, evaluation, reinforcement learning, agents, and batch processing; at deployment scale, even modest efficiency gains matter.
2. TTFT, per-request latency, and aggregate throughput measure different user or system objectives and can move in opposite directions.
3. Training sees a complete sequence and parallelizes across token positions. Autoregressive generation exposes only one unknown next position at a time.
4. Naively rerunning the full prefix gives cubic total attention work across a growing generation; causal KV caching removes redundant prefix projections.
5. Prefill remains a large parallel computation, while decode repeatedly reads model weights and request-specific KV state for one new token.
6. A thin matrix multiplication has intensity near its batch size. At batch one it resembles a matrix-vector product and is strongly memory-bound.
7. A generation MLP can gain arithmetic intensity from concurrent requests because its weights are shared across the batch.
8. Cached generation attention has intensity $S/(S+1)<1$. Batching does not improve it because every request adds a different KV cache and proportional bytes.
9. In the simplified bandwidth model, larger batches amortize parameter reads and raise throughput, but they increase per-step latency and KV capacity requirements.
10. GQA shares keys and values across query heads; MLA stores a learned latent cache; CLA shares KVs across layers; local, linear, and sparse attention reduce the number of history entries retained or read.
11. Cache-reduction arithmetic is predictable, but accuracy effects are empirical. Results from one model or benchmark should not be treated as universal.
12. Quantization reduces bytes per value, while pruning and distillation reduce model structure. Hardware-friendly packing and kernels determine whether nominal compression becomes measured speed.
13. Speculative decoding is exact in distribution: a cheap draft proposes several tokens, the target verifies them in parallel, and rejection-sampling corrections recover the target distribution.
14. Continuous batching admits and retires requests at decoding-step boundaries. Selective batching preserves ragged attention while combining shared non-attention work.
15. PagedAttention maps logical KV blocks to non-contiguous physical blocks, reducing fragmentation and enabling prefix sharing with copy-on-write.
16. The Transformer's growing KV cache is an architectural source of inference cost. Architectures designed around sequential serving may offer improvements beyond incremental kernel optimization.

# Key equations

## Matrix-multiplication intensity

For BF16 $X\in\mathbb{R}^{B\times D}$ and $W\in\mathbb{R}^{D\times F}$:

$$
F_{\mathrm{matmul}}=2BDF,
$$

$$
Q_{\mathrm{matmul}}=2(BD+DF+BF),
$$

$$
I_{\mathrm{matmul}}
=\frac{BDF}{BD+DF+BF}
\longrightarrow B
\quad (B\ll D,F).
$$

## Accelerator balance point

$$
I_{\mathrm{accelerator}}
=\frac{\text{peak FLOP/s}}
{\text{memory bandwidth}}.
$$

For the lecture's H100 values:

$$
I_{\mathrm{H100}}
=\frac{989\times10^{12}}
{3.35\times10^{12}}
\approx295\ \text{FLOPs/byte}.
$$

## KV-cache memory

With $B$ sequences, $S$ cached tokens, $L$ layers, $K$ KV heads, $H$ values per head, separate keys and values, and BF16 storage:

$$
M_{\mathrm{KV}}=4BSLKH\ \text{bytes}.
$$

## Gated-MLP intensity

$$
F_{\mathrm{MLP}}=6BTDF,
$$

$$
Q_{\mathrm{MLP}}=4BTD+4BTF+6DF,
$$

$$
I_{\mathrm{MLP}}
=\frac{6BTDF}{4BTD+4BTF+6DF}
\longrightarrow BT
\quad (BT\ll D,F).
$$

Thus prefill gives approximately $BS$, and one-token generation gives approximately $B$.

## Attention intensity

$$
F_{\mathrm{attn}}=4BSTD,
$$

$$
Q_{\mathrm{attn}}=4BD(S+T),
$$

$$
I_{\mathrm{attn}}=\frac{ST}{S+T}.
$$

The two phases specialize to

$$
I_{\mathrm{prefill}}=\frac{S}{2}
\quad (T=S),
$$

$$
I_{\mathrm{decode}}=\frac{S}{S+1}<1
\quad (T=1).
$$

## Simplified Transformer memory, latency, and throughput

The lecture's parameter count is

$$
P=2VD+3DFL+(2DNH+2DKH)L.
$$

For BF16 parameters,

$$
M(B)=2P+4BSLKH.
$$

Under the perfect-overlap, bandwidth-bound approximation,

$$
\ell(B)=\frac{M(B)}{W_{\mathrm{HBM}}},
\qquad
\tau(B)=\frac{B}{\ell(B)}.
$$

The cache-capacity bound is

$$
B_{\max}
=\left\lfloor
\frac{M_{\mathrm{device}}-2P}
{4SLKH}
\right\rfloor.
$$

## GQA cache reduction

For $N$ query heads and $K$ KV heads:

$$
G=\frac{N}{K},
\qquad
\frac{M_{\mathrm{KV,GQA}}}{M_{\mathrm{KV,MHA}}}
=\frac{K}{N}.
$$

## Affine quantization

$$
x_q=\operatorname{round}(x/s)+z,
\qquad
\hat{x}=(x_q-z)s.
$$

## Speculative-decoding acceptance

For draft distribution $p$ and target distribution $q$:

$$
a(x)=\min\left(1,\frac{q(x)}{p(x)}\right),
$$

and after rejection the replacement distribution is

$$
r(x)\propto(q(x)-p(x))_+.
$$

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Arithmetic intensity | Useful floating-point operations performed per byte transferred. |
| Autoregressive generation | Producing each next token conditional on the already known prefix. |
| AWQ | Activation-aware weight quantization, which protects or rescales weights associated with salient activation channels. |
| Batch step | One decoding iteration that produces one token for every active request in a batch. |
| Compute-bound | A regime where arithmetic throughput is the primary performance limit. |
| Continuous batching | Adding and removing requests at decoding-iteration boundaries instead of fixing a batch for whole generations. |
| Copy-on-write | Sharing a physical prefix block until one sequence diverges, then copying only the block that must be modified. |
| Cross-layer attention (CLA) | Reusing keys and values across layers to reduce the cache's depth factor. |
| Decode or generation | The sequential phase that produces new response tokens, usually one per request per step. |
| Distillation | Training a smaller or modified model to imitate a larger model's behavior. |
| Draft model | The cheaper proposal model used in speculative decoding. |
| External fragmentation | Free gaps between allocations that cannot be used effectively for a new request. |
| GQA | Grouped-query attention, in which several query heads share each KV head. |
| GPTQ | A post-training weight-quantization method that uses curvature-related information and error compensation. |
| HBM | High-bandwidth memory attached to an accelerator. |
| Internal fragmentation | Unused space inside a region reserved for a request, often because generation ends early. |
| KV cache | Persistent key and value vectors for prior tokens, stored so a causal prefix is not recomputed. |
| Latency | Per-request delay or seconds per generated token in the lecture's performance model. |
| Linear attention | A family of mechanisms that replaces a full growing KV history with a recurrent or factorized summary. |
| Local attention | Attention restricted to a fixed recent window or another sparse local pattern. |
| Memory-bound | A regime where moving bytes, rather than performing arithmetic, limits runtime. |
| MHA | Multi-head attention, with one key and value head for each query head in this notation. |
| MLA | Multi-head latent attention, which caches a compressed latent and reconstructs full KVs when needed. |
| MQA | Multi-query attention, the $K=1$ endpoint where every query head shares one key and value head. |
| PagedAttention | Block-based KV-cache management that maps logical sequence blocks to non-contiguous physical memory. |
| Post-training quantization (PTQ) | Quantizing an already trained model, usually using calibration data. |
| Prefill | Parallel processing of the known prompt to initialize the KV cache and produce the first distribution. |
| Pruning | Removing model structures judged less important, followed by repair training or distillation. |
| Quantization-aware training (QAT) | Simulating quantization during training so weights adapt to its errors. |
| RoPE | Rotary positional embeddings, whose key/query transformation requires special handling in MLA. |
| Selective batching | Batching shared non-attention operations while handling ragged attention separately per sequence. |
| Speculative decoding | Drafting several tokens cheaply and verifying them in parallel with an exact target-model correction. |
| Target model | The authoritative distribution that speculative decoding must reproduce exactly. |
| Throughput | Aggregate tokens completed per second across requests. |
| Time to first token (TTFT) | Delay from request submission until the first generated token becomes available. |

# Self-check questions

1. Why does an agentic workload remove the human reading-speed ceiling that limits the value of faster chatbot generation?
2. Give a workload that prioritizes TTFT and one that prioritizes aggregate throughput.
3. Why can training parallelize across sequence positions while ordinary autoregressive generation cannot?
4. Derive the $O(T^3)$ total cost of naive prefix recomputation from an $O(t^2)$ full-prefix attention pass.
5. Which axes determine BF16 KV-cache memory, and why are there two separate factors of 2?
6. Why does a BF16 $(B,D)$ by $(D,F)$ product have intensity near $B$ when $B\ll D,F$?
7. What does the H100 balance point of about 295 FLOPs per byte mean?
8. Why can increasing concurrent requests make the generation MLP more compute-efficient?
9. Derive $I_{\mathrm{attn}}=ST/(S+T)$ and specialize it to prefill and decode.
10. Why does the attention intensity not improve with batch size even though MLP intensity does?
11. Explain how a larger batch can simultaneously worsen latency and improve throughput.
12. What inconsistency appears in the lecturer's batch-one latency and throughput numbers, and what value is consistent with the formula?
13. Compare how GQA, MLA, and CLA reduce the KV cache along different axes.
14. Why can a local-attention layer use sequence-length-independent cache while still having an effective receptive field larger than one window?
15. What information-access tradeoff separates sliding-window attention from a recurrent linear-attention state?
16. Why does a top-$k$ sparse-attention indexer need high recall even if the later attention computation is exact?
17. Contrast QAT, naive PTQ, GPTQ, and AWQ.
18. Why can keeping a few FP16 weight channels preserve quality but fail to give a hardware-efficient quantized kernel?
19. Why is structured pruning generally easier to accelerate than arbitrary unstructured sparsity?
20. Show how speculative decoding's accepted mass becomes $\min(p(x),q(x))$.
21. Why can proposing too many speculative tokens reduce rather than improve speed?
22. How does selective batching preserve weight reuse when request lengths differ?
23. Distinguish internal from external KV-memory fragmentation.
24. How do block tables and copy-on-write permit several responses to share one prompt cache safely?

# Source coverage checklist

| Raw transcript span | Material covered | Covered above |
|---:|---|:---:|
| 1-327 | Inference uses, repeated cost, token-volume perspective, agents, providers, open-source systems | Yes |
| 328-481 | TTFT, latency, throughput, interactive and batch use cases, metric tradeoffs | Yes |
| 482-609 | Training-versus-inference sequence parallelism and lecture roadmap | Yes |
| 610-975 | Tensor notation, Transformer block, MLP conventions, query/KV heads, $S$ versus $T$ | Yes |
| 976-1260 | Matrix-multiplication FLOPs, bytes, arithmetic intensity, H100 balance point, batch-one limit | Yes |
| 1261-1354 | Audience correction of $K$ and $G$ in GQA notation | Yes |
| 1355-1665 | Naive repeated-prefix decoding, cubic cost, causal reuse, KV cache, prefill and generation | Yes |
| 1666-1927 | MLP FLOPs and bytes, $BT$ intensity, prefill efficiency, generation concurrency | Yes |
| 1928-2256 | Attention intensity, prefill/decode specialization, why batching does not help KV reads, memory-bound summary | Yes |
| 2257-2637 | Bandwidth runtime model, Llama 2 13B configuration, parameter and KV memory, latency and throughput forms | Yes |
| 2638-2922 | Batch-size examples, latency-throughput tradeoff, H100 capacity, model replication, TTFT | Yes |
| 2923-3000 | Cross-cutting optimization landscape and the cache-size versus accuracy objective | Yes |
| 3001-3291 | MHA, MQA, GQA, cache reduction, batch-size interaction, empirical-quality caution | Yes |
| 3292-3548 | MLA compression, RoPE complication, DeepSeek comparisons, question about reducing model dimension | Yes |
| 3549-3627 | CLA and sharing KVs across layers, reported Pareto frontier | Yes |
| 3628-4015 | Sliding, dilated, and global-local attention; linear attention; long-context and hybrid-architecture Q&A | Yes |
| 4016-4159 | DeepSeek compressed and sparse attention, KV-reduction recap, diffusion direction | Yes |
| 4160-4365 | Quantization, QAT, PTQ, GPTQ, AWQ, important activation channels | Yes |
| 4366-4647 | Pruning pipeline, 15B-to-8B example, distillation recipe, importance-statistics Q&A | Yes |
| 4648-4989 | Speculative decoding motivation, draft-target algorithm, acceptance rule, exactness, lookahead and distillation | Yes |
| 4990-5175 | Dynamic traffic, Orca continuous batching, ragged requests, selective batching | Yes |
| 5176-5472 | PagedAttention, internal/external fragmentation, non-contiguous blocks, prefix sharing, copy-on-write | Yes |
| 5473-5587 | Final summary, common optimization principle, systems analogies, inference-native architectures, next lecture | Yes |
