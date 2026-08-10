---
title: "Lecture 2 - PyTorch, Einops, and Resource Accounting"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 2
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 2 PyTorch (einops).txt"
slide_deck: "../lecture_02.pdf"
status: "complete"
---

# Lecture 2: PyTorch, Einops, and Resource Accounting

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves the lecturer's claims, examples, qualifications, numerical details, and substantive audience questions while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, derivation, intuition, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are shown so the paraphrase can be audited. The complete transcript was mapped before the slides were inspected. All 33 slides were then rendered and visually checked to verify names, shapes, code, formulas, hardware specifications, and figures. Material differences between speech and slides are recorded under `Source reconciliation`.

## Lecture map

The lecture develops resource accounting from the bottom up:

1. Treat tensors as the storage representation for model data, parameters, activations, gradients, and optimizer state.
2. Account for tensor memory through shape and numerical precision.
3. Use Einops to express tensor operations with named dimensions rather than fragile positional transposes.
4. Count floating-point operations, benchmark realized throughput, and define model FLOPs utilization (MFU).
5. Relate computation to memory traffic through arithmetic intensity and roofline analysis.
6. Apply the accounting to forward passes, backpropagation, optimizers, gradient accumulation, and activation checkpointing.

---

# Part I - Why resource accounting starts with tensors

## 1. A scaling-law forecast that matched the completed run

**Transcript coverage:** lines 1-67

### What the lecturer said - transcript only

The lecturer begins with an update from the Marin project. Its run had finished and matched the forecast made from smaller experiments. The underlying plot contains **iso-FLOP curves**: for each compute budget, several smaller models are trained so that the compute-optimal point can be estimated. A scaling law is then fitted through those optima and extrapolated.

For the completed run, the lecturer says that the measured loss was within 0.05 of the prediction. He regards this as a compelling demonstration that a scaling procedure can forecast a larger run before the compute is spent. The plot also extrapolates the fitted trend toward a GPT-5-level compute regime, although he cautions that the result depends on the validity of the scaling laws.

### Source reconciliation

The spoken claim is that the loss was within 0.05 of the prediction. Slide 2 labels displayed forecast errors of `+0.011` and `+0.005` at several extrapolated points and identifies a Marin 32B-equivalent point, but it does not make clear which label corresponds to the exact run described in speech. The transcript's 0.05 is therefore preserved as the spoken value rather than silently replaced by a chart label.

### Additional explanation

An iso-FLOP curve compares configurations whose approximate training compute is held constant. One can vary model size and token count along that curve, locate the configuration with minimum loss, and repeat at several compute budgets. If the optima move smoothly, a fitted relationship can be used to select a configuration at a larger budget.

The important achievement is not merely a low loss. It is **prospective accuracy**: the prediction was made before the target run completed. This is stronger evidence than drawing a curve after seeing the result because it tests whether the recipe can genuinely manage the risk of an expensive run.

## 2. Resource accounting as napkin math

**Transcript coverage:** lines 70-358

### What the lecturer said - transcript only

Lecture 1 covered the course overview and tokenization. This lecture moves to the systems side through **resource accounting**. The recurring goal is to train the best model possible with finite resources. Those resources can include compute, memory, and data, although data is not expected to be the binding constraint for the class. Before computational efficiency can be improved, the compute and memory characteristics of the computation must be understood.

Two motivating calculations illustrate the desired level of reasoning.

First, consider training a 70-billion-parameter model on 15 trillion tokens using 1,024 H100 GPUs. The lecture uses the approximation

$$
C \approx 6PT,
$$

where $P$ is parameter count and $T$ is training-token count. It then combines the H100 specification with an assumed MFU of 0.5. The resulting estimate is about 143 days.

Second, consider the largest model whose training state could fit on eight 80-GB H100s using AdamW. The lecture assigns, per parameter:

- 2 bytes for the parameter;
- 2 bytes for its gradient;
- 4 bytes for one Adam moment;
- 4 bytes for the other Adam moment.

This gives 12 bytes per parameter. Across $8 \times 80$ GB, the rough upper bound is about 53 billion parameters. It omits activations, whose memory depends on batch size and sequence length, so it is not a feasible end-to-end model-size guarantee.

These calculations are intentionally approximate. The purpose is not to count every byte and operation perfectly, but to understand the rough shape and scale of a system before making design decisions.

The lecture's three learning layers are:

1. **Mechanics:** how PyTorch and tensors work.
2. **Mindset:** make resource accounting habitual; whenever code is written, ask about its performance characteristics.
3. **Intuition:** develop a sense for where memory and computation are spent.

There is no new machine-learning mechanism in this lecture; architecture is deferred to the following lecture.

### Additional explanation

The first calculation can be written explicitly as

$$
C = 6(70 \times 10^9)(15 \times 10^{12}) = 6.3 \times 10^{24}\ \text{FLOPs}.
$$

If one H100 provides about $989.5 \times 10^{12}$ dense BF16 FLOP/s and only half of that peak is realized, then 1,024 devices provide approximately

$$
1024 \times 0.5 \times 989.5 \times 10^{12}
\approx 5.07 \times 10^{17}\ \text{FLOP/s}.
$$

Dividing total work by realized throughput gives about $1.24 \times 10^7$ seconds, or 143 days.

The memory estimate is a **capacity bound**, not a training plan. A real implementation may also store FP32 master weights, temporary buffers, communication buckets, allocator overhead, and more than one activation per layer. Conversely, optimizer-state sharding can distribute some of these bytes so that the per-GPU requirement is lower.

## 3. Tensors are the universal storage building block

**Transcript coverage:** lines 361-433

### What the lecturer said - transcript only

The bottom of the accounting stack is the tensor. Parameters, gradients, optimizer states, data, and activations are all represented as tensors. The lecturer points to the DeepSeek V3.2 model as an example: the saved model is a collection of tensors, each with a shape and a precision.

Vectors and matrices are special cases of tensors. A tensor generalizes them to any number of dimensions.

### Additional explanation

A tensor's resource footprint is determined by more than its semantic role. For accounting, the essential metadata is:

- **shape**, which determines the number of elements;
- **dtype**, which determines bytes per element and supported arithmetic;
- **device**, which determines where the bytes reside;
- **layout and strides**, which determine how logical indices map to memory.

The slides use a rank-4 Transformer example with shape `(batch, sequence, heads, hidden_per_head)`. Naming those axes becomes important later: two tensors can have the same total number of elements while requiring different operations because their axes mean different things.

## 4. FP32 storage and the basic memory formula

**Transcript coverage:** lines 436-699

### What the lecturer said - transcript only

Tensors can store integers and other types, but deep-learning tensors commonly store floating-point values. The conventional baseline is **float32**, also called FP32 or single precision. Its 32 bits contain:

- 1 sign bit;
- 8 exponent bits, which primarily determine dynamic range;
- 23 fraction or mantissa bits, which primarily determine resolution.

The name "single precision" comes from traditional scientific computing, where FP32 was a normal baseline and FP64 was double precision. Deep learning has moved in the opposite direction because its computations often do not require the precision used in numerical simulation.

PyTorch creates floating tensors as FP32 by default unless another dtype is requested. Tensor storage is

$$
\text{memory} = \text{number of elements} \times \text{bytes per element}.
$$

A $4 \times 8$ FP32 matrix contains 32 values at 4 bytes each, so it occupies 128 bytes. At model scale, individual tensors become large: the lecture notes that one GPT-3 feed-forward matrix occupies about 2.3 GB.

Reducing precision lowers storage. It can also reduce execution time because narrower arithmetic can be faster and fewer bytes need to move, although a 2x reduction in bit width does not guarantee a 2x speedup.

### Additional explanation

For a tensor with shape $(d_1,\ldots,d_k)$ and element size $s$ bytes,

$$
M = s\prod_{i=1}^k d_i.
$$

The slide's GPT-3 example is a matrix of shape $(4 \times 12288, 12288)$. In FP32 it contains $603{,}979{,}776$ values and occupies $2{,}415{,}919{,}104$ bytes, approximately 2.25 GiB or 2.42 decimal GB. This illustrates why the unit convention matters when a rough figure is compared with an exact allocator report.

Precision affects three different limits:

1. **Capacity:** how many tensors fit in accelerator memory.
2. **Bandwidth:** how long it takes to transfer their bytes.
3. **Arithmetic throughput:** how many operations the accelerator supports per second at that dtype.

## 5. FP16, BF16, mixed precision, FP8, and FP4

**Transcript coverage:** lines 700-1197

### What the lecturer said - transcript only

The direct way to halve FP32 storage is FP16. FP16 has 1 sign bit, 5 exponent bits, and 10 fraction bits. Its small exponent field gives it poor dynamic range. In the lecture's PyTorch example, $10^{-8}$ becomes zero. Very small values underflow, very large values overflow, and FP16 training can consequently produce instability and NaNs.

**BFloat16 (BF16)** was developed in 2018 to address this problem. It uses the same 16 total bits as FP16 but reallocates bits from the fraction to the exponent: 1 sign bit, 8 exponent bits, and 7 fraction bits. It therefore has roughly FP32's dynamic range but less resolution. For many deep-learning workloads, avoiding underflow and overflow is more important than retaining many fraction bits because training is already stochastic and tolerant of approximate values.

The practical training summary is:

- FP32 is safe and simple for a small model, but consumes 4 bytes per value.
- FP16 halves storage but is risky for training because of its limited range.
- BF16 is a useful middle ground, though it is not completely risk-free.
- Mixed-precision training uses different dtypes for different parts of the computation.

As a general mixed-precision rule, the lecture uses BF16 for parameters, activations, and gradients, while optimizer state remains FP32. PyTorch's automatic mixed precision (AMP) can cast operations selectively. Matrix multiplication is generally suitable for BF16, while sensitive operations such as exponentiation can remain FP32.

Precision can be reduced further. FP8 was standardized around 2022 and has two variants because a format can allocate its few bits toward either greater dynamic range or greater resolution. NVIDIA's Transformer Engine supports FP8 workflows.

NVIDIA introduced NVFP4 in 2025. Four bits permit only 16 local values, so naive per-value FP4 would be too restrictive. Instead, values are grouped into blocks and each block receives a separate scale. Each value still has four bits of local freedom, but the shared scale moves the representable range of the whole block. This extends effective range without allowing neighboring values in one block to choose completely independent ranges. The lecturer cites Nemotron 3 Super as a model trained in FP4 and notes that much of this functionality is implemented inside NVIDIA's software stack rather than exposed as an ordinary tensor dtype.

### Additional explanation

The central distinction is **range versus precision**:

- More exponent bits let a format represent values across more orders of magnitude.
- More fraction bits place more representable values within a given magnitude range.

BF16 keeps the exponent width that protects FP32 from common overflow and underflow, then accepts coarser spacing between nearby numbers. That is often a better failure mode for deep learning than FP16's narrower range.

A block-scaled low-precision representation can be written schematically as

$$
x_i \approx s_b q_i, \qquad i \in \text{block } b,
$$

where $q_i$ is one of a small set of FP4 values and $s_b$ is shared by the block. A smaller block lets scales adapt more locally but requires more scale metadata and potentially more complicated kernels.

AMP does not mean that every tensor is permanently stored in whichever dtype an operation uses. Storage dtype, input casts, accumulator dtype, and output dtype can differ. Numerically sensitive reductions are often accumulated in a wider format even when their inputs are narrow.

## 6. Questions about block scaling and one-bit models

**Transcript coverage:** lines 1198-1354

### What the lecturer said - transcript only

An audience member asks whether every value in an FP4 block is scaled together. The lecturer confirms that values within the block select among their four-bit levels and then share an additional scale factor. Looking at one value in isolation therefore reveals more effective dynamic range than four bits alone would suggest. The restriction is local: one value cannot use a scale far above that of its immediate neighbor when both belong to the same block.

A second question asks whether precision could be pushed down to one bit. The lecturer distinguishes **training precision** from **inference quantization**. Many low-bit approaches train a model in a format such as BF16 and quantize the trained model to one or two bits for inference. That is much easier than training a credible language model using one-bit values throughout. He is not aware of a convincing one-bit-trained language model, while allowing that it might eventually be possible.

### Additional explanation

Post-training quantization starts from a high-precision solution and approximates it. Low-bit training must also represent small updates, gradient statistics, and optimizer dynamics while the solution is still being discovered. This makes training much more sensitive than inference.

Claims such as "a 1-bit model" should therefore be unpacked carefully. The phrase may refer only to stored weights; activations, scales, accumulators, embeddings, or selected layers may remain at higher precision. Effective memory per parameter also includes scale and metadata overhead.

## 7. CPU and GPU tensor placement

**Transcript coverage:** lines 1357-1437

### What the lecturer said - transcript only

PyTorch tensors are created in CPU memory by default. To use GPU acceleration, they must either be moved to the GPU or created there. Forgetting this device placement prevents the expected speedup.

The lecturer notes that the executable slides had been run on a laptop without a GPU, so some GPU code could only be shown rather than executed during the lecture.

### Additional explanation

Moving an existing tensor and creating it directly on the target device are different operations:

```python
x = x.to(device)

with torch.device(device):
    y = torch.zeros(32, 32)
```

The first form performs a transfer when the source and destination differ. The second avoids constructing an unnecessary CPU copy. Device mismatches also cause correctness errors: most PyTorch operations require all participating tensors to live on compatible devices.

The CPU and GPU have separate memory systems connected by an interconnect such as PCIe or NVLink. A transfer is therefore itself a resource-consuming operation, not merely a metadata change.

---

# Part II - Einops and named tensor dimensions

## 8. Einsum as generalized matrix multiplication

**Transcript coverage:** lines 1438-1809

### What the lecturer said - transcript only

Before counting tensor-operation FLOPs, the lecturer introduces Einops. Positional tensor code can be difficult to audit: an expression such as `transpose(-2, -1)` forces the reader to remember what the last two axes mean. Einops instead names dimensions and draws on Einstein summation notation. The lecturer describes `einsum` as generalized matrix multiplication with good bookkeeping.

For a matrix $X$ of shape `(seq1, hidden)` and a matrix $Y$ of shape `(hidden, seq2)`, ordinary multiplication can be written as:

```python
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

The `hidden` dimension appears in the inputs but not the output, so it is summed out. Conceptually, the operation enumerates valid values of `seq1`, `hidden`, and `seq2`, multiplies the selected values of $X$ and $Y$, and accumulates into the corresponding output entry.

The advantage becomes clearer for batched tensors. If two tensors have shapes `(batch, seq1, hidden)` and `(batch, seq2, hidden)`, a positional implementation transposes one tensor before batched matrix multiplication. Einops expresses the intended contraction directly:

```python
z = einsum(
    x,
    y,
    "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2",
)
```

There is no separate transpose to reason about because the axis meanings determine the contraction. An ellipsis can stand for any number of broadcast or batch dimensions:

```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

This is useful in language models, where operations may be batched across batch, sequence, and head axes. The ellipsis also permits modular code that does not need to enumerate every leading dimension.

### Additional explanation

For the first example, the scalar equation is

$$
Z_{ij} = \sum_h X_{ih}Y_{hj}.
$$

Einops makes three facts visible in one string:

1. which semantic axis each input position represents;
2. which axes are contracted;
3. the order of axes in the output.

Named patterns do not remove the need to verify shapes, but they move that verification closer to the mathematical statement. This becomes especially valuable in attention, where batch, query position, key position, head, and per-head channel axes otherwise invite silent transposition mistakes.

## 9. `reduce`: named sums, means, minima, and maxima

**Transcript coverage:** lines 1810-1900

### What the lecturer said - transcript only

Einops `reduce` generalizes operations such as sum, mean, maximum, and minimum. Instead of asking PyTorch to sum positional dimension `-1`, one can state that the `hidden` dimension disappears:

```python
y = reduce(x, "... hidden -> ...", "sum")
```

The reduction name can be changed to `mean`, `max`, or `min`. In response to a question about speed, the lecturer says that this notation lowers to the same kinds of primitive operations. Its benefit is clearer expression, not an inherent computational speedup.

### Additional explanation

A reduction changes both shape and numerical behavior. Summing $n$ values performs roughly $n-1$ additions per output and may accumulate rounding error. A mean adds a division, while max and min use comparisons. The named pattern clarifies which axis disappears, but dtype and accumulation strategy still determine stability.

Syntactic clarity can indirectly improve performance by making it easier to recognize opportunities for fusion or to notice that the wrong axis is being reduced. That is an engineering benefit, not a guarantee that `reduce` itself launches a faster kernel.

## 10. `rearrange`: splitting and combining logical dimensions

**Transcript coverage:** lines 1903-2163

### What the lecturer said - transcript only

`rearrange` is useful when one stored dimension actually represents several logical dimensions. A matrix may have been flattened for storage or multiplication and later need to be unflattened so that one component can be operated on.

The lecture starts with a tensor of shape `(3, 8)`. Its final dimension is understood as two heads, each with hidden size four. Parentheses describe the flattened product:

```python
x = rearrange(
    x,
    "... (heads hidden1) -> ... heads hidden1",
    heads=2,
)
```

Because 8 could be decomposed in more than one way, `heads=2` disambiguates the split and implies `hidden1=4`. A $4 \times 4$ weight matrix can then transform the per-head hidden dimension:

```python
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
```

Finally, the logical head and output-hidden dimensions can be combined again:

```python
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

An audience member asks whether flattening a two-dimensional object uses row-major or column-major order. The lecturer answers that the order of the names in the pattern specifies how dimensions are combined. He acknowledges that Einops takes some practice but argues that it makes transposes, reductions, and shape changes much more fluid once the notation becomes familiar.

### Additional explanation

This pattern mirrors multi-head model code. A model-width axis $D$ is commonly interpreted as $H \times d_h$, where $H$ is the number of heads and $d_h$ is width per head:

$$
D = H d_h.
$$

`rearrange` documents that semantic factorization at the point where it occurs. It can often be implemented as a view with no data copy when the requested order is compatible with the tensor's strides. If axes are reordered in a way that makes the layout non-contiguous, a later operation may require a copy. Shape notation and physical memory movement are related but not identical.

---

# Part III - FLOPs, benchmarking, and utilization

## 11. FLOPs versus FLOP/s and hardware-scale intuition

**Transcript coverage:** lines 2164-2442

### What the lecturer said - transcript only

The lecture measures computational work in floating-point operations. A FLOP is a basic operation such as an addition or multiplication. GPUs perform many other operations, but additions and multiplications account for most of the work considered here.

The terminology is easy to confuse:

- **FLOPs** denotes an amount of work, such as the total operations required to train a model.
- **FLOP/s** denotes a rate, such as a GPU's peak operations per second.

The lecturer prefers writing the explicit `/s` suffix so that a hardware speed is not confused with a computation total. Model-training budgets such as $10^{22}$, $10^{23}$, or $10^{25}$ FLOPs describe total scale.

An H100 specification advertises 1,979 teraFLOP/s for BF16, but the headline figure assumes structured sparsity. Dense computation receives roughly half that rate, so the lecture uses

$$
P_{\text{H100,dense BF16}} \approx \frac{1979}{2}\ \text{teraFLOP/s}.
$$

The lecturer uses eight H100s running for approximately a week to build intuition for hardware budgets. Multiplying devices, seconds, and dense peak FLOP/s gives roughly $5 \times 10^{21}$ FLOPs. The exercise is described as napkin math rather than a precise runtime prediction.

### Source reconciliation

The spoken explanation first says "two weeks," then notices that the displayed multiplication contains only one week and explicitly proceeds with one week. Slide 13 retains the heading "8 H100s for 2 weeks" while its formula also multiplies by only seven days. The approximately $5 \times 10^{21}$ result is the one-week dense-peak calculation; a true two-week total would be twice as large.

### Additional explanation

Three quantities should remain separate:

$$
\text{work} = \text{rate} \times \text{time},
$$

$$
\text{realized rate} = \text{peak rate} \times \text{utilization},
$$

$$
\text{wall time} = \frac{\text{work}}{\text{realized rate}}.
$$

Specification-sheet peak is an upper bound for a particular dtype, sparsity pattern, and operation family. It is not the throughput of an arbitrary model. Dense versus sparse peak, tensor-core versus scalar-core arithmetic, and dtype all need to match the workload before a comparison is meaningful.

## 12. Counting FLOPs for matrix and elementwise operations

**Transcript coverage:** lines 2443-2760

### What the lecturer said - transcript only

Much of model FLOP accounting reduces to matrix multiplication. Consider a batch of $B$ data points, each with input dimension $D$, multiplied by a weight matrix with $K$ output dimensions:

$$
X \in \mathbb{R}^{B \times D}, \qquad
W \in \mathbb{R}^{D \times K}, \qquad
Y = XW \in \mathbb{R}^{B \times K}.
$$

For every output position, the computation performs $D$ multiplications and approximately $D$ additions. The lecture therefore counts

$$
\operatorname{FLOPs}(XW) \approx 2BDK.
$$

The exact addition count is smaller by one per dot product, but this difference is ignored in large-scale estimates. An elementwise operation on an $N \times M$ matrix costs on the order of $NM$ FLOPs. At sufficiently large dimensions, matrix multiplication dominates FLOP count, with an important caveat that FLOPs alone do not capture memory cost.

Because $DK$ is the number of weight parameters in the linear layer, the forward-pass formula can also be written as

$$
\text{forward FLOPs} \approx 2 \times (\text{number of data points})
\times (\text{number of parameters}).
$$

This is the shape of the later $6 \times \text{data} \times \text{parameters}$ training estimate.

Two audience questions refine the accounting:

- Although asymptotically faster matrix-multiplication algorithms exist, practical accelerator optimization usually focuses on co-design with the hardware rather than those sub-cubic algorithms.
- Addition may seem simpler than multiplication, but the relevant accelerator hardware supports them at roughly comparable rates for this accounting, so each is counted as one FLOP.

### Additional explanation

The exact count for conventional matrix multiplication is

$$
BK(2D-1),
$$

because each of the $BK$ outputs needs $D$ multiplications and $D-1$ additions. For large $D$, $2D-1 \approx 2D$.

Modern tensor cores commonly implement fused multiply-add operations. One hardware instruction may perform both operations, but conventional machine-learning accounting credits it as two FLOPs. FLOP counts are therefore a workload convention, not an instruction count.

Elementwise operations are cheap in FLOPs but can still be expensive in wall time. They often move an entire tensor to perform only one operation per value. Arithmetic intensity, introduced later, explains this apparent contradiction.

## 13. Benchmarking asynchronous GPU operations correctly

**Transcript coverage:** lines 2761-2937

### What the lecturer said - transcript only

A logical FLOP count is independent of hardware. To learn how long the operation actually takes, it must be benchmarked.

GPU work is asynchronous with respect to the CPU. A timing block must therefore synchronize before the operation and again after it. Without the final synchronization, the measured call can appear implausibly fast because the CPU only queued the GPU work and returned. The operation should also be repeated and averaged rather than measured once.

The realized rate is

$$
\text{actual FLOP/s}
= \frac{\text{logical FLOPs performed}}{\text{elapsed seconds}}.
$$

The lecture's code could not produce representative GPU results on the lecturer's laptop, but the synchronization procedure remains the important preview of later benchmarking material.

### Additional explanation

A minimal benchmark needs more than two timestamps:

1. warm up the operation so compilation, allocation, and cache effects do not dominate;
2. synchronize;
3. run enough iterations for stable timing;
4. synchronize again;
5. report a distribution or robust statistic, not only one measurement.

PyTorch CUDA events can time device work with less CPU-timer noise. A benchmark should also fix shapes, dtype, layout, and device because all of them can select different kernels.

## 14. Model FLOPs utilization (MFU)

**Transcript coverage:** lines 2938-3210

### What the lecturer said - transcript only

Actual throughput is normally below the rate promised by the specification sheet. The lecture defines **model FLOPs utilization** as

$$
\operatorname{MFU}
= \frac{\text{actual FLOP/s}}{\text{promised FLOP/s}},
$$

while setting aside communication and other overhead in the simplified definition.

The promised rate must be the relevant dense rate, not the H100's sparsity-enhanced headline. For BF16, the lecture first divides 1,979 teraFLOP/s by two. Practical model throughput is then some fraction of that dense number.

An MFU of about 0.5 is considered good for a modern model. A nearly pure matrix multiplication might reach roughly 0.8. An MFU near 0.1 suggests that something deserves investigation. MFU can be computed by counting the model's logical FLOPs, measuring wall-clock time, and comparing the resulting rate with the dtype-appropriate hardware peak.

Peak FLOP/s depends on both hardware and dtype. Current accelerators are optimized much more heavily for BF16 or FP8 than for FP32, so a model that uses FP32 can be far slower. The explanation for MFU being well below one is deferred until memory bottlenecks are introduced.

### Additional explanation

MFU is useful only when the numerator's FLOP convention is explicit. For example, if a model's attention computation is omitted from the logical count but still consumes time, its MFU will look lower. Comparing two reports requires consistent accounting for recomputation, sparsity, expert routing, and attention.

MFU is also not the same as "GPU busy." A GPU can be continuously occupied by memory transfers or low-throughput kernels while realizing few of its peak FLOPs. Conversely, a low MFU can be expected for an inherently memory-bound workload rather than evidence of a faulty implementation.

---

# Part IV - Arithmetic intensity and roofline analysis

## 15. Computation requires moving tensors to the arithmetic units

**Transcript coverage:** lines 3211-3658

### What the lecturer said - transcript only

To understand why MFU is not normally one, the lecturer introduces a simplified accelerator model. Tensors reside in high-bandwidth memory (HBM), while arithmetic is performed by separate compute units. An operation must:

1. move inputs from memory to the compute units;
2. perform the arithmetic;
3. move outputs back to memory.

Runtime therefore depends on both compute throughput and memory bandwidth. For the H100 example, the lecture uses about $1979 \times 10^{12}/2$ dense BF16 FLOP/s and about $3.3 \times 10^{12}$ bytes/s of HBM bandwidth. Memory size determines whether tensors fit, while memory bandwidth determines how quickly their bytes can be supplied.

Consider applying ReLU elementwise to a BF16 vector of length $N \approx 10^6$. Reading the input moves $2N$ bytes and writing the output moves another $2N$ bytes, for $4N$ bytes total. The operation performs roughly $N$ comparisons. The lecture estimates:

$$
t_{\text{memory}} = \frac{4N}{\text{memory bandwidth}}
\approx 10^{-6}\ \text{s},
$$

$$
t_{\text{compute}} = \frac{N}{\text{compute throughput}}
\approx 10^{-9}\ \text{s}.
$$

The simplified model assumes memory movement and arithmetic overlap perfectly, giving

$$
t_{\text{total}} = \max(t_{\text{memory}}, t_{\text{compute}}).
$$

Actual overlap is imperfect and has overhead, but the maximum is a useful first approximation. An operation is **memory-bound** when data movement takes longer and **compute-bound** when arithmetic takes longer. ReLU is strongly memory-bound in this example.

### Additional explanation

The overlap assumption describes a streaming implementation: while one tile is being computed, another tile can be loaded and a previous tile can be written. Without overlap, the simple estimate would be the sum of the two times. Good kernels try to approach the maximum by pipelining these stages.

The bytes in this accounting refer to transfers across the bottleneck being modeled. A real GPU has a memory hierarchy that includes registers, shared memory, caches, and HBM. Reuse in a faster level can reduce HBM traffic. This is why tiling and fusion can change arithmetic intensity even when the mathematical operation is unchanged.

## 16. Accelerator intensity and workload arithmetic intensity

**Transcript coverage:** lines 3661-3839

### What the lecturer said - transcript only

The lecturer expresses the same bottleneck test through intensity. **Accelerator intensity** is the peak work the hardware can perform per byte it can deliver:

$$
I_{\text{accelerator}}
= \frac{\text{peak FLOP/s}}{\text{memory bytes/s}}.
$$

For the H100 numbers used in the lecture, this is approximately 295 FLOPs per byte, or roughly 300 as a number to remember.

The **arithmetic intensity** of a workload is its actual work per byte moved:

$$
I_{\text{algorithm}}
= \frac{\text{FLOPs}}{\text{bytes moved}}.
$$

For BF16 ReLU, the lecture initially suggests one-half and then corrects the result to

$$
I_{\text{ReLU}} \approx \frac{N}{4N} = 0.25\ \text{FLOP/byte}.
$$

The bottleneck test becomes:

- memory-bound if $I_{\text{algorithm}} < I_{\text{accelerator}}$;
- compute-bound if $I_{\text{algorithm}} > I_{\text{accelerator}}$.

ReLU's 0.25 is far below 295, so it is deeply memory-bound. The lecturer emphasizes that a reported intensity of 0.25 should immediately look poor and that increasing useful work per byte moved is desirable.

### Additional explanation

The threshold follows directly from comparing times:

$$
\frac{Q}{B_w} \gtrless \frac{F}{P},
$$

where $Q$ is bytes moved, $B_w$ is bandwidth, $F$ is FLOPs, and $P$ is peak FLOP/s. Rearranging yields

$$
\frac{F}{Q} \gtrless \frac{P}{B_w}.
$$

The left side is workload intensity and the right side is accelerator intensity. The ratio is not a universal property of an abstract operation: it depends on dtype, tensor shape, memory reuse, and implementation.

## 17. Arithmetic intensity of common operations

**Transcript coverage:** lines 3840-4376

### What the lecturer said - transcript only

The lecture applies the intensity calculation to increasingly reusable computations.

#### 17.1 GELU

A BF16 GELU reads and writes the same $4N$ bytes as ReLU but performs roughly 20 FLOPs per value. Its estimated intensity is therefore

$$
I_{\text{GELU}} \approx \frac{20N}{4N} = 5\ \text{FLOP/byte}.
$$

Five is higher than ReLU's 0.25 but still far below the H100 threshold of about 295. GELU remains memory-bound. When executed as isolated elementwise kernels, the more complicated GELU can therefore take about the same time as ReLU because both wait on the same tensor traffic.

#### 17.2 Dot product

For BF16 vectors $x,w \in \mathbb{R}^N$, the dot product reads $2N$ bytes from each input and writes a 2-byte scalar. It performs $N$ multiplications and $N-1$ additions:

$$
Q \approx 4N + 2, \qquad F = 2N-1,
$$

so its intensity approaches one-half FLOP per byte. It is memory-bound.

#### 17.3 Matrix-vector product

For a BF16 $N \times N$ matrix multiplied by a length-$N$ vector:

$$
Q = 2N + 2N^2 + 2N,
$$

$$
F = N(2N-1).
$$

The matrix dominates the traffic, so the intensity is only about one FLOP per byte. Reusing the vector is not enough to offset reading the entire matrix; the operation remains memory-bound.

#### 17.4 Matrix-matrix multiplication

For two BF16 $N \times N$ matrices producing another $N \times N$ matrix:

$$
Q = 2N^2 + 2N^2 + 2N^2 = 6N^2,
$$

$$
F = N^2(2N-1) \approx 2N^3.
$$

Thus,

$$
I_{\text{matmul}} \approx \frac{N}{3}.
$$

At $N=1024$, this is roughly 340 FLOPs per byte, above the H100 threshold used in the lecture. Large matrix multiplication is consequently compute-bound. Its data grows quadratically while its arithmetic grows cubically, so intensity improves with matrix size.

This explains why large batches and large matrices help saturate accelerators. Below the intensity threshold, reducing matrix dimensions may not provide the proportional speedup expected from the FLOP count because memory traffic still dominates. Once the threshold is crossed, arithmetic units become the limiting resource.

Transformers are largely big matrix multiplications with other operations between them, and the lecturer describes their high arithmetic intensity as a deliberate strength. Training processes many tokens together and exposes matrix-matrix products. Autoregressive inference generates one token at a time and resembles matrix-vector multiplication, which helps explain why decoding is memory-bound. Intensity also changes with numerical precision.

Low arithmetic intensity can reduce MFU: the GPU cannot approach its promised FLOP/s while compute units wait for bytes.

### Additional explanation

The simple matmul traffic estimate assumes each matrix is read once from HBM and the result written once. A naive implementation that repeatedly reloads values would move much more data. Tiled kernels approach the favorable estimate by keeping tiles in registers or shared memory and reusing them for many multiply-add operations.

Batching converts many independent matrix-vector products into a matrix-matrix product. This is one reason serving systems batch requests: it increases weight reuse and arithmetic intensity. The tradeoff is latency, since a request may wait for a sufficiently useful batch.

The statement that isolated ReLU and GELU have the same time is a roofline approximation, not a universal guarantee. Launch overhead, vectorized special-function units, compiler fusion, and tensor size can still distinguish them.

## 18. Reading a roofline plot

**Transcript coverage:** lines 4377-4568

### What the lecturer said - transcript only

A roofline plot visualizes the relationship between arithmetic intensity and realized FLOP/s. The horizontal axis is arithmetic intensity, normally on a log scale, and the vertical axis is achieved FLOP/s, also normally logarithmic. Each vertical slice represents a workload at a particular intensity. Each piecewise line represents an accelerator.

At low intensity, performance rises with arithmetic intensity because memory bandwidth is the limit. At the **kink**, workload intensity reaches accelerator intensity. Beyond that point, the curve becomes horizontal at peak FLOP/s because the workload is compute-bound and cannot exceed the arithmetic hardware's peak.

An audience member observes that accelerators appear oversized relative to memory bandwidth because compute units can idle while waiting for memory. The lecturer says hardware design tradeoffs will become clearer in the GPU lecture and jokes that anyone with a better design should tell NVIDIA's Jensen Huang.

### Additional explanation

The idealized roofline is

$$
\text{attainable FLOP/s}
= \min\left(P_{\text{peak}},\ B_w I_{\text{algorithm}}\right).
$$

Dividing by peak compute gives an ideal utilization ceiling:

$$
\operatorname{MFU}_{\text{roofline}}
\leq \min\left(1,\frac{I_{\text{algorithm}}}{I_{\text{accelerator}}}\right).
$$

The roofline separates two optimization strategies:

- On the sloped, bandwidth-bound side, reduce bytes or increase reuse through fusion, tiling, caching, and batching.
- On the flat, compute-bound side, reduce FLOPs or use faster arithmetic and more efficient compute kernels.

A real roofline can include several ceilings for cache levels, instruction mixtures, occupancy, and communication. The lecture uses the single HBM-versus-compute roof because it is the clearest first model.

---

# Part V - Resource accounting for training

## 19. A simple deep network for accounting

**Transcript coverage:** lines 4569-4725

### What the lecturer said - transcript only

The lecture now applies the tensor accounting to training. Its running example takes an input batch of shape $(B,D)$ and passes it through $L$ layers. Every layer has a $D \times D$ weight matrix, followed by an elementwise activation such as ReLU. Each layer maps a $(B,D)$ activation to another $(B,D)$ activation, so the final output has the same shape as the input.

In PyTorch, the network is represented as a sequence of blocks. Each block owns a weight matrix and applies the linear transformation followed by the pointwise nonlinearity. The number of trainable parameters is

$$
P = LD^2.
$$

The forward pass applies these blocks sequentially. The model is intentionally simple so that memory and FLOP formulas can be derived without architectural distractions.

### Additional explanation

The model isolates the dominant dense-layer pattern while suppressing biases, normalization, residual connections, embeddings, and attention. This makes it a useful accounting model even though it is not a full language model.

If activations have BF16 dtype, one stored $(B,D)$ activation occupies $2BD$ bytes. If every layer boundary is retained for backpropagation, the leading activation-storage term is proportional to $2BDL$ bytes. Real frameworks may save additional intermediates required by the backward formulas.

## 20. PyTorch autograd mechanics

**Transcript coverage:** lines 4726-4812

### What the lecturer said - transcript only

The lecturer reviews gradients with an even simpler linear-regression example. The input is $x=(1,2,3)$, the weight vector is $w=(1,1,1)$, and the model forms their dot product and a squared-error loss. After the backward pass, PyTorch populates `w.grad` with $(1,2,3)$.

The example illustrates the basic autograd mechanism: tensors that participate in the computation graph can request gradients, and calling `backward()` fills the gradient fields needed for optimization.

### Additional explanation

The slide specifies

$$
\hat y = x^\top w = 6,
\qquad
L = \frac{1}{2}(\hat y-5)^2.
$$

Since $\hat y-5=1$,

$$
\frac{\partial L}{\partial w}
= (\hat y-5)x
= x
= (1,2,3).
$$

Autograd records operations during the forward pass, then applies local derivative rules in reverse order. The stored graph and saved intermediates are part of training's memory cost.

## 21. Why backpropagation leads to the $6PT$ rule

**Transcript coverage:** lines 4813-5325

### What the lecturer said - transcript only

To count gradient FLOPs, the lecture temporarily removes the nonlinearities and considers two $D \times D$ linear layers:

$$
H_1 = XW_1, \qquad H_2 = H_1W_2,
$$

where $X$ has shape `(batch, in)` and each weight has shape `(in, out)`. Intermediate gradients are retained for debugging, the loss is reduced to a scalar, and `backward()` computes the gradients.

Focus on the second layer, $H_2=H_1W_2$. Its forward pass is one matrix multiplication:

$$
F_{\text{forward,layer}} \approx 2BD^2.
$$

The backward pass must compute two quantities:

1. the gradient with respect to the layer input;
2. the gradient with respect to the layer parameters.

Using named dimensions, the first contraction is:

```python
h1_grad = einsum(
    h2.grad,
    w2,
    "batch out, in out -> batch in",
)
```

This is the matrix form of

$$
\frac{\partial L}{\partial H_1}
= \frac{\partial L}{\partial H_2}W_2^\top.
$$

The parameter gradient is:

```python
w2_grad = einsum(
    h2.grad,
    h1,
    "batch out, batch in -> in out",
)
```

or

$$
\frac{\partial L}{\partial W_2}
= H_1^\top\frac{\partial L}{\partial H_2}.
$$

Each contraction has the same leading FLOP count as the forward matmul. The backward pass therefore costs about twice the forward pass for the layer:

$$
F_{\text{backward,layer}} \approx 4BD^2.
$$

Applying this to all parameters gives:

$$
F_{\text{forward}} \approx 2BP,
$$

$$
F_{\text{backward}} \approx 4BP,
$$

$$
F_{\text{training step}} \approx 6BP.
$$

Across $T$ training data points or tokens, this becomes the familiar

$$
C \approx 6PT.
$$

The estimate also works reasonably well for Transformers when context length is not too large. At long context, attention contributes a sequence-length-squared term that this parameter-only approximation does not capture.

### Additional explanation

Einops helps distinguish the two backward contractions. Both use the incoming error signal, but one contracts over `out` to produce an input gradient, while the other contracts over `batch` to produce a parameter gradient. Their shapes identify which indices must remain.

The factor six is not a universal physical constant. It assumes dense parameter matmuls, conventional reverse-mode differentiation, one forward pass, and no activation recomputation. It omits or approximates embeddings, normalization, nonlinearities, attention score products, loss computation, optimizer updates, and communication. It remains valuable because large dense projections often dominate training FLOPs.

With activation checkpointing, some forward work is repeated during backward, so actual executed FLOPs can exceed the logical $6PT$ estimate.

## 22. AdaGrad and optimizer state

**Transcript coverage:** lines 5326-5532

### What the lecturer said - transcript only

After gradients have been computed, training needs an optimizer. The lecture uses AdaGrad, introduced in 2011, rather than exposing the exact Adam implementation required in Assignment 1. AdaGrad is described as lying between plain SGD and Adam: it augments SGD with accumulated squared gradients. Momentum tracks a first-order gradient average, while Adam combines first- and second-order information.

A PyTorch optimizer iterates through parameter groups. For each parameter, it retrieves optimizer state, updates that state from the current gradient, and then updates the parameter. AdaGrad's per-parameter state is the cumulative sum of squared gradients:

$$
G_t = G_{t-1} + g_t^2.
$$

The parameter update divides by the square root of this accumulated quantity:

$$
\theta_t
= \theta_{t-1}
- \eta\frac{g_t}{\sqrt{G_t}+\epsilon}.
$$

The key systems point is that the optimizer needs persistent tensors in addition to parameters and gradients.

### Additional explanation

AdaGrad scales frequently updated coordinates down more aggressively because their cumulative squared-gradient state grows. Unlike RMSProp and Adam, classical AdaGrad does not exponentially forget old squared gradients, so its effective learning rates can continually shrink.

The state dictionary is keyed by parameter identity. This is why optimizer checkpoints can be large and why restoring an optimizer is distinct from loading model weights alone. The optimizer's parameter groups can also carry different learning rates, weight decay values, or other hyperparameters.

## 23. Memory used by parameters, activations, gradients, and optimizer state

**Transcript coverage:** lines 5533-5793

### What the lecturer said - transcript only

For the $L$-layer, width-$D$ network, parameter count is $P=LD^2$. Under the lecture's mixed-precision assumptions:

| Training object | Leading memory |
|---|---:|
| BF16 parameters | $2P$ bytes |
| BF16 parameter gradients | $2P$ bytes |
| BF16 layer activations | $2BDL$ bytes |
| FP32 AdaGrad state | $4P$ bytes |
| FP32 Adam first and second moments | $8P$ bytes |

The lecture corrects a displayed variable while explaining the formulas: optimizer state should be measured from the number of parameters, not from parameter-memory bytes. FP32 state is preferred because squared gradients and statistics accumulated across many steps can be unstable in BF16.

Optimizer state is therefore a major part of training-memory capacity. AdaGrad stores one FP32 statistic per parameter, while Adam stores two. In the lecturer's simplified performance view, optimizer state is not the dominant source of matrix-multiplication compute, but its capacity cost can prevent a model from fitting in HBM.

The same accounting becomes more involved for a Transformer, and Assignment 1 asks students to carry it out more carefully.

### Additional explanation

Ignoring activations, the lecture's Adam-style state consumes

$$
2P + 2P + 8P = 12P\ \text{bytes},
$$

which is the 12-byte assumption used in the opening eight-H100 estimate.

Some mixed-precision implementations also maintain an FP32 master copy of the parameters. That adds another $4P$ bytes and would raise the simplified persistent total to 16 bytes per parameter. Exact accounting must inspect the optimizer and precision implementation rather than assume every stack has the same state.

Optimizer kernels themselves are often bandwidth-bound because they read and write several state tensors while performing relatively little arithmetic. The lecture's statement about optimizer state not dominating compute should be read as a comparison with the model's large matmuls, not as a claim that optimizer updates have no runtime cost.

## 24. Gradient accumulation trades activation memory for more micro-steps

**Transcript coverage:** lines 5794-5892

### What the lecturer said - transcript only

Larger batches can improve training stability up to a critical batch size, but activation memory grows with batch size and can exceed device capacity. Gradient accumulation simulates a larger batch using smaller **microbatches**:

1. compute gradients on one microbatch;
2. retain and accumulate those gradients instead of zeroing them;
3. repeat for `batch_size / microbatch_size` microbatches;
4. update the parameters once and then clear the gradients.

The lecturer presents this as a small code change that makes larger effective batches possible under the activation-memory limit.

### Source reconciliation

The spoken sentence at lines 5884-5890 says the technique "allows you to save on compute." Slide 29 and the surrounding argument present it as a way to reduce **peak activation memory**. Standard gradient accumulation does not reduce the leading FLOPs required for a fixed effective batch; it splits those FLOPs across micro-steps and can introduce extra overhead. The note therefore preserves the spoken procedure while treating memory, not total compute, as the intended saved resource.

### Additional explanation

If $K$ microbatches each contain $b$ examples, the effective batch is

$$
B_{\text{effective}} = Kb.
$$

Peak activation memory scales with $b$ rather than $Kb$, while the persistent gradient tensor remains the same size. To match a single large-batch mean loss, each microbatch loss must be scaled consistently, commonly by dividing by $K$ before `backward()`.

Microbatches that are too small can lower arithmetic intensity and throughput. Gradient accumulation therefore solves a capacity problem but may worsen execution efficiency; it does not create memory for free.

## 25. Activation checkpointing and rematerialization

**Transcript coverage:** lines 5893-6150

### What the lecturer said - transcript only

Training normally retains activations from all layers because backpropagation needs them. In the simplified network this gives activation memory proportional to $2BDL$ bytes. In ordinary inference there is no backward pass, so the lecturer contrasts this with needing only the current layer's activations.

**Activation checkpointing**, also called gradient checkpointing or rematerialization, exchanges additional computation for lower memory:

- During the forward pass, retain activations only at selected checkpoints.
- During the backward pass, recompute missing activations from the most recent checkpoint.

PyTorch can apply this behavior around a block with `torch.utils.checkpoint.checkpoint`. In the lecture's linear-plus-ReLU block, checkpointing stores the block boundary but not every intermediate, such as the pre-ReLU value. This can cut the illustrated activation storage roughly in half; the missing value is recomputed when backward needs it.

The tradeoff can be pushed farther. Storing every layer activation uses $O(L)$ activation memory and no rematerialization. Storing no intermediate layer activations uses $O(1)$ activation memory but can require $O(L^2)$ work if every needed value is recomputed from the beginning. The lecturer describes a balanced placement with about $\sqrt{L}$ checkpoints and segments of about $\sqrt{L}$ layers, giving $O(\sqrt{L})$ stored boundaries and an $O(\sqrt{L})$ maximum recomputation span.

### Source reconciliation

The transcript calls the balanced regime's "recomputation overhead" $\sqrt{L}$. Slide 31 states that checkpointing every $\sqrt{L}$ layers uses $O(\sqrt{L})$ activation memory and $O(L)$ total recomputation. These descriptions can be reconciled if the spoken $\sqrt{L}$ refers to the maximum distance recomputed within one segment, while the slide's $O(L)$ refers to total recomputation across all segments. If "overhead" is interpreted as total extra layer evaluations, the slide provides the clearer asymptotic statement.

### Additional explanation

Checkpointing changes the training-step FLOP count. If each omitted forward operation is recomputed once, the extra work is on the order of another forward pass, so total work remains $O(L)$ but grows by a constant factor. More extreme schedules can reduce memory further while repeating layers many times.

The inference comparison is simplified. Autoregressive Transformer inference does not need training activations, but it commonly retains a key-value cache whose memory grows with sequence length and layer count.

Checkpointed regions must also be safe to replay. Randomness and side effects need careful handling so that recomputation is consistent with the original forward pass.

## 26. Final synthesis and transition to architecture

**Transcript coverage:** lines 6151-6244

### What the lecturer said - transcript only

The lecture closes with five main messages:

1. Parameters, gradients, activations, optimizer state, and data are all tensors.
2. Einops provides a clearer way to reason about tensor operations through named dimensions.
3. The $6 \times \text{data points} \times \text{parameters}$ rule comes from one forward matmul and two corresponding backward matmuls.
4. Arithmetic intensity and roofline analysis diagnose whether computation is limited by memory bandwidth or arithmetic throughput. Large matrix multiplications are generally compute-bound, while isolated elementwise operations are generally memory-bound.
5. Gradient accumulation and activation checkpointing reduce peak memory requirements and thereby permit larger effective batches or models.

The following lecture, taught by Tatsu Hashimoto, turns from resource accounting to model architectures.

### Additional explanation

The lecture establishes a repeatable accounting workflow:

1. write down tensor shapes and dtypes;
2. count persistent and temporary bytes;
3. count leading FLOPs;
4. estimate bytes moved across the relevant memory boundary;
5. compare workload intensity with hardware intensity;
6. benchmark and compare realized throughput with the estimate.

Later architecture and systems choices can then be evaluated quantitatively rather than described only as "faster" or "smaller."

---

# Consolidated takeaways

1. Resource accounting is approximate by design, but a correct order-of-magnitude estimate can rule out impossible training plans before any code runs.
2. Tensor memory is the product of element count and bytes per element; shape, dtype, device, and layout are the essential accounting metadata.
3. BF16 preserves FP32-like range at half the storage by sacrificing resolution, which usually makes it a better training format than FP16.
4. Very low precision relies on mixed precision, wider accumulators, and often block-level scales; a nominal bit width does not describe the whole numerical system.
5. Einops replaces positional axis manipulation with named contractions, reductions, and rearrangements that more closely match the mathematics.
6. Dense matrix multiplication of $(B,D)$ by $(D,K)$ costs approximately $2BDK$ FLOPs.
7. A benchmark of CUDA work must synchronize because kernel launches are asynchronous.
8. MFU compares realized model FLOP/s with the relevant hardware peak, but its meaning depends on a consistent FLOP-counting convention.
9. Arithmetic intensity compares useful FLOPs with bytes moved. An operation is memory-bound below the hardware's FLOP-per-byte ratio and compute-bound above it.
10. Elementwise activations, dot products, and matrix-vector products have low intensity; sufficiently large matrix-matrix products achieve reuse and become compute-bound.
11. Backpropagating through a dense linear layer uses two leading matmuls for every forward matmul, yielding the approximate $6PT$ training rule.
12. Adam's FP32 moments can dominate persistent training memory even though model matmuls dominate arithmetic work.
13. Gradient accumulation reduces peak activation memory for a target effective batch but does not reduce its leading compute.
14. Activation checkpointing reduces saved intermediates by recomputing them during backward, explicitly trading FLOPs for memory.

# Key equations

## Tensor memory

$$
M = s\prod_{i=1}^{k} d_i,
$$

where $s$ is bytes per element and $(d_1,\ldots,d_k)$ is the shape.

## Matrix-multiplication FLOPs

$$
X \in \mathbb{R}^{B \times D},\quad
W \in \mathbb{R}^{D \times K}
\quad\Longrightarrow\quad
F \approx 2BDK.
$$

## Realized throughput and MFU

$$
\text{actual FLOP/s} = \frac{\text{logical FLOPs}}{\text{elapsed seconds}},
$$

$$
\operatorname{MFU}
= \frac{\text{actual FLOP/s}}{\text{promised FLOP/s}}.
$$

## Memory and compute time

$$
t_{\text{memory}} = \frac{Q}{B_w},
\qquad
t_{\text{compute}} = \frac{F}{P},
$$

$$
t_{\text{total}} \approx \max(t_{\text{memory}},t_{\text{compute}})
$$

under the perfect-overlap approximation.

## Arithmetic intensity

$$
I_{\text{algorithm}} = \frac{F}{Q},
\qquad
I_{\text{accelerator}} = \frac{P}{B_w}.
$$

The operation is memory-bound when $I_{\text{algorithm}} < I_{\text{accelerator}}$ and compute-bound when the inequality reverses.

## Roofline ceiling

$$
\text{attainable FLOP/s}
= \min(P, B_w I_{\text{algorithm}}).
$$

## Dense-training approximation

For $P$ parameters and $T$ training tokens or examples:

$$
C \approx 2PT + 4PT = 6PT.
$$

## Persistent mixed-precision state in the lecture's Adam example

$$
M_{\text{parameter+gradient+moments}}
= 2P + 2P + 8P
= 12P\ \text{bytes}.
$$

## Effective batch under gradient accumulation

$$
B_{\text{effective}}
= (\text{microbatches per step})(\text{microbatch size}).
$$

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Activation | An intermediate tensor produced by a layer and often retained for backpropagation. |
| Activation checkpointing | Saving selected activations and recomputing missing ones during backward to trade compute for memory. |
| Arithmetic intensity | Floating-point work performed per byte moved by a workload. |
| Autograd | PyTorch's system for recording a computation graph and automatically applying reverse-mode differentiation. |
| BF16 | A 16-bit floating format with 8 exponent bits and 7 fraction bits, giving FP32-like range with lower resolution. |
| Compute-bound | A regime in which arithmetic throughput, rather than memory traffic, limits runtime. |
| Dtype | The numerical representation used for a tensor's elements. |
| Einops | A tensor-manipulation library whose patterns name dimensions for contractions, reductions, and rearrangements. |
| FLOP | One floating-point operation, conventionally an addition or multiplication in this lecture. |
| FLOP/s | A rate of floating-point operations per second. |
| FP16 | IEEE half precision, with 5 exponent bits and 10 fraction bits; its limited range can destabilize training. |
| FP32 | IEEE single precision, the 32-bit floating baseline used in the lecture. |
| Gradient accumulation | Summing gradients across microbatches before one optimizer update to realize a larger effective batch. |
| HBM | High-bandwidth memory attached to an accelerator. |
| MFU | Model FLOPs utilization, the ratio of realized model FLOP/s to an appropriate hardware peak. |
| Memory-bound | A regime in which moving bytes takes longer than performing the associated arithmetic. |
| Mixed precision | Using different numerical formats for different tensors or operations in one training run. |
| Optimizer state | Persistent tensors, such as Adam moments or AdaGrad squared-gradient sums, used to compute parameter updates. |
| Quantization | Representing trained values with fewer bits, often for cheaper inference. |
| Rank | The number of dimensions in a tensor. |
| Rematerialization | Another name for recomputing activations instead of retaining all of them. |
| Roofline model | A model that bounds attainable throughput by the smaller of compute peak and bandwidth times arithmetic intensity. |
| Tensor | A multidimensional array carrying data, parameters, activations, gradients, or optimizer state. |

# Self-check questions

1. Why is the eight-H100, 53-billion-parameter result only an upper bound rather than a feasible model size?
2. Which FP32 fields primarily control dynamic range and local numerical resolution?
3. Why is BF16 generally safer for training than FP16 even though both use 16 bits?
4. What does a block scale add to an FP4 representation, and what local restriction remains?
5. How does an Einops output pattern reveal which dimensions are summed out?
6. Why can `rearrange` be clearer than a sequence of `view`, `transpose`, and `reshape` calls?
7. Derive the approximate $2BDK$ FLOP count for a $(B,D)$ by $(D,K)$ matrix multiplication.
8. Why must a CUDA benchmark synchronize after launching the operation?
9. What does an MFU of 0.1 tell you, and what does it not tell you?
10. Derive the memory-bound versus compute-bound test from memory time and compute time.
11. Why can isolated GELU take roughly as long as ReLU despite performing many more arithmetic operations?
12. Why is a matrix-vector product memory-bound while a sufficiently large matrix-matrix product can be compute-bound?
13. What are the two matrix multiplications required when backpropagating through a dense linear layer?
14. Under what assumptions does the $6PT$ training approximation hold, and what Transformer cost can violate it at long context?
15. Why are Adam moments commonly stored in FP32 when parameters and gradients use BF16?
16. Does gradient accumulation reduce total FLOPs for a fixed effective batch? What resource does it primarily reduce?
17. Compare the memory and recomputation behavior of storing every activation, no intermediate activations, and checkpoints spaced about $\sqrt{L}$ layers apart.
18. How would lowering precision change both workload arithmetic intensity and accelerator intensity?

# Source coverage checklist

| Raw transcript span | Material covered | Covered above |
|---:|---|:---:|
| 1-67 | Marin scaling run, iso-FLOP curves, forecast accuracy, extrapolation caution | Yes |
| 68-358 | Lecture goal, finite resources, 70B/15T runtime estimate, eight-H100 memory estimate, mechanics/mindset/intuition | Yes |
| 359-699 | Tensors as universal storage, ranks, FP32 representation, memory formula, GPT-3 matrix, reasons to reduce precision | Yes |
| 700-1197 | FP16 range failures, BF16 tradeoff, mixed precision and AMP, FP8, NVFP4, block scales, Nemotron 3 Super | Yes |
| 1198-1354 | Audience questions on FP4 block scaling and one-bit training versus inference quantization | Yes |
| 1355-1437 | Default CPU placement, moving or creating tensors on GPU, laptop limitation | Yes |
| 1438-1809 | Motivation for named dimensions, basic and batched einsum, ellipsis | Yes |
| 1810-2163 | `reduce`, speed question, `rearrange`, head/hidden split, flatten-order question, Einops learning curve | Yes |
| 2164-2442 | FLOPs versus FLOP/s, H100 sparse/dense specification, one-week/two-week inconsistency, hardware-scale intuition | Yes |
| 2443-2760 | Linear-layer FLOP counting, elementwise cost, parameter interpretation, sub-cubic matmul and addition-versus-multiplication questions | Yes |
| 2761-3210 | CUDA synchronization, repeated timing, realized throughput, MFU definition and interpretation, dtype-dependent peak | Yes |
| 3211-3839 | HBM-to-compute data movement, H100 bandwidth, overlap assumption, ReLU timing, accelerator and algorithm intensity | Yes |
| 3840-4568 | GELU, dot product, matrix-vector and matrix-matrix intensity, training/inference contrast, MFU connection, roofline plot, hardware-balance question | Yes |
| 4569-4812 | Deep-network construction, parameter count, simple autograd example | Yes |
| 4813-5325 | Backward contractions, gradient shapes, backward/forward FLOP ratio, $6PT$, Transformer long-context caveat | Yes |
| 5326-5793 | AdaGrad state and update, parameter groups, FP32 optimizer-state rationale, full memory breakdown, Transformer assignment connection | Yes |
| 5794-5892 | Critical batch size, microbatch procedure, gradient accumulation and memory/compute wording discrepancy | Yes |
| 5893-6150 | Training versus inference activations, checkpointing/rematerialization, PyTorch mechanism, full/none/$\sqrt{L}$ checkpoint schedules | Yes |
| 6151-6244 | Lecture summary and transition to architectures | Yes |
