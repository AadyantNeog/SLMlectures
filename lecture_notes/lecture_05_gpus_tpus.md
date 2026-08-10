---
title: "Lecture 5 - GPUs, TPUs, and Hardware-Aware Performance"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 5
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 5 GPUs, TPUs.txt"
slide_deck: "../lecture_05.pdf"
status: "complete"
---

# Lecture 5: GPUs, TPUs, and Hardware-Aware Performance

## How to read these notes

Every substantive topic has two layers:

1. **What the lecturer said - transcript only.** This is a cleaned, concise paraphrase of the spoken lecture. It preserves substantive definitions, claims, examples, qualifications, numbers, and audience questions while removing filler and repetition.
2. **Additional explanation.** This is independent intuition, derivation, or study guidance. It is not presented as something the lecturer said.

The physical line spans of the raw transcript are included for auditability. The complete slide deck was inspected after the transcript map was established. Slides were used to verify terminology, diagrams, equations, and structure. Material discrepancies are called out explicitly rather than silently folded into the transcript account.

## Lecture map

The lecture has three parts:

1. Build a useful mental model of GPU hardware, briefly compare GPUs with TPUs, and explain why matrix multiplication and data movement dominate modern ML performance.
2. Explain six practical performance ideas: avoiding control divergence, low-precision computation, operator fusion, recomputation, coalesced memory access, and tiling.
3. Combine those ideas to explain FlashAttention, including why online softmax and recomputation avoid materializing the quadratic attention matrix in global memory.

---

# Part I - How accelerators work

## 1. Why architecture researchers need a hardware model

**Transcript coverage:** lines 1-183

### What the lecturer said - transcript only

GPU systems can initially feel like a collection of mysterious rules. The lecturer's objective is to replace that feeling with a model that makes performance predictable. Even researchers who primarily design model architectures need this model: architecture choices determine matrix shapes, memory traffic, and the operations that hardware must execute.

The motivating puzzle is a plot of achieved throughput for square matrix multiplications. Performance increases unevenly, with repeated jumps, dips, and periodic patterns. By the end of the lecture, the class should be able to explain much of that plot in terms of compute intensity, tiling, alignment, and wave quantization. The same principles will then explain why FlashAttention is fast.

The lecture draws heavily on three resources: Horace He's writing on GPU performance, CUDA and GPU Mode materials, and *How to Scale Your Model*, which compares GPU and TPU systems. The material is organized into GPU hardware, methods for making ML workloads fast, and FlashAttention as a case study.

### Additional explanation

The point of a hardware model is not to memorize one GPU's specification sheet. It is to reason from a few durable constraints:

- arithmetic units can consume data much faster than distant memory can supply it;
- nearby memory is fast but small;
- many threads share execution machinery;
- specialized matrix units reward particular operations, data types, and shapes;
- a finite number of processors execute work in batches or "waves."

These constraints survive changes in GPU generation and also transfer, with different names and sizes, to TPUs and other accelerators.

## 2. Where language-model compute scaling comes from

**Transcript coverage:** lines 184-318

### What the lecturer said - transcript only

Language-model capability has a predictable relationship with the amount of useful computation invested in training. More compute can come from faster hardware, better utilization of each device, more devices, or algorithms that parallelize more effectively.

For decades, a major source of speed was **Dennard scaling**: shrinking transistors allowed clock rates and processor performance to rise without a proportional increase in power density. That traditional route largely ended in the 2000s because physical and power limits prevented clock speeds from continuing to rise in the same way.

Modern scaling therefore relies heavily on horizontal parallelism. GPUs provide many parallel workers, and GPU-based ML throughput grew by more than three orders of magnitude over roughly a decade. Several sources contributed: lower-precision number formats, specialized instructions such as tensor-core matrix operations, fabrication improvements, sparsity support, and improved models and software. The lecturer's central claim is that contemporary language-model scaling would not exist without GPU scaling.

### Additional explanation

Scaling a workload horizontally requires enough independent work to occupy many processors. Dense neural networks are unusually suitable because batches, tokens, heads, channels, and matrix tiles expose parallel dimensions. This does not make scaling automatic: synchronization, memory bandwidth, networking, and uneven work distribution can still leave processors idle.

The historical shift can be summarized as follows:

| Earlier emphasis | Modern accelerator emphasis |
|---|---|
| Make one instruction stream faster | Execute many instruction streams or data lanes together |
| Raise clock frequency | Add parallel arithmetic units and specialized matrix units |
| Optimize latency of one task | Optimize throughput across a large workload |
| Rely mainly on general-purpose operations | Co-design models around efficient accelerator primitives |

## 3. CPU versus GPU: latency and throughput

**Transcript coverage:** lines 319-472

### What the lecturer said - transcript only

A CPU is designed to finish a relatively small number of complicated threads quickly. It devotes substantial hardware to control flow, branch prediction, caching, and sophisticated instruction execution. This is appropriate for workloads with irregular decisions and strong sequential dependencies.

A GPU instead devotes much more of its area to many small arithmetic units. It supports enormous numbers of lightweight threads but provides less machinery for complicated branching. The optimization target is total throughput: some threads may wait for data, but the GPU can schedule other ready threads and keep the device busy.

The distinction is therefore not simply that a GPU is a "faster CPU." A CPU primarily minimizes the latency of each thread; a GPU primarily maximizes the amount of similar work completed per unit time.

### Additional explanation

An analogy is express checkout versus a large sorting center. A CPU gives a few jobs rich individual attention and tries to finish each quickly. A GPU builds many simpler lanes and wins when there are enough similar items to keep every lane occupied.

This is why GPU-friendly programs usually have:

- abundant parallel work;
- regular control flow;
- contiguous or reusable data;
- enough computation per memory transfer;
- large matrix operations that specialized units can execute.

## 4. GPU compute anatomy: SMs, processors, threads, blocks, and warps

**Transcript coverage:** lines 473-646

### What the lecturer said - transcript only

A GPU is divided into many **streaming multiprocessors** (SMs). Each SM is a largely independent compute unit that can execute assigned jobs, called blocks. Within an SM are many smaller streaming processors or arithmetic lanes that execute threads in parallel.

The CUDA execution model has three important levels:

- A **thread** performs one instance of the work. Threads use the same instruction program on different inputs, following the SIMT model: single instruction, multiple threads.
- A **block** is a group of threads. A block is assigned to one SM, and its threads can communicate through that SM's shared memory.
- A **warp** is the hardware scheduling group, normally 32 consecutively numbered threads. Those threads execute instructions together.

In response to a question, the lecturer clarified that the lockstep behavior is most usefully understood at warp granularity. Blocks are a programming and placement unit; warps are the groups that the hardware actually schedules through an instruction stream.

### Source reconciliation

The slides use several GPU counts for different purposes. One diagram labels a full GA100 die as having 128 SMs; the wave-quantization example uses an A100 with 108 available SMs; the GPU/TPU comparison table labels an H100 column with 132 SMs. The spoken explanation occasionally blurs these device labels. These are different die or product configurations, not one universal SM count. The performance arguments depend on the finite number of available SMs, not on one fixed number.

### Additional explanation

The hierarchy is:

```text
GPU
  SM
    block(s)
      warp(s)
        32 threads per warp
```

More than one block may reside on an SM when registers, shared memory, and thread limits permit it. Likewise, an SM can keep several warps ready. When one warp stalls on memory, the scheduler can issue an instruction from another. This latency-hiding mechanism is one reason threads are intentionally lightweight.

SIMT differs from fully independent scalar threads. Each lane has its own data and logical thread identity, but lanes in a warp share instruction issue. That shared issue machinery makes uniform work efficient and divergent control flow expensive.

## 5. The GPU memory hierarchy

**Transcript coverage:** lines 647-731

### What the lecturer said - transcript only

Memory becomes faster as it gets closer to the arithmetic units, but its capacity falls and its cost rises.

- **Registers** are private to a thread and are the fastest storage discussed.
- **Shared memory** is explicitly managed storage shared by threads in a block on the same SM.
- **L1 cache** is also close to the SM, but the hardware manages it automatically rather than the programmer placing values there explicitly.
- **L2 cache** is shared more broadly on the GPU die.
- **Global memory**, usually high-bandwidth memory (HBM), provides the large device-memory capacity but is much slower to reach than on-chip storage.
- **Host memory** belongs to the CPU side and is farther away still.

Each thread can access its registers, and threads within a block can exchange information through shared memory. Information that must cross block boundaries generally has to pass through global memory, which is expensive. This motivates grouping work so that values loaded from global memory can be reused within a block.

An audience question asked why both shared memory and L1 exist if they are physically similar kinds of fast memory. The answer was that shared memory gives the program direct placement and lifetime control, whereas cache is automatic. The exact organization also reflects distance, interconnect, and hardware tradeoffs.

### Source reconciliation

The memory slide summarizes SRAM as roughly 100 times more expensive and about 8 times faster than DRAM, while the spoken explanation uses looser language such as "hundreds" of times more costly and about an order of magnitude faster. These are illustrative comparisons, not a portable constant across chips. The stable point is that fast on-chip memory is scarce and expensive, so the hierarchy is unavoidable.

### Additional explanation

Capacity, bandwidth, and latency are different quantities:

- **Capacity** is how many bytes can be stored.
- **Bandwidth** is how many bytes can be transferred per second.
- **Latency** is how long one access takes before its data arrives.

HBM has high aggregate bandwidth, but a thread waiting on a single HBM access still experiences high latency. A GPU compensates with many active warps and with reuse in registers, shared memory, and caches. A fast kernel therefore often looks less like "do fewer FLOPs" and more like "make every expensive byte support many FLOPs."

## 6. A side thread on TPUs

**Transcript coverage:** lines 733-954

### What the lecturer said - transcript only

GPUs and TPUs are examples of convergent accelerator design. At a high level, both combine lightweight control, fast local memory, and large matrix-multiplication machinery connected to HBM. Concepts learned for GPUs therefore transfer substantially to TPUs even though the names and exact units differ.

The broad mapping is:

- a GPU SM corresponds roughly to a TPU tensor core as a processor-level unit;
- the GPU warp scheduler corresponds to TPU vector-processing control;
- GPU CUDA arithmetic lanes correspond to TPU vector ALUs;
- GPU shared/L1 memory corresponds to TPU vector memory;
- a GPU tensor core corresponds to the TPU matrix multiply unit (MXU);
- both use HBM as large high-bandwidth memory.

Terminology is especially confusing: a **GPU tensor core** is a matrix-multiplication circuit inside an SM, whereas a **TPU TensorCore** is a much larger processor-level unit containing control, vector, memory, and matrix units.

The main architectural difference emphasized in the lecture is granularity. A GPU has many relatively small SMs and matrix units, while a TPU has far fewer but much larger units. The TPU is consequently less flexible about matrix shapes and non-matrix work. The lecturer gives an example in which a TPU matrix dimension effectively needs to be at least around 64 to use its unit well. GPUs offer finer-grained scheduling, while TPUs lean more heavily into large matrix operations. Their networking topologies also differ, to be discussed in the parallelism lecture.

### Source reconciliation

The spoken comparison gives rough processor and matrix-unit counts and occasionally uses the GPU model name imprecisely. The slide's table compares an H100-like column with TPU v5p: 132 GPU SMs versus 2 TPU processor-level TensorCores, and 528 smaller GPU matrix units versus 8 larger TPU MXUs. These numbers illustrate granularity, not equivalent one-for-one units or a claim that the devices have identical overall behavior.

### Additional explanation

The comparison is best viewed as a spectrum:

| Property | GPU tendency | TPU tendency in this lecture |
|---|---|---|
| Number of processor-level units | Many | Few |
| Matrix unit size | Smaller, numerous | Larger, fewer |
| Scheduling granularity | Fine | Coarser |
| Flexibility for irregular work | Higher | Lower |
| Reward for large regular matmuls | High | Very high |

Neither side is universally better. A large regular training workload may map naturally to a coarse systolic array. Irregular kernels, small shapes, or diverse operators may benefit from the GPU's finer structure. Software, networking, and availability can dominate the hardware-only comparison.

## 7. Why GPUs became matrix-multiplication machines

**Transcript coverage:** lines 955-1160

### What the lecturer said - transcript only

The GPU model has three important strengths. It scales a large workload by adding SMs, its SIMT programming model exposes parallel work in a relatively direct way, and its lightweight threads can be paused and resumed so the machine can hide memory stalls.

Early GPUs were built for graphics and exposed programmable shaders. Researchers repurposed those shaders to perform matrix multiplication before GPUs had dedicated matrix hardware. Starting with NVIDIA's V100 generation, tensor cores made matrix multiplication a privileged operation. Matrix operations can now achieve more than ten times the floating-point throughput of ordinary non-matrix operations on the same broad class of device.

This creates a strong architectural incentive: a model that can express most of its work as large matrix multiplications can use the fastest-growing part of the hardware. The lecturer expects scalable near-future architectures to continue relying heavily on matrix multiplication.

At the same time, arithmetic throughput has improved much faster than memory and interconnect bandwidth. It is increasingly difficult to keep all matrix units supplied with data. Modern optimization must therefore respect the memory hierarchy and minimize data movement rather than count FLOPs alone.

### Additional explanation

This is a hardware-software contract. Hardware vendors make particular operations exceptionally fast because ML workloads use them heavily; model designers then choose architectures that use those operations, further increasing their importance.

The contract can become a constraint. An operation that is mathematically elegant may be unattractive if it decomposes into irregular gathers, branches, or many small kernels. Conversely, an approximation that yields dense matmuls can be faster even when it performs more nominal FLOPs.

## 8. Hardware questions: specialization, cache, and future inference chips

**Transcript coverage:** lines 1161-1283

### What the lecturer said - transcript only

Before tensor cores, GPUs still performed matrix multiplication using their general arithmetic lanes; there simply was no dedicated matrix circuit. Tensor cores add specialized hardware rather than making matrix multiplication newly possible.

Another question returned to L1 and shared memory. They use related on-chip storage technology but make different tradeoffs: cache placement is automatic, shared-memory placement is explicit, and physical distance and interconnect matter. TPUs may choose a different balance, including larger local vector memory and a different cache structure.

Looking ahead, the lecturer expects inference workloads to motivate more specialized systems. **Prefill**, which processes a prompt in parallel, is relatively compute-heavy. **Decode**, which produces tokens one at a time while reading model weights and the key-value cache, is relatively memory-heavy. These phases may be disaggregated onto different chips. Hardware could specialize even further for attention versus MLP computation. The lecturer expects inference to make memory constraints more prominent.

Warp and block parameters are hardware-dependent. The programming model exposes familiar concepts, but useful sizes and resource limits depend on the device.

### Additional explanation

Disaggregation is attractive because one universal chip must compromise. A prefill engine wants high matrix throughput and large batches; a decode engine wants excellent bandwidth, capacity, and latency at smaller per-request work. Separating them can improve utilization, but it adds transfer and scheduling costs between phases.

The larger lesson is to avoid treating GPU rules as eternal constants. The durable questions are: What work is parallel? Which operations have dedicated units? Where does each tensor live? How often does it move? How evenly is work distributed?

---

# Part II - Making ML workloads fast

## 9. The matrix-performance puzzle and the roofline model

**Transcript coverage:** lines 1284-1390

### What the lecturer said - transcript only

Even square matrix multiplication produces a complicated throughput curve. Small matrices achieve little of the advertised peak, performance then rises, and larger sizes show jumps and periodic drops. The first framework for interpreting this behavior is the **roofline model**.

An operation at low arithmetic intensity does too little computation for every byte fetched. It is **memory-bound**, so attainable performance rises with intensity and memory bandwidth. Once the operation supplies enough work per byte to saturate the arithmetic units, it becomes **compute-bound** and hits a flat hardware ceiling. The practical question for the rest of the section is how to avoid remaining memory-bound.

### Source reconciliation

Later, the low-precision slide labels its ReLU calculation as "8 bytes/FLOP" and calls that intensity. In standard roofline terminology, arithmetic or operational intensity is **FLOPs per byte**. The lecturer notes in speech that the displayed quantity is the inverse. These notes use the standard convention and call bytes/FLOP a memory-cost ratio.

### Additional explanation

Let

- $P_{\max}$ be peak compute throughput in FLOP/s,
- $B$ be memory bandwidth in bytes/s,
- $I$ be arithmetic intensity in FLOPs/byte.

The roofline bound is

$$
P \leq \min(P_{\max}, B I).
$$

The transition occurs near

$$
I^* = \frac{P_{\max}}{B}.
$$

Below $I^*$, reducing bytes or increasing reuse raises performance. Above it, more reuse alone cannot exceed the compute roof. Real kernels can sit below both roofs because of control divergence, instruction overhead, poor occupancy, synchronization, or uneven work.

## 10. Control divergence

**Transcript coverage:** lines 1391-1491

### What the lecturer said - transcript only

Control divergence is not primarily a memory bottleneck, but it follows directly from the SIMT execution model. Every thread in a warp is supposed to execute the same instruction. If a conditional sends some lanes through one branch and the rest through another, the hardware cannot simply run both groups independently at full width.

Instead, it executes the branch paths serially while masking lanes for which each path does not apply. During one branch, the other lanes are inactive; during the other branch, the first group is inactive. The result is wasted execution capacity. Conditionals are legal, but divergent branches inside a warp can create substantial overhead. When possible, express work as uniform operations or masks rather than long, different control paths.

### Additional explanation

Divergence matters at warp granularity. A conditional is cheap when every lane makes the same choice, even if another warp chooses differently. It becomes costly when lanes of the same warp disagree.

Short predicated expressions may be efficient because both candidate values can be computed and selected. That is not a universal rule: computing two very expensive paths may cost more than branching. The correct choice depends on branch coherence, path length, compiler behavior, and data distribution.

## 11. Low precision and mixed precision

**Transcript coverage:** lines 1492-1649

### What the lecturer said - transcript only

The first memory-focused technique is low-precision computation. Fewer bits mean fewer bits to move, more values fitting in fast memory, and often more arithmetic operations per cycle on tensor cores.

For an elementwise ReLU on FP32 values, the kernel reads a 4-byte input and writes a 4-byte output for roughly one comparison operation: about 8 bytes moved per operation. FP16 halves the storage to 2 bytes per value, so the same read and write move about 4 bytes. This doubles the conventional FLOPs-per-byte intensity.

Training does not have to use one format everywhere. Tensor cores can multiply low-precision inputs while accumulating into a higher-precision result. Many matrix multiplies and pointwise operations tolerate FP16 or BF16 storage, while reductions and numerically sensitive operations may need FP32 or another wider format. Adding small values into a large running sum is particularly vulnerable to rounding; softmax, normalization, exponentials, and losses may need care.

The correct precision policy is partly empirical. Different operations have different numerical requirements, and stable training normally uses a mixture rather than blindly converting the entire graph.

### Additional explanation

Low precision helps in three distinct ways:

1. **Bandwidth:** fewer bytes cross HBM and caches.
2. **Capacity:** more parameters and activations fit in a fixed memory budget.
3. **Compute:** tensor cores often provide higher peak throughput for smaller formats.

These gains are not automatically identical. A memory-bound operator may benefit mostly from bandwidth. A compute-bound matmul may benefit from tensor-core throughput. A tiny operator may see little gain because kernel launch and conversion overhead dominate.

BF16 and FP16 both use 16 bits but allocate exponent and mantissa bits differently. BF16 preserves an FP32-like exponent range with less significand precision; FP16 has more significand precision but a narrower range. This is why BF16 is often easier to train with even though both formats occupy the same number of bytes.

## 12. FP8, block scaling, MXFP8, and MXFP4

**Transcript coverage:** lines 1650-1872

### What the lecturer said - transcript only

Moving from 16 bits to 8 bits makes range and precision harder to balance. FP8 variants therefore choose different exponent and mantissa allocations. E4M3 provides more mantissa precision and less exponent range; E5M2 provides more range and less precision. A scaling factor maps a tensor's useful numerical range into the representable FP8 range so fewer values overflow or underflow.

**MXFP8** uses many local scaling factors rather than one scale for an entire tensor. Values are divided into blocks, with one scale for every 32 values. Because local scaling supplies range, the data can use E4M3 and retain more mantissa bits. The scales themselves use an E8M0-style power-of-two representation.

Block scaling complicates transposes. A block layout that groups consecutive values in one orientation does not generally form the right groups after transposition. Training therefore may maintain separately quantized representations of the original matrix and its transpose, or requantize them when needed.

In practical MXFP8 training, only selected matrix-heavy parts of a Transformer are quantized. Other layers remain in BF16 or wider precision. Quantization, dequantization, scaling, and maintaining alternate layouts have overhead, so the end-to-end improvement may be around 20-30%, not the naive 2x suggested by halving 16-bit storage.

The first and last layers are often difficult to quantize. The lecturer offers only an intuition for the last layer: its output directly affects the loss, so errors there can destabilize training.

**MXFP4** goes further. Its data values come from a very small representable set, roughly including magnitudes from zero through 6, with a scale per block of 16 values and an FP8 E4M3 scale. The lecturer expects such formats to matter but says that, at the time of the lecture, he does not know of a broadly demonstrated large-scale training success in the wild that makes the result routine.

### Source reconciliation

Automatic transcription occasionally renders the format names incorrectly. The slide deck verifies the intended names as FP8 E4M3, FP8 E5M2, MXFP8 with E8M0 scales, and MXFP4 with E4M3 scale factors.

### Additional explanation

Block scaling trades metadata and layout complexity for local adaptivity. If one scale covers a whole tensor, a single outlier can force most values into a tiny portion of the available code range. Per-block scales let unrelated regions choose different magnitudes.

For a value block $x$ and scale $s$, the conceptual operation is

$$
q = \operatorname{round}_{\mathcal{F}}(x/s),
\qquad
\hat{x} = s q,
$$

where $\mathcal{F}$ is the small representable value set. The training system must choose $s$, store or reconstruct it, and ensure the hardware can consume $q$ efficiently. The extra operations explain why a bit-width reduction does not translate directly into the same wall-clock speedup.

## 13. Questions about quantization and sparsity

**Transcript coverage:** lines 1873-2026

### What the lecturer said - transcript only

Quantization can in principle be applied to many operations, but it is most worthwhile where hardware provides an efficient low-precision implementation, especially matrix multiplication. Before conversion overheads and numerical safeguards, throughput gains can scale roughly with the reduction in bit width. End-to-end gains are smaller.

During training, scaling factors are normally selected from statistics of current values or running statistics. They are not generally ordinary parameters learned by gradient descent in the same manner as model weights.

Structured sparsity is another way hardware can skip work. It can provide nominal throughput gains, but the restrictions, bookkeeping, and model-quality effects often reduce the practical advantage. The lecturer characterizes the empirical outcome as frequently washing out rather than providing an automatic win.

State-of-the-art very-low-precision work may combine quantization-aware training, model scaling, and post-training quantization. Which recipe works best and why is still not fully understood. The lecture treats these formats as an active frontier rather than a settled drop-in replacement.

### Additional explanation

Quantization has at least four distinct error sources:

- clipping values outside the chosen range;
- rounding values within the range;
- stale or poorly chosen scales;
- accumulated error across operations and time.

A format can work for inference yet fail during training because gradients, optimizer states, reductions, and weight updates have different distributions and sensitivity. Claims about "FP8 training" should therefore be read carefully: they may refer only to matmul inputs while accumulators, master weights, normalization, and optimizer state remain wider.

## 14. Operator fusion

**Transcript coverage:** lines 2027-2131

### What the lecturer said - transcript only

Operator fusion addresses unnecessary trips between compute and global memory. The lecturer compares the GPU to a factory and memory to a warehouse. If every small processing step sends an item back to the warehouse and the next step fetches it again, transport dominates the useful work. A fused kernel keeps intermediate values near the compute units and performs several operations before writing the final result.

The example computes

$$
\sin^2(x) + \cos^2(x).
$$

A naive eager graph launches five pointwise kernels: sine, square, cosine, square, and add. Each kernel can write an intermediate tensor to global memory for the next kernel to read. A single fused kernel can load $x$, perform all five pointwise operations locally, and write only the final output.

Compilers such as `torch.compile` and JAX can automatically fuse straightforward pointwise graphs. More complicated fusion patterns may still require manual kernel design.

### Additional explanation

Fusion can improve performance through:

- fewer HBM reads and writes;
- fewer kernel launches;
- keeping intermediates in registers or shared memory;
- exposing optimization opportunities across operator boundaries.

Fusion also has limits. A very large fused kernel may use too many registers, reduce occupancy, duplicate computation, or require synchronization that prevents a legal fusion. Materializing an intermediate can also be useful when several consumers reuse it. The best fusion boundary is therefore an optimization decision, not "fuse everything."

## 15. Recomputation

**Transcript coverage:** lines 2132-2252

### What the lecturer said - transcript only

Backpropagation normally stores forward activations because the backward graph needs them to compute derivatives. Storing and later retrieving those activations can be more expensive than recomputing them.

The lecture uses three stacked sigmoid operations. A naive implementation reads the original input once, writes three forward activations, reads three saved values during backward, and writes the final gradient: eight memory operations in the simplified count.

An alternative discards the intermediate activations. During backward, it reads the original input and recomputes the sigmoid chain, then evaluates the original backward graph. In the lecture's count this uses five memory operations rather than eight, or $5/8$ as many accesses. It performs extra arithmetic but returns the same mathematical result.

Throwing away computed values can therefore be optimal when arithmetic is cheap and memory movement is the bottleneck.

### Additional explanation

This technique is also called activation checkpointing when applied across sections of a network. The system stores selected boundary activations and recomputes everything between them during backward.

The tradeoff is:

| Store activation | Recompute activation |
|---|---|
| More memory capacity and traffic | More FLOPs |
| Less backward latency | Longer backward compute |
| Useful for expensive or nondeterministic operations | Useful for cheap deterministic operations |

Recomputation can reduce both memory capacity and bandwidth pressure, but the exact benefit depends on whether the saved activation would have stayed in cache, whether the recomputed operator is itself memory-bound, and how much extra work is introduced.

## 16. DRAM burst access and memory coalescing

**Transcript coverage:** lines 2253-2453

### What the lecturer said - transcript only

Global DRAM does not efficiently return one arbitrary tiny value at a time. It reads in **bursts**, transferring a contiguous region that may be about 128 bytes. This behavior follows from DRAM's physical organization: opening a row and moving data to sensing circuitry is expensive, so neighboring bytes are delivered together.

Memory access is **coalesced** when threads in a warp request addresses that fit into the same burst region. The hardware can service those requests with one transaction. If the threads request addresses scattered across many regions, the warp requires many transactions and much of each transferred burst may be unused.

The matrix example assumes row-major storage. If consecutive warp lanes access neighboring locations along the contiguous memory dimension, their loads can coalesce. If the lanes stride through separate rows or columns so that their addresses are far apart, each may trigger a separate burst. Kernel writers therefore organize thread indices and matrix layouts so simultaneous lane accesses are contiguous.

An audience question prompted the lecturer to restate the row-versus-column intuition directly in terms of physical layout: follow the dimension that is contiguous in memory, not merely the visual direction in which a matrix is drawn.

### Source reconciliation

One slide says "threads that move along rows are not coalesced," while its diagrams distinguish the direction assigned across threads from the direction each individual thread traverses. Read literally, that sentence conflicts with the usual row-major rule. The spoken explanation resolves the ambiguity: coalescing depends on whether **simultaneous lanes** touch consecutive addresses within a burst. These notes use that address-based statement.

### Additional explanation

For a row-major matrix with width $W$, element $(i,j)$ has a linear position proportional to

$$
iW + j.
$$

If lane $t$ accesses $(i,j+t)$, adjacent lanes touch adjacent elements. If lane $t$ accesses $(i+t,j)$, adjacent lanes are separated by a stride of $W$. The first pattern is usually coalesced; the second may require one transaction per lane unless the layout or access is transformed.

Alignment also matters. Even a contiguous request can span two burst boundaries. Padding or changing the starting address can make one warp's requested region fit cleanly inside the hardware transaction size.

## 17. Tiling: reuse data in shared memory

**Transcript coverage:** lines 2454-2687

### What the lecturer said - transcript only

Tiling groups matrix work into submatrices so that data loaded from global memory can be reused in shared memory. In a naive matrix multiplication, many output threads independently reread the same rows and columns. The accesses may be repeated and poorly coalesced.

A tiled implementation proceeds in phases. A block cooperatively loads a tile of the left matrix and a tile of the right matrix into shared memory. The threads compute partial sums for an output tile using those local values. The block then loads the next pair of input tiles and continues until the output tile is complete.

For square matrices of size $N$ and tile width $T$, the simplified comparison is:

- without tiling, an input value may be fetched from global memory $N$ times;
- with tiling, it is fetched about $N/T$ times from global memory and then reused $T$ times locally.

This yields a factor-$T$ reduction in global-memory access under the simplified model. Tiling can also arrange the cooperative loads so they are coalesced.

Standard matrix-multiplication libraries already perform sophisticated tiling. Users of high-level matmul calls normally receive this optimization automatically. It becomes an explicit concern when writing custom kernels or when matrix shapes interact poorly with the library's available kernels.

### Additional explanation

For $C=AB$, a tile of $C$ depends on a strip of tiles from $A$ and $B$:

$$
C_{ij} = \sum_k A_{ik}B_{kj}.
$$

The key is that every loaded $A_{ik}$ tile contributes to multiple output elements across the tile, and every $B_{kj}$ tile does likewise. Shared memory acts as a programmer-managed reuse cache.

Tile size balances several constraints:

- larger tiles offer more reuse;
- larger tiles consume more shared memory and registers;
- too many resources per block reduce the number of resident blocks;
- hardware matrix instructions prefer particular internal shapes;
- loads should be aligned and coalesced;
- edge tiles may contain little useful work.

This is why production libraries maintain many kernels and select among them rather than using one universal tile.

## 18. Tiling complications, alignment, padding, and autotuning

**Transcript coverage:** lines 2688-2807

### What the lecturer said - transcript only

Real matrix dimensions may not be divisible by a kernel's tile shape. A small increase in dimension can create an extra row or column of mostly empty tiles, lowering utilization. Shared-memory capacity, coalescing requirements, and matrix divisibility all constrain the available tile sizes.

Alignment creates another issue. DRAM transfers fixed contiguous bursts. If a tile begins or ends across a burst boundary, fetching it can require two transactions instead of one. Padding a matrix to a favorable multiple can therefore make a larger matrix faster than the unpadded smaller one.

The lecturer cites Andrej Karpathy's nanoGPT example: padding the vocabulary dimension from 50,257 to 50,304, a multiple of 64, produced roughly a 25% speedup even though it introduced unused dimensions. The padded shape selected a more efficient kernel path with higher occupancy and better alignment.

Kernel selection has many interacting parameters, so compilers and libraries often use autotuning: benchmark several legal implementations on the actual shape and retain the fastest. Users should generally think about divisibility by convenient factors such as 16 or 32, while kernel authors must reason about the exact tile and transaction layout.

### Additional explanation

Padding is an example of trading nominal work for hardware efficiency. If the unpadded operation performs $F$ useful FLOPs at utilization $u_1$ and the padded operation performs $F+\Delta F$ at utilization $u_2$, padding wins whenever

$$
\frac{F+\Delta F}{u_2 P_{\max}} < \frac{F}{u_1 P_{\max}}.
$$

The extra arithmetic can be cheaper than the idle hardware and extra memory transactions it removes.

Powers of two are not magical in isolation. They are frequently useful because warp widths, transaction sizes, vector widths, tensor-core fragments, and tile dimensions are themselves built from such factors. A multiple larger than the relevant hardware unit does not automatically confer an additional benefit.

## 19. Explaining the matrix mystery: tiling and wave quantization

**Transcript coverage:** lines 2808-3033

### What the lecturer said - transcript only

The square-matmul performance plot can now be partially decoded. Its initial rise reflects increasing arithmetic intensity: larger matrices provide enough computation and reuse to amortize memory traffic. Sharp differences among nearby dimensions often reflect tile divisibility and aligned, coalesced access.

The periodic drops have another cause: **wave quantization**. Suppose the kernel uses output tiles of $256 \times 128$. A square matrix of size 1792 requires

$$
\frac{1792}{256}\frac{1792}{128}=7\times14=98
$$

output tiles. On the A100 configuration used in the example, 108 SMs can process those 98 tiles in one wave. Increasing the matrix size by one to 1793 requires ceiling division:

$$
\left\lceil\frac{1793}{256}\right\rceil
\left\lceil\frac{1793}{128}\right\rceil
=8\times15=120
$$

tiles. The first 108 occupy all SMs, and the remaining 12 require a second wave during which most SMs are idle. A one-element increase can therefore cause a conspicuous throughput drop.

The part concludes with a memory-centered summary:

- coalesce simultaneous accesses;
- fuse operations to avoid round trips;
- tile work so data is reused in shared memory;
- quantize to move fewer bits, accepting accuracy and conversion tradeoffs;
- recompute cheap values instead of storing and reloading them.

Questions asked why the device cannot simply use only SRAM. The answer is cost, area, power, and physical proximity: large fast memory cannot be placed everywhere economically. Another question asked whether divisibility beyond 32 always helps; it does not once the relevant burst or tile requirement is already met. A final question mentioned wafer-scale hardware such as Cerebras. The lecturer said it is substantially more complex and outside his expertise rather than offering a confident comparison.

### Additional explanation

Wave quantization is ordinary ceiling behavior amplified by parallel hardware. If a GPU has $S$ processors and a kernel launches $T$ similarly expensive tiles, its idealized number of waves is

$$
W=\left\lceil\frac{T}{S}\right\rceil.
$$

The last-wave utilization is

$$
u_{\text{last}} = \frac{T-(W-1)S}{S}.
$$

Performance can drop whenever $T$ crosses a multiple of $S$, even if the total amount of useful work changes only slightly. The exact curve is softened by multiple resident blocks, unequal tile costs, pipeline overlap, and library kernel changes, but the quantization principle remains.

---

# Part III - FlashAttention as a hardware-aware algorithm

## 20. Why ordinary attention moves too much data

**Transcript coverage:** lines 3034-3149

### What the lecturer said - transcript only

FlashAttention dramatically accelerates attention by applying the preceding ideas to the memory hierarchy. Its advantage does not come from reducing the mathematical attention computation to a radically different approximation. It uses fusion, tiling, and recomputation to reduce HBM traffic and avoid storing large intermediates.

Attention constructs queries, keys, and values, computes query-key scores, applies a softmax, and combines the normalized scores with values. Tiling the matrix multiplications is familiar: blocks of $Q$, $K$, and $V$ can be loaded into fast on-chip memory and used to compute blocks of the output.

The difficulty is softmax. A normal row-wise softmax depends on every score in the row because it needs the row maximum for numerical stability and the sum of all exponentials for normalization. If the entire score row must be materialized before softmax, the quadratic attention matrix returns to HBM and defeats the intended memory saving.

### Source reconciliation

The lecture and slide use the shorthand "three matrix multiplies" when recapping the broader $Q$, $K$, and $V$ attention computation. Once projected $Q$, $K$, and $V$ are given, the core attention expression

$$
\operatorname{softmax}(QK^\top)V
$$

contains two large matmuls: score formation and value aggregation. Projection matmuls account for the broader count. The distinction does not change the FlashAttention argument.

### Additional explanation

Standard attention can be written as

$$
S = \frac{QK^\top}{\sqrt{d}},
\qquad
P = \operatorname{softmax}(S),
\qquad
O = PV.
$$

For sequence length $N$, $S$ and $P$ contain $N^2$ elements per head or batch slice. Even when the matmuls are efficient, writing and rereading these matrices can dominate runtime and activation memory. FlashAttention targets these **I/O operations**, not merely the FLOP count.

## 21. Online softmax and the FlashAttention forward pass

**Transcript coverage:** lines 3150-3249

### What the lecturer said - transcript only

The key enabling idea is **online softmax**. Rather than seeing the whole row at once, the algorithm processes score tiles while maintaining a running maximum and a running normalization sum. When a new tile contains a larger maximum, the previously accumulated contribution is rescaled so that it remains consistent with the new maximum. This forms a telescoping update and produces the same normalized softmax as the full-row computation.

FlashAttention can therefore keep a score tile, its exponentials, the running maximum, the running normalizer, and a partial output in registers or shared memory. It multiplies the current normalized or appropriately rescaled score contribution by the corresponding $V$ tile and accumulates the result. At the end, the running quantities provide the correctly normalized output.

The forward pass combines:

- tile-wise query-key inner products;
- fusion of exponentiation and normalization bookkeeping;
- online, tile-wise softmax;
- tile-wise multiplication by values;
- no materialized full $N\times N$ attention matrix in HBM.

### Additional explanation

For one score row processed in chunks, maintain:

- running maximum $m$;
- running exponential sum $\ell$;
- running unnormalized value accumulator $u$.

For a new score chunk $x_j$ with corresponding values $v_j$, update

$$
m' = \max\left(m, \max_j x_j\right),
$$

$$
\ell' = e^{m-m'}\ell + \sum_j e^{x_j-m'},
$$

$$
u' = e^{m-m'}u + \sum_j e^{x_j-m'}v_j.
$$

After all chunks,

$$
o = \frac{u}{\ell}.
$$

The rescaling factor $e^{m-m'}$ is what preserves contributions accumulated under an older maximum. This is an exact rearrangement apart from ordinary floating-point rounding, not an approximation to softmax.

## 22. Backward recomputation and the final lesson

**Transcript coverage:** lines 3250-3324

### What the lecturer said - transcript only

The backward pass needs information related to the attention probabilities, but storing the full quadratic matrix would surrender the memory benefit. FlashAttention instead recomputes the required score and softmax tiles during backward. The extra arithmetic is worthwhile because it avoids writing and rereading the $N^2$ activation through HBM.

The lecture closes with three broad conclusions:

1. Hardware capability scales, but low-level execution details determine which workloads realize that capability.
2. Current GPU hardware strongly rewards a model of computation built from large matrix multiplications plus carefully managed data movement.
3. Reasoning about coalescing, tiling, fusion, precision, and recomputation explains real algorithms such as FlashAttention and is more useful than cargo-culting rules such as "make every dimension a multiple of 32."

### Additional explanation

FlashAttention is a model example of **algorithm-hardware co-design**. The mathematical function remains attention, but the evaluation order changes to fit the hierarchy:

```text
HBM: store Q, K, V, and final output
  -> load small tiles
SRAM/registers: form scores, update online softmax, accumulate output
  -> write final output only
Backward: reload inputs and recompute local scores instead of storing N^2 activations
```

The phrase "subquadratic memory" must be interpreted carefully. Exact dense attention still performs quadratic pairwise score work in sequence length. FlashAttention avoids **quadratic auxiliary storage and associated HBM traffic** for the attention matrix; it does not make the dense attention FLOP count linear.

---

# Consolidated takeaways

1. GPUs optimize aggregate throughput across many lightweight threads, whereas CPUs devote more hardware to low-latency execution of fewer complex threads.
2. SMs execute blocks, warps schedule groups of 32 threads, and blocks provide the scope for explicit shared-memory reuse.
3. Registers and shared memory are fast but scarce; HBM is large but expensive to access. Performance depends on reuse across this hierarchy.
4. GPUs and TPUs share the same broad accelerator logic but use different unit sizes, scheduling granularity, terminology, and networking.
5. Tensor cores make matrix multiplication far faster than general floating-point work, shaping the architectures that scale well on modern hardware.
6. The roofline model separates memory-bound from compute-bound regimes through arithmetic intensity.
7. Low precision, fusion, recomputation, coalescing, and tiling all improve the ratio of useful work to costly data movement, though each has numerical or resource tradeoffs.
8. Favorable divisibility and alignment help because they match concrete transaction and tile sizes, not because powers of two are inherently magical.
9. Wave quantization explains periodic performance drops when a small shape change creates a poorly utilized extra wave of tiles.
10. FlashAttention is exact dense attention reordered around the memory hierarchy: tile the matmuls, compute softmax online, fuse intermediate work, and recompute tiles in backward.

# Key equations

## Roofline bound

$$
P \leq \min(P_{\max}, BI),
$$

where $I$ is FLOPs per byte, $B$ is memory bandwidth, and $P_{\max}$ is peak compute throughput.

## Simplified tiling reduction

$$
\text{global reads per reused input: } N \longrightarrow \frac{N}{T},
$$

for matrix size $N$ and tile width $T$, giving an idealized factor-$T$ reduction.

## Number of tile waves

$$
W = \left\lceil\frac{T_{\text{tiles}}}{S_{\text{SMs}}}\right\rceil.
$$

## Scaled dot-product attention

$$
S=\frac{QK^\top}{\sqrt d},
\qquad
P=\operatorname{softmax}(S),
\qquad
O=PV.
$$

## Online softmax update

$$
m'=\max(m,\max_j x_j),
$$

$$
\ell'=e^{m-m'}\ell+\sum_j e^{x_j-m'},
$$

$$
u'=e^{m-m'}u+\sum_j e^{x_j-m'}v_j,
\qquad
o=\frac{u}{\ell}.
$$

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Arithmetic intensity | Useful floating-point operations performed per byte moved. |
| Block | A group of GPU threads assigned to one SM and able to share on-chip memory. |
| Coalescing | Combining simultaneous warp memory accesses into a small number of contiguous memory transactions. |
| Compute-bound | Limited mainly by arithmetic throughput rather than memory bandwidth. |
| Control divergence | A warp executing different branch paths serially while masking inactive lanes. |
| DRAM burst | A contiguous group of bytes returned by one underlying memory transaction. |
| FlashAttention | An exact, tiled attention implementation that minimizes HBM traffic through online softmax, fusion, and recomputation. |
| Fusion | Combining several operators into one kernel so intermediates remain near compute. |
| Global memory | Large device memory, normally HBM, visible across blocks but slower than on-chip memory. |
| HBM | High-bandwidth memory attached to an accelerator. |
| Memory-bound | Limited mainly by the rate at which bytes can be supplied or stored. |
| Mixed precision | Using different numerical formats for different inputs, operations, accumulators, or states. |
| MXFP4 / MXFP8 | Microscaling formats that share local scale factors across small blocks of low-bit values. |
| Operator fusion | See fusion; specifically, executing multiple graph operators in one kernel. |
| Recomputation | Recreating a value when needed instead of storing and later loading it. |
| Register | Very fast storage private to a thread. |
| Roofline model | A performance bound combining peak compute and bandwidth times arithmetic intensity. |
| Shared memory | Fast, explicitly managed on-chip memory shared by threads in a block. |
| SIMT | Single instruction, multiple threads: lanes share instruction issue while operating on separate data. |
| SM | Streaming multiprocessor, a GPU's processor-level unit for scheduling blocks and warps. |
| Tensor core (GPU) | A specialized matrix-multiplication circuit inside an SM. |
| TensorCore (TPU) | In TPU terminology, a processor-level unit that contains vector, memory, control, and matrix units. |
| Tile | A submatrix processed as a unit so loaded data can be reused locally. |
| Warp | A hardware scheduling group, normally 32 consecutively numbered GPU threads. |
| Wave quantization | Utilization loss when the number of work tiles requires a sparsely populated final wave across processors. |

# Self-check questions

1. Why is a GPU optimized for throughput rather than the latency of one thread?
2. What is the difference among a thread, block, warp, and SM?
3. Why can threads in one block exchange data cheaply while blocks generally cannot?
4. How do shared memory and L1 cache differ from the programmer's perspective?
5. Why does the phrase "tensor core" refer to different granularities on GPUs and TPUs?
6. Why do tensor cores influence model-architecture research rather than only kernel implementation?
7. State the roofline bound and explain its sloped and flat regions.
8. Why is bytes/FLOP the inverse of arithmetic intensity?
9. When does a branch produce warp divergence, and when is the same branch harmless?
10. Why can low precision improve bandwidth, capacity, and compute throughput by different amounts?
11. What do MXFP8's per-block scales solve, and why do transposes complicate the format?
12. Why is an end-to-end MXFP8 speedup much smaller than the naive bit-width ratio?
13. How does fusion reduce HBM traffic in the sine-and-cosine example?
14. Under what condition can recomputing an activation be faster than loading it?
15. What property of simultaneous addresses makes a warp access coalesced?
16. Why does matrix tiling reduce global reads by roughly a factor of the tile width in the simplified analysis?
17. How can padding from 50,257 to 50,304 dimensions make a matmul faster?
18. Why does the 1792-to-1793 example require 98 tiles and then 120 tiles?
19. Why does ordinary softmax make naive attention tiling difficult?
20. What three running quantities make online softmax possible?
21. What does FlashAttention save, and which quadratic aspect of exact dense attention remains?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-183 | Motivation, performance puzzle, sources, and three-part roadmap | Yes |
| 184-318 | Compute scaling, end of Dennard scaling, and GPU parallelism | Yes |
| 319-472 | CPU latency model versus GPU throughput model | Yes |
| 473-646 | SMs, streaming processors, threads, blocks, warps, and SIMT | Yes |
| 647-731 | Registers, shared memory, caches, global memory, and host memory | Yes |
| 733-954 | TPU comparison, terminology, granularity, and networking distinction | Yes |
| 955-1160 | GPU strengths, shaders, tensor cores, and the memory wall | Yes |
| 1161-1283 | Hardware Q&A, prefill/decode disaggregation, and specialization | Yes |
| 1284-1390 | Matrix-performance puzzle and roofline model | Yes |
| 1391-1491 | SIMT control divergence | Yes |
| 1492-1649 | Low precision, arithmetic intensity, and mixed precision | Yes |
| 1650-1872 | FP8 variants, scaling, MXFP8, transposes, and MXFP4 | Yes |
| 1873-2026 | Quantization and structured-sparsity questions | Yes |
| 2027-2131 | Operator fusion and the sine/cosine example | Yes |
| 2132-2252 | Backpropagation activations and recomputation | Yes |
| 2253-2453 | DRAM bursts, coalescing, matrix layout, and Q&A | Yes |
| 2454-2687 | Tiled matrix multiplication and reuse in shared memory | Yes |
| 2688-2807 | Tile-size complications, alignment, padding, autotuning, and Q&A | Yes |
| 2808-3033 | Matrix mystery, wave quantization, recap, and hardware questions | Yes |
| 3034-3149 | FlashAttention motivation, attention recap, and softmax obstacle | Yes |
| 3150-3249 | Online softmax and tiled forward pass | Yes |
| 3250-3324 | Backward recomputation and lecture conclusions | Yes |
