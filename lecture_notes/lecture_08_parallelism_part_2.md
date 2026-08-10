---
title: "Lecture 8 - Parallelism, Part 2"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 8
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 8 Parallelism.txt"
slide_deck: "../lecture_08.pdf"
status: "complete"
---

# Lecture 8: Parallelism, Part 2

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves the lecturer's claims, examples, qualifications, numerical details, and substantive questions and answers while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, derivation, or study guidance. It is not presented as something the lecturer said.

When the slides resolve notation, supply an exact displayed formula, or conflict with the spoken wording, a separate **Source reconciliation** subsection says so explicitly. The raw transcript's physical line spans are included for auditability. The complete 3,291-line transcript was mapped before the 73-slide deck was rendered and visually inspected page by page.

## Lecture map

The lecture has five connected parts:

1. **Communication and hardware:** collective operations, fast and slow network domains, TPU meshes, GPU tree-style networks, and why modern MoE traffic is changing accelerator interconnects.
2. **Data parallelism:** naive distributed data parallel, optimizer-state memory, ZeRO stages 1-3, and how FSDP overlaps parameter communication with computation.
3. **Model and activation parallelism:** pipeline, tensor, sequence, expert, and context parallelism, including their communication patterns and memory effects.
4. **Composing strategies:** roofline reasoning, 3D/4D parallelism, topology-aware rules of thumb, and questions about unusual architectures.
5. **Evidence from real runs:** OLMo, DeepSeek, Yi, Llama 3, Gemma 2, Mixtral, Nemotron, and Qwen configurations.

---

# Part I - Communication and hardware foundations

## 1. Scope: why large-model training needs many accelerators

**Transcript coverage:** lines 1-104

### What the lecturer said - transcript only

The previous lecture covered the mechanics of parallelism. This lecture focuses on the practical knowledge needed to train very large language models on very large clusters. Modern training can use four or even more parallelization dimensions at once. At the largest scales, most or all of them become necessary. A course assignment will ask students to choose an optimal strategy for a given model and network topology, and the lecture will end with published configurations from recent training runs.

Parallelism addresses two limits:

- **Compute:** one chip cannot supply the exaflop-scale compute available from a large supercomputer.
- **Memory:** large models and their training state do not fit in one accelerator's memory, so the state must be divided somehow.

The central systems distinction is between very fast **intra-node** links and slower **inter-node** links. Communication-heavy methods belong on the fast links. Methods used across slower links must be designed to respect their lower bandwidth and higher latency.

### Additional explanation

Parallel training is not one technique but a placement problem. A useful mental model is to treat compute, memory, batch size, and network bandwidth as separate resources. Each parallelism dimension spends some resources to save others. For example, data parallelism spends batch size and communicates gradients; tensor parallelism spends high-bandwidth collective communication; pipeline parallelism spends microbatches and accepts scheduling bubbles.

The network hierarchy matters because a configuration that is efficient inside an eight-GPU server can become communication-bound when extended across racks. The same algebra may be valid everywhere, while its performance depends on where the communicating ranks are placed.

## 2. Collective operations and the key all-reduce identity

**Transcript coverage:** lines 105-185

### What the lecturer said - transcript only

The discussion stays at the algorithmic level of collective communication rather than individual packets. Operations such as all-reduce, all-gather, and reduce-scatter will be used to account for communication. Efficient implementations ultimately reach down to the hardware, but the lecture treats each collective as a primitive.

One identity is especially important:

$$
\operatorname{all\mbox{-}reduce}
\equiv
\operatorname{reduce\mbox{-}scatter}
+
\operatorname{all\mbox{-}gather}
$$

in the bandwidth-limited accounting used here. Later, an algorithm replaces one all-reduce with those two phases. Because their total communication cost is equivalent, the change can deliver memory savings without increasing the leading communication volume.

**Question:** Why emphasize this particular decomposition when other decompositions exist?

**Answer:** Other decompositions are possible, but reduce-scatter followed by all-gather is the one needed for a later algorithm, so the lecturer highlights it in advance.

### Additional explanation

For a vector partitioned into rank-sized chunks, reduce-scatter sums corresponding chunks across ranks and leaves one reduced chunk on each rank. All-gather then distributes those reduced chunks so every rank reconstructs the full reduced vector. This is precisely the result of an all-reduce.

"Equivalent" here is a bandwidth-level statement, not a promise of identical latency. Two collectives can involve more launch and synchronization overhead than one fused collective, especially for small messages. The equivalence becomes most useful when messages are large enough that bytes moved dominate fixed overhead.

## 3. TPU meshes and GPU tree-style networks

**Transcript coverage:** lines 186-324

### What the lecturer said - transcript only

Although most of the lecture is hardware-agnostic, hardware topology changes which parallelism strategies are attractive.

Traditional TPU systems use a toroidal mesh. Chips communicate with fixed neighbors on a grid whose edges wrap around, with the physical system extending this idea into three dimensions. Each chip keeps the same number of neighbors as the system grows, which makes the design simple and scalable. For regular, local, and predictable traffic, the links can be made strong and cost-effective.

GPU clusters follow a more all-to-all philosophy. GPUs connect very quickly at the lowest level, groups of GPUs form nodes or pods, and higher-level leaf and spine switches connect those groups. As the cluster grows, the tree becomes larger and communication paths become more complicated. This design is more flexible when destinations are random or stochastic.

The workload determines which philosophy is favorable. Dense tensor-parallel computation has regular partitions and maps naturally to a mesh. Mixture-of-experts traffic routes different tokens to different experts and is less predictable, which favors flexible switched connectivity. The lecturer cites a discussion by Bill Dally and Jeff Dean to support this workload-dependent comparison.

### Additional explanation

A mesh has bounded physical degree: adding more chips does not require every chip to gain more direct links. Its regularity makes routing and collective schedules predictable. The cost is that distant traffic may traverse multiple hops.

A fat-tree-style network uses switches to provide high bisection bandwidth across larger groups. It can serve arbitrary destinations more naturally, but building and powering enough switching capacity is expensive. Neither topology is universally superior; locality, traffic entropy, and the collective pattern decide which one is better.

## 4. Topologies evolve with workloads: TPU8 and Huawei Ascend

**Transcript coverage:** lines 325-473

### What the lecturer said - transcript only

The lecturer had planned to leave the TPU/GPU contrast there, but Google announced TPU8i and TPU8t that morning. TPU8i appears closer to a tree or all-to-all topology. This makes sense because modern models are often MoEs, and both inference and training must route tokens to many experts. TPU8t also adds a switched cross-rack scale-out network called Virgo. The lecturer interprets this as convergent evolution: accelerator networks are changing in response to model workloads.

Huawei's Ascend 910 system illustrates a different tradeoff. An individual chip is substantially slower than an NVIDIA H200 on matrix multiplication and other measures. Huawei compensates by connecting 384 chips in a rack with extensive optical switching. Enough chips and communication can offset weaker per-chip performance, but the comparison shown has roughly four times the power consumption of an equivalent NVIDIA system. The lesson is that some problems can be brute-forced with scale-out connectivity and power, while a design optimized for manufacturing and energy ends up elsewhere.

The unit of compute should therefore be thought of as the whole data center, not one GPU. The goal is to control memory and compute across that system while keeping resources utilized, using collective-based algorithms to coordinate it.

### Additional explanation

This is an example of co-design: model architecture changes communication, and communication hardware then changes which model architectures are economical. Dense models reward predictable collective traffic; sparse expert models reward high-bandwidth permutation and all-to-all traffic.

Scaling out weaker devices can raise aggregate compute, but it also raises failure surface, power, cooling, switching, and software complexity. Peak FLOPs alone therefore do not describe a training system. Effective throughput per watt and per unit of capital depends on the whole cluster.

---

# Part II - Data parallelism and ZeRO/FSDP

## 5. Parallelism taxonomy and naive data parallelism

**Transcript coverage:** lines 474-569

### What the lecturer said - transcript only

The lecture groups methods into two broad families. Data parallelism distributes training examples across devices. Model parallelism divides the model itself. The boundary is not perfectly clean because some nominally data-parallel methods also shard model state. Both families are needed to scale compute and memory.

For naive synchronous data parallelism, temporarily imagine plain SGD. A global batch of size $B$ is split across $M$ machines, so each receives $B/M$ examples. Every machine computes gradients on its local examples, and the gradients are synchronized and summed before the update.

This gives ideal compute scaling as long as every GPU receives enough work. The leading communication volume is about twice the parameter count per batch because the full gradient is all-reduced. It provides no model-state memory scaling: every GPU holds a complete model copy, and the lecturer also describes the activation storage as replicated for its local computation.

### Additional explanation

If $P$ denotes the number of parameters, a ring-style all-reduce moves approximately

$$
2P\frac{M-1}{M}
$$

elements per rank, which approaches $2P$ for large $M$. The lecture rounds this to $2P$ to compare strategies cleanly.

The condition "enough work per GPU" matters. If $B/M$ becomes too small, fixed kernel and collective overheads dominate and matrix multiplications become too small to use the device efficiently.

## 6. Training-state memory is much larger than the model weights

**Transcript coverage:** lines 570-697

### What the lecturer said - transcript only

Naive data parallelism solves some compute scaling but leaves a severe memory problem. Training stores more than the model parameters:

- low-precision model parameters;
- a gradient buffer;
- possibly a higher-precision master or accumulation copy;
- Adam's first-moment estimate;
- Adam's second-moment estimate.

The precision choices change the exact total, but a rough rule is five weight-sized objects and about 16 bytes per parameter. The persistent optimizer state can be the largest component. Naive data parallelism replicates all of it on every GPU.

The natural response is to distribute increasingly many components. First shard optimizer state, then gradients, and finally parameters. In the slide's example, progressively sharding these objects lowers per-rank memory from 120 GB to 1.9 GB. The surprising claim is that much of this saving can be obtained with no increase in leading communication volume.

### Source reconciliation

The slide makes the 16-byte example explicit: 2 bytes for FP16/BF16 parameters, 2 for FP16/BF16 gradients, 4 for FP32 master weights, and 4 bytes each for FP32 Adam first and second moments. It notes that either moment may instead use 2 bytes in a lower-precision variant.

### Additional explanation

It is helpful to distinguish three memory classes:

1. **Parameters:** the values used by the forward and backward passes.
2. **Gradients:** one derivative value per parameter before or during the update.
3. **Optimizer state:** persistent values such as master weights and Adam moments.

Sharding only optimizer state can already save substantial memory because it is often the largest class. The ZeRO stages are ordered by how aggressively they shard these classes.

## 7. ZeRO stage 1: shard optimizer state for the cost of DDP

**Transcript coverage:** lines 698-824

### What the lecturer said - transcript only

ZeRO stage 1 shards only optimizer state. Every GPU still holds all parameters and all gradients, but each GPU is responsible for updating only one parameter slice and stores the optimizer state for that slice.

The update proceeds as follows:

1. Every worker computes a full gradient from its local data.
2. A reduce-scatter sums the gradients and delivers each reduced parameter slice to the worker responsible for it.
3. Each worker updates its slice using its local parameters, reduced gradients, and optimizer state.
4. An all-gather distributes all updated parameter slices so every worker again has the complete parameters.

Naive DDP uses one gradient all-reduce, with communication cost about $2P$. ZeRO stage 1 uses one reduce-scatter and one all-gather. Because those have the same leading bandwidth cost as one all-reduce, ZeRO stage 1 has the same communication volume while dividing optimizer-state memory across the GPUs. The lecturer calls this a free memory saving.

### Source reconciliation

With parameter count $\Psi$, $N_d$ data-parallel ranks, and $K\Psi$ bytes of optimizer state, the slide gives

$$
M_{\text{baseline}}=(2+2+K)\Psi,
$$

$$
M_{\text{ZeRO-1}}=2\Psi+2\Psi+\frac{K\Psi}{N_d}.
$$

The two unsharded $2\Psi$ terms are low-precision parameters and gradients.

### Additional explanation

ZeRO stage 1 changes ownership, not the mathematical update. Each rank becomes the optimizer for one shard, but after the all-gather every rank sees the same updated model. The important systems observation is that the DDP all-reduce already contains both a reduction phase and a distribution phase; ZeRO uses the intermediate sharded result rather than discarding it.

The saving is "free" only in the leading bandwidth model. Real implementations may differ in latency, launch count, buffering, or overlap quality.

## 8. ZeRO stage 2: keep gradients sharded as well

**Transcript coverage:** lines 825-877

### What the lecturer said - transcript only

ZeRO stage 2 additionally shards gradients. A rank must still compute derivatives for the whole model because it processes the whole model on its local data, but it need not materialize a persistent full gradient vector.

During the backward sweep, as soon as a layer's gradients are computed, they are reduced and sent to the worker that owns that slice. Once a gradient is no longer needed by the backward graph, it is freed immediately. At the end, each worker updates its own slice and the updated parameters are all-gathered. Incremental reduce-scatter has the same total leading communication as reducing all gradients at once, so this further memory saving adds no leading bandwidth cost.

### Source reconciliation

The slide summarizes the per-rank memory as

$$
M_{\text{ZeRO-2}}
=
2\Psi+
\frac{(2+K)\Psi}{N_d}.
$$

The full low-precision parameters remain replicated, while gradients and optimizer state are sharded.

### Additional explanation

The phrase "compute a full gradient" and the phrase "never instantiate a full gradient vector" are compatible. Autodiff visits every parameter and computes every local derivative, but a hook can immediately launch the reduction for one bucket and release that bucket. The peak resident gradient memory is therefore bounded by a small number of buckets rather than the entire model.

## 9. ZeRO stage 3/FSDP: gather parameters only when needed

**Transcript coverage:** lines 878-1000

### What the lecturer said - transcript only

ZeRO stage 3, also called fully sharded data parallelism (FSDP), shards parameters in addition to gradients and optimizer state. At rest, each rank sees only a slice of all three.

The method applies the same incremental idea used for gradients to parameters:

1. All-gather the parameters for the layer about to run.
2. Execute that layer's forward pass.
3. Free the full gathered parameters.
4. During backward, all-gather that layer's parameters again because its saved activations and weights are needed to compute derivatives.
5. Run backward, reduce-scatter the gradients immediately, and free the full parameters again.
6. Repeat for the remaining layers, then update the locally owned parameter shards.

One training step therefore uses two all-gathers and one reduce-scatter over the model parameters. This is one all-gather more than DDP or ZeRO stages 1 and 2. It also creates many layer-level communications, so at first it appears expensive.

### Source reconciliation

The slide gives the ideal resident model-state memory as

$$
M_{\text{ZeRO-3}}
=
\frac{(2+2+K)\Psi}{N_d}.
$$

Transiently gathered layers and communication buffers add to this idealized lower-level accounting.

### Additional explanation

FSDP shards storage while preserving data-parallel execution: every rank still performs the entire forward and backward computation for its own examples. This distinguishes it from pipeline parallelism, where different ranks permanently own different layer ranges.

The size of an FSDP wrapping unit matters. Very small units reduce peak gathered memory but create many small, latency-sensitive collectives. Very large units improve collective efficiency but increase transient memory. Implementations bucket or wrap layers to balance those effects.

## 10. How FSDP hides its extra communication

**Transcript coverage:** lines 1001-1121

### What the lecturer said - transcript only

FSDP relies on two systems ideas:

1. Request parameters incrementally and free them immediately after use.
2. Overlap communication with computation.

The illustrated execution has separate CPU scheduling, GPU compute, and GPU communication streams. Once layer 0 has been gathered, its forward computation begins while layer 1 is gathered. Forward 1 can then overlap the gathering of layer 2, and the pattern continues through forward and backward. Parameter frees and prefetches are similarly scheduled around useful computation.

If a layer's computation lasts longer than the communication needed for the next layer and the network is fast enough, most communication can be hidden. Some bubbles remain, but practical FSDP utilization can stay close to single-GPU utilization. The lecturer calls the result remarkable and notes that the assignment will implement an FSDP module wrapper built from repeated gather, compute, free, and backward operations.

Naive DDP, ZeRO stage 1, and ZeRO stage 2 communicate about $2P$ elements per step. ZeRO stage 3 communicates about $3P$, but overlap can mask much of the extra all-gather.

### Source reconciliation

Near the end of this explanation, the transcript says the overlap improves memory "without paying for almost any amount of memory use." In context and on the slides, the intended claim is that it improves memory use without paying much exposed **communication overhead**. The literal phrase would otherwise contradict the whole section.

### Additional explanation

For a prefetched layer $\ell+1$, the ideal hiding condition is approximately

$$
T_{\text{compute},\ell}
\ge
T_{\text{all-gather},\ell+1}.
$$

If this holds steadily, communication sits beneath useful compute on the timeline. If layers are too small, bandwidth is low, or latency dominates, prefetching cannot hide the collectives and the training step develops exposed gaps.

## 11. FSDP questions and concrete memory gains

**Transcript coverage:** lines 1122-1227

### What the lecturer said - transcript only

**Question:** Does backward pass a gradient from one GPU's layer to a previous GPU's layer?

**Answer:** That description is closer to pipeline parallelism. In FSDP every GPU traverses the entire model from beginning to end. No GPU permanently stores the full parameters. Instead, each requests a full layer when needed, computes it, and frees it. The mathematical computation is the same as ordinary data parallelism; parameter availability is managed between operations.

**Question:** Does every GPU hold a part of every layer, with the all-gather reconstructing it?

**Answer:** Yes. Parameters, gradients, and optimizer state are all sharded across the ranks.

**Question:** Why does communication cost not multiply by the number of layers?

**Answer:** The number of collective calls does multiply by the number of layers, but every call handles only one layer's smaller piece. Summed over the network, those pieces add up to the full model-sized communication volume.

In the A100 example, a baseline configuration cannot fit even a 7-billion-parameter model, while ZeRO stage 3 can fit roughly a 50-billion-parameter model. When asked about H100/H200-class devices, the lecturer declines to do the exact mental arithmetic but says the capacity scales linearly with available memory; the underlying calculation does not change.

### Source reconciliation

The slide's idealized example uses eight 80 GB A100 GPUs and 12 bytes per parameter. It lists maximum model sizes of approximately 6.66B for the baseline, 16B for ZeRO-1, 24.62B for ZeRO-2, and 53.33B for ZeRO-3. These numbers exclude other memory consumers such as activations and buffers.

### Additional explanation

The distinction between **operation count** and **byte volume** answers the layer question. Splitting one $P$-element collective into $L$ collectives of $P/L$ elements preserves total bytes but usually increases latency overhead. This is why practical FSDP chooses reasonably large buckets and overlaps them with compute.

The capacity figures are upper bounds from static model-state accounting. Real maximum trainable size is smaller once activations, temporary workspaces, fragmentation, and peak gathered parameters are included.

## 12. Why data parallelism is not enough

**Transcript coverage:** lines 1228-1306

### What the lecturer said - transcript only

FSDP is clean and effective, but two limits require other methods.

First, data parallelism consumes global batch size. With batch size 8, at most eight ranks can each receive a distinct example in a synchronous step. One cannot increase batch size without limit because beyond a **critical batch size**, another example in the same gradient estimate contributes less training progress than spending that computation on another SGD step. This creates a tradeoff between leaving GPUs idle at small batches and accepting optimization inefficiency at very large batches.

Second, sharding model state does not solve every memory problem. The lecturer says that the earlier ZeRO stages still leave important replicated memory, and that even the fully sharded method does not reduce activation memory. More fine-grained ways of cutting the model are needed.

### Source reconciliation

The spoken wording at lines 1288-1293 says that ZeRO stages 1 and 2 "don't really let you scale memory at all" and then says "ZeRO stage 2 lets you cut up the parameter memory." This conflicts with both the preceding explanation and the slide. The slide's intended distinction is:

- ZeRO-1 and ZeRO-2 reduce optimizer/gradient memory but keep parameters replicated, so they do not provide full parameter-memory scaling.
- ZeRO-3 shards parameters, gradients, and optimizer state, but it does not reduce activation memory.

### Additional explanation

The critical-batch argument is about statistical efficiency, not only hardware efficiency. Increasing batch size can raise tokens processed per second while increasing the tokens needed to reach a target loss. A training configuration must therefore optimize time-to-quality, not throughput alone.

FSDP also couples the data-parallel degree to the batch layout. Model-parallel methods can add ranks without assigning each one a separate batch shard, which is why they are needed after useful data parallelism has been exhausted.

---

# Part III - Model and activation parallelism

## 13. Model parallelism and the pipeline bubble

**Transcript coverage:** lines 1307-1452

### What the lecturer said - transcript only

Model parallelism also splits parameters across GPUs, but unlike FSDP it communicates activations. If one layer resides on GPU 0 and the next on GPU 1, the activation between those layers must cross the link. The lecture considers three cuts:

- pipeline parallelism cuts by layers or depth;
- tensor parallelism cuts matrices or width;
- expert parallelism assigns experts to different devices.

Naively putting successive layer groups on four GPUs gives terrible utilization. One GPU runs the first stage and then waits while later stages run. The same serialization occurs in reverse during backward. Most devices are idle; the empty portion of the schedule is the **pipeline bubble**.

The remedy is to divide a batch into microbatches. As soon as stage 0 hands the first microbatch to stage 1, stage 0 begins the second. Forward and backward work then flow through the stages like a pipeline. The bubble shrinks as the number of microbatches grows, so batch size is being spent to keep pipeline stages occupied.

### Source reconciliation

The transcript describes the utilization dependence loosely. The slide states the exact ratio of bubble time to useful compute as

$$
\frac{n_{\text{stages}}-1}{n_{\text{micro}}}.
$$

Thus increasing the number of microbatches drives the relative bubble toward zero.

### Additional explanation

Microbatching does not change the global batch if a fixed batch is merely subdivided, but more microbatches mean smaller per-microbatch matrix multiplications. There is therefore a second tradeoff: too few microbatches leave bubbles; too many can reduce kernel efficiency or raise scheduling overhead.

Pipeline stages must also be balanced. Even with many microbatches, the slowest stage sets the cadence, and faster stages wait for it.

## 14. Why pipeline parallelism is still useful

**Transcript coverage:** lines 1453-1554

### What the lecturer said - transcript only

Pipeline code is notoriously where otherwise understandable parallel training systems become complicated. It remains important for two reasons:

1. It partitions model memory by assigning different layers to different stages, and it composes with data parallelism.
2. Its communication can be much lighter than FSDP or tensor parallelism. A boundary sends an activation tensor of approximate size $bsh$, where $b$ is microbatch size, $s$ sequence length, and $h$ hidden size. This is point-to-point traffic between adjacent stages rather than a model-wide collective.

Pipeline parallelism is therefore placed on the slowest links, such as links across nodes, pods, or even data centers. The cited Megatron sweeps show that large batches can keep utilization close to the no-pipeline case even as pipeline degree rises, while small batches cause rapid degradation.

**Question:** What specifically makes pipeline communication better than FSDP communication?

**Answer:** It sends an activation-sized $bsh$ object, which is usually smaller than a whole parameter matrix, and the communication is between neighboring stages.

### Additional explanation

Pipeline traffic is attractive across slow links because it does not require every rank to participate in every transfer. The relevant boundary activation is sent once forward and its gradient once backward. In contrast, FSDP repeatedly materializes parameters through collectives, and tensor parallelism inserts collectives inside each Transformer block.

The claim that $bsh$ is smaller is workload-dependent. Very long sequences or large microbatches can make boundary activations large, and pipeline schedules may retain multiple microbatch activations. It remains a rule of thumb rather than a universal inequality.

## 15. Better schedules and zero-bubble pipeline parallelism

**Transcript coverage:** lines 1555-1677

### What the lecturer said - transcript only

Researchers reduce bubbles with more elaborate schedules that interleave forward and backward work from different microbatches. DeepSeek provides an example of improving utilization by carefully choosing when each stage runs each pass.

Zero-bubble pipeline parallelism uses a deeper observation about backpropagation. Backward work contains two distinct computations:

- $B$: propagate derivatives with respect to activations toward the preceding stage;
- $W$: compute gradients with respect to the current stage's weights.

The $B$ work lies on the pipeline's critical dependency chain because the previous stage cannot continue until it receives the activation derivative. The $W$ work is a leaf computation and can be postponed. A schedule therefore performs $B$ as early as possible and fills otherwise idle gaps with deferred $W$ work. Depending on the relative $B$ and $W$ workloads, this can almost completely fill the pipeline, although the implementation is substantially more complex.

### Additional explanation

For a layer $z=Wx$ followed by later computation, backpropagation forms both

$$
\frac{\partial \mathcal{L}}{\partial x}
=
W^\top
\frac{\partial \mathcal{L}}{\partial z}
$$

and

$$
\frac{\partial \mathcal{L}}{\partial W}
=
\frac{\partial \mathcal{L}}{\partial z}x^\top.
$$

The first value must travel to the previous stage; the second stays local until the optimizer step. Separating them exposes scheduling freedom that an undivided "backward" block hides.

## 16. Tensor parallelism: split width and exploit forward/backward duality

**Transcript coverage:** lines 1678-1801

### What the lecturer said - transcript only

Pipeline parallelism cuts the model along depth. Tensor parallelism cuts it along width. The underlying observation is that a matrix multiplication can be decomposed into smaller matrix multiplications whose partial results are combined.

For an MLP of the form

$$
Y=\operatorname{GeLU}(XA),
\qquad
Z=YB,
$$

split $A$ by columns and $B$ by rows. Each device receives the same input $X$, computes its local $XA_i$, applies GeLU, and computes $Y_iB_i$. The partial outputs are then summed.

There is an important forward/backward duality. In the forward pass, the input-side function $f$ is the identity or replication, while the output-side function $g$ is an all-reduce. In backward, the roles reverse: $g$ is the identity for incoming partial derivatives, and $f$ all-reduces the partial input gradients.

Within a Transformer block, the input projections of the MLP and the $Q/K/V$ attention projections are split column-wise. The MLP down-projection and attention output projection are split row-wise. Small operations such as layer normalization, nonlinearities, and MoE routers are replicated because splitting them is not worth the overhead.

### Additional explanation

The paired column/row partition avoids gathering the large intermediate activation between the two MLP matrices. Each rank keeps its own intermediate slice; only the final output needs reduction. This is why tensor-parallel layouts are designed across whole subgraphs rather than treating each matrix independently.

The backward duality follows from linear algebra. A row-partitioned forward sum becomes replicated incoming gradients, while a replicated forward input becomes a sum of per-rank input-gradient contributions.

## 17. Where tensor parallelism belongs and how it compares with pipelines

**Transcript coverage:** lines 1802-1904

### What the lecturer said - transcript only

Tensor parallelism is communication-hungry. Collectives occur around matrix multiplications in every layer, the communicated objects are activation-sized, and the synchronization is frequent. On GPU systems it is normally kept within the fastest interconnect domain, commonly the eight GPUs inside one node. Going beyond that boundary reaches slower inter-node links and causes a large performance drop.

TPUs differ because a large regular mesh can provide high bandwidth for predictable tensor-parallel traffic without the same sharp eight-GPU boundary. TPU users can therefore employ larger tensor-parallel domains, changing the balance between tensor and pipeline parallelism.

Compared with pipeline parallelism, tensor parallelism has no inherent pipeline bubble, can reach full utilization when the network is fast enough, and is conceptually a relatively simple matrix partition. It also does not require a large batch merely to fill a pipeline. Its disadvantage is much larger and more frequent collective communication rather than $bsh$ point-to-point transfers. The lecturer's rule is to use tensor parallelism on high-speed links and pipeline parallelism elsewhere.

### Source reconciliation

The slide quantifies tensor-parallel communication per layer as approximately

$$
8bsh\left(\frac{n_{\text{devices}}-1}{n_{\text{devices}}}\right),
$$

using all-reduce communication, while pipeline parallelism sends about $bsh$ point-to-point per microbatch. The spoken transcript gives only a rough verbal multiple at this point.

### Additional explanation

Tensor parallelism trades larger GEMMs for smaller per-rank GEMMs. Even with an excellent network, an excessive tensor degree can reduce arithmetic efficiency because each local matrix becomes too narrow. This is a second reason, beyond cross-node bandwidth, that practical GPU tensor-parallel degree often stops near the node size.

The "no bubble" advantage does not mean no waiting. A blocking all-reduce can still stall every rank. It means tensor parallelism lacks pipeline fill and drain bubbles as a structural scheduling cost.

---

# Part IV - Activation, expert, and context parallelism

## 18. Dynamic memory and the activation-memory baseline

**Transcript coverage:** lines 1905-2025

### What the lecturer said - transcript only

The naive picture of training memory as parameter storage is incomplete. Optimizer state adds a large persistent allocation, but the actual memory profile also contains dynamic allocations. Forward computation stores activations, and backward computation begins creating gradients while many activations are still live. Peak memory therefore tends to occur shortly after backward starts, when substantial activation and gradient memory coexist.

This matters increasingly at modern model and sequence sizes. In the cited comparison, activation memory can exceed parameter and optimizer-state memory by a wide margin. A model-state-only method therefore cannot solve the full memory problem.

If every activation is retained, the lecturer gives the following rule of thumb for one Transformer layer:

$$
M_{\text{act, baseline}}
=
sbh\left(34+5\frac{as}{h}\right),
$$

where $s$ is sequence length, $b$ is microbatch size, $h$ is hidden size, and $a$ is the number of attention heads. The $sbh$ dependence is fundamental: the computation has values for every sequence position, batch element, and hidden dimension. The additional $5as/h$ contribution comes from quadratic attention intermediates, including dropout-related storage. FlashAttention-style selective recomputation can remove this quadratic storage term.

### Source reconciliation

The slide defines the symbols used across the activation formulas: $a$ for attention heads, $b$ for microbatch size, $h$ for hidden dimension, $L$ for Transformer layers, $p$ for pipeline-parallel degree, $s$ for sequence length, $t$ for tensor-parallel degree, and $v$ for vocabulary size. The transcript says "34 times $sbh$ plus 5 times $as/h$" in a way that could be parsed ambiguously; the displayed slide resolves the intended grouping as $sbh(34+5as/h)$.

### Additional explanation

Activation memory is a lifetime problem, not just a tensor-size problem. A tensor contributes to peak memory only while it remains live. Backpropagation normally needs inputs or outputs from the forward pass, so retaining every intermediate creates a large live set. Checkpointing changes that live set by discarding selected intermediates and recomputing them later.

The formula is best treated as an architecture- and implementation-specific accounting rule, not a universal constant. The important structural terms are more durable:

- an $O(sbh)$ component for per-token hidden activations;
- an $O(as^2b)$ component for materialized attention scores or probabilities;
- a multiplier from the number and precision of saved tensors.

FlashAttention avoids materializing the full attention matrix in high-bandwidth memory, which is why the quadratic storage term can disappear even though exact attention computation remains quadratic in arithmetic work.

## 19. Tensor parallelism leaves residual activations; sequence parallelism shards them

**Transcript coverage:** lines 2026-2207

### What the lecturer said - transcript only

Tensor parallelism reduces activation memory associated with the large matrix multiplications in the MLP and attention blocks. Of the baseline coefficient 34, the lecturer attributes 24 to MLP-related activations. Tensor parallelism of degree $t$ divides that contribution by $t$ and likewise divides the attention-head storage term by $t$.

It does not, however, divide every activation. Layer normalization, dropout, residual inputs, and the inputs retained for attention and MLP backward passes are replicated. These lightweight operations leave an unsharded $10sbh$ term, so simply increasing tensor parallelism cannot make total activation memory fall linearly forever. Under tensor parallelism, the per-layer accounting becomes

$$
M_{\text{act, TP}}
=
sbh\left(10+\frac{24}{t}+5\frac{as}{ht}\right).
$$

The remedy is **sequence parallelism**. Despite its name, the lecturer distinguishes it from context parallelism. Sequence parallelism is usually paired with tensor parallelism and shards the inexpensive pointwise operations along the sequence axis. Activations for those operations remain sharded while they are not needed and are materialized through all-gather only when required. The corresponding reverse-mode operation uses reduce-scatter; in backward, the roles of the two communication functions reverse.

Combining tensor and sequence parallelism divides the entire coefficient by $t$:

$$
M_{\text{act, TP+SP}}
=
sbh\left(\frac{34}{t}+5\frac{as}{ht}\right).
$$

Selective activation recomputation can additionally remove the attention-storage term, giving the practical lower-bound rule of thumb

$$
M_{\text{act, TP+SP+recompute}}
=
sbh\frac{34}{t}.
$$

The lecturer recommends remembering this expression when estimating whether a training configuration will fit. It must be added to parameter, gradient, and optimizer-state memory.

### Source reconciliation

The slide deck provides a five-row summary that makes the progression explicit:

| Configuration | Activation memory per Transformer layer |
|---|---:|
| No parallelism | $sbh(34+5as/h)$ |
| Tensor parallel | $sbh(10+24/t+5as/(ht))$ |
| Tensor + sequence parallel | $sbh(34/t+5as/(ht))$ |
| Tensor parallel + selective recomputation | $sbh(10+24/t)$ |
| Tensor + sequence parallel + selective recomputation | $sbh(34/t)$ |

The slides further resolve the residual coefficient 10 as $4sbh$ for LayerNorm, $2sbh$ for dropout, and $4sbh$ for the saved inputs to attention and the MLP.

### Additional explanation

Sequence parallelism is an example of **layout switching**. Tensor-parallel matrix multiplications want a width-based layout, while inexpensive pointwise operations can execute independently on sequence shards. All-gather and reduce-scatter convert between those layouts without changing the mathematical function.

The analogy to FSDP is useful but the sharded object differs:

- FSDP stores parameter shards and gathers weights when a layer runs.
- Sequence parallelism stores activation shards and gathers the layout needed by a local operation.

Both exploit a common pattern: keep a large object sharded during inactive periods, materialize it near its use, and release or reshard it immediately afterward.

## 20. Recomputation choices and their cost

**Transcript coverage:** lines 2208-2247

### What the lecturer said - transcript only

An audience member asks why the $24/t$ MLP component is not also eliminated through recomputation. The lecturer says it can be recomputed, but doing so means running the MLP again during backward and is usually too computationally expensive. Selective recomputation of attention intermediates is generally cheaper and is the more attractive target.

The exchange ends with a somewhat compressed remark about the quadratic attention cost being painful. The substantive distinction is that more aggressive checkpointing can save additional memory, but it buys that memory with extra arithmetic, and not every recomputation target has a favorable tradeoff.

### Source reconciliation

The final few spoken sentences in this exchange are grammatically ambiguous: they can sound as though quadratic attention recomputation is both cheaper and something one does not want to pay for. The slides resolve the intended accounting result only: **selective** activation recomputation removes the $5as/(ht)$ stored-attention term, while the $24/t$ MLP term remains in the presented practical formula.

### Additional explanation

Checkpointing policies form a spectrum:

- Save everything: minimum recomputation, maximum activation memory.
- Selectively recompute cheap or memory-heavy intermediates: a favorable middle ground.
- Checkpoint only layer inputs: rerun nearly the whole layer during backward, saving more memory at a larger compute cost.

The best choice depends on the ratio

$$
\frac{\text{bytes of peak memory saved}}{\text{additional FLOPs and exposed time}}.
$$

An intermediate that is large to store but cheap to regenerate is an ideal checkpointing target. A large MLP GEMM is expensive to rerun, so discarding its useful intermediates may reduce throughput more than it helps.

## 21. Expert parallelism and why MoEs prefer it over tensor parallelism

**Transcript coverage:** lines 2248-2424

### What the lecturer said - transcript only

Expert parallelism is the last standard primitive introduced in detail. Because mixture-of-experts models are now common, their feed-forward experts can be divided across devices. The lecturer treats expert parallelism as analogous to tensor parallelism: both split the MLP side of the network, require high-bandwidth communication, and reduce activation or model-state pressure.

For an MoE layer, expert parallelism is normally preferred over tensor parallelism. Tensor parallelism cuts each matrix into smaller matrices; sufficiently fine cuts reduce GEMM size and hurt GPU utilization. Expert parallelism instead leaves larger local expert matrices intact and routes only the token activations assigned to them. If the experts are already distinct, placing different experts on different devices uses that structure directly.

The method is nevertheless difficult to implement efficiently. Every MoE layer performs latency-sensitive all-to-all dispatch and combine operations. Computation cannot begin until routed tokens arrive, so low dispatch latency is critical. The lecturer points to a DeepSeek expert-routing library, transcribed as "DPP," and NVIDIA's HybridEP as examples of highly specialized low-level implementations. DeepSeek engineers reportedly went as far as using undocumented PTX-level instructions to extract additional networking performance.

The broader point is that frontier parallelism efficiency can depend on hardware-level routing, operation fusion, and undocumented accelerator behavior, not just the high-level partition.

### Source reconciliation

The transcript's "DPP" is likely a transcription error for **DeepEP**, the name associated with DeepSeek's expert-parallel communication work; the slide deck does not print the library name, so this correction remains editorial rather than slide-verified.

The Megatron guidance shown on the slides lists four reasons to prefer EP over TP for expert layers: larger local GEMMs, lower communication for MoE layers, a simpler computation graph that is easier to overlap, and elimination of local token permutation when the EP degree equals the number of experts. Its example says `EP8 x TP1` outperforms `EP4 x TP2` for Mixtral 8x7B.

### Additional explanation

In tensor parallelism, every participating rank works on every token but owns only part of the matrix. In expert parallelism, a rank owns complete experts but receives only the tokens routed to those experts. That difference preserves large local matrix shapes, which is often better for accelerator efficiency.

The hard part is that token counts are data-dependent. An expert may receive too many or too few tokens, causing load imbalance or capacity overflow. Efficient EP therefore depends on the router, load-balancing loss, token packing, all-to-all implementation, and overlap schedule as a coupled system.

## 22. Combining expert, data, and tensor parallelism

**Transcript coverage:** lines 2425-2539

### What the lecturer said - transcript only

Most parallelism methods can be combined like building blocks, but expert parallelism introduces extra group-structure constraints. In a common naive layout, data-parallel replicas and expert-parallel splits share the same larger group: data is split over the data-parallel ranks, while experts are sharded within that domain and tokens are routed among those ranks. This relationship limits how independently the data- and expert-parallel degrees can be chosen.

There is a second imbalance. Ordinary MoEs replace MLPs with experts but leave attention dense. High tensor parallelism is useful for splitting attention, while low tensor parallelism is preferable for the expert MLPs because combining high TP with high EP makes the local expert matrices too small. The same model therefore wants different tensor-parallel degrees in its attention and MoE sublayers.

Recent systems decouple those choices. Attention receives one tensor-parallel layout, while MoE layers receive a different expert/tensor/data layout. This is more complicated but avoids forcing a single parallel degree onto subgraphs with different communication and matrix-shape needs.

### Source reconciliation

The slide on naive composition states that DP commonly shares replicas with EP splits and summarizes the constraint as $\mathrm{EP}<\mathrm{DP}$. Its next line reads "DP and TP can interact badly to lower utilization," but the slide title, diagram, following slide, and spoken explanation all concern **EP and TP**. The notes therefore treat "DP" in that sentence as a likely slide typo.

The next slide names Megatron Core's solution **MoE Parallel Folding** and displays separate group factorizations:

$$
\text{attention: } \mathrm{TP}\times\mathrm{CP}\times\mathrm{DP}\times\mathrm{PP},
$$

$$
\text{MoE: } \mathrm{ETP}\times\mathrm{EP}\times\mathrm{EDP}\times\mathrm{PP}.
$$

These exact names and factorizations are slide-supplied clarification; the transcript describes the decoupling without spelling them out.

### Additional explanation

Parallel groups are logical coordinate systems over the same physical ranks. A device can simultaneously have a data-parallel coordinate, pipeline-stage coordinate, tensor-shard coordinate, and expert coordinate. Composition works only when those groups cover the intended parameters and activations without accidental duplication or missing communication.

MoE Parallel Folding recognizes that attention and experts are different subgraphs. Requiring them to use one rank grid is convenient but unnecessarily restrictive. Separate grids let attention use enough TP to fit or accelerate its projections while experts preserve large GEMMs and use EP for sparse routing.

## 23. Context parallelism and the full tradeoff table

**Transcript coverage:** lines 2540-2658

### What the lecturer said - transcript only

Context parallelism, also called ring attention in the original formulation discussed, splits a very long sequence's activations across accelerators. Devices pass the required key/value or activation blocks around a ring, which maps naturally to a mesh topology. The lecturer says the approach worked especially well on TPUs and is now standard in long-context extension and serving. He does not develop it further because its sharding-and-communication ideas overlap with the methods already covered.

The lecture then compares all methods. No strategy strictly dominates:

- FSDP gives excellent model-state sharding but does not reduce activation memory and consumes global batch size as its parallel degree grows.
- Tensor parallelism reduces relevant activation and parameter slices without consuming batch size, but requires fast, high-bandwidth links.
- Pipeline parallelism reaches slower inter-node links with relatively light point-to-point traffic, but needs enough microbatches to control bubbles.
- Sequence and context parallelism reduce sequence-dimension activation or KV storage but do not solve parameter memory.
- Expert parallelism preserves efficient expert matrices, but its token-routing all-to-all is difficult and needs enough tokens per expert.

An audience member asks whether FSDP and pipeline parallelism still apply to MoEs. The lecturer answers yes: frontier MoE systems extensively combine data parallelism or FSDP, pipelines, tensor parallelism, and expert parallelism. The older rule was to keep EP, like TP, within a fast domain of roughly eight devices, although newer systems sometimes use much larger expert-parallel domains.

### Source reconciliation

The slide recap characterizes sequence and context parallelism in one row, while the spoken lecture distinguishes them: sequence parallelism is usually a TP add-on for pointwise activations; context parallelism or ring attention distributes long-context attention and KV work. The table below preserves that distinction.

### Additional explanation

| Method | Primary object sharded | Principal communication | Main limitation |
|---|---|---|---|
| DDP / ZeRO-1 | Data and optimizer state | Gradient reduction plus parameter distribution | Replicated parameters/activations; batch grows with DP |
| FSDP / ZeRO-3 | Parameters, gradients, optimizer state | Parameter all-gathers and gradient reduce-scatters | Activations remain; communication can dominate at small local batch |
| Pipeline parallel | Layer ranges | Point-to-point boundary activations and gradients | Pipeline bubbles and schedule complexity |
| Tensor parallel | Matrix width / heads | Activation-sized collectives inside each block | Requires fast links; small local GEMMs at high TP |
| Sequence parallel | Pointwise activation sequence shards | All-gather / reduce-scatter around TP regions | Usually not standalone; no parameter-memory benefit |
| Context parallel | Long-context sequence or KV blocks | Ring or block exchanges during attention | Communication and load scheduling for long sequences |
| Expert parallel | Complete experts | Token dispatch/combine all-to-all | Load balance, latency, and implementation complexity |

Context parallelism differs from simply giving each rank independent sequence examples. Tokens in one sequence attend across partitions, so ranks must exchange the information needed to compute exact attention while keeping the full $s\times s$ score matrix unmaterialized on any one device.

---

# Part V - Composing strategies and evidence from training runs

## 24. Roofline reasoning and the practical 3D/4D recipe

**Transcript coverage:** lines 2659-2830

### What the lecturer said - transcript only

Parallelism choices can be analyzed quantitatively. For each layer and sharding strategy, one can estimate computation and communication, assign those operations to links with known bandwidth, and compare their times. If useful computation lasts longer than communication, communication can in principle be hidden underneath it. Falling below that boundary means the device waits for the network.

The cited plot illustrates why multiple methods are needed. FSDP alone performs well when the batch per chip is very large; the lecturer gives a batch-size example of about 2,000. As local batch shrinks, FSDP becomes communication-bound. Adding model parallelism, represented there by tensor parallelism, extends the compute-bound region to smaller batch regimes. Adding further dimensions extends it again. Combining three or four dimensions in this way is called **3D or 4D parallelism**.

The lecturer reduces the design process to a practical prescription:

1. Until the model fits, partition it by whatever methods are necessary.
2. Use tensor parallelism for dense models or expert parallelism for MoEs on the fastest interconnect, commonly across the eight GPUs in one machine.
3. Use pipeline parallelism or FSDP/ZeRO-3 across the remaining ranks needed to fit the model.
4. Once the model fits, spend the remaining devices on data parallelism.
5. If the per-step batch is too small for efficient kernels or schedules, use gradient accumulation.

Megatron's practitioner guidance expresses the same hierarchy in reverse: minimize model parallelism and maximize data parallelism; keep TP and EP within the NVLink domain; use pipelines across nodes; prefer EP for expert layers; and use context parallelism for long sequences.

### Source reconciliation

The slide's roofline plot uses global batch divided by chip count, $B/N$, as the horizontal variable. Its annotations say no displayed scheme works below roughly 400, only mixed FSDP + model parallelism works from roughly 400 to 850, and both mixed and pure-FSDP choices can work above roughly 850. These thresholds belong to the plotted model and topology; the transcript uses the figure qualitatively rather than presenting them as universal cutoffs.

The Megatron screenshot adds that context parallelism is recommended for sequences of at least about 8K tokens in that guide. This exact threshold appears on the slide but is not stated aloud.

### Additional explanation

A compact roofline-style score is

$$
\rho = \frac{T_{\text{compute}}}{T_{\text{communication}}}.
$$

When $\rho\ge 1$ and scheduling is effective, communication may be fully hidden. When $\rho<1$, at least $T_{\text{communication}}-T_{\text{compute}}$ is exposed on the critical path. Real schedules also include latency, serialization, contention, and dependencies, so this is a feasibility test rather than a complete performance model.

The recipe has a clear priority order:

- **Fit first.** A configuration that exceeds memory is invalid regardless of theoretical throughput.
- **Use scarce fast links for frequent collectives.** TP and EP belong near the leaves of the network hierarchy.
- **Use low-traffic methods across slow links.** PP communicates only at stage boundaries.
- **Maximize statistically useful DP last.** DP is usually the cleanest compute scaler, but its degree is constrained by batch size.

## 25. Sequence parallelism as an add-on and the loop-Transformer question

**Transcript coverage:** lines 2831-2887

### What the lecturer said - transcript only

In response to a clarification, the lecturer says sequence parallelism is usually attached to tensor parallelism to reduce activations. It is often not treated as an independent, standalone parallelism dimension. Context parallelism would be the more natural name for a method that truly divides long-sequence attention work, but that name already refers to the ring-attention family.

Another audience member asks how the systems picture would change if a model reused the same Transformer block in a loop. The lecturer is skeptical of rumors about such architectures but considers the hypothetical. Reusing parameters could make FSDP's gather-discard pattern less suitable because the same weights remain needed across repeated applications. On the other hand, a recurrently reused block is more parameter-efficient, so less model parallelism may be required. The resulting system would therefore be meaningfully different from a conventional deep stack.

### Additional explanation

FSDP benefits from a one-way traversal over many distinct layer parameters: gather a layer, compute, discard it, and move on. If the same block is repeatedly reused, discarding and regathering identical weights can become needless traffic. Keeping that block resident may be better, trading some memory for fewer collectives.

This illustrates a general principle: a parallelism strategy is optimized for a computation graph, not merely a parameter count. Weight reuse, conditional branches, recurrence, and irregular depth can change tensor lifetimes and invalidate the assumptions behind an otherwise strong schedule.

## 26. Quantitative scaling evidence from Megatron-LM

**Transcript coverage:** lines 2888-3000

### What the lecturer said - transcript only

The lecturer recommends an older NVIDIA/Stanford scaling paper because network fundamentals have not changed enough to make its lessons obsolete. The paper benchmarks many large training configurations and exhibits the same hierarchy developed in the lecture:

- data parallelism is maximized when possible;
- tensor parallelism increases until it reaches eight, then stops;
- pipeline parallelism grows after TP has reached that limit;
- at the largest scales, data-parallel degree falls because more ranks are required simply to fit the model.

In the largest configuration discussed, data parallelism is only six. Even across extremely large GPU counts, the composed strategy keeps utilization approximately flat. The lecturer uses this as evidence that modern communication hardware and parallel schedules can efficiently exploit enormous data centers, including potentially cross-data-center training.

The paper also shows that TP beyond eight becomes unfavorable and that deeper pipelines need larger batches to remain efficient. Activation recomputation can help indirectly: although it adds arithmetic, the saved memory permits a larger batch, and that larger batch can improve device utilization enough to offset the recomputation cost.

### Source reconciliation

The slide table spans models from 1.7B to 1.008T parameters and GPU counts from 32 to 3,072. Tensor parallelism rises from 1 to 8 and remains at 8; pipeline parallelism rises as high as 64; the data-parallel degree falls from 32 to 6; and the reported percentage of theoretical peak remains roughly 43% to 52%. These exact values are supplied by the slide and support the transcript's qualitative summary.

One slide caption says that, for a 162.2B model across 64 GPUs, an $8\times8$ pipeline/tensor arrangement is best. The lecture's broader rule is not that TP=8 is mathematically universal, but that it aligns well with an eight-GPU high-bandwidth node and avoids overly small local matrices.

### Additional explanation

Recomputation can improve throughput even though it increases FLOPs because throughput is a system property. The causal chain is:

$$
\text{recompute}
\rightarrow
\text{less activation memory}
\rightarrow
\text{larger feasible micro/global batch}
\rightarrow
\text{larger GEMMs and smaller relative bubbles}
\rightarrow
\text{higher device utilization}.
$$

This is a useful warning against optimizing FLOP count in isolation. Extra arithmetic can be beneficial when it converts a memory- or communication-limited execution into a compute-efficient one.

## 27. OLMo, DeepSeek, and Yi configurations

**Transcript coverage:** lines 3001-3087

### What the lecturer said - transcript only

The lecture closes by examining actual model-training reports.

The lecturer first says "Dolma" and immediately corrects this to **OLMo**, the 7B open model trained on the Dolma dataset. OLMo used FSDP across many accelerators. This demonstrates that FSDP can scale surprisingly far for a relatively small model, and the lecturer suggests that many models around the 7B scale can be trained using FSDP alone.

DeepSeek uses more elaborate systems. Its earlier dense model used data parallelism with ZeRO stage 1 together with tensor, sequence, and pipeline parallelism. DeepSeek V3 is an MoE, so it combines pipeline parallelism with expert rather than primarily tensor parallelism for its experts. Its expert-parallel domain is 64-way and spans eight machines. DeepSeek uses pipeline-like overlap techniques so this large EP group does not suffer long low-utilization intervals.

Yi similarly uses ZeRO stage 1 with tensor and pipeline parallelism, which the lecturer presents as the classic DP + TP + PP combination. For MoE variants, expert parallelism takes over much of tensor parallelism's role because it serves the same broad purpose more efficiently for expert layers.

### Source reconciliation

The slide title says **Dolma - 7B model**, but the paper excerpt and the lecturer's immediate correction refer to **OLMo**, while Dolma is its training dataset. These notes use OLMo for the model and Dolma for the data.

The DeepSeek slide supplies exact details omitted from the spoken summary: the dense setup uses ZeRO-1 with tensor, sequence, and pipeline parallelism; DeepSeek V3 uses PP=16, EP=64 across eight nodes, and ZeRO-1; and its large EP communication is overlapped with a 1F1B-style all-to-all schedule.

The Yi slide adds that Yi-Lightning replaces tensor parallelism with expert parallelism. The spoken lecture states this as the general dense-to-MoE transition rather than naming that specific model.

### Additional explanation

These cases show that model size alone does not determine the layout:

- A 7B dense model may fit comfortably enough that FSDP provides all needed memory scaling.
- A much larger dense model may need TP and PP before DP can be applied.
- An MoE changes the efficient width partition from TP toward EP because experts are natural whole units.

The software burden rises with each additional dimension. Every new rank group affects checkpointing, initialization, random-number handling, gradient norms, logging, failure recovery, and the mapping from logical coordinates to physical topology.

## 28. Llama 3 405B, failures at scale, and Gemma 2 on TPUs

**Transcript coverage:** lines 3088-3173

### What the lecturer said - transcript only

Llama 3 405B is a very large dense model whose report provides unusually detailed parallel configurations for different training phases. After a small warm-up phase, the main pretraining configuration uses tensor parallelism 8, context parallelism 1, pipeline parallelism 16, and data parallelism 128. During long-context extension, context parallelism is increased and data parallelism is reduced to handle the much larger activation and memory burden. Because the model is dense, it uses tensor rather than expert parallelism.

Large training runs also face reliability problems. The lecturer cites 148 GPU failures during Llama 3 405B training. Efficient collectives are insufficient by themselves; the system also needs redundancy and recovery mechanisms for hardware and software interruptions.

Gemma 2 uses FSDP together with tensor and sequence parallelism and does not rely on pipeline parallelism. The lecturer interprets this as evidence for Google's claim that a sufficiently large TPU toroidal mesh can support broad tensor/model sharding without the sharp node boundary seen in GPU clusters. He remains unsure whether that approach scales indefinitely but says it works at Gemma's scale.

### Source reconciliation

The Llama table on the slide gives three rows:

| Phase | GPUs | TP | CP | PP | DP | Sequence length |
|---|---:|---:|---:|---:|---:|---:|
| Initial small-batch phase | 8,192 | 8 | 1 | 16 | 64 | 8,192 |
| Main pretraining | 16,384 | 8 | 1 | 16 | 128 | 8,192 |
| Long-context extension | 16,384 | 8 | 16 | 16 | 8 | 131,072 |

The failure slide reports 148 **interruptions categorized as faulty GPU** during a 54-day period, not necessarily 148 distinct physical GPUs. It also says about 78% of unexpected interruptions were attributed to confirmed or suspected hardware issues. The transcript compresses this into "GPUs failed 148 times."

The Gemma slide labels its ingredients ZeRO-3, model parallelism (TP + SP), and DP, with no pipeline. It shows model-sharding degrees of 1, 4, and 8 for the 2B, 9B, and 27B models, respectively. These exact sizes and degrees are slide-only clarification.

### Additional explanation

The Llama phase change demonstrates that parallelism is not fixed for an entire training run. When sequence length grows by $16\times$, activation and attention costs change sharply. Increasing CP while reducing DP reallocates devices from independent examples to cooperating on one long sequence.

Failure handling becomes part of utilization at this scale. A run that achieves high model FLOP utilization while healthy can still waste large amounts of wall-clock time if it cannot detect silent corruption, replace failed ranks, reload checkpoints quickly, and resume with consistent optimizer and data-loader state.

## 29. Mixtral, Nemotron, Qwen, and the emerging configuration pattern

**Transcript coverage:** lines 3174-3257

### What the lecturer said - transcript only

The lecturer recommends NVIDIA's Megatron Bridge repository as a source of practical configurations for many model shapes. Even within one broad method such as tensor parallelism, detailed configuration choices can substantially change performance, so systems integration remains important.

For the model transcribed as "Mistral 8x22B," the recommended configuration uses expert parallelism 8, pipeline parallelism 4, and tensor parallelism 4 for components such as attention. This keeps EP around the familiar fast-domain size while retaining TP where the dense attention path needs it.

Nemotron 3 Super follows a DeepSeek-like MoE pattern and, in its long-context phase, uses substantial expert and context parallelism. Qwen 3 similarly uses a large expert-parallel degree of 32 together with pipeline parallelism 8 and tensor parallelism 2 for attention matrices.

Across the surveyed models, the common pattern is to maximize data parallelism as far as memory and batch constraints permit. Tensor parallelism usually stays at or below eight. Expert parallelism can now be much larger, partly because DeepSeek V3 and related infrastructure made large-domain EP practical.

### Source reconciliation

The slide names the 8x22B model **Mixtral**, not Mistral. It provides the following exact or inferred configurations:

| Model / phase | TP | PP | CP | EP | DP |
|---|---:|---:|---:|---:|---:|
| Mixtral 8x22B recommendation | 4 | 4 | 1 | 8 | likely 2 to total 256 GPUs |
| Nemotron 3 Super long-context phase | 2 | 0 | 64 | 64 | not stated |
| Qwen 3 235B recommendation | 2 | 8 | 1 | 32 | not stated |

Only the TP/PP/EP values for Mixtral and Qwen, and the broad large-CP/large-EP description for Nemotron, are spoken. The remaining cells come from the slides and are therefore presented here as reconciliation rather than transcript content.

### Additional explanation

The pattern can be read as a topology-aware nesting order:

1. TP and the most latency-sensitive EP traffic use the fastest local links.
2. PP spans slower node boundaries.
3. CP expands during long-context phases when one sequence must occupy many ranks.
4. DP uses the outermost ranks that remain after the model and activations fit.

Large EP domains are an exception to the older "stay within one node" rule. They become viable only when routing libraries overlap dispatch, exploit specialized network paths, and keep token loads sufficiently balanced.

## 30. Final synthesis and transition to scaling laws

**Transcript coverage:** lines 3258-3291

### What the lecturer said - transcript only

The central lesson is that large-model training must be designed for multiple GPUs, nodes, and sometimes data centers. The system contains both fast and slow links, while different parallelism methods spend batch size, memory, communication, or scheduling flexibility in different ways.

Despite the number of methods, the lecturer argues that their composition follows fairly simple rules of thumb. Use the fastest links for the most communication-intensive partitions, use model parallelism only as much as needed to fit and utilize the model, and maximize data parallelism with the remaining resources. With effective composition, even very large systems can approach full utilization.

The next lecture begins scaling laws.

### Additional explanation

The lecture's unifying question is not "Which parallelism method is best?" It is:

> Which combination maps this model, batch, sequence length, and training phase onto this memory hierarchy and network topology with the least exposed idle time?

That framing makes parallelism an optimization problem with constraints rather than a catalog of independent algorithms.

---

# Consolidated takeaways

1. Large-model training is constrained by both aggregate compute and per-device memory, so the practical unit of design is the cluster or data center rather than one accelerator.
2. Network topology determines placement: frequent activation collectives belong on fast local links, while point-to-point pipelines tolerate slower links better.
3. An all-reduce can be decomposed into reduce-scatter plus all-gather with the same leading bandwidth cost, enabling ZeRO-1 and ZeRO-2 memory savings without more leading communication volume.
4. ZeRO progressively shards optimizer state, gradients, and parameters; FSDP/ZeRO-3 adds one model-sized all-gather but can hide much of it through layerwise prefetch and compute overlap.
5. Data parallelism is simple and effective but is bounded by useful global batch size and does not reduce activation memory.
6. Pipeline parallelism partitions depth and communicates boundary activations efficiently, but requires microbatches and careful schedules to control bubbles.
7. Tensor parallelism partitions width and reduces relevant parameter and activation memory, but inserts frequent collectives and usually remains within a high-bandwidth node.
8. Sequence parallelism complements tensor parallelism by sharding the pointwise activation terms that TP leaves replicated.
9. Selective activation recomputation can remove expensive stored intermediates and may increase total throughput by enabling larger, more efficient batches.
10. Expert parallelism is normally superior to TP for MoE experts because it preserves larger local GEMMs, but its latency-sensitive token all-to-all demands specialized infrastructure.
11. Context parallelism divides a long sequence across devices and becomes especially important during long-context training and serving.
12. No single method dominates. Practical 3D/4D configurations use enough TP/EP/PP/FSDP to fit, then maximize DP, changing the mix when model architecture or sequence length changes.
13. Real runs broadly follow the rules: GPU TP usually stops near eight, PP grows across nodes, DP fills remaining capacity, and modern MoEs sometimes use much larger EP domains.
14. At frontier scale, resilience, checkpointing, and recovery are part of training efficiency because hardware and software interruptions are routine.

# Key equations

## Collective identity

$$
\operatorname{all\mbox{-}reduce}
\equiv
\operatorname{reduce\mbox{-}scatter}
+
\operatorname{all\mbox{-}gather}.
$$

This is an equivalence in leading bandwidth volume; latency and overlap may differ.

## Model-state memory under ZeRO

Let $\Psi$ be the parameter count, $N_d$ the data-parallel degree, and $K\Psi$ the optimizer-state bytes:

$$
M_{\text{baseline}}=(2+2+K)\Psi,
$$

$$
M_{\text{ZeRO-1}}=2\Psi+2\Psi+\frac{K\Psi}{N_d},
$$

$$
M_{\text{ZeRO-2}}=2\Psi+\frac{(2+K)\Psi}{N_d},
$$

$$
M_{\text{ZeRO-3}}=\frac{(2+2+K)\Psi}{N_d}.
$$

These idealized expressions exclude activations, temporary buffers, fragmentation, and transient full-layer gathers.

## Pipeline bubble ratio

$$
\frac{T_{\text{bubble}}}{T_{\text{useful}}}
\approx
\frac{n_{\text{stages}}-1}{n_{\text{micro}}}.
$$

More microbatches reduce the relative fill/drain bubble, provided each microbatch remains large enough for efficient kernels.

## Tensor-parallel communication estimate

The slide's approximate per-layer activation communication is

$$
8bsh\left(\frac{n_{\text{devices}}-1}{n_{\text{devices}}}\right),
$$

compared with approximately $bsh$ point-to-point boundary traffic per pipeline microbatch.

## Activation-memory progression

$$
M_{\text{act, baseline}}
=
sbh\left(34+5\frac{as}{h}\right),
$$

$$
M_{\text{act, TP}}
=
sbh\left(10+\frac{24}{t}+5\frac{as}{ht}\right),
$$

$$
M_{\text{act, TP+SP}}
=
sbh\left(\frac{34}{t}+5\frac{as}{ht}\right),
$$

$$
M_{\text{act, TP+SP+recompute}}
=
sbh\frac{34}{t}.
$$

## Communication-hiding condition

$$
T_{\text{compute}}
\ge
T_{\text{communication}}.
$$

Meeting this inequality is necessary for full overlap but does not account for dependencies, latency, or contention.

# Glossary

| Term | Meaning in this lecture |
|---|---|
| 3D/4D parallelism | A composition of several rank dimensions, commonly data, tensor or expert, pipeline, and sometimes context parallelism. |
| All-gather | A collective in which ranks exchange distinct shards so every rank receives the complete concatenated object. |
| All-reduce | A collective that reduces values across ranks and returns the complete reduced result to every rank. |
| Context parallelism (CP) | Splitting a long sequence's attention or KV work across devices, often with ring-style block exchange. |
| Critical batch size | The regime beyond which increasing batch gives diminishing optimization progress per additional example. |
| Data parallelism (DP) | Replicating the model while assigning different examples to different ranks and synchronizing gradients or updates. |
| Expert parallelism (EP) | Placing different MoE experts on different devices and routing token activations to them. |
| FSDP | Fully sharded data parallelism, equivalent here to ZeRO stage 3: parameters, gradients, and optimizer state are sharded. |
| Gradient accumulation | Processing multiple microbatches before an optimizer step to create a larger effective batch without storing it all at once. |
| Microbatch | One subdivision of a batch used to fill a pipeline or limit peak activation memory. |
| Model parallelism | Any method that partitions model computation or parameters rather than only dividing examples. |
| NVLink domain | The group of GPUs connected by the node's fastest high-bandwidth interconnect, commonly eight in the lecture's examples. |
| Pipeline bubble | Idle time while a pipeline fills, drains, or waits on dependencies. |
| Pipeline parallelism (PP) | Assigning different layer ranges to stages and passing activations and gradients between them. |
| Reduce-scatter | A collective that reduces values across ranks and leaves each rank with one shard of the reduced result. |
| Selective activation recomputation | Discarding chosen forward intermediates and regenerating them in backward to save memory. |
| Sequence parallelism (SP) | A TP companion that shards pointwise activation work along the sequence axis. |
| Tensor parallelism (TP) | Partitioning matrices, heads, or hidden width across ranks and combining partial results with collectives. |
| Topology-aware placement | Mapping logical rank groups onto physical links according to their bandwidth and latency demands. |
| ZeRO | A staged family that shards optimizer state (stage 1), gradients (stage 2), then parameters (stage 3). |
| Zero-bubble scheduling | Pipeline schedules that prioritize activation-gradient propagation and defer local weight-gradient work into idle gaps. |

# Self-check questions

1. Why does reduce-scatter plus all-gather enable ZeRO-1 to match DDP's leading communication volume?
2. What is sharded at each of ZeRO stages 1, 2, and 3?
3. Why can FSDP communicate $3P$ elements yet expose much less than a $1.5\times$ slowdown over DDP?
4. Why does increasing the data-parallel degree eventually conflict with optimization efficiency?
5. What creates a pipeline bubble, and how does microbatching reduce it?
6. Why is pipeline parallelism often assigned to slower inter-node links?
7. How do column- and row-parallel matrix partitions compose across an MLP?
8. Why does tensor parallelism usually stop near the size of one high-bandwidth GPU node?
9. Which activation terms remain replicated under tensor parallelism alone?
10. How does sequence parallelism remove the residual $10sbh$ term?
11. Why is selective attention recomputation often preferable to recomputing an entire MLP?
12. Why does expert parallelism preserve GEMM efficiency better than high tensor parallelism in an MoE layer?
13. What makes expert dispatch latency-sensitive and difficult to balance?
14. Why might attention and MoE sublayers need different tensor-parallel degrees?
15. How does context parallelism differ from sequence parallelism despite both partitioning a sequence dimension?
16. What does it mean for a parallel configuration to be below the communication roofline?
17. In the lecture's rule of thumb, in what order should model-fitting and data-parallel scaling be applied?
18. How can recomputation increase end-to-end throughput while performing more arithmetic?
19. Why does Llama 3 increase context parallelism and reduce data parallelism during long-context extension?
20. Why must failure recovery be counted as part of utilization for a multi-week frontier run?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-104 | Scope, assignment framing, compute and memory limits, fast versus slow links | Yes |
| 105-185 | Collective primitives and all-reduce decomposition | Yes |
| 186-473 | TPU meshes, GPU switched networks, TPU8 changes, Huawei scale-out tradeoff | Yes |
| 474-697 | Parallelism taxonomy, naive data parallelism, training-state memory | Yes |
| 698-877 | ZeRO stages 1 and 2 | Yes |
| 878-1227 | ZeRO-3/FSDP mechanics, overlap, questions, and capacity example | Yes |
| 1228-1306 | Critical batch size and limits of data parallelism | Yes |
| 1307-1554 | Layer-wise and pipeline parallelism, microbatches, communication advantages | Yes |
| 1555-1677 | Interleaved and zero-bubble pipeline schedules | Yes |
| 1678-1904 | Tensor-parallel decomposition, Transformer layout, placement, and tradeoffs | Yes |
| 1905-2025 | Dynamic memory profile and baseline activation accounting | Yes |
| 2026-2207 | Tensor-parallel activation memory, sequence parallelism, final formulas | Yes |
| 2208-2247 | Recomputation question and cost tradeoff | Yes |
| 2248-2424 | Expert parallelism, advantages over TP, low-level routing systems | Yes |
| 2425-2539 | EP composition constraints and decoupled attention/MoE rank groups | Yes |
| 2540-2658 | Context parallelism, method comparison, MoE applicability question | Yes |
| 2659-2830 | Roofline analysis, 3D/4D composition, practitioner rules | Yes |
| 2831-2887 | Sequence-parallel naming and loop-Transformer question | Yes |
| 2888-3000 | Megatron scaling evidence, TP=8 pattern, pipelines, recomputation | Yes |
| 3001-3087 | OLMo, DeepSeek, and Yi training configurations | Yes |
| 3088-3173 | Llama 3 phases and failures; Gemma 2 TPU strategy | Yes |
| 3174-3257 | Mixtral, Nemotron, Qwen, Megatron Bridge, cross-model patterns | Yes |
| 3258-3291 | Final synthesis and transition to scaling laws | Yes |
