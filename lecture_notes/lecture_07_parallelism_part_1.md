---
title: "Lecture 7 - Parallelism, Part 1"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 7
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 7 Parallelism.txt"
slide_deck: "../lecture_07.pdf"
status: "complete"
---

# Lecture 7: Parallelism, Part 1

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves the lecturer's claims, examples, numerical details, cautions, and substantive questions and answers while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary intuition, derivation, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are shown so that the paraphrase can be audited. The complete transcript was mapped before the slides were inspected. All 26 slides were then rendered and checked for names, notation, code, diagrams, and numerical labels. Material discrepancies and verbal slips appear under **Source reconciliation** rather than being silently folded into the transcript-grounded account.

## Lecture map

The lecture extends last week's single-GPU optimization picture to distributed training. It has five main parts:

1. Recast parallel training as a problem of placing computation near data in a deeper hardware hierarchy.
2. Build a communication vocabulary from collective operations such as all-gather, reduce-scatter, all-reduce, and all-to-all.
3. Connect those abstractions to physical interconnects, NCCL, and `torch.distributed`, then measure effective communication bandwidth.
4. Implement three basic ways to cut a training problem: data parallelism across the batch, tensor parallelism across layer width, and pipeline parallelism across depth.
5. Explain why the right combination depends on interconnect speed, critical batch size, memory limits, and the ability to overlap communication with computation.

---

# Part I - From one GPU to a hierarchy of GPUs

## 1. Parallelism across GPUs is another data-movement problem

**Transcript coverage:** lines 1-196

### What the lecturer said - transcript only

The previous week studied how to make one GPU fast by writing kernels and reasoning about HBM, L2 cache, L1 cache or shared memory, registers, streaming multiprocessors, and tensor cores. This lecture expands the picture from one GPU to four, thousands, or more connected GPUs. The recurring systems problem does not change: arithmetic units are far from their inputs and outputs. On one GPU, the needed data may be in HBM; in a distributed system, it may be in another GPU's memory. Merely owning many GPUs is easy compared with using them effectively. Computation has to be orchestrated so that data transfer does not become the bottleneck.

The expanded speed hierarchy is:

- L1 cache and shared memory within one GPU are fastest;
- HBM is slower locally, although it now counts as fast relative to networking;
- NVLink and NVSwitch connect multiple GPUs in a node or fast switching domain;
- InfiniBand and Ethernet connect increasingly distant groups of GPUs and are slowest in this hierarchy.

Last week's remedies were fusion and tiling: load data into nearby memory, do as much work as possible, and write it back. This week's corresponding tools are replication and sharding, chosen to reduce communication across GPUs and nodes.

There are two distinct reasons to use more than one GPU. First, the parameters, optimizer state, gradients, and activations may not fit in one GPU's HBM. The lecturer gives a GPU with 192 GB and a trillion-parameter model as an obvious mismatch. Second, even a model that fits may train faster if its work is spread over more compute. That second case is a trade-off: more arithmetic units are available, but distributing the work introduces communication. A parallelization strategy must decide when the added compute repays that cost.

The executable lecture normally launches multiple processes and produces output from each one. The line-by-line trace used in class substitutes a special single-process mode, so its behavior should not be mistaken for the real multiprocessing run; separate standard output was linked for that run.

The lecture is organized in two halves. The first introduces the programming model, hardware, PyTorch implementation, and bandwidth measurement. The second demonstrates data, tensor, and pipeline parallelism on deep MLPs. The examples omit a full Transformer to expose the essential mechanics, but MLPs are a major Transformer compute bottleneck and therefore remain representative of the central distributed-compute problem.

### Source reconciliation

At line 118, the transcript renders the accelerator name as "B21." The hardware slides consistently identify the example generation as **B200**, so these notes treat B200 as the intended name while retaining the spoken 192 GB capacity figure. The slides also confirm that the traced path disables multiprocessing and distributed calls, whereas direct execution uses multiprocessing.

### Additional explanation

The hierarchy is best understood as a sequence of increasingly expensive placement mistakes:

| Where the needed data resides | Typical mechanism | Relative concern |
|---|---|---|
| Same GPU, on chip | Registers, shared memory, L1/L2 | Local kernel design |
| Same GPU, off chip | HBM | Memory bandwidth |
| Another GPU in a fast domain | NVLink/NVSwitch | Collective traffic |
| Another node or pod | InfiniBand or Ethernet | Network traffic and latency |

The two motives for scaling correspond to **memory scaling** and **throughput scaling**. They should not be conflated. A model that cannot fit requires some form of state sharding, offload, or recomputation even if speed is irrelevant. A model that already fits can use replication for simpler and often faster scaling, but only while communication and batch-size limits remain acceptable.

The trillion-parameter example illustrates why parameter count alone understates training memory. Even two bytes per parameter would require about 2 TB just for one low-precision parameter copy:

$$
10^{12}\ \text{parameters}\times 2\ \text{bytes/parameter}=2\ \text{TB}.
$$

Gradients, optimizer moments, master weights, and activations can multiply that requirement. Parallel training is therefore a joint problem in memory capacity, arithmetic throughput, and communication bandwidth.

---

# Part II - Collective communication primitives

## 2. Ranks, world size, and the collective programming model

**Transcript coverage:** lines 197-283

### What the lecturer said - transcript only

Collective operations are classic parallel-programming primitives dating to the 1980s; they were not invented for language models. A collective specifies a reusable communication pattern over many devices instead of requiring the programmer to manage every pairwise transfer. This interface is easier to use and lets the system choose a better implementation.

A **rank** identifies one participating process or device. In this lecture, a rank corresponds to one GPU. The **world size** is the total number of participating ranks. Four GPUs therefore give ranks 0 through 3 and a world size of 4.

The lecture introduces eight operations:

- broadcast;
- scatter;
- gather;
- reduce;
- all-gather;
- reduce-scatter;
- all-reduce;
- all-to-all.

Broadcast, scatter, gather, and reduce establish the concepts. All-gather, reduce-scatter, and all-reduce are the recurring workhorses for distributed language-model training. All-to-all is especially important for mixture-of-experts models, although this lecture only introduces it.

### Source reconciliation

The transcript briefly says "the first three" operations are warm-ups after naming broadcast, scatter, gather, and reduce. The slide groups all four as the foundational operations and the surrounding explanation covers all four before moving to the workhorses.

### Additional explanation

A collective is both an abstract contract and a real distributed call. The contract says what tensors each rank owns before and after the operation. It deliberately leaves the route unspecified. A backend may implement the same all-reduce with a ring, tree, hierarchical combination, or another topology-aware algorithm without changing application code.

It is useful to distinguish three properties:

| Property | Question |
|---|---|
| Participants | Which ranks join the operation? |
| Transformation | Are values copied, concatenated, or reduced? |
| Destinations | Does one root receive the result, or do all ranks receive it? |

All ranks in a collective generally must invoke compatible calls in a compatible order. If one rank skips or mismatches an operation, the others may wait indefinitely or fail. That coordination requirement is one reason collective libraries provide more than a convenient tensor API: they also encode a distributed execution protocol.

## 3. Broadcast, scatter, gather, and reduce

**Transcript coverage:** lines 284-406

### What the lecturer said - transcript only

**Broadcast** starts with a tensor at one root rank and copies it to every rank. With rank 0 holding `[0, 1, 2, 3]`, all four ranks end with that tensor. Broadcast is not normally on the central training path. A typical one-time use is loading an initial checkpoint on one rank and broadcasting it to the others.

**Scatter** starts with a larger tensor at one root, divides it into world-size pieces, and gives one piece to each rank. Scattering `[0, 1, 2, 3]` from rank 0 over four ranks leaves rank 0 with `[0]`, rank 1 with `[1]`, and so on. Although plain scatter is not the main operation later in the lecture, it prepares the idea of reduce-scatter. It is useful whenever local devices should compute on distinct pieces.

**Gather** is the inverse pattern. Each rank begins with one piece, and a designated root concatenates those pieces. Four scalar pieces `[0]`, `[1]`, `[2]`, and `[3]` become `[0, 1, 2, 3]` on rank 0. Gather is a stepping stone to all-gather.

**Reduce** also begins with data distributed across ranks, but combines the values with a reduction operation and writes the result to one root. Summing scalar values 0, 1, 2, and 3 produces 6 on rank 0. The lecturer notes that gather can loosely be viewed as a reduction whose operation is concatenation. Reduce prepares the idea of all-reduce.

In response to a question about NumPy broadcasting, the lecturer says the one-to-many intuition is related, but the instantiation is different. NumPy broadcasting describes how array shapes participate in local elementwise computation; this broadcast is collective communication among devices.

### Source reconciliation

At lines 340-341, the speaker says "the inverse of scatter is scatter" and immediately proceeds to define gather as that inverse. The slide explicitly says "Gather ... (opposite of scatter)." The intended statement is therefore that **gather is the inverse communication pattern of scatter**.

### Additional explanation

For a root rank $r$ and $p$ ranks, the four contracts can be summarized as follows:

| Operation | Before | After |
|---|---|---|
| Broadcast | Root owns $x$ | Every rank owns $x$ |
| Scatter | Root owns shards $x_0,\ldots,x_{p-1}$ | Rank $i$ owns $x_i$ |
| Gather | Rank $i$ owns $x_i$ | Root owns the ordered collection or concatenation |
| Reduce | Rank $i$ owns $x_i$ | Root owns $x_0\oplus\cdots\oplus x_{p-1}$ |

The reduction operator $\oplus$ is normally associative, and often commutative, so that the backend can reorder partial combinations. Floating-point addition is not perfectly associative because of rounding, so different reduction trees may produce tiny numerical differences even though they implement the same mathematical sum.

The root is an argument to root-oriented collectives, not a permanent special property of rank 0. Rank 0 is used in diagrams because it is an easy convention.

## 4. All-gather, reduce-scatter, and all-reduce

**Transcript coverage:** lines 407-520

### What the lecturer said - transcript only

The prefix **all** means that every rank is a destination.

**All-gather** performs a gather at every rank. If ranks 0 through 3 each hold one of `[0]`, `[1]`, `[2]`, and `[3]`, every rank receives `[0, 1, 2, 3]`. One later use is to let every rank reconstruct full parameters from parameter shards for a forward pass. Distributed training often alternates between gathering a full value for computation and returning to sharded storage.

**Reduce-scatter** reduces corresponding tensor components across ranks and distributes different components of the result. In the lecture's four-rank example, the input rows are:

$$
\begin{aligned}
r_0 &: [0,1,2,3],\\
r_1 &: [1,2,3,4],\\
r_2 &: [2,3,4,5],\\
r_3 &: [3,4,5,6].
\end{aligned}
$$

Elementwise summation gives `[6, 10, 14, 18]`; reduce-scatter leaves 6 on rank 0, 10 on rank 1, 14 on rank 2, and 18 on rank 3. The foreshadowed training use is to sum gradients produced from different data shards while leaving their storage distributed.

**All-reduce** can be understood as reduce-scatter followed by all-gather. It reduces values across ranks and then replicates the complete reduced result on every rank. In the same example, all four ranks end with `[6, 10, 14, 18]`. The first application later in the lecture is data parallelism, in which ranks combine their local gradients and retain replicated model state. More flexible schemes such as ZeRO and FSDP separate the operation into reduce-scatter and all-gather so they can manage sharded storage between those phases; basic distributed data parallelism can use a single all-reduce.

### Additional explanation

Let rank $i$ hold a tensor $x_i\in\mathbb{R}^n$. A sum all-reduce produces

$$
y=\sum_{i=0}^{p-1}x_i
$$

and stores $y$ on every rank. A reduce-scatter instead partitions $y$ into $p$ shards and gives shard $y^{(i)}$ to rank $i$. An all-gather reverses that storage layout by collecting all $y^{(i)}$ on every rank.

This decomposition explains an important memory distinction:

- all-reduce ends with a replicated result;
- reduce-scatter ends with a sharded result;
- all-gather starts from shards and ends with replicas.

The algebraic equality "all-reduce = reduce-scatter + all-gather" describes the logical result. A production backend may fuse, pipeline, or otherwise implement the operation without materializing those two calls exactly as written.

## 5. All-to-all and dynamic expert routing

**Transcript coverage:** lines 521-600

### What the lecturer said - transcript only

All-to-all is the most general collective introduced here: every rank sends a designated piece to every destination rank. In the balanced four-rank example, rank 0 sends 0, 1, 2, and 3 to ranks 0, 1, 2, and 3; rank 1 similarly sends 4, 5, 6, and 7; and the other ranks continue the pattern. The destinations receive columns of the original rank-by-destination arrangement:

$$
\begin{aligned}
r_0 &: [0,4,8,12],\\
r_1 &: [1,5,9,13],\\
r_2 &: [2,6,10,14],\\
r_3 &: [3,7,11,15].
\end{aligned}
$$

This pattern is useful for mixture-of-experts models. A rank may hold both a shard of the data and only a subset of experts. Because routing is data-dependent, activations must be sent to whichever ranks own the selected experts. With equal traffic to every destination, all-to-all is morally a matrix transpose. The operation can also accept unequal split sizes, but uneven traffic is undesirable. The load-balancing mechanisms discussed in the mixture-of-experts lecture are intended to keep the exchanges as balanced as possible.

### Additional explanation

If the local payload on each source rank is divided into destination blocks, all-to-all transposes the ownership layout:

$$
\text{source-major blocks}\quad [x_{i\rightarrow j}]_{i,j}
\quad\longrightarrow\quad
\text{destination-major blocks}\quad [x_{i\rightarrow j}]_{j,i}.
$$

For expert parallelism, this is commonly a **dispatch** phase. Tokens are grouped by selected expert and sent to expert-owning ranks; after expert computation, another exchange returns outputs to their original token positions. Balance matters for two reasons. The busiest destination determines the collective's completion time, and unequal token counts can also create unequal expert compute. A logically correct route can therefore have poor throughput if one expert or rank becomes a hotspot.

## 6. Remembering the collectives and choosing a root

**Transcript coverage:** lines 601-660

### What the lecturer said - transcript only

The terminology can be remembered with three rules. A **reduce** applies an associative and commutative operation such as sum, minimum, or maximum. **Scatter** distributes pieces, whereas **gather** centralizes them. The word **all** means that the result's destination is every device.

An audience member asks whether gather or reduce must always write to the same physical rank 0. The lecturer answers that rank 0 is only the example. The call specifies the destination rank when it executes, so another rank can be the root; the destination need not have been fixed far in advance.

Another question asks whether the operations are merely conceptual diagrams or actual code. They are both: the current presentation is conceptual, and the next portion shows concrete implementations.

### Additional explanation

A compact way to reconstruct the vocabulary is:

| Name component | Meaning |
|---|---|
| `scatter` | One collection becomes distributed shards |
| `gather` | Distributed shards become one collection |
| `reduce` | Multiple values become one combined value |
| `all-` | Every rank receives the resulting collection or reduction |

This naming system predicts the storage layout after an operation, which is often more useful than memorizing diagrams. When designing a distributed algorithm, first write down who owns each tensor before and after a step. The required collective usually follows directly from that ownership change.

---

# Part III - Hardware, NCCL, and PyTorch distributed execution

## 7. The interconnect hierarchy, RDMA, NVL72, and RoCE

**Transcript coverage:** lines 661-888

### What the lecturer said - transcript only

A traditional server diagram has CPUs and RAM, a PCIe bus with attached GPUs, and Ethernet connecting one server to another. GPUs in the same ordinary node communicate over PCIe; GPUs in different nodes must traverse Ethernet. The lecturer characterizes this as the sort of arrangement one might improvise from gaming machines rather than the hardware used for serious large-scale training.

A modern training node more typically contains eight GPUs connected by NVIDIA NVLink to an NVSwitch. NVLink 5 is quoted at about 1.8 TB/s of total bandwidth, compared with roughly 8 TB/s for B200 HBM, so even this fast GPU-to-GPU path is about four times slower than local HBM. NVSwitch makes the fast domain appear broadly any-to-any: the hardware routes traffic from a GPU through the switch to another GPU.

The domain cannot grow without bound. Groups of nodes are connected through InfiniBand, with a path involving PCIe, an InfiniBand adapter, and an InfiniBand cable. Larger groups may ultimately be joined through Ethernet, whose conventional path also involves the CPU. Each outward step in the hierarchy is slower. An NVSwitch cannot simply connect 100,000 GPUs at the same speed.

Traditional Ethernet requires the GPU's data to pass through the CPU-side networking stack. Data is copied into a kernel socket buffer, packetized, placed in a network-interface buffer, and transmitted. Here "kernel" means the operating-system kernel rather than a GPU kernel. The extra staging adds latency.

**Remote Direct Memory Access (RDMA)** lets one GPU directly read or write another GPU's memory without involving the CPU. NVLink/NVSwitch provide this within their domain, and InfiniBand supports it across nodes; standard Ethernet does not.

Two developments push against the usual limits:

- NVIDIA's NVL72 design connects nine trays of eight GPUs, giving 72 GPUs in one NVLink domain. It is expensive, but it expands the fast domain far beyond the usual eight-GPU node.
- **RDMA over Converged Ethernet (RoCE)** makes an Ethernet-based network bypass the CPU. It is presented as Ethernet's answer to InfiniBand: InfiniBand is generally expensive, while RoCE can still provide good performance. The lecturer mentions Meta research exploring this approach but the transcript does not preserve the associated model name.

The resulting practical picture is a fast NVLink/NVSwitch island of perhaps 8 or 72 GPUs, followed by InfiniBand at larger scale and potentially Ethernet farther out.

### Source reconciliation

The slide identifies the large system as **GB200/GB300 NVL72**, with 8 GPUs per tray and 9 trays per rack, and characterizes RoCE as similar to but cheaper and weaker than InfiniBand. This resolves the less clear product wording in the transcript and keeps the slide's comparative claim out of the transcript-only account. The older-topology slide additionally annotates PCIe 7.0 with 16 lanes at about 242 GB/s and a traditional cross-node Ethernet example at about 200 MB/s. The modern-topology slide labels an illustrative InfiniBand tier at about 0.05 TB/s and explicitly says its example of 256 nodes per pod is only a setup sketch, matching the lecturer's warning that 256 was made up.

### Additional explanation

The quoted numbers are useful as orders of magnitude, but bandwidth labels are not automatically comparable. A specification may be one-directional or bidirectional, per link or aggregate, theoretical or sustained, and measured for a particular message size. The durable lesson is the ratio: each move away from local memory consumes a scarcer resource.

The conceptual stack is:

| Layer | What it supplies |
|---|---|
| HBM | Local storage bandwidth on one accelerator |
| NVLink | High-bandwidth point-to-point GPU links |
| NVSwitch | Switching fabric across GPUs in an NVLink domain |
| InfiniBand | High-performance inter-node network with RDMA support |
| RoCE | RDMA semantics carried over a configured Ethernet fabric |
| Traditional Ethernet path | Packet transport that may stage through CPU memory and the OS stack |

RDMA is a memory-access capability, not the name of one cable. It removes CPU staging from the data path, but it does not make a remote access as cheap as a local HBM access. Link bandwidth, switching contention, message latency, and collective scheduling still matter.

## 8. NCCL turns collective intent into topology-aware GPU work

**Transcript coverage:** lines 889-1055

### What the lecturer said - transcript only

The NVIDIA Collective Communications Library, NCCL, pronounced "nickel," sits below the framework-level collective interface. A program asks for an operation such as all-reduce or broadcast. NCCL discovers the relevant hardware topology, chooses paths among nodes, switches, NVLink, and PCIe, and launches GPU communication kernels that send and receive the data. Communication is still implemented by kernels because all GPU work ultimately runs as kernels. The lecture does not descend further into NCCL internals.

The hardware discussion produces several audience questions:

- **What are a rack and a tray physically?** A rack is the literal data-center rack. In the NVL72 description, one tray contains two CPUs, each connected to four GPUs, for eight GPUs total; trays are stacked and connected to the NVSwitch. The expansion of the letter "G" is inaudible in the transcript.
- **How does RDMA differ from NVLink or InfiniBand?** RDMA describes the desired operation: one GPU directly accesses another GPU's memory. NVLink/NVSwitch, InfiniBand, and RoCE are different hardware or transport mechanisms that can provide that capability.
- **Is NCCL optimized for multi-node clusters?** The lecturer does not claim detailed knowledge of the implementation. He reasons that NVIDIA has optimized its stack heavily for large-model training and inference and would be surprising not to have considered those workloads.
- **What happens if there are nine GPUs when nodes contain eight?** If the ninth GPU is alone on another node across a much slower link, it adds little compute relative to its communication cost and is a poor arrangement. If all nine GPUs share a fast NVSwitch domain, the arrangement is much more reasonable.
- **How does this map to TPUs?** The lecturer says TPUs are generally simpler objects but does not know the component-level correspondence well enough to give a precise answer, so the discussion is deferred.

### Source reconciliation

The slide confirms the platform label **GB200/GB300 NVL72**, but neither the slide nor the audible transcript establishes what the speaker intended when expanding the letter "G" in the rack-and-tray answer. These notes therefore do not guess. The slide describes NCCL as translating collectives into low-level packets; the spoken explanation adds the important implementation detail that NCCL launches GPU send/receive kernels.

### Additional explanation

The layers should be kept distinct:

$$
\text{training algorithm}
\rightarrow \text{collective API}
\rightarrow \text{NCCL plan and kernels}
\rightarrow \text{links, switches, and NICs}.
$$

Data parallelism may request an all-reduce without knowing whether NCCL implements it as a ring, a tree, or a hierarchy that reduces within a node before crossing nodes. This separation is valuable because the training algorithm expresses **what ownership change is required**, while the backend adapts **how traffic should flow** to the actual machine.

The ninth-GPU answer is an example of topology-aware scaling. GPU count by itself is a misleading resource measure. A partition with eight equally connected workers may outperform a nominally larger partition whose extra worker lies behind a slow, oversubscribed link.

## 9. Processes, backends, setup, and barriers in `torch.distributed`

**Transcript coverage:** lines 1056-1175

### What the lecturer said - transcript only

PyTorch's `torch.distributed` library provides a clean interface to collectives without requiring application code to invoke NCCL directly. It supports different backends: NCCL for GPUs and Gloo for CPUs. It also contains higher-level distributed algorithms, but the course deliberately uses lower-level primitives so students can build the mechanisms from scratch.

The example `spawn` wrapper launches the same worker function `world_size` times in a normal execution, once for each rank. The lecture trace cannot step through multiprocessing, so the wrapper takes a special branch that invokes only rank 0 and temporarily disables distributed calls. In a real run, the workers execute asynchronously. Rank is an integer from 0 through `world_size - 1`, and there is one worker process per rank in the course's model.

The distributed environment is initialized with a master address and port. Those values support coordination and metadata exchange; the bulk tensor data does not flow through that endpoint. With GPUs available, the actual payload uses NCCL. On the lecturer's laptop, the example falls back to Gloo.

A `barrier()` is a process rendezvous. A rank that reaches it waits until all participating ranks arrive. This is useful when later code should not proceed until every worker has reached a known point, and it makes interleaved print examples easier to read. Barriers also have a cost: inserting too many forces fast ranks to wait unnecessarily and can destroy useful concurrency.

### Source reconciliation

At line 1085, automatic transcription renders the higher-level algorithm as "SFTP." The slide names **`FullyShardedDataParallel`**, or FSDP, as the intended example. The code slides also verify that `MASTER_ADDR` and `MASTER_PORT` are coordination settings, while NCCL carries actual GPU data.

### Additional explanation

A common process-per-GPU arrangement is:

```text
rank 0 process -> GPU 0
rank 1 process -> GPU 1
...
rank p-1 process -> GPU p-1
```

The process group defines membership and gives collective calls a shared communication context. Each process runs ordinary Python control flow, but matching collectives create cross-process dependencies. Consequently, distributed correctness includes control-flow correctness: ranks must reach compatible operations in a compatible sequence.

`barrier()` should not be used as a universal correctness patch. A collective already synchronizes the data dependencies required by its own contract. Extra global barriers can hide a rank-ordering bug during development and later become a performance bottleneck. They are most defensible for measurement boundaries, debugging, or a genuine phase transition that requires all ranks.

## 10. A concrete all-reduce in PyTorch

**Transcript coverage:** lines 1176-1239

### What the lecturer said - transcript only

Each rank creates a different four-element tensor by adding its rank to `[0, 1, 2, 3]`. Rank 0 therefore holds `[0, 1, 2, 3]`, rank 1 holds `[1, 2, 3, 4]`, and so on. The processes are asynchronous, so their "before" print statements can arrive in any order.

The program calls PyTorch's all-reduce with sum and disables asynchronous return for the example. The backend performs the communication and writes the reduced result **in place** into the input tensor. Each position is summed across ranks, so every rank ends with `[6, 10, 14, 18]`. The operation could be launched asynchronously, but the simple print sequence would then need explicit completion management. In response to a question, the lecturer confirms that a rank corresponds to a GPU for this class.

### Additional explanation

The essential code shape is:

```python
data = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device) + rank
dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
```

Because the call mutates `data`, any code that needs the unreduced local value must save it beforehand. With sum reduction, the result is the global sum. Data-parallel training often needs a global mean instead, which can be produced either by an average reduction when supported or by dividing the summed result by world size.

Nondeterministic print order is not evidence of an incorrect collective. It reflects independent host processes. The data result is constrained by the collective even when log messages are interleaved.

## 11. Reduce-scatter, all-gather, and asynchronous overlap in PyTorch

**Transcript coverage:** lines 1240-1339

### What the lecturer said - transcript only

For reduce-scatter, each rank creates a world-size input vector offset by its rank and allocates a separate one-element output tensor. Unlike the in-place all-reduce example, `reduce_scatter_tensor` leaves the input unchanged and writes the appropriate reduced component into the output for that rank.

An audience member asks what an asynchronous all-reduce means. Calling the operation can launch communication kernels and return control before they finish. There are two levels of asynchrony: CUDA work can be asynchronous with respect to its host process, and the rank processes are asynchronous with respect to one another. The process can perform independent work after launching the collective - for example, loading data for a later step - and wait when the result is actually required. This ability to overlap communication and computation is important even though the lecture does not yet implement it fully.

The output of reduce-scatter is then used as the input to all-gather. Each rank allocates space for the complete output and calls `all_gather_into_tensor`; afterward, every rank has all four reduced components. The example demonstrates concretely that reduce-scatter followed by all-gather has the same logical output as all-reduce. The process group is destroyed at the end as good cleanup practice.

### Additional explanation

An asynchronous PyTorch collective typically returns a work handle:

```python
work = dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=True)

# Perform work that does not read or overwrite `data`.
prepare_next_batch()

work.wait()
consume(data)
```

Overlap is safe only when the intervening computation is independent of the communicated buffer and when stream semantics are respected. Asynchrony by itself does not guarantee overlap: the hardware needs spare execution and link resources, the transfer must be large enough to matter, and dependencies must be scheduled so that neither side immediately waits.

All-gather and reduce-scatter are adjoint-like ownership changes. One expands sharded values into replicas; the other combines replicated contributions and returns shards. This pairing reappears in fully sharded training and in the backward pass of distributed tensor transformations.

## 12. Benchmarking collectives and defining effective bandwidth

**Transcript coverage:** lines 1340-1515

### What the lecturer said - transcript only

The lecture benchmarks a large all-reduce and reduce-scatter. As in kernel benchmarking, the operation is warmed up before timing. Because both CUDA kernels and worker processes are asynchronous, the code synchronizes CUDA and places a process barrier around the timed region so that unfinished work does not leak into or out of the measurement.

Each rank measures its own elapsed time and may report a slightly different value. If one summary number is needed, the lecturer suggests averaging the rank measurements.

Raw latency is difficult to interpret in isolation, so the lecture constructs an **effective bandwidth**, analogous in spirit to model FLOPs utilization. Let $S$ be the byte size of one all-reduce tensor, $p$ the world size, and $t$ one rank's measured duration. The lecture counts sent and received traffic as

$$
2S(p-1),
$$

then treats aggregate waiting time as $pt$. Its bandwidth expression is therefore

$$
B_{\mathrm{eff,AR}}
=\frac{2S(p-1)}{pt}
=\frac{2S}{t}\frac{p-1}{p}.
$$

As $p$ grows, $(p-1)/p$ approaches 1, leaving approximately $2S/t$. The lecturer describes this normalized form as independent of world size and of whether NCCL selects a ring or tree. The displayed run is roughly 400 GB/s.

The reduce-scatter benchmark follows the same pattern but omits the factor of 2 in its byte accounting. Its measured effective bandwidth is also in the 400 GB/s range, with ordinary run-to-run variation. Since all-reduce consists logically of reduce-scatter plus all-gather, it moves about twice the data and takes about twice the time in this comparison; those factors cancel, producing similar bandwidth.

### Source reconciliation

The speaker calls the payload "100 million elements." The code slide uses `100 * 1024**2`, which is 104,857,600 elements. The slides also make the exact accounting explicit: `size_bytes * 2 * (world_size - 1)` divided by `world_size * duration` for all-reduce, and no `2x` term in the reduce-scatter calculation.

### Additional explanation

An effective-bandwidth metric converts a collective into an equivalent amount of useful traffic per second. It is valuable for comparison only when the byte-count convention is stated, because libraries and papers use several conventions, including algorithm bandwidth and bus bandwidth.

The lecture's statement about topology requires a careful reading. The **algebraic normalization** does not include a ring-versus-tree term, but the observed duration $t$ absolutely can depend on topology, contention, message size, backend algorithm, and link speed. Thus the formula is topology-agnostic while measured performance is not.

Good collective benchmarks should control at least:

- warm-up and allocator effects;
- GPU completion before timing boundaries;
- rank rendezvous around the measurement;
- tensor dtype and payload size;
- world size and rank placement;
- repeated trials and an explicit aggregation rule, often maximum latency when step time is limited by the slowest rank.

The last point adds a useful alternative to the lecture's average: training cannot normally begin the next dependent phase until every required rank finishes, so the slowest-rank time can be the most operationally relevant statistic.

## 13. Q&A: CUDA synchronization versus a process barrier

**Transcript coverage:** lines 1516-1561

### What the lecturer said - transcript only

An audience member asks why CUDA synchronization is needed. Distributed workers still issue CUDA operations, and those operations are asynchronous by default. Reaching the next Python line does not prove that the GPU kernel has finished, so the timer must explicitly wait for device work.

A follow-up asks whether the process barrier should come before CUDA synchronization. The lecturer says he is not certain, then reasons that a barrier first may be ineffective: host calls can return while kernels are still running, allowing every process to reach the barrier even though their device work remains incomplete. Each rank would then synchronize independently afterward. The code therefore synchronizes device work before using the barrier to align processes.

### Additional explanation

The two calls answer different questions:

- **CUDA synchronization:** Has this rank's previously issued device work completed?
- **Distributed barrier:** Have all ranks reached this host-side rendezvous?

For a clean end boundary in a simple benchmark, synchronizing local device work and then rendezvousing ranks makes the elapsed interval include the collective's GPU work and prevents one rank from ending while another remains behind. Precise timing behavior can depend on the backend and stream semantics, so production benchmark harnesses should document the order rather than assume that any barrier implies completion of all prior device work.

---

# Part IV - Cutting training across batch, width, and depth

## 14. Three geometric cuts through an MLP

**Transcript coverage:** lines 1562-1600

### What the lecturer said - transcript only

The second half turns from communication building blocks to bare-bones distributed training. It uses multilayer perceptrons because MLPs are a Transformer compute bottleneck and display the core algorithms without the bookkeeping of a full model.

The three strategies can be pictured as different cuts through the same computation:

- **data parallelism** cuts the batch, so each rank processes different rows;
- **tensor parallelism** cuts the width of layers, so each rank owns part of every layer;
- **pipeline parallelism** cuts the depth, so each rank owns a consecutive subset of layers.

The diagrams are schematic rather than complete system designs. Their purpose is to make ownership of data and parameters visually clear before examining the code.

### Additional explanation

Each cut saves or replicates a different object:

| Strategy | Data on each rank | Parameters on each rank | Frequent communication |
|---|---|---|---|
| Data parallel | Different batch shard | Full model replica | Gradients or model-state shards |
| Tensor parallel | Often replicated or compatibly sharded | Slice of each participating layer | Activations and their gradients |
| Pipeline parallel | Microbatches flow through stages | Full-width subset of layers | Stage-boundary activations and gradients |

Real systems compose these axes. A rank coordinate might identify a data-parallel replica, a tensor-parallel shard within that replica, and a pipeline stage. The simple examples isolate one axis at a time so that the required ownership transitions remain visible.

## 15. Distributed data parallelism: local batches, global gradients

**Transcript coverage:** lines 1601-1716

### What the lecturer said - transcript only

The data-parallel example creates a matrix with batch size 128 and feature dimension 1,024. With world size 4, it partitions the rows into four local batches of 32 examples. Each rank receives one slice. In a real input pipeline, every rank should load its own data directly rather than creating a central loading and scattering bottleneck; the shared tensor and slicing are only illustrative.

Every rank constructs the full MLP, with a full square parameter matrix for every layer, and maintains its own optimizer state. A rank performs an ordinary forward pass and loss computation on its local data, then calls backward. Since the data shards differ, the local losses and initial parameter gradients differ.

The key DDP step comes after backward and before the optimizer update: for every parameter, all ranks all-reduce `param.grad` with an average. After this operation, every replica has the same gradient. Each rank then performs the same optimizer update, so their parameters remain identical. Conceptually, this synchronization is the only change from ordinary single-process training.

The result is elegant: every worker computes on only part of the global batch but updates as though it had processed the whole batch. The distributed step can be understood as ordinary local training plus one gradient synchronization between backward and the optimizer step.

### Source reconciliation

The slides make two implementation assumptions explicit. `get_init_params` resets the random seed to 0, ensuring that all ranks begin with identical parameter values, and each rank creates its own AdamW optimizer state. The toy code uses a four-layer MLP, GeLU, and an average squared-magnitude loss; these specifics support the spoken walkthrough but are not essential to DDP's general definition.

### Additional explanation

Suppose rank $r$ has $m$ examples and local mean loss

$$
L_r(\theta)=\frac{1}{m}\sum_{j=1}^{m}\ell(x_{r,j};\theta).
$$

With equal local batch sizes, averaging local gradients gives

$$
\frac{1}{p}\sum_{r=0}^{p-1}\nabla_\theta L_r(\theta)
=\nabla_\theta\left(
\frac{1}{pm}\sum_{r=0}^{p-1}\sum_{j=1}^{m}\ell(x_{r,j};\theta)
\right),
$$

which is exactly the gradient of the global mean loss over the combined batch. This equivalence requires replicas to start from the same parameters and apply the same update rule after synchronization.

If local batch sizes differ, an unweighted mean of rank gradients is not generally the sample-weighted global mean. The reduction must account for the number or total weight of valid examples on each rank. This is especially relevant for padding, variable-length examples, and uneven final batches.

DDP scales compute and input throughput, but it does not reduce per-rank storage of parameters, gradients, or optimizer state. Its simplicity comes from replicating those objects and communicating only what is needed to keep the replicas synchronized.

## 16. DDP Q&A, invariants, and the memory limit

**Transcript coverage:** lines 1717-1785

### What the lecturer said - transcript only

An audience question asks how large the batch must be. The global batch should be at least the world size for every rank to receive an example, and in practice it is usually much larger. It is convenient for batch size to be divisible by world size. If it is not, the lecturer suggests that padding can make the partition even, though divisibility is simpler.

Asked what changes for a Transformer, the lecturer says the method is essentially the same. DDP is modular because it does not care what the forward computation contains: the model runs locally, and distributed synchronization happens at the gradients.

The invariant is:

- local losses differ because local data differs;
- gradients initially differ;
- all-reduction makes the gradients equal;
- equal updates keep the parameters equal across ranks.

The limitation is memory. A monolithic all-reduce is simple, but DDP requires every rank to hold the full model and associated replicated state. It therefore does not solve the case in which the model parameters do not fit. The next lecture is assigned to more sophisticated data parallelism, FSDP and ZeRO, which use all-gather and reduce-scatter to avoid keeping all parameters resident on every rank.

### Source reconciliation

At lines 1753-1754, the transcript says DDP "averages the parameters." The surrounding explanation, the summary at lines 1759-1767, and slide code `dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)` all show that the operation averages **gradients**. Parameters remain synchronized because identical replicas apply those averaged gradients; they are not averaged at this point.

The transcript renders ZeRO as the digit "0." The slide spells the algorithm family `ZeRO`, which is the intended name.

### Additional explanation

Padding an uneven batch with zeros is only mathematically harmless if padded examples are excluded from the loss or otherwise contribute exactly zero weight. Treating a zero-valued input as an ordinary example can change both the loss and gradient. A robust implementation tracks valid-example counts and uses a weighted reduction.

The core DDP invariant can be proved inductively. If all ranks begin step $k$ with the same parameters $\theta_k$, average the same reduced gradient $g_k$, and execute the same deterministic update rule $U$, then

$$
\theta_{k+1}^{(r)}=U(\theta_k,g_k)
$$

is identical for every rank $r$. Differences in initialization, skipped collectives, overflow handling, or rank-specific optimizer behavior can break this invariant.

FSDP and ZeRO change the memory-communication point rather than the underlying optimization objective. They reconstruct state when it is needed and return it to shards afterward, trading more intricate communication and scheduling for lower per-rank memory.

## 17. Column tensor parallelism and the all-gather/reduce-scatter duality

**Transcript coverage:** lines 1786-1962

### What the lecturer said - transcript only

Tensor parallelism does not cut the batch in this example. Every rank receives the full data for simplicity, while each rank owns part of every layer. The input activation has shape `batch_size x num_dim`. With $p$ ranks, each layer's `num_dim x num_dim` weight matrix is split by columns, so a rank stores a `num_dim x local_num_dim` slice where `local_num_dim = num_dim / p`. The lecturer notes that row partitioning also exists but does not cover it here.

Each rank multiplies the full input by its local weight slice and applies the elementwise GeLU. The resulting local activation has shape `batch_size x local_num_dim`. Since the next sharded layer needs the full-width input, ranks allocate space for every activation shard, all-gather those shards, and concatenate them along the feature dimension to recover `batch_size x num_dim`. This communication occurs after every layer.

Unlike data parallelism, tensor parallelism is not model-agnostic. It reaches inside the model and relies on the algebraic fact that one large matrix multiplication can be decomposed into smaller multiplications on parameter slices, followed by communication that assembles their outputs.

In response to a question about backward propagation, the lecturer says the forward all-gather is paired with reduce-scatter in backward. The two operations have a dual relationship: activations are gathered on the forward path, and their gradient contributions are reduced and redistributed on the backward path.

Another question asks whether autograd performs this communication automatically. A plain `.backward()` call does not invent distributed communication. Higher-level PyTorch facilities can automate it, but in the from-scratch implementation shown here the programmer would explicitly manage the reduce-scatter. The lecture omits that backward implementation so the forward mechanism remains clear.

### Source reconciliation

The slide marks the tensor-parallel backward pass as a homework exercise, whereas the spoken Q&A states the required high-level idea: pair the forward all-gather with a backward reduce-scatter. These sources are complementary rather than contradictory; the deck omits code, not the communication requirement.

### Additional explanation

For a column-partitioned weight matrix,

$$
W=[W_0\;W_1\;\cdots\;W_{p-1}],
$$

so ordinary matrix multiplication decomposes as

$$
XW=[XW_0\;XW_1\;\cdots\;XW_{p-1}].
$$

Rank $r$ can compute $Y_r=XW_r$ locally. Concatenating all $Y_r$ reconstructs the same $Y$ that an unsharded multiplication would produce. An elementwise activation can be applied before that concatenation because it acts independently on each coordinate.

In the displayed layout, every rank consumes the gathered activation in its next local matrix multiplication. Backward therefore produces multiple contributions associated with the replicated gathered value. The communication adjoint of all-gather is reduce-scatter: it adds contributions and returns the shard belonging to each source rank. Exact production collectives vary with column-versus-row layouts and with whether intermediate activations remain sharded, so "tensor parallelism uses all-gather" is not a universal per-layer recipe. The more general rule is to follow the tensor algebra and make each ownership transition explicit.

Tensor parallelism lowers per-rank parameter storage and distributes a layer's matrix multiplications, but it places activation communication on the critical path of nearly every layer. That frequency is why it demands a fast, low-latency interconnect.

## 18. Pipeline parallelism, microbatches, and bubbles

**Transcript coverage:** lines 1963-2087

### What the lecturer said - transcript only

Pipeline parallelism cuts the network by depth. Each rank owns a subset of complete, full-width layers rather than a slice of every layer. The first stage receives the data, applies its local layers, and sends its output activations to the next stage. A later rank receives from `rank - 1`, computes its assigned layers, and sends to `rank + 1`. The example uses direct point-to-point `recv` and `send` calls rather than a collective.

The batch is also split into **microbatches**. Rank 0 chunks the input; downstream ranks allocate buffers for the incoming activations. For each microbatch, a stage receives, computes, and sends. Splitting a deep network this way is natural, but a naive schedule creates **pipeline bubbles**: stages sit idle while waiting for an upstream activation or for downstream work to clear. Smaller microbatches let the first stage send work onward sooner and begin the next piece, which reduces idle gaps.

The displayed implementation still misses a crucial optimization: communication/computation overlap. Blocking send and receive leave opportunities for stages to wait. Asynchronous send and receive require additional management but can move the next activation while other computation proceeds. The aim is to hide communication behind useful compute and reduce total waiting time.

### Source reconciliation

The code slides instantiate the illustration with 2 ranks, 4 layers, and 4 microbatches. They explicitly label the backward pass a homework exercise and state that overlapping communication with computation is not handled. The transcript explains the same limitations without depending on those exact example counts.

### Additional explanation

For $s$ stages and $m$ equal microbatches, a simplified forward-only pipeline with equal stage times takes about $m+s-1$ stage intervals. Its idealized utilization is

$$
U_{\mathrm{pipeline}}\approx\frac{m}{m+s-1}.
$$

Increasing $m$ reduces the relative fill-and-drain bubble. It is not free: smaller microbatches can make matrix multiplications less efficient, increase scheduling overhead, and require more activation bookkeeping. A practical schedule balances bubble reduction against per-microbatch efficiency and memory.

A complete training pipeline must schedule both forward activations and backward gradients. Different schedules trade peak activation memory, bubble size, and synchronization complexity. The lecture's forward-only example establishes the essential ownership change: full-width activations cross stage boundaries, while parameters stay with their assigned depth segment.

---

# Part V - Composing strategies and matching them to hardware

## 19. Overlap, additional axes, topology, and critical batch size

**Transcript coverage:** lines 2088-2199

### What the lecturer said - transcript only

Communication/computation overlap is missing not only from the toy pipeline but also from the simple DDP loop. That loop finishes the whole backward pass and then all-reduces gradients one parameter at a time. A better schedule begins communicating a gradient as soon as backward has produced it, while backward continues through earlier layers. Assignment 2 explores this form of overlap.

The MLP examples contain most of the basic distributed mechanisms. Adding attention and a full language model introduces substantially more bookkeeping, but does not replace the central ideas. The lecture also leaves several axes for later:

- **sequence parallelism** splits a sequence and can distribute attention computation;
- **expert parallelism** splits the experts in an MoE and uses the all-to-all routing introduced earlier;
- practical systems combine multiple parallelization methods.

The choice depends strongly on hardware. Tensor parallelism communicates large activations at every layer, so it is generally kept inside a fast NVLink domain. It is unattractive across a slow network. Pipeline parallelism can tolerate much slower links and therefore appears in decentralized training where GPUs may be geographically separated. A common composition is tensor parallelism within a node or fast domain, data parallelism or FSDP across a wider domain, and pipeline parallelism when another depth cut is needed.

Scaling data parallelism also encounters an optimization limit. Increasing the number of data-parallel ranks usually increases the global batch. Eventually training reaches a **critical batch size** beyond which a larger batch no longer provides a proportional benefit. Additional data-parallel compute is then wasted, and using those devices for tensor parallelism can be preferable. The correct strategy is therefore governed by both network topology and the statistical efficiency of the training batch.

### Additional explanation

Gradient overlap is often organized with buckets. Instead of launching one tiny collective per parameter or waiting for every gradient, a system groups gradients into communication-sized buffers. As soon as a bucket is ready, its all-reduce or reduce-scatter can overlap with continued backward computation. Bucket size trades launch overhead against how early communication can begin.

A topology-aware placement policy follows communication frequency:

| Parallel dimension | Typical traffic frequency | Preferred placement |
|---|---|---|
| Tensor parallel | Within many or every layer | Fastest links, commonly one NVLink domain |
| Expert parallel | At routed expert layers | High bisection bandwidth; balance is critical |
| Data parallel or FSDP | During backward and parameter materialization | Can span a wider domain, subject to state and traffic volume |
| Pipeline parallel | At stage boundaries | Can cross slower links if stage compute is large enough to amortize traffic |

Critical batch size adds a statistical constraint to this systems table. More data-parallel workers can reduce wall-clock time only if the training procedure can use the enlarged effective batch without losing useful update frequency or sample efficiency. Parallelism is therefore not a purely mechanical exercise in filling available GPUs.

## 20. Compiler-managed sharding as an alternative abstraction

**Transcript coverage:** lines 2200-2231

### What the lecturer said - transcript only

The course intentionally uses PyTorch collective operations at a primitive level so that students can see what distributed execution does mechanically. A different style, associated especially with JAX and TPUs, lets the programmer define the model and a sharding strategy. The compiler then infers which communication operations are required from where each piece of data must reside. This is appealing because it hides much of the manual communication logic, but it would also hide the from-scratch mechanisms the course wants students to understand.

### Source reconciliation

The transcript at line 2214 says "starting strategy," while the slide says **sharding strategy**. The surrounding discussion of data placement and compiler-inferred communication confirms that "sharding" is intended.

### Additional explanation

Compiler-managed sharding and explicit collectives solve the same ownership problem at different abstraction levels:

- with explicit collectives, the programmer says when to all-gather, reduce-scatter, or send;
- with declarative sharding, the programmer annotates desired tensor layouts, and the compiler inserts layout-changing communication.

The declarative approach can optimize across operation boundaries and reduce boilerplate, but performance still depends on understanding the underlying costs. A sharding annotation that forces an all-gather at every layer remains expensive even if the compiler inserts it automatically. Learning the primitives makes compiler output easier to predict and diagnose.

## 21. The persistent memory-compute-communication trade-off

**Transcript coverage:** lines 2232-2283

### What the lecturer said - transcript only

There are many ways to parallelize: cut the batch with data parallelism, cut model width or experts with tensor and expert parallelism, cut depth with pipeline parallelism, or cut sequence length with sequence parallelism. This lecture implements only DDP among the data-parallel family; FSDP and ZeRO are deferred to the next lecture.

Tensor parallelism requires very fast interconnects. Pipeline parallelism can use slower links, but only if its pipeline bubbles are controlled. These strategies extend a broader systems trade-off already seen in activation checkpointing. A value can be recomputed, stored in local memory, or stored in another GPU's memory and communicated when needed.

DDP makes one particular choice: every rank redundantly stores and updates all parameters and keeps its own optimizer state. That replication avoids moving optimizer state between ranks during each update, but costs memory on every GPU. Other schemes move to a different point on the same memory-versus-communication spectrum.

Hardware will continue getting faster, but desired models will also continue growing. For that reason, the hierarchical structure and the need to reason about it are unlikely to disappear. The following lecture will examine more advanced parallelism techniques in greater depth.

### Additional explanation

The central design question can be stated for every tensor or model state:

1. **Replicate it:** spend memory to make local access cheap.
2. **Shard it:** save memory, then communicate before or during use.
3. **Recompute it:** save storage and communication, then spend extra FLOPs.
4. **Offload it:** place it in a slower memory tier and pay transfer latency and bandwidth.

No option dominates universally. Parameters are reused extensively in forward and backward, activations have layer-specific lifetimes, gradients become ready in reverse order, and optimizer state is touched at update time. Effective systems choose separately for each object and schedule the resulting transfers around computation.

This is the distributed analogue of tiling. Tiling arranges local computation so a loaded value is reused before eviction. Parallelism arranges ownership so a remotely acquired value performs enough useful work before it must move again.

---

# Consolidated takeaways

1. Multi-GPU training extends the same principle as kernel optimization: computation should be organized to avoid expensive data movement.
2. Multiple GPUs are used either because model state does not fit on one device or because additional compute can reduce training time. The two motives lead to different design constraints.
3. A rank is one participating process or device, and world size is the number of participants.
4. Broadcast, scatter, gather, and reduce establish the collective vocabulary; all-gather, reduce-scatter, and all-reduce are the main training workhorses.
5. All-reduce logically equals reduce-scatter followed by all-gather. Separating those phases makes sharded storage schemes such as FSDP and ZeRO possible.
6. All-to-all redistributes source-by-destination blocks and is central to data-dependent expert routing. Its performance depends heavily on balanced traffic.
7. The hardware hierarchy runs from local HBM through NVLink/NVSwitch to InfiniBand and Ethernet. HBM, previously treated as slow, is fast relative to remote GPU memory.
8. RDMA is direct remote-memory access semantics; NVLink/NVSwitch, InfiniBand, and RoCE are mechanisms that can provide it.
9. NCCL translates collective intent into topology-aware communication paths and GPU kernels. PyTorch supplies the framework-level interface and chooses NCCL or Gloo as a backend.
10. GPU completion and process rendezvous are separate synchronization problems. CUDA synchronization and a distributed barrier serve different purposes.
11. Effective bandwidth is a normalized traffic rate, not a topology-free physical law. The formula may omit topology while the measured duration still reflects it.
12. DDP shards the batch but replicates model and optimizer state. Averaging gradients makes each equal-sized local batch contribute to the global-batch update.
13. Tensor parallelism shards layer algebra and communicates activations frequently, making fast interconnects essential.
14. Pipeline parallelism assigns complete layers to stages. Microbatches and communication/computation overlap reduce, but do not automatically eliminate, pipeline bubbles.
15. Sequence and expert parallelism add cuts along length and experts; practical training systems compose several parallel dimensions.
16. The best placement follows the physical topology: high-frequency tensor traffic stays on the fastest links, while pipeline boundaries can cross slower networks when enough computation amortizes each transfer.
17. Data parallelism cannot scale indefinitely if the required global batch exceeds the critical batch size.
18. Explicit collectives and compiler-managed sharding are different interfaces to the same tensor-ownership transitions.
19. Replication, sharding, recomputation, and offload exchange memory, FLOPs, bandwidth, and implementation complexity. Faster hardware does not remove this hierarchy because model ambitions grow with it.

# Key equations

## Sum all-reduce

For $p$ ranks, each holding $x_r$:

$$
y=\sum_{r=0}^{p-1}x_r,
$$

with $y$ stored on every rank.

## Reduce-scatter plus all-gather

If $\operatorname{shard}_r(y)$ is rank $r$'s output shard, then:

$$
\operatorname{reduce\text{-}scatter}(x_0,\ldots,x_{p-1})_r
=\operatorname{shard}_r\left(\sum_{i=0}^{p-1}x_i\right),
$$

and

$$
\operatorname{all\text{-}gather}
\left(\operatorname{shard}_0(y),\ldots,\operatorname{shard}_{p-1}(y)\right)
=y
$$

on every rank. Logically,

$$
\operatorname{all\text{-}reduce}
=\operatorname{all\text{-}gather}\circ\operatorname{reduce\text{-}scatter}.
$$

## Lecture's all-reduce effective-bandwidth convention

For payload size $S$ bytes, world size $p$, and duration $t$:

$$
B_{\mathrm{eff,AR}}
=\frac{2S(p-1)}{pt}
=\frac{2S}{t}\frac{p-1}{p}
\xrightarrow[p\to\infty]{}\frac{2S}{t}.
$$

The measured $t$ still depends on hardware, placement, topology, contention, message size, and backend choices.

## DDP global gradient for equal local batches

$$
g
=\frac{1}{p}\sum_{r=0}^{p-1}g_r
=\frac{1}{p}\sum_{r=0}^{p-1}\nabla_\theta L_r(\theta).
$$

Each replica then applies the same update:

$$
\theta_{k+1}=U(\theta_k,g_k).
$$

## Column tensor parallelism

$$
W=[W_0\;W_1\;\cdots\;W_{p-1}],
$$

$$
XW=[XW_0\;XW_1\;\cdots\;XW_{p-1}].
$$

Each rank computes one block $XW_r$; concatenation or an equivalent layout operation reconstructs the full output.

## Simplified forward-pipeline utilization

For $m$ microbatches and $s$ equal-time stages:

$$
U_{\mathrm{pipeline}}\approx\frac{m}{m+s-1}.
$$

This ignores backward scheduling, unequal stage times, communication overhead, and microbatch-efficiency effects.

# Glossary

| Term | Meaning in this lecture |
|---|---|
| All-gather | Collect shards from all ranks and place the complete collection on every rank. |
| All-reduce | Reduce values across ranks and replicate the full reduced result on every rank. |
| All-to-all | Have every rank send a designated payload to every destination rank. |
| Backend | Implementation selected beneath `torch.distributed`, such as NCCL for GPUs or Gloo for CPUs. |
| Barrier | Rendezvous at which each participating process waits for all the others. |
| Broadcast | Copy a tensor from one root rank to all ranks. |
| Collective operation | A communication pattern defined over a group of ranks. |
| Critical batch size | Batch-size regime beyond which further enlargement gives little or no proportional training benefit. |
| CUDA synchronization | Waiting for previously issued local GPU work to complete. |
| Data parallelism | Partition the batch while maintaining model replicas or sharded model states. |
| DDP | Distributed data parallelism; in this lecture, full model replicas synchronized by gradient all-reduce. |
| Effective bandwidth | Implied communication bytes divided by measured time under a stated counting convention. |
| Expert parallelism | Partition MoE experts across devices, commonly routing activations with all-to-all. |
| FSDP | Fully Sharded Data Parallel, a scheme that shards model states and materializes them as needed. |
| Gather | Collect distributed pieces at one designated root. |
| Gloo | PyTorch distributed backend used here for CPU execution. |
| HBM | High Bandwidth Memory local to a GPU. |
| InfiniBand | High-performance inter-node network that supports RDMA. |
| Microbatch | Subdivision of a batch scheduled separately through pipeline stages. |
| NCCL | NVIDIA Collective Communications Library, which implements topology-aware GPU collectives. |
| NVLink | NVIDIA high-bandwidth GPU interconnect. |
| NVSwitch | Switching fabric connecting GPUs in an NVLink domain. |
| Pipeline bubble | Time during which a pipeline stage is idle while the pipeline fills, drains, or waits. |
| Pipeline parallelism | Partition model depth so different ranks own different layer groups. |
| Rank | Identifier of a participating process or device; one GPU in this lecture. |
| RDMA | Remote Direct Memory Access, allowing direct remote memory reads or writes without CPU staging. |
| Reduce | Combine values from ranks using an operation such as sum, min, or max and place the result at a root. |
| Reduce-scatter | Reduce corresponding values across ranks and leave different output shards on different ranks. |
| RoCE | RDMA over Converged Ethernet. |
| Scatter | Divide a root-owned tensor and distribute one shard to each rank. |
| Sequence parallelism | Partition the sequence-length dimension across devices. |
| Tensor parallelism | Partition tensors or layer algebra, such as a weight matrix's columns, across ranks. |
| World size | Number of ranks participating in the process group. |
| ZeRO | A family of data-parallel memory optimizations that shards optimizer, gradient, and parameter state by stage. |

# Self-check questions

1. Why does the lecture call HBM slow in a single-GPU discussion but fast in a multi-GPU discussion?
2. What are the two distinct reasons for moving from one GPU to many?
3. What do rank and world size mean in the course's execution model?
4. How do broadcast, scatter, gather, and reduce change tensor ownership?
5. Why does the prefix `all-` change gather or reduce?
6. Reproduce the reduce-scatter outputs for the lecture's four input vectors.
7. Why is all-reduce logically equivalent to reduce-scatter followed by all-gather?
8. Why does separating all-reduce into two phases enable sharded model-state schemes?
9. In what sense does a balanced all-to-all resemble a matrix transpose?
10. Why is load balance both a communication and a compute concern in an MoE?
11. How is RDMA different from NVLink, InfiniBand, or RoCE?
12. What work does NCCL perform after the application requests a collective?
13. Why are `MASTER_ADDR` and `MASTER_PORT` not the path for bulk NCCL payloads?
14. What is the difference between waiting on CUDA work and waiting at a distributed barrier?
15. Why does asynchronous launch not guarantee useful communication/computation overlap?
16. Derive the lecture's effective all-reduce bandwidth expression and explain what its topology independence does and does not mean.
17. Under what assumptions does averaging equal-rank gradients reproduce the gradient of a global batch?
18. Why must DDP replicas begin with the same parameters and apply the same optimizer update?
19. Why does ordinary DDP fail to solve a model that cannot fit on one GPU?
20. Derive the column-partitioned identity $XW=[XW_0\;\cdots\;XW_{p-1}]$.
21. Why is reduce-scatter paired with forward all-gather in the tensor-parallel backward pass?
22. How do microbatches reduce pipeline bubbles, and what costs appear when microbatches become too small?
23. Why is tensor parallelism usually confined to the fastest interconnect domain?
24. How can critical batch size make more data parallelism statistically wasteful?
25. What information does declarative sharding expose to a compiler, and what communication cost remains even when the compiler inserts the collectives?
26. For a given tensor, how do replication, sharding, recomputation, and offload trade memory, FLOPs, and bandwidth?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-196 | Single-GPU to multi-GPU hierarchy, two scaling motives, communication trade-off, lecture execution and plan | Yes |
| 197-283 | Collective-programming history, rank, world size, foundational and workhorse operations | Yes |
| 284-406 | Broadcast, scatter, gather, reduce, examples, and NumPy-broadcasting Q&A | Yes |
| 407-520 | All-gather, reduce-scatter, all-reduce, examples, gradient use, ZeRO/FSDP foreshadowing | Yes |
| 521-600 | All-to-all mechanics, balanced transpose view, unbalanced splits, MoE routing and load balance | Yes |
| 601-660 | Terminology summary, root-rank Q&A, conceptual-versus-code Q&A | Yes |
| 661-888 | Classic and modern topologies, bandwidth hierarchy, CPU Ethernet path, RDMA, NVL72, RoCE | Yes |
| 889-1055 | NCCL role and hardware Q&A on racks, RDMA, multi-node optimization, a ninth GPU, and TPUs | Yes |
| 1056-1175 | `torch.distributed`, NCCL/Gloo, spawn and tracing, setup metadata, asynchronous workers, barriers | Yes |
| 1176-1239 | PyTorch all-reduce example, nondeterministic print order, in-place mutation, async option, rank Q&A | Yes |
| 1240-1339 | Reduce-scatter API, asynchronous-overlap Q&A, all-gather demonstration, cleanup | Yes |
| 1340-1515 | Collective benchmarking, warm-up, timing, effective-bandwidth accounting, 400 GB/s example, AR-versus-RS comparison | Yes |
| 1516-1561 | CUDA-synchronization and barrier-order Q&A | Yes |
| 1562-1600 | MLP motivation and the batch, width, and depth cuts | Yes |
| 1601-1716 | DDP data slicing, replicated model and optimizer, local forward/backward, averaged gradient all-reduce | Yes |
| 1717-1785 | DDP batch-size Q&A, Transformer modularity, synchronization invariant, full-model memory limit | Yes |
| 1786-1962 | Column tensor parallelism, per-layer activation all-gather, model-specific algebra, backward and autograd Q&A | Yes |
| 1963-2087 | Pipeline layer partition, microbatches, point-to-point send/receive, bubbles, asynchronous overlap | Yes |
| 2088-2199 | DDP overlap, sequence/expert/composed parallelism, topology-sensitive placement, critical batch size | Yes |
| 2200-2231 | Primitive PyTorch pedagogy versus compiler-managed JAX/TPU sharding | Yes |
| 2232-2283 | Parallel-axis summary, FSDP/ZeRO preview, recompute/store/remote-store trade-off, DDP replication, persistent hierarchy | Yes |
