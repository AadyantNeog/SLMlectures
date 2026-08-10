---
title: "Lecture 6 - Kernels, Triton, and XLA"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 6
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 6 Kernels, Triton, XLA.txt"
slide_deck: "../lecture_06.pdf"
status: "complete"
---

# Lecture 6: Kernels, Triton, and XLA

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves claims, examples, qualifications, numerical details, and substantive audience questions while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, derivation, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are shown for auditability. The complete transcript was mapped before the slides were inspected. All 22 slides were then rendered and visually checked to verify hardware numbers, code, kernel names, pointer arithmetic, diagrams, and tile shapes. Material differences between speech and slides are recorded under `Source reconciliation`.

## Source scope note

The transcript filename includes `XLA`, but neither the recorded lecture nor the 22-slide deck contains a substantive XLA section. The delivered lecture focuses on NVIDIA GPU hardware, benchmarking and profiling, Triton, PTX, reductions, and tiled matrix multiplication. These notes do not invent an XLA segment or attribute unspoken XLA material to the lecturer. A short compiler comparison appears only as clearly marked supplementary context.

## Lecture map

The lecture develops a practical path from hardware constraints to efficient GPU kernels:

1. Review the GPU memory hierarchy and the thread/block/grid programming model.
2. Explain why warps, register pressure, bank conflicts, memory coalescing, and launch geometry control performance.
3. Benchmark first, profile to identify the cause, change the code, and measure again.
4. Use GELU to show why kernel fusion and `torch.compile` can eliminate repeated HBM traffic.
5. Introduce Triton's block-level abstraction and inspect the PTX generated beneath it.
6. Progress from an elementwise kernel to row softmax, tiled row reduction, and tiled matrix multiplication with a fused activation.

---

# Part I - GPU hardware and the programming model

## 1. The memory hierarchy sets the performance environment

**Transcript coverage:** lines 1-246

### What the lecturer said - transcript only

The lecture continues the previous high-level treatment of GPUs by moving into code, Triton kernels, benchmarking, and profiling. The discussion focuses on NVIDIA GPUs, while acknowledging TPUs, AMD GPUs, and other accelerators.

Across generations such as A100, H100, and B200, a GPU contains on the order of 100 to 200 **streaming multiprocessors (SMs)**. Each SM has a register file and a combined L1-cache/shared-memory resource. Shared memory is explicitly managed by the program, while L1 caching is hardware-managed. A larger L2 cache is shared across the whole chip, and high-bandwidth memory (HBM) provides the largest storage.

The hierarchy trades capacity for proximity and speed:

- registers are local to an SM, very fast, and very small;
- L1 cache and shared memory are also SM-local, slightly larger, and slower than registers;
- L2 is chip-wide and larger again;
- HBM is global and very large, but has the lowest bandwidth and highest latency in this hierarchy.

HBM capacity has grown substantially across accelerator generations. B200 HBM bandwidth is about 8 TB/s, which is enormous in absolute terms but still slow relative to on-chip storage. The mental model is therefore: large memory is far away and slower, while fast local memory is scarce.

### Additional explanation

The slide table makes the scale concrete:

| Resource | A100 | H100 | B200 |
|---|---:|---:|---:|
| SMs | 108 | 132 | 148 |
| Register file per SM | 256 KB | 256 KB | 256 KB |
| L1 plus shared memory per SM | 192 KB | 256 KB | 256 KB |
| L2 cache | 40 MB | 50 MB | 96-126 MB |
| HBM capacity | 80 GB | 80 GB | 192 GB |
| HBM bandwidth | about 2 TB/s | about 3.35 TB/s | about 8 TB/s |

The B200's 65,536 32-bit registers occupy 256 KiB per SM. Counting entries and counting bytes are two descriptions of the same register file.

The performance objective is to load data from HBM infrequently, reuse it in registers or shared memory, and write final results back. Kernel fusion and tiling are two ways to increase that reuse.

## 2. Threads, thread blocks, and grids

**Transcript coverage:** lines 247-567

### What the lecturer said - transcript only

GPU kernels use a hierarchy of parallel work:

1. A **thread** executes code on a small part of the data.
2. Threads are grouped into **thread blocks**, also called cooperative thread arrays (CTAs).
3. A collection of thread blocks forms the **grid** launched by a kernel.

For an elementwise operation such as GELU, the natural mapping is one thread per element. More complicated operations such as softmax and matrix multiplication require values handled by different threads to interact. Those threads could communicate by repeatedly reading and writing HBM, but that would be slow.

A thread block provides the relevant collaboration unit. Its threads run together on one SM and can use that SM's shared memory. A block loads a region from HBM, lets its threads operate and communicate through local memory, and writes the result back. The later treatment of tiling follows directly from this pattern.

Triton encourages programmers to think natively in blocks rather than begin from individual threads. The lecturer argues that this becomes natural once the abstraction is familiar.

The hardware has additional concepts, including thread-block clusters on H100 and B200 and tensor memory on B200. Some are partly hidden from the programmer and are omitted from the introductory model.

### Additional explanation

The block is simultaneously:

- a scheduling unit that remains on one SM for its lifetime;
- a synchronization domain for its participating threads;
- a scope for explicitly managed shared memory.

There is no ordinary shared-memory communication across independent blocks. An algorithm that needs a global reduction must therefore use multiple kernel phases, atomics, or a specialized cooperative mechanism rather than assume all blocks can synchronize at will.

## 3. Correctness is abstracted; performance is not

**Transcript coverage:** lines 568-699

### What the lecturer said - transcript only

The grid/block/thread programming model is deliberately cleaner than the physical GPU. A programmer can write a correct kernel by defining the grid and the work of each block or thread without knowing every hardware detail.

High performance is different. Kernel speed is highly sensitive to the exact accelerator: memory capacities, register limits, scheduling, and execution units all matter. Since the reason to write custom kernels is to improve performance, a kernel author must understand both the computation expressed by the programming model and the hardware on which it runs.

The following examples are intended to give a flavor of the constraints rather than an exhaustive hardware manual.

### Additional explanation

This is a useful separation of contracts:

- The **semantic contract** says which output values a kernel computes.
- The **performance contract** depends on launch shape, memory layout, reuse, instruction selection, occupancy, and a particular GPU generation.

A kernel can therefore be portable in correctness while being non-portable in speed. Autotuning and architecture-specific kernel variants exist because one launch configuration rarely remains optimal across every shape and device.

## 4. Warps, control divergence, and latency hiding

**Transcript coverage:** lines 700-900

### What the lecturer said - transcript only

Threads inside a block are physically grouped into **warps** of 32 threads. A 64-thread block therefore contains two warps. Threads within one warp execute the same instruction in lockstep.

When threads in a warp take different branches, the warp experiences **control divergence**. If some threads require branch A and others require branch B, the hardware executes the relevant A path and B path sequentially under different masks. This wastes parallel execution and is why branching inside a warp is generally undesirable.

An SM keeps multiple warps resident and has a warp scheduler. If one warp waits on a long HBM access, which the lecturer describes as potentially taking on the order of 100 cycles, the SM can immediately issue useful instructions from another ready warp. This low-cost switching hides latency rather than leaving the arithmetic units idle.

### Additional explanation

Divergence is harmful only when active lanes disagree. A branch whose condition is uniform across the warp does not serialize paths in the same way. Divergence cost also depends on how much work lies on each path.

Latency hiding requires enough ready warps and enough independent work. High occupancy can help provide those warps, but register and shared-memory use constrain how many can be resident.

## 5. Register pressure, warp occupancy, and thread coarsening

**Transcript coverage:** lines 901-1149

### What the lecturer said - transcript only

Each thread can use at most 255 registers, while every SM has a fixed register file. If each thread consumes more registers, fewer threads, blocks, and warps can reside on the SM. This lowers **warp occupancy**.

Lower occupancy is not automatically bad. A thread that performs more useful work may justify its additional registers. **Thread coarsening** deliberately assigns several elements to each thread, for example eight instead of one. This reduces the number of threads and scheduling overhead while giving each thread a larger job.

The lecture works through an example with 128 threads per block and 160 registers per thread on an SM with 65,536 registers and a maximum of 64 resident warps:

$$
R_{\text{block}} = 128 \times 160 = 20{,}480\ \text{registers}.
$$

Only three such blocks fit by the register limit. Each block has four warps, so 12 warps can be resident. The occupancy is therefore approximately

$$
\frac{12}{64} = 18.75\%.
$$

The example shows how a memory resource can indirectly restrict available computation.

### Source reconciliation

Slide 4's prose bullet says "64 threads," but the displayed code sets `num_threads_per_block = 128`, matching the spoken example and the $20{,}480$-register calculation. The notes use 128. Interestingly, the mistaken 64-thread prose can also yield 12 warps through six smaller blocks, but it is not the configuration actually derived in speech or code.

### Additional explanation

A simplified register-limited calculation is

$$
B_{\text{resident}}
= \left\lfloor
\frac{R_{\text{SM}}}{T_{\text{block}}R_{\text{thread}}}
\right\rfloor,
$$

subject to separate limits on blocks, threads, warps, and shared memory. Occupancy is then resident warps divided by the hardware maximum.

Register spilling is another risk: if a thread needs more registers than can be allocated, some values may spill to much slower memory. A configuration with nominally more occupancy can still lose if it causes extra spills or reduces per-thread reuse.

## 6. Shared-memory bank conflicts

**Transcript coverage:** lines 1150-1350

### What the lecturer said - transcript only

Shared memory is divided into 32 banks, each four bytes wide. In one cycle, a bank can ordinarily service one thread access, with a special exception when threads read the exact same location. If threads in a warp request different addresses that map to the same bank, the requests are serialized. This is a **bank conflict**.

In the worst case, 32 threads try to read a column whose layout maps every requested value to one bank. This produces a 32-way conflict and discards the expected parallelism.

Simply accessing rows instead is not always possible. Matrix multiplication needs rows from one operand and columns from the other, while transpose-like operations also change access direction. **Swizzling** can rearrange shared-memory addresses so that accesses that would collide are distributed across banks.

Profiling can expose bank conflicts, just as it can expose occupancy problems.

### Additional explanation

For 32-bit values in a simple linear layout, a useful first model is

$$
\operatorname{bank}(i) = i \bmod 32.
$$

Consecutive indices then land in different banks, while indices separated by 32 land in the same bank. Padding a shared-memory tile or applying a more elaborate swizzle changes the mapping without changing the mathematical matrix.

Bank conflicts concern access to **shared memory**. They should not be confused with non-coalesced access to HBM, which is the next constraint.

## 7. Coalescing global-memory accesses

**Transcript coverage:** lines 1351-1476

### What the lecturer said - transcript only

When the 32 threads of a warp access HBM, nearby accesses are combined into cache-line transactions. The slide's example uses a 128-byte line. For 32 adjacent 4-byte values, all threads can be served by one fully coalesced transaction.

If those threads instead walk down a column of a row-major matrix, their addresses are far apart. The hardware fetches many cache lines containing values the warp does not use, wasting bandwidth.

This resembles the visual pattern of a bank conflict but occurs at a different level:

- bank conflicts concern shared memory;
- coalescing concerns HBM transactions.

### Additional explanation

Coalescing measures how effectively requested bytes fill each global-memory transaction. A kernel can be mathematically correct and still move many times more HBM data than necessary because its lanes access strided addresses.

Layout decisions should therefore be made with the thread mapping in mind. The fastest-varying logical dimension is usually assigned to adjacent lanes so their accesses are contiguous.

## 8. Block waves, tail effects, and the question of sharing an SM

**Transcript coverage:** lines 1477-1728

### What the lecturer said - transcript only

Thread blocks are scheduled onto a finite number of SMs in waves. A B200 has 148 SMs. If a kernel launches 160 blocks, the first wave can place 148 blocks, while a second tail wave contains only 12. Most SMs are idle during that tail, producing low **block occupancy** or wave quantization.

A rough mitigation is to choose a block count that divides the number of SMs more evenly. The exact geometry remains constrained by the work and by each block's resource use.

An audience member asks whether blocks can share an SM so the tail can be spread out. The lecturer says the answer depends on block demands. If one block already uses most of an SM's tensor-core capacity, placing another there will not improve throughput. A block must also remain intact rather than being split across several SMs. The practical response is to change block size or the total number of blocks so the final wave is less uneven.

The lecturer summarizes the conceptual ownership hierarchy:

- the grid can access global HBM;
- a block has access to its SM's shared memory;
- a thread owns its registers.

Warps, bank conflicts, coalescing, register use, and occupancy are hardware details layered beneath that elegant programming model. Profilers reveal part of the behavior, but scheduling and exact hardware limits can still make performance messy.

### Additional explanation

Multiple blocks can be resident on one SM when registers, shared memory, thread count, and block-count limits permit it. The lecture's one-block-per-SM wave picture is a simplified capacity model for explaining the tail. Blocks still cannot be split across SMs.

Tail effects matter most when blocks have similar, long runtimes and the final wave is very sparse. Increasing the number of smaller blocks can smooth the tail, but smaller blocks may reduce data reuse or increase overhead. Launch geometry is therefore another measured tradeoff rather than a rule to maximize one metric.

---

# Part II - Benchmark before optimizing

## 9. The measure-change-measure loop

**Transcript coverage:** lines 1729-1866

### What the lecturer said - transcript only

The lecturer's recipe for success is:

1. benchmark and profile the code;
2. make a change;
3. benchmark and profile again.

Measurement should precede custom-kernel work. A benchmark gives end-to-end wall-clock time. It does not explain where time is spent, but it measures the outcome that ultimately matters and reduces performance to a number that can be compared across implementations or plotted as dimensions change.

PyTorch offers benchmarking utilities, but the lecture builds a small benchmark from scratch to expose the important pitfalls.

### Additional explanation

Benchmarking and profiling answer different questions:

- **Benchmark:** Did the change improve the target metric?
- **Profile:** Which operations, transfers, or stalls explain the result?

A profile can suggest an optimization, but only a benchmark on the intended workload establishes whether the optimization helped. Measurement after every change also catches regressions caused by a different shape or hardware generation.

## 10. Warmup, CUDA events, synchronization, and scaling curves

**Transcript coverage:** lines 1867-2110

### What the lecturer said - transcript only

The example benchmark pre-allocates two random square matrices and returns a closure that performs only their matrix multiplication. Allocation and random-number generation are intentionally outside the timed region.

A correct steady-state GPU benchmark includes several steps:

1. **Warm up the operation.** Some work is compiled or initialized lazily. Initial cost should not contaminate the repeated-operation measurement.
2. **Run multiple trials.** Timings vary, so one observation is not enough.
3. **Use CUDA timing events.** Record a start event, execute the operation, and record an end event.
4. **Synchronize.** GPU work is asynchronous, so the CPU must wait until the device work has completed before reading the elapsed time.
5. **Aggregate the trials.** The lecture takes the mean, while noting that a careful analysis may inspect the entire distribution or a statistic such as the 95th percentile.

Plotting time against square-matrix dimension reveals two regimes. For small matrices, runtime is nearly constant because launch and underutilization effects dominate. In the displayed experiment, the curve does not clearly enter cubic scaling until dimensions approach roughly 2,000. Tiny matrices map poorly onto hardware designed for large matrix multiplications.

### Additional explanation

The benchmark wrapper has an important design property: it times the same callable repeatedly without recreating its inputs. This isolates the kernel from setup work.

For a square matmul, mathematical work is $O(n^3)$, but asymptotic work predicts runtime only after the device is sufficiently occupied. At small $n$, fixed launch overhead, dispatch, and minimum kernel latency can form a performance floor.

A representative timing skeleton is:

```python
# Warmup
for _ in range(num_warmups):
    run()
torch.cuda.synchronize()

times_ms = []
for _ in range(num_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    torch.cuda.synchronize()
    times_ms.append(start.elapsed_time(end))
```

For production benchmarking, inputs, dtype, strides, device clocks, competing work, and whether gradients are enabled should also be controlled.

## 11. Profiling reveals the kernels beneath PyTorch

**Transcript coverage:** lines 2111-2412

### What the lecturer said - transcript only

Profiling explains where time is spent and can reveal what a high-level program actually launches. This is valuable even when runtime is not the immediate concern, because frameworks hide implementation details.

PyTorch has a built-in profiler. The assignment uses NVIDIA Nsight for more detail, although the lecture does not demonstrate it fully.

Profiling `A + B` shows a vectorized CUDA addition kernel. Profiling `A @ B` shows a long, specialized matrix-multiplication kernel name. Changing the matrices from $2048 \times 2048$ to $128 \times 128$ selects a different kernel.

The kernel name itself carries implementation information:

- `CUTLASS` identifies NVIDIA's CUDA linear-algebra library;
- `sm100` identifies the Blackwell/B200 architecture target;
- `f32` identifies the data type;
- fields such as `64x64x16` or `32x32x16` identify internal tile shapes.

The lesson is that one PyTorch operator does not correspond to one universal implementation. Shape and hardware select among specialized kernels.

### Additional explanation

Dispatch systems choose kernels using properties such as:

- device architecture;
- dtype and accumulator type;
- matrix dimensions and alignment;
- transposition and strides;
- epilogue operations such as bias or activation.

This explains why performance can change discontinuously at particular dimensions. Crossing an alignment or tile threshold can select a different kernel rather than merely adding a little more arithmetic.

## 12. GELU, graph compilation, and kernel fusion

**Transcript coverage:** lines 2413-2921

### What the lecturer said - transcript only

GELU is an elementwise activation often evaluated with a compute-friendly tanh approximation. The lecture compares three equivalent implementations:

1. a naive PyTorch expression that spells out the approximation from primitive tensor operations;
2. PyTorch's built-in GELU;
3. `torch.compile` applied to the naive function.

The outputs are checked for agreement. Their performance is very different. The naive implementation takes roughly 3.75 in the displayed timing scale, while the built-in and compiled versions are much faster. In this run, the built-in kernel is faster than the compiler-generated version.

The profile explains the difference. Every primitive in the naive PyTorch computation graph launches a separate kernel. Each kernel reads its input from HBM, performs one stage, and writes an intermediate back to HBM for the next kernel. Most time is wasted moving the same tensor repeatedly.

The built-in GELU uses one dedicated CUDA kernel because GELU is common enough that someone implemented it in the standard library. `torch.compile` analyzes the naive graph and emits one fused Triton kernel. Both fused versions read each element from HBM once, perform the GELU stages locally, and write once.

An audience member asks whether the built-in implementation was written in CUDA. From its name, the lecturer infers that it probably was, while not claiming to have inspected its source. Another audience member asks why the Triton kernel is faster. The lecturer corrects the premise: in the displayed run, the compiled Triton kernel is slower than the built-in CUDA kernel. The relative gap had been different in the previous year's experiment and is hardware- and software-dependent; the example is intended to demonstrate fusion, not establish a permanent ranking.

### Additional explanation

If an elementwise expression has $k$ unfused stages over $N$ values, a rough traffic model is $O(kN)$ reads and writes. Fusion can reduce that toward one input read and one output write while keeping intermediate values in registers.

`torch.compile` does not merely remove Python overhead in this example. Its important action is graph transformation and code generation: the profiler shows a fused Triton kernel.

The source title mentions XLA, but the lecture never develops it. As supplementary context only, XLA is another graph compiler that can fuse tensor operations and generate accelerator code, especially in JAX and TPU-oriented stacks. The conceptual similarity is graph-level optimization; the compiler IRs, supported backends, and generated kernel paths differ. No XLA-specific claim in this paragraph is attributed to the lecturer.

---

# Part III - Writing and understanding Triton kernels

## 13. CUDA's thread view versus Triton's block view

**Transcript coverage:** lines 2922-3160

### What the lecturer said - transcript only

CUDA, developed by NVIDIA, asks the programmer to specify what each individual thread does. This thread-level model maps closely to the hardware and offers fine-grained control. Its cost is manual bookkeeping: when threads in a block communicate, the program must manage shared memory and synchronization explicitly.

For a purely elementwise operation, CUDA's individual-thread view is straightforward. The burden grows for reductions and matrix operations.

Triton, developed by OpenAI, asks what each **thread block** does. It is generally powerful enough for the course and for many Transformer kernels. It may not expose every feature of the newest hardware, but that loss of low-level control is not important for the introductory task.

The core Triton mental model is:

1. a block loads a tile from global memory;
2. it operates on the tile in local storage;
3. it writes the result back to global memory.

Triton sits between two abstraction levels. PyTorch treats a whole tensor operation such as matrix multiplication as atomic. CUDA can reason about one element in one thread. Triton operates on intermediate-size blocks, allowing code within a block to resemble vectorized PyTorch.

### Additional explanation

Triton's block program is often described as one **program instance** per output tile. The compiler maps the vectorized block operations to warps, threads, instructions, and memory locations. This raises the level of abstraction without eliminating the need to choose block shapes and access patterns.

The abstraction works especially well when the algorithm can be partitioned into tiles that communicate heavily internally and minimally with other tiles.

## 14. Building an elementwise Triton GELU kernel

**Transcript coverage:** lines 3161-3672

### What the lecturer said - transcript only

The first kernel applies GELU to a vector of 8,000 values. Python first allocates an output tensor because the Triton kernel writes through pointers rather than returning a functional tensor value.

The vector is too large to treat as one block, so it is partitioned into blocks of 1,024 elements. A ceiling division produces eight block instances. Triton's launch syntax supplies the grid shape in brackets, then passes input and output pointers, element count, and block size.

Inside the JIT-compiled kernel:

1. `tl.program_id(0)` identifies the current block.
2. The block start is `pid * BLOCK_SIZE`.
3. `tl.arange(0, BLOCK_SIZE)` creates the block-local positions.
4. Adding start and local positions produces global offsets.
5. `offsets < num_elements` masks positions beyond the true end.
6. `tl.load(x_ptr + offsets, mask=mask)` reads valid values from HBM.
7. The vectorized GELU computation is applied locally.
8. `tl.store(y_ptr + offsets, y, mask=mask)` writes valid outputs to HBM.

The mask matters whenever tensor length is not an exact multiple of block size. Blocks before the final one have all-valid masks; the last block ignores its padded lanes.

The lecturer summarizes the universal kernel form as: identify the input and output span, load, compute, and store.

### Additional explanation

A condensed version of the structure is:

```python
@triton.jit
def gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = gelu_expression(x)
    tl.store(y_ptr + offsets, y, mask=mask)
```

Pointers are addresses; adding an element offset performs pointer arithmetic in the pointee type. The program appears vectorized, while the compiler chooses how its values map to physical threads and registers.

## 15. Questions about CUDA, tensor units, and where loaded data goes

**Transcript coverage:** lines 3673-4000

### What the lecturer said - transcript only

Several questions probe the abstraction.

First, how does the elementwise Triton kernel differ from CUDA? For this operation, the difference is small and CUDA may even look simpler: a CUDA thread identifies one element and processes it. Triton expresses a vectorized block. The block abstraction becomes more valuable when computation requires communication or aggregation inside the block.

Second, how does this code use tensor units? The lecturer says the basic Triton program does not explicitly assign values to a particular unit; compilation and hardware determine the mapping. The generated lower-level code will reveal more.

Third, what happens step by step from HBM to shared memory or registers? The Triton source is a specification for the compiler, not code literally interpreted by the GPU. The input pointer names a region in HBM. A load reads the selected addresses and produces a local value that the compiler places in registers or shared memory as appropriate. The lecturer defers the exact execution and latency question until PTX is shown.

The main form remains stable across kernels: wake up as a block, determine indices, read, operate, and write.

### Additional explanation

Triton intentionally leaves some placement decisions implicit. This reduces source-level bookkeeping but makes compiled-code inspection and profiling important when register pressure or unexpected spills hurt performance.

An ordinary elementwise GELU does not need tensor-core matrix instructions. The tensor-unit question becomes more relevant for `tl.dot` in matrix multiplication, where the compiler can lower the block operation to hardware matrix instructions supported by the target.

## 16. PTX, thread coarsening, and warp scheduling

**Transcript coverage:** lines 4001-4535

### What the lecturer said - transcript only

The Triton compiler generates **PTX**, an intermediate assembly language for NVIDIA GPUs. At this level, the block abstraction has been lowered to code executed by a thread.

The displayed PTX includes:

- global-memory load instructions that move data from HBM into registers;
- integer registers and floating-point registers;
- move and multiply instructions operating on those registers;
- a global-memory store near the end.

The generated code also demonstrates thread coarsening. One physical thread processes eight elements rather than one because the elementwise job is light enough to make each thread larger.

The compiler generates the code once, and all relevant threads run the same program. Special identifiers distinguish instances: the CTA identifier specifies the block, and the thread identifier specifies the thread within that block.

PTX still does not specify every runtime decision, such as exactly which SM receives a block or how all warps are scheduled. Those remain hardware-controlled.

In response to questions:

- People can hand-write PTX, especially when a compiler is immature or a specialist believes it can be improved, but NVIDIA's compilers are mature enough that most programmers should not begin there.
- When a PTX global load stalls a warp, the SM's scheduler can issue another resident warp and return when the load completes. The lecturer confirms this as the intended latency-hiding model.
- Asked why the discussed architecture has four warp schedulers, the lecturer says he does not know the exact design reason.

### Additional explanation

PTX is a virtual instruction set. NVIDIA's toolchain further translates it to target-specific machine instructions, often called SASS. PTX inspection is therefore informative but not always the final word on instruction scheduling.

Thread coarsening exchanges parallel thread count for more independent work and register use per thread. It can reduce overhead and increase instruction-level parallelism, but may lower occupancy. The generated eight-elements-per-thread choice is a concrete example of the same tradeoff introduced earlier.

---

# Part IV - Row-wise operations and reductions

## 17. Stable softmax and the traffic of a naive implementation

**Transcript coverage:** lines 4536-4806

### What the lecturer said - transcript only

After elementwise GELU, the lecture moves to operations that aggregate multiple values. Softmax exponentiates and normalizes each row of a matrix. It is used in attention and to produce probability distributions.

For an $M \times N$ matrix, the stable row-wise procedure is:

1. compute the maximum of each row;
2. subtract that maximum from every row element;
3. exponentiate each shifted value;
4. sum the exponentials in each row;
5. divide each exponential by its row sum.

In naive eager PyTorch, these stages are separate kernels. Each stage reads inputs from HBM and writes an intermediate back. The lecture counts roughly $5MN$ reads and $3MN$ writes, far more traffic than a fused implementation should need.

### Additional explanation

For row vector $x$, stable softmax is

$$
m = \max_j x_j,
$$

$$
p_i = \frac{\exp(x_i-m)}{\sum_j \exp(x_j-m)}.
$$

Subtracting $m$ does not change the normalized result because the common factor $e^{-m}$ cancels. It prevents the largest exponential from overflowing.

The precise eager traffic includes small row-statistic vectors in addition to the leading $MN$ terms. The lecture's count intentionally emphasizes the dominant fact: every unfused pass streams the matrix through HBM again.

## 18. A fused Triton softmax when one row fits in a block

**Transcript coverage:** lines 4807-5174

### What the lecturer said - transcript only

The first Triton softmax assumes an entire row fits within one block. One block is assigned to each row because all values in that row must participate in max and sum reductions. Different rows are independent, and blocks do not share ordinary shared memory.

The Python wrapper:

- allocates the output matrix;
- reads the matrix dimensions $M$ and $N$;
- chooses a block size as the next power of two at least as large as $N$;
- launches a grid of $M$ blocks;
- passes row strides so the kernel can locate each row.

Inside the kernel, `tl.program_id(0)` identifies the row. Column offsets span the padded block. A masked load supplies $-\infty$ for positions beyond the real number of columns so that their exponentials contribute zero. The block then computes max, subtraction, exponentiation, sum, and division using vectorized Triton operations, and stores only valid columns.

When everything fits in one block, the computational portion resembles ordinary PyTorch. Triton handles the within-block reduction.

Questions clarify two cases:

- If a row is larger than the block can hold, a different tiled strategy is required and is addressed next.
- A column-wise softmax is conceptually possible by changing pointer offsets and strides so one program instance walks down a column rather than across a row.

### Additional explanation

The fused row program aims for one HBM read and one HBM write per matrix value. Max, exponentials, sum, and normalization operate on local values between those transfers.

Padding to a power of two can simplify reduction code and match hardware-friendly block shapes. The mask separates the logical row length from the physical block width. Using $-\infty$ is the correct neutral padding before softmax because

$$
\exp(-\infty)=0.
$$

## 19. Tiled row reduction when a row does not fit

**Transcript coverage:** lines 5175-5660

### What the lecturer said - transcript only

Suppose a row has about 4,000 values while the block size is 1,024. The lecture simplifies softmax to row sum and divides the row into four tiles. One block still owns one complete row, but its threads process successive tiles in a loop.

Each thread maintains an accumulator. On the first tile it reads one lane's value; on later tiles it reads the corresponding lane positions and adds them to that accumulator. After all tiles have been visited, a final within-block reduction sums the per-thread accumulators into the row's scalar result and writes it to HBM.

The kernel therefore:

1. identifies its row;
2. initializes a vector of accumulators;
3. loops with starts `0`, `BLOCK_SIZE`, `2 * BLOCK_SIZE`, and so on;
4. creates offsets and an end mask for each tile;
5. loads a tile from HBM and adds it to the accumulator;
6. applies `tl.sum` to the accumulator vector;
7. stores the row result.

The compiler chooses whether accumulator values reside in registers or shared memory. The lecturer says a sufficiently large block can force shared-memory use.

The crucial distinction is between **blocks** and **tiles**. In the elementwise GELU, pieces of the vector are independent blocks. In the long-row reduction, the entire row is one block and its tiles are sequential chunks processed by that same block because their partial results must be combined.

### Source reconciliation

The transcript describes a row of approximately 4,000 columns. Slide 16 uses the exact example `4096` columns with block size `1024`, yielding four complete tiles. The difference is spoken rounding, not an algorithmic disagreement.

### Additional explanation

For row length $N$ and block width $T$, the number of tile iterations is

$$
K = \left\lceil \frac{N}{T} \right\rceil.
$$

Each lane accumulates values at offsets separated by $T$. A final tree-style block reduction combines the $T$ lane accumulators. The total arithmetic remains $O(N)$, while the tile width controls parallelism, local storage, and the number of loop iterations.

---

# Part V - Tiled matrix multiplication and fusion

## 20. Naive and idealized matrix-multiplication kernels

**Transcript coverage:** lines 5661-6003

### What the lecturer said - transcript only

Matrix multiplication is the central deep-learning operation. The lecture multiplies

$$
A \in \mathbb{R}^{M \times K},
\qquad
B \in \mathbb{R}^{K \times N},
\qquad
C = AB \in \mathbb{R}^{M \times N},
$$

and later applies ReLU to the result.

A correct but naive kernel assigns work for one output element $C_{mn}$. It loops over $k$, repeatedly reads $A_{mk}$ and $B_{kn}$ from HBM, multiplies them, accumulates the dot product, and finally writes $C_{mn}$.

Across the matrix, this performs $O(MKN)$ HBM reads and $O(MN)$ writes. Arithmetic work is also $O(MKN)$, so arithmetic intensity remains $O(1)$. The kernel fails to reuse values: neighboring output elements reread the same rows of $A$ and columns of $B$.

An idealized alternative loads all of $A$ and $B$ into shared memory once, computes every output from those local copies, and writes $C$ once. Its leading traffic becomes $O(MK+KN+MN)$, which gives $O(N)$ intensity for square matrices. This is the favorable matrix-multiplication intensity derived earlier.

The idealized approach is usually impossible because full matrices do not fit in shared memory.

### Additional explanation

For square $N \times N$ matrices, the naive model moves $O(N^3)$ values for $O(N^3)$ arithmetic, while the ideal model moves $O(N^2)$ values for the same work:

$$
I_{\text{naive}} = O(1),
\qquad
I_{\text{ideal}} = O(N).
$$

The gap comes entirely from reuse. Each value of $A$ participates in $N$ output columns and each value of $B$ in $N$ output rows. The kernel should exploit that reuse before returning the values to HBM.

## 21. Tiling makes the kernel locally ideal

**Transcript coverage:** lines 6004-6224

### What the lecturer said - transcript only

Tiling keeps as much useful data in shared memory as fits. The lecturer describes the result as **globally like the naive algorithm but locally like the idealized one**.

The output matrix $C$ is divided into tiles. Each output tile is assigned to one thread block. That block loops along the shared $K$ dimension:

1. load a row-oriented tile from $A$ and the corresponding column-oriented tile from $B$ into local memory;
2. multiply the two tiles;
3. add the result to a partial output tile;
4. advance to the next pair of $K$ tiles;
5. after the sweep, write the completed output tile to HBM.

Different output tiles are computed independently by different blocks. Reuse inside each block raises arithmetic intensity to the order of the tile size. It cannot normally reach $O(N)$ because fitting entire operands would be required, but a large tile can still make the kernel compute-bound.

Since the output tile is already local, an elementwise activation can be applied before the HBM store at very little additional traffic. The lecture fuses ReLU after the matmul as an example of a kernel epilogue.

### Source reconciliation

The spoken lecture, function names, and kernel code use a fused **ReLU**. One bullet on slide 19 says `GELU(A @ B)`, which is inconsistent with the surrounding example and appears to be a slide typo. These notes follow the transcript and executable code.

### Additional explanation

Let the output tile have dimensions $T_M \times T_N$ and the reduction tile width be $T_K$. Each loop iteration loads about $T_MT_K + T_KT_N$ input values and performs about $2T_MT_NT_K$ FLOPs. Larger $T_M$ and $T_N$ improve reuse, but also consume more registers and shared memory.

Tile selection therefore balances:

- arithmetic intensity;
- register and shared-memory capacity;
- occupancy;
- alignment with tensor-core instruction shapes;
- edge masking for non-divisible dimensions.

## 22. Strides and the Triton matmul implementation

**Transcript coverage:** lines 6225-6476

### What the lecturer said - transcript only

A multidimensional tensor is linearized in memory. Strides map a logical row and column to an address:

$$
\operatorname{index}(r,c)
= r \cdot \operatorname{stride}_{r}
+ c \cdot \operatorname{stride}_{c}.
$$

For a contiguous row-major matrix with four columns, moving down one row advances four positions and moving across one column advances one. A transpose swaps those stride roles.

The Triton matmul launch creates a two-dimensional grid over output tiles. Each program instance wakes up responsible for tile $(m,n)$ of $C$. It derives:

- the rows of $A$ in its output tile;
- the columns of $B$ in its output tile;
- the sequence of indices along $K$;
- pointer matrices for the current $A$ and $B$ tiles.

It initializes an accumulator tile, then loops over $K$ tiles. Each iteration loads the current $A$ and $B$ tiles, calls a local dot operation, adds the partial result, and advances both pointer sets. Once the reduction dimension is complete, the kernel applies ReLU locally and stores the masked output tile.

Within the block, the matrix operation looks high-level: once tiles have been loaded, Triton can express their multiplication directly. The difficult part is correct index and pointer construction.

### Additional explanation

The slide implementation uses representative tile sizes

$$
T_M=64, \qquad T_N=64, \qquad T_K=32.
$$

Its grid dimensions are

$$
G_M = \left\lceil \frac{M}{T_M} \right\rceil,
\qquad
G_N = \left\lceil \frac{N}{T_N} \right\rceil.
$$

The accumulator is initialized in FP32 in the slide code even though input tiles may use a narrower dtype. Wider accumulation reduces numerical error. Actual placement of accumulator fragments and input tiles is decided by compilation and the target instruction path; the source-level labels "register" and "shared memory" remain a conceptual model.

Fusion works because ReLU is applied to the completed local accumulator before its only HBM write. A separate ReLU kernel would reread and rewrite the entire $M \times N$ output.

---

# Part VI - Synthesis and closing questions

## 23. What the single-GPU kernel progression established

**Transcript coverage:** lines 6477-6639

### What the lecturer said - transcript only

The programming stack ranges from PyTorch through Triton to PTX. These layers describe what a programmer can control, but none removes the finite hardware beneath them. A large model must fit the available SMs, banks, registers, shared memory, caches, and HBM behavior.

Benchmarking and profiling connect source code to that hardware reality. They show how scheduling, memory limits, and kernel selection translate into performance.

Triton offers a convenient thread-block language. It avoids much of CUDA's explicit thread synchronization and shared-memory bookkeeping. The recurring block recipe is to identify a tile, read from HBM into local storage, compute and reuse locally, and write the result back.

The examples increase in difficulty:

1. GELU is elementwise and its blocks are independent.
2. Softmax reduces one row that fits in a block.
3. A long row introduces a loop over tiles within one block.
4. Matrix multiplication uses two-dimensional output tiling and a reduction over paired input tiles.

This concludes single-GPU programming. The following lecture moves to multiple GPUs.

### Additional explanation

The examples form a reusable classification:

- **Pointwise:** no communication among elements; partition freely.
- **Local reduction:** values communicate within one block.
- **Tiled reduction:** one block iterates over more data than fits at once.
- **Blocked bilinear operation:** two tiled inputs are reused to build an output tile.

FlashAttention, mentioned as an assignment target, combines these ideas: tile the attention computation, perform row-wise normalizations stably, and avoid materializing the full attention matrix in HBM.

## 24. Alternatives to Triton

**Transcript coverage:** lines 6640-6735

### What the lecturer said - transcript only

An audience member asks what other kernel-writing options exist and how close they can get to optimal performance. The lecturer frames language choice through **inductive bias**: every language or DSL makes some patterns easy and others hard.

Triton was developed by people working on Transformer training, so Transformer-like computations are comparatively natural in it. At the lowest extreme, a programmer can hand-write PTX, but the lecturer does not recommend that as a first step.

Other choices include ThunderKittens, CuTe, and additional DSLs and libraries. They are not always arranged on one simple higher-versus-lower ladder; each exposes different characteristics and favors different kernel structures.

### Additional explanation

A practical choice can be organized by required control:

| Level | Typical advantage | Typical cost |
|---|---|---|
| PyTorch plus compiler | Fast development and graph-level fusion | Less predictable low-level control |
| Triton | Concise block programs and good Transformer fit | Compiler-dependent placement and feature coverage |
| CUDA/C++ and libraries such as CUTLASS or CuTe | Fine control and mature primitives | More synchronization and memory bookkeeping |
| PTX or machine-level specialization | Maximum instruction-level control | Highest development and portability burden |

The best layer is the highest one that can express the needed data movement and reach the target performance.

## 25. There is no shape-only answer for a high-dimensional tensor

**Transcript coverage:** lines 6736-6823

### What the lecturer said - transcript only

The final question asks whether a high-dimensional tensor operation should load whole tensors across many blocks or process individual components and write each back to HBM.

The lecturer says the question cannot be answered in the abstract. The best decomposition depends on the nature of the computation, so a more specific discussion is needed offline.

### Additional explanation

The relevant questions are not merely how many dimensions the tensor has, but:

1. Which axes are independent and which require reductions?
2. How often can each loaded value be reused?
3. What tile fits in registers and shared memory?
4. Are adjacent lanes accessing contiguous addresses?
5. Does the decomposition create enough blocks without a severe tail?
6. What does profiling identify as the bottleneck?

High rank is often flattened or grouped into a smaller number of semantic axes. The correct kernel follows data dependencies and reuse, not the visual number of tensor dimensions.

---

# Consolidated takeaways

1. The GPU programming model is clean, but performance depends on physical SM, warp, register, bank, cache, and HBM constraints.
2. Fast memory is small and local; efficient kernels maximize reuse before returning values to HBM.
3. Threads form 32-lane warps, warps form blocks, and blocks form a launch grid.
4. Divergent branches serialize paths within a warp, while multiple resident warps help hide memory latency.
5. Register use limits occupancy, but maximum occupancy is not always optimal because coarser threads can perform more useful work.
6. Shared-memory bank conflicts and uncoalesced HBM access are different failure modes at different levels of the hierarchy.
7. Block grids can suffer a sparse final wave when block count aligns poorly with the number of SMs.
8. Benchmarking measures the outcome; profiling reveals the kernels and bottlenecks that produced it.
9. Correct CUDA timing requires warmup, device events, repeated trials, and synchronization.
10. High-level PyTorch expressions may launch many kernels. Fusion eliminates intermediate HBM traffic by retaining values locally.
11. Triton raises kernel programming from individual threads to vectorized block programs.
12. Generated PTX reveals loads, stores, registers, thread identifiers, and compiler-selected thread coarsening.
13. A row-wise reduction fits naturally in one block when the row is small and requires tile iteration when it is large.
14. Tiled matrix multiplication raises arithmetic intensity by reusing input tiles while accumulating an output tile locally.
15. Elementwise epilogues such as ReLU are natural fusion opportunities before a matmul result is written to HBM.
16. No one kernel language or launch decomposition is universally best; the operation's dependencies and measured bottleneck determine the choice.

# Key equations and accounting rules

## Register-limited resident blocks

$$
B_{\text{resident}}
\leq
\left\lfloor
\frac{R_{\text{SM}}}{T_{\text{block}}R_{\text{thread}}}
\right\rfloor.
$$

This is one limit among additional constraints on blocks, threads, warps, and shared memory.

## Warp occupancy in the lecture example

$$
R_{\text{block}} = 128 \times 160 = 20{,}480,
$$

$$
B_{\text{resident}}=3,
\qquad
W_{\text{resident}}=3 \times \frac{128}{32}=12,
$$

$$
\text{warp occupancy} = \frac{12}{64}=18.75\%.
$$

## Stable softmax

$$
\operatorname{softmax}(x)_i
= \frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
\qquad
m=\max_j x_j.
$$

## Number of tiles in a long row

$$
K_{\text{tiles}}
= \left\lceil \frac{N}{T} \right\rceil.
$$

## Matrix multiplication

$$
A_{M \times K}B_{K \times N}=C_{M \times N}.
$$

For square matrices, the lecture's traffic models give:

$$
I_{\text{naive}}=O(1),
\qquad
I_{\text{ideal}}=O(N),
\qquad
I_{\text{tiled}}=O(T),
$$

where $T$ denotes a representative tile dimension.

## Stride-based address calculation

$$
\operatorname{address}(r,c)
= \operatorname{base}
+ r\,s_r
+ c\,s_c.
$$

## Two-dimensional output grid

$$
G_M = \left\lceil \frac{M}{T_M} \right\rceil,
\qquad
G_N = \left\lceil \frac{N}{T_N} \right\rceil.
$$

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Bank conflict | Serialization when warp lanes access different shared-memory addresses mapped to the same bank. |
| Benchmark | A measurement of end-to-end execution time or throughput for a defined workload. |
| Block occupancy | How effectively launched thread blocks fill available SM capacity across execution waves. |
| Cache line | A fixed-size unit transferred through a cache or global-memory transaction. |
| Coalescing | Combining nearby global-memory requests from warp lanes into fewer transactions. |
| CTA | Cooperative thread array, another name for a thread block. |
| CUTLASS | NVIDIA's CUDA template library for linear algebra. |
| CUDA event | A device-side timing marker used to measure asynchronous GPU work. |
| Control divergence | A warp executing different branch paths serially because its lanes disagree. |
| Grid | The full set of thread blocks launched for one kernel. |
| HBM | High-bandwidth global memory attached to the GPU. |
| Kernel | A function launched on the accelerator over a grid of parallel program instances. |
| Kernel fusion | Combining multiple operations into one kernel so intermediates need not round-trip through HBM. |
| L1/shared-memory pool | Fast per-SM storage used either as hardware-managed cache or programmer-managed shared memory. |
| Mask | A Boolean lane selector used to prevent out-of-bounds loads and stores in a padded block. |
| Occupancy | The fraction of an SM's possible resident execution resources, commonly expressed for warps or blocks. |
| Profiler | A tool that attributes time and hardware behavior to operations and kernels. |
| PTX | NVIDIA's virtual parallel-thread instruction set emitted beneath Triton in the lecture. |
| Register pressure | Demand for per-thread registers that can reduce resident threads or cause spilling. |
| SM | Streaming multiprocessor, the GPU unit that schedules warps and provides registers, shared memory, and compute units. |
| Stride | The linear-memory distance moved when a logical tensor index advances by one along an axis. |
| Swizzling | Remapping local-memory addresses to reduce structured bank conflicts. |
| Thread block | A group of threads scheduled together on one SM and able to cooperate through shared memory. |
| Thread coarsening | Assigning multiple data elements to one thread to increase work per thread. |
| Tile | A bounded data region loaded and processed locally to create reuse. |
| Triton | A block-oriented language and compiler for writing GPU kernels in a Python-like style. |
| Warp | A group of 32 NVIDIA GPU threads that executes instructions in lockstep. |
| Wave quantization | Underutilization when the final wave contains fewer blocks than the machine can run concurrently. |

# Self-check questions

1. Why can HBM offer several terabytes per second and still be the slow level of the GPU hierarchy?
2. Why are thread blocks unnecessary for a simple elementwise map but essential for efficient reductions and matmuls?
3. What is the difference between the semantic programming model and the hardware performance model?
4. How does warp divergence turn parallel branch paths into serialized work?
5. Reproduce the lecture's 18.75% register-limited occupancy calculation.
6. Why can lower occupancy sometimes outperform higher occupancy?
7. Distinguish a shared-memory bank conflict from an uncoalesced HBM access.
8. Why does launching 160 blocks on a simplified 148-SM machine create a tail problem?
9. Why are warmup and synchronization both necessary in a CUDA benchmark?
10. What can a long CUTLASS kernel name reveal about hardware target, dtype, and tiling?
11. Why does the naive PyTorch GELU move much more HBM data than the built-in or compiled version?
12. In what sense does Triton sit between PyTorch and CUDA?
13. What roles do program ID, offsets, and the mask play in the elementwise Triton kernel?
14. What information becomes visible in PTX, and what scheduling decisions remain hidden?
15. Why does one row map naturally to one block for the first softmax kernel?
16. Explain the difference between independent GELU blocks and tiles processed by one long-row reduction block.
17. Why does the naive matmul have $O(1)$ arithmetic intensity while the idealized version has $O(N)$ intensity for square matrices?
18. How do tile dimensions trade arithmetic intensity against occupancy and local-memory capacity?
19. Why is an activation function cheap to fuse into a matmul epilogue?
20. What information about a high-dimensional tensor operation is needed before choosing a block decomposition?

# Source coverage checklist

| Raw transcript span | Material covered | Covered above |
|---:|---|:---:|
| 1-246 | Lecture continuation, GPU generations, SMs, register/cache/shared-memory/L2/HBM hierarchy and bandwidth | Yes |
| 247-567 | Threads, CTAs, grids, why blocks exist, shared-memory collaboration, hidden hardware features | Yes |
| 568-900 | Programming abstraction versus hardware performance, warps, divergence, resident-warps latency hiding | Yes |
| 901-1149 | Register limits, occupancy tradeoff, thread coarsening, 128-thread/160-register worked example | Yes |
| 1150-1476 | Shared-memory banks, conflicts and swizzling, HBM coalescing and cache-line access | Yes |
| 1477-1728 | Block waves, 160-block/148-SM tail, programming-model summary, audience question about sharing an SM | Yes |
| 1729-2110 | Measure-change-measure philosophy, benchmark purpose, warmup, CUDA events, synchronization, repeated trials, scaling curve | Yes |
| 2111-2412 | Profiling purpose, PyTorch and Nsight, addition and matmul profiles, shape-specific CUTLASS kernels and names | Yes |
| 2413-2921 | GELU approximation, naive/built-in/compiled comparison, profiling, HBM traffic, fusion, CUDA/Triton questions | Yes |
| 2922-3160 | CUDA thread-level programming, Triton block abstraction, strengths and limitations | Yes |
| 3161-3672 | Output allocation, grid launch, program IDs, pointer arithmetic, offsets, masks, loads, GELU compute, stores | Yes |
| 3673-4000 | Questions on CUDA comparison, tensor units, HBM-to-local placement and compiler mapping; general kernel form | Yes |
| 4001-4535 | PTX loads/stores/registers, eight-element thread coarsening, thread and block IDs, PTX authorship, warp scheduling, unknown scheduler rationale | Yes |
| 4536-4806 | Softmax definition, stable algorithm, naive PyTorch kernel sequence and traffic | Yes |
| 4807-5174 | One-row-per-block Triton softmax, next-power-of-two block, strides, masks, column-softmax and long-row questions | Yes |
| 5175-5660 | Four-tile long-row sum, per-thread accumulation, final reduction, compiler-selected storage, tile-versus-block distinction | Yes |
| 5661-6003 | Matmul plus activation, naive output-element kernel, redundant HBM reads, idealized shared-memory approach | Yes |
| 6004-6224 | Output tiling, paired A/B input tiles, partial accumulation, tile-size intensity, fused activation | Yes |
| 6225-6476 | Strides, two-dimensional Triton grid, pointer construction, dot loop, pointer advancement, ReLU epilogue and output store | Yes |
| 6477-6639 | Programming stack versus finite hardware, role of measurement, Triton block model, progression of examples, transition to multi-GPU | Yes |
| 6640-6735 | Audience question on Triton alternatives, language inductive bias, PTX, ThunderKittens, CuTe, and DSL tradeoffs | Yes |
| 6736-6823 | Audience question on high-dimensional tensor decomposition and why the answer depends on the computation | Yes |
