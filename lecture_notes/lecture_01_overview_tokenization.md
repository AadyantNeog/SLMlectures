---
title: "Lecture 1 - Overview and Tokenization"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 1
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 1 Overview, Tokenization.txt"
slide_deck: "../lecture_01.pdf"
status: "complete"
---

# Lecture 1: Overview and Tokenization

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves the lecturer's claims, examples, qualifications, and numerical details while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, intuition, derivation, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are shown so the paraphrase can be audited. The slide deck was checked after the transcript map was complete, primarily to verify spellings, notation, code, charts, and lecture structure. When the slides and speech differ materially, an editorial note identifies the difference.

## Lecture map

The lecture has four broad purposes:

1. Explain why a from-scratch language-modeling course remains valuable even when frontier models are inaccessible and powerful coding agents already exist.
2. Place modern language models in historical and open-model context.
3. Preview the course's five major units: basics, systems, scaling laws, data, and alignment.
4. Begin the first technical unit by defining tokenization, comparing character-, byte-, and word-level schemes, and deriving byte pair encoding (BPE).

---

# Part I - Course motivation and perspective

## 1. The course and its teaching philosophy

**Transcript coverage:** lines 1-295

### What the lecturer said - transcript only

This is the third offering of CS336. The teaching team consists of Percy Liang and Tatsu Hashimoto as instructors, with Marcel Roed, Herman Brunnborg, and Steven Cao supporting the course. The introductions emphasize a range of relevant backgrounds: long experience with language models, model architectures, higher-order gradients, training, theory, and data efficiency. Herman also describes how working through the previous version of the course took him from little understanding of LLMs to doing LLM research.

The course retains its **from-scratch philosophy**: understanding comes from building the important parts of the system oneself. "From scratch" is necessarily selective because rebuilding absolutely everything would not fit into one quarter. The teaching team has therefore refined the course around components that offer the highest learning value for the time spent. Compared with prior versions, the course gives more attention to modern ingredients such as mixture-of-experts models, long context, and agents, while retaining the same underlying philosophy.

### Additional explanation

"From scratch" is best interpreted as **owning the important abstractions**. A student does not fabricate a GPU or reimplement an operating system, but should understand and implement enough of the tokenizer, model, optimizer, training loop, kernels, and distributed system to explain where the computation, memory, errors, and performance come from.

This is especially valuable in research. High-level libraries make standard workflows easy, but novel research often begins exactly where their assumptions stop being appropriate. Building the stack once gives a mental model for deciding which abstractions can safely be trusted and which must be opened.

## 2. Why understanding the underlying technology still matters

**Transcript coverage:** lines 298-415

### What the lecturer said - transcript only

Researchers have gradually moved farther away from the implementation details of their models. Roughly a decade ago, researchers commonly implemented and trained their own models. Later, they downloaded pretrained models such as BERT and fine-tuned them. Today, a great deal can be accomplished by prompting an API model.

Moving to higher levels of abstraction is productive and can produce impressive results. The problem is that language-model abstractions are **leaky**: a model can fail to do what the user wants, while the prompting interface offers little recourse. Restricting oneself to prompting also restricts the research design space. Fundamental research may require changing the architecture, objective, data, training process, or systems stack rather than treating the model as fixed. The course therefore treats full-stack understanding as a prerequisite for fundamental work and uses building as the route to that understanding.

### Additional explanation

A leaky abstraction hides details that later become relevant to behavior. Prompting hides tokenization, context construction, data provenance, optimization, inference scheduling, and architectural constraints. When a failure originates in one of those layers, prompt changes may only work around the symptom.

The point is not that every project should pretrain a model. For many applications, prompting or fine-tuning is the correct engineering choice. The point is that researchers should understand when a high-level interface has made a consequential design choice on their behalf.

## 3. The industrialization and inaccessibility of frontier models

**Transcript coverage:** lines 418-588

### What the lecturer said - transcript only

Language-model development has become industrialized. Frontier training runs are extremely expensive: the lecturer cites a reported figure of roughly 100 million US dollars for GPT-4 in 2023 and speculates that present costs may be on the order of a billion dollars. Major laboratories are also building clusters with enormous numbers of GPUs.

Cost is only one obstacle. Public reports often omit the details required to reproduce frontier systems. The lecturer points to the GPT-4 report's decision not to reveal architecture, hardware, training compute, data construction, training methods, and related details because of competitive and safety considerations.

Students can build small language models, but small models are not always representative of frontier models:

- **The allocation of computation changes with scale.** In the example shown, MLP layers account for about 44% of FLOPs at one small scale and about 80% at 175B parameters. Optimizing attention at small scale may therefore have a different total effect at large scale.
- **Some behaviors appear only after sufficient scaling.** In the cited zero- and few-shot examples, performance remains close to baseline at smaller scales and improves sharply only beyond a critical region. A small experiment may never expose the phenomenon of interest.

### Additional explanation

Scale changes more than the final accuracy. It can change the dominant bottleneck, the useful batch size, the balance between model and data, the sensitivity to optimization choices, and which failures are visible. This is why a small-model result should be treated as evidence under a particular scale regime rather than as an automatic forecast for a frontier run.

Two different kinds of extrapolation are involved:

1. **Quantitative extrapolation:** estimating how a metric such as loss changes as compute grows.
2. **Qualitative extrapolation:** assuming the same bottlenecks, behaviors, or optimal design choices continue to hold.

Scaling-law methods can help with the first. The second is often harder and is one reason the course distinguishes mechanics and mindset from empirical intuition.

## 4. What knowledge transfers across scale?

**Transcript coverage:** lines 589-768

### What the lecturer said - transcript only

The lecturer separates the course's knowledge into three categories:

1. **Mechanics:** how components work, such as Transformers and model parallelism.
2. **Mindset:** how to approach building models, especially squeezing value from hardware and taking scaling seriously.
3. **Intuitions:** which modeling and data choices produce strong performance.

The course can teach mechanics and mindset well, and these should transfer to larger scales. Its central habits are to profile, benchmark, and optimize for efficiency.

Empirical intuitions are less reliable across scale. Some design choices have no satisfying theoretical justification and are known mainly because experiments showed that they work. As an example, the paper introducing SwiGLU openly admits that its successful variants lack a clear explanation. Such intuition must be earned through experiments, ideally at the relevant scale.

### Additional explanation

This distinction suggests different standards of evidence:

- A mechanical claim can often be checked by derivation, implementation, or measurement: tensor shapes match, collectives move the expected bytes, or a kernel reduces memory traffic.
- A mindset is validated by repeated usefulness: profile before optimizing, state a resource budget, and test predictions.
- An empirical intuition should be treated as provisional. It needs ablations, multiple scales, and ideally multiple data distributions before becoming a reusable rule.

When reading later lectures, it is useful to label each lesson mentally as a mechanism, a method of working, or an empirical regularity. The label determines how confidently it can be transferred.

## 5. The bitter lesson and the central role of efficiency

**Transcript coverage:** lines 769-972

### What the lecturer said - transcript only

The lecturer warns against interpreting the "bitter lesson" as saying that scale is all that matters and algorithms do not. The intended interpretation is that **algorithms that scale are what matter**.

He summarizes the relationship as:

$$
\text{accuracy} = \text{efficiency} \times \text{resources}.
$$

Here, efficiency is output per unit of input, while resources are the available input. Efficiency becomes more important, not less, at large scale. A run taking twice as long may be tolerable in a small experiment; at frontier scale, the same factor can correspond to hundreds of millions of dollars. Even a 5% improvement can matter greatly. The lecturer cites a study estimating a 44-fold improvement in algorithmic efficiency on ImageNet between 2012 and 2019, in addition to improvements in hardware.

The course is therefore framed around one question: **What is the best model that can be built under a fixed data and compute budget?** Pretraining is usually treated as compute-constrained because available data exceeds affordable compute, although a group with unusually abundant hardware or limited data could instead be data-constrained.

### Additional explanation

The displayed equation is a conceptual decomposition, not a literal universal accuracy law. Its purpose is to prevent "more compute" from becoming the only explanation for progress. Hardware, numerical methods, optimizers, architectures, data selection, and systems software all determine how much model quality a resource budget buys.

Efficiency is also multidimensional:

- **Statistical efficiency:** quality gained per training example or token.
- **Compute efficiency:** quality gained per FLOP.
- **Memory efficiency:** useful work supported per byte of capacity or bandwidth.
- **Communication efficiency:** progress per byte transferred between devices.
- **Inference efficiency:** latency, throughput, energy, or cost per generated token.

Optimizing one dimension can harm another. For example, a larger vocabulary can shorten sequences but enlarge embedding and output matrices. The course repeatedly returns to these tradeoffs rather than treating efficiency as a single scalar.

---

# Part II - How language models reached the present landscape

## 6. From pre-neural language models to foundation models

**Transcript coverage:** lines 973-1233

### What the lecturer said - transcript only

Language models predate modern neural networks. Shannon used language modeling in the 1950s to study the entropy of English. For many years, n-gram language models were components of machine-translation and speech-recognition systems, helping those systems generate fluent text.

The lineage of modern models comes from a sequence of neural and systems ideas. The lecturer highlights LSTMs; Bengio's 2003 feed-forward neural language model over a short context; sequence-to-sequence modeling; Adam; the attention mechanism developed for machine translation; the Transformer, also developed in that context; mixture of experts; and model-parallel methods.

By the late 2010s, ELMo and BERT demonstrated that a model pretrained on large amounts of text could be fine-tuned for downstream tasks such as question answering with large gains. Google also explored casting tasks into a text-in/text-out form.

OpenAI then strongly embraced scaling. GPT-2 was followed by scaling-law work and GPT-3, a model more than ten times larger than its predecessor that exhibited in-context learning. Google trained PaLM at massive scale, while DeepMind's Chinchilla work clarified compute-optimal scaling. The lecturer presents this period as the point when scaling became a central, predictable development strategy rather than merely making one model larger.

### Additional explanation

Several changes are intertwined in this history:

- **Representation learning:** pretraining replaces many task-specific feature pipelines.
- **Interface unification:** many tasks can be expressed as predicting text conditioned on text.
- **Scale and transfer:** sufficiently capable pretrained models can adapt through fine-tuning, prompting, or in-context examples.
- **Systems co-design:** larger models become useful only when optimization, parallelism, and hardware utilization keep training feasible.

The modern language model is therefore not the result of one architectural invention. It is the product of a stack whose layers matured together.

## 7. Open weights, open development, and reproducibility

**Transcript coverage:** lines 1234-1458

### What the lecturer said - transcript only

After GPT-3, several groups attempted to reproduce large-model capabilities. EleutherAI produced open datasets and models but had limited compute. Meta's first 175B-parameter effort was visibly a replication attempt and encountered substantial hardware problems. Hugging Face's BigScience project produced another early open model. The lecturer characterizes these initial models as not especially strong.

The open ecosystem changed considerably over the next few years. Meta's Llama series helped lead the shift, followed by Mistral and a growing group of Chinese model families including DeepSeek and Qwen. The lecturer stresses that this list is incomplete and that the ecosystem is difficult to track. Depending on the evaluator and benchmark, present open-weight models may be slightly behind or comparable to closed models, but they are credible and widely used.

A further level of openness aims to release not only weights and a paper but also code and data. The lecturer names efforts from AI2, NVIDIA, and the Marin project. These projects make it possible to understand model construction more thoroughly.

The open ecosystem is essential to the course. Papers and artifacts describing large mixture-of-experts and reinforcement-learning systems provide partial visibility into frontier practice. Important details are still missing - especially exact data mixtures - and many systems cannot be fully reproduced, but partial evidence is much better than none.

### Additional explanation

"Open" is not a single property. Useful dimensions include access to:

- model weights;
- inference and training code;
- tokenizer and configuration;
- training data or a precise data recipe;
- optimizer state and checkpoints;
- evaluation code and raw results;
- logs, ablations, and failure reports.

Weights permit deployment and experimentation, but they do not by themselves make the training result reproducible. For scientific understanding, the data recipe, optimizer, systems configuration, and negative results can be as informative as the final parameters.

## 8. The changing interface to a language model

**Transcript coverage:** lines 1459-1608

### What the lecturer said - transcript only

The practical meaning of "language model" has changed over the last decade. It was first something users fine-tuned, then something they prompted, then something they conversed with in the ChatGPT era, and now something that can act as an agent. The lecturer remarks on the surprising strength of current coding agents, which can receive a page of instructions and execute a complicated task through a long action trace.

Despite the interface change, the fundamentals remain recognizable: GPUs, kernels, gradient-based optimization, Transformers, and attention still form the core. The specifications have changed. Models are expected to handle much longer contexts, which makes inference efficiency increasingly important. This stability in the fundamentals allows the course to evolve without being rebuilt from the ground up each year.

### Additional explanation

The interface has moved from **parameter adaptation** to **context conditioning** to **closed-loop action**:

1. Fine-tuning changes parameters for a task.
2. Prompting changes the input context while parameters remain fixed.
3. Conversation maintains an evolving context across turns.
4. An agent repeatedly observes, reasons, calls tools, and incorporates results.

Agents do not eliminate the underlying language-model stack. They amplify its constraints: long histories stress context and memory, repeated generation magnifies inference cost, and tool use makes latency and reliability system-level concerns.

---

# Part III - Course mechanics and roadmap

## 9. Executable lectures

**Transcript coverage:** lines 1609-1659

### What the lecturer said - transcript only

The lecture itself is an executable Python program whose execution delivers the presentation. This allows code to be run directly during the lecture and makes the lecture's hierarchical structure visible: completing a topic corresponds to returning from one function to the main program.

### Additional explanation

An executable lecture reduces the gap between explanation and artifact. Code examples are not disconnected screenshots; they can share definitions, execute in sequence, and expose the same decomposition used to organize the ideas. It also models a useful engineering habit: make explanations inspectable and reproducible whenever possible.

## 10. Workload, audience, and logistics

**Transcript coverage:** lines 1660-1944 and 2170-2226

### What the lecturer said - transcript only

CS336 is a five-unit course with five intensive assignments. A prior course review compared the first assignment alone with the total assignment workload of another full course, though the lecturer cautions that this may be exaggerated.

The strongest reason to take the course is an obsessive desire to understand how language models work. It is also intended to build research-engineering skill and the confidence to enter unfamiliar technical settings. The lecturer compares its role in today's empirical and systems-oriented field to the role that a deep statistical learning theory course once played for the mathematical side of machine learning.

There are also reasons not to take it. The workload may interfere with research. The course is not primarily a survey of the newest topics such as multimodality or agents, and it is not the shortest route to strong results in one application domain. For an application, the recommended order is to try prompting, then fine-tuning, and only pretrain a new model as a last resort because pretraining is difficult and expensive.

People outside the class can follow the posted materials and recordings, but the lecturer stresses that learning comes from doing the assignments rather than only watching the lectures. Enrolled students receive compute credits through Modal and instructions for using that platform. Unlike the previous year's cluster accessed through SSH, the new system is API-oriented; the lecturer reports that it appears pleasant to use despite his initial skepticism.

### Additional explanation

The recommended application workflow can be understood as an escalation ladder:

1. Prompt an existing model.
2. Add retrieval, tools, or structured prompting if needed.
3. Fine-tune when behavior must change systematically.
4. Pretrain only when the required knowledge, modality, data control, scale, or architecture cannot be obtained from existing models.

Each step increases control, but also increases data requirements, evaluation burden, infrastructure, and cost.

## 11. Assignment philosophy and AI policy

**Transcript coverage:** lines 1945-2169

### What the lecturer said - transcript only

The assignments provide no scaffolding implementation, but they do provide unit tests. This avoids a sparse-feedback situation in which students discover only at submission time whether an entire system works. Much of each assignment can be implemented and checked locally; the provided compute is then used for full training runs, performance benchmarks, or multi-GPU work. Several assignments include leaderboards, often phrased as minimizing perplexity under a fixed resource budget.

Coding agents have become capable enough to solve the assignments, but handing an assignment directly to an agent would defeat the learning objective. At the same time, AI can be useful for tutoring, answering questions, and clarifying code. Students who use AI must use the course-provided `AGENTS.md` prompt, which asks the system to behave pedagogically: it may explain and guide, but should not silently generate the component the student is supposed to learn by implementing. The teaching team is trying this policy for the first time and requests feedback.

### Additional explanation

The policy distinguishes **assistance that increases understanding** from **substitution that bypasses the learning step**. A productive tutor can:

- ask the student to predict tensor shapes before showing them;
- explain a failing test without supplying the whole implementation;
- generate smaller analogous exercises;
- review reasoning and identify gaps;
- compare complexity or numerical behavior of candidate approaches.

For a from-scratch course, retaining the struggle of decomposition and debugging is part of the curriculum, not incidental friction.

## 12. The five-part course roadmap

**Transcript coverage:** lines 2227-2256

### What the lecturer said - transcript only

The syllabus has five parts that mirror the five assignments:

1. **Basics:** tokenization, model architecture, and training.
2. **Systems:** kernels, parallelism, and inference.
3. **Scaling laws:** predicting and choosing configurations at larger compute budgets.
4. **Data:** evaluation, collection, processing, and mixtures.
5. **Alignment:** improving a pretrained model through preference or reward-based signals.

The ordering constructs a complete pipeline. First build a model, then make it run efficiently, learn how to scale it, decide what to train it on and how to evaluate it, and finally improve its behavior with weaker forms of supervision.

### Additional explanation

The units are dependent rather than isolated:

- Scaling laws are useful only when runs at different scales are comparable and systems measurements are trustworthy.
- Data decisions require evaluations that reflect desired capabilities.
- Alignment relies on efficient inference to generate candidate responses and on stable training to update the policy.
- Tokenization affects sequence length and therefore training cost, memory use, context length, and serving speed.

This dependency structure explains why the first lecture previews the entire stack before focusing on tokenization.

## 13. Unit 1 preview: building a basic language model

**Transcript coverage:** lines 2257-2922

### What the lecturer said - transcript only

The first two weeks aim to make students capable of training a language model from scratch. The three components are tokenization, model architecture, and training.

#### 13.1 Tokenization as the model's choice of atoms

Tokenization determines the units on which the model operates. Formally, a tokenizer maps raw input bytes to integer token IDs and maps those IDs back to the original input. Conceptually, it segments text. The course uses byte pair encoding, which groups frequently occurring parts of the byte stream.

From the efficiency perspective, tokenization compresses a long byte sequence into fewer model steps. It also enables **adaptive computation**: common, predictable spans can be represented compactly, while rare or information-rich regions can remain split into more units and receive more model computation.

The lecturer would prefer an end-to-end model that operates directly on bytes, and mentions recent H-Net work as a promising example. However, these alternatives have not displaced tokenizers at the frontier, so tokenization remains necessary course material.

#### 13.2 Architecture choices

The starting point is the Transformer, but many details have changed since the original architecture:

- activation functions;
- positional encodings;
- normalization methods and placement;
- alternatives to full quadratic attention;
- state-space or linear-attention mechanisms such as Mamba and Gated DeltaNet, often used in hybrids with attention;
- dense versus mixture-of-experts MLPs;
- model shape, including depth, hidden dimension, head count, and expert count.

Mixture of experts has become an important way to build compute-efficient Transformers, but it also introduces specialized training problems. Apparently simple shape hyperparameters have large consequences when models are scaled.

#### 13.3 Training choices

Training adds another set of interdependent decisions:

- next-token versus multi-token prediction objectives;
- optimizers, including the historical dominance of Adam and newer use of Muon in some open models;
- initialization;
- learning-rate schedules;
- regularization;
- batch size;
- mixture-of-experts-specific mechanisms.

These should not be approached as an arbitrary grid of hyperparameters. Principled settings can separate a stable, state-of-the-art run from a run that diverges and becomes useless.

#### 13.4 Assignment 1 and the three-way balance

Assignment 1 asks students to implement a BPE tokenizer, Transformer, loss, optimizer, and training stack; perform resource accounting; train on datasets such as TinyStories and OpenWebText; and compete to minimize perplexity under a time budget. The lecturer compares the leaderboard spirit to NanoGPT speedruns. By its end, a student should be able to build a complete small language model from scratch.

The high-level design problem balances three objectives:

1. **Expressivity:** the model must represent the data's complex dependencies.
2. **Stability:** parameter and gradient norms must stay in a useful range rather than exploding or vanishing.
3. **Efficiency:** the implementation must run quickly on the target hardware.

An architectural change may reduce computation, for example through a lower-dimensional projection, but the resulting model may perform worse. Managing such tradeoffs is central to language-model design.

### Additional explanation

The three objectives constrain one another:

- Greater expressivity can make optimization less stable or increase memory and computation.
- Strong stabilization can restrict the function class or slow learning if applied too aggressively.
- A mathematically elegant component can be a poor practical choice when it maps badly to hardware.

The practical target is therefore not the most expressive architecture in isolation. It is the best **trainable and executable** architecture under the actual budget.

## 14. Unit 2 preview: systems

**Transcript coverage:** lines 2923-3688

### What the lecturer said - transcript only

The systems unit asks how to extract the most useful work from the hardware. Its main topics are resource accounting, kernels, parallelism, and inference.

#### 14.1 Resource accounting and hardware bottlenecks

Students will track where FLOPs and memory are spent. A recurring approximation for dense Transformer training is:

$$
C \approx 6ND,
$$

where $N$ is the number of model parameters and $D$ is the number of training tokens.

The critical hardware fact is that memory and arithmetic units are physically distinct. Parameters or activations must be moved from memory to the compute units, processed, and often written back. This movement is frequently the bottleneck. As an example, the lecturer gives a B200's approximate BF16 throughput as 2.25 petaflops per second and its memory bandwidth as 8 terabytes per second. Roofline analysis helps determine whether an operation is limited by arithmetic throughput or memory movement, while benchmarking and profiling show what actually happens.

#### 14.2 Kernels and minimizing data movement

A kernel is a function executed on a GPU. PyTorch primitives already launch standard kernels, but custom kernels can make important operations faster. The main principle is to organize work so that less data moves between high-bandwidth memory and the compute units.

The lecturer's simple example contrasts two implementations of consecutive operations $A$ and $B$:

- A naive implementation reads data, computes $A$, writes it, reads it again, computes $B$, and writes again.
- A fused implementation reads once, computes both $A$ and $B$, and writes once.

Operator fusion and tiling are variations of this idea. Modern GPUs also have many architectural details that affect performance, and students will write kernels to gain an appreciation for those constraints.

#### 14.3 Parallelism across many GPUs

When a run uses many GPUs, the same principle remains, but inter-device movement is even more expensive. A DGX B200 contains eight GPUs connected by NVLink; a much larger cluster connects multiple such systems through InfiniBand or Ethernet. Distributed training is organized around collective operations such as gather, reduce, and all-reduce. Parameters, activations, gradients, and optimizer states must be split across devices, moved to the right place for computation, and updated efficiently.

The computation can be partitioned along multiple dimensions: data, tensors within a model, pipeline stages or layers, sequence positions, or experts. Each method makes different memory, computation, and communication tradeoffs.

#### 14.4 Inference

Inference is required not only for user-facing generation but also for reinforcement learning, rollouts, test-time computation, synthetic-data generation, and evaluation.

It has two phases:

1. **Prefill:** all prompt tokens are known and can be processed together while the key-value cache is constructed. This resembles training and is comparatively compute-heavy.
2. **Decode:** new tokens are generated one at a time. The model repeatedly reads its parameters and cache for very little new arithmetic, so decoding quickly becomes memory-bound.

Possible accelerations include pruning, quantization, distillation, inference-specific kernels, and speculative decoding. In speculative decoding, a cheaper draft model proposes several tokens; the full model evaluates those candidates in parallel, allowing multiple tokens to be accepted when the draft is correct. Serving systems must also batch requests that arrive at different times, unlike training where batches are scheduled predictably.

#### 14.5 Assignment 2

The spoken description says the assignment will involve Triton kernels and some form of parallel training, while warning that the exact design is still being revised. The lecturer recommends *How to Scale Your Model* for its conceptual treatment of Transformer arithmetic and roofline analysis. Although the book foregrounds TPUs, many of its high-level ideas also apply to GPUs.

### Source reconciliation

- In the spoken preview, the resource-accounting example mentions training a 7B model on one trillion tokens. Slide 17 displays the calculation for 70B parameters and one trillion tokens, giving $4.2 \times 10^{23}$ FLOPs. The formula, rather than the mismatched model size, is the conceptual point.
- Slide 20 lists a fused RMSNorm kernel, distributed data-parallel training, optimizer-state sharding, benchmarking, and profiling. The transcript explicitly says the assignment details may change, so those slide bullets are not treated as settled spoken requirements.

### Additional explanation

The roofline view can be summarized using **arithmetic intensity**:

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved}}.
$$

If $I$ is low, additional compute units do little because they wait for data. Fusion improves performance by increasing the useful computation performed for each read and write. Distributed training extends the same accounting to network links: an algorithm that saves FLOPs may still be slower if it increases communication.

The prefill/decode split is also why training throughput alone does not predict serving performance. Training and prefill expose large matrix multiplications with substantial parallelism. Decode exposes repeated, latency-sensitive steps with relatively low arithmetic intensity.

## 15. Unit 3 preview: scaling laws

**Transcript coverage:** lines 3691-4379

### What the lecturer said - transcript only

Suppose a team receives $10^{25}$ FLOPs - compute worth tens of millions of dollars - and must decide what model to train. Full-scale hyperparameter sweeps are impossible because the team may be able to afford only one target run. A mistake could waste the entire budget.

The key conceptual shift is to stop thinking about one model and instead define a **scaling recipe**: a mapping from a compute budget to a complete set of hyperparameters or configuration. For a candidate recipe:

1. Run experiments at smaller compute scales.
2. Measure the loss at those scales.
3. Fit a scaling law.
4. Extrapolate the expected loss at the target scale.

This allows researchers to optimize a large-scale recipe using smaller experiments and to predict the target run before spending the full budget.

Scaling laws do not arise automatically as laws of nature. Researchers must construct a recipe that behaves predictably across scale. Learning rate, batch size, parameterization, and other settings need either to remain stable or change according to predictable functions. This property is **hyperparameter transfer**. If the best learning rate changes erratically between scales, small experiments cannot identify the correct large-scale value. Consequently, predictability can be at least as important as achieving the absolute optimum at every small scale.

#### 15.1 Compute-optimal allocation between parameters and tokens

Given a fixed compute budget, one must decide whether to train a larger model or train on more tokens. The classic procedure evaluates several model sizes at each of several smaller compute budgets, identifies the loss-minimizing size for each budget, and fits a curve through those optima. If the points form a stable trend, the curve can be extrapolated.

The lecturer gives the rough rule:

$$
D \approx 20N.
$$

Thus, a 70B-parameter dense model would be trained on roughly 1.4 trillion tokens under this rule. The number depends on the data and architecture. It also ignores downstream inference cost. Models intended for heavy serving may deliberately be smaller and trained on many more tokens than the training-compute optimum, because smaller models are cheaper to run.

#### 15.2 Prediction and preregistration

The Marin project provides a live example. The team fits loss curves at multiple compute budgets, extrapolates to a larger run, and preregisters the predicted loss before training finishes. Comparing the final result with the preregistered forecast tests whether the scaling process is genuinely predictive rather than retrospectively fitted.

#### 15.3 Assignment 3

Because the course cannot fund many large training runs per student, it simulates the decision process through a training API. Students submit hyperparameters and receive losses drawn from a cache of models trained offline. They gather points under a budget, fit scaling laws, extrapolate hyperparameters and loss, and are evaluated on how well the chosen large-scale configuration performs. The low-stakes simulation is intended to reproduce the reasoning required when a real run carries a very large financial risk.

### Additional explanation

Combining the dense-training approximation $C \approx 6ND$ with $D \approx 20N$ gives:

$$
C \approx 120N^2,
\qquad
N \approx \sqrt{\frac{C}{120}},
\qquad
D \approx 20\sqrt{\frac{C}{120}}.
$$

This derivation is only a rough planning tool. The constants and exponents can change with architecture, data quality, optimizer, sparsity, and the definition of compute. A mixture-of-experts model also separates total parameters from parameters activated per token, which complicates the simple dense-model accounting.

The emphasis on predictability is a form of risk management. A configuration that is marginally worse at small scale but extrapolates smoothly may be more valuable than a brittle configuration whose optimum cannot be forecast.

## 16. Unit 4 preview: evaluation and data

**Transcript coverage:** lines 4380-4935

### What the lecturer said - transcript only

After learning how to train, accelerate, and scale a model, the next question is what to train it on. Data quality strongly determines model quality, and the data distribution encodes what the builder wants the model to do: speak multiple languages, converse naturally, or complete long coding-agent tasks.

#### 16.1 Evaluation defines the desired capabilities

The data unit begins with evaluation because evaluation defines the capabilities being pursued. The lecturer distinguishes two purposes:

1. **Internal evaluation** guides model development. Metrics should be smooth across scales and useful for comparing alternatives; relative performance is often more important than an intuitively meaningful absolute value.
2. **External evaluation** communicates real-world quality to users, reviewers, or customers. Ecological validity - whether the evaluation reflects the actual use case - matters greatly.

Perplexity remains useful for internal development because it captures intrinsic model quality without dependence on benchmark prompting. Ideally, evaluation data should not already be available on the public internet, reducing contamination. More advanced external use cases require task-specific evaluations. Because language models are general-purpose, they need a diverse evaluation suite. A single average can conceal major differences among capabilities.

#### 16.2 Data does not arrive ready for training

Large datasets must be actively curated from sources such as web pages, books, archived papers, and GitHub repositories. Collection raises legal questions, including fair use, licensing, and how to treat code repositories with no explicit license.

The raw sources are usually not plain training text. Web data is HTML, documents may be PDFs, and code exists in directory structures. Processing therefore includes:

- **transformation:** extracting useful text or structure from raw formats;
- **filtering:** retaining high-quality material and removing unsuitable content;
- **deduplication:** avoiding repeated documents or passages;
- **mixing:** choosing the proportions of different sources;
- **synthetic-data generation or rewriting:** transforming real data into formats closer to desired downstream tasks or other preferred styles.

The lecturer distinguishes three stages of model data:

- **Pretraining data:** broad, large-scale data.
- **Mid-training data:** higher-quality material used near the end of pretraining, including long-context examples such as large code repositories or books.
- **Post-training data:** conversations, reasoning or agent traces, and examples involving tool use.

#### 16.3 Assignment 4

Students start from a raw source such as a web crawl and perform the unglamorous but essential work of transformation, filtering, deduplication, and cleanup. This "dirty work" is presented as an indispensable part of genuinely building a language model from scratch.

### Additional explanation

Training data is better viewed as a **behavior specification** than as passive fuel. The model learns probability mass from what appears, how frequently it appears, and in what format. Changing a mixture weight is therefore a modeling decision.

The evaluation/data relationship forms a loop:

1. Define desired behaviors with evaluations.
2. Identify failures and underrepresented capabilities.
3. Collect, filter, or synthesize targeted data.
4. Retrain and reevaluate.

This loop also creates a danger: optimizing repeatedly against a visible benchmark can overfit the development process even when the benchmark is not literally included in training. Private and rotating evaluations help preserve honest measurement.

## 17. Unit 5 preview: alignment and weak supervision

**Transcript coverage:** lines 4936-5148

### What the lecturer said - transcript only

Pretraining and related stages use full supervision in the sense that the target is the next token, or perhaps the next several tokens. Once this produces a reasonable model, behavior can be improved with weaker signals. Weak supervision is useful because evaluating a response can be easier than generating the ideal response directly.

The basic procedure is:

1. Generate candidate responses with the model.
2. Score them using humans, verifiers, or language-model judges.
3. Update the model to prefer the better responses.

This pattern can be implemented with reinforcement-learning methods such as PPO or GRPO, or with a simpler preference-data method such as DPO.

Reinforcement learning is unstable and difficult to tune. The lecturer personally prefers to remain in fully supervised training for as long as possible and use RL only when necessary. At scale, RL also creates major systems problems. An inference service must generate rollouts while a training service updates the model. Environments may include code execution. If rollout workers lag behind the trainer, their data becomes off-policy. The system must continually trade off fresh, on-policy data against maximum throughput.

The precise fifth assignment was still being decided at lecture time. The previous year's version implemented DPO and GRPO for a mathematical benchmark, and the staff hoped to make the new version more realistic.

### Source reconciliation

Slides 25-26 list DPO and GRPO as Assignment 5 tasks, while the spoken lecture explicitly says that the current year's exact assignment had not been finalized.

### Additional explanation

"Weak" does not mean useless. A scalar score, pairwise preference, unit test, or verifier supplies less information than a complete ideal response, but it may be much cheaper or more reliable to obtain.

The generate-score-update loop also clarifies why alignment is a systems problem:

- generation consumes inference capacity;
- scoring may invoke humans, tools, environments, or other models;
- training consumes accelerator capacity;
- model versions and data freshness determine how on-policy the updates are.

DPO avoids an explicit online RL loop by learning directly from preferred and rejected response pairs. PPO and GRPO more directly optimize reward-based objectives, but introduce rollout and policy-update machinery. None of these algorithms by itself guarantees broad human-value alignment; here, "alignment" denotes shaping model behavior toward the preferences or reward signals supplied by the training process.

## 18. Efficiency as the unifying lens

**Transcript coverage:** lines 5149-5305

### What the lecturer said - transcript only

The model builder has a set of resources: data and hardware, with hardware contributing compute, memory, and communication bandwidth. The goal is to achieve the best evaluation result with those fixed resources.

The major course decisions can all be viewed through this lens:

- **Systems** directly improves compute, memory, and communication efficiency.
- **Tokenization** avoids the computational inefficiency of operating on every raw byte with current architectures.
- **Architecture** often changes to reduce FLOPs, memory use, or inference cost.
- **Data filtering** prevents a fixed compute budget from being spent on redundant or low-quality examples. Even harmless bad data has an opportunity cost because it displaces better data.
- **Scaling laws** use smaller, cheaper models to choose hyperparameters for expensive runs.

The field may become data-constrained rather than compute-constrained, which would change the optimal design decisions. The durable lesson is to identify the active resource constraint and reason about the efficiency of the entire approach.

### Additional explanation

Optimization changes bottlenecks. Faster kernels may make communication dominant; shorter token sequences may make the output vocabulary more expensive; abundant hardware may expose a data shortage. Efficiency work should therefore be iterative:

1. State the objective and fixed resources.
2. Measure the current bottleneck.
3. Change one part of the stack.
4. Measure again, because the bottleneck may have moved.

---

# Part IV - Tokenization

## 19. What problem does a tokenizer solve?

**Transcript coverage:** lines 5320-5379

### What the lecturer said - transcript only

The lecturer recommends Andrej Karpathy's tokenization video as a useful companion. Raw text is represented as a Unicode string, whereas a language model defines a probability distribution over sequences of tokens, which are usually stored as integer indices. A tokenizer supplies the bridge in both directions:

$$
\operatorname{encode}: \text{string} \rightarrow \text{list of token IDs},
$$

$$
\operatorname{decode}: \text{list of token IDs} \rightarrow \text{string}.
$$

A correct tokenizer must support a round trip:

$$
\operatorname{decode}(\operatorname{encode}(s)) = s.
$$

If encoding and then decoding does not reconstruct the original string, the tokenizer is broken for that input.

### Additional explanation

Token IDs are **categorical labels**, not measurements. If the IDs for two strings are 500 and 501, those numbers being adjacent says nothing about the meanings of the strings. The embedding table learns a vector for each ID; semantic relationships live in the learned vectors, not in numerical proximity between IDs.

There are two distinct design questions:

1. **Coverage:** Can every valid input be represented and reconstructed?
2. **Segmentation:** How many tokens represent the input, and where are its boundaries?

Byte-based BPE solves coverage by beginning with all 256 byte values, then learns a more efficient segmentation by merging frequent byte sequences.

## 20. Tokenization is visible in surprising ways

**Transcript coverage:** lines 5380-5518

### What the lecturer said - transcript only

Real tokenizers have unintuitive behavior:

- A word and the same word preceded by a space may be different tokens. Many vocabulary entries effectively represent "space + word."
- The same visible word at the beginning of a string and in the middle can map to unrelated token IDs because one version includes a leading space.
- Numbers may be split into groups of a few digits. The grouping is predictable in some tokenizers and irregular in others. Making every digit a separate token would simplify the rule but would also lengthen numeric sequences.

The lecture demonstrates a GPT tokenizer mapping a multilingual string to integer IDs and decoding the IDs back to exactly the same string.

### Additional explanation

These quirks are consequences of a tokenizer optimizing corpus frequency rather than human linguistic elegance. Spaces are highly predictive and common, so merging them with following text can improve compression. Digit groupings reflect which substrings were frequent and which pre-tokenization rules were used during training.

Token boundaries matter because the model performs one prediction step per token. A simple arithmetic string, identifier, or non-English phrase may require very different numbers of steps under different tokenizers even when it has the same number of visible characters. This can change effective context length and task difficulty.

## 21. Compression ratio and vocabulary size

**Transcript coverage:** lines 5519-5607

### What the lecturer said - transcript only

The lecture defines tokenizer compression ratio as:

$$
r = \frac{\text{number of UTF-8 bytes in the string}}{\text{number of tokens}}.
$$

For the example in the lecture, the string contains 20 bytes and becomes 8 tokens, so:

$$
r = \frac{20}{8} = 2.5 \text{ bytes/token}.
$$

A larger ratio means fewer tokens for the same byte string. Shorter sequences are valuable because attention cost grows quadratically with sequence length.

Compression can be increased by enlarging the vocabulary, but this produces a sparsity tradeoff. Every vocabulary element is treated as a distinct category, so rare large tokens receive relatively little evidence. Modern multilingual tokenizers may contain roughly 100,000 to 200,000 distinct tokens.

### Additional explanation

If a byte sequence of length $B$ has compression ratio $r$, then its token length is:

$$
T = \frac{B}{r}.
$$

For full attention, the attention matrix scales approximately as $T^2$. Holding everything else fixed, doubling $r$ halves $T$ and reduces this quadratic term by about four times. Not every part of a Transformer is quadratic, so total runtime will improve by less than that, but the effect can still be substantial.

A larger vocabulary has costs:

- the input embedding table grows approximately as $Vd$, where $V$ is vocabulary size and $d$ is hidden dimension;
- the output classifier or softmax also grows with $V$ unless weights or computation are specially structured;
- rare tokens receive fewer updates;
- a token that memorizes a long surface form does not automatically provide useful compositional structure.

Compression ratio should also be compared carefully across languages. A byte-normalized measure is more informative than tokens per visible character because UTF-8 uses different numbers of bytes for different code points.

## 22. Three simple tokenization schemes and why each is inadequate

### 22.1 Character-level tokenization

**Transcript coverage:** lines 5608-5712

#### What the lecturer said - transcript only

A Unicode string is already a sequence of Unicode characters. Each character can be mapped to an integer code point with Python's `ord` operation and reconstructed with `chr`, so a character tokenizer is easy to implement and can round-trip a string.

However, Unicode contains roughly 150,000 characters. A character vocabulary can therefore be very large, while many entries are extremely rare. The model spends vocabulary capacity on symbols that receive little training. At the same time, common text still requires many tokens, so compression is poor. The lecturer describes this as an unfavorable combination of a large vocabulary and low compression.

#### Additional explanation

"Character" is also more ambiguous than it appears. Unicode code points do not always correspond to user-perceived characters. A visible symbol can be composed from multiple code points, and visually identical text can have different normalized representations. A code-point tokenizer is deterministic over the input string, but it does not automatically discover grapheme clusters, morphemes, or words.

Its main advantage is conceptual simplicity: token boundaries do not depend on a learned corpus. Its disadvantage is that the model must learn common multi-character units only after several Transformer steps have already processed them separately.

### 22.2 Byte-level tokenization

**Transcript coverage:** lines 5713-5775

#### What the lecturer said - transcript only

Unicode strings can be encoded as UTF-8 byte sequences. Each byte is an integer from 0 through 255, giving a compact fixed vocabulary of 256 values. Some characters use one byte and others use several bytes.

A pure byte tokenizer has a compression ratio of exactly one byte per token. Its vocabulary is excellent for coverage and size, but sequences are long. With present Transformer architectures and quadratic attention, this makes raw byte modeling computationally inefficient.

#### Additional explanation

Bytes provide a universal fallback: any file or UTF-8 string is ultimately representable without an unknown token. The difficulty is not expressibility but allocation of computation. A common word, emoji, or Chinese character may require several byte-level model steps before the model can treat it as one higher-level pattern.

Byte-level modeling can still be attractive when robustness to unusual text is more important than sequence length, or when an architecture learns to pool bytes into larger internal chunks. The lecture's later argument is precisely that replacing tokenization does not remove the need for some form of learned abstraction.

### 22.3 Word-level tokenization

**Transcript coverage:** lines 5776-5877

#### What the lecturer said - transcript only

Classical NLP often segmented text into words or regex-defined chunks and assigned an integer to each distinct chunk. This has an appealing property: human-created words usually carry stable meaning, and representing whole words yields good compression.

The vocabulary, however, can be enormous and is not naturally bounded. A new word can appear at test time even if it was absent from training. Traditional systems map such items to an unknown (`UNK`) token, discarding their identity. This is inelegant and can also distort perplexity calculations. Rare words receive too little evidence for the model to learn them well.

#### Additional explanation

Word vocabularies also struggle with productive morphology, names, code identifiers, misspellings, and languages in which whitespace does not reliably mark word boundaries. Treating every surface form as atomic prevents sharing between related forms such as `compute`, `computer`, and `computing` unless the model learns the relationship independently from scarce whole-word examples.

Subword tokenization is a compromise: it keeps frequent words or word pieces intact while decomposing rare forms into reusable components.

### Comparison

| Scheme | Base units | Vocabulary | Sequence length | Coverage problem | Main weakness |
|---|---|---:|---:|---|---|
| Unicode code points | Code points | Potentially about 150K | Long | Rare but representable code points | Large sparse vocabulary and weak compression |
| UTF-8 bytes | Bytes 0-255 | 256 | Very long | None for byte strings | Too much computation per raw byte |
| Words or regex chunks | Whole chunks | Huge or unbounded | Short | Unseen words require `UNK` | Poor handling of rarity and productive forms |
| Byte-level BPE | Learned byte sequences | Fixed, chosen size | Intermediate | None if all bytes remain available | Greedy, corpus-dependent segmentation |

## 23. Byte pair encoding: history and objective

**Transcript coverage:** lines 5878-5960

### What the lecturer said - transcript only

Byte pair encoding was introduced long before modern language models as a data-compression method. It was later adapted to neural machine translation and then used for language models in GPT-2.

The tokenizer is trained on raw text to construct a vocabulary tailored to the data. Its desired behavior is:

- common byte sequences become single tokens;
- rare sequences remain decomposed into smaller tokens;
- any input remains tokenizable, because rare material can fall back all the way to bytes rather than becoming `UNK`.

Conceptually, BPE starts with each byte as a token and repeatedly merges the most frequent adjacent pair of current tokens.

### Additional explanation

BPE does not explicitly optimize linguistic units. It performs greedy corpus compression. A learned token may coincide with a word or morpheme because those strings are frequent, but it may also be a space-prefixed fragment, punctuation pattern, code substring, or arbitrary byte sequence.

The learned vocabulary contains a hierarchy. Initial tokens represent individual bytes. A later token is created by concatenating the byte strings of two earlier tokens, so every token can ultimately be expanded back into bytes. This construction is what preserves complete coverage.

## 24. Training BPE step by step

**Transcript coverage:** lines 5961-6143

### What the lecturer said - transcript only

Assume the corpus is one long byte sequence. Initially, each byte is its own token and the vocabulary contains the 256 byte values.

For the example string `the cat in the hat`, training proceeds iteratively:

1. Count every adjacent pair of current token IDs.
2. Find the pair with the largest count. In the example, the byte IDs `(116, 104)`, corresponding to `t` followed by `h`, occur twice. There are ties, and the simple implementation selects the first applicable pair.
3. Create a new token ID. The first learned token is 256, representing the byte sequence `th`.
4. Replace every applicable occurrence of `(116, 104)` with token 256.
5. Recount pairs on the shortened token sequence and repeat. A later merge can combine token 256 with byte 101 (`e`) to build a token representing `the`.

With each merge, the represented training sequence becomes shorter and the vocabulary grows by one entry. In the toy example, three merges produce a compression ratio of about 1.5 bytes per token.

### Additional explanation

A compact formalization is:

```text
tokens = UTF8_BYTES(corpus)
vocabulary = {0, 1, ..., 255}
merges = []

repeat until the desired vocabulary size is reached:
    counts = frequencies of adjacent pairs in tokens
    pair = deterministically chosen most frequent pair
    new_token = next unused token ID
    record pair -> new_token
    replace non-overlapping occurrences of pair in tokens
    add concatenated byte representation to vocabulary
```

Important implementation details follow from this seemingly simple loop:

- **Counts change after every merge.** Merging creates new neighboring pairs and removes old ones.
- **Tie-breaking must be deterministic.** Otherwise the learned tokenizer can differ across runs even on identical data.
- **Replacements are non-overlapping.** A scan must define what happens when occurrences share an element.
- **The merge list is ordered.** Later merges may use tokens created by earlier merges.

A naive implementation repeatedly scans the entire corpus to recount and replace pairs, which becomes prohibitively expensive for large training corpora. Efficient implementations update only counts and positions affected by the latest merge.

## 25. Encoding and decoding with learned BPE merges

**Transcript coverage:** lines 6144-6187

### What the lecturer said - transcript only

After training produces a vocabulary and an ordered set of merges, new text is tokenized by first converting it to bytes and then applying the learned merges. The lecture applies the tokenizer to `the quick brown fox`, obtains a token-ID sequence, and decodes it back to the original string.

Decoding reverses the representation: look up the bytes associated with each token ID, concatenate those byte strings, and decode the result as UTF-8. The reconstructed string should equal the input.

### Additional explanation

The order or **rank** of merges matters. Encoding cannot simply merge whichever pair has the largest ID or happens to look longest. It must reproduce the priority learned during training. A common conceptual implementation repeatedly applies the highest-priority available merge until none remains.

Decoding is simpler because it does not need to invert the sequence of merge operations. Every token already stores its final byte string, so concatenation directly reconstructs the original byte stream.

Lossless decoding does not imply that every string has an intuitively good segmentation. It guarantees identity preservation, while segmentation quality is judged by compression, downstream model quality, multilingual behavior, robustness, and implementation speed.

## 26. Making a practical BPE tokenizer fast and correct

**Transcript coverage:** lines 6188-6310

### What the lecturer said - transcript only

The lecture's simple implementation is complete enough to work, but it is extremely slow. Its encoder loops over every learned merge, and the number of learned merges is approximately the vocabulary size minus the 256 base byte tokens. Most merges are irrelevant to any one input, so a practical implementation should identify and process only merges that can actually occur.

Modern tokenizers also need special-token handling. This is not conceptually deep but is important for correctness. In addition, production tokenizers do not usually run BPE across one entire unstructured string. They first split text into chunks and apply the tokenizer to each chunk, which greatly improves speed and controls where merges can occur.

Assignment 1 asks students to optimize these components as aggressively as possible. If Python becomes the limiting factor, students may implement performance-critical parts in a compiled language such as Rust or C.

### Additional explanation

Three implementation concerns should be kept separate:

1. **BPE training:** learn merge ranks from a large corpus.
2. **Pre-tokenization:** split new text using a deterministic rule, often a regular expression.
3. **BPE encoding:** apply relevant merge ranks within each pre-tokenized chunk.

Pre-tokenization changes the model's possible vocabulary. If chunks are separated before BPE, no learned token can cross those boundaries. This is one reason spaces, contractions, and digit groups behave differently across tokenizer families.

Efficient encoders can maintain neighboring symbols in a linked structure and use a priority queue keyed by merge rank. When a merge occurs, only the local neighbors need to be reconsidered. The exact data structure varies, but the objective is the same: avoid rescanning every merge rule over the whole input.

Special tokens must be treated as reserved control symbols rather than ordinary byte substrings. Examples include end-of-text markers or role separators. Their recognition, escaping, and permitted use affect both correctness and security, because accidental or adversarial insertion of control tokens can alter the structure seen by the model.

## 27. Why tokenization may eventually change, but abstraction will remain

**Transcript coverage:** lines 6311-6432

### What the lecturer said - transcript only

The lecture summarizes tokenization as a bidirectional mapping between strings and token IDs. Character-, byte-, and word-based schemes are each highly suboptimal in a different way, while BPE is an effective data-driven heuristic.

The lecturer hopes tokenization may eventually be replaced by an end-to-end byte-level approach. Nevertheless, any replacement must satisfy two requirements:

1. The Transformer or other sequence model should operate on useful **chunks or abstractions**, not remain forever at raw low-level units.
2. Chunk size should be variable so computation is allocated adaptively. Not every byte deserves the same amount of modeling capacity.

This is especially clear beyond text. In video or DNA, individual low-level units can have a poor signal-to-noise ratio, so the model must lift them into higher-level representations before effective modeling. A tokenizer-free system that treats every byte identically would remain inefficient.

### Additional explanation

Eliminating a separately trained tokenizer does not eliminate segmentation; it moves segmentation inside the model. A future system might learn pooling boundaries, compress predictable spans, or assign more depth and computation to surprising regions. The central design problem becomes **adaptive resolution** rather than a fixed text preprocessing rule.

An ideal learned system would try to combine:

- byte-level coverage and robustness;
- subword-level compression;
- context-dependent boundaries;
- fair capacity across languages and data types;
- differentiable, end-to-end learning;
- efficient execution on accelerator hardware.

The hard part is achieving these properties without making the model or training procedure more expensive than the tokenization it replaces.

## 28. Transition to the next lecture

**Transcript coverage:** lines 6433-6457

### What the lecturer said - transcript only

The next lecture begins resource accounting, described as an introductory systems unit. After resource accounting, the course returns to model architectures.

### Additional explanation

This transition is deliberate. Before comparing architectures, students need to count their parameter memory, activation memory, FLOPs, and data movement. Otherwise, statements such as "this architecture is efficient" cannot be evaluated quantitatively.

---

# Consolidated takeaways

1. Full-stack understanding expands the research design space beyond prompting a fixed model.
2. Small models teach mechanics and methodology, but not every empirical intuition transfers to frontier scale.
3. Algorithms matter when they convert resources into capability efficiently and continue to work as scale increases.
4. Modern language-model development is a co-design problem spanning data, architecture, optimization, hardware, distributed systems, inference, and evaluation.
5. Scaling requires a predictable recipe, not merely a larger target model.
6. Data and evaluation jointly specify the capabilities the model is being built to acquire.
7. Alignment methods use cheaper evaluative signals when ideal target responses are difficult to supply, but create both optimization and systems challenges.
8. A tokenizer must round-trip text while choosing units that balance coverage, sequence length, vocabulary size, and model efficiency.
9. Character, byte, and word tokenization each optimize one property while failing badly on another.
10. Byte-level BPE preserves universal byte coverage and learns frequent larger units through repeated adjacent-pair merges.
11. Practical tokenizers require deterministic merges, pre-tokenization, special-token handling, and efficient data structures.
12. Even if explicit tokenization disappears, models will still need learned abstractions and adaptive computation over low-level inputs.

# Key equations

## Conceptual progress decomposition

$$
\text{accuracy} = \text{efficiency} \times \text{resources}.
$$

This is a framing device from the lecture, not a literal universal law.

## Dense Transformer training estimate

$$
C \approx 6ND.
$$

- $C$: approximate training FLOPs
- $N$: model parameters
- $D$: training tokens

## Rough compute-optimal token rule

$$
D \approx 20N.
$$

This is a rule of thumb whose constant depends on the architecture, data, and objective, and it does not account for inference cost.

## Tokenizer compression ratio

$$
r = \frac{\text{UTF-8 bytes}}{\text{tokens}}.
$$

A larger $r$ means fewer model tokens for the same byte sequence.

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Abstraction | A higher-level interface that hides lower-level implementation details. It is leaky when hidden details still affect outcomes. |
| Adaptive computation | Allocating different amounts of model computation to different parts of an input rather than treating every raw unit equally. |
| Arithmetic intensity | FLOPs performed per byte moved; useful for reasoning about compute- versus memory-bound operations. |
| BPE | Byte pair encoding, a greedy procedure that repeatedly merges frequent adjacent token pairs. |
| Collective operation | A distributed communication primitive such as gather, reduce, or all-reduce. |
| Compression ratio | UTF-8 bytes represented per model token. |
| Decode | The autoregressive inference phase that generates tokens one at a time. |
| Ecological validity | The extent to which an evaluation reflects the actual real-world use case. |
| Hyperparameter transfer | The ability to reuse or predict hyperparameters across model scales. |
| Kernel | A function executed on an accelerator such as a GPU. |
| Mid-training | A higher-quality or specialized training stage near the end of broad pretraining. |
| On-policy data | Rollouts generated by the current policy, rather than an older version of the model. |
| Open weights | Publicly available model parameters; this need not include training code or data. |
| Prefill | The inference phase that processes all prompt tokens together and constructs the key-value cache. |
| Pre-tokenization | Splitting text into chunks before applying learned BPE merges. |
| Scaling recipe | A mapping from compute budget to model and training hyperparameters. |
| Speculative decoding | Using a cheaper draft model to propose tokens that a larger model verifies in parallel. |
| Token ID | An integer label for one vocabulary entry; numerical proximity between IDs has no semantic meaning. |

# Self-check questions

1. Why can a small-model architectural improvement fail to produce the same benefit at frontier scale?
2. What is the difference between mechanics, mindset, and empirical intuition?
3. Why is "algorithms that scale matter" different from "algorithms do not matter"?
4. How do internal and external evaluations serve different purposes?
5. Why can a compute-optimal training configuration be suboptimal once inference cost is included?
6. What does $C \approx 6ND$ count, and what important details does it ignore?
7. Why is decoding typically more memory-bound than prefill?
8. Why does byte-level tokenization have perfect coverage but poor compute efficiency?
9. Why does a word-level tokenizer require an unknown-token strategy?
10. In BPE, why must adjacent-pair counts be updated after each merge?
11. Why must BPE tie-breaking and merge ordering be deterministic?
12. How can a larger vocabulary improve attention cost while increasing other costs?
13. What is the difference between removing a separate tokenizer and removing segmentation altogether?
14. Why does filtering bad but harmless data still improve a model trained under a fixed compute budget?
15. Why does large-scale RL training create a tradeoff between throughput and on-policyness?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-295 | Teaching team, third offering, from-scratch philosophy, course updates | Yes |
| 298-972 | Motivation, industrialization, scale transfer, knowledge types, efficiency | Yes |
| 973-1608 | Historical development, open ecosystem, agents and stable fundamentals | Yes |
| 1609-2226 | Executable lecture, logistics, assignments, AI policy, compute | Yes |
| 2227-2922 | Syllabus and basics unit | Yes |
| 2923-3688 | Systems, kernels, parallelism, inference, Assignment 2 | Yes |
| 3691-4379 | Scaling recipes, compute-optimal laws, prediction, Assignment 3 | Yes |
| 4380-4935 | Evaluation, data curation and processing, Assignment 4 | Yes |
| 4936-5305 | Alignment, RL systems, Assignment 5, unifying efficiency lens | Yes |
| 5308-5320 | Pause for questions and transition to tokenization; no substantive question was recorded | Yes |
| 5320-5607 | Tokenizer interface, quirks, round-trip, compression, vocabulary | Yes |
| 5608-5877 | Character, byte, and word tokenization | Yes |
| 5878-6187 | BPE motivation, training, encoding, and decoding | Yes |
| 6188-6432 | Implementation performance, special tokens, pre-tokenization, future requirements | Yes |
| 6433-6457 | Next lecture: resource accounting | Yes |
