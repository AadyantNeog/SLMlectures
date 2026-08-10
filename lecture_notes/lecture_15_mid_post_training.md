---
title: "Lecture 15 - After Pretraining: Mid- and Post-Training"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 15
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 15 MidPost-Training.txt"
slide_deck: "../lecture_15.pdf"
status: "complete"
---

# Lecture 15: After Pretraining - Mid- and Post-Training

## How to read these notes

Each substantive topic has two layers:

1. **What the lecturer said - transcript only.** A concise paraphrase of the complete spoken lecture, preserving substantive claims, examples, qualifications, numerical details, and audience questions while removing filler and repetition.
2. **Additional explanation.** Independent intuition, derivation, organization, and study guidance. This material is not attributed to the transcript.

The raw transcript's line spans are shown for auditability. All 65 slides were rendered and visually inspected after the transcript map was established. Slides were used to verify paper names, figures, data examples, objectives, and equations. Material differences or slide-only precision appear under **Source reconciliation**.

## Lecture map

The lecture explains how a strong base model becomes a useful instruction-following system:

1. Use supervised fine-tuning (SFT) demonstrations to elicit desired behavior.
2. Understand how SFT data evolved from converted NLP benchmarks to chat, synthetic instruction data, and structured agent/tool traces.
3. Treat style, knowledge, factuality, safety, scale, and annotator composition as separate data-design problems.
4. Blur the boundary between pretraining and post-training through high-quality decay-stage or mid-training mixtures.
5. Move from imitating demonstrations to optimizing preferences through reinforcement learning from human feedback (RLHF).
6. Compare PPO with direct preference optimization (DPO), then study reward overoptimization, mode collapse, entropy loss, and calibration.

The next lecture develops reinforcement learning with verifiable rewards (RLVR) and reasoning models in greater depth.

---

# Part I - From a base model to instruction following

## 1. Why post-training is necessary, powerful, and opaque

**Transcript coverage:** lines 1-239

### What the lecturer said - transcript only

The course has built enough machinery to produce something like a stronger GPT-3: a large, capable base model. Yet GPT-3's practical utility was limited. It could perform copywriting and playful completions where reliability and precise control were not essential, but steering it through few-shot prompts was fragile. The transition from GPT-3 to ChatGPT made language models feel like useful systems rather than impressive text completers.

Instruction following is remarkable because a user can provide a long, detailed, program-like request and a later model can satisfy many constraints in one attempt. Pretraining is indispensable: it supplies the broad, diverse capabilities, and no amount of downstream steering can replace a weak base. Post-training then extracts desired behaviors from what the lecturer calls the "primordial soup" of pretraining through explicit data collection and engineering.

The lecture will move from a GPT-3-like base toward a ChatGPT-like assistant. The next lecture will move from that assistant toward o1-style thinking models. The broad process from base model to deployed assistant is called **post-training**.

Modern post-training is unusually opaque. Older papers provide far more operational detail. Stiennon et al.'s work on learning to summarize from human feedback includes annotation instructions, and Anthropic's 2022 helpful-and-harmless work describes safety annotation. As competition intensified after ChatGPT, vendors stopped disclosing comparable information because post-training data became a trade secret. Open-source recipes reveal more artifacts but often depend on distillation from closed models, which differs from frontier laboratories' direct human-data collection.

The lecturer gives a 2023 example involving leaked Scale AI material. Workers trying to improve Google Bard were reportedly asked to study GPT-4 and produce responses with greater detail and quality. The episode illustrates the competitive reverse engineering around post-training data.

Algorithms are comparatively well understood; the main leverage and secrecy lie in the data. That constraint means the lecture can describe the public algorithmic recipe more confidently than current frontier annotation practice.

### Source reconciliation

The slides verify the older detailed references as **Stiennon et al. (2020)** and **Bai et al. (2022)**. They also show the leaked Scale AI excerpt describing a comparison of Bard rewrites against GPT-4, which grounds the anecdote that is only summarized in speech.

### Additional explanation

Pretraining and post-training solve different distribution problems:

```text
pretraining: learn broad regularities from naturally occurring sequences
post-training: concentrate probability on behaviors useful in an interface
```

The second problem is smaller in token count but more judgment-intensive. A demonstration silently specifies many things at once: correctness, style, refusal policy, tool syntax, answer length, citation behavior, and what the model should do under ambiguity. This is why a small post-training corpus can exert disproportionate influence.

## 2. The two-part RLHF recipe and the role of SFT

**Transcript coverage:** lines 240-320

### What the lecturer said - transcript only

The classical RLHF recipe has two broad phases. First, collect **demonstration data**: for each prompt, an annotator writes a reference response, and the model is fine-tuned on those prompt-response pairs. This is supervised fine-tuning. Second, collect judgments over model responses and use reinforcement learning to shape the model toward what raters consider better behavior.

The lecture begins with SFT because its optimization is familiar. Apart from minor variations, training looks like pretraining: predict tokens and use gradient descent. The consequential difference is the data distribution. The discussion will therefore focus on what SFT examples contain, how instruction datasets developed, how much scale is needed, and which collection pitfalls resemble classical supervised-learning failures.

### Additional explanation

The three-stage implementation often shown in RLHF papers is:

```text
prompts + expert demonstrations -> SFT policy
prompts + sampled candidates + rankings -> reward or preference signal
SFT policy + preference signal -> optimized policy
```

Calling SFT "the same as pretraining" refers to the next-token loss, not to data semantics. Implementations may mask prompt tokens and optimize only assistant tokens, or they may predict the complete formatted conversation. That choice changes which behavior receives direct supervision.

## 3. The historical progression of open SFT data

**Transcript coverage:** lines 321-460

### What the lecturer said - transcript only

The lecturer sketches a chronology of open instruction data:

- **FLAN** converted many existing supervised NLP datasets into instruction tasks and used them to train T5. The core idea was visionary: collect downstream tasks and train on all of them.
- **Self-Instruct** asked whether a language model could generate its own instruction data. As models improved, their responses could rival or exceed ordinary annotators.
- **Alpaca** distilled ChatGPT-style traces into prompt-response pairs.
- **Vicuna** used prompts and responses shared by online users through ShareGPT-like sources.
- **OpenAssistant** used a Wikipedia-like volunteer effort to write prompts and responses directly.
- **WizardLM**, **Tulu 3**, and related projects designed increasingly elaborate synthetic instruction-generation pipelines.
- Newer **Nemotron**-style corpora include agent and tool-use traces rather than only textual chat.

Closed laboratories may also distill other systems, but human-data collection remains a major private ingredient that this open chronology does not reveal.

An audience member asked how much correctness matters in prompt-response pairs. The lecturer gives a deliberately nuanced answer. Builders should seek the best responses they can because bad content teaches bad behavior. Yet models can learn instruction following from surprisingly strange or incomplete examples, even in work that removed much of the normal response signal. A capable pretrained model generalizes enough that response quality and behavior induction are not identical questions.

### Source reconciliation

The progression slide also names **ShareGPT/Vicuna** explicitly and labels the final category **Nemotron tool use**. These names clarify several compressed or automatically transcribed phrases in the spoken chronology.

### Additional explanation

This history changes the source of supervision:

| Era | Main source of targets | Principal advantage | Principal risk |
|---|---|---|---|
| Converted benchmarks | Human labels collected for older NLP tasks | Diversity and clear evaluation tasks | Unnatural prompts and inherited benchmark artifacts |
| Crowdsourced chat | Volunteers or paid workers | Natural interaction and richer prose | Variable expertise, cost, bias, and quality control |
| Distillation | Strong teacher model | Cheap, scalable, coherent outputs | Teacher imitation, correlated errors, legal or policy constraints |
| Agentic synthesis | Models plus tools and structured simulators | Teaches workflows, calls, plans, and state transitions | Interface overfitting and synthetic-reality gaps |

The key distinction is between learning the **interface behavior** of following instructions and adding the **underlying capability** needed to solve them. SFT can often reveal an existing capability with weak targets; it is less reliable as a way to create missing knowledge or reasoning skill.

---

# Part II - What SFT data teaches

## 4. FLAN: scale from benchmarks, with inherited unnaturalness

**Transcript coverage:** lines 461-589

### What the lecturer said - transcript only

FLAN established multitask instruction tuning: if a model will face many downstream tasks, gather those tasks and train them together. Looking inside the dataset, however, reveals unnatural interactions.

One example takes an Enron email body, appends an instruction to write its subject line, and uses the original subject as the target. The task exists because the source dataset happens to contain bodies and subjects, not because users naturally prompt assistants that way. Another converts a news summarization dataset by appending "highlights for this article" and treating its short reference summary as the answer. Compared with ChatGPT, these summaries are unusually terse and can include details unsupported by the source. FLAN inherits both quality deficiencies and awkward structure from the older datasets it reformats.

Early instruction tuning also imported pretraining's assumption that scale was essential. FLAN combined many tasks and many examples. Later work showed that a sufficiently capable base model can acquire useful instruction behavior from far fewer, higher-quality demonstrations. The field had to explore the scale-quality space to discover that the original mixture occupied the wrong point on that tradeoff.

### Additional explanation

Converting a dataset into instructions does not erase its collection objective. A benchmark built to test subject prediction, classification, or short-reference overlap still carries those assumptions after a natural-language wrapper is added.

An audit should therefore ask:

- Would a real user request this task in this form?
- Is the target a genuinely good assistant answer or merely an available label?
- Does the source contain hallucinated, truncated, or stylistically obsolete references?
- Is task scale achieved by duplicating templates rather than adding behavioral diversity?

## 5. Alpaca and OpenAssistant: natural chat and expert detail

**Transcript coverage:** lines 590-681

### What the lecturer said - transcript only

After ChatGPT, Stanford Alpaca distilled ChatGPT traces into instruction-response pairs. Its prompts looked more natural and its answers were longer and more conversational than converted benchmarks. When applied to the original LLaMA models, these examples reliably induced ChatGPT-like behavior. Both sides mattered: a strong pretrained model and suitable post-training data.

The result produced optimism that a large, high-quality open instruction corpus might close the gap with proprietary systems. OpenAssistant was a major crowdsourced attempt. Volunteers wrote difficult prompts and detailed responses, resembling Wikipedia's community-production model. The project produced on the order of ten thousand or more examples before losing momentum. Its answers could be lengthy, expert, and well sourced.

The lecturer regards OpenAssistant as admirable and useful, while postponing its important pitfalls: detailed responses can train behaviors that the model cannot reliably generalize.

### Additional explanation

Alpaca showed that **behavioral distillation** is cheap relative to pretraining. The student need not copy every teacher fact; it can learn surface policies such as answering directly, using explanatory prose, following imperative prompts, and ending cleanly.

OpenAssistant demonstrated the opposite source tradeoff. Direct human creation avoids dependence on one teacher model, but coordinating experts, enforcing consistent policies, and sustaining volunteer effort are difficult. High-quality community data is not automatically high-throughput data.

## 6. Agentic SFT and the shift toward structured outputs

**Transcript coverage:** lines 682-764

### What the lecturer said - transcript only

The desired product has shifted from a text-only chat interface toward an agent that uses tools, manages tasks, and acts through structured APIs. Systems such as Claude Code or Codex create to-do lists and update them as work progresses. New SFT datasets must teach these behaviors explicitly.

Nemotron examples include assistant text alongside structured tool calls, sometimes allowing calls to occur in parallel. These are still supervised targets: the serialized calls, arguments, plans, and status updates are directly predicted during SFT.

Across the history of instruction data, the lecturer identifies three high-level changes:

1. **Chattiness:** older NLP datasets map inputs to programmatic labels, while users expect detailed, human-like communication.
2. **Detail and annotator quality:** later projects ask knowledgeable annotators for richer responses.
3. **Tool use:** contemporary data must specify an interface for planning and acting, not only a prose style.

### Source reconciliation

The slide's Nemotron-SFT-OpenCode examples show JSON-like `tool_calls`, function names and arguments, and an explicit to-do list with IDs, priorities, and statuses. These concrete schemas are visible on the slide but not read aloud in the transcript.

### Additional explanation

Tool-use data teaches a protocol with at least four separable decisions:

```text
whether to call a tool
which tool to select
how to serialize valid arguments
how to use the result in the next state
```

A corpus that shows only successful calls may not teach recovery from invalid arguments, timeouts, partial results, or permission failures. Agentic SFT benefits from trajectories containing state changes and correction loops, not merely isolated tool-call strings.

## 7. Style, length, and the difference between preference and capability

**Transcript coverage:** lines 765-887

### What the lecturer said - transcript only

Anyone collecting SFT data must make conscious choices about formatting, knowledge, scale, safety, length, and style. Differences in Claude's tone or ChatGPT's chattiness are not accidents; they reflect post-training data and preference decisions. Public instruction datasets vary widely in response length.

Style strongly affects pairwise evaluation. Human and model judges often prefer bullet-pointed, longer, or more detailed responses when two answers are placed side by side. That preference can be reasonable in a comparison task, yet repeated selection creates a chatbot tone unlike ordinary human conversation.

Engagement or win-rate signals can therefore be misleading. Training on some instruction datasets yields large gains on preference-based evaluations such as AlpacaEval while barely changing conventional capability benchmarks. A model can become more appealing without becoming more knowledgeable or better at reasoning. Style control and capability control should be analyzed separately.

### Source reconciliation

The slides verify substantial completion-length variation across datasets and show strong evaluator preferences for lists and longer answers. A benchmark table illustrates the spoken warning: instruction mixtures can differ greatly on AlpacaEval while their factuality, reasoning, multilingual, and code results move differently.

### Additional explanation

Preference score is a confounded metric:

$$
\text{observed win rate}
= f(\text{correctness},\text{relevance},\text{style},\text{length},\text{judge biases},\text{prompt set}).
$$

Length-controlled evaluation, rubric decomposition, and task benchmarks can reduce this ambiguity. None is sufficient alone: a benchmark may miss helpfulness, while a preference judge may reward confident verbosity.

## 8. Citations, tail knowledge, and hallucination pressure

**Transcript coverage:** lines 888-1049

### What the lecturer said - transcript only

OpenAssistant provides an instructive high-quality example: an answer includes a detailed academic citation. SFT on that answer teaches two things simultaneously. Through next-token prediction, it teaches the specific citation and surrounding facts. Through behavioral imitation, it teaches that a polished answer should emit a reference.

These signals can become entangled. If the model does not reliably know whether a citation is correct, it may generalize the format without the knowledge and hallucinate a plausible reference. The lecturer describes a supported piece of folklore: fine-tuning on facts the model does not already know can increase hallucination and cause overfitting, whereas training on known facts does not show the same failure as strongly.

John Schulman offers an RL-centered argument. A demonstration is externally selected and can force the model to emit information regardless of its own uncertainty. Reinforcement learning acts on outputs sampled from the model's policy and can reward citing when the model is in a state associated with knowing, while penalizing citations when it is not. This may align the output policy with internal knowledge.

The resulting caution is counterintuitive: the most expert-looking SFT answer is not always the safest target if it contains tail knowledge absent from the base model. Markers such as "References:" can pressure a model to complete an unknown citation. Knowledge storage and knowledge extraction are separate, messy problems.

### Additional explanation

This is a multi-task interference problem hidden inside one target. The target sequence asks the model to learn:

- a fact;
- whether the fact is known in the current context;
- when a citation is appropriate;
- the exact syntax and content of the citation;
- a policy for abstaining when confidence is low.

SFT supplies only the positive sequence. It does not directly show counterfactual cases where the model should omit a citation or say that it cannot verify one. Contrastive or policy-dependent feedback can make that boundary more explicit.

## 9. What "tail knowledge" means and why RL can help only conditionally

**Transcript coverage:** lines 1050-1152

### What the lecturer said - transcript only

Asked for a definition of **tail knowledge**, the lecturer says there is no formal boundary. Article length on Wikipedia can serve as a rough proxy for how widely known a fact is, but knowledge itself cannot be cleanly divided into head and tail. Empirical work can compare fine-tuning on more versus less common information and observe more hallucination on the latter without producing a universal definition.

Another audience member asks why reinforcement learning would help. The lecturer gives a folk account. Suppose the model's activations contain some direction correlated with "I know this" versus "I do not know this." SFT forces a reference target in every supervised example, so the model may learn to emit references regardless of that direction. With RL, outputs produced in a know-like state can receive positive reward and references produced in an uncertain state can receive negative reward. The policy can then learn to connect internal calibration to visible behavior.

RL cannot create calibration if the model contains no relevant signal. It can only extract and use information that exists somewhere in the model.

Asked where SFT penalizes a bad reference, the lecturer says SFT is not incorrectly optimizing the provided example. It assigns loss to all alternatives to the exact target. The problem is generalization: the target entangles a citation template with factual content, and the model may transfer the easier template while failing to transfer the knowledge.

### Additional explanation

Policy-dependent feedback matters because the same prompt can expose different states across models or checkpoints. A target written by an external expert may be easy for the expert but outside the student's competence. Evaluating the student's own candidate makes uncertainty and error visible in the data-generation loop.

This does not imply that RL always fixes hallucinations. The reward must detect factual errors, the sampled policy must explore abstention or a correct answer, and the model must have enough internal signal to distinguish them.

## 10. Safety tuning as a violation-versus-over-refusal tradeoff

**Transcript coverage:** lines 1153-1249

### What the lecturer said - transcript only

Post-training must confront how deployed systems will be misused. The lecture mentions political manipulation, disinformation, individualized spear phishing, and other harmful applications. Safety controls are usually enforced by training the assistant to refuse malicious requests, making the post-training team a last line of defense.

Public information about safety SFT is even sparser than information about capability SFT. Llama 2 provides one of the more detailed descriptions, yet does not clearly disclose how many safety examples it used. The central tuning problem balances two errors:

- **violation rate:** unsafe requests that receive harmful assistance;
- **false refusal rate:** benign requests that trigger a refusal, such as "How do I kill a Python process?"

These objectives form a Pareto tradeoff. Dataset builders create adversarial, borderline, and benign examples to improve one side without damaging the other. Typical safety collections contain thousands to tens of thousands of examples; the lecturer estimates Llama 2 used a few thousand.

### Additional explanation

Safety is not a single binary label. A good collection varies:

- user intent and ambiguity;
- whether information is dual use;
- request specificity and operational detail;
- legitimate professional or educational context;
- indirect, encoded, multilingual, and multi-turn attacks;
- benign words that superficially resemble harmful requests.

A scalar violation rate can hide category failures. The Pareto curve should be inspected by domain and user population, not only as one aggregate point.

## 11. Tulu 3 and WildChat: mine real failures, then compose targets

**Transcript coverage:** lines 1250-1313

### What the lecturer said - transcript only

Because most model reports remain vague, the lecturer points to AI2's open OLMo/Tulu post-training work as one of the few reasonably strong public pipelines with meaningful detail. Its safety component includes roughly 50,000-scale datasets.

The data strategy is straightforward. WildChat offered free chat access and, with collection as part of the arrangement, recorded many real interactions. Builders mined those logs for unsafe requests and attempted jailbreaks. They then created preferred responses that resist a jailbreak or refuse an unsafe request.

The lecturer believes closed companies follow the same broad pattern: inspect usage logs, find new failure modes, and have annotators play whack-a-mole by adding examples that address them.

### Source reconciliation

The speech loosely calls this an OLMo pipeline; the slide names the public recipe **Tulu 3**, an AI2 post-training pipeline used with OLMo-family models. Its safety table lists **Tulu 3 CoCoNot (10,983)**, **Tulu 3 WildJailbreak (50,000)**, and **Tulu 3 WildGuardMix (50,000)** with different subsets used at SFT and preference-training stages. The slide also distinguishes two WildTeaming operations: mine user-written jailbreak tactics and compose adversarial attacks from those tactics.

### Additional explanation

Production logs create an adaptive data loop:

```text
deploy -> observe attacks and false refusals -> cluster failure modes
       -> write or synthesize corrected examples -> retrain -> reevaluate
```

This loop is effective because it follows the real attack distribution. It also requires privacy safeguards, clear consent, retention limits, and controls against training on secrets or sensitive user content.

## 12. Small-data steering, long-tail precision, and the SFT/RL boundary

**Transcript coverage:** lines 1314-1472

### What the lecturer said - transcript only

A sufficiently capable pretrained model can be steered with remarkably few examples. The lecturer says that roughly 500 well-designed safety examples can dramatically reduce compliance with malicious, hate-speech, or other unsafe instructions. This suggests that a safe-versus-unsafe behavioral direction already exists in the pretrained model and needs only a small push.

That result does not make large collections unnecessary. Frontier products need fine-grained distinctions across many scenarios, languages, policies, and borderline cases. Hundreds of examples can establish a broad mode; extensive data is still required to cover the long tail.

SFT works best when it extracts behavior already latent in pretraining rather than trying to install a genuinely absent capability. Factually correct additions can sometimes make hallucination worse, and quality often matters more than quantity.

An audience member asks how one can know whether a behavior is already in pretraining. The lecturer admits that the claim is imprecise. One can identify some missing abilities by failures on extremely rare programming languages or other domains, but cannot prove that a general safety feature literally existed beforehand.

Another question asks whether SFT destroys features that RL later revives. The lecturer rejects a sharp algorithmic boundary. Expert iteration and related procedures blur SFT and RL. A more useful distinction is the feedback:

- SFT provides dense supervision from an external target sequence.
- RL trains on outputs sampled from the model's own policy and supplies a more outcome-oriented signal.

Because RL reinforces the policy's own candidates, it may move less abruptly away from the model's current behavior.

### Source reconciliation

The safety-steering slide shows a sweep from 0 to 2,000 added examples across several safety datasets and highlights the large improvement by approximately **500 Alpaca-style safety examples**. The speech communicates the headline rather than each plotted condition.

### Additional explanation

The "latent direction" language is a hypothesis about representation, not proof that the desired policy is stored as one clean neuron or vector. Small-data steering can also arise from broad generalization, strong priors, and shared wording across evaluation sets.

A practical test is to evaluate both breadth and robustness:

- Does the behavior transfer to paraphrases and unseen categories?
- Does it persist under jailbreaks and long contexts?
- Does it damage unrelated capabilities?
- Does the model preserve legitimate help near the safety boundary?

## 13. SFT mechanics and the rise of mid-training

**Transcript coverage:** lines 1473-1586

### What the lecturer said - transcript only

The basic SFT method is simply gradient descent on next-token prediction. The lecturer jokingly reduces it to calling `loss.backward()` and taking optimizer steps. The important nuance is an industry trend: instruction data is increasingly mixed into the tail of pretraining rather than held for an entirely separate post-training stage.

In a two-phase recipe, ordinary web-heavy pretraining is followed by a decay or **mid-training** phase that emphasizes high-quality data and can include instruction-tuning corpora. This scales instruction exposure and shifts the model toward deployment-relevant distributions while the learning rate decays. The practice is widespread in model reports under labels such as mid-training, continued pretraining, or second-phase pretraining.

The lecturer argues that the term **base model** has become misleading. A model advertised as base may already have seen UltraChat or other synthetic chat data during its final pretraining phase. It is no longer a pure next-token model of ordinary Internet text in the historical sense.

MiniCPM is used as an example. Its stable phase is dominated by broad web, code, and C4-like data. The decay phase reduces generic data and adds more Wikipedia, Stack Exchange question-answering, UltraChat, code SFT, math, and other higher-quality or instruction-like sources.

Asked whether prompts are masked, the lecturer says this phase is pure pretraining and predicts prompt tokens as well as assistant tokens. Some ordinary SFT recipes also train on prompt tokens, so the difference is not universal.

### Source reconciliation

The slide states the combined recipe explicitly: pretrain on broad data, mix instruction data into pretraining, and still perform a short dedicated instruction-tuning round. It presents the benefit as scaling instruction tuning without catastrophic forgetting. The MiniCPM mixture charts supply the exact component proportions; the transcript discusses their qualitative change rather than reading the percentages.

### Additional explanation

The stages differ along a continuum rather than a binary boundary:

| Dimension | Broad pretraining | Mid-training | Dedicated SFT |
|---|---|---|---|
| Data mix | Large, diverse, noisy | Smaller, higher-quality, capability-targeted | Explicit prompt-response behavior |
| Loss | Usually all-token next-token loss | Usually all-token next-token loss | Often assistant-only, but not always |
| Learning rate | Stable or scheduled high regime | Decay or lower-rate continuation | Short, low-rate adaptation |
| Purpose | Build broad representation | Reweight capabilities and deployment domains | Make the interface behavior explicit |

## 14. Choosing mixtures through decay-stage ablations

**Transcript coverage:** lines 1587-1736

### What the lecturer said - transcript only

An audience member asks why data quality might be lower in the decay phase. The lecturer says the usual intuition is the opposite. The decay is closest to the deployed checkpoint and uses the lowest learning rates, so builders often reserve their best data for it. Wikipedia and Stack Exchange are commonly treated as high quality.

Another question asks how mid-training and post-training mixtures are chosen. Despite papers proposing automatic methods, the lecturer says mixture design remains dominated by trial, error, and intuition. The advantage of a short decay phase is experimental cost: a team can run many mixture ablations instead of repeating full pretraining.

Teams often remove or alter one domain, rank the downstream changes, and use that evidence to choose both the decay mixture and sometimes the earlier pretraining mixture. Court documents from litigation over Meta's book data reportedly exposed exactly this kind of internal ablation work on book subsets.

There are systematic models that map ablation deltas to weights, but they can be brittle. High-quality-only pretraining is usually impossible because there are not enough distinct tokens; a model cannot train its entire budget on Wikipedia without exhausting and repeating the source.

### Additional explanation

Ablation results are not mixture gradients. Removing domain $i$ entirely measures a large intervention and can interact with every other domain. Translating that result into a new sampling weight requires judgment.

A disciplined workflow can still improve reliability:

1. fix the checkpoint, token budget, optimizer, and evaluation suite;
2. run repeated domain removals and controlled weight changes;
3. measure capability gains, regressions, and variance;
4. inspect interactions and contamination;
5. validate the selected mix on a held-out checkpoint or scale.

---

# Part III - From imitation to preference optimization

## 15. Generative imitation versus reward maximization

**Transcript coverage:** lines 1737-1929

### What the lecturer said - transcript only

After SFT, RLHF changes the conceptual objective. Pretraining and SFT fit a reference distribution by predicting its next token. RLHF treats the language model as a **policy** and searches for a distribution whose sampled outputs obtain high reward.

This distinction permits behavior that generative modeling would regard as pathological. For a given prompt, a reward-maximizing policy could collapse to one answer with no diversity and still be optimal if that answer always scores well. RLHF does not inherently need to imitate a human response distribution.

Why optimize preferences rather than collect demonstrations forever? First, people do not always produce the answer they prefer. In a news-summarization study, several freelance writers preferred Instruct-Davinci summaries over their own work. Interviews suggested that they had written competent summaries but recognized a better approach only after seeing it. Judgment and generation are different skills.

Second, verification can be easier than generation. It may be easier to check a mathematical proof than to construct it. DeepSeek's self-verifiable mathematical reasoning is mentioned as an example. The next lecture will cover RL with verifiable rewards; this lecture stays with human-feedback RLHF.

The remainder has three components: preference-data collection, PPO and DPO algorithms, and failure modes.

### Source reconciliation

The slide formalizes the distinction as

$$
\hat p(y\mid x) \approx p^*(y\mid x)
$$

for imitation, versus choosing $\hat p$ to maximize

$$
\mathbb{E}_{y\sim p(\cdot\mid x)}[R(y,x)]
$$

for optimization. Another slide reports low overall agreement, $\alpha=0.07$, among the summary writers' preferences, illustrating the generation-verification gap beyond the spoken anecdote.

### Additional explanation

Imitation asks, "What would the reference writer emit?" Optimization asks, "Which emission scores best under this evaluator?" These agree only when the reference distribution and reward define the same optimum.

Reward optimization can exceed typical demonstrations, but it also exploits evaluator blind spots. That opportunity-risk pair drives the rest of RLHF.

## 16. The RLHF data loop and annotation guidelines

**Transcript coverage:** lines 1930-2049

### What the lecturer said - transcript only

Starting from an SFT model, builders sample several responses to each prompt, often at temperature 1 because the SFT policy remains diverse. A rater ranks the candidates, sometimes with only a binary preference. A reward model is trained from these comparisons, and reinforcement learning then changes the policy to score better under that model. The intermediary is useful because repeatedly evaluating a learned verifier can be easier than repeatedly asking a human.

Pairwise comparison interfaces show the same prompt and two responses and ask which is better, sometimes allowing a tie or strength-of-preference choice. InstructGPT's appendix is recommended as one of the last detailed public views into a frontier company's process.

InstructGPT asked annotators to balance **helpfulness, truthfulness, and harmlessness**. Helpfulness includes following intent, writing clearly, respecting international context, and avoiding needless length. Truthfulness penalizes hallucination. Harmlessness favors appropriate refusal on unsafe prompts. These dimensions conflict, so raters must make an overall judgment.

Leaked Google Bard instructions reveal a broadly similar structure but use separate rating scales, including helpfulness and presentation. They ask whether an answer addresses the prompt, is factually accurate, coherent, and easy to consume. These early guidelines are not extremely long, but they show the behavioral objectives embedded in preference labels.

### Source reconciliation

The InstructGPT slide reproduces more detailed rules than the spoken summary: follow the user's intention; clarify misleading premises; avoid unsupported facts; consider harmful physical, psychological, social, or financial effects; and rank candidates with judgment when objectives trade off. The Bard slide shows multi-level helpfulness and presentation scales rather than only a pairwise winner.

### Additional explanation

A pairwise label compresses a multidimensional comparison into one bit. If answer A is more factual and answer B is clearer, the selected winner depends on an implicit weighting that may vary across raters.

Storing rubric-level annotations alongside the overall preference enables later audits and specialized reward models. It also exposes disagreement that a single winner label would hide.

## 17. The modern annotation workforce, compensation, and labor constraints

**Transcript coverage:** lines 2050-2191

### What the lecturer said - transcript only

The annotation workforce has shifted toward higher education, higher skill, and higher cost, though one survey from a Scale AI platform is not representative of the entire industry. In that survey, roughly 70% of respondents had bachelor's or master's degrees, the modal age was around 35, and common work included creative and technical writing.

Demand for domain-specific deployment has also created bespoke expert annotation. Companies want doctors, lawyers, scientists, and other professionals to write demonstrations and judge specialized answers. Median pay across several topics can exceed \$50 per hour, while some experts receive more than \$100 per hour. The old mental model of RLHF as only cheap overseas pairwise labeling is incomplete. Annotation now forms a pyramid that still includes large amounts of lower-paid scalable work beneath a smaller layer of expensive experts.

High pay reflects real difficulty. Companies need to verify that annotators have the claimed expertise, produce correct work under time pressure, and are not quietly using a language model. Modern crowdsourcing and survey research are saturated with plausible AI-generated submissions, making human verification extremely difficult.

The lecturer recounts a Google Bard labor dispute in which workers said they had less than a minute to check long answers for correctness, a task inconsistent with the written guidelines. Large vendors have also faced criticism for outsourcing low-paid annotation. Like the broader economy, the annotation market is bifurcated between well-paid specialists and a sizable low-wage workforce.

### Source reconciliation

The worker-distribution slide gives exact figures for one Outlier/Scale AI survey: **44% bachelor's**, **32% master's**, and **34% aged 35-44**. Thus the combined degree share shown is 76%, while the transcript rounds it to "70% some." A compensation slide cites Project Stargate/Handshake AI, describes 3,000-4,000 freelancers, and shows many specialist midpoints above \$100 per hour. A separate ethics slide documents reports of Kenyan workers earning under \$2 per hour, reinforcing the spoken bifurcation.

### Additional explanation

Annotation quality is jointly constrained by task design and labor conditions:

$$
\text{usable judgment}
= f(\text{expertise},\text{time},\text{instructions},\text{incentives},\text{tools},\text{review}).
$$

Hiring a qualified person does not make an impossible time budget reasonable. Likewise, detailed guidelines cannot compensate for incentives that reward speed over verification. Data quality and labor ethics are coupled system-design choices.

## 18. Demographics, expertise, subtle transfer, and annotator quality

**Transcript coverage:** lines 2192-2497

### What the lecturer said - transcript only

Post-training is the final shaping step before deployment, so annotators have substantial influence over model behavior. In earlier work, the lecturer and colleagues compared language models' answers to public-opinion questions with demographic groups. Base models were closer to Protestant and Roman Catholic response patterns than to Buddhist or Hindu patterns. After instruction tuning, several systems shifted away from the former and toward Buddhist, Hindu, and atheist patterns. The InstructGPT appendix described an annotator population including many people from Southeast Asia and the United States West Coast, which the researchers saw as a plausible connection.

The lecturer cautions that measurements of model political opinion can be fragile. The broader point is that annotator selection can transmit values and preferences.

Even data that appears innocuous can carry hidden signals. Research on emergent misalignment or subliminal transfer shows that outputs generated by a model with a preference such as liking owls can cause a student trained on those outputs to inherit that preference, even when the visible data does not overtly express it. Synthetic data can therefore transmit properties that simple filtering misses.

Expertise changes error detection as well as ideology. Work by Hosking and collaborators compared highly engaged expert annotators with ordinary crowdworkers. Non-experts overemphasized surface formatting, while experts paid more attention to factuality and inconsistency, which require more effort and domain knowledge.

Asked what makes one annotator higher quality, the lecturer gives two partial answers:

1. A detailed guideline can define semi-objective criteria. For factuality, a procedure might require checking whether credible search results contradict the answer.
2. Inter-annotator agreement measures variance across a group.

Neither is a gold standard. Agreement does not expose shared bias; subjective preference naturally has high variance; factual questions should agree more; and a group of workers all using ChatGPT may agree perfectly while violating the intended process.

Asked why companies recruit experts, the lecturer says the primary reason is that certain tasks genuinely require them: a lawyer may be needed to check Bluebook citations, and specialized tacit knowledge may be impossible for a generalist to supply. General annotation is also moving toward verified workers because otherwise people can use the cheapest model to generate plausible submissions.

Asked whether domain-specific models should replace annotators, the lecturer separates model annotation from domain specialization. Strong model-based annotation is generally excellent for catching up to a frontier system and can beat random crowdworkers. A domain model has no automatic advantage if it is weaker than the strongest general model available.

### Source reconciliation

The demographic slide attributes the opinion-comparison work to **Santurkar et al. (2023)**. The expertise slide attributes the error analysis to **Hosking, Blunsom, and Bartolo (2024)** and makes the directional result precise: crowdsourced annotations underestimate inconsistency and factuality errors while showing relatively more formatting emphasis.

### Additional explanation

Annotator quality has at least three axes:

- **validity:** does the label measure the intended property?
- **reliability:** would repeated qualified raters reach similar judgments?
- **representativeness:** whose preferences and risk tolerances define the target?

High agreement can coexist with low validity and poor representativeness. A robust program combines calibrated test items, adjudication, rationale review, demographic analysis, expertise verification, and explicit treatment of legitimate disagreement.

## 19. Model-generated feedback, distillation, self-training, and judge bias

**Transcript coverage:** lines 2498-2731

### What the lecturer said - transcript only

Language models are surprisingly effective annotators. In comparisons conducted around GPT-4's release, GPT-4 judgments produced strong system rankings, agreed reasonably with the human majority, and cost roughly an order of magnitude less than carefully curated human annotation.

At the time, it was unclear whether open projects would continue investing in human collection or shift to distillation. The lecturer now sees the answer as settled for catching up to the frontier: human-only data is not competitive on cost and speed. Hugging Face's Zephyr project initially tried to avoid model distillation and purchased human data from the same kinds of vendors used by frontier labs. The process was slow and expensive, and the results were not better than model feedback, so the project ultimately used AI-generated preferences.

UltraChat and UltraFeedback became standard synthetic sources for SFT and RLHF. Tulu 3 uses model-based annotation throughout its open pipeline. Strong models follow detailed collection instructions consistently and scale to many targeted prompts.

This conclusion has an important boundary. Distillation is excellent for matching an existing frontier, but it cannot by itself push beyond the strongest available teacher. Frontier capability still relies on new human expertise and world knowledge in areas such as law or science.

The lecturer responds to a question about the 7B-scale Zephyr experiment. Seven billion parameters may seem small now, but it was a respected open-model size at the time. The lecturer doubts that moving to a larger student would suddenly reverse the human-versus-model annotation result.

Model generation can also be self-training rather than teacher distillation. Anthropic's Constitutional AI prompted a model to produce critiques and safer revisions, then trained on the resulting data. Self-Instruct similarly bootstrapped capability-oriented instructions. These methods can extract behavior from a model, but they cannot conjure specialist world knowledge absent from the process.

Models reproduce evaluator biases. Several studies show that simply making answers longer improves model-judged win rates. One can optimize length alone with RLHF and perform well on preference benchmarks without a comparable capability gain. ChatGPT 3.5 appears as an outlier in one analysis because its success was not explained only by length.

### Source reconciliation

The GPT-4 judge slide reports a **Spearman system-rank correlation of 0.98** and $R^2=0.87$, with agreement near human inter-annotator levels. The length-effects slide shows an illustrative SFT answer of 59 tokens versus an RLHF answer of 243 tokens with similar substance. These exact values support, but are not enumerated in, the spoken account.

### Additional explanation

Using a model judge creates a scalable measurement instrument, not ground truth. It can introduce:

- self-preference for its own style or model family;
- position and order effects;
- verbosity and formatting bias;
- susceptibility to persuasive but false reasoning;
- shared blind spots between generator and judge;
- contamination from seeing benchmark-like data during pretraining.

Calibration against expert humans remains necessary, especially when the target is to surpass rather than imitate the judge.

---

# Part IV - PPO and direct preference optimization

## 20. KL-regularized RLHF, policy gradients, TRPO, and PPO

**Transcript coverage:** lines 2732-2873

### What the lecturer said - transcript only

The basic RLHF goal is to maximize the reward obtained by samples from the current policy. For language-model RLHF this resembles a contextual bandit more than rich multi-step reinforcement learning: a prompt defines the context, one response is generated, and a reward is assigned.

InstructGPT's objective combines the learned reward with a KL-divergence term that keeps the optimized policy near the SFT or pretrained reference. Without that constraint, reward maximization can drive the model into degenerate regions. Stiennon et al.'s summarization setup similarly learns a pairwise reward classifier and hill-climbs its score while penalizing policy drift.

The policy-gradient identity converts the gradient of expected reward into a sample expectation: increase the log-probability of sampled sequences in proportion to their reward. Conceptually, this resembles weighted SFT.

Vanilla policy gradients require fresh samples for every update, and language-model generation is expensive. Builders would prefer to reuse rollouts for several optimizer steps. Off-policy reuse becomes unreliable if the updated policy moves too far from the policy that generated the data.

TRPO addresses this with importance weighting and an explicit constraint that keeps the new policy close to the old one. PPO replaces the difficult trust-region constraint with a heuristic clipped objective that discourages excessively large probability-ratio changes. The lecture postpones detailed PPO mechanics to the next class but emphasizes the progression:

```text
policy gradient -> reuse rollouts with a trust region -> PPO clipping
```

### Source reconciliation

The slides provide the exact policy-gradient identity:

$$
\nabla_\theta \mathbb{E}_{z\sim p_\theta}[R(z)]
=
\mathbb{E}_{z\sim p_\theta}
\left[R(z)\nabla_\theta\log p_\theta(z)\right].
$$

They also show that InstructGPT's published combined objective contains not only reward and a KL penalty but an optional pretraining-log-likelihood term. The spoken explanation focuses on reward plus KL. A later slide verifies the TRPO constrained probability-ratio objective and PPO's clipped surrogate.

### Additional explanation

For a prompt $x$, response $y$, learned reward $r_\phi$, reference policy $\pi_{\mathrm{ref}}$, and trainable policy $\pi_\theta$, the core regularized objective is

$$
\max_{\pi_\theta}
\mathbb{E}_{x,y\sim\pi_\theta}
\left[
r_\phi(x,y)
-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right].
$$

The KL term serves both as a quality prior and as an optimization brake. It says that the learned reward must provide enough evidence to justify departing from the reference model.

## 21. Attempts to avoid PPO and the intuition behind DPO

**Transcript coverage:** lines 2874-2972

### What the lecturer said - transcript only

PPO's equations and systems requirements motivated many attempts to reduce preference learning to ordinary supervised training. Several plausible ideas underperformed:

- Prepend a `[GOOD]` token to chosen examples and `[BAD]` to rejected examples, train on both, and condition on `[GOOD]` at inference.
- Fine-tune only on the preferred responses.
- Train a reward model, sample responses, keep its preferred candidates, and fine-tune on those.

The last approach works to some degree, but none matched the desired behavior reliably enough.

Direct preference optimization, or **DPO**, is the successful simplification highlighted in the lecture. It removes a separately trained reward model and removes on-policy rollout optimization. At an intuitive level, DPO raises the log-probability of the chosen response and lowers the log-probability of the rejected response, with a weighting that keeps the update appropriately scaled relative to a reference policy.

The lecturer describes this as positive SFT on good material plus negative SFT on bad material, but stresses that the weighting is what turns the naive idea into a useful algorithm.

### Source reconciliation

The slide of proposed PPO replacements contains one additional option not spoken in this transcript segment: sample **1,024** outputs from the language model and keep the reward model's best candidate. This is a slide-only best-of-$n$ alternative.

### Additional explanation

Training only on winners discards information about why the winner was preferred and how close the loser was. It can also increase all winner probabilities without constraining what happens to rejected alternatives.

DPO uses the log-probability **margin** between the two responses relative to a fixed reference. This creates a direct contrast and an implicit regularizer.

## 22. DPO derivation from the KL-regularized RLHF objective

**Transcript coverage:** lines 2973-3098

### What the lecturer said - transcript only

The derivation starts from expected reward minus KL distance from a reference policy. It makes one strong assumption: optimize over the set of all possible policies rather than directly restricting the policy to a particular neural-network parameterization.

Under that nonparametric assumption, the optimal policy has a closed form. Take the reference policy and exponentially tilt each response probability by its reward. High-reward responses are upweighted exponentially, and low-reward responses are downweighted.

The closed form can be rearranged to express the unknown reward in terms of the optimized policy and reference policy. That **implied reward** is substituted into the ordinary pairwise reward-model likelihood. The result is the DPO objective, which depends only on the trainable policy, reference policy, and chosen-rejected pairs.

The gradient has a simple interpretation for each pair. Increase the likelihood of the winner, decrease the likelihood of the loser, and scale the step by how wrong the policy's current implied preference is. If the policy already strongly separates the pair in the correct direction, the update is small. If it treats them as equal or favors the loser, the update is larger.

### Source reconciliation

The slides provide the exact derivation. For partition function $Z(x)$, the nonparametric optimum is

$$
\pi_r(y\mid x)
=
\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{1}{\beta}r(x,y)\right),
$$

which implies

$$
r(x,y)
=
\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

For chosen response $y_w$ and rejected response $y_l$, the displayed DPO loss is

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right].
$$

### Additional explanation

The derivation works because pairwise comparison cancels the prompt-dependent normalizer:

$$
r(x,y_w)-r(x,y_l)
=
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
$$

The unknown $\beta\log Z(x)$ appears in both rewards and disappears. A Bradley-Terry preference likelihood then turns this reward difference into the logistic DPO loss.

The reference ratios matter. Without them, the optimizer could treat a response that was already extremely likely under the SFT model the same as one requiring a major behavioral change.

## 23. DPO in practice, expert iteration, and fragile PPO comparisons

**Transcript coverage:** lines 3099-3188

### What the lecturer said - transcript only

DPO is operationally much simpler than PPO and works reasonably well. The lecturer argues that PPO-versus-DPO differences often matter less than implementation quality unless one is training a frontier model.

LLaMA's post-training report uses DPO inside an outer loop. The model is supervised-fine-tuned and DPO-trained, then generates new candidates. Rejection sampling chooses stronger candidates for another SFT round, and the process repeats. DPO is therefore one primitive inside a larger expert-iteration pipeline rather than the complete recipe.

Many DPO variants followed. **SimPO** changes the weighting and removes the explicit reference policy. **Length-normalized DPO** divides sequence scores by response length to reduce verbosity exploitation. The lecturer is unconvinced that most variants produce consistently important gains.

Empirical comparisons are highly contingent. AI2 work has reported PPO beating DPO under one setup and, in Tulu 2 work, well-executed DPO beating PPO. These apparently conflicting conclusions show how learning rates, batch sizes, epochs, data, reference models, and evaluation choices dominate small algorithmic differences.

The robust core is simple: move toward chosen responses, move away from rejected responses, and scale those moves sensibly.

### Source reconciliation

The LLaMA slide depicts the full outer loop: collected prompts, $K$ generations per prompt, reward-model rejection sampling, SFT data and an SFT model, DPO training with specialized binary-preference data, then the best model feeding the next round. The variants slide confirms that SimPO removes the reference term and that length-normalized DPO divides log-probability ratios by response length.

### Additional explanation

"Offline" DPO still depends on the policy that generated its comparison data. If the candidate set is weak, narrow, or far from the current policy, DPO cannot discover responses that never appear. Regenerating candidates in an outer loop restores some of the adaptive exploration that a purely static preference dataset lacks.

Algorithm comparisons should match rollout source, preference pairs, compute, hyperparameter search effort, and evaluation controls. Otherwise, "PPO versus DPO" may actually compare two unrelated recipes.

## 24. Reward overoptimization, mode collapse, calibration, and the bridge to RLVR

**Transcript coverage:** lines 3189-3321

### What the lecturer said - transcript only

The first major RLHF hazard is **overoptimization**. Early excitement around InstructGPT raised the possibility that collecting enough thumbs-up and thumbs-down judgments might produce unlimited improvement. In practice, pushing optimization too far overfits the learned reward model. The policy discovers responses that exploit imperfections in the proxy rather than satisfy the intended human objective. The KL regularizer is therefore critical when the optimizer is strong.

The second hazard is **mode collapse**. RLHF policies often become less diverse and concentrate probability on a small set of outputs. This follows from the conceptual shift discussed earlier: the objective no longer fits a naturally diverse response distribution, so one high-reward answer can dominate.

Calibration is another casualty. GPT-4-era reporting showed RLHF models becoming less calibrated, and the lecturer does not believe the problem has been fully solved. Anthropic researchers have argued that some loss of calibration follows naturally from reward optimization, although recalibration may sometimes help.

Entropy and exploration become especially important for RLVR. A reasoning model must sample diverse candidate solutions to discover successful paths on hard problems. A collapsed policy cannot explore enough to improve.

The lecture closes by reiterating that both SFT and RLHF are dominated by difficult data work. PPO is algorithmically and operationally complex. GRPO offers a simpler recent alternative that will appear in the assignment and next lecture. The transition to RLVR is motivated by a question: can we use rewards that resist proxy overoptimization, so that additional compute continues to improve real task performance?

### Source reconciliation

The overoptimization slide distinguishes three evaluator regimes. Optimization eventually hurts performance under human preferences and a noisy model preference, while a cleaner single-prompt GPT-4 proxy continues to track improvement in the plotted range. The mode-collapse slide combines calibration curves and entropy distributions to show both miscalibration and concentration; the transcript discusses the qualitative conclusion rather than the individual plots.

### Additional explanation

Goodhart's law captures the failure:

```text
human intent -> finite preference data -> learned reward proxy
                                      -> strong optimizer finds proxy loopholes
```

Monitoring reference KL is useful but not sufficient. A small policy change can still exploit a reward bug, and a large beneficial change may be unnecessarily suppressed. Reliable training uses held-out human judgments, adversarial evaluation, multiple reward signals, entropy monitoring, and early stopping.

---

# Consolidated takeaways

1. **Post-training extracts utility from pretraining.** A strong base model is essential, but SFT makes its behavior controllable enough for an assistant interface.
2. **Modern post-training is data-secret, not algorithm-secret.** Public methods are understandable; frontier collection guidelines, mixtures, and feedback loops are sparsely documented.
3. **SFT data evolved from tasks to interactions.** FLAN converted benchmarks, Alpaca distilled chat, OpenAssistant crowdsourced richer answers, and newer corpora supervise tools and agent state.
4. **Naturalness matters.** Reformatting an old benchmark does not remove its awkward objectives, reference errors, or narrow response style.
5. **A small number of demonstrations can steer a capable model.** Broad behavior may shift with hundreds of examples, while the policy long tail still requires large, careful collections.
6. **Style and capability are distinct.** Length, bullets, and polished detail can dominate preference scores without improving reasoning or factuality.
7. **SFT can force unsupported knowledge emission.** Citation templates and tail facts may entangle format imitation with hallucination.
8. **Safety is a Pareto problem.** Builders must reduce harmful compliance without making benign users suffer false refusals.
9. **Mid-training blurs stage boundaries.** High-quality and instruction data increasingly enter the decay phase, so a modern "base model" may already be chat-shaped.
10. **Mixtures remain empirical.** Short decay-stage ablations are valuable, but mapping ablation deltas into final weights is still trial-and-error engineering.
11. **Preference and generation are different skills.** People and models can recognize an answer better than one they could produce directly.
12. **Annotators define part of the product.** Their expertise, demographics, incentives, working conditions, and hidden tool use all affect the learned policy.
13. **Model feedback dominates open catch-up recipes.** It is cheap and strong for distillation, but cannot independently create knowledge beyond the teacher frontier.
14. **PPO optimizes a learned reward while constraining policy drift.** Policy gradients, rollout reuse, and clipping make it more complex than SFT.
15. **DPO converts preference learning into a contrastive likelihood objective.** It removes the explicit reward model and on-policy optimizer while preserving a reference-relative preference margin.
16. **No optimizer escapes data quality.** DPO and PPO comparisons are fragile because candidate generation, pairs, hyperparameters, and judges matter at least as much as the named loss.
17. **Reward optimization changes the distribution.** It can overfit the proxy, reduce entropy, collapse modes, and destroy calibration.
18. **RLVR promises a different reward regime.** Verifiable outcomes may support longer optimization, provided the policy retains enough exploration.

# Key equations

## Imitation and optimization

SFT approximately fits a reference conditional distribution:

$$
\hat p(y\mid x)\approx p^*(y\mid x).
$$

RLHF instead seeks a policy with high expected reward:

$$
\max_p\;\mathbb{E}_{y\sim p(\cdot\mid x)}[R(y,x)].
$$

## Pairwise reward-model loss

For preferred item $y_i$ over $y_{1-i}$,

$$
\mathcal L_{\mathrm{RM}}(r_\phi)
=
-\mathbb E_{(x,y_0,y_1,i)\sim\mathcal D}
\left[
\log\sigma\left(r_\phi(x,y_i)-r_\phi(x,y_{1-i})\right)
\right].
$$

## KL-regularized policy reward

$$
R(x,y)
=
r_\phi(x,y)
-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

## Policy-gradient identity

$$
\nabla_\theta\mathbb E_{z\sim p_\theta}[R(z)]
=
\mathbb E_{z\sim p_\theta}
\left[R(z)\nabla_\theta\log p_\theta(z)\right].
$$

## PPO clipped surrogate

Let

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

The standard clipped surrogate maximizes

$$
\mathbb E_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

## DPO objective

$$
\mathcal{L}_{\mathrm{DPO}}
=
-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right].
$$

# Glossary

**Behavioral distillation**
Training a student on demonstrations or preferences generated by a stronger teacher so that the student imitates the teacher's interaction policy.

**Calibration**
Agreement between stated probabilities and empirical correctness frequencies. RLHF can damage this relationship.

**Constitutional AI**
A self-training approach that uses written principles to generate critiques, revisions, preference data, and safer policy updates.

**Decay phase**
The lower-learning-rate tail of pretraining, increasingly used for high-quality, capability-targeted, and instruction-like data.

**DPO**
Direct preference optimization, an offline objective that increases a chosen-versus-rejected log-probability margin relative to a reference policy.

**Expert iteration**
An outer loop that generates candidates, selects or verifies stronger outputs, retrains on them, and repeats.

**False refusal rate**
The frequency with which a safety-tuned model rejects benign requests.

**FLAN**
A family of multitask instruction-tuning mixtures built largely by converting existing supervised NLP datasets into prompted tasks.

**GRPO**
Group relative policy optimization, a PPO-like method that uses relative rewards within sampled groups and is developed further in the next lecture.

**Instruction following**
Producing an output that respects a user's explicit content, format, and behavioral constraints.

**KL regularization**
A penalty that discourages an optimized policy from moving too far from a reference model.

**Length hacking**
Improving an evaluator score by increasing response length rather than improving substantive quality.

**Mid-training**
A continued-pretraining phase that changes the data mixture toward higher-quality or specialized domains before final post-training.

**Mode collapse**
Concentration of policy probability on a narrow set of outputs, reducing useful diversity and exploration.

**OpenAssistant**
A crowdsourced project in which volunteers produced prompts, detailed responses, and conversational preference data.

**Pairwise feedback**
A judgment that one response is better than another for the same prompt.

**Policy**
In this setting, the language model's conditional distribution over responses given a prompt.

**Policy gradient**
An estimator that weights gradients of sampled log-probabilities by reward or advantage.

**PPO**
Proximal policy optimization, which reuses rollout data while clipping policy-ratio changes to limit destructive updates.

**Preference model / reward model**
A learned scalar scorer trained from human or model comparisons and used as an optimization proxy.

**RLHF**
Reinforcement learning from human feedback, broadly covering preference-data collection and policy optimization against human-derived reward signals.

**RLVR**
Reinforcement learning with verifiable rewards, in which correctness can be checked programmatically or otherwise objectively.

**Self-Instruct**
A method for bootstrapping instruction data from a language model's own generated tasks and responses.

**SFT**
Supervised fine-tuning on demonstration sequences, usually prompt-response or conversational examples.

**Tail knowledge**
Informal term for rare or weakly represented information that a model may not reliably store or retrieve.

**TRPO**
Trust-region policy optimization, which constrains policy updates using a divergence from the behavior policy.

**Violation rate**
The frequency with which a safety-tuned model complies with requests that should be refused or constrained.

**Weak evaluator / proxy reward**
A measurable score that imperfectly represents the actual human objective and can be exploited by optimization.

# Self-check questions

1. Why can a very strong base model still be much less useful than a post-trained assistant?
2. What information about post-training is public, and what tends to remain secret?
3. How do the demonstration and preference phases of classical RLHF differ?
4. Why did FLAN inherit unnatural tasks and low-quality targets from older NLP datasets?
5. What did Alpaca and OpenAssistant reveal about synthetic versus human SFT data?
6. Which separate decisions must an agentic tool-use example supervise?
7. Why can preference win rate improve while capability benchmarks remain flat?
8. How can a detailed citation target increase hallucination?
9. Under what assumptions might policy-dependent RL feedback improve calibration?
10. What are violation rate and false refusal rate, and why do they form a tradeoff?
11. Why can 500 safety examples have a large effect yet still be insufficient for a deployed product?
12. In what sense does mid-training blur the definition of a base model?
13. Why are decay-stage mixture ablations useful, and why are their conclusions still brittle?
14. Explain the conceptual difference between fitting a response distribution and maximizing reward.
15. What dimensions did InstructGPT ask preference annotators to balance?
16. Why is inter-annotator agreement not a complete quality metric?
17. When is model-generated feedback especially effective, and when does it reach a ceiling?
18. Why does PPO include a reference-policy or trust-region constraint?
19. Which information is lost by training only on preferred responses?
20. What nonparametric assumption makes the DPO closed-form derivation possible?
21. Why does the DPO pairwise loss not need the partition function $Z(x)$?
22. How can an expert-iteration outer loop compensate for static DPO data?
23. What are reward overoptimization and mode collapse?
24. Why will exploration and entropy matter even more for RLVR reasoning models?

# Source coverage checklist

- [x] **Lines 1-239:** GPT-3 to ChatGPT motivation, instruction control, pretraining dependence, post-training messiness, public-information limits, leaked Scale AI anecdote, and data as the secret ingredient.
- [x] **Lines 240-320:** demonstration-plus-RLHF recipe, SFT focus, historical plan, and data-collection pitfalls.
- [x] **Lines 321-460:** FLAN, Self-Instruct, Alpaca, Vicuna, OpenAssistant, WizardLM, Tulu 3, tool-use data, closed-lab caveat, and correctness question.
- [x] **Lines 461-589:** FLAN examples, Enron subject prediction, summarization defects, inherited unnaturalness, and early scale assumptions.
- [x] **Lines 590-681:** Alpaca distillation, LLaMA dependence, ChatGPT-style behavior, open-data optimism, and OpenAssistant crowdsourcing.
- [x] **Lines 682-764:** agentic applications, Nemotron structured calls, chattiness, detail, expert responses, and tool interfaces.
- [x] **Lines 765-887:** collection dimensions, response-length variation, list and verbosity preferences, engagement confounds, and style-versus-capability separation.
- [x] **Lines 888-1049:** OpenAssistant citation example, simultaneous knowledge and behavior learning, unknown-fact hallucination, Schulman's RL argument, and tail-knowledge caution.
- [x] **Lines 1050-1152:** tail-knowledge definition question, RL folk mechanism, limits of internal calibration, and SFT reference-template generalization.
- [x] **Lines 1153-1249:** deployed misuse, safety SFT opacity, violation and false-refusal tradeoff, Llama 2, and example scale.
- [x] **Lines 1250-1313:** AI2/Tulu public pipeline, WildChat, unsafe behavior and jailbreak mining, preferred refusals, and production whack-a-mole.
- [x] **Lines 1314-1472:** 500-example steering, fine-grained long tail, SFT takeaways, latent-behavior question, and feedback-based SFT/RL distinction.
- [x] **Lines 1473-1586:** SFT gradient descent, instruction data in pretraining, two-phase training, base-model terminology, MiniCPM mix, prompt masking question.
- [x] **Lines 1587-1736:** decay data quality, mixture trial and error, cheap mid-training ablations, reflection into pretraining, token scarcity, and Meta book-subset example.
- [x] **Lines 1737-1929:** shift from imitation to reward, possible distribution collapse, generation-verification gap, writer study, math verification, and RLHF lecture plan.
- [x] **Lines 1930-2049:** sampling and ranking loop, reward model, pairwise interface, InstructGPT helpful/truthful/harmless rubric, and Bard guidelines.
- [x] **Lines 2050-2191:** workforce education and age, expert growth and pay, worker verification, AI use, time pressure, outsourcing, and labor bifurcation.
- [x] **Lines 2192-2497:** annotator demographics, opinion shifts, subliminal transfer, expert versus crowdworker errors, quality and agreement questions, expert need, and model-annotator question.
- [x] **Lines 2498-2731:** GPT-4 judging, cost, Zephyr human-data attempt, UltraChat/UltraFeedback/Tulu 3, frontier limit, 7B question, Constitutional AI, Self-Instruct, and length bias.
- [x] **Lines 2732-2873:** contextual-bandit objective, reward plus KL, Stiennon reward model, policy gradient, rollout cost, off-policy reuse, TRPO, and PPO clipping.
- [x] **Lines 2874-2972:** failed control-token/preferred-only/rejection approaches and DPO's positive-versus-negative gradient intuition.
- [x] **Lines 2973-3098:** KL-regularized objective, nonparametric optimum, exponential reward tilt, implied reward, DPO loss, and gradient interpretation.
- [x] **Lines 3099-3188:** DPO practicality, LLaMA outer loop, SimPO, length-normalized DPO, PPO comparisons, experiment fragility, and common update principle.
- [x] **Lines 3189-3321:** overoptimization, KL importance, diversity and mode collapse, calibration, entropy, RLVR exploration, data difficulty, GRPO, assignment, and transition to verifiable rewards.

All 3,321 transcript lines and all 65 slides were covered.
