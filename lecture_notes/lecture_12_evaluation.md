---
title: "Lecture 12 - Evaluation"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 12
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 12 Evaluation.txt"
slide_deck: "../lecture_12.pdf"
status: "complete"
---

# Lecture 12: Evaluation

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves claims, examples, numerical details, cautions, and substantive audience questions and answers while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, derivation, or study guidance. It is not presented as something the lecturer said.

The raw transcript's physical line spans are shown so that the paraphrase can be audited. The complete transcript was mapped before the slides were inspected. All 54 slides were then rendered and visually checked for names, notation, formulas, charts, benchmark construction, and numerical labels. Slide-only details and material discrepancies appear under **Source reconciliation** rather than being silently imported into the transcript-grounded account.

## Lecture map

The lecture treats evaluation as the act of turning a desired but abstract model property into an operational measurement. It has five main parts:

1. Define competing meanings of "good" and study perplexity as the intrinsic metric closest to the language-model objective.
2. Follow the progression from exam benchmarks to evaluations of open-ended chat.
3. Extend evaluation from what models say to what agents do, then examine pure-reasoning and safety benchmarks.
4. Ask whether an evaluation is realistic, uncontaminated, and built from valid tasks and graders.
5. Match the evaluation to its purpose and state clearly whether the object being compared is a method, model, system, or agent.

---

# Part I - What does it mean for a model to be good?

## 1. Evaluation turns an abstract construct into a concrete metric

**Transcript coverage:** lines 1-174

### What the lecturer said - transcript only

The course has already covered the architecture, optimizer, training loop, kernels, parallelism, scaling laws, and some inference optimization needed to train a language model. The remaining major ingredient is training data. Data shapes behavior: code data should improve coding, while a model trained only on DNA sequences should not be expected to speak English. Before choosing data, however, a model builder must decide which behaviors are desired. That is the role of evaluation.

The apparent question is simple: given a trained model, how good is it? A mechanical description would define prompts, obtain model responses, and compute an accuracy. The topic deserves much more care because evaluations become development North Stars. Open and closed model developers use them to measure progress, and the chosen metric implicitly directs what capabilities receive attention.

The core difficulty is moving from an **abstract construct** such as conversation quality or reasoning ability to a **concrete metric** supported by prompts, answers, judges, or environments. Several plausible meanings of "good" illustrate why there is no automatic answer:

- A model may rank highly on a benchmark aggregation such as Artificial Analysis.
- It may provide strong benchmark performance at low inference cost. Quality and price are correlated, but not perfectly aligned.
- People may prefer its responses in Arena AI, formerly Chatbot Arena.
- People may choose to use and pay for it. OpenRouter usage statistics provide one economic lens, although that service is not representative of all model use.

None of these definitions is declared correct. Their purpose is to make the choice of evaluative objective explicit.

### Additional explanation

The abstract-to-concrete step is a problem of **construct validity**. A metric is not the construct itself; it is evidence about the construct. For example, multiple-choice accuracy may provide evidence about factual knowledge and constrained reasoning, but it does not define intelligence.

An evaluation pipeline can be decomposed into:

$$
\text{construct}
\rightarrow \text{task distribution}
\rightarrow \text{model interaction}
\rightarrow \text{scoring rule}
\rightarrow \text{reported statistic}.
$$

Every arrow introduces assumptions. A benchmark can fail even when its arithmetic is flawless: prompts may not represent the intended use, the interaction protocol may handicap some systems, or the statistic may collapse important capability differences.

Cost, preference, usage, and benchmark accuracy also answer different questions. A practical model decision is often multi-objective:

| Dimension | Example question |
|---|---|
| Capability | Can the system complete the target task? |
| Quality | How correct, useful, and clear is the output? |
| Cost | What does each successful task or token cost? |
| Latency | Is the response fast enough for the workflow? |
| Preference | Which response do intended users choose? |
| Risk | What failures are unacceptable even if average quality is high? |

A single ranking hides how those dimensions were weighted.

## 2. Perplexity from in-distribution testing to GPT-2 zero-shot transfer

**Transcript coverage:** lines 175-316

### What the lecturer said - transcript only

A language model defines a probability distribution $p(x)$ over token sequences. The most natural intrinsic evaluation asks how much probability the model assigns to a test dataset and normalizes that probability by dataset length. Likelihood, log loss, and perplexity are closely related views of this question.

Traditional language-model research minimized perplexity on a training split and reported perplexity on a held-out split from the same dataset. Common benchmarks included Penn Treebank, WikiText-103, and the One Billion Word Benchmark. A notable 2016 result showed a large perplexity reduction from pure CNN and LSTM models on the One Billion Word Benchmark, helping settle the then-open comparison with n-gram and hybrid approaches. In that paradigm, training and evaluation were both in distribution and progress meant reducing test perplexity.

GPT-2 changed the evaluation setup. It was trained on WebText, about 40 GB of websites linked from Reddit, and evaluated zero-shot on standard language-model datasets rather than training separately on each benchmark. This was out-of-distribution evaluation. Its 1.5B-parameter model reached about 35 perplexity on the small Penn Treebank benchmark against a previous state of the art near 46. It did not beat a strong in-distribution model when the target dataset was large, such as One Billion Word, but transfer to smaller datasets was impressive.

The lecturer notes uncertainty about whether WebText overlapped any evaluation sources because careful train-test decontamination was not known. Even with that caveat, GPT-2 helped establish the now-common paradigm of broad pretraining followed by evaluation on many standard benchmarks without benchmark-specific training.

### Source reconciliation

The slides state the perplexity definition as

$$
\operatorname{PPL}(D)=\left(\frac{1}{p(D)}\right)^{1/|D|}.
$$

They also identify Penn Treebank as Wall Street Journal text, WikiText-103 as Wikipedia, and the One Billion Word Benchmark as material derived from WMT11 sources including Europarl, United Nations text, and news. The displayed 2016 comparison reduces One Billion Word perplexity from 51.3 to 30.0. These exact labels and values support, but are more detailed than, the spoken account.

### Additional explanation

For tokens $x_1,\ldots,x_N$, the usual autoregressive definition is:

$$
\operatorname{PPL}(x_{1:N})
=\exp\left(
-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i\mid x_{<i})
\right).
$$

Perplexity is the exponential of average negative log-likelihood. Lower is better. In an informal interpretation, perplexity behaves like the model's effective average number of plausible next-token choices, although that interpretation is exact only in special cases.

Perplexity comparisons require a common tokenization and evaluation protocol. Changing the tokenizer changes $N$ and the events whose probabilities are scored, so token-level perplexities from two unrelated tokenizers are not directly comparable. Byte- or character-normalized log loss is safer when tokenizations differ.

The GPT-2 shift also separates two questions:

- **In-distribution modeling:** How well does the model fit held-out data drawn from the same source as training?
- **Transfer:** How well does broad pretraining assign probability to a distinct target distribution without target-specific fitting?

The second is closer to the modern foundation-model setting, but it introduces contamination and distribution-mismatch questions that the old fixed-split setup largely avoided.

## 3. "Perplexity is all you need" - and perhaps more than you need

**Transcript coverage:** lines 317-423

### What the lecturer said - transcript only

The lecturer presents "perplexity is all you need" as a mindset driven more by faith than by settled science. Suppose there is a true distribution $t$ and the model distribution is $p$. He argues that the best achievable value occurs uniquely when $p=t$. If the model learns the true distribution, then any task can be expressed by conditioning: condition on a problem and generate a solution, or condition on a question and generate its answer. Under this view, continually lowering perplexity eventually yields all desired capabilities.

This belief helped motivate language-model scaling before the practical gains were obvious. Before GPT-3, it was not clear that scaled language models would have such broad impact, but some researchers trusted that lower perplexity would eventually produce useful behavior.

Perplexity may also measure **more** than a developer needs. In the sentence "Stanford was founded in 1885," predicting `1885` tests useful world knowledge and resembles question answering. Perplexity also penalizes less interesting choices, such as the sentence's first word or a routine word such as `founded`. It charges for every deviation from the data distribution rather than only for the capability-relevant token.

Conditional perplexity partially addresses this. A benchmark can provide a prompt and score only the response tokens, focusing the metric on a selected conditional behavior while ignoring incidental prompt tokens.

### Source reconciliation

Both the transcript and slide say that the "best possible perplexity is $H(t)$." Strictly, the **minimum cross-entropy or average negative log-likelihood** is the entropy $H(t)$, while the minimum perplexity is its exponential, $\exp(H(t))$ for natural-log entropy or $2^{H_2(t)}$ for entropy in bits. The claim that $p=t$ is the unique optimum, up to zero-probability events, is the substantive point.

The conditional-perplexity slide writes $p(\text{response}\mid\text{prompt})^{1/|\text{response}|}$ without the reciprocal shown in its earlier perplexity definition. The conventional conditional perplexity uses the negative exponent:

$$
p(\text{response}\mid\text{prompt})^{-1/|\text{response}|}.
$$

### Additional explanation

The population cross-entropy decomposes as:

$$
H(t,p)
=\mathbb{E}_{x\sim t}[-\log p(x)]
=H(t)+D_{\mathrm{KL}}(t\Vert p).
$$

Since KL divergence is nonnegative, minimizing expected log loss recovers $p=t$ when the model class, data, and optimization permit it. This is why log loss is called a **proper scoring rule**.

The theoretical argument does not by itself settle practical evaluation. Real models have finite capacity, finite data, imperfect optimization, and a training distribution that may differ from the desired deployment distribution. A small improvement in average web-text loss can come from modeling frequent stylistic details rather than improving a rare, high-value capability.

Conditional evaluation makes the allocation explicit. For prompt $q$ and response tokens $r_{1:M}$:

$$
\operatorname{PPL}(r\mid q)
=\exp\left(
-\frac{1}{M}\sum_{j=1}^{M}\log p(r_j\mid q,r_{<j})
\right).
$$

It still depends on the chosen response distribution, but it no longer spends score on predicting the prompt itself.

## 4. Perplexity in disguise and the difficulty of verifying probability submissions

**Transcript coverage:** lines 424-564

### What the lecturer said - transcript only

Several apparent downstream benchmarks are sharpened forms of next-token evaluation. LAMBADA is a cloze task in which a model fills the final missing word of a passage. The spoken illustration ends with a fragment like "Do you honestly think I will want you to have a ____?" The examples are selected so that resolving the answer requires substantial preceding context. Early GPT work valued this structure because long-distance dependencies were thought to be important for reasoning and other capabilities. The benchmark reports accuracy, but it focuses language-model probability on positions chosen to test a particular phenomenon.

HellaSwag is multiple-choice sentence completion. The lecture's example describes a woman outside with a bucket and a dog, then asks which sentence best completes "She ____." It is not full-distribution perplexity, but it is closely related: candidate continuations are compared according to how well they complete a context.

Perplexity also creates an unusual leaderboard contract. A participant may receive test data and return log probabilities. The evaluator must trust that those numbers come from a valid normalized distribution. A dishonest system could return probability 1, or log probability 0, for every submitted sequence and appear perfect even though no such language-model distribution exists. Verifying the implementation or code may therefore be necessary.

Response-based downstream evaluation is easier to audit through a black-box interface: the evaluator sends a prompt, receives an answer, and scores that answer. Distributional evaluation is fundamentally different because the submitted probabilities carry mathematical validity requirements. For latent-variable models such as VAEs, an evaluator may receive only a likelihood bound rather than an exact probability and must also trust that the bound was computed correctly.

### Source reconciliation

The HellaSwag slide shows that its candidate continuations were created with adversarial filtering, a construction detail not stated in the transcript. The slide's leaderboard warning summarizes the required contract as trusting that submitted probabilities are valid and sum to 1.

### Additional explanation

Multiple-choice scoring can be written as:

$$
\hat{k}=\arg\max_k\log p(c_k\mid q),
$$

where $q$ is the context and $c_k$ is candidate continuation $k$. This looks like conditional language modeling, but implementation choices matter. Longer candidates accumulate more negative log probability, so length normalization, answer formatting, and whether the score includes shared prefixes can change results.

The leaderboard warning distinguishes **verifiable outputs** from **unverifiable internal claims**. A generated answer can be checked directly against a task. A claimed probability distribution cannot generally be validated from a small set of queried sequences because normalization ranges over all possible sequences. A rigorous probability leaderboard may therefore need trusted execution, submitted code, cryptographic or systems controls, or a restricted model interface.

## 5. What perplexity is good for

**Transcript coverage:** lines 565-591

### What the lecturer said - transcript only

Perplexity remains heavily used during language-model development. It varies smoothly with scale, which makes it especially useful for fitting scaling laws. Nevertheless, developers still need benchmarks that capture real-world situations, particularly to persuade people who do not accept low perplexity as sufficient evidence of a good model. The lecturer jokes that a believer may be convinced by an exceptionally low perplexity alone, then closes the section without a recorded substantive question.

### Additional explanation

Perplexity is strongest as an internal diagnostic when:

- the tokenizer and held-out distribution are fixed;
- small differences need to be measured with low variance;
- models at several scales must be placed on one smooth curve;
- one wants a task-independent signal before generation behavior is polished.

It is weaker as the sole external evaluation because users care about conditional actions, factuality, preference, latency, safety, and task completion under deployment conditions. A useful evaluation suite often combines one stable intrinsic metric with targeted behavioral and systems metrics.

---

# Part II - Exams and open-ended chat

## 6. MMLU and the move to harder exam benchmarks

**Transcript coverage:** lines 592-747

### What the lecturer said - transcript only

Exams are a familiar way to test humans and have several convenient properties for language-model evaluation. Their subject matter and difficulty can be controlled, questions can be designed to have unambiguous correct answers, and grading is straightforward. Much of early language-model benchmarking therefore adopted exam-style tasks.

Massive Multitask Language Understanding, or MMLU, was influential because it anticipated language models as general-purpose task solvers around the release of GPT-3. Students assembled questions from online sources across 57 subjects, including mathematics, history, law, and morality. Despite "language understanding" in its name, the benchmark mainly tests knowledge and reasoning rather than fluency alone.

GPT-3 was evaluated with few-shot prompts: a subject description, several question-answer examples, and then a final multiple-choice question. This looks ordinary now, but expecting a language model to follow such a constructed prompt was unusual at the time. Small models performed near chance, while the largest GPT-3 model was clearly above chance. Over the following years, MMLU rose into the 90% range and became saturated for frontier models, although it remained useful for smaller-model development and scaling-law work.

Around 2024, a harder successor removed noisy and trivial MMLU questions, expanded each item from four answer choices to ten, and evaluated models with chain-of-thought reasoning. The changes initially reduced performance substantially, but the lecturer reports that scores have again climbed to roughly 88%-90%. The familiar pattern is that a benchmark becomes too easy, is repaired or replaced, and then begins saturating again.

### Source reconciliation

The transcript refers to the harder successor without clearly preserving its name. The slides identify it as **MMLU-Pro**. They report model-accuracy drops ranging from 16% to 33%, rather than saying that every model's absolute accuracy became exactly 33%. The slides also confirm 57 MMLU subjects and describe MMLU-Pro's chain-of-thought protocol.

### Additional explanation

Exam benchmarks offer strong **scoring reliability** but only partial **construct coverage**. A four-choice item has a 25% chance baseline, while a ten-choice item has a 10% baseline. Increasing the number of choices can reduce lucky guessing, but it also changes the distractor-writing problem and may alter what the task measures.

Prompting protocol is part of the benchmark. Few-shot examples, chain-of-thought instructions, answer order, sampling temperature, and the answer parser can all affect accuracy. A result is reproducible only when those details are fixed alongside the question set.

Saturation does not make a benchmark useless in every regime. It means the benchmark has little resolution among systems near its ceiling. The same questions can remain informative for smaller models, ablations, or scaling curves whose performance lies well below that ceiling.

## 7. GPQA, expert difficulty, and subtle contamination

**Transcript coverage:** lines 748-872

### What the lecturer said - transcript only

GPQA was designed to be "Google-proof": if a non-expert could answer a question easily with a web search, then a model trained on the internet might find it too easy as well. The benchmark moved toward explicitly PhD-level questions and required considerable human labor. Sixty-one PhD contractors hired through Upwork wrote questions. Each question passed through expert validation and feedback, author revision, another expert review, and tests with non-experts who had Google access.

The DIAMOND subset contains especially well-vetted questions. The lecturer describes inclusion as requiring agreement from both expert validators and at least one correct answer among the non-experts. PhD experts achieved about 65% accuracy. Non-experts with 30 minutes and Google reached about 34%, only modestly above the four-way random baseline. GPT-4 initially achieved about 39%. The lecturer reports that current frontier results are around 94%, illustrating the short shelf life even of a benchmark built to be extremely difficult.

An audience member asks how anyone knows the questions were not used for training. The lecturer's short answer is that outsiders generally do not know what is in a closed model's training set. Contamination is subtler than literally copying the benchmark file: questions may be derived from books, papers, or websites that were themselves included in training. Benchmark numbers should therefore be treated with skepticism, and the lecture returns to mitigation strategies later.

### Source reconciliation

The slides expand GPQA as **Graduate-Level Google-Proof Q&A**. They confirm 61 PhD contractors, expert accuracy of 65%, non-expert accuracy of 34% with 30 minutes and Google, and the original GPT-4 result of 39%.

### Additional explanation

Contamination can occur at several levels:

| Level | Example |
|---|---|
| Exact | The test question and answer appear verbatim in training. |
| Near-duplicate | A reformatted or lightly edited version appears. |
| Source | The paper, textbook, or web page from which the item was derived appears. |
| Procedural | Many closely related templates or solutions teach the benchmark's construction pattern. |

The last two are difficult to classify. Learning a subject from the underlying literature is legitimate generalization, while memorizing an item's exact derivation undermines its intended novelty. This is why a single binary "contaminated or clean" label often hides a continuum.

Expert accuracy is also not an absolute human ceiling. It depends on whether the evaluator matches the subfield, how much time and tooling are allowed, and whether disagreement reflects difficulty or a defective question. The multi-stage validation process is valuable because it separates hard questions from merely unclear ones.

## 8. Humanity's Last Exam and the limits of multiple choice

**Transcript coverage:** lines 873-1015

### What the lecturer said - transcript only

Humanity's Last Exam, introduced the previous year, made another deliberate attempt to create questions that frontier models could not solve. It spans many subjects, includes multimodal content, and combines multiple-choice with some short-answer questions. Contributions were crowdsourced with monetary rewards for some authors and coauthorship for others, then passed through multiple review stages and filtering by frontier models.

The benchmark also retained a private set that was not publicly released to reduce the chance of training contamination. This still requires trust that prompts sent to model APIs do not later enter training data. The lecturer observes a recurring dataset-paper pattern: prior benchmarks are shown with high model scores, while the new benchmark is shown with very low scores. Humanity's Last Exam initially produced single-digit results. At lecture time, even the displayed Mythos system was only at about 64.7%, so the benchmark still had room.

Exam benchmarks have trended toward harder questions as older sets saturate. Multiple choice itself can be arbitrarily difficult; adding answer choices does not prevent a question from requiring deep expertise. Its main limitation is not difficulty but expressiveness. It restricts the kinds of questions that can be asked. Real users usually submit open-ended, ambiguous, or imperfectly formed requests that may not have one correct answer, and Humanity's Last Exam is rarely used outside the act of benchmarking it.

In Q&A, a student asks how model output is compared with ground truth. For multiple choice, the generated answer letter is checked against the correct letter. A model may directly sample a letter or generate a chain of thought, explanation, and final answer. In the latter case, extracting the answer matters, and evaluation can be sensitive to that parser even though the lecture does not explore the issue further.

### Source reconciliation

The slides specify **2,500 questions**, a **500,000 US dollar prize pool plus coauthorship**, and both public and private HLE sets. A construction diagram shows roughly 70,000 launch attempts, 13,000 submissions, 6,000 candidates after expert organization and approval, and 2,500 final questions. These counts are slide-only details and are not presented as spoken claims above.

### Additional explanation

The benchmark-replacement cycle is an evaluation arms race:

1. A benchmark separates current systems.
2. Researchers optimize models, prompts, and scaffolds against it.
3. Scores rise and variance near the ceiling collapses.
4. Audits reveal noise or shortcuts.
5. A harder or cleaner successor restores discrimination.

This cycle measures progress but also changes the target. Comparing percentages across MMLU, MMLU-Pro, GPQA, and HLE does not form one continuous scale because the item populations and protocols differ.

Answer extraction is part of measurement error. If a correct explanation ends with an unexpected format, a brittle parser can score it wrong; a permissive parser may accidentally infer a correct answer from contradictory text. Robust evaluations define an output schema, validate it, and report parser failures separately from substantive errors.

## 9. Chatbot Arena and Elo-style preference rankings

**Transcript coverage:** lines 1016-1225

### What the lecturer said - transcript only

Open-ended chat responses cannot usually be evaluated by exact match because there may be no single ground-truth answer. The lecture's motivating example asks which herbs work well or poorly in a beet and cheese salad. Many different responses could be useful.

Chatbot Arena addressed this by collecting pairwise human preferences. A person on the internet enters a prompt and receives answers from two randomly selected, anonymized models. The person can say that A is better, B is better, both are good, or both are bad. The accumulated comparisons are fit with an Elo-style model: a higher rating for A relative to B implies a higher probability that A wins. Fitting the ratings to the observed pairwise outcomes yields a model ranking.

This setup has several attractions. Users receive free model access, giving them some incentive to enter prompts they genuinely care about. Sparse comparisons are sufficient; every model need not answer every prompt or face every other model as long as the comparison graph remains connected. New prompts and new models can enter continuously, making the ranking dynamic.

The weaknesses are equally important. The population is an uncontrolled subset of internet users, and demographics alone do not fully characterize its tasks or motivations. Bias, spam, or deliberate leaderboard gaming may enter. A binary preference conflates style and correctness. The prompt author understands personal intent, but may have asked the question precisely because they do not know the correct answer. Pleasing or sycophantic responses can therefore beat honest but less flattering ones.

### Source reconciliation

The transcript leaves the exact rating equation partially inaudible. The slide gives:

$$
P(A\text{ beats }B)
=\frac{1}{1+10^{(E_B-E_A)/400}},
$$

where $E_A$ and $E_B$ are the two Elo ratings. The ratings are fit to maximize the likelihood of the observed pairwise comparisons.

### Additional explanation

Elo here is a pairwise probabilistic model closely related to Bradley-Terry ranking. Only rating **differences** determine probabilities, so adding a constant to every rating changes nothing. An implementation must fix an origin or normalization.

The connected-graph requirement matters. If one group of models is never compared, directly or indirectly, with another group, their relative rating offset is unidentified. Even in a connected graph, sparse or unbalanced matchups create uncertainty that a point estimate alone conceals.

Arena results estimate preference under a joint distribution of:

$$
\text{users}\times\text{prompts}\times\text{model pairs}\times\text{presentation protocol}.
$$

A product with a different user population or task mix may have a different ranking. Preference is not a context-free property of a model.

## 10. AlpacaEval, LLM judges, length bias, and WildBench rubrics

**Transcript coverage:** lines 1226-1364

### What the lecturer said - transcript only

AlpacaEval evaluates a model's win rate against a baseline on a fixed instruction set. In its initial form, GPT-4 Preview served both as the baseline response generator and as the judge deciding which answer was better. This raises possible self- or family-bias concerns. Using multiple judges and ensembling can reduce reliance on one judge.

The first version exposed a concrete judge bias: language-model judges favored longer responses. Fine-tuned models exploited this and rose on the leaderboard by producing increasingly verbose answers. AlpacaEval 2.0 applied a simple regression-based correction to debias the metric.

This leads to a meta-evaluation question: how does one know whether an evaluation metric is good? There is no definitive answer. A sanity check is correlation with another metric. AlpacaEval 2.0 had a Spearman correlation of about 0.98 with human Chatbot Arena rankings on the models then tested. That makes it a useful proxy when a developer cannot or does not want to wait for Arena votes. The correlation is conditional on that model population and need not extrapolate to models stronger or qualitatively different from the GPT-4-era systems used in the comparison. The leaderboard itself had not been maintained recently.

WildBench combines real human-chatbot conversations with an LLM judge. Its main addition is a prompt-specific checklist or rubric. Asking a judge whether an answer is simply "good" leaves the task ill-defined; a checklist scopes the criteria and makes judging more reliable. WildBench also correlated well with Chatbot Arena, although treating Arena as ground truth is somewhat circular.

### Source reconciliation

The slides specify **805 AlpacaEval instructions**. They identify GPT-4 Preview as both baseline and judge in the original setup. For WildBench, the slides specify **1,024 examples sampled from one million human-chatbot conversations** and a GPT-4 Turbo judge supplied with a checklist. These exact counts and model labels are not all stated in the transcript.

### Additional explanation

LLM judges can introduce several systematic effects:

- verbosity preference;
- position bias toward the first or second answer;
- self-preference for outputs resembling the judge's family or style;
- sensitivity to formatting and confident language;
- failure to verify specialist factual claims;
- prompt-injection or rubric manipulation inside the candidate response.

Controls include swapping answer order, hiding model identity, length-controlled analysis, using multiple diverse judges, adding domain-specific verifiers, and auditing disagreements with humans. Regression can remove an observed correlation with length, but it cannot guarantee removal of every latent style bias.

Correlation validates a proxy only over the sampled systems and prompts. A metric can correlate strongly because all systems lie on one familiar capability frontier, then fail on a new model that deliberately optimizes the proxy. Meta-evaluation should therefore include adversarial and out-of-distribution tests, not just one historical correlation coefficient.

## 11. Principles for evaluating open-ended responses

**Transcript coverage:** lines 1365-1422

### What the lecturer said - transcript only

There is no clean solution to open-ended response evaluation, but several practices help. Pairwise comparison is usually higher signal than assigning an absolute score. When two responses are similar, a judge may reliably say that one is slightly better even if deciding whether either deserves 7/10 or 8/10 is arbitrary.

Human and language-model judges have different biases, and both require scrutiny. Agreement across multiple judges, including humans and models, provides stronger evidence, though not certainty. Rubrics and checklists improve reliability by defining what counts as success. This is true for either type of judge: crowdsourcing experience shows that asking a person to rate an answer without criteria often produces incoherent results.

The underlying evaluation problem remains partly ill-defined. A rubric does not discover the desired construct automatically; it forces the evaluator to state it more clearly.

### Additional explanation

A good rubric separates dimensions that should not be collapsed prematurely, for example:

| Dimension | Possible criterion |
|---|---|
| Correctness | Claims are factually and logically supported. |
| Relevance | The response addresses the user's actual request. |
| Completeness | Required parts and constraints are covered. |
| Clarity | The response is understandable without needless complexity. |
| Calibration | Uncertainty and limitations are represented honestly. |
| Safety | The answer avoids defined unacceptable harms. |

Judges can score each dimension before an overall preference. This makes disagreements diagnosable: two judges may agree on correctness but value concision differently. Reliability should be measured with repeated or overlapping judgments, and evaluation reports should distinguish judge disagreement from model failure.

---

# Part III - Evaluating actions, reasoning, and safety

## 12. Agents and SWE-bench: evaluating what systems do

**Transcript coverage:** lines 1423-1499

### What the lecturer said - transcript only

Chat benchmarks evaluate what language models say. Agentic benchmarks evaluate what systems do. An agent is a language model plus a scaffold: the logic that decides when and how to call the model, which tools it may use, and how it iterates over time.

SWE-bench is a canonical coding-agent benchmark. A system receives a codebase and a GitHub issue description, then must produce a pull-request-style patch. Evaluation runs unit tests designed so that relevant tests fail before the patch and pass after a correct repair, while previously working behavior should remain intact. The binary or test-based grader is comparatively straightforward.

The benchmark helped establish a realistic way to measure coding agents in an actual repository environment. The lecturer reports progress on SWE-bench Verified from about 16% in 2024 to about 93% at lecture time, reflecting the enormous investment in coding-agent capability.

### Source reconciliation

The slides spell the benchmark **SWE-bench** and specify **2,294 tasks across 12 Python repositories**. The spoken lecture does not give those counts. The slides link the later cleaned variant, SWE-bench Verified; the reason for that verification pass appears in the dataset-quality section near the lecture's end.

### Additional explanation

An agentic score is a property of a full experimental configuration:

$$
\text{outcome}
=f(\text{model},\text{scaffold},\text{tools},\text{budget},\text{environment},\text{grader}).
$$

Changing the retry limit, available commands, repository setup, context policy, or test harness can change the outcome without changing model weights. Reporting only the model name is therefore insufficient.

Executable tests are attractive because they replace subjective judgment with reproducible behavior. They are only as complete as their assertions, however. A patch can overfit visible tests, exploit the harness, or preserve no behavior outside tested cases. SWE-bench's later verification work illustrates why even executable graders need audit.

## 13. Terminal-Bench, CyBench, and MLE-bench

**Transcript coverage:** lines 1500-1601

### What the lecturer said - transcript only

Terminal-Bench broadens agent evaluation beyond repository repair. The environment is a computer terminal, a simple and nearly universal interface through which an agent can solve diverse tasks by issuing commands. Tasks were crowdsourced from 93 people around the world, and 89 were selected for Terminal-Bench 2.0. Depending on the task and worker expertise, human completion times range from under an hour to more than a week. Its leaderboard also shows that the same language model can achieve different accuracies under different agent implementations.

CyBench contains 40 cybersecurity Capture the Flag tasks. An agent can inspect source code, issue commands, and interact with a web server. Success means extracting a unique flag that proves the vulnerability was exploited. The initial scaffold was simple: maintain one growing response-observation history, generate an action, execute it, append the environmental feedback, and repeat. That history eventually becomes too long and motivates more sophisticated context management. Leading systems began near 10% and, by lecture time, had solved the benchmark completely.

A machine-learning-engineering benchmark uses Kaggle competitions involving data processing, model training, debugging, and submission. An agent reads the competition description and data, writes and executes code, trains models, and submits a result for grading. The familiar frontier language models dominate, but performance varies substantially across their agent scaffolds.

### Source reconciliation

The Terminal-Bench slides state that **229 tasks** were crowdsourced from 93 contributors and that 89 form Terminal-Bench 2.0. They show separate expert and junior completion-time distributions. The CyBench slides add that first-solve time is used as a difficulty measure. The slides name the machine-learning-engineering suite **MLE-bench** and specify **75 Kaggle competitions**, details absent from the transcript.

### Additional explanation

Long-horizon benchmarks require more than final-answer accuracy. Useful secondary measures include:

- success rate over repeated runs;
- tokens, tool calls, wall time, and monetary cost per attempt;
- number and duration of environment failures;
- recovery from an incorrect intermediate action;
- fraction of tasks solved within several fixed budgets;
- variance across random seeds or sampling temperatures.

A single best-of-many score can hide an unreliable system. If one run succeeds only after many expensive failures, its practical value differs from a system with the same headline success rate on first attempts.

Human-time bins provide task context but are not a perfect machine difficulty scale. Humans and agents have different strengths, tools, and parallelism. First-solve time can also reflect which humans attempted the task rather than intrinsic complexity.

## 14. Agent scaffolds are part of the evaluated system

**Transcript coverage:** lines 1602-1666

### What the lecturer said - transcript only

Sophisticated tasks require more than an unstructured stream of chain-of-thought text. Explicit planning helps an agent retain its objectives, for example by maintaining and checking off a to-do list. A long accumulated context can cause the agent to lose track of its state.

Hierarchical delegation provides encapsulation. A master agent gives a sub-agent a focused task with clean context; the sub-agent performs the detailed work and returns only its result. The master does not need every intermediate token. Persistent memory can also move information into files that the system later reads, rather than forcing everything to remain in the context window.

More generally, **context engineering** determines when to delegate, which strategy to try, what process instructions to supply, and which information to retain. These decisions are often optimized for a particular language model.

Agents enlarge the capability surface of language models, but scaffolds matter greatly. Evaluating an agent therefore evaluates the language model and scaffold together, not the model alone.

### Additional explanation

A fair comparison should declare which components are fixed:

| Comparison goal | Hold fixed | Vary |
|---|---|---|
| Model comparison | Scaffold, tools, budgets, prompts, environment | Model |
| Scaffold comparison | Model, tools, budgets, environment | Planning and context logic |
| Product/system comparison | Only task distribution and allowed external constraints | Any component |

The last comparison is useful to downstream users, while the first two support causal research claims. Without this distinction, a better agent score may be incorrectly attributed to a model when it came from more retries, better tools, or domain-specific instructions.

Context management is a resource-allocation problem. Keeping everything preserves information but consumes attention and may bury relevant facts. Summarizing or delegating compresses context but can discard details. Persistent files extend memory but create retrieval and consistency problems. Evaluation should include the cost and failure modes of these mechanisms.

## 15. ARC-AGI and the attempt to isolate reasoning from knowledge

**Transcript coverage:** lines 1667-1852

### What the lecturer said - transcript only

The previous benchmarks all require language and world knowledge. ARC-AGI asks whether reasoning or fluid intelligence can be isolated from memorized facts. It was introduced in 2019, during the GPT-2 era, with tasks intended to be fully solvable by humans but challenging for AI. Each task is meant to be unique enough that memorizing earlier solutions offers little direct benefit.

The first version uses visual grid transformations. In one simple example, a person can infer that a yellow cell should be added to complete a rectangle. A later version introduced more complicated, multi-step problems.

For years, pretrained language models made almost no progress. This matched the benchmark's premise: internet facts and linguistic patterns were not directly enough. Performance changed abruptly with OpenAI's o1 and o3 reasoning models in 2024. ARC-AGI-1 became essentially solved, and ARC-AGI-2 appeared to be moving in the same direction after its release. Pretraining may not have been directly visible in early scores, but it was arguably still a prerequisite for the reasoning-model advances.

ARC-AGI-3, released shortly before the lecture, turns the task into an interactive game-like environment without natural language. Current scores are extremely low, continuing the pattern in which a new version opens a fresh capability gap.

The attempt to disentangle reasoning from knowledge is valuable but difficult. No task comes from no prior at all, and concentrated optimization can benchmark-engineer almost any fixed suite. ARC-AGI is also deliberately restricted to human-solvable reasoning. It does not measure superhuman reasoning such as winning more mathematical olympiad gold medals or solving open mathematical problems. Even with those limits, it clearly exposes gaps in current systems.

In Q&A, a student asks how a language model receives the highly graphical ARC-AGI-3 state. The lecturer says the input may be supplied as an image or as ASCII art or another textual encoding. Either way, the model must reason spatially about something unlike ordinary English.

### Source reconciliation

The slides date ARC-AGI-2 to **March 2025** and ARC-AGI-3 to **March 2026**. They explicitly attribute the sharp progress to reasoning models such as o1 and o3. The ARC-AGI-3 table shows top listed scores below 1%, with 0.50% as the highest displayed value, providing scale for the transcript's phrase "extremely low."

### Additional explanation

ARC-style tasks attempt to reduce direct knowledge dependence by generating a new rule-induction problem from a broad family. Success still depends on priors: objectness, symmetry, counting, spatial adjacency, search strategies, and representation choices all reflect learned or engineered knowledge.

The modality question reveals another confound. Image input tests visual perception plus reasoning; ASCII input tests serialization understanding plus reasoning. If two systems receive different representations, their scores do not isolate the same capability.

A stronger evaluation can report both final success and generalization structure:

- performance on unseen task families;
- adaptation from the provided demonstrations;
- compute used at test time;
- sensitivity to representation;
- errors attributable to perception, search, or rule induction.

This turns "reasoning" from one opaque number into several testable mechanisms.

## 16. Safety benchmarks, jailbreaks, context, and dual use

**Transcript coverage:** lines 1853-2002

### What the lecturer said - transcript only

Vehicle safety has concrete crash tests and rating standards developed over decades of engineering, policy, and lobbying. AI safety lacks an equally settled definition.

HarmBench represents one common view: prompt a model with harmful requests and expect refusal. This focuses on preventing malicious users from using language models for harmful acts. AIR-Bench takes a broader approach. It draws from regulatory frameworks in the European Union, China, and the United States, along with company policies, and builds a taxonomy from which prompts and evaluations can be constructed.

Jailbreaking tests whether a model trained to refuse can be induced to comply. Early work used an automatic coordinate-wise optimization method, GCG, to search for adversarial prompt suffixes. Prompts optimized against open models transferred to closed models. At the time, apparently nonsensical text could make a closed model answer a request such as producing a step-by-step plan to destroy humanity, where refusal was the expected behavior. The lecturer hopes those attacks no longer work.

The deeper problem is that safety is contextual. Politics, law, and social norms differ across countries. The risks also differ: hallucinations in medical, legal, or financial use; sycophancy; aiding crime; inequality; and loss of critical thinking. Some risks decrease when capabilities improve, while others may increase or conflict with capability. Treating safety as whether AI goes well for society is therefore much broader than one refusal benchmark.

Language models are also dual-use. A cybersecurity agent can attack a system or perform penetration testing that makes it safer. The same capability can be harmful or beneficial depending on authorization, intent, and setting.

### Source reconciliation

The slides specify that HarmBench covers **510 harmful behaviors** that violate laws or norms. They give AIR-Bench **314 risk categories and 5,694 prompts**. They expand GCG as **Greedy Coordinate Gradient** and explicitly state transfer from an open-weight Llama model to a closed GPT-4 model. These exact names and counts clarify the spoken account.

### Additional explanation

Safety evaluation should begin with a threat model:

1. Who may cause harm: an ordinary user, malicious actor, model operator, or autonomous system?
2. What assets or people can be harmed?
3. What capabilities and access does the system have?
4. Which context makes the action authorized or prohibited?
5. What false-positive and false-negative costs are acceptable?

A refusal benchmark measures one slice of this space. Over-refusal can itself be harmful when a benign medical, educational, or defensive-security request is blocked. Reporting only attack success omits utility; reporting only helpfulness omits misuse resistance.

Dual use makes context part of the label. The same command sequence may be legitimate in a sandboxed security exercise and criminal on an unauthorized target. A safety benchmark that strips away permission and environment details can reward superficial keyword refusal rather than sound risk reasoning.

---

# Part IV - Realism, validity, and benchmark auditing

## 17. Ecological validity and the privacy-realism tension

**Transcript coverage:** lines 2003-2112

### What the lecturer said - transcript only

**Ecological validity** asks how well an evaluation captures real-world use. Exam benchmarks such as GPQA are far removed from normal interactions. Chatbot Arena obtains prompts from real people, but its user and use-case distribution is uncontrolled and may not be the distribution a developer cares about.

GDPVal tries to improve realism at the level of occupations and sectors. It covers 44 occupations from the nine largest sectors by United States GDP. Professionals with about 14 years of experience created tasks. The examples include nursing, concierge work, real estate, film and video editing, manufacturing engineering, financial analysis, customer service, and other jobs.

A medical effort similarly moves beyond standardized exams. It sources 121 realistic clinical tasks from 29 clinicians. Passing a medical school examination is not equivalent to being ready to operate on patients; models likewise require evaluation on the work clinicians would actually ask them to perform.

Real deployment data would be even more representative, and model providers possess it. One project uses language models themselves to analyze private user conversations and summarize broad patterns in how people use Claude. This avoids direct human inspection of individual conversations.

Realism and privacy conflict. A random sample from the true query stream would be valuable for measuring deployment performance and understanding errors, but it contains sensitive user information. More realistic evaluation may therefore require stronger privacy controls or more aggregate reporting.

### Source reconciliation

The slides name the occupational benchmark **GDPVal**, the clinical suite **MedHELM**, and Anthropic's usage-analysis project **Clio**. They confirm 44 occupations, 9 sectors, and about 14 years of professional experience for GDPVal. The MedHELM slide states that its 121 tasks from 29 clinicians use a mixture of private and public datasets, a detail not stated in the transcript.

### Additional explanation

Ecological validity has several components:

- **population validity:** intended users are represented;
- **task validity:** requests resemble real work;
- **environment validity:** tools, documents, latency, and interaction constraints match deployment;
- **consequence validity:** the grader reflects what success or failure actually costs;
- **temporal validity:** tasks remain representative as use changes.

A benchmark can be realistic along one dimension and artificial along another. Occupational tasks may represent genuine workflows while still using a clean sandbox, generous time, or simplified documents.

Privacy-preserving evaluation can use consented samples, access-controlled enclaves, de-identification, aggregation, differential privacy, or model-mediated summaries. Each intervention may reduce fidelity. The important practice is to report what transformation separated the benchmark from raw use data and which behaviors it may have removed.

## 18. Four routes for addressing train-test contamination

**Transcript coverage:** lines 2113-2247

### What the lecturer said - transcript only

Scientific validity is harder when foundation models train on the internet and providers do not disclose their data. Before foundation models, datasets such as ImageNet and SQuAD had explicit training and test splits. Everyone trained on the designated training set and evaluated on the same held-out set. That clean separation is no longer guaranteed.

The lecture presents four responses.

1. **Infer exposure from model behavior.** Benchmark item order should be exchangeable or arbitrary. If a model assigns unusually high probability to the canonical published order compared with shuffled orders, it may have seen the benchmark sequence during training.
2. **Create reporting norms.** Statistics routinely reports confidence intervals so readers can assess uncertainty. Similarly, a provider claiming a GPQA score should disclose or justify the degree of train-test overlap rather than leaving the issue unaddressed.
3. **Use fresh evaluations.** LiveCodeBench and UncheatableEval construct tasks from webpages, papers, or GitHub material created after a model's training cutoff. This is relatively robust, but timestamps do not prove conceptual freshness: a newly posted repository may copy older material.
4. **Use private evaluations.** Companies can evaluate on internal code that is not on the internet and was not used for training. Individuals can use unpublished personal writing; the lecturer's example is a collection of rejected graduate-school papers that he never posted online. Private text is especially convenient for perplexity because any coherent held-out corpus can be scored without creating answers or graders.

No route is perfect. Together they reduce reliance on trust in an unknown pretraining corpus.

### Source reconciliation

The slides describe the first method as exploiting **exchangeability of data points** and depict higher likelihood for a benchmark's canonical item order than for a shuffled order as contamination evidence. They spell the fresh benchmark **UncheatableEval**, correcting the less clear automatic transcription.

### Additional explanation

The ordering test uses a signal that should not carry task information. If items are independently assembled, their publication order is arbitrary. Memorizing the exact serialized benchmark can make that order more probable. The test is clever because it does not require access to training data, but it detects only certain forms of exposure. A model could see each item separately without memorizing canonical order, or acquire order preference from another published artifact.

Fresh and private evaluations trade transparency for cleanliness:

| Strategy | Contamination resistance | Public reproducibility | Long-term maintenance |
|---|---:|---:|---:|
| Old public benchmark | Low to uncertain | High | Easy |
| Continuously fresh benchmark | Medium to high near release | Medium | Expensive |
| Secret/private benchmark | High if securely managed | Low | Requires governance |
| Behavioral contamination test | Detects some exposure | Medium | Must be validated |

Reporting overlap should distinguish exact matches, near-duplicates, source documents, and benchmark-specific synthetic variants. A raw overlap percentage without the matching rule, corpus coverage, and denominator is difficult to interpret.

## 19. Dataset quality, executable-grader failures, and qualitative audits

**Transcript coverage:** lines 2248-2325

### What the lecturer said - transcript only

Evaluation can fail because the benchmark itself is wrong. SWE-bench contained tasks and tests that were not rigorous enough, motivating SWE-bench Verified. GSM8K, MMLU, and other benchmarks have also been audited and cleaned. Example defects include a mathematics question that refers to a missing curve and a visual question asking whether a baby wears socks when the image does not reveal the answer.

Agentic benchmarks are harder to audit because the evaluator must inspect an entire environment, tool interface, and execution trace rather than one question and answer. Unit tests provide an appealing binary grader, but tests can be incomplete: a patch may pass every test without being a genuinely correct solution.

The lecturer cites a TorchBench case in which an agent that returns an empty response can receive 38%, exposing a trivial shortcut. Some benchmark errors are therefore large enough to dominate reported system failures.

Docent uses a language model to inspect agent traces and detect benchmark problems. This is a qualitative complement to quantitative scores. Whenever developing a benchmark or running a model on one, the lecturer recommends examining actual outputs and traces to verify that the benchmark measures what its designers believe it measures.

### Source reconciliation

The slides explicitly recommend fixing benchmarks into verified or **Platinum** versions. They show audited examples from SVAMP, GSM8K, VQA, and MMLU, and a table where cleaning substantially reduces measured errors on many agent benchmarks. The transcript supplies the main lesson and representative defects but not every table entry.

### Additional explanation

A benchmark audit should include at least four layers:

1. **Item audit:** Is the prompt answerable, unambiguous, and correctly labeled?
2. **Environment audit:** Are dependencies, files, network services, and permissions deterministic and complete?
3. **Grader audit:** Do tests reject known wrong solutions and accept diverse correct ones?
4. **Trace audit:** Did the system solve the intended task rather than exploit leakage, stale state, or a scoring bug?

Useful audit techniques include running trivial baselines, mutation testing the grader, inserting deliberately wrong solutions, replaying traces in a clean environment, and manually sampling both successes and failures. The empty-response result demonstrates why a benchmark should always be tested against degenerate agents before serious model comparison.

LLM-based trace analysis can increase audit coverage, but the auditor is itself fallible. Its findings should point humans toward suspicious cases rather than serve as unquestioned ground truth.

---

# Part V - Choosing the evaluation for the decision

## 20. Purpose, object of comparison, and the rules of the game

**Transcript coverage:** lines 2326-2425

### What the lecturer said - transcript only

There is no single evaluation that rules them all. The correct design depends on the purpose, which is often left unstated. The lecture distinguishes at least four goals:

1. A user or company wants to choose model A or B for a concrete use case, such as a customer-service chatbot.
2. A researcher wants to measure an abstract capability such as intelligence.
3. A business or policymaker wants to understand benefits and harms.
4. A model developer wants diagnostic feedback that guides improvement.

These goals can require different benchmarks or combinations of evidence.

The object of evaluation has also changed. Before foundation models, standardized train-test splits made it natural to compare **methods**: the data and task were fixed, and the algorithm varied. Today, most leaderboards compare complete **models and systems**, where architecture, data, post-training, inference procedure, tools, and scaffolds may all differ. This is useful to downstream users because the complete system is what ships, but it is weaker evidence about which algorithm caused an improvement.

The NanoGPT speedrun is an exception that compares methods under fixed data and a fixed target validation loss, measuring how quickly a model can be trained. Such a benchmark encourages algorithmic innovation. Complete-model evaluation instead supports purchase and deployment decisions.

The lecture closes by rejecting a universal metric. Perplexity, exams, chat, agentic tasks, pure reasoning, and safety evaluations differ in difficulty, realism, and validity. These desirable properties can conflict. Highly realistic tasks may be hard to grade, public tasks may be contaminated, and private tasks may be difficult to reproduce. The evaluator must choose which compromise serves the goal and define the rules of the game clearly. The next lectures turn to training data.

### Additional explanation

An evaluation report should function as a decision record. At minimum it should declare:

| Field | Question to answer |
|---|---|
| Purpose | What decision will this result support? |
| Construct | What property is being operationalized? |
| Evaluation unit | Method, base model, post-trained model, system, or agent? |
| Task distribution | Which users, domains, languages, and time period are represented? |
| Protocol | Prompting, tools, sampling, budgets, retries, and answer extraction? |
| Grader | Exact match, tests, humans, LLM judge, verifier, or combination? |
| Uncertainty | How many items and runs, and what confidence interval? |
| Validity risks | Contamination, judge bias, missing cases, privacy transformation? |
| Cost | Tokens, wall time, hardware, and money per evaluated task? |

This prevents two common category errors: using a product benchmark to claim one algorithm is better, and using a controlled method benchmark to claim one deployed product will serve users better.

Difficulty, realism, and validity should be treated as axes rather than a checklist with one perfect endpoint. A benchmark suite is often more credible than a single average because different components can deliberately occupy different points in this space.

---

# Consolidated takeaways

1. Evaluation shapes AI development because metrics become North Stars. Choosing a metric is choosing which behavior receives optimization pressure.
2. The central design problem is converting an abstract construct into tasks, interactions, graders, and statistics without losing the intended meaning.
3. Benchmark score, inference cost, human preference, and observed use are different definitions of "good" and need not rank models identically.
4. Perplexity is a proper, smooth intrinsic metric that supports scaling analysis, but it spends probability mass on every token rather than only on deployment-relevant behavior.
5. Conditional perplexity and cloze or continuation tasks focus language-model scoring on selected responses or positions.
6. Probability-based leaderboards require trust or controlled execution because arbitrary submitted log probabilities cannot easily be verified as a normalized distribution.
7. Exam benchmarks provide controlled difficulty and objective grading, but their multiple-choice format covers only a narrow part of real use.
8. MMLU, MMLU-Pro, GPQA, and Humanity's Last Exam illustrate a recurring cycle of saturation, cleanup, and harder replacement.
9. Public benchmarks can be contaminated through exact examples, near-duplicates, or underlying source material even when the benchmark file itself was not used for training.
10. Open-ended chat evaluation commonly uses pairwise preference. It improves relative judgment but inherits user-distribution, correctness, style, sycophancy, and judge-bias problems.
11. LLM judges can make evaluation scalable, but verbosity bias and leaderboard gaming show that judge behavior must itself be evaluated.
12. Prompt-specific rubrics and checklists make both human and model judgments more defined and auditable.
13. Agent scores measure a language model together with its scaffold, tools, budget, environment, and grader. The model name alone does not specify the evaluated system.
14. Agent benchmarks gain realism from executable environments, but incomplete tests, nondeterminism, and trivial exploits make auditing harder than for static questions.
15. ARC-AGI attempts to isolate fluid reasoning from knowledge, yet representations, learned priors, test-time compute, and benchmark-specific optimization prevent a perfect separation.
16. AI safety is not one refusal rate. It is contextual, spans many risk types, and must address both misuse and the dual-use value of capable systems.
17. Ecological validity asks whether users, tasks, environments, and consequences resemble deployment. Realism often conflicts with privacy.
18. Contamination can be mitigated through behavioral inference, reporting norms, fresh evaluations, and private data; no one route is complete.
19. Benchmark quality is an empirical question. Inspect prompts, labels, environments, graders, outputs, and traces, and test trivial baselines before trusting a score.
20. Method evaluation supports causal and algorithmic claims; system evaluation supports deployment decisions. Every report should state which game is being played.
21. There is no one true evaluation. A credible suite is purpose-specific and explicit about its compromises among difficulty, realism, validity, reproducibility, and cost.

# Key equations

## Perplexity

For a token sequence $x_{1:N}$:

$$
\operatorname{PPL}(x_{1:N})
=\exp\left(
-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i\mid x_{<i})
\right)
=p(x_{1:N})^{-1/N}.
$$

## Conditional perplexity

For response $r_{1:M}$ conditioned on prompt $q$:

$$
\operatorname{PPL}(r\mid q)
=\exp\left(
-\frac{1}{M}\sum_{j=1}^{M}\log p(r_j\mid q,r_{<j})
\right)
=p(r\mid q)^{-1/M}.
$$

## Cross-entropy optimum

$$
H(t,p)=H(t)+D_{\mathrm{KL}}(t\Vert p)\ge H(t).
$$

The expected log loss is minimized at $p=t$; the corresponding minimum perplexity is $\exp(H(t))$ under natural logs.

## Multiple-choice continuation scoring

$$
\hat{k}=\arg\max_k \log p(c_k\mid q).
$$

The exact result may depend on length normalization, shared prefixes, prompting, and answer extraction.

## Arena Elo probability

$$
P(A\text{ beats }B)
=\frac{1}{1+10^{(E_B-E_A)/400}}.
$$

Only $E_A-E_B$ affects the probability; ratings require a chosen offset or normalization.

## Empirical benchmark mean

For per-item scores $s_1,\ldots,s_n$:

$$
\hat{\mu}=\frac{1}{n}\sum_{i=1}^{n}s_i.
$$

This estimates performance only for the sampled item distribution and stated protocol. Repeated runs or clustered sampling may be needed when agent outcomes are stochastic or items are not independent.

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Agent | A language model combined with scaffold logic, tools, memory, and an iterative environment interaction. |
| AlpacaEval | Fixed-prompt open-ended evaluation using LLM-judged win rate against a baseline. |
| Answer extraction | Procedure that maps a generated explanation or free-form response to the benchmark's scored answer. |
| ARC-AGI | Visual or interactive task family intended to test novel rule induction and human-solvable reasoning. |
| Benchmark saturation | Regime in which top systems approach the ceiling and the benchmark no longer distinguishes them well. |
| Chatbot Arena | Dynamic pairwise human-preference evaluation, now called Arena AI. |
| Checklist | Prompt-specific list of criteria used to make open-ended judgment more defined. |
| Conditional perplexity | Perplexity of response tokens conditioned on a supplied prompt. |
| Construct validity | Degree to which an operational metric supports the intended abstract concept. |
| Contamination | Evaluation information appearing in training directly, through duplicates, sources, or related benchmark artifacts. |
| CyBench | Agent benchmark built from cybersecurity Capture the Flag tasks. |
| DIAMOND set | More stringently validated GPQA subset based on expert agreement and non-expert testing. |
| Ecological validity | Degree to which evaluation reflects actual users, tasks, environments, and consequences. |
| Elo rating | Parameter in a pairwise win-probability model used to aggregate comparisons. |
| Exam benchmark | Evaluation with controlled questions and typically unambiguous answers, often multiple choice. |
| GCG | Greedy Coordinate Gradient, an automated prompt-suffix optimization method used for jailbreak research. |
| GDPVal | Occupational benchmark built from tasks across major economic sectors. |
| GPQA | Graduate-Level Google-Proof Q&A, a difficult expert-written question set. |
| HLE | Humanity's Last Exam, a difficult multimodal, multi-subject exam benchmark. |
| LLM as a judge | Use of a language model to compare, score, or critique candidate responses. |
| MLE-bench | Agent benchmark based on end-to-end machine-learning competitions. |
| MMLU | Massive Multitask Language Understanding, a 57-subject multiple-choice benchmark. |
| MMLU-Pro | Harder MMLU successor with cleaned questions, more answer choices, and chain-of-thought evaluation. |
| Perplexity | Exponential of average negative log-likelihood over a token sequence. |
| Pairwise preference | Judgment that one response is better than another rather than an absolute numeric score. |
| Private evaluation | Test data kept out of public release and, ideally, out of training corpora. |
| Rubric | Explicit scoring dimensions and standards for a response or behavior. |
| Scaffold | Logic around a model that plans, calls tools, manages context, delegates, and persists memory. |
| SWE-bench Verified | Audited version of the code-repair agent benchmark with more reliable tasks and tests. |
| Sycophancy | Tendency to favor agreement or pleasing responses over honest, correct, or appropriately critical ones. |
| Terminal-Bench | Agent benchmark whose tasks are performed through a computer terminal. |
| WildBench | Open-ended benchmark using real chat-derived prompts and checklist-guided LLM judgment. |

# Self-check questions

1. Why is converting an abstract construct into a metric the core evaluation challenge?
2. How do benchmark quality, price, human preference, and paid usage answer different questions about whether a model is good?
3. Derive perplexity from average autoregressive negative log-likelihood.
4. Why is token-level perplexity not directly comparable across unrelated tokenizers?
5. What is the difference between in-distribution and zero-shot out-of-distribution language-model evaluation?
6. Use the cross-entropy decomposition to explain why $p=t$ minimizes expected log loss.
7. Why is the minimum perplexity $\exp(H(t))$ rather than $H(t)$ under natural-log entropy?
8. How does conditional perplexity avoid scoring irrelevant prompt tokens?
9. In what sense are LAMBADA and HellaSwag perplexity in disguise?
10. Why can a black-box answer leaderboard verify submissions more easily than a black-box perplexity leaderboard?
11. What properties make exams easy to grade, and what real-world behaviors do they omit?
12. How do MMLU-Pro, GPQA, and HLE restore difficulty after older benchmarks saturate?
13. Why can source contamination matter even when no exact test question appears in training?
14. What answer-extraction errors can occur when a model generates chain of thought before a multiple-choice answer?
15. How does an Elo-style model aggregate sparse pairwise comparisons, and why must the comparison graph be connected?
16. Why can Arena preference conflate style, correctness, and sycophancy?
17. What did AlpacaEval's length bias reveal about optimizing against an LLM judge?
18. Why does high correlation with a historical Arena ranking not universally validate an automatic judge?
19. How does a prompt-specific checklist improve human or LLM judgment?
20. Why is an agent benchmark score a property of more than the base language model?
21. What additional statistics should accompany a stochastic agent success rate?
22. How can incomplete unit tests allow a wrong agent solution to pass?
23. What is ARC-AGI trying to isolate, and why can reasoning never be made completely prior-free?
24. How can image versus ASCII representation confound a pure-reasoning comparison?
25. Why is refusal on harmful prompts only one part of safety evaluation?
26. How does dual use make authorization and context part of a safety label?
27. What are the population, task, environment, and consequence aspects of ecological validity?
28. Why are realism and privacy often in tension?
29. What does a canonical-order contamination test detect, and what kinds of exposure can it miss?
30. Compare fresh, private, public, and behaviorally tested evaluations in contamination resistance and reproducibility.
31. What should be checked in an item, environment, grader, and trace audit?
32. When should an evaluation fix the system and vary the method, and when should it allow complete systems to compete?
33. Why can no single benchmark simultaneously maximize difficulty, realism, validity, reproducibility, and low cost?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-174 | Course transition to data, evaluation's influence, abstract constructs, benchmark/cost/preference/usage definitions of good | Yes |
| 175-316 | Perplexity, classic in-distribution datasets, neural progress, GPT-2 WebText and zero-shot transfer | Yes |
| 317-423 | Perplexity-is-all-you-need argument, true distribution, scale motivation, over-broad scoring, conditional perplexity | Yes |
| 424-564 | LAMBADA, HellaSwag, probability-leaderboard trust, black-box responses, latent-variable bounds | Yes |
| 565-591 | Perplexity summary, smooth scaling-law role, need for real-world benchmarks | Yes |
| 592-747 | Exam advantages, MMLU construction and few-shot GPT-3, saturation, harder MMLU successor | Yes |
| 748-872 | GPQA construction, DIAMOND review, human/model scores, contamination Q&A and source-level nuance | Yes |
| 873-1015 | HLE construction and private set, benchmark replacement pattern, multiple-choice limits, answer-extraction Q&A | Yes |
| 1016-1225 | Open-ended chat, Chatbot Arena collection and Elo ranking, population/style/correctness/sycophancy issues, sparse dynamic comparisons | Yes |
| 1226-1364 | AlpacaEval, LLM judges, length gaming, regression debiasing, metric correlation, WildBench checklists | Yes |
| 1365-1422 | Open-ended evaluation summary, pairwise versus absolute judgments, multiple judges, rubrics | Yes |
| 1423-1499 | Agents as model plus scaffold, SWE-bench task and unit-test evaluation, progress on Verified | Yes |
| 1500-1601 | Terminal-Bench, CyBench, MLE-bench, human difficulty, agent/model variation and task mechanics | Yes |
| 1602-1666 | Explicit planning, hierarchical delegation, persistent memory, context engineering, scaffold importance | Yes |
| 1667-1852 | ARC-AGI 1/2/3, reasoning versus knowledge, performance history, limitations, image/ASCII Q&A | Yes |
| 1853-2002 | Vehicle-safety analogy, HarmBench, AIR-Bench, GCG jailbreaks, contextual risks, dual use | Yes |
| 2003-2112 | Ecological validity, GDPVal, clinical tasks, model-mediated usage analysis, realism/privacy conflict | Yes |
| 2113-2247 | Train-test overlap, exchangeability test, reporting norms, fresh and private evaluations | Yes |
| 2248-2325 | Dataset defects, SWE-bench Verified, audited static and agent benchmarks, empty-response exploit, Docent and qualitative inspection | Yes |
| 2326-2425 | Evaluation purposes, methods versus models/systems, NanoGPT speedrun, trade-offs, rules of the game, transition to data | Yes |
