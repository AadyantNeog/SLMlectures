---
title: "Lecture 14 - Data Processing, Filtering, Deduplication, and Mixing"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 14
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 14 Data.txt"
slide_deck: "../lecture_14.pdf"
status: "complete"
---

# Lecture 14: Data Processing, Filtering, Deduplication, and Mixing

## How to read these notes

Every substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a cleaned and compressed paraphrase of the spoken lecture. It preserves the lecturer's claims, examples, qualifications, numbers, warnings, and substantive questions and answers while removing filler, false starts, and repetition.
2. **Additional explanation.** This is supplementary interpretation, derivation, intuition, or study guidance. It is not presented as something the lecturer said.

When a slide supplies exact notation, resolves a spoken ambiguity, or materially differs from the speech, a separate **Source reconciliation** subsection records that fact. The complete 2,531-line transcript was mapped before the 30-slide deck was rendered and visually inspected page by page.

## Lecture map

The lecture follows data from raw artifacts to training examples in four stages:

1. **Transformation and filtering:** convert HTML and PDFs into text, define a target distribution, and use cheap models to select language-, quality-, domain-, or toxicity-relevant documents.
2. **Deduplication:** detect exact and near duplicates efficiently with hashes, Jaccard similarity, MinHash, and locality-sensitive hashing.
3. **Data mixing:** choose sampling weights across finite sources while controlling diversity, repeated epochs, downstream-target overfitting, and small-to-large-scale transfer.
4. **Post-training data:** construct task-specific synthetic or semi-synthetic examples, with recent reasoning and software-engineering datasets as case studies.

---

# Part I - From raw artifacts to filtered text

## 1. Where this lecture fits in the data pipeline

**Transcript coverage:** lines 1-37

### What the lecturer said - transcript only

The previous lecture established that data does not simply appear. Internet data begins in live services, must be obtained through a dump or crawl, and then must be processed. Its collection also involves terms of service, copyright, licensing, and possible fair-use arguments.

This second data lecture studies the next processing stages: transformation, filtering, deduplication, and mixture construction. It ends with post-training data, especially the current use of synthetic data. The first and larger part should be understood mainly as a pretraining-data pipeline.

### Additional explanation

The pipeline can be viewed as a sequence of increasingly semantic decisions:

$$
\text{raw artifacts}
\rightarrow \text{text representations}
\rightarrow \text{selected documents}
\rightarrow \text{unique documents}
\rightarrow \text{training distribution}.
$$

Transformation asks how an artifact should be represented. Filtering asks whether it belongs. Deduplication asks whether its information is already present. Mixing asks how often each retained source should be sampled. These decisions are coupled: a transformation error can change a filter score, and mixture weights determine whether a small filtered source is repeated too often.

## 2. HTML-to-text transformation is fast, heuristic, and lossy

**Transcript coverage:** lines 38-129

### What the lecturer said - transcript only

Scraped raw data is not necessarily text. Common Crawl records are commonly HTML; GitHub data can be directory structures; and some web artifacts are PDFs. Because most web pages are HTML, much transformation work focuses on converting HTML into text.

Typical processors remove boilerplate such as navigation, advertisements, headers, footers, and menus, then extract what appears to be the page's main content. The boundary is not absolute. Navigation elements may themselves help a model learn what web pages look like, so labeling them as non-content is a task-dependent judgment.

Images and tables create further problems. HTML is hierarchical, while its rendered form is visual; either form must be linearized into a token sequence. This makes the conversion inherently lossy. A simple table may be rendered as Markdown, but nested tables are much harder and eventually require approximation or abandonment.

HTML-to-text systems are usually rule-based because rules are very fast and the task often does not seem to require much intelligence. Model-based intervention could improve the output if it remained extremely cheap and added genuinely useful judgment. In practice, every rule system has a failure rate, and inspecting processed examples exposes imperfections. The choice of tool therefore matters: on the extended DCLM evaluations shown previously, Resiliparse performed better than the alternatives discussed, including Trafilatura.

### Source reconciliation

Slide 2 gives the exact DCLM comparison that the speech only summarizes. On CORE, Resiliparse, Trafilatura, and WET score 24.1, 24.5, and 20.7. On EXTENDED, they score 13.4, 12.5, and 12.2. Thus Trafilatura is slightly higher on CORE, while Resiliparse is highest on EXTENDED, matching the lecturer's qualified claim.

### Additional explanation

HTML extraction is not merely tag deletion. A robust system has to infer which DOM regions are content, preserve useful boundaries, decide how lists and links are represented, and avoid concatenating unrelated regions. Two opposing error types matter:

- **Under-extraction:** useful prose, labels, captions, or table cells disappear.
- **Over-extraction:** menus, cookie banners, repeated sidebars, or unrelated recommendations enter the corpus.

The appropriate tradeoff depends on the model. A prose-focused model may value aggressive boilerplate removal; a web-navigation agent may need controls, link anchors, and layout cues that such a cleaner discards. Sampling and reading transformed records is indispensable because aggregate extraction scores can hide systematic failures on unusual languages, templates, or page types.

## 3. PDF extraction adds retrieval, OCR, layout, and cleanup problems

**Transcript coverage:** lines 130-205

### What the lecturer said - transcript only

Hugging Face's FinePDFs work illustrates a different transformation pipeline. PDFs occur on ordinary web pages and sometimes appear in Common Crawl. A crawler may not know that a URL points to a PDF until it fetches it, especially when the URL lacks a file extension. Common Crawl often truncates PDFs because they are large, so building a PDF corpus may require recrawling the source URLs.

After retrieval, the file still has to be converted into text. Some PDFs contain text objects, while others are effectively page scans. The methods studied in the FinePDFs work therefore include OCR, often using a vision-language model, and are much more expensive than ordinary HTML extraction.

PDFs form only a small fraction of web data, but the lecturer argues that their average quality is higher: producing a PDF often signals that the author had more substantial material than an average web page. Even so, the extracted corpus requires extensive cleanup and filtering.

PDF conversion can lose more semantic information than HTML conversion. HTML tags such as `h1` and `p` explicitly identify structural roles. A PDF is designed primarily to position visual elements, so semantic reading order and document structure may not be preserved. Obtaining text is therefore only an early stage, not the completion of data processing.

### Source reconciliation

Slide 3 names RolmOCR as the OCR example and also lists VLM-based extraction and Docling. The transcript mentions OCR with a VLM but does not name RolmOCR or Docling, so those names are slide-only implementation examples.

### Additional explanation

A PDF pipeline may need to solve four distinct tasks:

1. recover all bytes rather than an incomplete crawl response;
2. distinguish embedded text from image-only or mixed pages;
3. infer reading order across columns, captions, headers, equations, and footnotes;
4. serialize the result without destroying relationships such as a caption's connection to a figure.

OCR accuracy alone is not enough. A system can recognize every character yet interleave two columns or detach table values from their headers. For language-model training, structural errors can produce fluent-looking but semantically nonsensical sequences, making visual spot checks and layout-aware validation important.

## 4. The general filtering problem and two model families

**Transcript coverage:** lines 206-352

### What the lecturer said - transcript only

Filtering begins with a small, high-quality target set $T$ and a much larger raw pool $R$ produced by transformation. The goal is to find a subset of $R$ that resembles $T$. This template covers several applications:

- language identification, such as retaining German for a German model;
- quality selection, such as preferring encyclopedic information over spam;
- toxicity filtering, when harmful internet content is not wanted in training.

A useful filter must generalize beyond the target examples themselves and must run over an enormous raw pool, potentially 100 trillion tokens. It therefore has to be extremely fast. Filtering commonly leaves only a single-digit percentage of the original data.

The general recipe is to estimate a model from $R$ and $T$, derive a score for each raw document, and retain documents according to that score. Two model families are common:

1. **Generative target model.** Fit a cheap language model to $T$ and score a document by how probable it is under that model. KenLM is a typical choice and can fit a 5-gram model rather than a large neural LM.
2. **Discriminative classifier.** Label target examples as positive and random raw examples as negative, possibly balance the classes, and train a classifier. fastText is common because it is fast; in this setting it is essentially a linear bag-of-words classifier.

Each raw document is scored, then retained above a chosen quality threshold, either deterministically or stochastically. This is deliberate model-based filtering. Several older datasets avoided it to reduce selection bias. The lecturer says it is now nearly universal because most projects are compute-poor: spending training FLOPs on low-quality tokens is otherwise wasteful. A compute-plentiful project could filter less and train on more of the pool.

### Additional explanation

The two approaches estimate related quantities. A target language model approximates $p_T(x)$ and treats high target likelihood, or low target perplexity, as evidence of membership. A classifier approximates something like $p(T\mid x)$ directly. By Bayes' rule, the classifier can exploit how the target differs from the raw background rather than merely what is common in the target.

That distinction matters. A target LM may assign high probability to generic grammatical prose even when it is not distinctive of the desired domain. A discriminative classifier can learn features that separate mathematical prose from ordinary web prose, but its score depends on how negatives were sampled and how class balance was constructed.

Stochastic retention can preserve diversity among middling-scored documents instead of creating a hard boundary. It also allows the score to define a sampling distribution, but it does not eliminate the need to audit which communities, languages, or formats the model systematically ranks lower.

## 5. Language identification and targeted mathematical filtering

**Transcript coverage:** lines 353-456

### What the lecturer said - transcript only

Meta provides off-the-shelf fastText language-identification models supporting 176 languages. They were trained on multilingual sources including Wikipedia, translation sites, and language-specific sites. Language identification is relatively easy compared with many LM data problems because a few words often distinguish languages such as Spanish and Japanese. Code-switching and dialects remain subtle, so the problem is not absolutely solved, but the lecturer does not view it as the main bottleneck to a good language model. Once a classifier exists, its retention threshold is generally chosen heuristically.

OpenMathText, a 2023 effort to build a large mathematical corpus, shows that filters can define quality for a particular capability. Its pipeline combines several signals rather than relying on one classifier:

1. rules check for LaTeX commands;
2. a KenLM model trained on ProofPile provides a target-domain perplexity score, and documents below a threshold are retained;
3. a fastText classifier predicts whether the writing is mathematical.

The fastText threshold is lower when LaTeX is already present and higher when it is absent. The resulting corpus contained about 15 billion tokens. Models trained on this targeted corpus outperformed mathematical models trained on 20 times as much data that had not been selected this way. The example illustrates that there is no universal definition of quality. If the goal is mathematical capability, quality can intentionally mean mathematical relevance.

### Source reconciliation

Slide 5 gives Dolma's concrete language-ID rule as retaining pages with $p(\mathrm{English})\ge 0.5$, an example of the heuristic threshold mentioned in speech. It also supplies exact OpenMathText values omitted or rounded in speech: ProofPile KenLM perplexity below 15,000; fastText threshold 0.17 when mathematical or LaTeX evidence is present and 0.8 when it is absent; and a final corpus of 14.7B tokens rather than the spoken rounded 15B. The slide states that 1.4B-parameter models trained on this data beat models trained on 20 times as much unfiltered data.

### Additional explanation

This pipeline is a cascade. Cheap, high-recall rules identify obvious candidates; domain likelihood and discriminative evidence then make finer decisions. Cascades can reduce cost because expensive or detailed scoring is applied only after obvious non-matches have been removed.

The example also separates two meanings of quality:

- **intrinsic quality:** coherence, correctness, originality, or absence of spam;
- **utility for a target capability:** how much the example helps a model learn math, code, legal language, or another domain.

A document can be excellent prose yet have little utility for mathematical pretraining. Conversely, a terse proof fragment may be valuable for math despite looking unlike polished general prose.

## 6. GPT-3, LLaMA, phi-1, and toxicity as filtering instances

**Transcript coverage:** lines 457-525

### What the lecturer said - transcript only

GPT-3's quality filter fits the same target-versus-raw framework. Positive examples came from Wikipedia, WebText pages linked from high-scoring Reddit posts, and books; negative examples were sampled from the web. A linear classifier scored documents, and sufficiently high-scoring ones were retained.

The first LLaMA paper used pages referenced by Wikipedia as positives, rather than the Wikipedia articles themselves. This again uses a small, trusted selection mechanism to identify desirable pages outside the seed set.

Microsoft's phi-1 pipeline began with the Python subset of The Stack as raw data. A prompt defined educational value, and GPT-4 labeled a 100,000-example subset. Positively classified examples became the target set. A much cheaper Random Forest classifier was then trained on those labels and applied to the rest of the raw corpus. The filtered data produced better performance in fewer steps than the unfiltered Python source.

Toxicity filtering follows the same pattern. Jigsaw Toxic Comments contains annotations derived from Wikipedia talk pages, including heated discussions on controversial topics. Those labels define positive and negative examples for a classifier that can then be applied to a larger crawl.

### Source reconciliation

Slide 5 names GPT-3's four positive components as Wikipedia, WebText2, Books1, and Books2. Slide 6 quantifies phi-1's comparison: a 1.3B-parameter model trained on raw Python reaches 12.19% HumanEval after 96,000 steps, while the filtered-data model reaches 17.68% after 36,000 steps. The slide also lists the Jigsaw label categories used by Dolma: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. These exact names and values are not spoken in the transcript.

### Additional explanation

The phi-1 design is a common weak-supervision pattern:

$$
\text{expensive teacher on a sample}
\rightarrow \text{cheap student filter}
\rightarrow \text{mass scoring}.
$$

Its success depends on more than teacher quality. The labeled sample must span the raw distribution, the prompt must operationalize the desired property consistently, and the cheap model must have features capable of reproducing the distinction. A cheap classifier is attractive because inference cost is multiplied by every candidate document.

Toxicity also shows why label semantics matter. A classifier trained on Wikipedia talk-page norms may not transfer perfectly to quoted speech, reclaimed language, historical documents, or code. Filtering can reduce exposure to unwanted content, but false positives can erase dialects or discussions about harmful material rather than harmful use itself.

## 7. The right filtering threshold depends on the training-token budget

**Transcript coverage:** lines 526-684

### What the lecturer said - transcript only

Once a desired data type is identified, a classifier can be trained and used to filter Common Crawl. However, no classifier threshold, such as 0.9, is universally optimal. The correct quality bar depends on the model's planned number of training tokens. A short run generally benefits from concentrating on the highest-quality data. A much longer run can tolerate lower-quality data because the finite high-quality pool would otherwise have to be repeated. More high-quality data would be preferable, but the available pool is fixed.

The lecturer illustrates this with Michael Ryan's preliminary experiment on a 157M-parameter model and a very small pool of 100 WARC files. The DCLM-filtered data begins with better loss. As training continues, epoch boundaries show that the same small filtered pool is being reused; a second look still helps, but repeated epochs eventually lead to overfitting. Resiliparse, described here as nearly unfiltered, performs worse initially. Because it contains much more data, however, a sufficiently long run continues improving and can eventually beat a run that has exhausted the smaller high-quality set. The intended lesson is not to keep training the overfitting filtered run, but that data quality and available unique volume must be evaluated together.

An audience member asks whether each plot point is a single run and whether confidence intervals are needed. Each point is indeed one run. Repeating runs and reporting confidence intervals would be ideal practice, but full pretraining runs are expensive, so papers often omit them. The lecturer says that in their experience pretraining results tend to be fairly stable.

Another audience member asks whether training longer on high-quality data would still show diminishing returns. Any finite dataset eventually does. A larger or better high-quality set would shift the curve to a better loss and extend its useful region, but it would still eventually be exhausted.

### Additional explanation

Filtering creates a quality-quantity frontier. If a score threshold $\tau$ retains $N(\tau)$ unique tokens with average utility $q(\tau)$, increasing $\tau$ usually raises $q$ while lowering $N$. For a training budget $D$, the approximate number of passes is $D/N(\tau)$. A threshold that is excellent for $D \ll N$ can be harmful for $D \gg N$ because repeated exposure replaces new, moderately useful information.

This is why filter evaluation should specify the downstream regime. Reporting only the retained fraction or average score is insufficient. A useful experiment plots validation loss against tokens trained, marks epoch boundaries, and compares methods at the intended training horizon.

Single-run stability should also be separated from confidence in the conclusion. Optimization noise may be small, yet uncertainty can still arise from the chosen crawl slice, evaluation domains, filter training data, or hyperparameters. Those sources often dominate seed variance.

## 8. Practical filtering recipe

**Transcript coverage:** lines 685-719

### What the lecturer said - transcript only

Filtering is critical when compute is constrained, as it is for most projects. With literally infinite compute, one could train a giant model on everything and filter less, but realistic projects must avoid wasting compute.

The practical recipe is to define what good data looks like, train a classifier, and extrapolate that judgment to the larger pool. The target can come from an existing dataset whose qualities one wants to reproduce. Alternatively, an LM prompt can label an initial sample of the raw pool; those high-quality labels then train a smaller, cheaper classifier for the complete pool.

### Additional explanation

A disciplined filtering workflow records the target definition, negative sampling method, score calibration, threshold, retained fraction, and examples near both sides of the boundary. It should also keep enough provenance to recreate or revise the selection later. Because a filter changes the model's effective training distribution, it is part of the model specification, not merely a cleaning script.

---

# Part II - Exact and near deduplication

## 9. What duplicates look like and why they matter

**Transcript coverage:** lines 720-851

### What the lecturer said - transcript only

Filtering can leave a high-quality corpus that still contains many duplicates. Exact duplicates arise naturally from mirrored websites. A crawler may not recognize two mirrors as the same source and can collect identical content repeatedly. Repository forks are also largely duplicates even when a few files change; 99% of the fork may remain the same.

Near duplicates contain mostly the same text with small differences, often because of copying or a shared source. Common examples include:

- terms of service and licenses, such as the MIT license, embedded in otherwise different pages;
- repeated site headers and footers;
- articles differing only in punctuation, as in examples from the One Billion Word benchmark;
- templated advertising-like text that substitutes entities such as `Canada` and `USA` while preserving the rest of the passage.

Training on many entity-substituted versions of one low-quality template wastes GPUs. The lecturer emphasizes inspecting concrete records and gives an extreme C4 example: a product description for a gas mask appeared about 61,000 times in the Common Crawl-derived dataset. The web contains strange repetition that an abstract pipeline may not reveal.

Deduplication reduces dataset size without meaningfully discarding information, so it improves training efficiency. It can also reduce memorization, including verbatim reproduction of frequently repeated copyrighted or private content. The lecturer treats saved FLOPs as the main motivation. A closely related operation is decontamination: ensuring that evaluation examples do not occur in training data.

### Source reconciliation

Slide 8 reports the exact C4 count as 61,036 occurrences; the transcript rounds it to 61,000.

### Additional explanation

Duplicates distort the empirical training distribution. If a document appears $k$ times, standard sampling gives it roughly $k$ times the gradient influence of a unique document even when the repetitions contain no new information. Deduplication therefore changes both efficiency and weighting.

The right unit depends on the risk being addressed. Document deduplication catches mirrors but may miss a copied paragraph inside otherwise different pages. Span-level deduplication catches shared boilerplate but can damage coherence. Code needs repository- and file-aware treatment because forks share structure legitimately. Decontamination additionally demands normalization compatible with the benchmark: superficial whitespace or formatting changes should not conceal test leakage.

## 10. Deduplication design choices and hash functions

**Transcript coverage:** lines 852-934

### What the lecturer said - transcript only

A deduplication system makes three groups of choices:

1. **Item granularity:** compare sentences, paragraphs, documents, or another unit.
2. **Match rule:** require exact equality, the presence of a common subitem, or a sufficiently large fraction of common subitems.
3. **Removal policy:** remove every member of a duplicate group or retain one representative.

Filtering scores each item independently and can be parallelized in linear time. Deduplication compares items with other items, so an all-pairs $O(n^2)$ comparison is impossible at web scale. The algorithms must remain close to linear, and the main mechanism is hashing.

A hash function maps a large value, such as a string, to a much smaller string or integer. A collision occurs when distinct inputs receive the same hash value. Cryptographic hashes devote substantial work to collision resistance and are used in applications such as cryptography and Bitcoin. Deduplication can instead use much faster hash-table-oriented functions, where occasional collisions are manageable.

### Source reconciliation

Slide 9 contrasts SHA-256 as a cryptographic example with DJB2, MurmurHash, and CityHash as faster alternatives, and its code example uses `mmh3.hash`. The transcript describes these categories but does not name the fast hash functions.

### Additional explanation

Hashing turns content comparison into key grouping. In a distributed implementation, each worker computes a compact key; records with the same key are shuffled to the same reducer or bucket. This avoids comparing each item with every other item.

Hash collision and semantic collision should not be confused. A **hash collision** is an artifact of the key function. A **candidate collision** in locality-sensitive hashing is intentional evidence that two items may be similar. Systems can reduce accidental hash risk by using sufficiently wide keys or verifying candidate pairs against their original features before deletion.

## 11. Exact deduplication and the coherence problem

**Transcript coverage:** lines 935-981

### What the lecturer said - transcript only

Exact deduplication is conceptually simple: hash each string, group exact matches, and remove all but one. It is transparent and can be expressed in a MapReduce-like form that parallelizes well. Its limitation is that messy web duplicates often differ slightly, so exact equality misses them.

C4 performed exact deduplication on three-sentence spans and retained only one occurrence of a repeated span. This creates a structural problem. When a repeated span occurs in the middle of a document, deleting those three sentences from all but one document can break the surrounding coherence. Nevertheless, that is the procedure the dataset used.

### Additional explanation

Exact deduplication has high precision after canonicalization, but canonicalization itself changes its behavior. Lowercasing, whitespace normalization, Unicode normalization, or stripping markup can expose superficial duplicates; overly aggressive normalization can also merge genuinely different content.

The C4 example illustrates why a match unit and a removal unit need not be identical. One safer design is to use a repeated span as evidence for scoring or rejecting the complete document rather than surgically deleting the span. Another is to retain document boundaries and mark boilerplate regions before language-model sequence construction. Each choice trades data retention against local coherence.

## 12. Jaccard similarity and the MinHash identity

**Transcript coverage:** lines 982-1153

### What the lecturer said - transcript only

Near deduplication first requires a definition of approximate equality. The lecture uses Jaccard similarity for two sets $A$ and $B$:

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

For $A=\{1,2,3,4\}$ and $B=\{1,2,3,5\}$, the intersection is $\{1,2,3\}$ and the union is $\{1,2,3,4,5\}$, giving $J=3/5=0.6$. Jaccard similarity lies between 0 and 1: 0 means disjoint sets and 1 means identical sets. A near-duplicate rule might declare documents duplicates above a threshold such as 0.99.

MinHash connects this similarity to linear-time hashing. It is a random hash construction with the property

$$
\Pr[h_{\min}(A)=h_{\min}(B)]=J(A,B).
$$

Unlike an ordinary hash, where collisions are undesirable, MinHash deliberately makes similar sets collide more often. To compute it, hash every member of a set and retain the minimum hash value. The maximum would work as well; selecting an extreme is simply a deterministic way to identify the first member under the random ordering induced by the hash.

The proof views a random hash as a random permutation of the union. The first union element belongs to both sets exactly when it lies in the intersection. Every union element is equally likely to be first, so the probability that the two sets select the same minimum is the fraction of union elements in the intersection, namely Jaccard similarity.

The lecture checks the claim with 100 independently seeded hashes. The fraction of matching MinHashes for the example is approximately 0.6. Each set's signatures can be computed independently and equal values can be grouped, avoiding an all-pairs scan. One MinHash alone, however, is stochastic and does not reliably establish that Jaccard similarity exceeds a threshold.

### Additional explanation

For documents, the sets are typically shingles, such as normalized word $n$-grams. The shingle width determines what counts as local agreement. Short shingles are robust to edits but create incidental matches; long shingles are more specific but miss reordered or lightly rewritten passages.

With $k$ independent MinHashes, the unbiased estimator

$$
\widehat{J}(A,B)=\frac{1}{k}\sum_{i=1}^{k}
\mathbf{1}\!\left[h_i(A)=h_i(B)\right]
$$

has variance $J(1-J)/k$. Computing this estimate for every pair would still be quadratic. Locality-sensitive hashing uses the signatures to retrieve only plausible pairs.

## 13. MinHash locality-sensitive hashing and its S-curve

**Transcript coverage:** lines 1154-1444

### What the lecturer said - transcript only

A single MinHash collision occurs with probability equal to Jaccard similarity, but it does not directly answer whether similarity exceeds a threshold such as 0.99. Locality-sensitive hashing (LSH) sharpens the probabilistic decision by combining independent MinHashes.

Use $n=br$ hashes and divide them into $b$ bands of $r$ rows. For example, 12 hashes can form 3 bands of 4 hashes. Two items become a candidate pair when **every** hash agrees within **at least one** band. This is an AND within each band and an OR across bands.

If the true Jaccard similarity is $s$, the probability that all $r$ hashes in one fixed band match is $s^r$. The probability that no band matches is $(1-s^r)^b$, so the candidate-collision probability is

$$
P_{\mathrm{candidate}}(s)=1-(1-s^r)^b.
$$

For $s=0.8$, $b=5$, and $r=10$, the lecture reports a collision probability of about 0.4. Plotting this probability against similarity produces an S-shaped transition: very dissimilar pairs rarely become candidates, while sufficiently similar pairs almost always do.

With $b=10$ and $r=10$, the displayed similarities from 0.7 to 0.98 produce candidate probabilities ranging from about 0.25 to nearly 1. Pairs below a desired threshold can therefore still collide as false positives and may need a verification step, while most pairs above it are retained.

Increasing $r$ requires agreement on more hashes in a band. It makes matching harder, moves the transition right, and sharpens it; in the illustrated comparison a low-similarity collision probability falls from 0.25 to 0.008. Increasing $b$ supplies more chances for a band match, moving the transition left and making matching easier; one illustrated probability near similarity 0.9 rises from 0.72 to 0.92. Increasing both can make the transition arbitrarily sharp, but raises computational and storage cost.

A real deduplication setting cited in the lecture used $b=20$ bands and $r=450$ rows, or 9,000 hashes. The approximate transition location is

$$
s_*\approx \left(\frac{1}{b}\right)^{1/r}.
$$

At this location $s_*^r=1/b$, and the candidate probability is

$$
1-\left(1-\frac{1}{b}\right)^b \approx 1-\frac{1}{e}\approx 0.64.
$$

As the transition sharpens, similarities below this region tend toward candidate probability 0 and those above it tend toward 1. The method is called MinHash LSH because LSH is the general framework and MinHash is the particular family that approximates Jaccard similarity for language-model deduplication.

### Source reconciliation

Slides 13-16 confirm the exact AND-OR construction and formulas. For the spoken example $s=0.8$, $b=5$, $r=10$, the formula gives approximately 0.433, so the lecturer's 0.4 is a rounded value. Slide 15 attributes the $n=9{,}000$, $b=20$, $r=450$ setting to Lee et al. (2021). The exact limiting midpoint is about $1-1/e=0.632$, consistent with the spoken rounded 0.64.

### Additional explanation

LSH is primarily a **candidate-generation** mechanism. A production system can place documents sharing a band signature in the same bucket, generate pairs only within those buckets, compute exact Jaccard similarity for those candidates, and then apply the true threshold. This second-stage verification removes many false positives while preserving near-linear behavior when buckets remain controlled.

The parameters encode an operational precision-recall tradeoff:

- larger $r$ reduces candidate volume and false positives but risks missing genuine duplicates;
- larger $b$ improves recall but creates more candidate pairs and verification work.

The threshold approximation locates the S-curve but does not alone determine resource use. Shingle design, document length, skewed common shingles, bucket caps, and connected-component clustering all affect the actual deduplication result.

## 14. Deduplicate across the complete mixture

**Transcript coverage:** lines 1445-1458

### What the lecturer said - transcript only

Deduplication is sometimes run independently within each incoming dataset. That is insufficient because separate datasets can overlap. It should be applied across the complete combined corpus, not only within individual sources. The lecture then moves from per-source transformation, filtering, and deduplication to the problem of mixing sources.

### Additional explanation

Cross-source deduplication also requires a representative policy. Retaining one arbitrary copy may erase useful provenance or keep a lower-quality extraction. A better rule can prefer the copy with clearer licensing, cleaner text, richer metadata, or the most authoritative source. Cross-source overlap statistics are additionally useful for auditing whether supposedly distinct mixture components are genuinely distinct.

---

# Part III - Constructing a data mixture

## 15. Mixture weights balance volume, quality, and diversity

**Transcript coverage:** lines 1459-1579

### What the lecturer said - transcript only

After raw HTML or PDFs have been transformed, filtered, and deduplicated, the result is a smaller set of high-quality documents for each source. A language model is trained on many such sources. The Marin source viewer, for example, includes Nemotron, FinePDFs, books, code, and other components. The Pile similarly assigns an explicit weight to each of its components. A data mixture is therefore a probability distribution over sources.

Several baseline strategies are common:

- **Manual or "vibes-based" weights:** choose values from intuition and adjust them. The lecturer says this is more common, even in recent work, than one might expect.
- **Uniform sampling:** give every source the same probability.
- **Size-proportional sampling:** make a source's probability proportional to its token count.

Size-proportional sampling is defensible, but a huge low-quality source can consume most of the training budget. Higher-quality sources should intuitively receive more weight. Two constraints complicate that intuition. First, diversity matters: literature, code, and papers may be incomparable rather than totally rankable, and a general model should not place all probability on one type. Second, every source is finite. Too much weight on a small source exhausts it and forces repeated epochs over the same tokens.

### Additional explanation

For sources $s=1,\ldots,m$, a mixture is a simplex vector

$$
p_s\ge 0,\qquad \sum_{s=1}^{m}p_s=1.
$$

The vector is not merely a bookkeeping choice. It defines the expected gradient distribution and therefore the model's capability allocation. A source can be upweighted because its examples are cleaner, because its domain is strategically important, or because it is underrepresented relative to desired use. These rationales should be distinguished: a single scalar "quality" score cannot fully express a multi-capability objective.

It is useful to report both probability and implied token demand. A 5% weight can be conservative for a trillion-token source but extreme for a million-token source.

## 16. Mixture weights imply source-specific epoch counts

**Transcript coverage:** lines 1580-1731

### What the lecturer said - transcript only

Consider two sources. The low-quality source contains 10 trillion tokens and the high-quality source only 10 billion. Suppose a run trains for 1 trillion token presentations with a uniform 50/50 mixture. "One trillion training tokens" counts tokens consumed by the optimizer, not necessarily unique tokens.

The low-quality source is asked to provide 500 billion tokens, only 5% of its 10-trillion-token pool. The high-quality source is also asked to provide 500 billion tokens, but it contains only 10 billion unique tokens. Every high-quality token must therefore be repeated about 50 times.

This is not a hypothetical bookkeeping detail: some large runs have missed it by selecting weights from perceived quality without checking source size. An audience member asks why 50 epochs are needed if performance saturates earlier. They are not needed. In the best case they merely waste compute; in the worst case they cause overfitting. They occur automatically when the requested mixture mass exceeds the source supply. The lesson is to audit the epoch count implied by every mixture.

Another audience member asks how a mixture is realized during training. In general, the loader fills a batch by sampling a component for each sequence or batch element. A sequence normally comes from one component; sources are not resampled token by token. A batch should contain a mix of sources rather than dedicating a complete optimizer step to one source, because mixing within the batch reduces variance.

### Additional explanation

If source $s$ has $N_s$ unique tokens, the run consumes $D$ tokens total, and the mixture weight is $p_s$, its implied epoch count is

$$
e_s=\frac{p_sD}{N_s}.
$$

For the example:

$$
e_{\mathrm{low}}=\frac{0.5\times10^{12}}{10^{13}}=0.05,
\qquad
e_{\mathrm{high}}=\frac{0.5\times10^{12}}{10^{10}}=50.
$$

This equation should accompany every mixture proposal. If documents vary in length or are packed, actual repeat patterns can differ from the token-level expectation, so loaders should also measure observed document and span reuse.

Mixed batches make the stochastic gradient closer to the intended mixture at each step. Source-homogeneous steps can still have the right long-run expectation, but introduce high step-to-step variance and can interact with optimizer state, normalization, or scheduling.

## 17. UniMax uses a hard epoch cap

**Transcript coverage:** lines 1732-1774

### What the lecturer said - transcript only

Multilingual training made finite-source imbalance especially visible because low-resource languages have far fewer tokens. Earlier work often flattened size-proportional weights by raising source sizes to a power. UniMax makes the protection explicit: sample sources uniformly, but impose a hard maximum on the number of epochs over any source.

In the lecturer's example, a source may be used for at most 20 epochs. Once it reaches that cap, it supplies no more tokens and training allocation moves elsewhere. This acts as a safety net against unnoticed oversampling. The source probability multiplied by the total training-token budget must remain within the amount allowed by the cap, and a simple procedure constructs weights subject to this restriction.

### Source reconciliation

Slide 19 states the earlier temperature-style rule as $p(s)\propto N_s^\alpha$ for $\alpha\in[0,1]$. Its UniMax constraint is visually abbreviated as `p(s) * num_training_tokens <= C`; the spoken explanation makes clear that $C$ is the source-specific allowance implied by the epoch cap. With source size shown explicitly, the constraint is $p_sD\le C N_s$ when $C$ denotes the maximum number of epochs.

### Additional explanation

An epoch cap converts a hidden consequence into an explicit constraint:

$$
p_sD\le C_sN_s
\quad\Longleftrightarrow\quad
e_s\le C_s.
$$

Uniform preference plus caps can be implemented by allocating equally among still-eligible sources, saturating small sources at their maximum, then redistributing unused mass among the rest. The cap does not prove that $C_s$ repetitions are beneficial; it only prevents a mixture from exceeding a chosen tolerance.

## 18. Regression-based mixing learns a small-scale surrogate

**Transcript coverage:** lines 1775-1984

### What the lecturer said - transcript only

With perhaps 50 sources, selecting 50 mixture weights by hand is difficult. Weighting by a separate quality estimate remains heuristic. Regression-based methods such as RegMix and OLMix provide a more systematic framework:

1. choose many candidate mixtures over the source components;
2. train a swarm of small proxy models, with the lecturer giving about 300M parameters as an example;
3. evaluate each proxy on a target such as validation loss, perplexity, or downstream evaluations;
4. fit a regression from mixture weights to the measured target;
5. optimize the cheap regression surrogate and use its predicted best mixture for the large run.

This resembles scaling laws: cheap experiments are used to make a decision at expensive scale. Important design choices include how candidate mixtures are sampled, often with a Dirichlet distribution; whether the regression is linear, log-linear, or a boosted tree; which target is optimized; and how far proxy scale is from final scale.

Downstream targets can cause objective overfitting. If the target suite contains many code evaluations, the optimizer will unsurprisingly put more mass on code. The resulting model may then be worse at an unmeasured use such as poetry. Uniform and proportional mixing avoid this specific failure because they never inspect downstream evaluations. Proxy size also presents a cost-accuracy tradeoff: very small models may not be representative, while proxy models near full scale defeat the purpose by making mixture search as expensive as the final run.

The OLMix paper compares methods within this common framework. Proxy models are often in the tens of millions of parameters; the number of swarm runs varies with the number of domains; sampling may be Dirichlet or exponential; log-linear regression often works well; and optimization can use search, gradient methods, or an exact solver.

Two leaps of faith remain. First, the regression may predict ordinary sampled mixtures well but fail at the extreme mixture found by optimization, where training coverage is sparse. Second, the mixture that minimizes loss for small proxies may not minimize it at large scale. At scales available to the open community this transfer appears plausible rather than plainly false, but scale-dependent data effects clearly exist. In particular, longer training can make lower-quality data more useful, so the true optimum need not be invariant.

### Source reconciliation

Slide 21 gives exact OLMix comparison entries that the transcript summarizes. Across the displayed methods, proxy sizes span 1M to 410M parameters; regression families include LightGBM, log-linear, power-law, and Gaussian-process models; and optimization includes search, gradient descent, exact solvers, and an exact solver with KL regularization. OLMixBase is the displayed method that explicitly includes data-repetition constraints. These are slide-only table details, not additional spoken claims.

### Additional explanation

Let $L_{\mathrm{proxy}}(p)$ be the target measured after training a proxy on mixture $p$, and let $\widehat L(p)$ be a fitted surrogate. The method chooses

$$
p^*=\arg\min_{p\in\Delta^{m-1}}\widehat L(p),
$$

possibly subject to repetition and minimum-diversity constraints. This is a form of experimental design followed by black-box optimization.

The first leap of faith is an **optimization-distribution shift**: minimizing a learned function deliberately seeks regions where prediction errors can be exploited. Cross-validation on randomly held-out mixtures does not fully test accuracy at the optimizer's solution. Useful safeguards include bounds on each weight, regularization toward a baseline, repeated proxy runs near the predicted optimum, and trust-region iterations.

The second leap is a **scale transfer** assumption. It can be tested by fitting at several proxy scales and checking whether rankings or optima stabilize. Even then, the final run may have a different token budget, and therefore different repetition regime, from all proxies.

## 19. Epoch-aware scale transfer and within-source mixing

**Transcript coverage:** lines 1985-2185

### What the lecturer said - transcript only

The finite-source effect creates a known failure in proxy mixture search. With 10 trillion low-quality tokens and only 10 billion high-quality tokens, a small-token proxy may favor a mixture heavily concentrated on the high-quality source because it does not repeat it. Applying that mixture to a much longer large run would repeatedly epoch the small source and overfit.

One remedy, used in OLMix, is to cap epochs. Another is **simulated epoching**, whose guiding principle is to make the small experiment resemble the large run. The lecturer connects this to muP: transfer improves when the small and large regimes are parameterized to have corresponding behavior.

Suppose proxy runs train on 10B tokens and the final run trains on 1T, a ratio of 1:100. Downsample every source to 1/100 of its original size for the proxy. A scarce source then becomes proportionally scarce at small scale and is repeated just as it would be in the final run. Mixtures overconcentrated on that source receive poor proxy loss, encouraging a more balanced optimum. This simulates large-scale data scarcity without actually training the large model.

The section's main recipe is to fit a small-scale mapping from mixture weights to loss, optimize it, and transfer the result cautiously. Epoching and overfitting must be addressed through a cap or simulated epoching. More generally, optimizing any measurable target risks optimizing the wrong thing.

An audience member asks whether proportional downsampling can make a source too small to support generalization. It can. In principle the optimum may assign it correspondingly little mass, but discrete rounding can accidentally turn one intended pass into zero. A workaround is to ensure that the source is trained on at least once.

Another question asks whether mixing can be applied within one diverse dataset rather than only across named sources. The Nemotron work does this with Common Crawl: group pages by topic or domain, also divide them by quality, and treat the resulting two-dimensional grid cells as mixture components. Hand-collected sources can then be added as further components.

### Source reconciliation

Slide 22 writes the illustrative source sizes as 10T low-quality and 10B high-quality tokens, the small and large budgets as 10B and 1T, and the proportional downsampling ratio as $10^{-2}$. Its made-up mixture shifts from 0.1 low / 0.9 high before simulated epoching to 0.7 low / 0.3 high afterward. The transcript explains the mechanism but does not assert these example weights as a measured result.

### Additional explanation

If the target run's data-to-training ratio is $D_{\mathrm{small}}/D_{\mathrm{large}}=\rho$, simulated epoching sets

$$
N_{s,\mathrm{small}}=\rho N_{s,\mathrm{large}}.
$$

Then the proxy and final epoch counts match for any fixed mixture:

$$
\frac{p_sD_{\mathrm{small}}}{N_{s,\mathrm{small}}}
=
\frac{p_sD_{\mathrm{large}}}{N_{s,\mathrm{large}}}.
$$

This alignment preserves one important scale-dependent variable, not every aspect of scale. Model capacity, optimization dynamics, and capability priorities can still change. It is therefore best understood as removing a predictable confound from the proxy study.

Splitting a broad crawl into domain-quality cells makes the mixture space more expressive, but also higher-dimensional. Sparse or correlated cells can make regression unstable, so taxonomy design, minimum weights, and interpretability become part of the method.

---

# Part IV - Synthetic and semi-synthetic post-training data

## 20. Post-training data is task-specific and usually teacher-generated

**Transcript coverage:** lines 2186-2240

### What the lecturer said - transcript only

The preceding pipeline applies mainly to pretraining and perhaps mid-training. Its examples are relatively task-agnostic and intended to build basic skills, even when RegMix uses a target loss to choose proportions. Post-training data is much more task-dependent.

The lecturer gives a general post-training recipe:

1. define environments, such as GitHub repositories for software tasks;
2. define tasks or prompts in those environments;
3. collect responses from a strong teacher model.

In the open community, most post-training data is synthetic. A human can replace the teacher model, but human response collection is slow and expensive. Frontier laboratories formerly paid large human workforces for responses and can now use hybrid human-AI processes. In every case, some teacher supplies the response that the student model will imitate or learn from.

### Additional explanation

Post-training example construction has three independent axes:

- **environment provenance:** synthetic, curated real, or naturally occurring real environments;
- **task provenance:** authored by humans, generated by a model, or mined from behavior such as pull requests;
- **response provenance:** human demonstrations, teacher-model trajectories, verified solutions, or hybrids.

Calling a dataset "synthetic" can obscure these distinctions. A generated answer to a real human problem is different from a generated task inside a real repository, and both differ from a fully generated problem with no executable environment.

## 21. OpenThoughts: source selection, multiple answers, and teacher quality

**Transcript coverage:** lines 2241-2326

### What the lecturer said - transcript only

OpenThoughts was motivated by the attention to reasoning after o1, especially for mathematics and science. It ultimately released 1.2 million teacher-generated examples. Questions and tasks came from many sources: human-origin sources such as Stack Exchange and NuminaMath, synthetic sources, and collections spanning code, math, and chemistry. The lecturer notes that it was a large collaborative project and that its code sources ranged from artificial exercises such as CodeGolf to more realistic work such as code review.

The project's analysis produced several non-obvious findings:

- using a smaller selection of sources could be better than using every available source;
- sampling multiple generations per prompt, such as 16, was useful;
- a model that is stronger overall is not necessarily a better teacher - QwQ-32B outperformed the then-stronger DeepSeek-R1 as a teacher in this setting;
- basic answer filtering did not help.

The pipeline combines source datasets, processes the question pool, samples a smaller set of questions, and generates multiple answers for each. The transcript appears to say that questions are "duplicated," but the surrounding discussion treats this as a deduplication stage. Because the final 1.2M count includes roughly 16 responses per question, the number of distinct underlying questions is approximately $1.2\text{M}/16=75{,}000$.

### Source reconciliation

Slides 23-25 resolve and extend several points:

- Slide 23 explicitly names QwQ-32B as the teacher and says the questions came from 27 human and synthetic sources.
- Slide 25 labels the ambiguous stage **Deduplicate Questions**, confirming that the transcript's "duplicate" wording is a transcription error. It also summarizes the observed source result as smaller high-quality sources, such as OpenMath-2-Math, being better than a large diverse collection.
- Slide 24 catalogs code-question sources and counts not spoken in detail: StackExchange CodeGolf (85.9K), OpenCodeReasoning (459K), `cognitivecomputations/dolphin-coder` (101K), `m-a-p/CodeFeedback-Filtered-Instruction` (150K), KodCode-V1 (384K), McEval-Instruct (35.8K), `christopher/rosetta-code` (75.4K), `glaive-code-assistant-v3` (946K), StackExchange CodeReview (183K), and Coder-Stat (41.9K), plus the OpenCoder SFT mixture without a displayed question count.

These slide details document the source pool; they are not presented in the transcript as claims about how many examples from each source entered the final 1.2M set.

### Additional explanation

Generating multiple responses separates **task diversity** from **solution diversity**. Sixteen trajectories for one problem can teach alternative reasoning paths, error patterns, or implementation styles, but they do not replace sixteen independent problems. Dataset reports should therefore state both prompt count and response count.

Teacher quality is conditional on the teaching objective. A very capable model may produce terse, stylistically inconsistent, overly advanced, or hard-to-distill traces. A smaller teacher may generate clearer demonstrations that better match the student's capacity. Teacher selection should be evaluated by the student's learning outcome, not only by the teacher's benchmark score.

The failed basic answer filter is also instructive: a filter can remove diversity or use a proxy that is weakly related to educational value. Every extra selection stage should be validated by an ablation rather than assumed to improve data.

## 22. SWE-smith creates verified software tasks from repositories

**Transcript coverage:** lines 2327-2359

### What the lecturer said - transcript only

Recent work increasingly targets **agentic coding**: not merely generating a code snippet, but carrying out software-development work in a repository. SWE-smith starts from a repository and uses an LM to generate tasks. An agent first makes the repository usable, including installing dependencies. Task generation then modifies code, possibly introducing bugs, and verifies the resulting task instances. The tasks are synthetic but grounded in real software environments. The pipeline produced 50,000 tasks, which was large for this kind of dataset when released the previous year.

### Source reconciliation

Slide 26 says that 128 GitHub repositories yielded the 50K tasks. Its diagram distinguishes environment creation from task-generation strategies such as procedural modification, LM generation, combining bugs, and pull-request mirroring, followed by verified tests. The transcript describes the overall pipeline but does not state the repository count or enumerate all four strategies.

### Additional explanation

Repository-grounded task generation must establish both a problem and an oracle. Introducing a change is easy; producing a task with a reproducible failing state, a valid repair, and tests that distinguish correct from superficial fixes is harder. Verification filters out tasks whose environment does not build, whose tests are flaky, whose bug is trivial, or whose stated requirement does not match the test behavior.

This makes environment preparation part of data quality. Dependency pinning, build scripts, test commands, and container metadata determine whether the same example remains usable later.

## 23. SWE-Zero, SWE-Hero, SWE-rebench, and scaling software trajectories

**Transcript coverage:** lines 2360-2496

### What the lecturer said - transcript only

Software-engineering tasks have much heavier environment requirements than math or isolated coding problems. Many GitHub repositories no longer run cleanly, dependencies are outdated, and reconstructing the state around an old pull request can be an infrastructure nightmare.

SWE-Zero exploits the observation that strong models can solve many repository tasks without executing code. In the cited comparison, allowing execution yields scores around 80 for a strong model, while forbidding execution still gives almost 70. This suggests that models contain a useful internal model of code semantics even without test feedback.

The work generated 300,000 agent trajectories from real GitHub pull requests and SWE-smith examples using the OpenHands scaffold. The execution-free instruction forbids Python and testing, permitting only basic shell-style inspection and editing operations such as `sed` and `grep`. Considerable care is required to prevent agent hacking. The trajectories were distilled from a large Qwen model, then filtered because the teacher sometimes ignored the restrictions and attempted execution anyway.

The authors also collected 13,000 agent trajectories that do require execution feedback. The transcript says models were first fine-tuned on SWE-Zero examples and then "fine-tuned again on these SWE-Zero examples." That repeated name is ambiguous in the spoken record, although the surrounding comparison distinguishes the 300K execution-free set from the 13K execution-based set. The resulting models made progress but remained below the strongest frontier systems in the displayed comparison.

SWE-rebench is another pipeline that gathers large numbers of GitHub pull requests, tries to install and test their repositories, retries the many failures, and uses an LM to supply responses for usable tasks. The lecturer then describes work released that same day that scales the lightweight, execution-free SWE-Zero idea to 12 million agent trajectories. SWE-rebench recovered only 32,000 executable tasks and a much larger group that did not execute; SWE-Zero can use both groups because it does not require repository-specific execution.

These examples show a progression from environment-free mathematical reasoning to increasingly sophisticated coding data. Prompts may be fully synthetic, semi-synthetic with a real environment and synthetic task, or real. Responses generally come from capable teacher models, but those models must also be effective teachers. Code environments remain painful, and the short survey omits substantial filtering and engineering detail.

### Source reconciliation

Slides 26-29 clarify exact values and one likely transcript error:

- Slide 26 reports execution and no-execution results for several models. For MiniMax-M2.5, for example, SWE-bench Verified is 80.2 with execution and 69.5 without it; SWE-bench Multilingual is 74.1 with execution and 57.2 without it. This supports the spoken rounded comparison but shows that the amount of degradation varies by model and benchmark.
- Slide 27 says the 300K trajectories originate from 150K GitHub pull requests. It contrasts the standard OpenHands workflow with the SWE-Zero execution-free workflow and explicitly prohibits Python commands and test execution.
- Slide 28 labels the 13K execution-requiring trajectories **SWE-Hero**. This strongly suggests that the transcript's second "SWE-Zero" in the two-stage fine-tuning sentence should be SWE-Hero, but the transcript alone does not establish the intended training order beyond doubt.
- Slide 29 gives SWE-rebench's exact pipeline scale: 21K interactive Python SWE tasks from 3.4K repositories, starting from 450K pull requests from GitHub and GH Archive. For SWE-Zero-12M, it says **32K executable tasks + 120K nonexecutable tasks**. The transcript's "120 ... didn't execute" is therefore missing the `K`, almost certainly through ASR loss. The slide also identifies the small model as mini-coder-1.7B and reports 50.4 pass@100 using a mini-swe-agent scaffold; these details are not spoken.

### Additional explanation

Execution-free trajectories exchange one supervision source for scale. They can cover repositories that cannot be reproduced, but they cannot learn directly from compiler errors, test failures, or runtime behavior. A two-stage design is therefore sensible: use abundant execution-free traces for broad repository reasoning, then use scarcer executable traces to teach feedback-driven correction.

Preventing "agent hacking" is a data-validity problem. If the teacher uses prohibited tools, its answer no longer demonstrates the intended capability. Filtering only the final text may miss hidden violations; the full action trace must be checked against an allowlist and against environment state.

Real pull requests improve task realism but can leak future information. An agent that can inspect later commits, merged fixes, or generated artifacts may recover the answer rather than solve the task. Repository snapshots and tool permissions must therefore enforce the information boundary represented by the benchmark.

## 24. Closing synthesis and the reality of data work

**Transcript coverage:** lines 2497-2531

### What the lecturer said - transcript only

The lecture closes with four main points:

1. Define desired data and train a lightweight classifier to find a matching subset of a web crawl.
2. Deduplicate to save FLOPs and reduce overfitting.
3. Try data mixtures at small scale and extrapolate cautiously to large scale.
4. Post-training datasets increasingly use carefully constructed synthetic examples.

The lecturer warns that real data work is grungy, domain-specific, and driven by inspection of concrete examples. The lecture's abstract overview does not represent the amount or character of labor required to build a high-quality dataset; its purpose is to map the landscape.

### Additional explanation

The repeated methodological lesson is to connect aggregate methods to individual records. A classifier score, MinHash collision, mixture optimum, or synthetic-data filter can look clean mathematically while failing for a recurring template, language, repository, or task family. High-quality data pipelines combine scalable algorithms with provenance, audits, ablations, and domain expertise.

---

# Consolidated takeaways

1. Raw web artifacts must be transformed into sequences, and that transformation inevitably chooses which structural and visual information to preserve.
2. Filtering is target-relative: language, toxicity, general prose quality, mathematical relevance, and educational value are different objectives.
3. Cheap target language models or discriminative classifiers let a small trusted set guide selection from an enormous raw pool.
4. A filter threshold is inseparable from the planned training-token budget because stricter filtering reduces unique data and increases repetition.
5. Deduplication saves compute, corrects accidental example weighting, reduces memorization risk, and supports test-set decontamination.
6. Jaccard similarity defines set overlap; MinHash turns it into a collision probability; banded LSH turns many MinHashes into efficient near-duplicate candidates.
7. Deduplication must span the combined corpus because independently collected sources can overlap.
8. Mixture weights define both capability allocation and source-specific epoch counts. A source's finite token count must be audited alongside its perceived quality.
9. UniMax caps repetition explicitly, while regression-based methods use proxy-model swarms to learn a mixture-to-loss surrogate.
10. Proxy mixture optimization has two transfer risks: the fitted surrogate can fail at its optimized extreme, and a small-scale optimum can change at large scale.
11. Epoch caps or simulated epoching make proxy experiments better reflect the data-scarcity regime of the final run.
12. Post-training data combines environments, tasks, and teacher responses; each can be real, synthetic, or hybrid.
13. Multiple teacher answers increase trajectory diversity but do not increase the number of distinct prompts.
14. Software-agent data is constrained by environment reproducibility, tool permissions, task verification, and leakage prevention.
15. Scalable algorithms do not replace record-level inspection. Data quality remains domain-specific engineering work.

# Key equations

## Mixture simplex

$$
p_s\ge 0,
\qquad
\sum_{s=1}^{m}p_s=1.
$$

$p_s$ is the probability of drawing a training sequence from source $s$.

## Implied epoch count

$$
e_s=\frac{p_sD}{N_s},
$$

where $D$ is the total number of consumed training tokens and $N_s$ is the number of unique tokens in source $s$.

## Epoch-cap constraint

$$
p_sD\le C_sN_s
\quad\Longleftrightarrow\quad
e_s\le C_s.
$$

$C_s$ is the maximum permitted number of epochs over source $s$.

## Jaccard similarity

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

## MinHash collision identity

$$
\Pr[h_{\min}(A)=h_{\min}(B)]=J(A,B).
$$

## Empirical MinHash estimator

$$
\widehat J(A,B)=\frac{1}{k}\sum_{i=1}^{k}
\mathbf{1}\!\left[h_i(A)=h_i(B)\right].
$$

## Banded LSH candidate probability

For similarity $s$, $b$ bands, and $r$ rows per band:

$$
P_{\mathrm{candidate}}(s)=1-(1-s^r)^b.
$$

## Approximate LSH transition

$$
s_*\approx\left(\frac{1}{b}\right)^{1/r},
$$

and at that point

$$
P_{\mathrm{candidate}}(s_*)
=1-\left(1-\frac{1}{b}\right)^b
\approx 1-\frac{1}{e}.
$$

## Regression-mixture optimum

$$
p^*=\arg\min_{p\in\Delta^{m-1}}\widehat L(p),
$$

where $\widehat L$ is a surrogate fitted from proxy training runs.

## Simulated epoching

For $\rho=D_{\mathrm{small}}/D_{\mathrm{large}}$:

$$
N_{s,\mathrm{small}}=\rho N_{s,\mathrm{large}},
$$

which preserves the implied epoch count at fixed $p_s$.

# Glossary

| Term | Meaning in this lecture |
|---|---|
| Boilerplate | Repeated page material such as navigation, headers, footers, menus, and advertisements. |
| Candidate pair | A document pair retrieved by LSH for possible exact similarity verification. |
| Data mixture | A probability distribution determining how often training sequences come from each source. |
| Decontamination | Removing training examples that overlap with evaluation or test data. |
| Deduplication | Detecting repeated or nearly repeated content and removing some or all redundant copies. |
| Discriminative filter | A classifier trained to separate target examples from raw-background examples. |
| Epoch | One complete pass over a finite source; mixture sampling can create fractional or repeated epochs. |
| fastText | A fast linear text-classification toolkit commonly used for language, quality, or domain filters. |
| Generative filter | A target-domain language model whose likelihood or perplexity scores candidate documents. |
| Jaccard similarity | Intersection size divided by union size for two sets. |
| KenLM | A toolkit for efficient $n$-gram language models, used here as a cheap target model. |
| Locality-sensitive hashing (LSH) | A hash-family framework in which similar items collide more often, enabling efficient candidate retrieval. |
| MinHash | A random set signature whose pairwise collision probability equals Jaccard similarity. |
| Near duplicate | Content that is highly similar but not byte-for-byte identical. |
| OCR | Optical character recognition, used to recover text from scanned or visually rendered pages. |
| Proxy model | A relatively small model trained to predict or choose a configuration for a more expensive final run. |
| RegMix | The general regression-based approach of fitting mixture weights to proxy-model outcomes. |
| Simulated epoching | Proportionally shrinking proxy data sources so their repetition rates match a longer final run. |
| Synthetic data | Examples whose tasks, answers, trajectories, or some combination are produced by a model or program. |
| Teacher model | A capable model that generates labels, answers, or trajectories for training another model. |
| UniMax | A mixture strategy that prefers uniform source sampling while enforcing a hard repetition cap. |
| WARC | A web-archive file format used to store crawled web responses and metadata. |

# Self-check questions

1. Why is HTML-to-text conversion lossy even when every visible word is retained?
2. How do a target generative model and a discriminative target-versus-raw classifier differ?
3. Why can a high language-identification threshold mishandle code-switching or dialectal text?
4. What does the OpenMathText example show about the meaning of data quality?
5. Why can an expensive teacher label only a sample and still support filtering of an entire crawl?
6. Why does the best filter threshold depend on the final training-token budget?
7. How can an almost unfiltered dataset eventually outperform a small, high-quality dataset in a very long run?
8. What are the efficiency, memorization, and evaluation reasons for deduplication?
9. Why can deleting a repeated three-sentence span damage a document even when the span is an exact duplicate?
10. Prove that one MinHash collision has probability equal to Jaccard similarity.
11. Why does a single MinHash collision not establish that similarity exceeds 0.99?
12. Derive $1-(1-s^r)^b$ for banded LSH.
13. How do increasing $r$ and increasing $b$ affect the LSH S-curve?
14. Why should deduplication be run across sources rather than independently within each one?
15. For a source with $N_s$ tokens, how many epochs does weight $p_s$ imply in a $D$-token run?
16. Why can a manually reasonable 50/50 mixture repeat a small source dozens of times?
17. What does a UniMax epoch cap protect against, and what does it not guarantee?
18. What are the two main leaps of faith in regression-based mixture optimization?
19. How does simulated epoching make a small proxy run resemble a longer final run?
20. Why might optimizing code-heavy downstream evaluations harm a general pretraining mixture?
21. What is the difference between prompt count and response count in OpenThoughts?
22. Why might a nominally stronger model be a worse teacher?
23. What capabilities can execution-free software trajectories teach, and what feedback do they omit?
24. Why must a software-agent data pipeline prevent access to future commits or prohibited tools?

# Source coverage checklist

| Transcript span | Topic | Covered above |
|---:|---|:---:|
| 1-37 | Prior data lecture, legal context, and today's pipeline | Yes |
| 38-129 | HTML transformation, boilerplate, tables, and extraction tools | Yes |
| 130-205 | PDF retrieval, OCR, layout loss, cleanup, and FinePDFs | Yes |
| 206-352 | General filtering objective, applications, models, and compute motivation | Yes |
| 353-456 | Language identification and OpenMathText | Yes |
| 457-525 | GPT-3, LLaMA, phi-1, and toxicity filters | Yes |
| 526-684 | Token-budget-dependent thresholds, epoching plot, and audience questions | Yes |
| 685-719 | Filtering summary and teacher-to-cheap-classifier recipe | Yes |
| 720-851 | Exact and near duplicates, examples, motivations, and decontamination | Yes |
| 852-934 | Deduplication design space, complexity, and hash families | Yes |
| 935-981 | Exact deduplication, MapReduce form, and C4 span coherence | Yes |
| 982-1153 | Jaccard similarity, MinHash construction, proof, and simulation | Yes |
| 1154-1444 | Banded MinHash LSH, probability, parameter effects, and transition | Yes |
| 1445-1458 | Cross-source deduplication | Yes |
| 1459-1579 | Source mixtures, baseline weights, diversity, and finite sources | Yes |
| 1580-1731 | Epoch-count example, overfitting warning, and batch realization | Yes |
| 1732-1774 | UniMax and hard epoch caps | Yes |
| 1775-1984 | Regression-based mixing, design choices, and transfer assumptions | Yes |
| 1985-2185 | Epoch-aware transfer, simulated epoching, and audience questions | Yes |
| 2186-2240 | Task-specific post-training recipe and teacher responses | Yes |
| 2241-2326 | OpenThoughts sources, multiple generations, teachers, and filtering | Yes |
| 2327-2359 | SWE-smith repository-grounded task generation | Yes |
| 2360-2496 | SWE-Zero, SWE-Hero, SWE-rebench, and 12M trajectories | Yes |
| 2497-2531 | Lecture summary and warning about real data work | Yes |
