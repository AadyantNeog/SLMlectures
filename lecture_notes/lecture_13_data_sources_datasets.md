---
title: "Lecture 13 - Data I: Sources and Datasets"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 13
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 13 Data (Sources, Datasets).txt"
slide_deck: "../lecture_13.pdf"
status: "complete"
---

# Lecture 13: Data I - Sources and Datasets

## How to read these notes

Each substantive topic is divided into two layers:

1. **What the lecturer said - transcript only.** This is a concise paraphrase of the spoken lecture. It preserves substantive claims, examples, qualifications, numerical values, and audience questions while removing filler and repetition.
2. **Additional explanation.** This adds independent organization, intuition, caveats, or study guidance. It is not presented as part of the lecture transcript.

The raw transcript's physical line spans are shown for auditability. The complete 33-slide deck was inspected after the transcript map was established. Slides were used to verify names, dates, dataset sizes, tables, and pipeline diagrams. Material differences are recorded under **Source reconciliation**.

> **Legal-snapshot warning:** The copyright discussion below records what was said in this Spring 2026 lecture and explains the lecturer's framework. It is not current legal advice. Copyright duration, fair use, licensing, terms of service, and case status depend on jurisdiction, facts, and time.

## Lecture map

The lecture follows data from origin to model-ready dataset:

1. Explain why data is a central but unusually secret part of language-model development.
2. Distinguish the live web from the material a crawler can actually obtain, including technical, contractual, copyright, and ethical constraints.
3. Introduce copyright, licensing, fair use, and the then-current legal disputes around model training.
4. Survey important raw sources: Common Crawl, Wikipedia, GitHub, Software Heritage, and arXiv.
5. Trace the evolution of major pretraining datasets from BERT and WebText through DCLM, Nemotron-CC, The Stack, and Common Pile.
6. End with the central lesson: raw availability does not make a training dataset. Selection, extraction, filtering, deduplication, mixture design, and provenance determine what the model actually learns.

This is **Data I**. The next lecture continues with data processing and filtering in greater depth.

---

# Part I - Why data is difficult

## 1. Data as the hidden part of the training recipe

**Transcript coverage:** lines 1-160

### What the lecturer said - transcript only

Data may be the most important part of a language-model system to get right. Model developers often publish architecture and training details yet reveal much less about their data. Llama 3 is given as an example: its public documentation says much about the model but comparatively little about exactly what it was trained on.

The lecturer offers two main reasons for this secrecy. First, companies regard data composition and processing as competitive "secret sauce." Second, precise disclosure may expose copyright or licensing risk. This is a major change in emphasis from older machine learning. Before foundation models, dataset creation was often equated with direct human annotation. Pretraining now needs less label-by-label annotation, but it still requires substantial human effort in collection, curation, cleaning, policy design, and evaluation. That long tail of judgment creates a bottleneck and supports large data teams.

The lecture separates model development into three approximate stages:

| Stage | Typical data described in the lecture | Broad purpose |
|---|---|---|
| Pretraining | Very large collections of relatively low-quality web documents | Learn broad language and world regularities |
| Mid-training | Smaller, higher-quality capability data; long-context, synthetic, and instruction-like material | Refine capabilities and prepare the base model for later use |
| Post-training | Chat demonstrations, reinforcement-learning environments, task-specific data, reasoning, code, and safety data | Shape behavior and produce an instruct or chat model |

These boundaries are blurry. A model after pretraining and mid-training is often called a **base model**, while the post-trained result is called an **instruct** or **chat model**. The terminology is becoming less reliable because some developers expose only a final model and use different names for intermediate stages. Open OLMo is highlighted as a more transparent project that releases intermediate artifacts and data information.

The lecture's organizing questions are therefore: Where does raw data come from? How is it selected? How is it processed into something suitable for training?

### Source reconciliation

Slides 4-6 provide concrete OLMo and Tulu 3 mixture tables that the spoken lecture does not enumerate row by row. They illustrate the scale contraction described in speech: trillions of tokens for pretraining, hundreds of billions for mid-training, and a much smaller collection of prompts for supervised and preference-based post-training. Those table entries are slide context, not transcript-derived details.

### Additional explanation

It is useful to treat the model recipe as a joint object:

```text
model behavior = architecture + optimization + data distribution + training stages
```

Two teams can use nearly identical transformers and obtain meaningfully different models because their data distributions differ. Data decisions determine which languages, domains, styles, abilities, errors, and social assumptions receive training weight. The dataset is therefore not merely fuel for a fixed algorithm; it is part of the specification of the learned system.

The stage boundaries are best understood by function rather than by file format. Synthetic question-answer pairs could appear in mid-training or post-training depending on whether the goal is to expand a base capability or align interactive behavior.

## 2. "The Internet" is not a downloadable dataset

**Transcript coverage:** lines 161-294

### What the lecturer said - transcript only

People often say that language models are trained on "the entire Internet." The lecturer calls this slightly more accurate than saying only "the web," but still wrong. The Internet consists partly of live services. Ordinary pretraining does not continuously interact with those services; it trains on stored artifacts. A model can browse later if placed inside an agent, but that is different from the pretraining dataset.

A web crawler begins with seed URLs, discovers links, downloads reachable pages, and expands through the link graph. Even a large crawler cannot obtain everything:

- Modern sites may be dynamic applications whose content appears only after button presses, forms, or client-side interactions. Discord and Weights & Biases are mentioned as examples.
- Pages may require authentication or payment. Facebook, X, LinkedIn, and The New York Times illustrate large walled gardens.
- A platform owner may train on its internal data even when an outside crawler cannot reach it.
- Some content is not connected by discoverable public links, changes rapidly, or exists in formats a crawler does not understand.

Thus, a web-derived corpus is a selective snapshot of accessible artifacts, not a neutral copy of all online human activity.

### Additional explanation

The phrase "trained on the Internet" collapses at least four transformations:

```text
live services and files
    -> crawlable URLs
    -> successfully downloaded responses
    -> extracted documents
    -> retained training examples
```

Each arrow changes the distribution. Search visibility, link structure, geography, platform policy, crawler design, extraction quality, and filters all decide what survives. A later statement such as "web data" should therefore prompt two questions: **which snapshot?** and **which processing pipeline?**

## 3. Technical, contractual, and consent restrictions

**Transcript coverage:** lines 295-429

### What the lecturer said - transcript only

Web access is constrained even when a page has a public URL. The lecturer distinguishes several mechanisms:

- A site's `robots.txt` file can ask automated agents not to crawl particular paths or can reject named bots. It is described as a voluntary social convention rather than, by itself, a universal law or contract.
- Cloudflare and similar services detect bot traffic, present CAPTCHAs, or block it.
- Sites can block IP ranges or countries and impose rate limits.
- Terms of service may contractually forbid automated downloading or particular reuse.
- Copyright and licenses can restrict copying even when the page is technically reachable.

The lecture cites Shane Longpre and collaborators' **Consent in Crisis** study. It examined URLs in datasets such as C4, RefinedWeb, and Dolma and reported that restrictions increased sharply after the rise of generative AI, especially from mid-2023 onward. The accompanying slide shows a growing fraction of restrictive `robots.txt` policies and terms that prohibit crawling or AI-related use. The lecturer interprets this as a decline in practical consent for unrestricted collection.

The restrictions are not interchangeable. A crawler may be technically able to fetch a resource while violating a site's stated preferences, terms, license, or legal rights. Conversely, a block may prevent access without resolving the underlying legal question.

### Additional explanation

A useful provenance record should keep these layers separate:

| Layer | Question |
|---|---|
| Technical access | Could the collector retrieve the bytes? |
| Site preference | Did the publisher ask crawlers to stay away? |
| Contract | Did access or reuse breach accepted terms? |
| Copyright or license | Was copying and reuse permitted, restricted, or disputed? |
| Research ethics | Even if technically or legally possible, was collection responsible? |

Passing one layer does not imply passing the others. This is why "it was publicly visible" is not a complete provenance argument.

## 4. Crawler behavior and shadow libraries

**Transcript coverage:** lines 430-478

### What the lecturer said - transcript only

Crawler behavior also affects the people operating websites. The lecture gives reports that Anthropic's crawler contacted one site roughly one million times within 24 hours and that Read the Docs experienced similarly heavy traffic. The problem was not only uncompensated copying: excessive requests consumed server resources, raised costs, and could degrade service. Rate limits and responsible scheduling therefore matter independently of later model training.

The lecture then turns to **shadow libraries**, including Library Genesis, Z-Library, Anna's Archive, and Sci-Hub. These collections try to make books or papers freely available, often by bypassing paywalls. They have faced takedown orders, lawsuits, blocks in different countries, and technical attempts to evade those blocks. Some supporters frame them as access projects, but the lecturer emphasizes that, from a legal perspective, using pirated copies can constitute copyright infringement.

The section's conclusion is that the online world is enormous, but both technical and legal restrictions sharply limit what a responsible data collector can treat as usable.

### Additional explanation

This produces two distinct risk paths:

1. **Acquisition risk:** the bytes were obtained through abusive crawling, circumvention, or an unauthorized copy.
2. **Training-use risk:** the acquired material was then copied, transformed, or used in a way that may require permission or a legal defense.

A claim that model training is transformative does not automatically cure an unlawful acquisition path. Later legal examples in the lecture make this separation central.

---

# Part II - Copyright and model training

## 5. Copyright basics, public-domain material, and licenses

**Transcript coverage:** lines 479-695

### What the lecturer said - transcript only

The lecturer begins with a disclaimer that the legal situation is active and evolving. Intellectual-property law is presented as a mechanism intended to encourage production of intellectual goods. Copyright is the most relevant category here, alongside patents, trademarks, and trade secrets.

The lecture traces modern copyright to England's Statute of Anne in 1709 and, in the United States, to the Copyright Act of 1976. Copyright protects **original works of authorship fixed in a tangible medium of expression**. It protects expression rather than bare facts, ideas, or functional algorithms. A telephone directory without creative selection or arrangement is used to illustrate that merely collecting facts may be insufficient, although creative arrangement can itself matter.

Protection arises automatically when a work is fixed; unlike a patent, it does not depend on an initial application. The lecturer says that United States registration is needed before suing for infringement and gives a registration fee of about \$65. The lecture uses "75 years" as a simplified duration before a work enters the public domain, then points to Shakespeare, Beethoven, and much of Project Gutenberg as examples of material no longer under ordinary copyright restrictions.

Because most online expression is protected at a very low threshold, two broad routes for reuse are introduced:

1. obtain permission through a license;
2. rely on a doctrine such as fair use where it applies.

A license is described informally as a copyright holder's promise not to sue for specified uses. Creative Commons licenses offer standardized ways to allow redistribution subject to chosen conditions. Wikipedia, OpenCourseWare, Khan Academy, Free Music Archive, and openly licensed media collections are examples. Model developers can also negotiate paid licenses; deals involving platforms, publishers, and model companies are mentioned.

### Source reconciliation

The slide and transcript both use "75 years" as shorthand. That is not a universal copyright-duration rule. Duration varies by jurisdiction, date, authorship, publication status, and work type. The faithful transcript account above preserves the lecturer's wording, while this note prevents the classroom simplification from being mistaken for general legal guidance.

### Additional explanation

Three categories should not be conflated:

- **Public domain:** no applicable copyright restriction, though other rights or access conditions can still matter.
- **Permissively licensed:** copyright still exists, but the holder grants specified permissions and may impose conditions such as attribution or share-alike terms.
- **Publicly accessible:** anyone can view the page, but no training or redistribution permission follows merely from visibility.

Collections introduce another layer. Individual documents can have their own copyrights, while a sufficiently creative selection or arrangement can have a separate collection copyright. A license on a database wrapper does not necessarily grant rights in every underlying item.

## 6. Fair use and why it is not a text-overlap test

**Transcript coverage:** lines 696-891

### What the lecturer said - transcript only

The lecture introduces United States fair use through the four factors in Section 107:

1. **Purpose and character of the use.** Educational and transformative uses tend to receive more favorable consideration than purely commercial or reproductive uses.
2. **Nature of the copyrighted work.** Factual material tends to receive less protection in this analysis than highly creative expression.
3. **Amount and substantiality used.** A limited snippet can be more favorable than copying the whole work, although importance as well as quantity matters.
4. **Effect on the actual or potential market.** A use that substitutes for or harms the market for the original weighs differently from one that does not.

No single factor is presented as a mechanical switch. The lecturer gives several examples. A person may watch a movie and write a summary; a programmer may reimplement an idea without copying its source code; Google Books scanned books to build an index and show snippets. Authors Guild v. Google took about eleven years and ultimately favored Google.

Copyright is not limited to verbatim memorization. Plots and characters, such as elements associated with Harry Potter, can be protected expression. Parody can qualify as fair use because it transforms a work for commentary or criticism. The underlying issue is semantic and economic, not simply whether a particular n-gram appears.

For language models, the lecturer separates several questions. Copying data is an act that can matter even if training never occurs. Training can be argued to be transformative because it extracts broad patterns and ideas rather than reproducing a database, but commercial purpose and market effects can weigh in the opposite direction. A model can also affect the market for writers or artists even without reproducing exact passages.

Terms of service remain a separate layer. The lecture uses YouTube as an example: a video's license might permit reuse, yet the platform's terms can still prohibit downloading it in a particular way.

### Additional explanation

The key mistake to avoid is replacing a multifactor legal inquiry with a similarity threshold. These are different questions:

```text
Was a protected work copied?
Was the source copy lawfully acquired?
Does a license authorize this use?
If not, does a limitation or exception apply?
Does the trained system later emit protected expression?
```

The answers can differ for the same project. Dataset provenance, training use, and model output are related but separate stages of analysis.

## 7. The Spring 2026 litigation snapshot and audience questions

**Transcript coverage:** lines 892-1029

### What the lecturer said - transcript only

The lecture surveys several cases as a time-specific snapshot:

- **The New York Times v. OpenAI**, filed in 2023, alleged unauthorized training and included prompted examples resembling Times articles. The lecturer describes the case as pending.
- In litigation involving authors and **Anthropic**, a narrow ruling treated the particular training use of the plaintiffs' books as fair use. The lecturer stresses that this was not a general permission to pirate books. Pirated acquisition remained a separate problem. Anthropic had also purchased and scanned physical books, an acquisition route the lecturer says was treated more favorably, but doing so later did not erase the earlier piracy issue. The lecture reports a roughly \$1.5 billion settlement, around \$3,000 per book.
- In litigation involving authors and **Meta**, a narrow summary judgment treated the training use at issue as fair use, while allegations concerning torrenting remained unresolved in the lecture's account.

The lecturer's summary is deliberately limited: some specific instances of training had been found fair use, not model training in general; piracy was clearly a serious legal problem; and the area remained active and unsettled.

An audience member asked about voice data and services such as ElevenLabs. The lecturer declined to generalize beyond their expertise and suggested discussing it separately. Another question concerned a source whose license changes over time. The lecturer's non-lawyer intuition was that an earlier snapshot obtained under a permissive license would normally retain that permission, while later material published under changed terms would follow the later conditions.

### Additional explanation

The pedagogical value of these cases is the distinction between **use** and **provenance**. A court can view a training transformation favorably yet still find fault with how the source copy was acquired. A responsible dataset audit should therefore preserve source URLs, collection times, license snapshots, access method, and downstream transformations.

Because litigation and statutes change, these notes should be used to understand the lecture's analytical framework, not to determine the present legal status of any company, case, or dataset.

---

# Part III - Major raw data sources

## 8. Common Crawl and the mechanics of web collection

**Transcript coverage:** lines 1030-1158

### What the lecturer said - transcript only

Model developers sometimes operate their own crawlers, but Common Crawl is a foundational public source. It is a nonprofit founded in 2007 that runs large crawls approximately monthly. The lecture reports roughly three to five billion pages added per crawl and about 300 billion pages accumulated. A recent dump is described as approximately 2.19 billion pages and 372 terabytes. The lecturer notes that simple multiplication of monthly crawl sizes makes some of the headline totals hard to reconcile and that overlap between crawls matters. Google's index is described as at least 100 petabytes, emphasizing that even Common Crawl is not the whole web.

A crawler starts from a large set of seed URLs. A scheduler and queue select URLs; many download workers fetch them; discovered hyperlinks are normalized and returned to the queue; responses and metadata go to storage. A production crawler needs policies for:

- selecting which pages to download;
- respecting `robots.txt`, rate limits, and server load;
- deciding when changing pages should be revisited;
- handling dynamic URLs, mirrors, and many URLs that resolve to nearly identical content.

Common Crawl distributes two important representations:

- **WARC** files preserve raw HTTP responses, including HTML.
- **WET** files contain extracted text and are therefore smaller and easier to use, but the conversion is lossy.

HTML-to-text extraction is itself a modeling decision. The lecture cites experiments in which `trafilatura` and `resiliparse` produced better downstream language-model results than simply using Common Crawl's WET text. The web is uneven, but it contains concentrated high-quality regions such as Wikipedia, GitHub, and arXiv.

### Source reconciliation

The slide identifies the crawl snapshot as **April 2026** and gives **2.19 billion pages (372.2 TB)**. The transcript conveys these quantities while also recording the lecturer's uncertainty about how Common Crawl's cumulative-page claim should be interpreted. The notes retain both the published slide figures and that spoken qualification.

### Additional explanation

The extraction result can be written conceptually as

$$
d = E(r; \theta_E),
$$

where $r$ is the raw HTTP response, $E$ is an extractor, and $\theta_E$ represents its rules. Changing $E$ changes the training document $d$: navigation may be retained or removed, paragraphs can merge, tables can disappear, and boilerplate can dominate. "Same crawl" therefore does not imply "same corpus."

WARC is valuable when future extraction improvements are expected because it preserves more information. WET trades that recoverability for convenience and smaller storage.

## 9. Wikipedia: curated, downloadable, and still attackable

**Transcript coverage:** lines 1159-1253

### What the lecturer said - transcript only

Wikipedia began in 2001 and is described as containing about 67 million articles across many languages. Its scope excludes original thought and generally requires notability and citations to reliable sources. Anyone can edit, yet a relatively small community of frequent contributors, administrators, and bots performs a large share of maintenance. The lecturer mentions an exceptionally prolific contributor with roughly five million edits.

For dataset builders, Wikipedia offers periodic downloadable dumps, so crawling the live site is unnecessary. Its editorial structure and broad topical coverage make it an unusually high-quality web source, but it is not immune to manipulation.

The lecture discusses a data-poisoning attack described by Nicholas Carlini and collaborators. An attacker could insert malicious text shortly before a predictable snapshot, wait for the dump to capture it, and then allow the live article to be reverted. The visible encyclopedia would recover while the poisoned snapshot remained. A trigger phrase could then be associated with a chosen behavior, such as negative sentiment toward a product. The lecturer believes this specific vulnerability was subsequently addressed, but uses it to show that even respected sources can contain adversarial data.

### Additional explanation

A periodic dump creates a **snapshot boundary**. Security review must examine the exact snapshot used, not merely the current page. The general attack pattern is:

```text
predict snapshot time
    -> inject targeted content
    -> snapshot captures it
    -> live page is reverted
    -> training pipeline still sees the injected version
```

This connects data provenance to supply-chain security. Version IDs, timestamps, edit histories, and anomaly detection can matter as much as the source's general reputation.

## 10. GitHub, Software Heritage, and arXiv

**Transcript coverage:** lines 1254-1434

### What the lecturer said - transcript only

GitHub, founded in 2008 and acquired by Microsoft in 2018, is valuable not only for programming syntax but also for structured problem solving. The lecture reports more than 420 million repositories, with about 28 million public at the stated time. A repository can include source files, commit history, issues, pull requests, reviews, and comments. It also contains extensive duplication through copied code, mirrors, and forks.

The lecturer recommends using repositories under permissive licenses such as MIT or Apache when building a lower-risk code corpus. Repository content should be obtained through the Git protocol rather than by scraping GitHub's web interface. Metadata is a distinct source: the GitHub API supplies issues, pull requests, and comments, while GitHub Archive publishes an hourly event stream.

**Software Heritage** is a nonprofit archive founded in 2016 that preserves software from GitHub, GitLab, Bitbucket, PyPI, and other sources. Unlike a platform-centered archive, it focuses on repository content and metadata across hosting services. The lecture reports roughly 28.8 million source files at the cited moment.

**arXiv**, founded in 1991, lets researchers share papers in physics and later many other fields. It has roughly three million submissions. Records include metadata, a PDF, and sometimes LaTeX source. Its approval process is lighter than peer review. Authors select rights terms: some papers reserve ordinary copyright, while others use Creative Commons licenses. Metadata such as title and abstract is available under a permissive license, and bulk downloads are available, so a responsible collector need not crawl the site. PDF text must be extracted, whereas LaTeX source may provide cleaner structure but also needs macro and bibliography processing.

During questions, the lecturer says synthetic data will probably become increasingly important and will be discussed later. Asked how to ensure crawled data contains no pirated material, the lecturer says this cannot be guaranteed. Common Crawl can include copyrighted or improperly posted material. Both books and web pages can be copyrighted, though enforcement incentives differ. This motivates the later discussion of Common Pile, which deliberately adopts a more risk-averse provenance policy.

### Source reconciliation

The slide confirms the spoken GitHub figures as **420M+ repositories (28M public)**. It also clarifies that the Software Heritage figure is **28.8M source files**, not repositories.

### Additional explanation

These sources differ in both structure and lawful-use signals:

| Source | Native structure | Useful metadata | Central challenge |
|---|---|---|---|
| GitHub | Repository graph and development history | Licenses, commits, issues, pull requests, reviews | Licensing, secrets, personal data, duplicates, generated code |
| Software Heritage | Cross-platform archival graph | Origins, revisions, files | Mapping archived content back to usable provenance and licenses |
| arXiv | Scholarly records and document/source files | Authors, subjects, dates, rights | PDF or LaTeX conversion, versioning, mixed licenses |

Code training improves when the pipeline represents software as more than isolated files. A pull request connects a problem statement, a proposed change, review feedback, and a resolution. That sequence can teach development processes that raw code snapshots cannot.

---

# Part IV - How pretraining datasets evolved

## 11. BERT's BooksCorpus and GPT-2's WebText

**Transcript coverage:** lines 1435-1513

### What the lecturer said - transcript only

The historical survey begins with BERT. Its 2018 training mix used English Wikipedia and BooksCorpus. BooksCorpus had been assembled around 2015 by scraping self-published books offered for free on Smashwords. It contained about 7,000 books and roughly 985 million words. Although the books were free to read, the collection was later taken down for violating Smashwords' terms of service. This is an early illustration of the distinction between zero price, public accessibility, and permission to collect or redistribute.

Books were attractive because BERT needed document-level sequences rather than a benchmark made of disconnected sentences. The lecture contrasts BooksCorpus with the one-billion-word benchmark, whose sentences originated in a machine-translation context and did not preserve full-document structure.

GPT-2's 2019 **WebText** took a different approach. The authors considered raw Common Crawl too noisy and used Reddit community judgment as a quality proxy. They collected pages linked by Reddit posts that had at least three karma, yielding about eight million pages and 40 GB of text. WebText itself was not released. **OpenWebTextCorpus** later attempted an open replication by extracting external URLs from Reddit submissions, applying Facebook's fastText language classifier, and removing near duplicates.

### Additional explanation

These datasets embody two lasting ideas:

- **Document structure matters.** Long-form books preserve discourse, topic continuity, and references across sentences.
- **Metadata can act as weak supervision.** Reddit votes do not directly label prose quality, but they provide a noisy signal about what a community found worth sharing.

Both ideas also carry bias. Self-published books represent a particular author and genre distribution. Reddit votes reflect the users, norms, and ranking dynamics of specific communities. Weak supervision scales cheaply precisely because it inherits the imperfections of its proxy.

## 12. CCNet and C4: language-model filtering versus hand-written rules

**Transcript coverage:** lines 1514-1648

### What the lecturer said - transcript only

Facebook's 2019 **CCNet** work aimed to construct large, high-quality multilingual pretraining corpora, including for lower-resource languages. Its pipeline performed three major operations:

1. normalize text lightly and deduplicate paragraphs;
2. identify language with a fastText classifier and retain the desired language;
3. score document quality using a KenLM five-gram language model trained on Wikipedia, keeping text that looked more Wikipedia-like.

The result supported a key argument: sufficiently large, filtered Common Crawl data could outperform training only on Wikipedia. CCNet refers both to the processing tools and to corpora produced with them.

Google's 2019 **Colossal Clean Crawled Corpus**, or **C4**, became influential through T5. The T5 paper unified many NLP tasks under a text-to-text format, but C4 itself was a major contribution. Starting from an April 2019 Common Crawl snapshot of about 1.4 trillion tokens, it applied simple rules. It kept lines that ended in punctuation and contained at least five words; required a page to contain at least three sentences; removed pages containing terms from a bad-word list; removed common boilerplate such as JavaScript, cookie, and terms-of-use text; rejected pages containing a curly brace, which removed much code; and retained English with high language-identification confidence. The final corpus contained about 806 GB and 156 billion tokens.

Later analysis showed that C4's domains were highly uneven. The lecture notes that patents, Wikipedia, news sites, and many other sources contribute at very different scales. T5 also built a WebText-like subset by following pages in OpenWebText links. Even using 12 Common Crawl dumps, this produced only about 17 GB compared with WebText's 40 GB, evidence that Common Crawl is incomplete and snapshot-dependent. Nevertheless, C4 enabled strong gains on benchmarks such as GLUE and SuperGLUE.

### Additional explanation

CCNet and C4 represent two filter philosophies:

| Approach | Mechanism | Advantage | Failure mode |
|---|---|---|---|
| Reference-distribution filtering | Keep text that resembles Wikipedia under a language model | Learns many correlated quality cues automatically | Narrows the corpus toward Wikipedia's style and topics |
| Rule-based filtering | Apply explicit thresholds and pattern tests | Fast, interpretable, easy to audit | Brittle; innocent text can trigger a rule and useful domains can be erased |

The curly-brace example is especially instructive. A rule designed to remove malformed or non-natural text can silently delete programming material. Every quality filter is also a domain-selection policy.

## 13. GPT-3, The Pile, books, and Stack Exchange

**Transcript coverage:** lines 1649-1793

### What the lecturer said - transcript only

GPT-3 used a processed Common Crawl collection, an expanded WebText variant called WebText2, two undisclosed internet-book corpora called Books1 and Books2, and Wikipedia. The lecture gives a total of roughly 570 GB or 400 billion tokens. For Common Crawl, the pipeline trained a binary quality classifier to distinguish high-quality sources such as WebText, Wikipedia, and the book corpora from the rest. It also used fuzzy document deduplication, including checks against WebText and evaluation benchmarks.

In response to GPT-3, the volunteer-driven EleutherAI community produced **The Pile**, an open mixture of 22 domains. Contributors coordinated through Discord and collected sources such as Common Crawl, PubMed Central, arXiv, GitHub, Wikipedia, Stack Exchange, Ubuntu IRC, Enron email, BooksCorpus2, philosophy papers, and subtitles. The released mixture contained about 825 GB of raw text and used explicit component weights and multiple epochs for some smaller sources.

The lecture pauses on three components:

- **Project Gutenberg** began in 1971 to broaden access to literature. By the lecture's account it held about 75,000 books, mostly in English, that had received copyright clearance and were largely public domain. PG-19 uses Gutenberg books published before 2019.
- **Books3** contained roughly 196,000 books from the shadow library Bibliotik, including contemporary authors. It appeared in The Pile and later datasets but was removed after copyright disputes and lawsuits. The lecturer's recommendation is unambiguous: do not use it.
- **Stack Exchange** grew from Stack Overflow into a network of user-contributed question-and-answer communities. Its format resembles instruction data and real applications more closely than ordinary prose. Votes, badges, users, and comments provide metadata for filtering. Official data dumps are available in XML, so crawling is unnecessary.

The Pile also made the diversity of web-derived data concrete. Its Common Crawl portion started from WARC and used `jusText` rather than accepting WET extraction; PubMed Central contributed open research papers; arXiv supplied LaTeX; and the Enron corpus supplied about 500,000 emails from roughly 150 senior employees released during the investigation of the company.

### Source reconciliation

The slide gives The Pile's size as **825.18 GiB raw** and **1,254.20 GiB effective** after component weights or repeated epochs, corresponding to roughly **275 billion tokens**. These exact table values are slide-only verification of the spoken overview.

### Additional explanation

The Pile popularized a useful abstraction: a training corpus is a **mixture of named distributions**, not one undifferentiated text file. If source $i$ contains $D_i$ tokens and is assigned sampling weight $w_i$, its effective exposure depends on both size and sampling:

$$
p_i = \frac{w_i}{\sum_j w_j},
$$

where $p_i$ is the probability of sampling from component $i$ under a simple mixture sampler. Small, valued sources can be upsampled, but repeated exposure raises memorization and overfitting risk.

The Books3 episode also shows how provenance choices propagate. Once a corpus becomes a standard component, later recipes may copy it without independently revisiting the original acquisition path.

## 14. Gopher, LLaMA 1, and RedPajama

**Transcript coverage:** lines 1794-1911

### What the lecturer said - transcript only

DeepMind's 2021 **MassiveText** dataset was created for Gopher. Gopher was later overshadowed by Chinchilla and neither model's corpus was released, but the paper described processing in useful detail. The mixture included MassiveWeb, C4, books, news, GitHub, and Wikipedia, although the exact provenance of several sources remained unclear.

MassiveWeb retained English, deduplicated documents, removed train-test overlap, and applied manual quality rules rather than a learned quality classifier. One example required at least 80% of words to contain an alphabetic character. A Google SafeSearch classifier was used for toxicity. The resulting collection was about 10.5 TB, but Gopher trained on only around 300 billion tokens, roughly 12% of the available material. This illustrates that constructing a dataset and choosing the sampled training subset are different decisions.

LLaMA 1, released in 2022 in the lecture's chronology, documented a more explicit mixture totaling about 1.2 trillion tokens. It included:

- Common Crawl processed with CCNet, but classified by whether pages were **references linked from Wikipedia**, rather than Wikipedia articles themselves;
- C4, providing a broader rule-filtered web distribution;
- GitHub repositories with permissive licenses and manual filtering;
- Wikipedia snapshots in 20 languages;
- Project Gutenberg and Books3 inherited from The Pile;
- arXiv LaTeX with comments removed and macros and bibliographies expanded;
- Stack Exchange answers, sorted by score, across 28 large sites.

The Books3 component drew legal attention and contributed to later reluctance to disclose precise data recipes. **RedPajama v1** reproduced the LLaMA recipe openly. Its initial version also included Books3 and later removed it. Cerebras' SlimPajama produced a 627-billion-token subset using deduplication. The lecturer uses this chain to show how one dataset's legal and quality decisions are inherited by downstream replications.

### Additional explanation

The Wikipedia-reference signal is a form of graph-based weak supervision. Instead of asking whether a page linguistically resembles an encyclopedia, it asks whether encyclopedia editors considered the page worth citing. That captures a different notion of quality and inherits different biases: notability, citation practice, link rot, and subject-area coverage.

Reproducibility requires more than naming sources. A usable recipe should record snapshots, extractors, filter versions, deduplication thresholds, licenses, mixture weights, and exclusions. "We used Common Crawl" leaves nearly every consequential decision unspecified.

## 15. RefinedWeb, FineWeb, and Dolma

**Transcript coverage:** lines 1912-1982

### What the lecturer said - transcript only

Falcon's **RefinedWeb** advanced the thesis that carefully processed web data could be sufficient for a strong language model. It extracted text from WARC files with `trafilatura`, applied Gopher-like rules, and used fuzzy deduplication with MinHash over five-grams. The builders deliberately avoided learned quality filters because they worried that a model trained on a narrow reference distribution would amplify bias and erase useful variety. RefinedWeb processed about five trillion tokens and released a 600-billion-token subset.

Hugging Face's **FineWeb** began as an attempt to reproduce RefinedWeb and then improve it. It covered 95 Common Crawl dumps; applied URL filtering and English language identification; combined Gopher, C4, and additional manual rules; used MinHash deduplication; and anonymized email addresses and public IP addresses as personally identifiable information. The result contained approximately 15 trillion tokens.

AI2's **Dolma** combined its own Common Crawl processing with The Stack, C4, Reddit data from Pushshift, research papers from Semantic Scholar, Project Gutenberg, and Wikipedia or Wikibooks. For Common Crawl it used fastText language identification, Gopher and C4 quality rules, toxicity rules plus a Jigsaw classifier, and Bloom-filter deduplication. The lecture gives a final scale of roughly three trillion tokens.

### Additional explanation

These corpora reveal a recurring tension:

```text
aggressive learned filtering
    -> higher average similarity to the reference notion of quality
    -> less volume and possibly less diversity

lighter explicit filtering
    -> more volume and broader tail coverage
    -> more noise, toxicity, boilerplate, and extraction failures
```

There is no universally correct point on this curve. The best threshold depends on available compute, desired domains, repetition policy, and whether later training stages can repair weaknesses. Dataset evaluation must therefore train models or strong proxies; counting retained bytes alone cannot determine quality.

## 16. DCLM: a controlled benchmark for data recipes

**Transcript coverage:** lines 1983-2061

### What the lecturer said - transcript only

**DataComp for Language Models**, or **DCLM**, was designed as a standardized environment for comparing data-processing algorithms. Rather than changing both the raw pool and the model recipe, participants could process a shared Common Crawl pool and evaluate the resulting data under common training conditions. The unfiltered **DCLM-pool** contained about 240 trillion tokens.

The DCLM baseline applied English identification, heuristic cleaning, deduplication, and model-based quality filtering. Only about 1.4% of original documents survived the complete pipeline. The final baseline contained about 3.8 trillion tokens. This enormous contraction makes the filter recipe, not the raw pool size, the central object of study.

For the learned quality filter, the positive set combined roughly 200,000 examples each from OpenHermes 2.5 and ELI5. OpenHermes contains mostly GPT-4-generated instruction data; ELI5 contains highly rated explanatory questions and answers from Reddit. RefinedWeb supplied negative examples. A fastText linear classifier trained on these examples was then run over DCLM-pool. In the benchmark, this relatively simple classifier outperformed several more elaborate alternatives, including PageRank, semantic-deduplication scores, embedding classifiers, AskLLM, perplexity filtering, and top-$k$ average logits.

The lecturer treats DCLM as an open gold standard because it does not merely release one corpus. It exposes a pool, a baseline pipeline, and an evaluation protocol, making data algorithms more comparable.

### Source reconciliation

The slide's flow diagram confirms that the final baseline retains **1.4%** of original documents after heuristic cleaning, deduplication, and model-based filtering. The following slide gives **3.8T tokens** for the result and identifies the best comparison row as the fastText filter trained on OpenHermes 2.5 plus ELI5 positives.

### Additional explanation

DCLM turns dataset construction into an experimental variable:

```text
fixed raw pool + fixed model/evaluation recipe
                 |
                 v
          compare data pipelines
```

If two studies use different crawls, model sizes, tokenizers, compute budgets, and benchmarks, one cannot attribute their difference to filtering. Standardization removes some of these confounders.

The fastText result is also a caution against assuming that a more expensive judge is automatically better. A simple model can work well when its labels encode a strong target distribution. The choice of positives and negatives may matter more than classifier capacity.

## 17. Nemotron-CC and synthetic transformation of web data

**Transcript coverage:** lines 2062-2171

### What the lecturer said - transcript only

NVIDIA's **Nemotron-CC** argues that FineWeb-Edu and DCLM filter too aggressively, discarding roughly 90% or more of available documents. The goal is to preserve substantially more tokens without giving up the quality advantages of strong filtering.

Its quality system ensembles two signals. First, Nemotron-340B-Instruct scores FineWeb documents for educational value; those judgments are distilled into a faster classifier. Second, the DCLM quality classifier supplies an additional score. The pipeline then applies different synthetic transformations by quality level:

- Lower-quality documents are rewritten by a language model to resemble clearer, higher-quality prose.
- Higher-quality documents are used to generate additional tasks such as question-answer pairs, summaries, or key-information extraction.

The resulting corpus contains about 6.3 trillion tokens, including an approximately 1.1-trillion-token high-quality subset. The slide compares this with the reported 15 trillion training tokens for Llama 3 and 36 trillion for Qwen 3, while the lecturer warns that such numbers may count repeated epochs rather than distinct raw data.

In downstream comparisons, higher-quality Nemotron subsets outperform less selective FineWeb variants. The lecturer's broader conclusion is that a useful dataset lies between the extremes of retaining an enormous raw pool, such as 240 trillion tokens, and filtering down to roughly one trillion pristine tokens. Quantity and quality must be balanced.

### Additional explanation

Synthetic rewriting changes the pipeline from selection to transformation:

```text
traditional: raw document -> keep or discard
synthetic:   raw document -> score -> rewrite, augment, keep, or discard
```

This can recover information from poorly written pages, but it adds a new provenance chain. The generated document reflects both the source and the rewriting model. Errors can be polished rather than removed, rare styles can collapse toward the generator's style, and duplicated facts can multiply through several generated tasks.

Token totals should therefore be annotated with at least three quantities when possible:

- unique source tokens;
- synthetic or transformed tokens;
- total sampled training tokens, including repeats.

Without that distinction, dataset sizes are not directly comparable.

## 18. The Stack v1 and v2: from files to software-development trajectories

**Transcript coverage:** lines 2172-2292

### What the lecturer said - transcript only

The original **Stack**, released in 2022, used repository names from GitHub Archive and cloned a very large set of repositories. It kept repositories identified as permissively licensed with `go-license-detector`, removed near duplicates using MinHash and Jaccard similarity, and produced about 3.1 TB of code.

**The Stack v2**, released in 2024, expanded beyond repository files. It combined issues, comments, and pull requests from GitHub Archive; repository content preserved by Software Heritage; documentation crawled from sites associated with ecosystems such as PyPI, npm, and DevDocs; and existing code datasets such as GSM8K, code contests, Stack Overflow, arXiv, Wikipedia, and OpenWebMath. Processing removed binary files, malware, bot activity, duplicates, and personally identifiable information, and subsampled pull requests.

One augmentation pairs source code in lower-resource languages, such as Nim, with a shared low-level intermediate representation. Because several languages compile to a common intermediate form, aligned source-to-IR examples can help the model connect scarce high-level syntax to a representation also seen for richer languages.

Pull-request data is not naturally a single text document. The Stack v2 linearizes structured fields into a token sequence: title, status, repository, base files, changed files, diff hunks, review comments, comment relationships, review state, and nearby code context. Context must be selected carefully because complete repositories and long diffs can exceed sequence limits. This data can teach the process of changing and reviewing software rather than only the final code.

### Source reconciliation

The automatic transcript drops the scale marker and can sound as though the first Stack cloned only "137 repositories." The slide verifies **137M repositories**, approximately **5 TB of files**, and about **5 billion unique files**, followed by a filtered result of **3.1 TB of code**.

### Additional explanation

The Stack v2 illustrates a general principle for structured data: serialization is part of the learning problem. A good sequence must preserve relationships that matter to prediction.

For a pull request, a useful causal order is approximately:

```text
problem and repository context
    -> proposed base-to-diff change
    -> line-specific review feedback
    -> follow-up discussion
    -> approval, rejection, or requested changes
```

If comments are detached from their code lines or review states appear without the triggering change, the model sees tokens but loses the development logic. More data fields are not automatically better; the representation must make dependencies recoverable within the context window.

## 19. Common Pile and the limits of permissive-only collection

**Transcript coverage:** lines 2293-2428

### What the lecturer said - transcript only

The lecture returns to the earlier legal discussion with a deliberately conservative question: Can a strong model be trained only on public-domain or permissively licensed material? **Common Pile** assembles about 8 TB of such data. Its sources include The Stack v2, government and legal proceedings, wikis, permissively licensed news, academic papers, online forums, public-domain books, and educational resources.

The approach reduces some risks but contains subtleties:

- **License laundering:** a person can upload copyrighted material and attach a Creative Commons label they had no authority to grant. This is difficult to detect automatically.
- **Collection versus item rights:** a dataset may carry a license for its compilation or metadata while the individual works retain separate rights. The lecture uses Dolma's ODC-By collection license to illustrate that a collection-level label need not authorize every item.
- **Synthetic provenance:** Common Pile excludes synthetic data when the generating model may have been trained on unlicensed material, because the legal effect of that lineage is unclear.
- **Metadata reliability:** a license field on Hugging Face or another host is evidence, not proof that the uploader had the required rights.

Models trained on Common Pile perform reasonably and beat several older 2023-era systems in the presented comparison, but they lag Qwen 3, especially on coding benchmarks. The lecturer says the result is promising rather than decisive: a stronger permissive-only model may simply require more tokens and better curation.

### Source reconciliation

The slide's source chart confirms the **8 TB** total and shows that code is the largest category, followed by government and legal material, wikis, web collections, academic papers, forums, public-domain books, other media, and educational resources. The performance slide adds the caution that the available permissive corpus can do decently but is difficult to compare fairly against models trained on many more tokens.

### Additional explanation

Common Pile demonstrates that provenance is a claim requiring evidence at the document level. A robust record might include:

| Field | Purpose |
|---|---|
| Source URL or archive identifier | Locate the acquired object |
| Timestamp and snapshot | Determine which terms or license statement applied |
| Rights holder or publisher | Assess whether the licensor plausibly had authority |
| License text and version | Preserve exact permissions and obligations |
| Acquisition method | Separate lawful access from later use |
| Transformations and parent IDs | Track derived, extracted, or synthetic documents |
| Removal or dispute status | Support downstream deletion and audit |

No metadata system can prove every rights claim. It can, however, make uncertainty visible and enable corrections instead of burying provenance inside a monolithic corpus.

## 20. From raw services to training data: the lecture's synthesis

**Transcript coverage:** lines 2429-2507

### What the lecturer said - transcript only

The closing message is that data does not fall from the sky as a finished download. What begins as a live service, crawl, archive, repository, or scholarly collection must pass through extraction, normalization, filtering, deduplication, mixture construction, and often synthetic transformation before it becomes training data.

The magnitude of this processing deserves more attention. A pool can shrink from around 200 trillion raw tokens to fewer than three or four trillion selected tokens. Decisions made during that contraction determine the final distribution. Because most modern systems share broadly similar transformer architectures, data is one of the most important remaining differentiators between models.

Dataset construction remains legally and ethically difficult. It is also scientifically immature: many filters are based on hand-written rules, classifier thresholds, or informed intuition. The lecturer describes this as involving a surprising amount of "vibes," which makes it a fertile research area. Better evaluation, provenance, quality measurement, and filtering can produce major gains.

The next lecture will discuss post-training data and continue the treatment of filtering and processing.

### Additional explanation

The complete pipeline can be summarized as:

```text
raw sources
    -> lawful and responsible acquisition
    -> snapshot and provenance capture
    -> format-specific extraction
    -> language, safety, privacy, and quality filters
    -> exact and fuzzy deduplication
    -> benchmark-contamination controls
    -> domain mixture and sampling weights
    -> optional synthetic rewriting or augmentation
    -> tokenization, shuffling, and training shards
    -> model-based dataset evaluation
```

Every stage should be treated as a versioned transformation. This makes experiments reproducible, enables deletion or correction, and allows an observed model behavior to be traced back to plausible data causes.

---

# Consolidated takeaways

1. **A model is not trained on "the Internet."** It is trained on a particular stored corpus produced by access constraints, snapshots, extraction, filtering, and sampling.
2. **Public visibility is not permission.** Technical access, site preferences, terms of service, copyright, licenses, and research ethics are separate layers.
3. **Acquisition and training use are separate questions.** A transformative-use argument does not automatically resolve piracy, abusive crawling, or contractual restrictions.
4. **Raw-source reputation does not guarantee snapshot integrity.** Wikipedia's poisoning example shows that even curated sources require version-aware security.
5. **Text extraction is consequential.** WARC, WET, `trafilatura`, `resiliparse`, LaTeX processing, and repository serialization can yield different training distributions from the same nominal source.
6. **Metadata provides scalable but biased supervision.** Reddit karma, Wikipedia references, Stack Exchange votes, and license fields are useful proxies, not ground truth.
7. **Dataset history is cumulative.** Books3 shows how a provenance problem can propagate through The Pile, LLaMA recipes, and replications.
8. **Filtering chooses a domain, not only a quality level.** Wikipedia-like classifiers and hand-written rules remove different kinds of content and encode different preferences.
9. **Standardized data benchmarks matter.** DCLM makes processing methods comparable by fixing the raw pool and evaluation setting.
10. **Synthetic processing expands the design space.** Documents can be rewritten or converted into tasks, but this introduces new error, diversity, and provenance questions.
11. **Token totals need definitions.** Unique source tokens, transformed tokens, and repeated sampled tokens should not be treated as interchangeable.
12. **Permissive-only training is promising but not solved.** Common Pile reduces some risk while revealing license laundering, item-level rights, and coverage limitations.
13. **Data is a first-class research object.** With transformer architectures converging, dataset construction can be a primary source of model differentiation.

# Key quantitative relationships and reference figures

This lecture is primarily conceptual rather than mathematical. The following expressions organize its quantitative claims.

## Retention through a filter pipeline

If a raw pool contains $N_0$ documents and $N_k$ survive $k$ processing stages, the final document-retention rate is

$$
r = \frac{N_k}{N_0}.
$$

For the DCLM baseline, the slide reports approximately

$$
r \approx 0.014 = 1.4\%.
$$

Retention is not the same as token retention because removed documents have different lengths.

## Mixture sampling

For named components with nonnegative sampling weights $w_i$, a simple normalized mixture samples component $i$ with probability

$$
p_i = \frac{w_i}{\sum_j w_j}.
$$

An epoch count or upsampling policy can make effective exposure much larger than a component's unique token count.

## Reference figures from the lecture

| Resource or dataset | Figure reported in lecture or slides | Interpretive caution |
|---|---:|---|
| Common Crawl recent monthly snapshot | 2.19B pages, 372.2 TB | Snapshot-specific; pages overlap across crawls |
| C4 | 156B tokens, about 806 GB | English-only, heavily rule-filtered |
| GPT-3 training pool | 400B tokens, about 570 GB | Includes undisclosed Books1 and Books2 |
| The Pile | About 275B tokens, 825.18 GiB raw | Some components upsampled; Books3 later removed |
| LLaMA 1 | 1.2T tokens | Mixture inherited Books3 in the original recipe |
| RefinedWeb | 5T processed, 600B released | Avoided learned quality filters |
| FineWeb | 15T tokens | Built from 95 Common Crawl dumps |
| Dolma | About 3T tokens | Mixed web, code, Reddit, papers, books, and wikis |
| DCLM-pool | 240T raw tokens | Shared unfiltered comparison pool |
| DCLM baseline | 3.8T tokens, 1.4% documents retained | Quality target defined by its positive and negative examples |
| Nemotron-CC | 6.3T tokens, 1.1T HQ subset | Includes synthetic rewriting and task generation |
| The Stack v1 | 137M cloned repositories, 3.1 TB final code | Repository and file duplication is substantial |
| Common Pile | 8 TB | Public-domain or permissive intent does not eliminate provenance uncertainty |

# Glossary

**Acquisition provenance**
Evidence describing where an item came from, when and how it was collected, and which rights or restrictions were recorded.

**Books3**
A roughly 196,000-book corpus derived from the shadow library Bibliotik. It appeared in influential mixtures and was later removed amid copyright disputes.

**C4**
The Colossal Clean Crawled Corpus introduced with T5, built from Common Crawl using explicit English and quality rules.

**CCNet**
A multilingual Common Crawl processing pipeline using paragraph deduplication, language identification, and Wikipedia-like language-model scoring.

**Common Crawl**
A nonprofit that publishes recurring large web snapshots, commonly distributed as raw WARC and extracted WET files.

**Common Pile**
An approximately 8 TB collection designed to use public-domain or permissively licensed sources.

**Creative Commons**
A family of standardized licenses through which copyright holders grant selected reuse permissions subject to stated conditions.

**DCLM**
DataComp for Language Models, a standardized pool and benchmark for comparing data-processing strategies.

**Deduplication**
Removal of identical or near-identical documents, paragraphs, or code objects to reduce repetition, leakage, and wasted compute.

**Fair use**
A fact-specific United States doctrine evaluated through purpose and character, nature of the work, amount used, and market effect.

**FineWeb**
A large Hugging Face web corpus built across many Common Crawl dumps using language, quality, deduplication, and PII processing.

**GitHub Archive**
An event stream containing public GitHub activity metadata such as pushes, issues, and pull requests.

**License laundering**
Applying or propagating a permissive label to material when the person supplying that label may not own the underlying rights.

**Mid-training**
An informal stage between broad pretraining and behavioral post-training, often using higher-quality capability or long-context data.

**Nemotron-CC**
A web corpus that combines learned quality signals with synthetic rewriting and task generation to retain more useful tokens.

**Public domain**
Material not restricted by applicable copyright, distinct from material that is copyrighted but permissively licensed.

**RefinedWeb**
A Falcon-era web corpus emphasizing strong extraction, rules, and fuzzy deduplication while avoiding learned quality filtering.

**Robots exclusion protocol**
The `robots.txt` convention through which a site states which automated agents or paths should not be crawled.

**Shadow library**
An access-oriented collection of books or papers that may include unauthorized copies and may evade publisher or state restrictions.

**Software Heritage**
A nonprofit cross-platform archive preserving source code and repository history.

**The Pile**
EleutherAI's 22-component open mixture spanning web, books, code, papers, forums, and other domains.

**The Stack**
A family of large code datasets; v2 adds repositories, issues, pull requests, comments, documentation, and structured development data.

**WARC**
Web ARChive format containing raw web responses and metadata.

**WET**
Common Crawl's extracted-text representation, convenient but lossy relative to WARC.

**Weak supervision**
Labels inferred from indirect proxies such as votes, links, or source membership instead of direct expert annotation.

# Self-check questions

1. Why is saying that a model was trained on "the Internet" technically misleading?
2. What distinct restrictions can apply even when a URL is publicly viewable?
3. Why should acquisition legality and training-use legality be tracked separately?
4. What are the four United States fair-use factors as presented in the lecture?
5. Why is fair use not reducible to measuring verbatim overlap?
6. What information is lost when using WET instead of WARC?
7. How did the Wikipedia snapshot-poisoning attack exploit timing?
8. What different structures do GitHub, Software Heritage, and arXiv contribute?
9. How did WebText use Reddit karma as weak supervision, and what bias can that introduce?
10. Compare CCNet's quality model with C4's manual rules.
11. Why is Books3 important to the history of dataset provenance?
12. What does the DCLM benchmark hold fixed, and why does that improve comparison?
13. How does Nemotron-CC go beyond binary keep-or-discard filtering?
14. Why must unique, synthetic, and repeated training-token counts be distinguished?
15. What additional dependencies become learnable when pull requests are serialized well?
16. Why can a collection-level license fail to authorize the individual works inside it?
17. What does Common Pile demonstrate, and what does it leave unresolved?
18. Name at least five versioned transformations needed between a raw source and training shards.

# Source coverage checklist

- [x] **Lines 1-160:** motivation, data secrecy, labor, pretraining/mid-training/post-training stages, terminology, and OLMo transparency.
- [x] **Lines 161-294:** why the Internet is not a stored dataset, crawling, dynamic content, authentication, and walled gardens.
- [x] **Lines 295-429:** `robots.txt`, Cloudflare, IP blocks, rate limits, terms, licenses, and the decline-of-consent study.
- [x] **Lines 430-478:** abusive crawler traffic, server costs, shadow libraries, circumvention, and legal restrictions.
- [x] **Lines 479-695:** intellectual property, copyright threshold, fixation, registration, duration shorthand, public domain, and licenses.
- [x] **Lines 696-891:** four fair-use factors, examples, semantics beyond verbatim overlap, model-training considerations, and terms of service.
- [x] **Lines 892-1029:** New York Times, Anthropic, and Meta litigation snapshot; voice-data and changing-license questions.
- [x] **Lines 1030-1158:** Common Crawl scale, crawler architecture and policies, WARC/WET, and HTML extraction.
- [x] **Lines 1159-1253:** Wikipedia scope, contributors, dumps, and snapshot poisoning.
- [x] **Lines 1254-1434:** GitHub, GitHub Archive, Software Heritage, arXiv, synthetic-data question, and piracy-in-crawls question.
- [x] **Lines 1435-1513:** BERT, BooksCorpus, GPT-2 WebText, and OpenWebText replication.
- [x] **Lines 1514-1648:** CCNet, C4 rules and scale, domain analysis, WebText-like subset, and benchmark impact.
- [x] **Lines 1649-1793:** GPT-3 mixture and filtering, The Pile, Gutenberg, Books3, Stack Exchange, and component details.
- [x] **Lines 1794-1911:** Gopher/MassiveText, LLaMA 1 mixture, RedPajama, SlimPajama, and inherited provenance decisions.
- [x] **Lines 1912-1982:** RefinedWeb, FineWeb, and Dolma sources and processing.
- [x] **Lines 1983-2061:** DCLM-pool, baseline contraction, positive and negative examples, fastText classifier, and standardized evaluation.
- [x] **Lines 2062-2171:** Nemotron-CC quality ensemble, synthetic rewriting, task generation, sizes, and quantity-quality tradeoff.
- [x] **Lines 2172-2292:** The Stack v1/v2, licenses, deduplication, Software Heritage, intermediate representations, and pull-request serialization.
- [x] **Lines 2293-2428:** Common Pile, source categories, license laundering, collection rights, synthetic provenance, and performance limits.
- [x] **Lines 2429-2507:** end-to-end dataset transformation, filtering scale, data as differentiator, legal and ethical issues, research opportunities, and next lecture.

All 2,507 transcript lines and all 33 slides were covered.
