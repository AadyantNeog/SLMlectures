---
title: "Lecture 17 - Alignment and Multimodal Models"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: 17
primary_source: "../transcripts/Stanford CS336 Language Modeling from Scratch  Spring 2026  Lecture 17 Alignment - Multimodality.txt"
slide_deck: "../lecture_17.pdf"
status: "complete"
---

# Lecture 17: Alignment and Multimodal Models

## How to read these notes

Each substantive topic has two deliberately separate layers:

1. **What the lecturer said - transcript only.** A clean paraphrase of the spoken lecture that preserves substantive claims, examples, qualifications, numerical details, and audience questions while removing filler and repetition.
2. **Additional explanation.** Independent intuition, derivations, connections, and study guidance. This material is not attributed to the transcript.

The raw transcript's physical line spans are included for auditability. The complete 2,187-line transcript was mapped before the 36-slide deck was inspected page by page. Slides were used to verify names, model diagrams, objectives, training stages, data scales, token budgets, and examples. Slide-only precision is isolated under **Source reconciliation**.

## Lecture map

The lecture follows the path from a text-only transformer to models that understand and sometimes generate other modalities:

1. Recast images, video, and audio as sequences of continuous or discrete tokens.
2. Learn semantic image representations with CLIP and the more decomposable SigLIP objective.
3. Connect a pretrained vision encoder to a pretrained language model through an adapter, using LLaVA as the simplest template.
4. Extend the template to high-resolution images, multiple images, and video in LLaVA-OneVision and the Qwen-VL family.
5. Study Qwen3-VL's long-context, positional, loss-balancing, and deep-fusion refinements.
6. Contrast input-only vision-language models with Chameleon's unified discrete-token approach to text and image generation.
7. End with the unresolved asymmetry between high-level understanding and fine-grained generation.

---

# Part I - The multimodal interface problem

## 1. From text-to-text models toward an omni model

**Transcript coverage:** lines 1-137

### What the lecturer said - transcript only

The original plan was to continue reinforcement learning, but a course on modern language models would be incomplete without multimodality. The topic could fill an entire course, so this lecture is an overview with an emphasis on images.

Language models are already broad within text: they can map arbitrary textual input to arbitrary textual output across natural languages, code, poetry, and even sequence-like domains such as DNA. The world, however, also contains images, audio, and video. The long-term "omni model" goal is to accept any combination of modalities and produce any combination of modalities. Such a system might jointly inspect an image and video, answer questions about them, generate an image, or transform information across media.

Transformers remain the strongest scalable architecture across these modalities, so the practical problem is how to express non-text data in a form they can consume. A transformer processes tokens. The lecturer broadens "token" beyond a discrete text symbol to include a continuous embedding, but still wants each token to represent a meaningful unit of information. A subword has some semantic content; a raw pixel generally does not.

Every modality must therefore be converted into a sequence of discrete or continuous tokens. Text already required this step through BPE. The analogue for images, audio, or video is much less obvious.

There are two separate questions:

1. How can non-text data be encoded as transformer input?
2. How can a model generate non-text data as output?

Most of the lecture addresses the first question.

### Source reconciliation

The slides visualize text, images, audio, and video and define an omni model as one that accepts and emits arbitrary combinations. The outline names two technical routes: continuous image encoders followed by language-model injection, and discrete image tokens for more uniform multimodal generation.

### Additional explanation

The encoder is an information bottleneck:

$$
x_{\mathrm{modality}}
\xrightarrow{E}
(z_1,\ldots,z_m)
\xrightarrow{\mathrm{Transformer}}
\text{prediction}.
$$

Its token sequence must balance three pressures:

- **Fidelity:** preserve details needed by the downstream task.
- **Compression:** avoid producing an impractically long sequence.
- **Compatibility:** give the transformer vectors or IDs with learnable structure.

No single tokenization is optimal for every objective. Image classification can discard font strokes and exact textures. OCR cannot. Image generation needs still more local detail. This task dependence becomes the lecture's central theme.

## 2. CLIP turns image-text matching into symmetric in-batch classification

**Transcript coverage:** lines 138-244

### What the lecturer said - transcript only

CLIP, short for Contrastive Language-Image Pre-Training, is a foundation of modern vision-language models. Around 2021, language modeling had entered the foundation-model era, while computer vision still relied heavily on carefully annotated datasets such as ImageNet, architectures such as ResNet, and extensive augmentation.

The CLIP project asked whether naturally occurring image-caption pairs on the web could play the role that scraped text played for language models. The training data are aligned image-text pairs. For a large batch, an image encoder maps each image to a vector and a text encoder maps each corresponding caption to a vector.

For image $i$, the similarity with its paired text $i$ should exceed its similarity with every other text in the batch. Symmetrically, text $i$ should be more similar to its paired image than to every other image. This produces $2N$ multiclass classification problems for a batch of $N$ pairs.

Implementation is compact. The image and text encoders produce two $N$-by-$d$ matrices. Their rows are normalized, their matrix product forms all pairwise similarities, a learned temperature rescales the logits, and cross-entropy is applied in both directions.

### Source reconciliation

The slide uses a batch example of **32,768** image-text pairs. Its pseudocode projects both encoder outputs into a joint embedding space, applies L2 normalization, multiplies the image matrix by the transposed text matrix, scales by an exponentiated learned temperature, and averages image-to-text and text-to-image cross-entropies.

### Additional explanation

Let normalized image and text embeddings be $u_i$ and $v_j$, with temperature $\tau$. Define

$$
s_{ij}=\frac{u_i^\top v_j}{\tau}.
$$

The symmetric CLIP loss is

$$
\mathcal L_{\mathrm{CLIP}}
=
\frac{1}{2N}
\sum_{i=1}^{N}
\left[
-\log
\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}
-\log
\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}
\right].
$$

The diagonal pairs are positives and all off-diagonal batch pairs act as negatives. Large batches are valuable because they make each classification problem harder and provide many negative comparisons without a separate negative dataset.

The learned space supports retrieval in both directions:

$$
\text{image}\rightarrow\text{nearest captions},
\qquad
\text{text}\rightarrow\text{nearest images}.
$$

It also supports zero-shot classification by turning class names into text prompts and selecting the most similar text embedding.

## 3. CLIP data, OpenCLIP, and fixed-resolution preprocessing

**Transcript coverage:** lines 245-320

### What the lecturer said - transcript only

The CLIP paper gives limited detail about its data collection. Broadly, queries were used to search the web and mine image-text pairs, producing 400 million examples. The dataset was not released.

OpenCLIP later reproduced and extended the approach with public code and the LAION-5B dataset, containing roughly five billion images with associated text. CLIP itself was used to filter that corpus before OpenCLIP was trained, creating a bootstrapping loop in which a learned representation selects the next representation's data.

Internet images have arbitrary aspect ratios and resolutions, while neural networks are easiest to train with fixed shapes. CLIP uses a convenient heuristic: resize until the shorter side reaches the target, such as 224 or 336 pixels, then take a square center crop. This can discard information but was acceptable for an ImageNet-oriented classification setting in which the primary object is often near the center. Later models improve on this choice.

### Source reconciliation

The data slide states that CLIP was trained on 400 million pairs and that OpenCLIP used LAION-5B with CLIP-based filtering. The preprocessing slide specifies bicubic resizing of the shorter side to 336 pixels followed by a $336\times336$ center crop.

### Additional explanation

Center cropping makes compute predictable:

$$
\text{fixed image size}
\rightarrow
\text{fixed patch count}
\rightarrow
\text{fixed attention cost}.
$$

Its failure is not only aesthetic. A wide receipt, a dense page, or text near an edge can lose task-critical evidence. Classification may tolerate that loss because global object identity is redundant. Document understanding and OCR cannot.

CLIP-based filtering also creates selection bias. The retained web pairs are those an earlier CLIP model already considers compatible. This can improve alignment quality while narrowing the data toward the predecessor's semantic preferences and blind spots.

## 4. Vision and text encoders, patch tokens, semantic supervision, and position

**Transcript coverage:** lines 321-496

### What the lecturer said - transcript only

An audience member asks why CLIP uses image-text pairs rather than image-only self-supervision. The lecturer contrasts CLIP with methods such as SimCLR, where two augmented versions of the same image are encouraged to share an embedding. Cropping, rotating, and perturbing an image teaches useful low-level invariances, but augmentation cannot turn one dog breed into another. Text supplies a higher-level semantic relation and therefore encourages semantic image representations.

CLIP tested ResNets and the newly introduced Vision Transformer, or ViT, and found the ViT variants strongest. A ViT divides an image into fixed patches. Each flattened patch is linearly projected into a vector, receives positional information, and enters an ordinary transformer encoder. The patches are the image tokens.

The encoder produces a sequence of patch vectors. Instead of plain mean pooling, CLIP uses attention pooling: the global average activation serves as a query that attends to the keys and values from all positions, producing a more informed single image vector.

The strongest reported CLIP variant discussed is ViT-L/14 at 336-pixel resolution. "L" denotes a large ViT and 14 denotes $14\times14$ pixel patches. High-resolution training is expensive, so resolution was increased in a later phase.

In response to a question about two-dimensional positional embeddings, the lecturer says CLIP explored spatially aware alternatives but found little difference for its classification goal. Later models use more explicit multidimensional position schemes.

The text encoder is a standard GPT-2-style transformer. It adds beginning- and end-of-sequence markers and uses the final layer's end-of-sequence activation as the representation of the complete caption.

With both encoders defined, training repeatedly encodes a batch and optimizes the two directional cross-entropies.

### Source reconciliation

The slides specify **ViT-L/14@336px** as the best CLIP vision model in this comparison. They describe the text encoder as a 63M-parameter, 12-layer GPT-2-style transformer and confirm that the final end-of-sequence activation is returned as the text representation.

### Additional explanation

For image height $H$, width $W$, and square patch size $P$, the ViT sequence length is

$$
M=\frac{H}{P}\frac{W}{P}.
$$

The patch projection maps each vector in $\mathbb R^{3P^2}$ to the transformer width. Smaller patches preserve more detail but increase self-attention cost approximately as $M^2$.

Text pairing changes what the representation is optimized to preserve. If captions usually mention "a dog playing in snow" but omit exact fur texture, CLIP receives a strong signal for dog, action, and scene semantics and little signal for texture reconstruction. That bias is useful for retrieval and classification and incomplete for OCR or generation.

Attention pooling is task-specific compression:

$$
(h_1,\ldots,h_M)
\rightarrow
z_{\mathrm{image}}.
$$

Once the entire image is reduced to one vector, fine spatial correspondences are difficult to recover. Later VLMs therefore inject grids or sequences of vision features rather than only one pooled CLIP vector.

## 5. Zero-shot classification, noisy negatives, and why caption generation was less efficient

**Transcript coverage:** lines 497-616

### What the lecturer said - transcript only

CLIP's headline 2021 result was that zero-shot CLIP outperformed a ResNet-50 trained on 1.2 million labeled ImageNet examples. ImageNet annotation required substantial Mechanical Turk labor. CLIP instead leveraged naturally occurring web data and the labor implicit in producing that public content.

For zero-shot prediction, an image is compared with text representations of candidate labels or prompts, and the label with the highest similarity is chosen.

An audience member asks whether other captions containing the same concept, such as another dog, confuse the contrastive loss. The data and objective are noisy, but a batch contains many unrelated concepts, so individual false negatives need not dominate on average. Web captions are noisy for a deeper reason: adjacent text or alt text does not necessarily state literally what is visible. If an image obviously contains a dog, a human author may discuss something else. Considerable filtering is needed before arbitrary web image-text pairs become usable.

CLIP also compared contrastive ranking with objectives that predict text from images, either as a bag of words or with a language model. The stronger sequence model was less compute-efficient for ImageNet accuracy, while contrastive bag-of-words-style learning performed better. Precisely modeling caption word order was not necessary for the coarse semantic representation measured by classification.

### Source reconciliation

The slide verifies the zero-shot comparison against a ResNet-50 trained on 1.2M ImageNet images. Its efficiency plot shows contrastive bag-of-words training above bag-of-words prediction and autoregressive caption language modeling for zero-shot ImageNet accuracy at matched image counts.

### Additional explanation

Zero-shot classification constructs a classifier from language:

$$
\hat y
=
\arg\max_{c}
\operatorname{sim}
\left(
E_{\mathrm{img}}(x),
E_{\mathrm{text}}(\text{prompt}(c))
\right).
$$

Prompt wording matters because the text encoder represents a sentence, not an abstract class ID. Ensembling prompts such as "a photo of a dog" and "an image of the dog" can reduce template sensitivity.

An in-batch false negative occurs when an off-diagonal pair is actually semantically compatible. Contrastive learning tolerates some such noise, but systematic duplicates or frequent concepts can distort the geometry. Deduplication, caption filtering, and hard-negative design matter at scale.

The generative objective spends capacity on syntax, function words, and one arbitrary caption realization. For a classification metric, those details are nuisance variation. This does not prove generation is universally inferior; it shows that a coarser discriminative objective was better aligned with that representation benchmark.

## 6. SigLIP replaces global softmax classification with pairwise logistic loss

**Transcript coverage:** lines 617-817

### What the lecturer said - transcript only

CLIP produces semantic image encodings, but its design is oriented toward classification and is not fine-grained. It also needs very large batches, around 30,000 examples, and its softmax couples the loss across the full batch. A batch of one cannot define useful alternatives, and distributed training cannot decompose as cleanly as ordinary language-model examples.

Google's SigLIP, or Sigmoid Loss for Language-Image Pre-Training, simplifies the objective. CLIP asks which one of many texts matches an image and which one of many images matches a text. SigLIP asks a binary question for each pair: are this image and this text aligned? Diagonal pairs receive positive labels and off-diagonal pairs receive negative labels. Normalized similarities feed a log-sigmoid loss.

An audience member asks about sophisticated negative sampling. In the initial paper, the same pairwise batch matrix was used without an elaborate strategy, although contrastive learning more generally may need balanced or difficult negatives.

SigLIP used Google's WebLI corpus, on the order of a billion image-text pairs. It included OCR-derived text, filtering, and multilingual data.

The primary contribution was efficiency. The lecture compares CLIP's ten days on 256 TPUv3 chips with SigLIP's five days on 32 TPUv4 chips and notes that a TPUv4 is not individually faster in raw FLOPs in a way that explains the full difference. SigLIP distributes the similarity matrix in blocks: each device computes local pairs, then text embeddings rotate among devices so all off-diagonal blocks are eventually scored.

The pairwise formulation decouples the definition of the objective from batch size. CLIP's multiclass denominator changes when the batch changes. With SigLIP, a smaller batch gives a noisier estimate of the same pairwise loss. SigLIP is better than CLIP below roughly 16K batch size. Very large batches, even up to one million, do not continue helping; around 32K behaves like a critical batch size.

### Source reconciliation

The slides give the pair labels as $+1$ on the diagonal and $-1$ elsewhere, with a learned temperature and bias. They add slide-only WebLI details: automatic OCR, retention of the highest-quality 10 percent, and support for 100 languages. The distributed diagram shows cross-device rotation of text blocks. The batch summary says SigLIP can be trained at much smaller batches, can be run as high as one million, and gains little beyond about 32K.

### Additional explanation

Let $z_{ij}=+1$ for a matched pair and $-1$ otherwise. A simplified SigLIP objective is

$$
\mathcal L_{\mathrm{SigLIP}}
=
-\frac{1}{N}
\sum_{i,j}
\log\sigma
\left[
z_{ij}
\left(
t\,u_i^\top v_j+b
\right)
\right],
$$

where $t$ and $b$ are learned scale and bias terms.

CLIP normalizes each example against the alternatives present in that batch. SigLIP treats sampled pairs as observations from a fixed binary problem. This makes blockwise distributed computation natural:

$$
\text{local image block}
\times
\text{rotating text block}
\rightarrow
\text{independent pair losses}.
$$

Batch size is not irrelevant. It still controls the number and diversity of negatives and the variance of the gradient. The benefit is that changing it no longer changes a global softmax class set in the same way.

---

# Part II - Injecting continuous vision tokens into a language model

## 7. LLaVA stitches together CLIP, Vicuna, synthetic conversations, and a projector

**Transcript coverage:** lines 818-951

### What the lecturer said - transcript only

CLIP and SigLIP map a fixed-size image into semantically useful vectors. The next step is to build a vision-language model by injecting those vectors into a language model. The lecture studies the LLaVA and Qwen families, which share the broad template of reusing a pretrained vision encoder and pretrained LLM rather than training an entire multimodal model from scratch.

LLaVA appeared in 2023, when closed GPT-4 systems had demonstrated visual reasoning. It was exciting because an open model exposed a workable approximation of that capability.

LLaVA has three pieces:

1. a CLIP vision encoder;
2. Vicuna, a Llama model fine-tuned on conversations shared by ChatGPT users;
3. a learned projection from CLIP's vision-feature space into the language model's embedding width.

Its data begin with MS COCO, whose images have human captions and object bounding boxes. GPT-4 receives captions or detected objects and synthesizes three kinds of targets: conversations, detailed descriptions, and complex reasoning questions. The generated text is paired back with the original images, producing 158,000 examples.

LLaVA uses CLIP ViT-L/14. The resulting vision features do not initially inhabit the same representational space or dimensionality as text embeddings, so a matrix $W$ maps them into vectors the language model can accept. Those visual vectors and ordinary token embeddings form one sequence processed by the standard autoregressive transformer. In this practical sense, the image is converted into continuous "textual" tokens that reuse the language model's capabilities.

### Source reconciliation

The slides expand LLaVA as **Large Language and Vision Assistant**. They confirm CLIP ViT-L/14, Vicuna based on ShareGPT conversations, MS COCO captions and boxes, GPT-4 synthesis, and 158K examples. They also note that Flamingo and Q-Former use more complex connectors than LLaVA's linear projection.

### Additional explanation

If the vision encoder emits

$$
Z_v\in\mathbb R^{M\times d_v},
$$

and the LLM has width $d_{\mathrm{lm}}$, LLaVA learns

$$
H_v=Z_vW,
\qquad
W\in\mathbb R^{d_v\times d_{\mathrm{lm}}}.
$$

The projected rows can then be concatenated with text embeddings:

$$
[H_v;H_q]
\rightarrow
\text{causal language model}
\rightarrow
\text{text response}.
$$

The word "alignment" here has a concrete operational meaning: learn a connector that makes pretrained visual features useful inside the fixed LLM interface. It does not mean that a particular visual vector becomes identical to a word embedding. The downstream transformer can learn a distributed interpretation across many projected vision tokens.

Synthetic instruction data converts static annotations into an interface task. A caption says what is present; a conversation teaches how users ask about it; reasoning examples teach how the answer should combine visual evidence with language-model knowledge.

## 8. LLaVA's alignment and fine-tuning stages

**Transcript coverage:** lines 952-1009

### What the lecturer said - transcript only

LLaVA training has two stages. In the first, both the vision encoder and language model are frozen. Only the projection matrix $W$ is trained. This is called alignment: a random projection does not produce vectors the language model can interpret, so the connector learns to translate the fixed vision features into a useful input distribution.

In the second stage, the vision encoder remains frozen while both $W$ and the language model are fine-tuned on image-plus-text inputs and conversation, description, or reasoning outputs.

The paper's "extreme ironing" example shows a person ironing clothes on the back of a vehicle. LLaVA recognizes what is unusual and can continue discussing it even when a follow-up prompt does not repeat the word "unusual." GPT-4 also succeeds, while several other contemporary open vision-language systems give weaker or incorrect descriptions. This illustrated the strength of the simple open recipe at the time.

### Source reconciliation

The training diagram confirms:

- Stage 1: freeze the CLIP encoder and LLM; train only $W$.
- Stage 2: freeze CLIP; train $W$ and the LLM.

The example slide compares LLaVA with GPT-4, BLIP-2, and OpenFlamingo on the same unusual-image prompt.

### Additional explanation

Freezing components constrains what each stage can solve:

| Stage | Trainable component | Main adaptation |
|---|---|---|
| Alignment | Projector only | Match interface, scale, and coarse semantics |
| Instruction fine-tuning | Projector plus LLM | Teach multimodal tasks, response style, and reasoning behavior |

If the projector alone cannot preserve a visual distinction, later LLM tuning cannot reconstruct it. Conversely, a rich vision encoder may already contain the information, while only the LLM must learn where to attend and how to verbalize it.

The second stage risks weakening text-only capabilities if its mixture is too narrow. Modern systems therefore mix text data, multimodal data, and regularization or replay rather than treating multimodal fine-tuning as an isolated small task.

## 9. LLaVA-OneVision adds multiple images, video, AnyRes, and token budgeting

**Transcript coverage:** lines 1010-1181

### What the lecturer said - transcript only

The lecture jumps from the original LLaVA to LLaVA-OneVision in 2024, absorbing intermediate LLaVA 1.5 and LLaVA-NeXT innovations. The broad architecture remains the same, but its scope expands to single images, multiple images, and video.

The components are upgraded: SigLIP replaces CLIP as the vision encoder, Qwen2 replaces Vicuna as the text decoder, and a two-layer MLP replaces the single linear projector.

High resolution is essential for OCR. CLIP's square resize and crop can make document text unreadable. **AnyRes**, introduced earlier in the LLaVA series, handles this by splitting an image into crops at the vision encoder's expected resolution. Each crop is encoded independently and the resulting vision tokens are concatenated. A downsampled view of the entire image is also retained so the model has global context. If a very high-resolution image produces too many tokens, bilinear interpolation reduces the vision grid.

The design exploits a transformer's ability to accept variable sequence lengths: different image resolutions can produce different numbers of visual tokens rather than being forced into one lossy square.

Token allocation is deliberately modality dependent. A single image can receive a global view plus up to nine high-resolution crops. Multiple images receive only the base resolution per image. Video frames receive still fewer tokens, and the system uses at most 32 frames. This prevents a long or repetitive video from exhausting the context and makes single images, image sets, and videos more comparable in total token cost.

The lecturer emphasizes that handling long context becomes central once visual input expands from one thumbnail to many crops or frames.

### Source reconciliation

The architecture slide specifies SigLIP grid features, a Qwen2 **72B** decoder, and a two-layer MLP projector. The AnyRes slide gives concrete maximum examples:

- single image: one global view plus up to nine crops, at most 7,290 vision tokens;
- multiple images: up to twelve base-resolution images, at most 8,748 tokens;
- video: up to 32 frames at 196 tokens each, at most 6,272 tokens.

The transcript explains the balancing principle but not all of these exact counts.

### Additional explanation

AnyRes creates a visual pyramid:

$$
\text{global thumbnail}
\;+\;
\{\text{high-resolution tiles}\}.
$$

The thumbnail answers "where is the relevant region?" while a tile answers "what exactly is written or drawn there?" Concatenating both lets the LLM integrate coarse layout and fine detail.

The tradeoff is sequence cost. If an image is split into $K$ tiles and each produces $M$ tokens, visual self-attention and cross-modal processing see roughly $KM$ additional positions. A token budget is therefore an allocation policy, not merely preprocessing:

$$
\text{more views}
\leftrightarrow
\text{more fidelity}
\leftrightarrow
\text{more context and compute}.
$$

Video has high temporal redundancy. Spending the same per-frame resolution as a single still image would overweight nearly repeated content and reduce the duration that fits.

## 10. LLaVA-OneVision data curriculum and transfer across visual formats

**Transcript coverage:** lines 1182-1315

### What the lecturer said - transcript only

LLaVA-OneVision emphasizes curated quality and quantity, but much of the corpus remains targeted by task: visual question answering, chart and table questions, OCR, comparisons between images, and video tasks. This is post-training-like data construction, because the desired downstream behaviors are represented explicitly.

The project also distills GPT-4 heavily to obtain strong synthetic examples without an equivalent human annotation budget. The lecturer notes that this is not ideal as an independent source of supervision, but it is a practical choice for an open effort.

Training has three stages rather than the original two. First, train only the projector for image-language alignment. Next, train the full model on high-quality data emphasizing knowledge. Finally, train the full model on examples that resemble downstream visual instruction tasks. The lecturer does not claim that the exact boundary between the last two stages is theoretically necessary; it is an easy-to-hard curriculum.

An important result is transfer across input formats and task families. Diagram and chart training may use single images, yet the model can answer a question that combines a diagram in one image with a table in another. OCR learned on single images and relational reasoning learned from image sets can combine in GUI-agent scenarios involving multiple screenshots.

Visual prompting supplies another example. A circle or mark inside a single image indicates which object to discuss. Although this supervision is available only for still images, the model can follow a highlighted player across video frames.

The collection looks like task-specific supervised learning, but sufficient task diversity produces useful compositional transfer. The standard architecture is less novel than the data curation. A major strength of the LLaVA line is that it releases both weights and data, enabling replication and analysis.

### Source reconciliation

The data slides report a 3.2M-example single-image mixture and a 1.6M-example OneVision mixture spanning single-image, multi-image, and video sources. The stage table gives:

- Stage 1 language-image alignment with 558K LCS examples and projector-only training;
- an intermediate high-quality knowledge stage using 4M image examples;
- final single-image and OneVision instruction stages using the 3.2M and 1.6M mixtures.

The transfer slides show the diagram-plus-table problem, a multi-screen mobile GUI sequence, and a video whose highlighted player must be described.

### Additional explanation

The reported transfer is compositional:

$$
\text{OCR skill}
+\text{multi-image relation skill}
\rightarrow
\text{multi-image document reasoning}.
$$

It is not zero supervision in the broad sense. Each component skill and the shared interface are trained. What is new is their combination across a format not shown in exactly that form.

A task mixture can function like a basis. If tasks cover localization, reading, comparison, temporal tracking, and response generation, new evaluations may be expressible as combinations of those operations. The risk is benchmark-shaped overfitting: a long list of named datasets can look diverse while sharing annotation conventions and synthetic teacher biases.

## 11. Qwen-VL uses cross-attention, multilingual data, and three training stages

**Transcript coverage:** lines 1316-1408

### What the lecturer said - transcript only

Qwen began releasing multimodal models in 2023. The first Qwen-VL follows the same overall pattern but changes the connector. Its vision encoder is OpenCLIP, the open reproduction of CLIP. A single cross-attention layer with two-dimensional positional information maps the visual representation to a fixed sequence of 256 vectors.

Special tokens mark images, bounding boxes, and referred descriptions. This lets the text decoder emit structured grounding coordinates in addition to ordinary prose.

Training again has three stages. In the first, large-scale lower-quality image-text data train the vision encoder and adapter while the language model is frozen. This differs from LLaVA, which initially freezes the vision encoder. Qwen-VL uses about 1.4 billion cleaned examples at this stage.

The second stage raises image resolution, switches to higher-quality task-specific data, and trains all parameters. The mixture includes visual question answering, charts, grounding, captions, OCR, and pure text. The third stage uses instruction-tuning data, freezes the vision encoder, and trains the adapter and language model.

The resulting model handles Chinese and English, code shown in images, object grounding through bounding boxes, OCR, comparison, and other language-model-like behaviors. It still outputs text or coordinates rather than rendered images.

### Source reconciliation

The slide names the vision encoder as OpenCLIP ViT-bigG with $14\times14$ patches. Its adapter is one cross-attention layer with learned queries, 2D position information, and a fixed output length of 256.

The data table shows approximately **5B original pairs cleaned to 1.4B**, or 28 percent retained. The second-stage table includes large captioning, VQA, grounding, referring-expression, grounded-caption, OCR, and text-autoregression subsets. The final stage freezes the visual encoder for supervised instruction tuning.

### Additional explanation

Cross-attention uses a fixed set of learned queries to compress a variable grid:

$$
Q_{\mathrm{learned}}
\operatorname{Attn}
K_{\mathrm{vision}},V_{\mathrm{vision}}
\rightarrow
256\text{ adapter tokens}.
$$

This bounds LLM context cost but imposes a fixed-capacity bottleneck. Dense documents and tiny images receive the same 256-slot interface. Later dynamic-resolution systems relax that constraint.

Training the vision encoder in stage 1 can adapt it beyond CLIP's classification-oriented semantics. It may improve OCR and localization but requires more compute and risks forgetting useful pretrained structure. Freezing it in the final stage stabilizes the visual front end while the LLM learns instruction behavior.

## 12. Qwen2-VL introduces native dynamic resolution and multimodal RoPE

**Transcript coverage:** lines 1409-1504

### What the lecturer said - transcript only

Qwen2-VL is an upgrade with a larger vision encoder and native dynamic resolution. As in AnyRes, the same model should handle a dense document, a tiny equation, an ordinary photograph, or a video without assigning them all the same token count. One example can map to roughly 11,000 tokens while a small equation uses only eight.

Each $224\times224$ region is encoded by a ViT. To control context length, groups of $2\times2$ visual features are compressed into one, yielding 66 tokens per region in the described setup. Video is sampled at two frames per second with an overall maximum around 16,000 vision tokens.

Qwen2-VL also uses multimodal rotary position embedding, or M-RoPE. Ordinary RoPE represents one-dimensional token distance. Visual positions have width and height, and video adds time. Each patch therefore has a time-height-width coordinate. RoPE is computed for the three axes and the resulting components are concatenated.

The language model is initialized from Qwen2 and the vision encoder from a pretrained vision model. Training resembles Qwen-VL: first train the visual side, then all parameters, then the language model on instruction-following data. The resulting system supports video understanding, math and code, grounding, OCR, function calls, and interface interaction.

### Source reconciliation

The slides specify a **675M-parameter** vision encoder. They give the video cap as 16,384 tokens and illustrate native-resolution examples using 11,427, 8, and 1,125 image tokens plus 2,208 video tokens. The M-RoPE diagram assigns coordinates along time, height, and width.

### Additional explanation

For a token at coordinate $(t,h,w)$, M-RoPE can assign disjoint channel groups to axis-specific rotations:

$$
\operatorname{M\!-\!RoPE}(t,h,w)
=
\operatorname{concat}
\left[
R_t(t),R_h(h),R_w(w)
\right].
$$

The attention dot product then contains relative displacement information for all axes. Text tokens can use a consistent degenerate coordinate convention.

Dynamic resolution makes token count proportional to information demand, but resolution is an imperfect proxy. A visually simple high-resolution photo may receive more tokens than a compact equation whose few pixels carry high semantic density. Data and routing policies still determine whether the budget is well spent.

## 13. Qwen3-VL refines long context, position frequencies, loss balance, and fusion

**Transcript coverage:** lines 1505-1698

### What the lecturer said - transcript only

Qwen3-VL keeps the established architecture but makes several quality-critical refinements. It uses the stronger Qwen3 family, including dense and mixture-of-experts language models, and extends context length to 256K, which is especially important for long video.

The vision encoder becomes SigLIP-2, designed to remain architecture-compatible with SigLIP. The original M-RoPE assigned contiguous channel blocks to time, width, and height. Because RoPE channels correspond to different frequencies, that allocation could give one axis mostly low-frequency channels and another mostly high-frequency channels. Qwen3-VL interleaves the axes so every axis receives a mixture of low and high frequencies.

Video timestamps also become explicit text-like tokens. Earlier models represented time only implicitly through positional coordinates. A token such as "0 seconds" or "2 seconds" can be referred to directly in a question and answer.

The model uses square-root-normalized per-token loss. A video example may contain far more tokens than a single-image example, so treating every token identically lets long videos dominate the mixture. The lecturer says the report's details are not fully clear, but interprets the normalization as downweighting very long examples by the square root of their length.

The adapter also becomes deeper. LLaVA used a linear map, OneVision an MLP, and Qwen-VL cross-attention. Qwen3-VL uses DeepStack, from the DeepSeek team, to inject features from multiple vision-encoder layers into several language-model residual layers. The vision encoder is no longer only a black box that emits one final sequence.

The training pipeline has four pretraining stages and three post-training stages. It first aligns the adapter, then trains at progressively larger context lengths of roughly 8K, 32K, and 256K. Most pretraining tokens occur in the middle stages. Post-training includes long-chain-of-thought SFT, knowledge distillation, and reinforcement learning.

The report has become a complex systems recipe rather than one architectural insight. On a large benchmark suite, Qwen3-VL is highly competitive with closed systems such as Gemini, GPT-5, and Claude Opus 4.1.

### Source reconciliation

The slides identify Qwen3 dense and MoE decoders up to **235B-A22B** and a 256K context. They state:

- SigLIP-2 vision encoder;
- interleaved M-RoPE over temporal, width, and height axes;
- explicit video timestamp tokens;
- square-root-normalized per-token loss;
- DeepStack cross-layer visual fusion.

The pretraining table gives:

| Stage | Objective | Trainable | Token budget | Sequence length |
|---|---|---|---:|---:|
| S0 | Vision-language alignment | Adapter or merger | 67B | 8,192 |
| S1 | Multimodal pretraining | All | about 1T | 8,192 |
| S2 | Long-context pretraining | All | about 1T | 32,768 |
| S3 | Ultra-long-context adaptation | All | 100B | 262,144 |

The benchmark slide supports the lecturer's qualitative claim of state-of-the-art open performance but should not be read as a single universal ranking across every row.

### Additional explanation

Interleaving axes prevents a frequency allocation artifact. Instead of

$$
[t_{\mathrm{low}},\ldots,t_{\mathrm{high}},
h_{\mathrm{low}},\ldots,
w_{\mathrm{high}}],
$$

the channels cycle through axes across the frequency spectrum. Each axis can then represent both broad displacement and fine local changes.

If a completion has length $L$, three common loss aggregations are:

$$
\sum_{t=1}^{L}\ell_t
\quad\text{(long examples dominate linearly)},
$$

$$
\frac{1}{L}\sum_{t=1}^{L}\ell_t
\quad\text{(every example has equal total weight)},
$$

$$
\frac{1}{\sqrt L}\sum_{t=1}^{L}\ell_t
\quad\text{(intermediate square-root weighting)}.
$$

The last retains more total weight for longer examples while preventing a video from dominating purely in proportion to token count.

DeepStack supplies multi-scale visual features. Earlier vision layers often preserve local edges and texture; later layers emphasize semantics. Injecting several depths gives the LLM access to both, at the cost of tighter architectural coupling.

## 14. Q&A: the discussed VLMs understand non-text inputs but still emit text

**Transcript coverage:** lines 1699-1720

### What the lecturer said - transcript only

A student asks about video generation and how a model decides whether to output video or text. The lecturer clarifies that the LLaVA and Qwen systems discussed so far do not generate video or images. Multimodality is on the input side; the autoregressive output remains text.

Except during reinforcement-learning stages, training directly supervises the target tokens from the dataset. There is not necessarily a separate language-model judge rating whether a free-form visual description is good. RL can introduce other reward choices, but ordinary supervised stages learn from whatever answer appears in the corpus.

### Additional explanation

These systems implement

$$
(x_{\mathrm{text}},x_{\mathrm{image}},x_{\mathrm{video}})
\rightarrow
y_{\mathrm{text}},
$$

not the full omni mapping

$$
\{\text{any inputs}\}
\rightarrow
\{\text{any outputs}\}.
$$

The output vocabulary and decoder determine what can be emitted. A text-only LLM with continuous vision prefixes has no image-token vocabulary or pixel decoder. Image generation therefore needs a discrete image tokenizer, a diffusion decoder, or another dedicated output head.

## 15. Q&A: multimodal systems add data-loading and mixture-weighting problems

**Transcript coverage:** lines 1721-1781

### What the lecturer said - transcript only

A student asks whether multimodal training is harder than pure language-model training from a systems perspective. It is certainly not easier. Image and especially video datasets are large, and simply loading and decoding video can become a bottleneck. Text loading was cheap enough not to dominate earlier course discussions. Multimodal input pipelines must decode, augment, batch, and transfer data asynchronously so accelerators do not wait.

The student also asks whether modalities with many more tokens dominate. The square-root normalization is one response. Builders can also downweight a modality through the data-mixture machinery discussed earlier in the course.

The lecturer does not believe multimodal tokens necessarily outnumber text tokens in current large-model training. Text corpora still contain tens of trillions of tokens. The practical issue is choosing weights that reflect learning value rather than allowing raw token length to decide the mixture.

### Additional explanation

Multimodal throughput has at least three bottlenecks:

$$
\text{storage bandwidth}
\rightarrow
\text{decode and augmentation}
\rightarrow
\text{accelerator compute}.
$$

Compressed video is cheap to store relative to decoded frames but expensive to decode. Variable resolutions complicate packing. A robust pipeline prefetches and decodes asynchronously, groups similar shapes, and monitors accelerator idle time separately from model FLOPs.

Mixture weight is not the same as dataset size:

$$
p(m)
\ne
\frac{\text{raw tokens in modality }m}
{\text{all raw tokens}}.
$$

The sampling probability, tokens per example, and loss normalization jointly determine the effective gradient contribution.

## 16. Q&A: alignment assumes pretrained components, and most capacity remains in the LLM

**Transcript coverage:** lines 1782-1852

### What the lecturer said - transcript only

A student asks how the adapter alignment stage knows the language space and whether the language model is already pretrained. It must be pretrained; otherwise there is no stable language representation to align with. In the basic alignment stage, the language model is frozen and only the adapter is trained to connect a chosen vision encoder to that fixed LLM. Training uses a predetermined token budget rather than an adaptive stopping threshold; the Qwen3 example uses 67 billion tokens.

Another question asks whether the vision encoder is much smaller than the language model. Usually it is. The lecturer explains that a vision encoder performs comparatively local patch processing and does not contain most of the broad knowledge or reasoning capability. Those abilities remain in the language model.

In the cited LLaVA-scale comparison, the projector is very small, the vision transformer is generally below a billion parameters, and the language model can be tens of billions of parameters. The lecturer corrects an initially misread unit while making this comparison.

### Source reconciliation

The LLaVA-OneVision training table illustrates the scale separation: projector sizes range from about 1.8M to 72M depending on the LLM, while the example language models range from 0.5B to 72.7B. The Qwen3-VL table verifies the 67B-token alignment stage.

### Additional explanation

Parameter count reflects the chosen division of labor:

- The vision encoder converts local pixels into structured features.
- The adapter translates and compresses those features.
- The LLM supplies language, world knowledge, instruction following, and much of the cross-token reasoning.

Calling the vision encoder "local" does not mean it is unimportant or literally limited to one patch after its transformer layers. It means the system architecture concentrates general-purpose capacity in the LLM. A weak or lossy encoder still caps visual performance, especially on OCR, counting, and spatial grounding.

## 17. The Qwen progression is mostly better data, scale, context, and refinements

**Transcript coverage:** lines 1853-1881

### What the lecturer said - transcript only

Qwen3-VL is the final vision-language model in the lecture and represents state-of-the-art open performance. Substantial data work underlies it, but later Qwen reports reveal relatively little about the exact mixture. The LLaVA papers and an AI2 report offer more data detail for readers who want to study construction.

From Qwen-VL through Qwen2-VL and Qwen3-VL, the high-level framework stays stable. Improvements come from scaling, stronger curation, longer context, dynamic-resolution processing, and architectural sharpening rather than abandoning the vision-encoder, adapter, and language-model template.

### Source reconciliation

The Qwen summary slide lists four themes: state-of-the-art performance, extensive but incompletely documented data work, small potentially important architectural improvements, and scale.

### Additional explanation

This is a mature-system pattern:

$$
\text{stable macro-architecture}
+\text{better components}
+\text{better data}
+\text{more context}
+\text{more compute}.
$$

When reports omit the mixture, it becomes difficult to attribute benchmark gains to architecture. A new position encoding may matter, but so may a newly distilled OCR corpus or hidden evaluation overlap. Open data releases are therefore scientifically valuable even when their absolute scores trail a closed system.

---

# Part III - Discrete image tokens and multimodal generation

## 18. Chameleon makes text and images one autoregressive discrete-token stream

**Transcript coverage:** lines 1882-1940

### What the lecturer said - transcript only

Meta's 2024 Chameleon takes a different route. The vision-language models discussed so far encode images into continuous vectors, inject them into a language model, and generate only text. One could attach a diffusion head for image generation. Chameleon instead asks whether every modality can be mapped to discrete tokens.

This approach is aesthetically appealing from a language-model perspective because understanding and generation use one interface. A text prompt can be followed by generated image tokens. Text and images can alternate repeatedly in one response. The paper demonstrates interleaved outputs, such as a request for interesting birds followed by descriptions and generated bird images.

In this vision of an omni model, text and images live in the same autoregressive stream because both have been made to look like discrete token sequences.

### Source reconciliation

The slides show one mixed-modal autoregressive language model consuming text and tokenized image spans, then emitting either text tokens or image tokens delimited by start- and end-image markers. An image de-tokenizer converts generated image-token spans back into pixels.

### Additional explanation

The factorization becomes

$$
p(s_1,\ldots,s_T)
=
\prod_{t=1}^{T}
p(s_t\mid s_{<t}),
$$

where each $s_t$ may be a text token, image code, or modality delimiter. The transformer no longer needs a separate continuous-prefix interface.

Uniform syntax does not imply uniform statistics. Text and image codes have different entropy, local dependence, and perceptual error. Chameleon demonstrates that a common vocabulary interface simplifies the model graph while shifting difficulty into tokenization, balancing, and optimization.

## 19. VQ-VAE converts a continuous image representation into codebook IDs

**Transcript coverage:** lines 1941-2021

### What the lecturer said - transcript only

Chameleon needs a discrete image tokenizer. It uses the older vector-quantized variational autoencoder, or VQ-VAE, idea introduced by van den Oord and colleagues in 2017.

An encoder maps an image to a grid of continuous latent vectors. Each latent is replaced by the nearest entry in a learned codebook, for example one of about 8,000 prototype vectors. The selected codebook index is the discrete image token. A decoder maps the quantized grid back toward the original image.

Training minimizes reconstruction error along with extra terms needed because nearest-neighbor quantization is not differentiable. The lecture does not derive those additional terms.

In the described Chameleon setup, a $512\times512$ image becomes 1,024 tokens drawn from a vocabulary of roughly 8,000 entries. The model also trains a new text BPE tokenizer because the joint corpus differs from ordinary text-only data.

Once both media are discrete, the main transformer training looks like ordinary next-token language modeling. There is no separate vision adapter in the mixed-modal LM. Training still has stages: a large mostly unsupervised phase over text, paired text-image data, and interleaved documents, followed by a phase that mixes in higher-quality data.

### Source reconciliation

The slide gives the exact image codebook size as **8,192** and confirms 1,024 codes per $512\times512$ image. Its stage budgets are:

- Stage 1, 80 percent of training: 2.9T text tokens, 1.5T text-image tokens, and 400B interleaved text-image tokens.
- Stage 2, 20 percent: half resampled Stage 1 data and half higher-quality data.

### Additional explanation

For encoder latent $z_e(x)_{hw}$ and codebook vectors $e_k$, quantization chooses

$$
k^*(h,w)
=
\arg\min_k
\left\|
z_e(x)_{hw}-e_k
\right\|_2^2.
$$

The discrete token is $k^*(h,w)$ and the decoder reconstructs from $e_{k^*}$. A common training objective contains:

$$
\mathcal L
=
\mathcal L_{\mathrm{recon}}
+\mathcal L_{\mathrm{codebook}}
+\beta\mathcal L_{\mathrm{commit}}.
$$

Straight-through gradient estimation is commonly used to pass learning signal across the non-differentiable nearest-code selection. These details are additional background; the lecture only notes that extra terms are required.

Compression is substantial. A $512\times512$ RGB image contains 786,432 channel values but is represented by 1,024 categorical IDs. Whether the loss is acceptable depends on the task. A plausible reconstruction can still alter a small printed character and fail OCR.

## 20. Unified discrete modeling is elegant but unstable and lossy

**Transcript coverage:** lines 2022-2101

### What the lecturer said - transcript only

Chameleon's uniform formulation has several problems. Training was unstable because text and image tokens behave differently even after both are assigned integer IDs.

Next-text-token entropy is comparatively low: many words are predictable from context. Exact image codes have higher entropy because local color, texture, and shade remain uncertain. Mixing these prediction problems caused parameter norms to grow and logits or loss to become unstable. QK normalization and z-loss regularization controlled norm growth enough to mitigate the problem.

The model was also less performant than leading alternatives, and image discretization loses information. Small print may be destroyed, making OCR impossible even if the reconstructed image looks reasonable. Multimodal weighting is difficult in the continuous-encoder Qwen systems and becomes more severe when every image code directly participates in the same autoregressive loss.

VQ-VAE image generation was popular when transformers required discrete outputs. Diffusion models later became effective and widely adopted for generation, reducing the need to force images into categorical autoregressive tokens.

The lecturer includes Chameleon for its conceptual elegance: one model treats modalities uniformly. Its disadvantages explain why the approach has not displaced continuous encoders for understanding and diffusion for high-fidelity generation.

### Source reconciliation

The final Chameleon slide explicitly identifies low-entropy text versus high-entropy image tokens, norm growth, and logit drift. It lists QK norm and z-loss as the stabilizers. Its summary calls the method elegant but not as performant, highlights OCR information loss, and notes the difficulty of training multiple modalities together.

### Additional explanation

The expected cross-entropy scale differs by modality:

$$
H_{\mathrm{text}}
\ne
H_{\mathrm{image\ codes}}.
$$

If one optimizer and unweighted token average combine them, image positions can contribute larger gradients or different logit pressure. The model may increase representation norms to sharpen a hard categorical distribution.

QK normalization controls attention-logit scale by normalizing queries and keys. A z-loss penalizes growth in the log-partition term, schematically:

$$
\mathcal L_z
=
\lambda
\left[
\log\sum_j e^{z_j}
\right]^2.
$$

Neither restores information discarded by the tokenizer. Optimization stability and representation fidelity are separate failure modes.

Diffusion avoids committing to a single coarse code sequence early. It iteratively denoises a continuous or latent representation and can refine low- and high-frequency detail, which is well matched to perceptual generation.

## 21. Final synthesis: understanding and generation need different information

**Transcript coverage:** lines 2102-2187

### What the lecturer said - transcript only

Frontier systems are now expected to be multimodal, natively multimodal, or omni. Gemini and GPT releases advertise such abilities but disclose little about their construction. The lecturer speculates that strong closed systems probably combine continuous encoders for understanding with diffusion for generation, but labels this as speculation rather than known fact.

The fundamental challenge is representing non-text modalities. Understanding and generation are asymmetric. CLIP-style classification needs high-level semantics, so a compact representation can discard local details. OCR requires exact fine structure. Image generation requires still more low- and high-frequency detail, which is one reason diffusion works well.

There is no universal encoder that is best for all these tasks. Multimodal mixtures also need careful weighting. Video has lower information density than text because adjacent frames repeat much of the same content, so raw video-token count should not overwhelm the text objective.

The current practical pattern is continuous encoders, often still based on CLIP-like ideas, feeding transformers for semantic understanding, with diffusion models used for generation. Students are encouraged to experiment with training these systems even though the course has no homework on the topic.

### Source reconciliation

The lecture outline and final summary slides state the same four conclusions:

1. frontier models are expected to be multimodal or omni;
2. encoding non-text modalities is the foundational problem;
3. understanding and generation demand different levels of semantic and fine-grained information;
4. multimodal training must balance image, video, and text while current strong systems favor continuous encoders plus transformers and diffusion generation.

### Additional explanation

The design space can be summarized by the representation contract:

| Goal | Useful representation | Main risk |
|---|---|---|
| Retrieval or classification | Compact semantic embedding | Loses spatial and textual detail |
| Visual question answering | Sequence or grid of continuous features | Long context and adapter bottleneck |
| OCR and grounding | High-resolution local features plus position | Excessive vision-token count |
| Autoregressive image generation | Discrete image code sequence | Quantization loss and high entropy |
| Diffusion generation | Continuous or latent spatial field | Separate generation architecture and iterative cost |

An omni system need not use one internal representation for every direction. It can route each input through a modality-appropriate encoder, reason in a shared transformer, and invoke a modality-specific decoder. "Native" can describe an integrated training and interface even when internal modules differ.

---

# Consolidated study material

## Key takeaways

1. Multimodality begins with tokenization: transformers require every input to become a sequence of discrete IDs or continuous vectors.
2. The right visual token depends on the task. Classification needs semantics; OCR and generation need fine detail.
3. CLIP learns a shared image-text space with symmetric in-batch contrastive classification.
4. CLIP's large web corpus replaces explicit class labeling but introduces noisy captions, false negatives, and filtering dependence.
5. ViTs turn image patches into transformer tokens, while CLIP's objective determines which information those tokens preserve.
6. SigLIP replaces the batch-global softmax with pairwise logistic classification, improving decomposability and small-batch behavior.
7. The standard open VLM template is a pretrained vision encoder, an adapter or projector, and a pretrained autoregressive LLM.
8. LLaVA first aligns a connector with frozen components, then tunes the connector and LLM on synthetic visual instructions.
9. AnyRes and native dynamic resolution trade token budget for high-resolution evidence rather than cropping everything to one square.
10. Multi-image and video systems deliberately reduce tokens per view to prevent visual input from exhausting context.
11. Diverse targeted tasks can transfer compositionally across single images, image sets, video, OCR, and GUI sequences.
12. Qwen-VL evolves from a fixed cross-attention bottleneck to dynamic resolution, 3D M-RoPE, interleaved frequencies, timestamp tokens, and deep fusion.
13. Loss normalization and modality sampling jointly decide whether long videos dominate training.
14. The LLaVA and Qwen systems discussed are multimodal on input and text-only on output.
15. Most broad reasoning capacity remains in the LLM, but visual fidelity is capped by the encoder and connector.
16. Chameleon unifies text and images as discrete autoregressive tokens and therefore supports interleaved generation.
17. VQ-VAE quantization simplifies the decoder interface but discards information and creates a high-entropy image-token objective.
18. A common token vocabulary does not erase statistical differences between modalities.
19. Continuous semantic encoders currently remain strong for understanding, while diffusion is well suited to fine-grained generation.
20. "Omni" is an input-output capability goal, not a requirement that every modality share one tokenizer or decoder.

## Key equations

### CLIP pairwise similarity

$$
s_{ij}
=
\frac{
E_{\mathrm{img}}(x_i)^\top
E_{\mathrm{text}}(y_j)
}{\tau},
$$

with both embeddings L2 normalized.

### Symmetric CLIP loss

$$
\mathcal L_{\mathrm{CLIP}}
=
\frac{1}{2}
\left(
\mathcal L_{\mathrm{image\rightarrow text}}
+\mathcal L_{\mathrm{text\rightarrow image}}
\right).
$$

### SigLIP binary objective

$$
\mathcal L_{\mathrm{SigLIP}}
=
-\frac{1}{N}
\sum_{i,j}
\log\sigma
\left[
z_{ij}
\left(
t\,u_i^\top v_j+b
\right)
\right].
$$

### Vision-token projection into an LLM

$$
H_v=E_{\mathrm{vision}}(x)W,
\qquad
W\in\mathbb R^{d_v\times d_{\mathrm{lm}}}.
$$

### ViT patch count

$$
M
=
\left(\frac{H}{P}\right)
\left(\frac{W}{P}\right).
$$

### Autoregressive mixed-modal factorization

$$
p(s_{1:T})
=
\prod_{t=1}^{T}
p(s_t\mid s_{<t}).
$$

### Vector quantization

$$
k^*
=
\arg\min_k
\|z_e(x)-e_k\|_2^2.
$$

### Square-root example weighting

$$
\mathcal L_{\mathrm{example}}
=
\frac{1}{\sqrt L}
\sum_{t=1}^{L}\ell_t.
$$

## Glossary

- **Adapter or projector:** A trainable module that maps vision-encoder features into the LLM's embedding interface.
- **Alignment stage:** Training that connects frozen pretrained modalities, usually by fitting only the adapter.
- **AnyRes:** A high-resolution strategy that combines a global thumbnail with independently encoded image tiles.
- **Attention pooling:** Compressing a sequence of vision features with an attention query rather than a plain average.
- **Chameleon:** Meta's mixed-modal autoregressive model using discrete text and image tokens.
- **CLIP:** Contrastive Language-Image Pre-Training, which learns aligned image and text embeddings.
- **Codebook:** A learned set of prototype vectors whose indices form discrete visual tokens.
- **Contrastive learning:** Learning representations by increasing similarity for aligned pairs and decreasing it for alternatives.
- **Continuous visual token:** A vector produced by a vision encoder and supplied directly to a transformer.
- **DeepStack:** Cross-layer fusion that injects features from several vision depths into several LLM layers.
- **Discrete visual token:** A categorical code representing a quantized image region.
- **Dynamic resolution:** Allowing image size and visual token count to vary with the input.
- **False negative:** An off-diagonal pair treated as unrelated even though its image and text are semantically compatible.
- **Image-text pair:** An image and associated caption, alt text, OCR text, or other language supervision.
- **Interleaved multimodal data:** A sequence in which text and image spans alternate within one document or response.
- **LLaVA:** Large Language and Vision Assistant, an open VLM built from CLIP, a projector, and an LLM.
- **M-RoPE:** Multimodal rotary position embedding over time, height, and width.
- **Omni model:** A system intended to accept and emit arbitrary combinations of modalities.
- **OpenCLIP:** An open reproduction and extension of CLIP trained with public large-scale datasets.
- **Patch token:** A vector obtained by flattening and projecting a fixed image patch.
- **QK norm:** Normalization of attention queries and keys to control attention-logit scale.
- **SigLIP:** Sigmoid Loss for Language-Image Pre-Training, using pairwise binary classification.
- **Temperature:** A learned or fixed scale controlling the sharpness of similarity logits.
- **VLM:** Vision-language model, typically accepting visual and textual input and generating text.
- **ViT:** Vision Transformer, which processes a sequence of projected image patches.
- **VQ-VAE:** Vector-quantized variational autoencoder, which represents encoder latents with nearest codebook entries.
- **WebLI:** Google's large multilingual web image-language dataset used by SigLIP.
- **z-loss:** A penalty on log-partition magnitude used to discourage logit norm growth.
- **Zero-shot classification:** Selecting a label by comparing an image embedding with embeddings of label prompts without task-specific classifier training.

## Self-check questions

1. What two separate problems must an omni model solve for each non-text modality?
2. Why is a raw pixel a poor analogue of a text subword token?
3. How does CLIP turn a batch of $N$ pairs into two sets of classification problems?
4. Why do large batches help the original CLIP objective?
5. What information can center cropping destroy while preserving ImageNet classification?
6. Why does text pairing encourage more semantic representations than image augmentation alone?
7. How does a ViT convert an image into a transformer sequence?
8. Why can CLIP perform zero-shot classification from class-name prompts?
9. What is an in-batch false negative?
10. Why was autoregressive caption prediction less efficient for CLIP's ImageNet representation goal?
11. How does SigLIP's loss differ from CLIP's global softmax?
12. Why is SigLIP easier to decompose across devices?
13. What does LLaVA's projector align, and what remains frozen in its first stage?
14. Why can a two-stage LLaVA recipe reuse a pretrained LLM's knowledge?
15. How does AnyRes preserve both global layout and local OCR detail?
16. Why are video frames assigned fewer tokens than a single still image?
17. What kind of transfer did LLaVA-OneVision show across single-image, multi-image, and video data?
18. How does Qwen-VL's fixed cross-attention interface differ from native dynamic resolution?
19. What coordinates does M-RoPE encode?
20. Why does Qwen3-VL interleave axes across RoPE frequencies?
21. What benefit do explicit video timestamp tokens provide?
22. How does square-root loss normalization sit between token averaging and example averaging?
23. What additional information can DeepStack expose compared with a final-layer-only vision adapter?
24. Why can the input-side VLMs not generate an image?
25. Which systems problems appear when video enters the training mixture?
26. Why is the vision encoder typically much smaller than the language model?
27. How does Chameleon make interleaved text and image generation possible?
28. What does VQ-VAE quantization discard?
29. Why do text and image token losses have different entropy and optimization behavior?
30. What do QK norm and z-loss address, and what do they not address?
31. Why might continuous encoders and diffusion decoders coexist in an omni system?
32. How would you choose a representation differently for classification, OCR, and image generation?

## Source coverage checklist

| Topic | Transcript lines | Slides checked | Coverage note |
|---|---:|---:|---|
| 1. Omni-model goal and token interface | 1-137 | 1-3 | Scope, modalities, transformer tokens, and input versus output questions |
| 2. CLIP objective | 138-244 | 4-5 | Historical motivation, batch construction, symmetric classification, and pseudocode |
| 3. CLIP data and preprocessing | 245-320 | 5-6 | 400M pairs, OpenCLIP or LAION, resize, and crop |
| 4. CLIP encoders and Q&A | 321-496 | 6-7 | SimCLR contrast, ViT patches, pooling, 2D position, and GPT-2 text encoder |
| 5. CLIP results and noise | 497-616 | 7 | Zero-shot ImageNet, noisy captions, false negatives, and generative-objective ablation |
| 6. SigLIP | 617-817 | 8-9 | Pairwise sigmoid loss, WebLI, negative-sampling Q&A, distributed computation, and batch behavior |
| 7. LLaVA construction | 818-951 | 10-11 | CLIP, Vicuna, MS COCO, GPT-4 synthesis, projector, and 158K examples |
| 8. LLaVA training and result | 952-1009 | 12-13 | Alignment, fine-tuning, and extreme-ironing example |
| 9. OneVision architecture and AnyRes | 1010-1181 | 14-15 | SigLIP, Qwen2, MLP, high resolution, multiple images, and video token budgets |
| 10. OneVision data and transfer | 1182-1315 | 16-21 | Mixtures, staged curriculum, GPT-4 distillation, cross-format transfer, and open data |
| 11. Qwen-VL | 1316-1408 | 22-25 | OpenCLIP, cross-attention, cleaning, three stages, multilingual and grounding abilities |
| 12. Qwen2-VL | 1409-1504 | 25-28 | Dynamic resolution, token compression, video sampling, M-RoPE, and capabilities |
| 13. Qwen3-VL | 1505-1698 | 29-31 | Qwen3, SigLIP-2, interleaved position, timestamps, loss weighting, DeepStack, and seven stages |
| 14. Output-modality Q&A | 1699-1720 | 3, 32 | Text-only output and supervised target tokens |
| 15. Systems and mixture Q&A | 1721-1781 | 15, 30 | Video loading, asynchronous pipelines, token length, and weighting |
| 16. Alignment and parameter-size Q&A | 1782-1852 | 12, 18, 30 | Frozen pretrained LLM, 67B-token alignment stage, and capacity split |
| 17. Qwen family summary | 1853-1881 | 31-32 | Data opacity, stable framework, refinement, and scale |
| 18. Chameleon motivation | 1882-1940 | 32-34 | Discrete unification and interleaved generation |
| 19. VQ-VAE and Chameleon training | 1941-2021 | 35 | Codebook, reconstruction, 1,024 tokens, 8,192 codes, BPE, and stage budgets |
| 20. Chameleon limitations | 2022-2101 | 35-36 | Entropy mismatch, norm growth, QK norm, z-loss, quantization loss, and diffusion |
| 21. Final synthesis | 2102-2187 | 3, 36 | Frontier expectations, disclosed versus speculative design, task-dependent fidelity, mixture balance, and closing |

**Transcript accounting:** the 21 spans cover lines 1-2187 exactly once, with no gaps or overlaps.

**Slide accounting:** all 36 pages were rendered and visually inspected. Every slide is included in at least one checklist range.
