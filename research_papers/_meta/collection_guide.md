# Research-paper collection guide

## Purpose

Create a curated companion bibliography for each lecture in Stanford CS336: Language Modeling from Scratch, Spring 2026. The collection should help a reader move from the lecture to the most useful foundational, representative, and modern research papers.

## Source and verification rules

- Read the companion file in `../../lecture_notes/` before selecting papers.
- Prefer primary paper pages: arXiv abstract pages, OpenReview, proceedings, ACL Anthology, ACM/IEEE, USENIX, JMLR, or an official project/publisher page.
- Verify every title, year, and link on the primary page. Do not link to search-result pages, blogs, or citation aggregators when a primary page exists.
- A paper may appear in more than one lecture when it genuinely connects the topics, but each lecture-specific explanation must say why it matters there.
- Include foundational papers and modern work through 2026 where useful. Do not force novelty when an older paper remains the clearest reference.
- Summaries must be original paraphrases, not copied abstracts.

## Target scope

- Usually 10-14 papers per lecture; use fewer or more only when the lecture's breadth warrants it.
- Cover the lecture's major conceptual clusters rather than trying to attach a paper to every minor aside or classroom question.
- Order entries as a coherent reading path: foundations first, then core systems/method papers, then modern extensions.

## File structure

```markdown
---
title: "Lecture N - Topic: Research Paper Guide"
course: "Stanford CS336: Language Modeling from Scratch, Spring 2026"
lecture: N
companion_notes: "../lecture_notes/lecture_XX_name.md"
status: "complete"
---

# Lecture N: Topic - Research Paper Guide

> Summaries are concise original paraphrases. Links point to primary paper or proceedings pages.

## Reading map

Short explanation of the intellectual path and a small table grouping papers by lecture theme.

## Curated papers

### 1. [Exact paper title](primary URL) - Year

**Connects to:** Lecture concepts.

**Very short abstract:** Two or three concise sentences stating the problem, method, and main contribution without inventing numerical results.

**Why it is useful:** One concise sentence about what the reader gains.

**Bigger picture:** One concise sentence explaining how it relates to earlier/later work or the field's trajectory.

## Suggested reading order

Numbered tiers such as `Start here`, `Then deepen`, and `Modern extensions`, referencing entry numbers and titles.
```

## Quality bar

- Exact title and working primary link for every entry.
- No copied abstracts and no unsupported performance claims.
- Clear coverage of the lecture's main themes.
- Sequential entry numbering, clean Markdown, no trailing whitespace, and a final newline.
- Companion-note path resolves.
