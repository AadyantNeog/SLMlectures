# Lecture Notes Production Standard

## Canonical example

Use `../lecture_01_overview_tokenization.md` as the structural and quality reference.

## Required source order

1. Read and map the complete transcript before drafting.
2. Write the transcript-grounded account from the transcript only.
3. Inspect every rendered slide after the transcript map is established.
4. Use slides to verify structure, names, notation, equations, code, charts, and figures.
5. Keep slide-only facts out of transcript-only sections. Put material discrepancies in `Source reconciliation`.
6. Add independent explanation only after the grounded account is defined.

## Required per-topic structure

```markdown
## N. Topic

**Transcript coverage:** lines X-Y

### What the lecturer said - transcript only

Clean, concise paraphrase preserving every substantive claim, definition,
example, qualification, number, warning, and useful question or answer.

### Source reconciliation

Only when the slides and spoken lecture differ materially.

### Additional explanation

Clearly separate intuition, derivation, examples, connections, and caveats.
```

Nested concepts may use level-four headings inside the transcript-only section, but source boundaries must remain unambiguous.

## Completeness standard

- Preserve all substantive content, not all verbal repetition.
- Remove filler, false starts, and duplicated phrasing.
- Include course logistics only when they convey meaningful pedagogy, requirements, or context.
- Include substantive audience questions and answers.
- Mark genuinely ambiguous transcript passages instead of guessing.
- Add a final source-coverage checklist that spans the complete raw transcript.

## Required closing material

- Consolidated takeaways
- Key equations, when applicable
- Glossary
- Self-check questions
- Source coverage checklist

## Formatting

- Markdown is the canonical output.
- Use YAML front matter matching Lecture 1.
- Use zero-padded filenames: `lecture_02_...md`.
- Use `$...$` for inline mathematics and `$$...$$` for display mathematics.
- Use fenced code blocks with a language tag when appropriate.
- Use ASCII hyphens rather than typographic dash characters.
- Keep tables compact and readable.
- Use relative source paths in front matter.

## Verification

- Count transcript-only and supplementary sections; they should correspond conceptually.
- Confirm source files exist.
- Check balanced code fences and math delimiters.
- Check heading hierarchy and trailing whitespace.
- Verify representative names, numbers, equations, and examples against the transcript.
- Remove temporary slide renders after inspection.
