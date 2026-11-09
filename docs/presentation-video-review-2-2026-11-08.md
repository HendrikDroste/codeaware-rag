# Second presentation recording: proposed improvements

Source: `/Users/hendrikdroste/Desktop/Codeaware_2.mp4`.
Reviewed slide frames and a locally generated automated transcript. Timestamps
are approximate; transcription errors are not evidence of pronunciation errors.
No presentation edits were made.

## Overall recommendation

Keep the five-slide structure. The added walkthrough and engineering decisions
make the story more complete. Focus the next rehearsal on precise technical
claims, less repetition and a stronger final decision. Animations can wait.

The file lasts 10:49.9; the transcribed main explanation finishes around 10:20.
Avoid depending on speaking faster. Recover 40–60 seconds by removing repetition,
particularly on architecture, rather than rushing the evidence or conclusion.

## Priority corrections to the explanation

1. **3:22–3:45: chunks are not guaranteed to contain exactly one function.**
   The implementation uses Python-aware recursive character splitting with a
   900-character limit and zero overlap. Complete definition content is stored
   separately. Say: “The vector index stores small code chunks. The graph stores
   definitions and resolved relationships.” A chunk can split a large function
   or contain multiple small definitions.
2. **4:25–4:45: explain the direction and role of tool use.**
   The host queries the index through read-only tools. It does not create the
   database through these tools. Say: “I build the index beforehand. The host
   selects tools to retrieve context and uses the returned evidence to answer.”
3. **Around 9:21–9:40: a quality tie is not an identical answer.**
   The review found equal rubric scores on 17 question pairs, not identical
   wording or an independent equivalence result. Say: “The review found 17 ties
   and one baseline completeness advantage.” Do not dismiss the baseline result
   as negligible; identify it briefly as an omitted detail if asked.
4. **Around 9:40–10:00: bound the conclusion.**
   A working tool and lower reported tokens in one run justify further testing;
   they do not establish financial impact or confirm the full efficiency
   hypothesis. Keep the headline scoped to total reported tokens in this run.
   The unchanged underlying benchmark accounts for cached input; do not turn
   the number into a claim about costs or newly supplied context.
5. **Around 8:18–8:45: avoid broad claims about language portability.**
   Say: “The architecture could be extended, but each language needs suitable
   extraction and resolution. I have implemented and tested Python only.”
6. **Around 8:45–9:05: distinguish freshness detection from updates.**
   Say: “Source hashes detect changes. Retrieval then rejects the stale index
   until I rebuild and reconnect.” Hashes do not update the graph automatically.

## Slide-by-slide proposals

### 1. Problem and hypothesis

- Keep the screenshots and hypothesis.
- Add or say one concrete user situation: a developer investigates dependencies
  before assessing a change. This establishes intended value, not measured savings.
- Explain only that Blueprint's context spans files. Leave the detailed resolution
  sequence to the walkthrough, so the same inheritance chain is not explained twice.
- Avoid presenting inheritance as a universal requirement for good-quality code.
  It is an example of a cross-file dependency, not an argument for one design style.

Suggested opening: “Coding agents already search and write code. I wanted to test
whether a code index combining semantic search and explicit relationships could
help them gather useful context more efficiently.”

### 2. Architecture

- Keep the two rows: building the index and querying it.
- Clarify roles in speech: Tree-sitter extracts syntax; Jedi resolves definitions;
  embeddings support search by meaning.
- Add compact capability labels if changing the slide: “Find by meaning” beside
  chunks/vectors and “Follow relationships” beside code definitions/relationships.
- Replace “trace data” with “follow code relationships”: the implementation is
  not a data-flow analysis system.
- Cut the extended explanation of embeddings and the general graph-database
  description. These are natural code-discussion topics.

### 3. Walkthrough

- Keep the three steps and actual tool names.
- Remove the subtitle that repeats the title. Keep the concrete question.
- Explain disambiguation explicitly: choose the public Blueprint using its path,
  then pass the returned definition ID. The tool does not guess the intended match.
- Explain that a second inheritance call follows the first returned parent.
- Distinguish a graph definition node from an embedding chunk. A class node need
  not correspond to one vector chunk.
- Avoid claiming that this example demonstrates faster performance; it shows
  functionality, while the separate evaluation tests the combined tools.

### 4. Engineering decisions

- Preserve the explanation that Python support limits prototype complexity.
- Make the database trade-off correspond to the choice: one service means fewer
  components, but couples both retrieval capabilities to Neo4j. An external
  service is a property shared by both the one- and two-database alternatives.
- Explain snapshots as a deliberate prototype boundary. There is no need to
  speculate about support for almost every programming language.

### 5. Evaluation and proposed pilot

- Keep the native chart and quality counts. Do not expand the main slide into a
  methodology lecture; one clear sentence can state the paired comparison.
- Replace the conclusion bullet repeating the token headline with:
  **“Continue with broader validation.”**
- Make the second bullet more diagnostic:
  **“More repositories, relationship-heavy questions and repeated runs.”**
- The proposed developer pilot is too easy to miss as a long gray footer line.
  Integrate it into the right-hand conclusion area and shorten it:
  **“Developer pilot: dependency investigations”**
  **“Measure correctness, completion time and operating effort.”**
- Keep it explicitly proposed. The recorded technical evaluation did not measure
  developer productivity, completed changes or financial savings.

## Speaking improvements

- Prefer present-tense implementation statements: “The index stores…”, “The host
  queries…”, “The tool returns…”. Use conditional language for untested benefits.
- Replace repeated “I would say”, “kind of” and “for example” with a direct statement
  followed by one example. Do not compensate by speaking faster.
- Use one transition per slide: problem → implementation → example → choices →
  evidence and next decision.
- End with a firm next decision rather than an open-ended claim of impact:
  “The prototype works. This run motivates broader validation. Next I would test
  relationship-heavy tasks across repositories, then evaluate developer outcomes.”

## Evidence checked

- Current code's indexing/chunking/tool behavior, as reviewed in the code checklist.
- `benchmarks/results/2026-11-07-neo4j-low-full-guided/quality-review.md`: 17 rubric
  ties and one baseline completeness advantage; retrospective unblinded review.
- `benchmarks/flask/README.md`: the questions retain the original dataset wording;
  source labels were updated and pinned. No new question set was generated here.

No slide updates, benchmark runs or animation changes were performed for this review.
