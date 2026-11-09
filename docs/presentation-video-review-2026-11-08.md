# Presentation video review — 8 November 2026

Source: `/Users/hendrikdroste/Desktop/Codeaware_Interview_1.mp4`.
Reviewed slide frames and locally generated narration transcription.
Timestamps are approximate; transcription errors were not treated as speaking errors.
No slides or animations were changed.

## Overall recommendation

Keep the cover and three main content slides. The visual sequence is coherent:
problem → implementation → evaluation. Keep the original Flask screenshots,
architecture diagram and native results chart. The strongest improvements are a
concrete running example, accurate technical explanations and a clearer evaluation
decision. The recording lasts about 5:41; the spoken main presentation ends around
5:03, followed by appendix navigation. Aim for approximately six minutes of main
content, without adding slides.

## Slide 1 — Code context spans multiple files

Current segment: approximately 0:13–1:28.

- Keep the title and screenshots.
- Replace the generic left-hand explanation with a concrete question:
  **“Which classes does Flask's Blueprint inherit from, and where are they defined?”**
- Retain a short goal below it: **“Find relevant code. Follow its relationships.”**
- Walk through the alias in the first screenshot, then the parent and grandparent
  definitions. Explain why relationships add context beyond a retrieved fragment.
- Briefly state a potential application: helping developers investigate dependencies
  when onboarding or assessing changes. Present this as intended value, not measured
  productivity improvement.
- Avoid claiming that coding agents generally lack navigation or IDE capabilities.
  Use: “Coding agents already search repositories. I wanted to test whether an
  explicit code index could give them useful context alongside that search.”
- Add a short, accurate ownership statement in the narration: distinguish the
  original personally built seminar project from the later AI-assisted extension.

## Slide 2 — The agent queries one Neo4j code index

Current segment: approximately 1:28–3:25.

Keep both rows and the shared Neo4j database. Make component roles explicit inside
the build box, if space permits:

- Tree-sitter: extract syntax and call sites.
- Jedi: resolve referenced definitions.
- Embeddings: support semantic retrieval.

Do not enlarge this into a tool catalogue. Explain indexing first, then show the
same Blueprint question passing through the lower row:

1. The host selects the intended Blueprint definition using `find_definitions`.
2. It requests inheritance with `get_inheritance`.
3. Codeaware returns the indexed definitions and relationships with locations.
4. The host uses that context to formulate its answer.

The known-name example does not require semantic search. Explain separately in one
sentence that semantic search helps when the relevant symbol or location is unknown.

Technical corrections to the narration:

- Around **1:49–2:02**, remove the statement that these components are based on
  LSP. This implementation calls Tree-sitter and Jedi directly; it does not
  implement an LSP client or server. The embedding model is also separate from LSP.
- Around **1:09–1:27**, replace “I built an agent” with “I built a code-context
  tool server that a coding agent can query.” The host owns tool selection and
  the agent loop.
- Around **2:21–2:33**, distinguish retrieval chunks from graph definitions.
  The current text splitter uses up to 900 characters and zero overlap with
  Python-aware separators; it does not guarantee one complete function per chunk.
  Full definition content is stored separately for symbol excerpts.
- Say **Model Context Protocol** when expanding MCP.
- State the existing operating limit aloud: this is a partial static graph and
  changed source requires a manual rebuild. Keep implementation details in backup.

## Slide 3 — Evaluation and lessons learned

Current segment: approximately 3:25–5:03.

Keep the chart and quality counts. Improve the interpretation:

- Retain “6.7% fewer reported tokens,” explicitly scoped to this run.
- Move **“Includes cached input; uncached input + output: +2.5%”** from the small
  footer to a readable annotation immediately below the headline or chart.
- Keep one-run/unblinded-review limitations in the footer and mention them briefly.
- Replace the first conclusion bullet, which currently repeats the headline, with
  **“Continue with broader validation.”**
- Make the next step **“More repositories, relationship-heavy questions and repeated runs.”**
  Larger repositories alone do not establish that graph retrieval is responsible
  for any benefit. This run used only one inheritance query.

Suggested narration:

> “I compared the same host on 18 Flask questions, once with native tools and once
> with Codeaware available. Total reported tokens were 6.7% lower, including cached
> input. Uncached input plus output was 2.5% higher, so this does not establish cost
> savings. My retrospective review found 17 ties and one baseline completeness
> advantage. This is enough to justify a broader test, rather than a deployment
> claim. Next I would test more repositories, more relationship-heavy questions and
> repeated runs, including quality, latency and actual cost.”

Do not call the quality identical, the hypothesis confirmed, or scaling easier as
an established finding. A possible advantage on larger repositories remains a
hypothesis; index construction, freshness and unresolved relationships also matter.

## Revised speaking outline

| Section | Target | Purpose |
|---|---:|---|
| Cover | 0:15 | Name the project, origin and contribution |
| Problem and Flask example | 1:30 | Establish a concrete need and intended user value |
| Architecture and query walkthrough | 2:15 | Explain the implementation and one operating limit |
| Evaluation and decision | 1:45 | Present the comparison, bounds and next experiment |
| Handover | 0:15 | Invite the code discussion |
| Total | 6:00 | Three content slides plus holding cover |

Stay on the results slide when handing over. Open appendix slides in response to
discussion questions; do not browse them automatically at the end. Keep notes to
short prompts rather than a written script. Finish content changes before animation.

## Evidence checked

- `src/index_repository.py`: chunk settings and direct Jedi configuration.
- `src/indexing/file_processor.py`: Python-aware recursive text splitting.
- `src/indexing/build_graph.py`: direct Tree-sitter and Jedi use.
- `benchmarks/results/2026-11-07-neo4j-low-full-guided/README.md`: token accounting,
  quality review summary, tool use and limitations.

The Blueprint walkthrough is an explanation of tool use, not an additional
benchmark result or evidence of graph-specific savings.
