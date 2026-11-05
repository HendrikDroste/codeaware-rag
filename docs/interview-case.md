# Interview case: repository context as an evaluated engineering experiment

## Positioning

The hypothesis was that semantic retrieval plus structured Python relationships
would help a coding agent answer repository questions more efficiently. The
system works, but the current experiments do not establish that hypothesis or
a business-value improvement. Present the evaluation and resulting decisions
as the outcome; avoid claiming a successful productivity product.

The original seminar work was personally built. The subsequent tool-server
extension used AI assistance. Be prepared to explain and change the relevant
code, including analysis, query boundaries and measurement accounting.

The host supplies the agent loop. Codeaware is a read-only context provider;
the host chooses tools and performs any editing or test execution. The recorded
benchmarks evaluate repository Q&A, not successful autonomous code changes.

## Short presentation: 5–10 minutes

1. Problem and hypothesis: locating relevant implementation in unfamiliar code.
   Developer onboarding/maintenance is a possible application, not a validated
   customer requirement or realized saving.
2. Architecture: manual indexing, Chroma semantic retrieval, Neo4j relationships,
   fixed read-only tools, and the host agent's decisions.
3. Evidence: pinned Flask source, paired native/Codeaware runs, reported tokens,
   tool adoption, and scoped answer-quality review.
4. Decision: reduce speculative functionality and retain the understandable core.
   More tools did not automatically improve the evaluated workflow.

## Code discussion route

| File | Discussion |
|---|---|
| `src/index_repository.py` | Shared source list, build identity and incomplete-build handling |
| `src/embeddings/build_graph.py` | Tree-sitter extraction, Jedi resolution, ambiguous targets and scoped replacement |
| `src/retrievers/query_graph.py` | Explicit parameterized queries and one-hop relationships |
| `src/mcp_server.py` | Tool contracts, limits, excerpts and error handling |
| `src/index_state.py` | Freshness, configuration compatibility and snapshot limitations |
| `src/benchmark_agent.py` | Fair paired prompts, raw traces, usage and cache accounting |

Keep the distinction between semantic retrieval metrics, host token usage,
answer completeness and business outcomes explicit. A lower token count is
not a correct-answer score or a currency-cost estimate.

## Findings worth defending

- The full low-reasoning run used 17.85% more total tokens with Codeaware.
  Six of 18 pairs saved tokens and tied on the scoped quality rubric.
- The retrospective, unblinded review found 15 ties and three native-baseline
  completeness advantages. It is not an independent or calibrated quality grade.
- The final ten-tool pilot used none of the three new on-demand Jedi tools.
  Their tests demonstrate functionality, not agent-level benefit.
- Native searches can bundle information that requires several context-tool
  calls. Large class prefixes may hide useful implementation behind docstrings.
- Missing static edges do not prove absence of runtime calls. Dynamic dispatch,
  callbacks, aliases and version differences require explicit uncertainty.

Source artifacts are linked from the main README. Keep single-observation
pilots separate from repeated runs and full-dataset aggregates.

## Why simplify

Remove only the on-demand outlines, definition and reference tools. Keep Jedi
inside the existing graph builder: it resolves targets that Tree-sitter alone
does not identify semantically. Keep the measured experiment history intact.
Keep small excerpt controls and hints that make the core contracts explicit.

The remaining seven tools have a direct explanation: semantic discovery,
indexed symbol lookup, callers, callees, inheritance, imports and index status.
The simplification itself has no new benchmark result.

The proposed live-workspace/versioned-library catalog is a future architecture
option. Its unresolved questions include actual user demand, dependency version
selection, local type information and cross-boundary references. A full rewrite
before establishing a use case would increase uncertainty and review scope.

## Preparation still needed

- Rehearse one bounded agent engineering task with an observable result and a
  focused test. The existing Q&A benchmarks do not establish that demonstration.
- Explain what happens after an edit: current retrieval rejects a stale index,
  then requires a manual rebuild and reconnect. Do not imply automatic updates.
- Keep a recorded or scripted fallback for the demo, and know the core code
  well enough to discuss failure cases without depending on an agent.
- Assess whether another eligible project has stronger evidence of a real user
  need. This document does not resolve that project-selection question.
