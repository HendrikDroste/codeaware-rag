# Joint discussion: likely questions and answer cues

Preparation hypotheses, not a prediction of BCG's questions. Use the six backup
slides only when useful, then open the relevant code. The appendix is a separate
six-slide appendix plus a divider so the three-slide main presentation stays unchanged.

## Highest priority

| Likely question | Short answer cues | Backup / code |
|---|---|---|
| What did you build yourself? | Original seminar implementation independently built. Later tool-server extension used AI assistance. Explain concrete decisions and code you understand. | Main slide 1; repository history |
| Is Codeaware an agent? | Host owns the loop and edits/tests. Codeaware supplies seven read-only context tools. No independent chat model. | A2; `src/mcp_server.py` |
| How does a call become a graph edge? | Two passes. Extract definitions first, then resolve calls. Jedi infers a target, matched by definition file/line/column. | A1; `src/indexing/build_graph.py` |
| How do you distinguish two methods named run? | Repository-scoped definition ID includes relative path, definition line and qualified name. Name alone is insufficient. | A1; `tests/fixtures/repository/helpers.py` |
| Why both Tree-sitter and Jedi? | Syntax locations and enclosing scopes versus semantic name resolution. Explicit division of responsibilities. | A1/A2 |
| Why Neo4j? | One store for relationships and vector search, parameterized queries, repository isolation. Trade-off: running a service and building snapshots. | A2; `src/retrievers/` |
| Why not use native search or an LSP? | Existing tools are a strong baseline. This prototype tests semantic discovery plus stored relationships. No demonstrated universal advantage over language intelligence. | A2/A3 |
| What happens after a file changes? | Hash checks reject stale retrieval. Manual rebuild and reconnect. No automatic updating yet. | A2; `src/index_state.py` |
| Did your hypothesis succeed? | Integration worked. One run had fewer reported tokens and no quality advantage. Graph-specific benefit was not isolated. | A3 |
| Does 6.7% fewer tokens mean lower cost? | Cached input explains the reduction. Uncached input plus output rose 2.47%. No monetary saving established. | A3 |
| Why is the median higher despite the lower total? | Different summaries of heterogeneous questions. Total fell 6.71%; condition median rose 4.16%; 13/18 pairs used fewer tokens. | A3; run report |
| Where is the business case? | Potential dependency-investigation pilot. Measure correct outcomes, developer time, indexing and maintenance effort. Customer demand and ROI remain unvalidated. | A6 |

## Technical follow-ups

- **Which relationships are covered?** `DEFINED_IN`, `CALLS`, `INHERITS_FROM`
  and resolved local file-to-file `IMPORTS`. This is a partial static graph.
- **What is omitted?** Unresolved, ambiguous and external targets are counted
  and omitted. Missing edges do not prove runtime absence. Inheritance base
  order is not a computed Python MRO.
- **How do aliases and Unicode work?** Resolve bindings with Jedi. Convert
  Tree-sitter byte columns to Python character columns. Show `worker.entry`.
- **Why 900-character chunks and zero overlap?** Reused seminar baseline,
  not a newly optimized choice. Benchmark alternatives before claiming optimality.
- **How does semantic ranking work?** Same embedding configuration for build
  and query, repository-scoped Neo4j ANN index, returned squared L2 distances.
- **What happens on a failed build?** Manifest remains incomplete. Publish
  completion only after graph, vectors, vector index readiness and final hashes.
  The multi-step rebuild is not atomic and must not overlap queries/rebuilds.
- **How are tool errors handled?** Explicit errors for stale/missing/incomplete
  indexes and incompatible settings. Unknown definition differs from no neighbors.
- **How is accidental mutation limited?** Fixed parameterized Cypher, bounded
  responses, relative path checks, no arbitrary Cypher/shell/rebuild tool.
  This does not make the host or a future deployment fully secure.
- **What would you change for scale?** Measure first: model/index cost,
  database size, latency and hashing all source files around each tool call.
  Consider incremental updates only after identifying the dominant costs.

## Evaluation follow-ups

- **Was the comparison controlled?** Same pinned Flask revision, host/model,
  low reasoning and prompt within each pair. Order alternated. One observation
  per question/condition. All 36 runs completed, with no MCP errors.
- **How was quality graded?** Retrospective, unblinded implementing-assistant
  review against the unchanged checklist. Source reviewed; examples not executed.
- **What was the quality difference?** 17 ties, one baseline completeness
  advantage. Q15 omitted response-tuple status/header details. A narrower literal
  question rubric would score that pair as a tie.
- **Did agents use the graph?** Eight semantic searches, 19 definition lookups and
  one inheritance query. No graph-specific benefit can be inferred from that.
- **What experiment would isolate the graph?** Hold tasks and host settings
  fixed. Compare native tools, semantic-only tools and the complete toolset on
  relationship-heavy questions. Repeat and review outcomes independently.
- **Why did the historical run differ?** Host, prompt, tool metadata and storage
  changed. Within-run comparisons are meaningful; cross-run differences do not
  identify which change caused the improvement.

## Potential live code changes to rehearse

1. Add a fixture with two identically named methods and explain the expected IDs.
2. Follow one aliased call from Tree-sitter location through Jedi to Cypher.
3. Change a source file and explain the stale-index error and rebuild lifecycle.
4. Change a result bound and explain validation and truncation.

These are rehearsal ideas, not claims that a live coding exercise has been
completed. Keep the demo independent of an unreliable live model call.

## Evidence

- [Current architecture and limits](../README.md)
- [Current run and exact usage](../benchmarks/results/2026-11-07-neo4j-low-full-guided/README.md)
- [Quality method and Q15 sensitivity](../benchmarks/results/2026-11-07-neo4j-low-full-guided/quality-review.md)
- [Graph construction](../src/indexing/build_graph.py)
- [Build lifecycle](../src/index_repository.py)
- [Freshness checks](../src/index_state.py)

## Expanded benchmark appendix

A3 explains the paired method, A4 shows representative question coverage,
and A5 explains Q15 grading. A6 contains the next evaluation and potential
business pilot. The expanded file includes an Appendix divider. The merge is complete in `presentation/Codeaware_Interview_Revised.pptx`: four main slides, the divider and six backup slides.
