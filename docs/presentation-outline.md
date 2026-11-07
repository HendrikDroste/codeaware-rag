# Codeaware interview notes

Updated 7 November 2026. Three project slides, about 7–8 minutes total.
Prepare a spoken introduction separately. The supplied brief gives the
introduction and project presentation separate 5–10 minute slots, followed
by 35–45 minutes of code discussion. Introduction slides are optional.

## Project framing

Focus on code relationships as context tools for a coding agent. Semantic search
helps discover an implementation. The graph helps follow explicit relationships.
Both capabilities now use Neo4j.

The original seminar work was personally built. The later tool-server extension
used AI assistance. Explain your own decisions and be ready to modify the code.
The host owns the agent loop. Codeaware provides read-only context and does not
run its own chat model, edit files or execute tests.

The original retrieval experiments are optional background. One sentence is
enough: “The project started with semantic code retrieval; I later exposed code
relationships as tools for a coding agent.” Keep their results on the archive
branch for follow-up questions. They do not establish an agent-level benefit.

## Slide 1: Problem, hypothesis and contribution

**Time: about 2 minutes**
**Visual:** a concrete maintenance question beside a small Blueprint inheritance
chain spanning several Flask files.

Explain the developer's problem: finding a relevant fragment does not always
provide the relationships needed to understand it. Use the concrete Blueprint,
its Sansio base and Scaffold as an example. Distinguish the two classes named
Blueprint and the SansioBlueprint import alias.

State the hypothesis: semantic discovery plus explicit relationships could help
a coding agent investigate unfamiliar code more efficiently. The host can also
start with an exact symbol name; semantic search is not mandatory before every
graph call.

Explain what you built and which parts came from the seminar versus the later
extension. Maintenance and onboarding are potential applications. Customer
demand, developer time savings and business ROI have not been validated.

## Slide 2: Architecture and agent workflow

**Time: about 3 minutes**
**Visual:** one diagram showing the host agent, Codeaware tools and Neo4j, with
a short explanation of offline indexing.

Speaking route:

1. Discover Python source and tests. Split code into chunks of up to 900
   characters with zero overlap and generate embeddings using the configured
   sentence-transformers model.
2. Tree-sitter extracts definitions and call sites. Jedi resolves targets,
   explicit bases and imported names. Neo4j stores symbols, relationships,
   chunks and embeddings, with a vector index scoped to each repository.
3. The host chooses a semantic search, symbol lookup or relationship query,
   receives source evidence and decides its next action. Explain one request
   through this path instead of listing every tool.
4. Explain fixed Cypher, bounded responses and source/build checks. Freshness
   checks reject changes until a manual rebuild and reconnect. The graph is
   partial static analysis; missing edges do not establish runtime absence.

The storage consolidation reduces components to operate, but rebuilding remains
multi-step. A manifest is complete only after graph extraction, vector indexing
and the final source check succeed. One database does not make the rebuild atomic.

Keep an actual tool trace available for the joint discussion. The presentation
need not depend on a live application switch. Recorded Q&A runs do not establish
successful autonomous code changes.

## Slide 3: Evaluation, lessons and next decision

**Time: about 2 minutes 30 seconds**
**Visual:** a compact token comparison with the quality-review findings and the
resulting engineering decision.

| Recorded full benchmark | Native tools | Native tools plus Codeaware |
|---|---:|---:|
| Total reported tokens | 979,194 | 1,154,004 |
| Relative token change | Reference | +17.85% |

The same host/model answered 18 pinned Flask questions with low reasoning,
once per question and condition. The retrospective, unblinded implementer
review found 15 ties and three baseline completeness advantages. Six pairs
saved tokens with Codeaware and tied on the scoped rubric. No overall
quality advantage or reliable productivity improvement was established.

These results predate the Neo4j-only migration and final tool simplification.
The comparison evaluates the combined added tools, without isolating the graph's
contribution. Cached input is included, indexing is excluded, and token totals
are not monetary costs. Preserve these qualifications alongside the chart.

Explain the decisions the evidence informed:

- Removed three on-demand Jedi tools after a small pilot used none of them.
  Jedi remains inside graph construction.
- Consolidated vector and graph storage in Neo4j. The embedding model and chunk
  boundaries remain unchanged. Functional verification does not establish
  unchanged approximate rankings or an agent-performance improvement.
- Next, validate a focused maintenance workflow against the same native-tools
  baseline. Verify outcomes with tests, review quality independently and
  measure task time alongside usage and indexing effort.

Business investment requires verified benefits exceeding operation and
maintenance costs. Time saved is not automatically cash saved.

Suggested closing:

> “I built a working integration, but the benchmark challenged my efficiency
> hypothesis. I simplified the system and would validate one maintenance
> workflow before expanding it. We can now look at the graph construction,
> tool contracts or evaluation.”

Evidence: [full run](../benchmarks/results/2026-11-05-low-full-guided/README.md),
[quality review](../benchmarks/results/2026-11-05-low-full-guided/quality-review.md)
and [archive index](../benchmarks/results/README.md).

## Joint code discussion

| File | Discussion |
|---|---|
| `src/index_repository.py` | Shared file list, build identity and incomplete builds |
| `src/embeddings/build_graph.py` | Extraction, target resolution, uncertain calls and imports |
| `src/retrievers/vector_search.py` | Vector storage, distance and repository isolation |
| `src/retrievers/query_graph.py` | Parameterized queries and one-hop relationships |
| `src/mcp_server.py` | Tool contracts, source excerpts and limits |
| `src/index_state.py` | Freshness and configuration compatibility |
| `src/neo4j_connection.py` | Shared driver and short-lived sessions |
| `benchmarks/run_agent.py` | Fair paired settings and usage accounting |
| `benchmarks/dataset.py` | Immutable question/source checks before and after runs |

Be ready to explain why combine Jedi and Tree-sitter, what dynamic Python
behavior the graph misses, what happens after an edit, and what evidence would
justify continuing or stopping. A focused code change with a verified outcome
still needs its own rehearsal; the Q&A benchmark does not demonstrate it.

The Neo4j migration was functionally checked against a real database, with
stdio tool calls and a real-model Flask rebuild of 83 files and 933 chunks.
Those checks support integration correctness, not an efficiency or quality claim.

## Sources

- Supplied BCG Platinion interview brief, pages 2–3.
- Supplied university presentation, particularly slides 5, 16 and 18.
- [Current architecture and setup](../README.md).
- [Benchmark method](../benchmarks/README.md) and retained evidence linked above.
