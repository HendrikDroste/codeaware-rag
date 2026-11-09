# Codeaware interview notes

Updated 8 November 2026. Five project slides, about 9–9½ minutes total.
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

**Time: about 1 minute 30 seconds**
**Visual:** a concrete maintenance question beside a small Blueprint inheritance
chain spanning several Flask files.

Explain the developer's problem: finding a relevant fragment does not always
provide the relationships needed to understand it. Use the concrete Blueprint,
its Sansio base and Scaffold as an example. Distinguish the two classes named
Blueprint and the SansioBlueprint import alias.

State the hypothesis: semantic discovery plus explicit relationships could help
a coding agent investigate unfamiliar code more efficiently. The host can also
start with an exact definition name; semantic search is not mandatory before every
graph call.

Explain what you built and which parts came from the seminar versus the later
extension. Maintenance and onboarding are potential applications. Customer
demand, developer time savings and business ROI have not been validated.

## Slide 2: Architecture and agent workflow

**Time: about 1 minute 45 seconds**
**Visual:** one diagram showing the host agent, Codeaware tools and Neo4j, with
a short explanation of offline indexing.

Speaking route:

1. Discover Python source and tests. Split code into chunks of up to 900
   characters with zero overlap and generate embeddings using the configured
   sentence-transformers model.
2. Tree-sitter extracts definitions and call sites. Jedi resolves targets,
   explicit bases and imported names. Neo4j stores code definitions, relationships,
   chunks and embeddings, with a vector index scoped to each repository.
3. The host chooses a semantic search, definition lookup or relationship query,
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

## Slide 3: Following a question through Codeaware

**Time: about 2 minutes**
**Visual:** three steps, with the answer inside the final step.

1. `find_definitions(name="Blueprint")`: choose the public class from the
   matching definitions, using its path, and keep its returned definition ID.
2. `get_inheritance` twice: query the selected class, then its parent. Each
   call returns direct parents and subclasses; follow the parent results.
3. The host explains the chain: public Blueprint inherits from the Sansio
   Blueprint, which inherits from Scaffold. The Sansio name is an import alias.

Keep paths and line numbers off the slide. They remain available in tool results.
The ID is called `symbol_id` in the unchanged API/database schema.

## Slide 4: Engineering decisions and trade-offs

**Time: about 1 minute 45 seconds**
**Visual:** three decisions with a reason and consequence for each.

- One Neo4j database: vectors and relationships together; an external service
  still needs operating.
- Python-focused analysis: limit language support to keep the prototype simple;
  other languages need additional extraction and resolution logic.
- Manual snapshots: simple indexing with an explicit source version; source edits
  require a rebuild and reconnect. Source hashes reject stale context.

This is the former A2 slide, moved into the main presentation without duplication.

## Slide 5: Evaluation, lessons and next decision

**Time: about 2 minutes**
**Visual:** a compact token comparison with the quality-review findings and the
resulting engineering decision.

| Current Neo4j-only benchmark, 7 November 2026 | Native tools | Native tools plus Codeaware |
|---|---:|---:|
| Total reported tokens | 1,027,206 | 958,262 |
| Relative token change | Reference | -6.71% |
| Uncached input plus output | 162,566 | 166,582 |

The same host/model answered 18 pinned Flask questions with low reasoning,
once per question and condition. All 36 runs completed with reported usage
and no MCP errors. Codeaware used fewer tokens on 13 of 18 questions. The
retrospective, unblinded implementing-assistant review found 17 ties and one
baseline completeness advantage on the unchanged historical rubric. The
Q15 difference is a missing explanation of response-tuple status and headers;
a narrower literal-question rubric would score it as a tie.

Cached input accounts for the aggregate reduction. Uncached input plus output
rose 2.47%, and the median token count rose 4.16%. Token totals are not monetary
costs. Indexing is excluded. The comparison evaluates the combined added tools,
with eight semantic searches, 19 definition lookups and one inheritance query.
It does not isolate the graph's contribution or demonstrate productivity.

Use the neutral title **Evaluation and lessons learned**. Show the result
clearly, then connect it to your engineering decisions. Avoid describing the
project as failed or claiming proven savings from a single trial.

The earlier benchmark remains optional discussion material: +17.85% tokens,
15 ties and three baseline completeness advantages. Host version, prompts,
tool metadata and storage changed between the two runs. Their difference
cannot isolate which change caused an improvement.

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

> “I built a working code-context integration and tested it against native
> tools. The current run used fewer reported tokens, with no quality advantage
> and no proven cost saving. I would next validate one maintenance workflow
> before expanding it.”

Evidence: [current full run](../benchmarks/results/2026-11-07-neo4j-low-full-guided/README.md),
[quality review](../benchmarks/results/2026-11-07-neo4j-low-full-guided/quality-review.md)
and [archive index](../benchmarks/results/README.md).

## Joint code discussion

| File | Discussion |
|---|---|
| `src/index_repository.py` | Shared file list, build identity and incomplete builds |
| `src/indexing/build_graph.py` | Extraction, target resolution, uncertain calls and imports |
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
Those integration checks support functional correctness. The separate paired
Q&A run above measures agent usage and reviewed answer quality within its scope.

## Sources

- Supplied BCG Platinion interview brief, pages 2–3.
- Supplied university presentation, particularly slides 5, 16 and 18.
- [Current architecture and setup](../README.md).
- [Benchmark method](../benchmarks/README.md) and retained evidence linked above.
