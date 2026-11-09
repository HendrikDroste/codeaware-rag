# Codeaware: code-understanding checklist for the interview

Updated 8 November 2026. Based on the current implementation, including the
`find_definitions` rename.

Tick an item when you can explain it in your own words, locate the implementation
and trace an example. Library internals do not need memorizing; your own control
flow, data model, assumptions and failure handling do.

## A short route through graph construction

Open `src/indexing/build_graph.py` and follow this order:

1. `build_graph()`: setup, definitions, relationships and returned counts.
2. `_store_definitions()`: store classes/functions and remember their IDs by
   declaration location. `function_locations` and `class_locations` are separate.
3. `_store_relationships()`: dispatch imports, inheritance and calls.
4. `process_function_calls()`: identify the enclosing caller, let Jedi resolve
   the target, match its declaration location, then write the edge.
5. `_jedi_position()`: explain why Unicode requires byte-to-character conversion.

Keep the Cypher constants available for questions about the database. Counter
initialization in `_new_counts()` is bookkeeping, not a separate algorithm.

## 1. Explain the whole system first — essential

- [ ] Explain the lifecycle: Python files → chunks and resolved definitions →
  embeddings and relationships in Neo4j → read-only context tools → host answer.
- [ ] Distinguish the two retrieval capabilities: semantic search finds code by
  meaning; graph queries follow indexed relationships or look up exact names.
- [ ] Explain the responsibility boundary: Codeaware supplies context. The host
  chooses tools, reasons, answers and may use its own editing/testing tools.
  Codeaware has no independent agent loop or answer-generating LLM.
- [ ] Explain why this prototype supports Python only: keeping extraction and
  resolution manageable with Tree-sitter and Jedi. There is no LSP integration.
- [ ] Distinguish your original seminar implementation from the later AI-assisted
  extension. Identify the decisions and changes you can personally defend.

Read: [README](../README.md), [server](../src/mcp_server.py).

## 2. Trace one complete index build — essential

- [ ] Walk through `index_repository()`: resolve paths, discover files, hash the
  corpus, read embedding settings, verify Neo4j connectivity, publish `building`,
  load the model, split files, build the graph, store vectors, recheck hashes,
  then publish `complete`.
- [ ] Explain the manifest fields: repository root, build ID, schema version,
  embedding settings, chunk settings, file hashes, counts and vector dimensions.
- [ ] Explain why both graph and vector storage use the same discovered file list.
- [ ] Explain failure behavior: after `building` is published, a failed build
  stays incomplete and retrieval rejects it. The previous index is not retained
  as an automatically usable fallback.
- [ ] Distinguish atomic manifest replacement from the multi-step Neo4j rebuild.
  The entire build is not a single atomic transaction.

Read: [`index_repository()`](../src/index_repository.py),
[`write_manifest()`](../src/index_state.py).

## 3. Understand file discovery and chunk locations — essential

- [ ] Explain inclusion of source, tests, `__init__.py` and `cli.py`, and exclusion
  of configured cache/environment directories, detected virtual environments
  and symlinks. The generic loader's defaults differ from the indexing path.
- [ ] Explain the Python-aware recursive character splitter: 900 characters,
  zero overlap. This does not guarantee one complete function/class per chunk.
- [ ] Show how the start character index is checked against the original text and
  converted into inclusive, one-based line ranges. Repeated text must retain its
  actual location.
- [ ] Distinguish chunk content from full definition content stored in the graph.
  Definition excerpts can be larger or smaller than retrieval chunks.

Read: [file processing](../src/indexing/file_processor.py),
[`source_files()`](../src/index_state.py).

## 4. Explain the Neo4j data model — essential

- [ ] Sketch these nodes: index metadata, files, functions/methods, classes and
  chunks. Explain the `Codeaware` label and repository/build metadata.
- [ ] Draw edge directions: definition/chunk → `DEFINED_IN` → file;
  caller → `CALLS` → callee; child → `INHERITS_FROM` → parent;
  importing file → `IMPORTS` → resolved target file.
- [ ] Explain the definition ID: relative path + definition start line + qualified
  name. Its stored/API key is still `symbol_id`; semantic results use `symbol_ids`.
  These names did not change with `find_definitions`.
- [ ] Explain how qualified names distinguish nested functions and methods in
  different classes. IDs are tied to source locations, not stable across edits.
- [ ] Show why rebuilding deletes only this repository's `Codeaware` nodes and
  replaces its vector index, preserving unrelated database data.

Read: [graph construction](../src/indexing/build_graph.py),
[vector storage](../src/retrievers/vector_search.py).

## 5. Trace graph extraction and resolution — essential

- [ ] Explain the two passes: create all definitions first, then resolve edges.
  Otherwise targets in later files may not exist when relationships are created.
- [ ] Distinguish Tree-sitter's syntax extraction from Jedi's target resolution.
  Detecting a call expression does not by itself identify the called definition.
- [ ] Follow a call through enclosing-function selection, `script.infer()`,
  target-location matching and parameterized `CALLS` creation. Method calls reuse
  this logic with a different `call_type`.
- [ ] Explain the two location representations: Tree-sitter uses zero-based rows
  and UTF-8 byte columns; Jedi uses one-based lines and character columns.
  Point to the byte-to-character conversion and the Unicode fixture.
- [ ] Explain why targets are matched by definition file/line/column, rather than
  their name, and why relationships retain call-site locations.
- [ ] Explain syntax validation: `tree.root_node.has_error` aborts indexing.
  This checks parsing, not types, runtime behavior or program correctness.

Read: `build_graph()`, `_read_source()`, `_store_definitions()`,
`_store_relationships()`, `_jedi_position()`, `_enclosing_function()`,
`process_function_calls()`, `process_method_calls()` and
`get_qualified_name()` in [graph construction](../src/indexing/build_graph.py).

## 6. Explain what the graph cannot establish — essential

- [ ] Distinguish unresolved, ambiguous, external and out-of-scope call counts.
  For example, calls outside a supported enclosing function are not stored as
  ordinary function-to-function edges.
- [ ] Explain why missing edges do not prove a runtime relationship is absent.
  Give an example such as `handlers[user_input]()` or runtime monkey-patching.
- [ ] Explain inheritance limits: explicit resolvable bases, direct parents and
  subclasses, written base order. No computed MRO or inherited-method lookup.
- [ ] Explain imports: Jedi `goto(..., follow_imports=True)` follows bindings and
  re-exports to indexed target files; aliases and sites are retained. This is not
  runtime import order or a complete trace through intermediate re-export modules.
- [ ] Explain omission of external/unresolved/ambiguous imports and wildcard
  imports. Conditional imports do not establish which branch executes.

Read: `process_imports()` and `process_inheritance()` in
[graph construction](../src/indexing/build_graph.py),
[graph queries](../src/retrievers/query_graph.py).

## 7. Trace semantic retrieval — essential

- [ ] Identify the configured embedding model and how `get_embedding_model()`
  creates it. Explain `embed_documents()` during indexing versus `embed_query()`
  during retrieval, and why the settings must match.
- [ ] Explain batching, dimension checks and rejection of nonfinite vectors.
  The configured model currently allows remote model code when loading through
  Hugging Face; understand the `trust_remote_code=True` setting.
- [ ] Explain repository-derived vector index/label names, waiting for the index
  to become `ONLINE`, and checking stored dimensions and build identity.
- [ ] Follow `db.index.vector.queryNodes()` and the returned squared Euclidean
  distance: lower is better. It is not a probability or confidence score.
  Candidate retrieval is approximate, even though distances are recomputed.
- [ ] Explain how overlapping definitions are attached to each snippet, and why
  a chunk can overlap zero, one or several definitions.

Read: [model loader](../src/indexing/model.py),
[configuration](../config/app_config.yaml),
[vector retrieval](../src/retrievers/vector_search.py),
`find_snippet_symbols()` in [graph queries](../src/retrievers/query_graph.py).

## 8. Trace exact lookup and graph queries — essential

- [ ] Explain `find_definitions`: exact simple/qualified names, optional exact
  path, distinct IDs for ambiguous matches, no assignment/attribute lookup.
- [ ] Explain `include_code`, `max_lines` and `start_line`. The definition range
  covers the whole definition; the excerpt has its own range and truncation flag.
- [ ] Trace `get_callers` versus `get_callees` and explain their edge direction.
  `get_callers` delegates through `query_references()` to the same call-query code.
- [ ] Explain one-hop inheritance/import traversal, unknown-ID errors, empty
  neighbor lists, and the hint after an empty exact-name lookup.
- [ ] Explain fixed query structure and parameter binding. Generated index names
  and fixed internal labels are narrowly controlled exceptions to parameter use;
  the host cannot supply arbitrary Cypher.

Read: [graph queries](../src/retrievers/query_graph.py),
[tool wrappers](../src/mcp_server.py).

## 9. Understand MCP dispatch and response bounds — essential

- [ ] Explain `create_server()`, nested tool functions, `@server.tool`, type/schema
  descriptions and stdio transport. stdout carries protocol messages; logs use
  stderr. MCP connects the tools to the host; it does not implement reasoning.
- [ ] Explain the seven tools and when the host should choose each. The guidance
  influences tool choice but does not enforce a deterministic workflow.
- [ ] Explain limits: semantic search at most 20, other result lists at most 100.
  Fetching `limit + 1` lets the wrapper indicate result-list truncation.
- [ ] Explain lazy embedding loading and the model lock. Tool discovery and graph
  lookup do not need to load the embedding model.
- [ ] Distinguish read-only tool exposure from indexing, which mutates Neo4j.
  No shell execution, rebuilding or arbitrary Cypher is exposed through these tools.

Read: [MCP server](../src/mcp_server.py),
`CONTEXT_GUIDANCE` in [shared state](../src/index_state.py).

## 10. Explain freshness, errors and resource management — essential

- [ ] Trace `snapshot()` and `read_manifest()`: validate repository, completion,
  schema, backend, model settings and current file hashes. Recheck after retrieval
  and reject a changed build until the client reconnects.
- [ ] Explain that additions/deletions/edits to included Python files invalidate
  the snapshot. Non-Python files are outside this fingerprint. There is no watcher
  or automatic incremental update.
- [ ] Explain the cost/limit: retrieval hashes the Python corpus; this may become
  expensive on larger repositories. Before/after checks are not filesystem locks
  or a fully atomic read transaction across source, manifest and database.
- [ ] Explain that `index_status` reports manifest information and freshness;
  it does not prove Neo4j connectivity or vector-index readiness.
- [ ] Explain environment-based credentials, required password, shared driver,
  session cleanup, `atexit` cleanup and restoration of Jedi's cache setting.

Read: [shared state](../src/index_state.py), [server](../src/mcp_server.py),
[Neo4j connection](../src/neo4j_connection.py),
[indexing](../src/index_repository.py).

## 11. Understand the benchmark and its claim boundaries — essential

- [ ] Explain the paired comparison: same host/model/prompt/native tools, with
  Codeaware added in one condition; alternating pair order and one repetition.
- [ ] Explain pinned dataset validation: question-file hash, corpus hashes,
  repository revision and valid evidence line ranges.
- [ ] Trace usage extraction: one `turn.completed` event, total = input + output,
  cached input is already included in input, uncached input = input − cached.
  Missing/ambiguous usage is not silently treated as zero.
- [ ] Explain tool-call deduplication by item ID, tool-error counts, successful
  excerpt use and excerpt truncation. A truncated excerpt is not a tool error.
- [ ] Defend the recorded findings: 18 questions; 6.71% fewer total reported
  tokens; uncached input + output 2.47% higher; 17 quality ties and one baseline
  completeness advantage. Explain caching and the retrospective unblinded review.
- [ ] Explain what remains unproven: monetary savings, developer productivity,
  generalization and graph-specific benefit. Only one inheritance query occurred.
  Index construction is not included in the agent token comparison.
- [ ] Explain why archived tool names/results remain unchanged after a rename.
  A new name and description version do not constitute a new measured run.

Read: [runner](../benchmarks/run_agent.py),
[dataset validation](../benchmarks/dataset.py),
[recorded run](../benchmarks/results/2026-11-07-neo4j-low-full-guided/README.md),
[quality review](../benchmarks/results/2026-11-07-neo4j-low-full-guided/quality-review.md).

## 12. Know the tests and be ready to change the code — practice

- [ ] Find coverage for duplicate method names, nested functions, aliased imports,
  Unicode positions, syntax errors, stale/incomplete indexes, bounded excerpts
  and Neo4j repository isolation.
- [ ] Explain deterministic test embeddings and mocked Neo4j sessions versus the
  opt-in real Neo4j integration test. Mocked tests do not prove backend behavior.
- [ ] Trace the stdio test: initialize a client, discover `find_definitions`, inspect
  schemas, call tools and confirm the embedding model stays unloaded for discovery.
- [ ] Explain benchmark tests preventing double-counted tokens and mismatched
  settings between conditions.
- [ ] Practice a small change: adjust an excerpt default, extend a parameter
  description or add a validation case. Identify which layer and existing test
  should change, without altering the graph schema unnecessarily.

Read: [MCP tests](../tests/test_mcp.py),
[fixture](../tests/fixtures/repository/worker.py),
[benchmark tests](../tests/test_agent_benchmark.py),
[dataset tests](../tests/test_dataset.py).

## Final rehearsal: three traces without notes

1. **Index build:** start at `index_repository()` and show exactly when retrieval
   becomes allowed, including what happens if vector storage fails.
2. **Blueprint question:** `find_definitions` → select public Blueprint's ID →
   `get_inheritance` → query its parent's ID → host explains the chain. Explain
   how Jedi resolved the imported alias during indexing, not during the query.
3. **Unknown implementation:** `search_code` → query embedding → Neo4j candidates
   → snippets/line ranges/definition IDs → optional targeted definition excerpt.
   Explain when the host already has enough evidence to answer.

For each trace, answer: **What enters? What is stored or returned? Why this design?
What can fail? What would you change for a larger production system?**
