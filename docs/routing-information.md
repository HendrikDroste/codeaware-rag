# Routing information in the existing index

The existing Flask index contains the information needed to explain route
registration and its setup restriction. However, it is not all returned or
connected by the current method-level tools. This was checked directly in
Neo4j and the persisted Chroma collection, with no index rebuild or new model
calls. Both backends matched build `19dddedc-b63d-4410-bdf0-79213658ea46`.

## What is stored

| Evidence | Stored location |
|---|---|
| Concrete rule registration, method/options selection and handler mapping | Neo4j function `App.add_url_rule`, `src/flask/sansio/app.py:606-662` |
| Route decorator returning the original function | Neo4j function `Scaffold.route`, `src/flask/sansio/scaffold.py:344-373` |
| Nested call to `self.add_url_rule` | Neo4j function `Scaffold.route.decorator`, lines 368-371 |
| Setup wrapper calling `_check_setup_finished` | Neo4j functions `setupmethod` and `setupmethod.wrapper_func`, lines 42-49 |
| Rejection after the first request | Neo4j function `App._check_setup_finished`, `src/flask/sansio/app.py:411-421` |
| `@setupmethod` applied to the routing methods | Full Neo4j class content and Chroma source chunks covering the decorator lines |

The Chroma chunks containing the relevant decorators cover app lines 598-617
and scaffold lines 327-349. This confirms storage, not that a particular
semantic query will rank those chunks highly; no embedding query was run.

## What the agent does not automatically receive

`build_graph.py` extracts a Tree-sitter `function_definition` starting at its
`def` line. The preceding `@setupmethod` belongs to a surrounding
`decorated_definition` and is outside the stored method content. Consequently,
`find_symbols(include_code=true)` for `App.add_url_rule` or `Scaffold.route`
does not show that decorator. Their complete enclosing classes do contain it,
but bounded class excerpts may end well before the relevant method.

The only stored relationship types are `DEFINED_IN`, `CALLS` and
`INHERITS_FROM`. There is no decorator-application relationship. Relevant
existing call edges are:

```text
App.add_url_rule                -> _endpoint_from_view_func
Scaffold.route.decorator       -> Scaffold.add_url_rule
setupmethod.wrapper_func       -> Scaffold._check_setup_finished
```

The latter two target the base Scaffold methods under static resolution.
They do not establish dispatch to concrete App overrides. `get_callees` for
the outer `Scaffold.route` also does not include the call inside its separate
nested decorator symbol. A graph-only traversal therefore does not supply a
complete explanation of runtime registration and setup checking.

The setup restriction is present in storage, but retrieving only the selected
method body can hide the clue that connects it to that restriction. This is a
retrieval/extraction gap, not missing setup-check implementation code. It is
not established as the sole reason for omissions in the generated answers.

A small future indexing improvement would preserve decorator text or metadata
alongside each definition, while keeping its existing symbol identity and
definition location. Stored-content changes would require rebuilding the
index. This inspection and prompt update do not implement that change.

[routing-evidence.json](../data/benchmark_results/2026-11-05-excerpts-guided/routing-evidence.json)
preserves the stored definitions, class-content checks, call edges and the two
Chroma source chunks. Prompt instructions now explicitly avoid verification
rereads of already-returned code, but retain targeted reads for essential
information actually absent from tool results.
