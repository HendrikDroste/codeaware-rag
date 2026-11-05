# Q11, Q17 and Q18 after prompt and excerpt-navigation updates

All six turns completed successfully with reported usage and zero MCP errors:
three native baseline turns and three Codeaware turns, low reasoning, guided
mode, one observation per condition/question. Codeaware used **21.79% more total
tokens** than this baseline. The preceding low run used 56.56% more on these
same three questions, not the 17.85% excess across all 18 questions.

## Implementation and setup

- Implementation commit: `7948609`.
- Host: `gpt-6.1-sol`, low reasoning, CLI `0.166.0-alpha.6`.
- Prompt `repository-qa-v4`; tool metadata `excerpt-navigation-v7`.
- Pinned Flask `49943b5a94413a530761350f04e896f76e8d620e` and unchanged index
  `a1670bcd-c98a-48f3-9ab5-f687f093015e`, schema 3.
- `find_symbols` now returns a hint on empty matches and accepts an optional
  absolute `start_line` within a definition, with `include_code=true`.
  The 1–100 line bound, source freshness checks and symbol identity remain.
- With an offset, `code_excerpt.truncated` means any definition content is
  omitted, including preceding lines. No index rebuild was needed.
- All 79 tests passed, including real Chroma/Neo4j integration and MCP stdio
  offset retrieval. A separate Flask stdio preflight verified a class excerpt
  starting at line 237 and a hint for an unsupported attribute lookup.
- Source hashes and the previous quality rubric were recorded before agent
  calls; both remained unchanged. Corpus checks ran before/after each turn.
- Pair prompts were byte-identical, with the same native tools, model and
  reasoning level. Sessions were fresh and ephemeral, source access read-only,
  and web/apps/plugins/memories/delegation disabled. Only Codeaware added MCP.
- Pair order alternated across questions. There is one observation per pair;
  provider-cache, order and sampling effects are not controlled away.

## Token and tool results

| Metric | Native baseline | Codeaware |
|---|---:|---:|
| Input tokens | 172,621 | 210,482 |
| Cached input, included above | 141,568 | 179,968 |
| Uncached input | 31,053 | 30,514 |
| Output tokens | 1,617 | 1,720 |
| Total tokens | 174,238 | 212,202 |
| Native commands | 7 | 4 |
| MCP calls | 0 | 8 |
| Reported MCP errors | 0 | 0 |
| Excerpt-returning calls | 0 | 5 |
| Excerpt rows | 0 | 6 |
| Truncated excerpt rows | 0 | 3 |
| Mean seconds/turn | 32.23 | 35.69 |

| Question | Native tokens | Codeaware tokens | Current excess | Previous low excess |
|---|---:|---:|---:|---:|
| 11: Blueprint inheritance | 53,119 | 64,347 | +21.14% | +22.03% |
| 17: configuration defaults | 51,565 | 73,254 | +42.06% | +67.17% |
| 18: request creation | 69,554 | 74,601 | +7.26% | +75.15% |

For these questions, previous totals were 169,542 native and 265,430 Codeaware.
The current absolute Codeaware total fell 20.05%; native rose 2.77%. MCP calls
fell from 18 to eight; Codeaware native commands rose from one to four. The
extra native searches now directly cover information missing from graph tools.
These batches changed prompt and tools together, so this is not an isolated
measurement of any one feature or a stable saving estimate. Codeaware still
uses more total tokens than its current baseline on every question.

## Actual feature use and answer review

The agent made **zero `start_line` calls** and received **zero empty-result
hints**, because it made no empty symbol lookups. Both features work in the
preflight and tests, but this batch provides no evidence of their agent-level
benefit. Do not attribute the token reduction to unused features.

The changed workflow is visible in the traces: Q17 starts with semantic search,
uses targeted native text search for attributes, and retrieves `App.make_config`.
Q18 starts with semantic discovery and selects `AppContext.from_environ`
directly, avoiding the previous repeated `RequestContext` guesses. Relevant
native reads provide the request-class and proxy details.

All six excerpts matched the pinned source, inclusive bounds and line caps.
The structural citation audit checked 31 source destinations and bounds, with
none invalid. That check is not an exhaustive claim-support grade.

The implementing assistant manually reviewed all six answers against the
previously recorded question criteria, using the same 0–2 dimension anchors.
This review is unblinded and not independent; examples were not executed.
All six scored 2 on reviewed correctness and essential-claim citation support.
Completeness is:

| Question | Native /2 | Codeaware /2 | Finding |
|---|---:|---:|---|
| 11 | 2 | 1 | Correct hierarchy and alias mapping, but Codeaware still omits explicit public `flask.Blueprint` export. |
| 17 | 2 | 2 | Both locate the loading mechanism and the concrete `Flask.default_config` mapping. |
| 18 | 2 | 2 | Both explain construction, customization and the context-bound `LocalProxy`. |

Q17 and Q18 no longer have the completeness omissions recorded in the previous
low run. Q11 still does. The public-export and proxy criteria extend beyond the
minimal literal hierarchy/construction questions and are retained for rubric
continuity. These observations do not establish causal quality improvement.

## Artifacts

- [summary.json](summary.json), [usage.csv](usage.csv), [run.json](run.json):
  aggregates, all pairs and exact settings/prompt.
- [answers.md](answers.md), [quality-review.json](quality-review.json),
  [rubric.json](rubric.json): unchanged answers and manual review criteria.
- [tool-calls.json](tool-calls.json), [excerpt-audit.json](excerpt-audit.json),
  [citation-audit.json](citation-audit.json): actual use and source checks.
- [preflight.json](preflight.json), [postflight.json](postflight.json),
  [provenance.json](provenance.json), [index-manifest.json](index-manifest.json):
  discovery, freshness and source lineage.

Raw events, prompts, answers and stderr remain unchanged under ignored
`data/agent_runs/2026-11-05-navigation-guided/q*-{baseline,codeaware}/`.
The isolated local Neo4j instance was stopped after the run. Personal client
settings and the default database were not changed. No further runtime changes
were made during the benchmark, and no additional grading model was called.
