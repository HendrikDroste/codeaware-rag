# Tool guidance pilot, 4 November 2026

The updated descriptions explain tool selection, parameter formats and partial
static-graph coverage. The benchmark now supports `auto` and `guided` prompt
modes. This local pilot measures their use on questions 11, 16 and 18: Blueprint
inheritance, route registration and request creation.

All 12 turns completed successfully and reported usage. With the updated
descriptions and neutral `auto` prompt, Codex still chose native tools in all
three Codeaware runs. With the conditional `guided` prompt, it used Codeaware
in all three runs, making 10 successful MCP calls with no tool errors. This
shows adoption in these observations, not an improvement in correctness or
efficiency.

## Conditions and results

Both conditions within each mode received byte-identical prompts for each
question. They used `gpt-6.1-sol`, medium reasoning effort, CLI
`0.166.0-alpha.6`, fresh ephemeral sessions, and the pinned Flask corpus
`49943b5a94413a530761350f04e896f76e8d620e`. Pair order alternated across
questions. The automatic pilot ran before the guided pilot; shared cache and
order effects are not eliminated.

| Mode | Condition | Reported total tokens | Mean seconds/question | MCP calls |
|---|---|---:|---:|---:|
| auto | Native baseline | 148,117 | 24.94 | 0 |
| auto | Codeaware available | 182,785 | 26.09 | 0 |
| guided | Native baseline | 213,979 | 30.93 | 0 |
| guided | Codeaware available | 249,923 | 32.37 | 10 |

Each row contains three runs. Total means reported input plus output, including
cached input once. Across all 12 turns, 794,804 tokens were reported. These
figures exclude local indexing/embedding work and are not monetary costs.

Within `guided`, the Codeaware condition used 16.80% more total tokens than its
native baseline. Within `auto`, the difference was +23.41%, although no MCP
tools were called. Neither result establishes savings. Changing modes changes
the prompt for both conditions, so cross-mode differences do not isolate the
effect of tool metadata. One observation per question/condition gives no
estimate of repeat variability. These three structural questions do not replace
the original 18-question evaluation.

The guided calls were:

- Q11: `find_symbols` and three `get_inheritance` calls.
- Q16: two `find_symbols` and two `get_callees` calls.
- Q18: two `find_symbols` calls, followed by source inspection.

The agent did not call `search_code`, `get_callers` or `index_status` in this
pilot. It did not call every tool just to satisfy a fixed sequence. The Q16
graph query returned no callees for `Scaffold.route`, whose actual call is in
its nested decorator; the agent used source reads to explain registration.
Both guided Q11 answers identified the test-only `MyBlueprint` subclass, so
its appearance in the Codeaware answer is not evidence of an exclusive benefit.
The guided Q18 answer identified the current `AppContext.from_environ` path.
No exhaustive or independent answer-quality grading was performed.

## Artifacts and versions

- [summary.json](summary.json): aggregate usage and adoption.
- [auto-usage.csv](auto-usage.csv), [guided-usage.csv](guided-usage.csv): all
  per-question counts, including cached and uncached input.
- [auto-run.json](auto-run.json), [guided-run.json](guided-run.json): model
  settings and exact common prompt prefixes.
- [answers.md](answers.md): all 12 answers; source-link destinations normalized
  to the pinned Flask revision for portability. Raw answers remain unchanged.
- [citation-audit.json](citation-audit.json): source-link file and line-bound
  checks, which do not grade the claims supported by those links.
- [tool-calls.json](tool-calls.json): arguments, result counts and errors for
  every MCP call.
- [tool-definitions.json](tool-definitions.json): actual discovery metadata
  exported by the SDK, including parameter descriptions and read-only hints.
- [provenance.json](provenance.json): source commit, source hashes, schema
  checksum, index identity and raw trace locations.

The source commit is `b4adffbf115aa9b93f6a472da6868be598dd2107`. Both modes
record `prompt_version=repository-qa-v1` and
`tool_description_version=selection-guidance-v2`. The complete prompt for each
turn is retained under the Git-ignored raw directories:

```text
data/agent_runs/2026-11-04-guidance-auto/
data/agent_runs/2026-11-04-guidance-guided/
```

The existing index build `19dddedc-b63d-4410-bdf0-79213658ea46` was reused.
Corpus freshness was checked before and after every turn; metadata changes
required no reindexing. Retrieval and graph-query logic, the embedding model,
chunking and graph extraction were unchanged. The isolated local Neo4j instance was
stopped after the pilot, leaving the personal default database untouched.

For another run, follow the [benchmark instructions](
../../../docs/agent-benchmark.md#tool-selection-modes), use the same pinned
corpus and a running database, choose the mode with `--tool-policy auto` or
`--tool-policy guided`, and use a fresh output directory. Evaluate correctness
and repeat variability before interpreting usage differences as business value.
