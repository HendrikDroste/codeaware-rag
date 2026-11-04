# Local Flask benchmark run, 4 November 2026

All 18 pinned questions were evaluated with the original embedding
configuration. A separate paired agent comparison completed all 36 runs with
reported usage. The Codeaware-enabled agent chose native search in every run.
This comparison therefore establishes a usable native-agent baseline and
measures tool adoption; it does not establish a benefit from Codeaware retrieval
or graph queries.

## Setup and provenance

- Dataset: `flask-49943b5a-v2`, revision
  `49943b5a94413a530761350f04e896f76e8d620e`.
- Codeaware source: `72ccfd6e1afd112f51b8fa3c5d424ea5f4e7aaeb`.
- Embeddings: `Lajavaness/bilingual-embedding-large`, downloaded snapshot
  `e83179d7a66e8aed1b3015e98bb5ae234ed89598`; 900-character chunks, no overlap.
- Index build: `19dddedc-b63d-4410-bdf0-79213658ea46`.
- Host: Codex CLI `0.166.0-alpha.6`, `gpt-6.1-sol`, medium reasoning effort.
- Python 3.11.14 on the local Mac. Index embedding used MPS; the standalone
  retrieval evaluator used CPU. Codex inference used the authenticated account.
- The graph used an isolated local Neo4j instance on port 17687, with data
  separate from the personal default database. The index is stored in
  `.codeaware-index/flask`; the embedding cache used a temporary local directory.
  This temporary Neo4j instance was stopped after the measurements; querying
  the saved index requires restarting its database or rebuilding against your
  configured running Neo4j instance.

The index contains 83 files, 933 chunks, 1,462 functions/methods and 160 classes,
with 779 resolved call-site edges and 76 inheritance edges. Its statistics also
record 1,407 unresolved calls, 50 ambiguous calls, 1,431 external calls,
237 calls outside function scope, and 50 unresolved bases. These counts describe
a partial static graph, not runtime coverage.

[provenance.json](provenance.json) records dependency versions, the full index
manifest and local artifact locations. The Flask checkout retained only its
pre-existing untracked `.DS_Store`; no source changes were made.

## Retrieval results

The original embedding pipeline retrieved five snippets for each question.

| Metric | Mean across 18 questions |
|---|---:|
| MRR | 0.4722 |
| Unique labelled-line coverage | 25.14% |
| Expected-range hit ratio (`file_match_ratio`) | 41.20% |

Seven questions had no overlap with the selected gold ranges in the top five.
The labels are manually reviewed seed evidence, and some broad questions admit
other valid evidence. These scores measure overlap with those ranges, not final
answer correctness. They are not directly comparable to the historical CSVs,
which use older labels, corpus revisions and coverage accounting.

See [retrieval-summary.json](retrieval-summary.json) for per-question scores,
[retrieval.csv](retrieval.csv) for the ranked snippets, and
[models.csv](models.csv) for the aggregate export.

## Agent comparison

Both conditions used the same questions and prompt, fresh ephemeral sessions,
the same native repository tools and model settings. The second condition also
had Codeaware MCP available. Pair order alternated across questions. The
runner checked corpus hashes before and after each run. Model downloads and
index construction finished before the comparison started.

| Measurement | Native Codex | Codex + Codeaware available |
|---|---:|---:|
| Completed runs / questions | 18 | 18 |
| Reported input tokens | 871,979 | 897,346 |
| Cached input, included above | 637,568 | 691,968 |
| Uncached input | 234,411 | 205,378 |
| Reported output tokens | 11,897 | 11,594 |
| Total input + output tokens | 883,876 | 908,940 |
| Mean total tokens per question | 49,104 | 50,497 |
| Mean elapsed seconds per question | 19.59 | 19.51 |
| Native command calls | 49 | 48 |
| Codeaware calls | 0 | 0 |

The Codeaware-enabled condition used 2.84% more total tokens in this single run.
It used fewer uncached input tokens, but cache/order effects and ordinary
variation prevent interpreting that as savings. Cached tokens are already part
of input; reasoning tokens are already part of output. Neither is added again.
The main comparison consumed 1,792,816 reported tokens in total. Setup and
diagnostic turns are separate from these totals.

Most importantly, adding the tools did not cause this model to choose them for
these questions. The observed token and time differences do not measure the
effect of semantic retrieval or graph traversal. The separate tool-directed
walkthrough uses different instructions and must not be merged into this table.

All runs exited successfully, produced nonempty answers, and reported usage.
A structural audit found 197 Markdown source links with valid files and line
bounds, with no invalid links among those checked. This audit does not check
whether the cited source supports each claim or cover unlinked references.
Selected answers for questions 11, 16 and 18 were also compared with the relevant
source: they identified the Blueprint inheritance chain, route registration
and the current `AppContext.from_environ` request-creation path. No exhaustive
or independent answer-quality grading was performed.

See [agent-usage.csv](agent-usage.csv), [agent-summary.json](agent-summary.json),
[agent-run.json](agent-run.json), and [citation-audit.json](citation-audit.json).
Raw prompts, answers, JSONL events and stderr remain in the Git-ignored local
directory `data/agent_runs/2026-11-04-full/`.

## Separate tool-directed walkthrough

After the comparison, question 16 was run with explicit instructions to use
Codeaware's semantic search, symbol lookup, callers, callees and inheritance.
All 12 MCP calls succeeded and covered all six exposed tool types. The agent
also made two native command calls. The turn reported 172,832 total tokens
(170,864 input, including 123,136 cached; 1,968 output) and took 57.34 seconds.
Its extra instructions and workflow make it a diagnostic rather than a matched
comparison with the native baseline.

The graph established `Flask -> App -> Scaffold`, returned callers of
`App.add_url_rule`, and resolved `_endpoint_from_view_func` as a callee. It did
not return the decorator-to-concrete-method edge or external rule/map calls.
The answer used source inspection to fill those gaps and explicitly described
the graph as partial. This is a useful interview example of the boundary
between static evidence and the agent's source-based reasoning.

The [answer](walkthrough-answer.md), [actual prompt](walkthrough-prompt.txt),
[tool-call summary](walkthrough-tools.json), and
[usage](walkthrough-usage.json) are saved separately. Full raw events remain in
`data/agent_runs/2026-11-04-tool-walkthrough/`.

Successful setup checks and this diagnostic consumed another 274,091 reported
tokens, for 2,066,907 known reported tokens across all successful host turns in
this local exercise. The initial sandbox-blocked CLI attempt reported no usage;
its usage remains unknown rather than being counted as zero.
[supplemental-usage.json](supplemental-usage.json) preserves those separate
records. Local embedding and indexing work is outside these host totals.

## Reproduce

Follow the main installation instructions, use the pinned Flask corpus, build
the index and start Neo4j. Set `NEO4J_*` and the appropriate embedding cache
environment before launching the agent runner. For offline embedding use,
the selected model must already be cached.

```bash
python -m src.benchmark_agent \
  --repo /Users/hendrikdroste/PycharmProjects/flask \
  --index-dir /Users/hendrikdroste/PycharmProjects/codeaware-rag/.codeaware-index/flask \
  --model gpt-6.1-sol \
  --questions 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
  --codex /Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex \
  --output-dir data/agent_runs/new-run \
  --timeout 300
```

For retrieval, [run_retrieval.py](run_retrieval.py) uses the existing pipeline
and evaluator. Copy it into a new sibling result directory before rerunning,
so the recorded results remain intact, and launch it from the repository root
with `PYTHONPATH=.`. The script writes its exports alongside itself.

One run per question/condition gives no estimate of repeat variability. The
questions mostly concern familiar Flask APIs and named symbols; they do not
establish performance on unfamiliar repositories, code changes or debugging.
Before making a business-value claim, assess answer quality independently and
repeat a controlled comparison on representative engineering tasks.
