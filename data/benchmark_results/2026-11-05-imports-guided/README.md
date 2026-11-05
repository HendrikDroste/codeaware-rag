# Guided rerun after import tracking and evidence-reuse instructions

All 18 agent turns completed successfully: questions 11, 16 and 18, three
repetitions, with a fresh native baseline and Codeaware condition in each pair.
Codeaware used **15.46% more total tokens** than its baseline. It made **zero
native command calls**, but added 50 MCP calls. Only one of nine pairs used
fewer tokens. No overall answer-quality improvement is established.

The import tool was available in all Codeaware turns but was **never selected**.
The preflight verified it directly. This benchmark does not establish its
usefulness for an agent or measure import-specific question quality.

## Setup and controls

- Source commit: `74829c5e18b0e6d83de72bfebc2f11c41caf3909`.
- Pinned Flask: `49943b5a94413a530761350f04e896f76e8d620e`, dataset
  `flask-49943b5a-v2`.
- Host: Codex CLI `0.166.0-alpha.6`, `gpt-6.1-sol`, medium reasoning.
- Guided prompt: `repository-qa-v3`; server metadata: `imports-v5`.
- Index build: `a1670bcd-c98a-48f3-9ab5-f687f093015e`, graph schema 3.
- Original Lajavaness embedding model, 900-character chunks, zero overlap:
  83 files, 933 chunks, 1,622 symbols, 76 inheritance edges, 781 call edges,
  308 import edges. Counts and source hashes are in
  [index-manifest.json](index-manifest.json).
- The 648 imported bindings comprise 308 stored edges, 106 unresolved,
  two ambiguous and 232 external/unindexed bindings. There were no wildcards.
- A real stdio preflight checked a source excerpt and import-site locations
  before model calls. All seven tools were discovered. See
  [preflight.json](preflight.json).
- The previous checklist was copied unchanged before model calls. Source
  fingerprints and its checksum were recorded in [provenance.json](provenance.json)
  and verified unchanged after the batch.
- Both conditions received byte-identical prompts within each pair, the same
  model and native tools, fresh ephemeral sessions and read-only source access.
  Web, apps, plugins, memories and delegation were disabled. Only the Codeaware
  condition added MCP.
- Corpus hashes were validated before and after every turn. The final index
  remained fresh, and actual relationship counts matched its manifest. See
  [postflight.json](postflight.json).
- Pair order alternated across questions and repetitions; three repeats give
  each question a 2:1 order split. This does not remove cache/order effects.

The agent runs took place on 5 November 2026, approximately 08:14–08:24
Europe/Berlin. Artifact timestamps are UTC. The original embedding model was
loaded from the local cache. Indexing and the preflight are outside agent token
usage. Neo4j ran in the isolated test instance on port 17687; it was stopped
after verification, without changing the personal default database or clients.

## Aggregate results

| Measurement | Native baseline | Codeaware available |
|---|---:|---:|
| Completed turns | 9 | 9 |
| Input tokens | 612,253 | 708,243 |
| Cached input (included above) | 500,480 | 523,264 |
| Uncached input | 111,773 | 184,979 |
| Output tokens | 7,678 | 7,527 |
| Total input + output tokens | 619,931 | 715,770 |
| Native command calls | 27 | 0 |
| MCP calls | 0 | 50 |
| Reported MCP errors | 0 | 0 |
| Calls returning excerpts | 0 | 29 |
| Excerpt rows | 0 | 43 |
| Truncated excerpt rows | 0 | 13 |
| Mean elapsed seconds/turn | 30.88 | 31.52 |

All nine Codeaware turns called MCP and received excerpts. All 43 excerpt rows
matched the pinned source, inclusive ranges, requested line caps and truncation
flags. The structural citation audit checked 138 Markdown source-link targets
and bounds, with no invalid destinations. That audit does not establish support
for every claim or cover bare textual citations.

The 18 turns reported **1,335,701 tokens** in total. Cached input is already part
of input, and reasoning output is not added again. Uncached input was **65.50%
higher** with Codeaware. Total tokens and uncached input are not currency costs;
provider-cache and subscription effects prevent a reliable cost claim.

## Per-question results and remaining overhead

| Question | Native mean tokens | Codeaware mean tokens | Change in sums | Paired change range |
|---|---:|---:|---:|---:|
| 11 | 62,448.0 | 65,986.0 | +5.67% | +0.38% to +18.05% |
| 16 | 72,616.0 | 71,336.0 | -1.76% | -25.85% to +31.83% |
| 18 | 71,579.7 | 101,268.0 | +41.48% | +23.31% to +72.31% |

- Q11, inheritance: four MCP calls per turn, 12 in total. Correct production
  hierarchy, with consistent discovery of the test-only subclass.
- Q16, routing: 13 MCP calls across three turns. Its negative aggregate change
  comes from one large native baseline run; the other two pairs used more
  tokens with Codeaware. This is not a reliable saving.
- Q18, request creation: 25 MCP calls across three turns, including four empty
  lookups for `RequestContext` or `RequestContext.__init__`. This revision uses
  `AppContext.from_environ`. The host recovered through further graph/symbol
  queries and semantic search. Repetition 3 made nine MCP calls and reported
  120,633 tokens versus its baseline's 70,009.

The current overhead is not from reopening native source: there were no native
command calls in any Codeaware turn. The traces instead show more context-tool
operations: 50 versus 27 native commands in the baseline. One command may bundle
multiple searches/reads, and one operation need not be a separate model request.
This supports extra query steps as a plausible explanation, without attributing
an exact token amount to a particular call.

Tool use: 33 `find_symbols`, six `get_inheritance`, seven `get_callees` and four
`search_code` calls. `get_imports` was unused. The unknown-argument audit found
zero calls with arguments absent from the discovered schemas; the previous
`include_source` mistake did not recur. Zero reported MCP errors does not imply
all useful facts were retrieved, and an empty exact-name lookup is not an error.

## Answer quality

The implementing agent manually reviewed all 18 answers against the unchanged
[core-fact checklist](rubric.json). This review was not blinded or independent,
and it is not an exhaustive sentence-level grade or aggregate numerical score.
See [quality-review.json](quality-review.json).

| Observed coverage, out of three answers per question/condition | Native | Codeaware |
|---|---:|---:|
| Q11: correct production inheritance chain | 3 | 3 |
| Q11: explicit public `flask.Blueprint` export | 2 | 0 |
| Q11: optional test-only `MyBlueprint` | 1 | 3 |
| Q16: all required registration/dispatch criteria | 3 | 3 |
| Q16: optional setup guard after the first request | 1 | 0 |
| Q18: construction and JSON-provider/context storage | 3 | 3 |
| Q18: explicit context-bound `LocalProxy` explanation | 3 | 0 |

All request answers explain the default/customizable request class, but the
Codeaware answers say `flask.request` becomes available without explaining the
proxy. Their combined third required criterion is therefore marked partial.
Missing facts are omissions, not false claims. No incorrect required facts were
identified within this scoped review.

Codeaware consistently adds the test subclass, but omits selected required
details more often in this batch. The results do not support an overall quality
improvement or a business-value claim.

## Comparison with the preceding batch

The [previous repeated excerpt batch](../2026-11-05-excerpts-guided/README.md)
used prompt `repository-qa-v2` and metadata `source-excerpts-v3`:

| Measurement | Previous batch | Current batch |
|---|---:|---:|
| Native total tokens | 580,928 | 619,931 |
| Codeaware total tokens | 745,541 | 715,770 |
| Codeaware excess over its paired baseline | +28.34% | +15.46% |
| Codeaware native command calls | 20 | 0 |
| Codeaware MCP calls | 41 | 50 |
| Codeaware turns receiving excerpts | 6/9 | 9/9 |

Codeaware's absolute total fell 3.99%, while the native total rose 6.71%.
Both shifts contribute to the smaller relative gap. These are separate small
batches, not a controlled ablation: the prompt, tool inventory/descriptions and
index build changed together, and sampling/cache/order effects remain. The
rebuilt call graph also contains 781 edges versus the previous build's 779;
that resolution-count change was not isolated. Import-tool benefit cannot be
inferred from a batch with zero agent import calls.

The results support testing questions that explicitly require dependency or
public-export analysis before judging import tracking. They also motivate
improving discovery after an empty symbol lookup and returning missing decorator
or module context. No further runtime changes were made during this batch.

## Artifacts and reproduction

- [usage.csv](usage.csv), [run.json](run.json), [summary.json](summary.json):
  all 18 measured runs, exact prompt/settings, aggregates and all nine pairs.
- [answers.md](answers.md), [quality-review.json](quality-review.json): answers
  with source links normalized to the pinned revision, and manual review.
- [tool-calls.json](tool-calls.json), [native-commands.json](native-commands.json):
  actual operations and arguments. Response-character counts are not tokens.
- [excerpt-audit.json](excerpt-audit.json), [citation-audit.json](citation-audit.json),
  [argument-audit.json](argument-audit.json), [import-audit.json](import-audit.json):
  source/range checks, structural citations and argument/import-use audits.
- [provenance.json](provenance.json), [index-manifest.json](index-manifest.json),
  [preflight.json](preflight.json), [postflight.json](postflight.json): controls.

Unmodified raw prompts, events, answers and stderr remain in the Git-ignored
`data/agent_runs/2026-11-05-imports-guided/repeat-*/q*-{baseline,codeaware}/`.
The index-build log is retained under the ignored `.codeaware-index/flask/`.

With the pinned Flask checkout, matching rebuilt index and configured Neo4j:

```bash
python -m src.benchmark_agent \
  --repo /Users/hendrikdroste/PycharmProjects/flask \
  --index-dir /Users/hendrikdroste/PycharmProjects/codeaware-rag/.codeaware-index/flask \
  --model gpt-6.1-sol \
  --reasoning-effort medium \
  --questions 11 16 18 \
  --tool-policy guided \
  --repeats 3 \
  --codex /Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex \
  --output-dir data/agent_runs/new-imports-batch \
  --timeout 300
```
