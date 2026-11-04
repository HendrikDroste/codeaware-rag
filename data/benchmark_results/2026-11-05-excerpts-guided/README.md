# Repeated guided benchmark with source excerpts

Questions 11, 16 and 18 were run in guided mode, with three repetitions of each
native baseline/Codeaware pair: 18 successful agent turns and nine pairs.
Every turn reported token usage. The new instrumentation records successful
excerpt-returning calls, excerpt rows, truncations and MCP errors.

The combined excerpt feature and evidence-reuse prompt did not reduce overall
reported token usage in this batch. Codeaware used 28.34% more total tokens;
only one of nine pairs used fewer tokens. This is a small development benchmark,
not evidence of a production quality or business-value improvement.

## Setup and controls

- Source commit: `70d58de57814cbefdc4414e077b34fd9b6923d37`.
- Dataset: `flask-49943b5a-v2`, pinned Flask revision
  `49943b5a94413a530761350f04e896f76e8d620e`.
- Host: Codex CLI `0.166.0-alpha.6`, model `gpt-6.1-sol`, medium reasoning.
- Prompt: `repository-qa-v2`, guided policy. Tool metadata:
  `source-excerpts-v3`. Excerpt/error metrics: `codeaware-tool-usage-v1`.
- Index: existing build `19dddedc-b63d-4410-bdf0-79213658ea46`.
- Before model calls, a real stdio MCP lookup returned a three-line excerpt
  from Neo4j, checked against local Flask source. The corpus and index were
  fresh. See [preflight.json](preflight.json).
- Both conditions received identical prompts within each pair, fresh ephemeral
  sessions, the same native tools, read-only source access and disabled web,
  apps, plugins, memories and delegation. Only Codeaware added MCP tools.
- Pair order reversed across questions and repetitions. With three repeats,
  which condition goes first is split 2:1 for each question, not perfectly
  balanced. Provider caching and order effects remain possible.
- Source hashes and a core-fact checklist were recorded before model runs;
  they were checked unchanged after completion. Corpus hashes were checked
  before and after every turn. The batch evaluates excerpts and the updated
  prompt together; it cannot isolate their individual effects.

The local run began after midnight on 5 November 2026 in Europe/Berlin.
Machine-readable provenance timestamps are stored in UTC. The temporary local
Neo4j instance used port 17687 and was stopped after the measurements; its data
is separate from the personal default database.

## Aggregate results

| Measurement | Native baseline | Codeaware available |
|---|---:|---:|
| Completed turns | 9 | 9 |
| Input tokens | 572,613 | 736,669 |
| Cached input, included above | 461,952 | 637,952 |
| Uncached input | 110,661 | 98,717 |
| Output tokens | 8,315 | 8,872 |
| Total input + output tokens | 580,928 | 745,541 |
| Mean elapsed seconds/turn | 26.06 | 30.32 |
| Native command calls | 29 | 20 |
| MCP calls | 0 | 41 |
| MCP errors | 0 | 0 |
| Calls returning excerpts | 0 | 19 |
| Returned excerpt rows | 0 | 25 |
| Truncated excerpt rows | 0 | 10 |

All nine Codeaware turns called MCP. Six of those turns received excerpts;
the other three used existing symbol/graph tools without receiving excerpts.
Counters describe delivered results, not proof that the model used the text.
An MCP call returning several matching symbols can deliver multiple excerpts.
The same symbol returned in another call is counted again. Structured and
text representations of the same result are not counted twice.

The 18 turns reported 1,326,469 tokens in total. Cached input is counted once
inside input; reasoning output is not added again to output. Uncached input
was 10.79% lower with Codeaware, but total usage was higher and caching/order
effects remain; this is not a currency-cost estimate or a reliable saving.

Native command calls fell by nine, but 41 MCP calls were added. That leaves
61 completed tool operations with Codeaware versus 29 in the native baseline.
An operation is not necessarily a separate model request: calls may be batched,
and one shell operation can contain several reads. The traces still show that
the tools did not consistently replace enough inspection to reduce usage.

## Repeated pair variation

| Question | Native mean tokens | Codeaware mean tokens | Change in summed tokens | Range of paired changes |
|---|---:|---:|---:|---:|
| 11: Blueprint inheritance | 60,812.7 | 86,941.7 | +42.97% | -5.86% to +90.91% |
| 16: Route registration | 66,391.0 | 74,963.3 | +12.91% | +2.02% to +38.78% |
| 18: Request creation | 66,439.0 | 86,608.7 | +30.36% | +9.16% to +59.27% |

Each question has three matched observations. The large ranges show why one
pair should not be treated as a stable estimate. No significance test or
confidence interval is claimed. Do not compare these aggregates directly to
the earlier single-run pilot as proof that a feature helped or harmed: the
sample size, prompt and optional feature use differ.

## Evidence and answer quality

All 25 returned excerpts were checked against the local pinned source for
content, inclusive ranges, requested line caps and truncation flags. All
matched. A structural audit checked 148 Markdown source-link destinations and
line bounds, with no invalid destinations among those checked. This audit
does not establish that the cited lines support every claim.

The [rubric](rubric.json) states core facts before this batch. A manual check of
the first repetition's six answers found that both conditions covered those
facts. Codeaware's inheritance answer additionally included the optional
test-only subclass. This check was neither blinded nor independent, and the
remaining repetitions were not fully graded. No overall answer-quality
improvement is established. See [quality-review.json](quality-review.json).

## Artifacts

- [usage.csv](usage.csv): all 18 rows, including repetition and new counters.
- [run.json](run.json): exact common prompt and host settings.
- [summary.json](summary.json): condition aggregates, question means and all
  nine paired changes.
- [tool-calls.json](tool-calls.json): actual completed MCP calls and arguments.
- [excerpt-audit.json](excerpt-audit.json): source checks for every excerpt.
- [citation-audit.json](citation-audit.json): structural source-link audit.
- [answers.md](answers.md): all answers, with source-link destinations
  normalized to the pinned Flask revision for portability.
- [provenance.json](provenance.json): source hashes, checklist checksum and
  pre-run UTC timestamp.

Full raw answers, prompts, events and stderr remain unchanged under the
Git-ignored directory `data/agent_runs/2026-11-05-excerpts-guided/`, grouped
under `repeat-01/`, `repeat-02/` and `repeat-03/`.

To reproduce, use the pinned corpus and a running configured database:

```bash
python -m src.benchmark_agent \
  --repo /Users/hendrikdroste/PycharmProjects/flask \
  --index-dir /Users/hendrikdroste/PycharmProjects/codeaware-rag/.codeaware-index/flask \
  --model gpt-6.1-sol \
  --questions 11 16 18 \
  --tool-policy guided \
  --repeats 3 \
  --codex /Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex \
  --output-dir data/agent_runs/new-excerpt-batch \
  --timeout 300
```
