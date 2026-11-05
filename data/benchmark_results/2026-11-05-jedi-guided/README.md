# Small guided rerun with Jedi navigation tools

Q11, Q17 and Q18 were run once per condition: six successful turns, low
reasoning, guided mode. Codeaware used **26.74% more total tokens** than its
new native baseline. In the preceding navigation pilot, the gap on the same
questions was 21.79%. These separate single-observation batches do not establish
a causal regression or a stable effect size.

**None of the three new Jedi tools was selected by the agent.** Their preflight
and implementation tests establish functionality, but this batch demonstrates
no agent-level benefit from adding them. The quality checklist results are
unchanged from the preceding pilot.

## Setup and controls

- Source commit: `2777b92`.
- Model: `gpt-6.1-sol`, low reasoning, CLI `0.166.0-alpha.6`.
- Prompt `repository-qa-v5`; metadata `python-navigation-v8`; ten MCP tools.
- Pinned Flask revision `49943b5a94413a530761350f04e896f76e8d620e`.
- Unchanged schema-3 index `a1670bcd-c98a-48f3-9ab5-f687f093015e`:
  83 Python files, 933 chunks and 308 import edges. No rebuild was required.
- A real stdio preflight discovered ten tools, returned a Neo4j source excerpt
  matching the pinned source, and a Jedi outline containing `request_class`.
- Source hashes and the previous question rubric were recorded before model
  calls and verified unchanged afterward. Corpus validation ran before and
  after every turn; the index remained fresh after the batch.
- Pair prompts were byte-identical. Both variants used the same native tools,
  fresh ephemeral sessions, read-only access and disabled web/apps/plugins/
  memories/delegation. Only the Codeaware variant added MCP tools.
- Pair order alternated by question. Three pairs, no repetitions; sampling,
  cache/order effects and combined prompt/tool changes prevent attribution.

## Results

| Metric | Native baseline | Codeaware |
|---|---:|---:|
| Successful turns | 3 | 3 |
| Input tokens | 191,865 | 243,990 |
| Cached input, included above | 157,184 | 209,280 |
| Uncached input | 34,681 | 34,710 |
| Output tokens | 1,973 | 1,681 |
| Total input + output | 193,838 | 245,671 |
| Native commands | 8 | 4 |
| MCP calls | 0 | 8 |
| Reported MCP errors | 0 | 0 |
| Excerpt-returning calls | 0 | 5 |
| Excerpt rows | 0 | 6 |
| Truncated excerpt rows | 0 | 3 |
| Mean seconds/turn | 29.36 | 35.41 |

| Question | Native tokens | Codeaware tokens | Current excess | Previous excess |
|---|---:|---:|---:|---:|
| 11: inheritance | 54,231 | 70,469 | +29.94% | +21.14% |
| 17: configuration | 68,439 | 93,279 | +36.30% | +42.06% |
| 18: request creation | 71,168 | 81,923 | +15.11% | +7.26% |

Codeaware added 51,833 total tokens: 52,125 more input and 292 fewer output.
The extra reported input consists of 52,096 cached tokens and 29 uncached tokens.
Cached input is not counted twice. These are reported token measurements,
not currency costs or subscription charges.

Both the current and preceding Codeaware pilots made eight MCP calls and four
native commands. This time the baseline made eight native commands rather than
seven. The larger tool inventory and changed guidance may affect request
context, but full model requests are not exposed in these traces; no precise
attribution of the token difference to extra tool definitions is possible.

## Actual tool use and quality

Codeaware used two `search_code`, five `find_symbols` and one
`get_inheritance` call. It made zero calls to `list_symbols`,
`resolve_definition` or `find_references`. There were also zero offset requests,
empty symbol matches or delivered empty-result hints.

Q17 used native text search to locate assignments, and both Q17/Q18 used
relevant native reads. The new tools were available, but the host retained
its existing workflow. The next useful experiment would target a task requiring
symbol outlines or non-call references, rather than assuming availability
implies adoption. No additional model experiments were run in this batch.

All six delivered excerpts matched source content, inclusive line ranges,
requested caps and truncation flags. The structural citation audit checked
36 source destinations and line bounds, with none invalid. This structural
check is separate from claim support.

The implementing assistant manually reviewed all six answers against the same
three essential facts per question recorded before this batch. On the coarse
0–2 rubric, all six score 2 on reviewed correctness and essential-claim citation
support. Completeness is unchanged from the previous navigation pilot:

| Question | Native /2 | Codeaware /2 | Finding |
|---|---:|---:|---|
| 11 | 2 | 1 | Correct chain and alias mapping; Codeaware still omits explicit public `flask.Blueprint` export. |
| 17 | 2 | 2 | Both identify `make_config` and the actual `Flask.default_config` mapping. |
| 18 | 2 | 2 | Both explain construction, customization and context-bound `LocalProxy`. |

The export and proxy requirements extend beyond minimal literal responses and
are retained for consistency with the earlier checklist. The review is
unblinded, performed by the implementing assistant and not an independent or
exhaustive quality evaluation. Examples were source-reviewed, not executed.
No quality improvement was observed on this rubric; no aggregate numerical
quality ranking or business-value claim is assigned.

## Artifacts

- [usage.csv](usage.csv), [run.json](run.json), [summary.json](summary.json):
  all six observations, exact prompt/settings and previous matched totals.
- [answers.md](answers.md), [quality-review.json](quality-review.json),
  [rubric.json](rubric.json): unchanged answers and scoped manual assessment.
- [tool-calls.json](tool-calls.json), [excerpt-audit.json](excerpt-audit.json),
  [citation-audit.json](citation-audit.json): actual calls and source checks.
- [preflight.json](preflight.json), [postflight.json](postflight.json),
  [provenance.json](provenance.json), [index-manifest.json](index-manifest.json):
  source lineage, discovery and freshness.

Raw events, prompts, answers and stderr remain unchanged under ignored
`data/agent_runs/2026-11-05-jedi-guided/q*-{baseline,codeaware}/`.
The isolated local Neo4j instance was stopped after the run. Personal client
settings and the default database were not changed. Runtime source remained
unchanged throughout; no separate model-based grading calls were made.
