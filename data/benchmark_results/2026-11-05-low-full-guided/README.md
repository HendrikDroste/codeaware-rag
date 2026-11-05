# Full guided benchmark with low reasoning

All 18 pinned Flask questions, one observation per condition: 36 agent turns.
Model: `gpt-6.1-sol`, reasoning `low`, guided prompt `repository-qa-v3`, tool
metadata `imports-v5`. Both conditions use identical prompts within each pair;
Codeaware adds MCP to the same native tools. Pair order alternates by question.

Codeaware used +17.85% total tokens
relative to its new baseline. It used fewer tokens in 6/18 pairs.
This is a single observation per question/condition, without variance estimates
or an independent quality grade. No business-value improvement is established.

## Measurements

| Measurement | Native baseline | Codeaware |
|---|---:|---:|
| runs | 18 | 18 |
| completed | 18 | 18 |
| reported_usage | 18 | 18 |
| input_tokens | 969,246 | 1,144,742 |
| cached_input_tokens | 754,048 | 880,896 |
| uncached_input_tokens | 215,198 | 263,846 |
| output_tokens | 9,948 | 9,262 |
| total_tokens | 979,194 | 1,154,004 |
| command_calls | 41 | 3 |
| mcp_calls | 0 | 53 |
| codeaware_calls | 0 | 53 |
| mcp_errors | 0 | 0 |
| code_excerpt_calls | 0 | 27 |
| code_excerpts | 0 | 39 |
| excerpt_truncations | 0 | 12 |
| Mean seconds/turn | 24.22 | 27.24 |

Cached input is included in input; total is input plus output. Reasoning output
is not added again. These are reported host tokens, not currency costs.
Indexing is outside the host token totals.

## Question pairs

| Question | Native tokens | Codeaware tokens | Change |
|---|---:|---:|---:|
| 1 | 51,891 | 45,155 | -12.98% |
| 2 | 50,512 | 40,149 | -20.52% |
| 3 | 49,916 | 55,320 | +10.83% |
| 4 | 49,832 | 44,892 | -9.91% |
| 5 | 53,351 | 64,245 | +20.42% |
| 6 | 51,071 | 40,039 | -21.60% |
| 7 | 50,062 | 45,860 | -8.39% |
| 8 | 51,795 | 44,212 | -14.64% |
| 9 | 50,648 | 65,420 | +29.17% |
| 10 | 53,239 | 64,441 | +21.04% |
| 11 | 51,691 | 63,080 | +22.03% |
| 12 | 39,143 | 47,347 | +20.96% |
| 13 | 73,342 | 92,180 | +25.69% |
| 14 | 76,814 | 89,126 | +16.03% |
| 15 | 52,400 | 77,587 | +48.07% |
| 16 | 55,636 | 72,601 | +30.49% |
| 17 | 50,916 | 85,114 | +67.17% |
| 18 | 66,935 | 117,236 | +75.15% |

## Controls and scope

The Flask revision is `49943b5a94413a530761350f04e896f76e8d620e`.
The same existing schema-3 index was reused, build
`a1670bcd-c98a-48f3-9ab5-f687f093015e`, with 83 files, 933 chunks and 308 import
edges. A real Neo4j excerpt was checked before model calls. The corpus was
validated before and after every turn. Runtime source hashes remained unchanged.
Fresh ephemeral sessions used read-only source access; web, apps, plugins,
memories and delegation were disabled. The personal clients were not changed.
The isolated Neo4j test instance was stopped after final verification.

This full single-pass low run is not directly comparable to the aggregate of
the preceding medium run, which covers only questions 11, 16 and 18 with three
repetitions. Question mix, sampling and provider-cache/order effects differ.
Only compare matching questions when exploring reasoning-level differences,
and do not infer causality or stable savings from one low observation.

Answers were ungraded at initial publication. The subsequent
[quality review](quality-review.md) covers all 36 answers: 15 ties and three
baseline advantages on completeness, with no substantive errors identified
in the reviewed claims. This retrospective implementer review is unblinded;
it does not establish a quality advantage for Codeaware. No separate grading
API calls were made. The original usage and answers remain unchanged.

## Artifacts

- [summary.json](summary.json), [usage.csv](usage.csv), [run.json](run.json):
  all measurements and exact settings/prompt.
- [answers.md](answers.md): all 36 saved answers.
- [tool-calls.json](tool-calls.json): actual MCP calls and arguments.
- [provenance.json](provenance.json), [preflight.json](preflight.json),
  [index-manifest.json](index-manifest.json): source lineage and setup checks.

Unmodified raw events, prompts, answers and stderr remain in the ignored
`data/agent_runs/2026-11-05-low-full-guided/q*-{baseline,codeaware}/`.
The previous medium results remain unchanged. No runtime code was changed.

The [token-overhead investigation](token-analysis.md) examines every saved
trace. Additional discovery operations, unsupported attribute lookups and
docstring-heavy class prefixes explain specific wasted steps; larger returned
text or native verification rereads do not explain the aggregate increase.
Exact per-model-request token attribution is unavailable.
