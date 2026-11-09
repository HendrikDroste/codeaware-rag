# Neo4j-only benchmark, 7 November 2026

The current configuration answered all 18 pinned Flask questions once with native
Codex tools and once with native tools plus Codeaware. All 36 runs completed with
reported usage and no MCP errors. Codeaware used **6.71% fewer total reported
tokens**, with fewer tokens on 13 of 18 questions. The retrospective, unblinded
[quality review](quality-review.md) found **17 ties and one baseline completeness
advantage**. These are observations from one run per condition, not a stable
effect estimate or evidence of business ROI.

## Measurements

| Measurement | Native baseline | Native + Codeaware |
|---|---:|---:|
| Completed runs with reported usage | 18 | 18 |
| Input tokens | 1,016,580 | 949,045 |
| Cached input tokens, included above | 864,640 | 791,680 |
| Uncached input tokens | 151,940 | 157,365 |
| Output tokens | 10,626 | 9,217 |
| Total reported tokens | 1,027,206 | 958,262 |
| Uncached input + output | 162,566 | 166,582 |
| Tokens per question, Q1 | 50,890.25 | 41,839.25 |
| Tokens per question, median | 52,521.5 | 54,707.5 |
| Tokens per question, Q3 | 61,270.0 | 57,879.75 |
| Native command calls | 45 | 21 |
| Codeaware calls | 0 | 28 |
| MCP errors | 0 | 0 |
| Sum of elapsed seconds | 390.118 | 384.185 |

Total tokens fell by 68,944. Cached input fell by 72,960, while uncached input
plus output **rose by 4,016 (+2.47%)**. Do not describe this as monetary cost
savings. The median per-condition token count rose by 4.16%, despite the lower
total. The median of the 18 paired percentage changes is -14.99%, a different
statistic. Quartiles use `statistics.quantiles(n=4, method="inclusive")`.

Codeaware made eight semantic searches, 19 symbol lookups and one inheritance
query. It used fewer native commands, but more total native-plus-MCP calls
(49 versus 45). This evaluates the combined tools, with very little graph
traversal; it does not establish a graph-specific benefit. Eight semantic
searches returned 25 snippets. The runner also recorded 19 successful symbol
excerpt calls, 25 symbol excerpts and six truncated excerpts. A truncated
excerpt is not a tool error.

## Paired observations

| Question | Native tokens | Codeaware tokens | Change | Quality |
|---|---:|---:|---:|---|
| Q1 | 51,381 | 40,761 | -20.67% | tie |
| Q2 | 38,737 | 54,482 | +40.65% | tie |
| Q3 | 50,790 | 42,785 | -15.76% | tie |
| Q4 | 50,119 | 40,531 | -19.13% | tie |
| Q5 | 54,711 | 44,387 | -18.87% | tie |
| Q6 | 49,861 | 41,418 | -16.93% | tie |
| Q7 | 51,329 | 40,899 | -20.32% | tie |
| Q8 | 53,353 | 56,853 | +6.56% | tie |
| Q9 | 40,067 | 54,933 | +37.10% | tie |
| Q10 | 62,094 | 43,363 | -30.17% | tie |
| Q11 | 67,253 | 57,693 | -14.21% | tie |
| Q12 | 52,867 | 41,524 | -21.46% | tie |
| Q13 | 86,809 | 85,525 | -1.48% | tie |
| Q14 | 84,605 | 59,415 | -29.77% | tie |
| Q15 | 52,176 | 57,942 | +11.05% | baseline completeness advantage |
| Q16 | 71,065 | 65,392 | -7.98% | tie |
| Q17 | 51,191 | 73,197 | +42.99% | tie |
| Q18 | 58,798 | 57,162 | -2.78% | tie |

## Controls and provenance

- Dataset: `flask-49943b5a-v2`, revision
  `49943b5a94413a530761350f04e896f76e8d620e`.
- Model: `gpt-6.1-sol`, reasoning `low`, guided prompt `repository-qa-v6`,
  tool descriptions `neo4j-navigation-v10`, one repetition, 300-second timeout.
- Host: `codex-cli 0.167.0-alpha.3.2`. Both conditions used the same prompt,
  model and native tools. Codeaware alone received the extra MCP server.
  Pair order alternated by question. Personal configuration, web, apps,
  plugins, memories and delegation were disabled by the runner.
- Fresh current Neo4j index: build `dbc32ccf-f6e0-4c20-9a2c-12b304a53300`,
  83 Python files, 933 chunks, 1,024-dimensional
  `Lajavaness/bilingual-embedding-large` embeddings. Cached local model files
  were used offline. The vector index was ONLINE and an actual semantic
  query succeeded before execution. The index was not rebuilt for this run.
- Dataset validation ran before and after every condition. Runtime source
  fingerprints remained unchanged during execution. The pre-existing removal
  of an unused logging import was retained. The isolated local Neo4j instance
  was used without altering the personal database.
- The shell's npm Codex wrapper was broken. The runner used the already
  installed desktop Codex executable, recorded with its hash in
  [provenance.json](provenance.json). No client installation or personal
  configuration was changed.

The exact host prompt is in [run.json](run.json), per-condition measurements in
[usage.csv](usage.csv), and all original answer text in [answers.md](answers.md).
[analysis.json](analysis.json) contains derived figures. The raw prompts, JSONL
host events and stderr remain under the ignored local directory
`benchmarks/runs/2026-11-07-neo4j-low-full-guided/`; their hashes are retained in
[trace-hashes.json](trace-hashes.json).

## Interpretation for the interview

Use the title **Evaluation and lessons learned**. Explain the hypothesis, the
working implementation, the comparison and the next decision. The current
run suggests that bounded retrieval can reduce reported context use on some
repository questions. It found no quality advantage, does not prove monetary
savings, and does not measure completed maintenance work.

The [historical run](../2026-11-05-low-full-guided/README.md) remains unchanged:
it used 17.85% more reported tokens and its review found 15 ties and three
baseline completeness advantages. The host version, prompts, tool metadata and
storage changed between runs. Their difference cannot identify which change
caused an improvement. Do not combine both runs into one estimate.

Quality grading was retrospective and unblinded by the implementing assistant,
using the unchanged historical rubric. Examples were source-reviewed, not
executed. Cached input is included, local indexing/embedding effort is excluded,
and elapsed time includes startup/model loading. No repeated observations,
independent quality review, graph ablation, productivity study or business ROI
measurement was performed. A useful next evaluation would verify a focused
maintenance outcome with tests, measure task time and include indexing effort.
