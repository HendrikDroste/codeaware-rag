# Paired agent benchmark

`python -m benchmarks.run_agent` compares the same Codex host in two conditions:

| Condition | Available repository tools |
|---|---|
| `baseline` | Native Codex tools |
| `codeaware` | The same native tools plus Codeaware's seven MCP tools |

The runner uses the [pinned 18-question Flask dataset](flask/README.md). It
records usage and answers; it does not automatically grade answer quality.
No pandas, requests, retrieval-metric framework or additional Python package is
needed beyond the runtime installation in the [main README](../README.md).

## Run a pair

1. Install and authenticate Codex CLI. The runner reuses its authentication,
   without reading credential files or changing personal client configuration.
2. Use the Flask revision pinned in [the dataset manifest](flask/manifest.json).
   Keep the checkout unchanged during a run.
3. Build its Neo4j index using the main README and set `NEO4J_*` in the
   environment. Keep Neo4j running.
4. Choose a model available to your account and a fresh output directory
   outside Flask. Run from the Codeaware project root:

```bash
python -m benchmarks.run_agent \
  --repo /absolute/path/to/flask \
  --index-dir /absolute/path/to/codeaware-index \
  --model YOUR_CODEX_MODEL \
  --questions 16 \
  --tool-policy auto \
  --output-dir benchmarks/runs/trial-01
```

This starts two model-backed runs and consumes host tokens or subscription
allowance. The default question is 16, reasoning effort is `medium`, repetitions
are 1 and the timeout is 300 seconds per condition. No benchmark starts merely
by importing the package or running `--help`.

Useful options:

- `--variant baseline`: native baseline only; no index or Neo4j required.
- `--variant codeaware`: added-tool condition only.
- `--questions 11 17 18`: select one-based question numbers.
- `--reasoning-effort low --repeats 1`: explicitly select those settings.
- `--config /absolute/model.yaml`: same embedding configuration as indexing.
- `--codex /absolute/path/to/codex`: select the host executable.

The CLI must support `exec --json --ephemeral --ignore-user-config` and the
feature toggles used by the runner. The recorded full run used CLI
`0.166.0-alpha.6`; verify compatibility with your installed host before a run.

## Controls and tool selection

Each question/condition gets a fresh ephemeral session. The model, reasoning,
question and prompt match within each pair. Pair order alternates across
questions and repetitions. This does not eliminate provider-cache or order
effects, and one observation does not estimate variance.

Source access is read-only. Personal configuration, apps, plugins, memories,
delegation and web/browser tools are disabled. A checkout containing a project
`.codex/config.toml` is refused. MCP calls in the baseline make that run invalid.
The Codeaware condition may use native tools as well as MCP tools.

`--tool-policy auto` uses neutral tool-choice guidance. `guided` adds the same
conditional capability guidance to both conditions: semantic discovery for
unknown locations, exact lookup for known functions/classes and native text
search for assignments. It encourages reuse of returned source, relevant
methods over large class excerpts and graph calls only for unresolved questions.
It does not mandate MCP calls or impose a hard call limit.

Prompt text and versions are recorded in each run's metadata. The current
prompt is `repository-qa-v6`; current tool metadata is `neo4j-navigation-v10`.
Both runner and server share `CONTEXT_GUIDANCE` from `src/index_state.py`.
Recorded older versions remain unchanged in the evidence.

## Integrity and outputs

`dataset.py` checks the CSV checksum, dataset format, all Python file hashes,
question count, pinned source URL revisions and inclusive label ranges. Checks
run before and after each condition. Missing/stale indexes are rejected before
paired runs begin. The runner neither rebuilds indexes nor changes the corpus.
Existing output directories are refused.

A run records:

- `run.json`: dataset, model, settings, prompts and build/version identity.
- `usage.csv`: usage, timing, status, tool/excerpt counts and errors.
- Per-condition folders: raw `events.jsonl`, stderr, prompt and answer.

Input tokens include cached input. Total tokens are input plus output;
reasoning output is not added again. Uncached input is input minus cached input.
Missing usage remains unknown, not zero. Multiple completed turns are flagged
as ambiguous. Started/completed tool events are deduplicated by item ID.

The runner also records MCP calls, successful source-excerpt use, excerpt
truncations and tool errors. A completed process does not prove a correct
answer or successful tool use. Inspect the saved evidence.

## Interpretation and retained evidence

The [7 November Neo4j-only rerun](results/2026-11-07-neo4j-low-full-guided/README.md)
covers all 18 questions once per condition with low reasoning: 36 completed turns,
all with reported usage and no MCP errors. Codeaware used 6.71% fewer total
reported tokens, with fewer tokens on 13 of 18 questions. Its median token count
rose 4.16%. Cached input accounts for the aggregate reduction; uncached input
plus output rose 2.47%. The retrospective, unblinded
[quality review](results/2026-11-07-neo4j-low-full-guided/quality-review.md) found
17 ties and one baseline completeness advantage on the unchanged rubric.

The [historical run](results/2026-11-05-low-full-guided/README.md) used 17.85%
more total tokens and its review found 15 ties and three baseline completeness
advantages. It predates storage consolidation and tool simplification. Host,
prompt and tool metadata also changed, so cross-run differences cannot isolate
the effect of an individual change. Neither run isolates the graph's benefit.
The [evidence index](results/README.md) links both runs and archived pilot history.

Token totals are not monetary cost. Cached tokens may have different pricing,
and local embeddings/indexing are outside host usage. Wall time includes fresh
host/server startup and model loading. Repository Q&A does not establish
successful code changes, developer productivity or business ROI.

For a new claim, compare matching tasks and settings, verify outcomes, review
quality independently and collect repeated observations on representative work.
