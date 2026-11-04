# Agent baseline and token usage

The original `data/results_baseline.csv` compares embedding retrieval settings.
It contains no agent token usage, and its historical scores use old labels and
metrics. It is not an end-to-end coding-agent baseline.

`python -m src.benchmark_agent` runs the same pinned Flask questions in two
conditions, using Codex CLI as the host:

| Condition | Available repository tools |
|---|---|
| `baseline` | Native Codex tools, without Codeaware MCP |
| `codeaware` | The same native tools plus the six Codeaware MCP tools |

Every question/condition starts a fresh ephemeral session. Both use the same
model, reasoning effort, question and prompt prefix, with read-only source
access. User configuration, apps, plugins, memories, delegation and web/browser
tools are disabled for this controlled comparison. The Codeaware condition may
choose whether to call its additional tools; `codeaware_calls` records actual
use. A baseline trace containing MCP calls is marked invalid.

This compares native repository tools with Codeaware added. It does not compare
against a personalized plugin/LSP setup or a different agent/model. The runner
requires a checkout without project `.codex/config.toml` settings, preventing
that file from introducing extra MCP servers or model configuration.

## Run a pair

1. Install and authenticate Codex CLI. The runner reuses its authentication; it
   does not read or copy credential files or modify personal client settings.
2. Use the Flask revision pinned in
   [the dataset manifest](../data/benchmarks/flask/manifest.json).
3. Rebuild the Codeaware index after updating Flask, following the main README.
   Set `NEO4J_*` in the environment and keep Neo4j running.
4. Choose an explicit model available to your Codex account and a new output
   directory outside the Flask repository. Keep the index outside Flask too.

From the Codeaware root with its virtual environment active:

```bash
python -m src.benchmark_agent \
  --repo /absolute/path/to/flask \
  --index-dir /absolute/path/to/codeaware-index \
  --model YOUR_CODEX_MODEL \
  --questions 16 \
  --output-dir data/agent_runs/trial-01
```

The default is question 16 in **both** conditions: two model-backed agent runs.
This consumes your host's tokens or subscription allowance. Select more with
`--questions 11 12 16`; omitting `--questions` does not run all 18. Use
`--variant baseline` to collect only the agent baseline; that mode requires no
index or Neo4j connection. `--variant codeaware` runs only the added-tool case.

`--codex /absolute/path/to/codex` selects a CLI executable if it is not on PATH.
For example, the macOS desktop bundle may provide
`/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex`.
The runner needs a CLI supporting `exec --json --ephemeral --ignore-user-config`
and feature toggles; command construction was checked against local version
`0.166.0-alpha.5.1`. `--config` selects the same embedding configuration used at
index time. `--reasoning-effort` defaults to `medium`; `--timeout` defaults to
300 seconds per condition.

The corpus is checked before and after every condition. An outdated/missing
index is rejected before a paired run starts. It is not rebuilt automatically.
An existing output directory is refused to preserve previous traces. When
running multiple questions, which condition runs first alternates across them.
For repeats, use new directories and inspect variation rather than relying on
one pair. This does not eliminate shared provider-cache or order effects.

## Collected data

`usage.csv` has one row per question/condition:

- `input_tokens`: all input tokens reported by the host.
- `cached_input_tokens`: the reported cached subset of input tokens.
- `uncached_input_tokens`: input minus cached input, when both are reported.
- `output_tokens`: the host's reported output count.
- `reasoning_output_tokens`: the separate reasoning field, when reported.
- `total_tokens`: input plus output. Cached input and reasoning are not added
  again to their parent counts.
- Elapsed wall time, exit status, usage status, command/MCP call counts and
  actual Codeaware call count.
- Question, model, reasoning effort, CLI version, dataset/repository revision,
  index build identifier and usage metric version.

Usage comes from the single `turn.completed` event in the official
[Codex JSONL output](https://learn.chatgpt.com/docs/non-interactive-mode).
It covers the agent turn, including repeated model inputs during tool use;
counting only the visible answer or retrieved snippets would miss that work.
Reported output can include non-visible tokens, as explained in the official
[token-counting documentation](https://developers.openai.com/api/docs/guides/token-counting).

Missing fields remain blank, not zero. Multiple completion events are marked
ambiguous instead of potentially double-counting cumulative totals. Failed,
timed-out and malformed runs retain their logs; any available usage is recorded,
but absent usage remains unknown and must not be treated as free execution.

Each run also saves:

```text
trial-01/
  run.json
  usage.csv
  q16-baseline/
    prompt.txt
    events.jsonl
    stderr.log
    answer.md
  q16-codeaware/
    ...
```

`run.json` records the common settings. The traces preserve the raw reported
usage for inspection. `data/agent_runs/` is ignored by Git; retain selected runs
deliberately if you want to share them.

## Interpreting the comparison

The runner collects usage; it does not automatically grade answer quality.
Review the saved answers against the pinned evidence and score correctness and
source references before interpreting token differences. A successful CLI exit
does not prove a correct answer or successful MCP queries; inspect tool errors
and whether Codeaware was used. A wrong answer with fewer tokens is not a win.

Total tokens are not a currency cost. Cached input can have different pricing,
and subscription usage need not translate directly into a per-run bill. Local
embedding work and one-time indexing are outside these host token totals.
Wall time includes host/MCP startup and model loading for each fresh session;
it does not isolate steady-state retrieval latency.

Automated tests use simulated CLI traces and subprocess responses. The
[local run from 4 November 2026](
../data/benchmark_results/2026-11-04-local/README.md) records a separate live
18-question paired evaluation and tool-directed diagnostic. Codex chose native
tools in every paired run; the separate diagnostic successfully used Codeaware.
The report preserves this distinction when interpreting token usage.
