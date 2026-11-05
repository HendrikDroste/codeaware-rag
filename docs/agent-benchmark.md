# Agent baseline and token usage

The original `data/results_baseline.csv` compares embedding retrieval settings.
It contains no agent token usage, and its historical scores use old labels and
metrics. It is not an end-to-end coding-agent baseline.

`python -m src.benchmark_agent` runs the same pinned Flask questions in two
conditions, using Codex CLI as the host:

| Condition | Available repository tools |
|---|---|
| `baseline` | Native Codex tools, without Codeaware MCP |
| `codeaware` | The same native tools plus the Codeaware MCP tools (currently seven) |

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
  --tool-policy auto \
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

## Tool-selection modes

`--tool-policy auto` is the default. It preserves the original neutral prompt:
the agent may use any available repository tools. Use this mode to measure
whether it chooses Codeaware with the current tool descriptions.

`--tool-policy guided` adds the same conditional workflow guidance to both
conditions. It asks the agent to choose tools by capability: semantic search
for unknown implementation locations, function/class lookup for those symbols,
and targeted native text search for assignments or attributes. After one empty
exact-name lookup, switch to discovery instead of guessing name variants.
Prefer a relevant method over a large class prefix, which may mostly contain
documentation. Query graph relationships only for unresolved relationship
questions. Start with 3..5 results and narrow by path when possible; neither
mode requires calling every tool, mandates an MCP call or sets a hard call cap.

The current guided template explicitly treats Codeaware's returned code as
indexed source evidence. It asks for `include_code=true` in the initial lookup
when implementation code is needed, prohibits reopening files merely to verify
already-returned code, and permits answering directly from returned paths and
line ranges. Truncation alone does not require another read. A targeted source
read or another query is warranted only for a specific essential missing fact
or unavailable tool capability. Stop when the answer and citations are supported,
and continue when essential evidence is missing. `auto` retains its original
neutral text.

These instructions are recorded as `prompt_version=repository-qa-v4` and
`tool_description_version=capability-guidance-v6`. The shared policy lives in
`src/index_state.py` as `CONTEXT_GUIDANCE`; the guided benchmark and server
instructions both reuse it. Both benchmark conditions still receive identical
prompts within each mode. The update contains no Flask-specific answers.

This combination has not been benchmarked yet. Earlier results retain their
original versions and measurements. Reconnect clients to discover the updated
server guidance; a prompt/description change does not require an index rebuild.

For a small pilot, run questions 11, 16 and 18 in `auto`, then repeat with
`--tool-policy guided` and a new output directory:

```bash
python -m src.benchmark_agent \
  --repo /absolute/path/to/flask \
  --index-dir /absolute/path/to/codeaware-index \
  --model YOUR_CODEX_MODEL \
  --questions 11 16 18 \
  --tool-policy guided \
  --output-dir data/agent_runs/guided-trial-01
```

Each three-question paired run makes six model-backed turns. Compare baseline
and Codeaware within the same mode first. Comparing modes changes the workflow
instructions too, so it does not isolate the effect of tool metadata. Do not
treat zero MCP calls as a retrieval-quality score, or an MCP call as proof of
an improved answer. Check adoption, correctness, usage and tool errors together.

Tool selection guidance lives in `src/mcp_server.py`: server instructions,
tool docstrings and parameter descriptions. The SDK derives tool descriptions
from docstrings. `Annotated[..., Field(description=...)]` adds parameter help
to the input schema; the shared `SymbolId` and `ResultLimit` aliases avoid
repeating that help. The existing argument validation and retrieval functions
still handle the calls.

Metadata changes require a fresh connection to rediscover tools; they do not
require rebuilding embeddings or the graph. The CLI starts a new server for
each Codeaware turn. Keep backend logic and corpus fixed when testing a
description change.

## Collected data

`usage.csv` has one row per question/condition/repetition:

- `input_tokens`: all input tokens reported by the host.
- `cached_input_tokens`: the reported cached subset of input tokens.
- `uncached_input_tokens`: input minus cached input, when both are reported.
- `output_tokens`: the host's reported output count.
- `reasoning_output_tokens`: the separate reasoning field, when reported.
- `total_tokens`: input plus output. Cached input and reasoning are not added
  again to their parent counts.
- Elapsed wall time, exit status, usage status, command/MCP call counts and
  actual Codeaware call count.
- `mcp_errors`: calls with transport errors, failed/cancelled status, or MCP
  `isError`/`is_error` flags. A recovered tool error is still counted even if
  the host turn succeeds. Host startup failures are recorded by turn status.
- `code_excerpt_calls`: completed, non-error Codeaware calls returning at
  least one nonempty `code_excerpt`.
- `code_excerpts`: number of those returned excerpts; `excerpt_truncations`
  counts excerpt rows whose `truncated` flag is true. These count returned
  rows across calls, not unique symbols or proof that the model used the text.
- Question, model, reasoning effort, CLI version, dataset/repository revision,
  index build identifier and usage metric version.
- `tool_policy`, `prompt_version` and `tool_description_version`, identifying
  the workflow prompt and server metadata used for the comparison.
- `tool_metric_version` identifies the excerpt/error accounting rules.

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

`--repeats` defaults to 1. For example, `--questions 11 16 18 --repeats 3`
performs 18 turns when both variants are selected. All rows go into the same
`usage.csv`, with a one-based `repetition` column. With multiple repetitions,
raw turn directories are grouped under `repeat-01/`, `repeat-02/`, etc.; the
single-repetition layout above remains unchanged. Pair order reverses across
both questions and repetitions. With three repetitions each question has a
2:1 split of which condition goes first, rather than perfect balance.

Before spending model tokens, verify a fresh corpus/index and make one actual
`find_symbols` call with `include_code=true` against the configured Neo4j.
`index_status` alone does not check backend connectivity. Completed tool items
are deduplicated by item ID before excerpt/error counting. Structured results
are preferred; JSON text results are a fallback and are not counted again.

The exact common prompt prefix is saved in `run.json`; each turn also keeps its
complete `prompt.txt`. Prompt templates and `PROMPT_VERSION` live in
`src/benchmark_agent.py`. `TOOL_DESCRIPTION_VERSION` lives with the shared
constants in `src/index_state.py` and is also exposed by `index_status`.
Increment the corresponding version when changing a template, server
instructions, tool docstrings or parameter descriptions. Historical artifacts
retain their original metadata; older runs without these fields are not
silently relabelled. The description version on a baseline row identifies the
paired setup, even though that condition has no Codeaware tools.

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

A subsequent [three-question guidance pilot](
../data/benchmark_results/2026-11-04-guidance/README.md) recorded all 12 turns
with the versioned descriptions and both prompt modes. Automatic runs made no
MCP calls; all three guided Codeaware runs did, with 10 successful calls. Token
usage increased relative to the guided native baseline in that small sample.

The [three-repetition excerpt benchmark](
../data/benchmark_results/2026-11-05-excerpts-guided/README.md) records 18
successful turns on questions 11, 16 and 18 with its recorded prompt/tool
versions and excerpt/error counters. Codeaware delivered 25 excerpts with
zero MCP errors but used more total tokens overall. Its checklist, per-pair
variation and limited quality review are preserved in that report.

The [subsequent import-tracking rerun](
../data/benchmark_results/2026-11-05-imports-guided/README.md) records another
18 successful turns with prompt v3 and imports-v5 metadata. Codeaware made
zero native command calls but used 15.46% more total tokens than its new
baseline. No agent selected `get_imports`; no overall answer-quality
improvement is established. The report includes all answers, a manual review,
source/excerpt checks and an audit of unknown tool arguments.

The [full low-reasoning run](../data/benchmark_results/2026-11-05-low-full-guided/README.md)
uses `--reasoning-effort low --repeats 1` with questions 1 through 18 in guided
mode. All 36 turns completed and reported usage. Codeaware used 17.85% more
total tokens and fewer tokens in 6/18 pairs, with no reported MCP errors.
A subsequent [quality review](../data/benchmark_results/2026-11-05-low-full-guided/quality-review.md)
grades all 36 answers, finding 15 ties and three baseline completeness
advantages under an unblinded retrospective rubric. This is not a like-for-like
aggregate comparison with the three-question repeated medium batch.
