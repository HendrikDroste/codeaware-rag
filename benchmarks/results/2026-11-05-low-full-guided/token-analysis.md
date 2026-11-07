# Why Codeaware used more tokens

The latest full low-reasoning run added 174,810 reported tokens (+17.85%).
The increase comes from input: Codeaware used 175,496 more input tokens and
686 fewer output tokens. The strongest trace evidence is inefficient discovery
through additional context-tool operations, especially on questions 17 and 18.
The traces do not expose full model requests or per-request usage, so an exact
causal allocation among tool schemas, history and result text is unavailable.

## Measured decomposition

| Reported measurement | Native | Codeaware | Difference |
|---|---:|---:|---:|
| Input tokens | 969,246 | 1,144,742 | +175,496 |
| Output tokens | 9,948 | 9,262 | -686 |
| Total tokens | 979,194 | 1,154,004 | +174,810 |
| Cached input, already inside input | 754,048 | 880,896 | +126,848 |
| Uncached input | 215,198 | 263,846 | +48,648 |
| Native commands | 41 | 3 | -38 |
| MCP operations | 0 | 53 | +53 |
| Total tool operations | 41 | 56 | +15 |

Uncached input rose 22.61%. Cached input is counted once, not added again to
input. The separate reported reasoning-output field totals 44 versus 82;
it is a subset of output and is not added again. That small field difference
cannot explain an increase of 174,810 total tokens. These are not cost figures.

## Concrete sources of extra work

| Question | Native operations | Codeaware operations | Extra tokens |
|---|---:|---:|---:|
| 18: request creation | 3 | 8 MCP + 1 native | 50,301 |
| 17: default configuration | 2 | 6 MCP | 34,198 |
| 15: response conversion | 2 | 3 MCP | 25,187 |
| 13: Scaffold usage | 5 | 8 MCP | 18,838 |

Q17 and Q18 contribute 84,499 tokens, or 48.34% of the net excess. This is a
sum of observed paired differences, not a causal attribution to particular
calls. Other questions offset some positive differences with savings.

### Q17: the tool cannot look up the requested attribute

The agent starts with `find_symbols(name="App", max_lines=100)` and receives
lines 60–159 of `sansio/app.py`: 95 of these 100 lines belong to docstrings.
It then searches semantically and looks up `App.make_config` and `App.__init__`.
It also tries `Flask.default_config`, which returns an empty result.

That last lookup is a capability mismatch, not a missing name in Flask:
`find_symbols` indexes functions, methods and classes, not assignments or
attributes. The defaults exist at `src/flask/app.py:205–235`. A final semantic
search does not lead the answer to state that concrete definition location.
The answer still has the completeness omission recorded in the quality review.

The native baseline finds `default_config` and `make_config` together with one
`rg` search, then reads the relevant regions of both files in one command.

### Q18: outdated name guesses plus an unhelpful class prefix

Two empty exact-name lookups ask for `RequestContext.__init__` and
`RequestContext`. This revision instead constructs the request through
`AppContext.from_environ`. The agent recovers using graph and symbol queries.

Later, `find_symbols(name="Flask", max_lines=100)` returns lines 108–207 of
`app.py`, again with 95 docstring lines. It does not reach `request_class` at
line 239, so the agent performs another semantic search for the default class.
The native baseline gathers symbol matches and reads several targeted regions
across four files in a single final shell command.

The Codeaware answer remains incomplete about the request proxy despite the
extra calls. This is more discovery work, not better coverage on the rubric.

### Q15 and Q13: additional traversal after discovery

Q15 uses semantic search, a 100-line `Flask.make_response` excerpt and then
`get_callers`. The baseline uses two commands, bundling multiple source regions.
Q13 combines symbol lookup, reverse imports, three inheritance queries,
semantic search and two further symbol lookups. These calls are relevant, but
relevance alone does not make them token-efficient. Their answers do not score
higher on the existing rubric.

## What the current evidence does not support

**Native verification rereads are not the main issue in this batch.** Only
three native commands occur in Codeaware turns: Q7 and Q18 list files; Q14
reads/searches attributes not exposed by the selected class prefix. The previous
explanation of graph queries followed by native rereads does not account for
the current run's overhead.

**Larger returned text alone does not explain the aggregate.** Counting each
logical MCP payload once gives 135,568 serialized JSON characters, plus 5,993
native-output characters: 141,561 versus the baseline's 170,271 native-output
characters. Codeaware returns roughly 16.9% fewer characters under this proxy,
yet reports more input tokens. Formats and tokenization differ; this is not a
measurement of actual model input serialization.

The source-location audit also finds only 56 repeated line locations across
calls out of 2,517 delivered locations (deduplicated inside each call). Repeated
returned source is therefore not the dominant measured volume. This audit
cannot measure history being included again in later model requests.

**Tool-operation count is not model-request count.** Calls can be batched or
parallel; a shell command can bundle searches and reads. Questions 3, 5 and 10
have equal operation counts but still use more tokens with Codeaware. Additional
MCP definitions/instructions, different request grouping, serialization and
history growth remain plausible contributors, without exact percentages.
The JSONL traces do not expose complete prompts for each model request.

**Do not assume the host passes both result representations to the model.**
Raw MCP events contain structured and text forms. The analysis counts the
logical payload once; it cannot establish whether host serialization duplicates
those representations. There is no basis here to blame double encoding.

## What works, and the smallest useful next changes

Q1, Q2, Q4, Q6 and Q8 each finish with one MCP call and no native commands;
all five save tokens. Q7 also saves tokens with one MCP call and one file-list
command. All six tie the baseline on the current quality rubric. These are
single observations, not proof of a stable performance advantage.

The next changes should reduce discovery steps and expose the facts that are
currently absent from the tool interface:

1. Support assignment/attribute lookup, or explicitly direct the agent to a
   targeted native text search after the first unsupported lookup. This
   directly addresses `Flask.default_config` and `request_class`.
2. Return a compact class/member summary or allow selecting a source region;
   a class's first 100 lines can be almost entirely documentation.
3. After an empty symbol lookup, switch to discovery once rather than trying
   variations of the same unverified name. Q18's two RequestContext guesses
   illustrate the avoidable sequence.
4. Batch already-known symbol lookups when practical, while preserving the
   existing per-result bounds. Avoid requesting graph neighbors solely to
   reconfirm an implementation fact already supported by returned code.

These are recommendations, not implemented changes or measured savings.
Smaller excerpts alone would not fix missing attributes or wrong initial names.
Do not weaken evidence requirements merely to reduce call counts.

## Analysis artifact

[token-cause-analysis.json](token-cause-analysis.json) contains every run's
reported usage, operation list, logical response sizes, empty symbol results,
class-excerpt/docstring counts and source-location overlap, plus raw-trace
hashes. The analysis used existing local traces only, without new agent runs,
model calls, Neo4j startup or changes to the original measurements.
