# Why token usage increased, and whether answers improved

This review examines the three guided question pairs already recorded in the
pilot. It uses saved events and answers, checked against the pinned local Flask
source. No additional model calls were made. The quality assessment is a
retrospective manual review by the implementing agent, not a blinded or
independent evaluation.

## What increased

The Codeaware condition reported 35,944 more total tokens than the guided
native baseline. Of that difference, 35,923 were input tokens and only 21 were
output tokens: 99.94% of the increase was input. Output includes the host's
reported generation, not only the final visible answer, so these figures are
not a final-answer word count.

| Question | Native total | Codeaware total | Change | Tool operations: native / Codeaware |
|---|---:|---:|---:|---:|
| 11: Blueprint inheritance | 62,989 | 90,810 | +44.2% | 3 / 7 |
| 16: Route registration | 75,746 | 96,581 | +27.5% | 5 / 8 |
| 18: Request creation | 75,244 | 62,532 | -16.9% | 4 / 4 |

An operation here is one completed command-execution or MCP item. A command can
contain several shell reads, and several tool calls can be batched. These counts
are not counts of model requests.

The strongest trace-supported explanation is additional inspection on top of
native source reading. In Q11, four symbol/inheritance calls were added while
the agent still made three native command calls, the same native-call count as
the baseline. In Q16, four MCP calls replaced only one native command call.
The agent still read the route decorator, registration implementation, dispatch
and blueprint code. More context-bearing interaction steps are consistent with
the higher accumulated input, even where individual tool replies were small.

For Q11 and Q16, a simple measure of tool-result text recorded once per
completed operation was actually smaller with Codeaware: native shell output plus one
structured MCP rendering totalled about 19.7k versus 24.7k characters for Q11,
and 21.6k versus 23.5k for Q16. These are sums of result sizes, not deduplicated
source content. They are not token counts or reconstructed model inputs.
Larger individual replies alone therefore do not explain those increases.

Q18 was different. Symbol lookup and a callee query replaced two native command
calls. The agent obtained the request-context and dispatch functions from the
graph, then made a focused set of source reads. That run used fewer tokens and
was faster, despite a larger single graph reply. This is an encouraging
observation for that question, not a repeatable savings estimate.

The saved [Codex JSONL format](https://learn.chatgpt.com/docs/non-interactive-mode)
reports usage in `turn.completed`. Our traces do not include the complete
model-visible requests or a token breakdown per request/tool. We cannot
precisely assign the extra tokens to schema text, repeated conversation context,
tool serialization or individual queries. Additional tool metadata is a
possible contributor, not a measured explanation for a fixed number of tokens.

Caching does not remove the finding: cached input increased by 23,808 tokens,
but uncached input also increased by 12,115, from 48,271 to 60,386 (+25.1%).
These are usage counts, not a currency-cost calculation.

## Are the answers better?

The relevant criteria were core correctness, coverage of the question, useful
additional detail, source support and appropriate uncertainty. On these three
pairs, there is no clear overall quality improvement.

**Q11, inheritance:** Both answers correctly identify
`object -> Scaffold -> sansio.Blueprint -> public Blueprint`, explain the base
alias and public export, and mention the test-only `MyBlueprint` subclass.
Codeaware puts that subclass into the diagram, which is easier to scan. The
baseline also distinguishes nested-blueprint registration from inheritance;
Codeaware instead clarifies that `BlueprintSetupState` is a separate helper.
These are useful presentation/coverage differences, not a new core finding
available only with MCP. See the declarations at
[public Blueprint](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/blueprints.py#L10-L18),
[sans-I/O Blueprint](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/blueprints.py#L119-L133),
[Scaffold](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L52-L54),
and the [test subclass](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/tests/test_blueprints.py#L221-L224).

**Q16, routing:** Both explain decorator equivalence, returning the same view
function, endpoint defaults, HTTP methods, automatic OPTIONS, the URL map and
view-function mapping, endpoint collisions, deferred blueprint registration,
and request dispatch. Codeaware gives automatic OPTIONS its own step and adds
the setup guard after the first request. The baseline additionally explains
registering an endpoint without a view and binding the view later. Neither
omission prevents answering the question. The core explanations agree with
[route](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L344-L373),
[registration](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L606-L662),
the [setup guard](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/app.py#L411-L421),
and [late view binding](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/sansio/scaffold.py#L443-L465).

One graph query selected `Scaffold.route` and returned no callees. The actual
`self.add_url_rule(...)` call belongs to its nested `decorator` function, a
separate symbol. An empty outer-function result was therefore not enough to
explain the decorator's behavior; source inspection supplied the answer.

**Q18, request creation:** Both correctly trace WSGI entry through
`wsgi_app -> request_context -> AppContext.from_environ`, identify
`app.request_class(environ)` as the actual construction, explain the JSON
provider and replaceable request class, and describe context pushing and the
request proxy. Codeaware is shorter. The baseline adds the ContextVar mechanism
and the current merger of request and app contexts. Those details are supported
by the source but are not required for the core construction explanation. See
[construction](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L340-L348),
[ContextVar binding](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L428-L434),
and the [merge documentation](https://github.com/pallets/flask/blob/49943b5a94413a530761350f04e896f76e8d620e/src/flask/ctx.py#L285-L288).

The prior link audit verified 126 source-link destinations and line bounds;
that alone did not establish that answers were correct or better. This review
checked the core claims above against source. It does not award an independent
numeric quality score or claim that every sentence was adjudicated.

## What to improve next

First test whether Codeaware can replace native inspection instead of adding
another inspection sequence. Encourage the agent to stop relationship queries
when it has sufficient evidence, and read only the needed source ranges. The
current prompt already asks for relevant queries, but the traces show remaining
redundancy; stronger wording still needs measurement.

Compact graph replies are a secondary option. `query_calls` returns complete
callee/caller function bodies. Q18's callee reply included five bodies,
including exception handling and context push/pop, totalling about 9.3k
characters with metadata. Some of that was outside the narrow construction
question and some was read again from source. Returning locations first and
making content optional could reduce duplication, but that should be tested:
Q18 was already the token-saving case, while the expensive Q11/Q16 replies were
much smaller. Large graph replies are not established as the main cause here.

No `search_code` calls occurred in this pilot. These results do not evaluate
semantic retrieval quality and do not justify changing embedding models.
Broader tasks on unfamiliar code and repeated, independently graded runs are
needed before claiming a quality or business-value advantage.

[usage-analysis.json](usage-analysis.json) preserves the exact per-question
counts, operation counts and result-size measurements used here. Raw traces,
answers, benchmark settings and product code remain unchanged; the README's
Q18 call sequence was corrected to `find_symbols` followed by `get_callees`.
