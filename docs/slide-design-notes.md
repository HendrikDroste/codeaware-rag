# Slide design notes

## Problem slide

Use one goal and one short explanation of the approach. Keep the three original
Flask screenshots as the concrete example. Present potential applications aloud
rather than adding another paragraph to the left column.

Suggested left-side copy:

> Help AI agents understand code and its dependencies
>
> Find relevant code by meaning.
> Follow its explicit relationships.

Keep a compact ownership disclosure: the original HPI project was built
independently, and the later tool-server extension used AI assistance.

Two other layouts would also work:

- **Concrete question:** one question beside the screenshots, such as “Where does
  Blueprint get its shared behavior?” Explain the broader project goal aloud.
- **Example takeaway:** “Three files explain one class hierarchy.” Use the
  screenshots to introduce semantic discovery and relationship navigation.

The goal-and-approach layout keeps the broad project scope visible while leaving
space for the example.

## Alternative titles

| Slide | Recommended title | Alternatives |
|---|---|---|
| Problem and goal | Code context spans multiple files | Connecting code context for AI agents; Code relationships add context |
| Architecture | The agent queries one Neo4j code index | How Codeaware supplies code context; Semantic search and graph navigation in one index |
| Evaluation | Evaluation and lessons learned | Benchmark findings and next steps; What the benchmark supports |

Keep the evaluation title neutral. Do not turn one observation into a broad
performance or business-value claim. Keep subtitle spacing consistent across the
three content slides, and preserve the screenshots and automatic date fields.

## Results chart

Use an editable column chart for the two token totals, rather than a table.
Both bars start at zero. Native tools are gray and Native + Codeaware is teal.
Label the bars with the exact totals: 1,027,206 and 958,262 reported tokens.
The headline states the observed 6.7% reduction. Keep the answer-quality result
and the cache qualification visible. Cached input explains the aggregate
reduction; uncached input plus output rose 2.5%. This single trial does not
establish monetary savings.
