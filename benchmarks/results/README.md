# Retained benchmark evidence

The [7 November Neo4j-only rerun](2026-11-07-neo4j-low-full-guided/README.md)
measures the current configuration. It contains 18 questions, one observation
per condition, with low reasoning and guided tools. Codeaware used 6.71% fewer
reported tokens. Its unblinded review found 17 ties and one baseline completeness
advantage. Cached input accounts for the token reduction; uncached input plus
output rose 2.47%. This does not establish monetary savings or business ROI.

The [historical 18-question run](2026-11-05-low-full-guided/README.md) provides
the earlier comparison. Its 17 evidence files are retained byte for
byte, including original answers, usage, prompts, source provenance and the
retrospective, unblinded quality review. Directory moves do not change the
recorded run identity. Absolute paths and old source paths inside the records
refer to their original execution environment.

The historical run predates the Neo4j-only migration and later tool simplification.
Host, prompt and tool metadata differ between runs. Compare the two conditions
within each run; the change between runs cannot isolate an individual cause.
Neither comparison isolates the graph's contribution.

Six exploratory pilot folders, their 88 files, and the earlier experiment
implementation are preserved in the [archive branch](https://github.com/HendrikDroste/codeaware-rag/tree/codex/archive-before-presentation-cleanup-2026-11-07).
Their original paths on that branch are under `data/benchmark_results/`.
No pilot measurements were combined with the full-run results.

Historical raw host traces remain in the ignored local `data/agent_runs/`
directory at the recorded paths. Current traces are in the ignored
`benchmarks/runs/2026-11-07-neo4j-low-full-guided/` directory, with their hashes
retained alongside the new report. New runs belong in fresh `benchmarks/runs/`
directories. Markdown files explain the findings; CSV and JSON retain raw
measurements and machine-readable provenance.
