# Retained benchmark evidence

The [full 18-question run](2026-11-05-low-full-guided/README.md) is the reference
experiment for the presentation. Its 17 evidence files are retained byte for
byte, including original answers, usage, prompts, source provenance and the
retrospective, unblinded quality review. Directory moves do not change the
recorded run identity. Absolute paths and old source paths inside the records
refer to their original execution environment.

This run predates the Neo4j-only migration and later tool simplification. It
does not measure the current configuration or isolate the graph's contribution.

Six exploratory pilot folders, their 88 files, and the earlier experiment
implementation are preserved in the [archive branch](https://github.com/HendrikDroste/codeaware-rag/tree/codex/archive-before-presentation-cleanup-2026-11-07).
Their original paths on that branch are under `data/benchmark_results/`.
No pilot measurements were combined with the full-run results.

Raw host traces remain in the ignored local `data/agent_runs/` directory at
the recorded paths. New runs belong in fresh `benchmarks/runs/` directories.
