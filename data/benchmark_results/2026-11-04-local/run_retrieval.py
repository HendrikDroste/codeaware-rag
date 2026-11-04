"""Reproduce the local pinned-Flask retrieval measurement."""

import json
import time
from pathlib import Path

from src.index_state import COLLECTION, DEFAULT_CONFIG, read_manifest
from src.pipelines.embedding_pipeline import EmbeddingPipeline
from src.retrievers.validate_retriever import (
    load_benchmark,
    save_model_results,
    save_results_to_csv,
    validate_retriever,
)

PROJECT = Path(__file__).resolve().parents[3]
REPOSITORY = Path("/Users/hendrikdroste/PycharmProjects/flask")
INDEX = PROJECT / ".codeaware-index/flask"
OUTPUT = Path(__file__).resolve().parent

started = time.monotonic()
data = load_benchmark(PROJECT / "data/benchmarks/flask", REPOSITORY)
manifest = read_manifest(REPOSITORY, INDEX, DEFAULT_CONFIG)
pipeline = EmbeddingPipeline(
    config_path=str(DEFAULT_CONFIG),
    collection_name=COLLECTION,
    chroma_persist_directory=str(INDEX / "chroma"),
    explicit_embeddings=True,
    existing_only=True,
)
if pipeline.collection.metadata.get("build_id") != manifest["build_id"]:
    raise ValueError("Collection and manifest belong to different builds")
results = validate_retriever(pipeline, data)
save_results_to_csv(results, str(OUTPUT / "retrieval.csv"))
save_model_results(
    results, manifest["embeddings"]["name"], str(OUTPUT / "models.csv")
)
summary = {
    **data.attrs["benchmark"],
    "embedding_settings": manifest["embeddings"],
    "chunking": manifest["chunking"],
    "index_build_id": manifest["build_id"],
    "index_counts": manifest["counts"],
    "questions": len(data),
    "results_per_question": 5,
    "elapsed_seconds_including_model_load": time.monotonic() - started,
    "means": {
        metric: sum(results[metric]) / len(results[metric])
        for metric in ("mrr", "line_coverage", "file_match_ratio")
    },
    "per_question": [
        {
            "question_id": number + 1,
            "question": question,
            **{
                metric: results[metric][number]
                for metric in ("mrr", "line_coverage", "file_match_ratio")
            },
        }
        for number, question in enumerate(results["question"])
    ],
}
(OUTPUT / "retrieval-summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary["means"], indent=2))
