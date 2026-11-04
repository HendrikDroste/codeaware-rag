"""
Manually index one repository using the existing embedding and graph
functions.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.index_state import (
    COLLECTION,
    DEFAULT_CONFIG,
    GRAPH_SCHEMA,
    embedding_settings,
    file_hashes,
    source_files,
    write_manifest,
)


def index_repository(repository_root, index_dir, config=DEFAULT_CONFIG):
    import jedi

    from src.embeddings.build_graph import build_graph
    from src.embeddings.file_processor import embed_python_directory
    from src.neo4j_connection import graph_connection
    from src.pipelines.embedding_pipeline import EmbeddingPipeline

    repository_root = Path(repository_root).resolve(strict=True)
    index_dir, config = Path(index_dir).resolve(), Path(config).resolve(
        strict=True
    )
    if not repository_root.is_dir():
        raise ValueError("--repo must be a directory")
    files = source_files(repository_root)
    if not files:
        raise ValueError("Repository contains no Python files")
    hashes = file_hashes(repository_root, files)
    settings = embedding_settings(config)
    if (index_dir / "manifest.json").exists():
        old = json.loads(
            (index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if old["repository_root"] != str(repository_root):
            raise ValueError(
                "Index directory belongs to another repository; use a "
                "separate directory"
            )
    driver, _ = graph_connection()
    with driver:
        driver.verify_connectivity()
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "state": "building",
        "graph_schema": GRAPH_SCHEMA,
        "repository_root": str(repository_root),
        "build_id": str(uuid4()),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embeddings": settings,
        "chunking": {"size": 900, "overlap": 0},
        "file_hashes": hashes,
        "counts": {},
    }
    # Publish 'building' before replacing either backend, so failures
    # cannot be served.
    write_manifest(index_dir, manifest)
    pipeline = EmbeddingPipeline(
        config_path=str(config),
        collection_name=COLLECTION,
        chroma_persist_directory=str(index_dir / "chroma"),
        reset_collection=True,
        explicit_embeddings=True,
    )
    pipeline.collection.modify(metadata={"build_id": manifest["build_id"]})
    chunks = embed_python_directory(
        str(repository_root),
        pipeline,
        chunk_size=900,
        chunk_overlap=0,
        file_paths=files,
        repository_root=str(repository_root),
    )
    if not chunks:
        raise ValueError(
            "Repository contains no nonempty Python code to embed"
        )
    previous_cache = jedi.settings.cache_directory
    try:
        jedi.settings.cache_directory = str(index_dir / "jedi-cache")
        counts = build_graph(
            str(repository_root),
            file_paths=files,
            build_id=manifest["build_id"],
        )
    finally:
        jedi.settings.cache_directory = previous_cache
    if hashes != file_hashes(repository_root):
        raise ValueError(
            "Source files changed during indexing; rebuild from a stable "
            "checkout"
        )
    manifest.update(state="complete", counts={**counts, "chunks": chunks})
    write_manifest(index_dir, manifest)
    return manifest


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    args.repo = args.repo.resolve(strict=True)
    if not args.repo.is_dir():
        parser.error("--repo must be a directory")
    args.index_dir, args.config = (
        args.index_dir.resolve(),
        args.config.resolve(strict=True),
    )
    return args


if __name__ == "__main__":
    args = arguments(__doc__)
    logging.basicConfig(level=logging.INFO)
    try:
        result = index_repository(args.repo, args.index_dir, args.config)
        logging.info("Index complete: %s", result["counts"])
    except Exception:
        logging.exception(
            "Index build failed; incomplete indexes will not be served"
        )
        raise SystemExit(1)
