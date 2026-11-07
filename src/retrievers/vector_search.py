"""Store chunks and search vectors in the same Neo4j graph."""

import hashlib
import math
from pathlib import Path

from src.neo4j_connection import graph_connection
from src.retrievers.query_graph import check_graph


def vector_names(repository_root):
    # Identifiers cannot be query parameters in Neo4j 5. The generated
    # names contain only a fixed prefix and hexadecimal characters.
    suffix = hashlib.sha256(str(repository_root).encode()).hexdigest()[:24]
    return f"codeaware_vector_{suffix}", f"CodeawareChunk_{suffix}"


def validate_vectors(vectors, dimensions=None):
    if not vectors:
        raise ValueError("The embedding model returned no vectors")
    dimensions = dimensions or len(vectors[0])
    if not 1 <= dimensions <= 4096:
        raise ValueError("Neo4j vectors must have 1..4096 dimensions")
    if any(
        len(vector) != dimensions
        or any(not math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise ValueError("Embedding dimensions differ or values are nonfinite")
    return dimensions


def store_chunks(chunks, model, *, repository_root, build_id):
    """Write embedded chunks, then wait for the vector index."""
    if not chunks:
        raise ValueError("Repository contains no nonempty Python code")
    index_name, label = vector_names(repository_root)
    driver, database = graph_connection()
    dimensions = None
    with driver.session(database=database) as session:
        check_graph(session, repository_root, build_id)
        session.run(f"DROP INDEX {index_name} IF EXISTS").consume()
        for offset in range(0, len(chunks), 100):
            batch = chunks[offset : offset + 100]
            vectors = model.embed_documents([d.page_content for d in batch])
            if len(vectors) != len(batch):
                raise ValueError(
                    "The embedding model returned too few vectors"
                )
            dimensions = validate_vectors(vectors, dimensions)
            rows = [
                {
                    "id": offset + number,
                    "path": doc.metadata["source"],
                    "file": str(
                        Path(repository_root) / doc.metadata["source"]
                    ),
                    "line_start": doc.metadata["line_start"],
                    "line_end": doc.metadata["line_end"],
                    "content": doc.page_content,
                    "embedding": list(vector),
                }
                for number, (doc, vector) in enumerate(zip(batch, vectors))
            ]
            session.run(
                "UNWIND $rows AS row "
                f"CREATE (c:Codeaware:Chunk:{label}) "
                "SET c = row, c.repository = $repo, c.build_id = $build "
                "WITH c, row "
                "MATCH (f:Codeaware:File {repository:$repo, path:row.file}) "
                "CREATE (c)-[:DEFINED_IN]->(f)",
                rows=rows,
                repo=repository_root,
                build=build_id,
            ).consume()
        session.run(
            f"CREATE VECTOR INDEX {index_name} "
            f"FOR (c:{label}) ON c.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dimensions, "
            "`vector.similarity_function`: 'euclidean', "
            "`vector.quantization.enabled`: false}}",
            dimensions=dimensions,
        ).consume()
        session.run(
            "CALL db.awaitIndex($name, 120)", name=index_name
        ).consume()
        session.run(
            "MATCH (i:CodeawareIndex {repository:$repo, build_id:$build}) "
            "SET i.vector_dimensions = $dimensions, i.chunks = $count",
            repo=repository_root,
            build=build_id,
            dimensions=dimensions,
            count=len(chunks),
        ).consume()
    return dimensions


def search_chunks(vector, *, repository_root, build_id, limit):
    """Search only this repository's index; distance is squared L2."""
    dimensions = validate_vectors([vector])
    index_name, _ = vector_names(repository_root)
    driver, database = graph_connection()
    with driver.session(database=database) as session:
        index = check_graph(session, repository_root, build_id)
        if not index.get("chunks"):
            raise ValueError("Vector index is missing or empty; rebuild it")
        if index.get("vector_dimensions") != dimensions:
            raise ValueError("Query embedding dimensions differ; rebuild it")
        status = session.run(
            "SHOW VECTOR INDEXES YIELD name, state "
            "WHERE name = $name RETURN state",
            name=index_name,
        ).single()
        if status is None or status["state"] != "ONLINE":
            raise ValueError("Vector index is missing or offline; rebuild it")
        result = session.run(
            "CALL db.index.vector.queryNodes($index, $limit, $vector) "
            "YIELD node, score "
            "WHERE node.repository = $repo AND node.build_id = $build "
            "RETURN node.path AS path, node.line_start AS line_start, "
            "node.line_end AS line_end, node.content AS content, "
            "reduce(d = 0.0, i IN range(0, size($vector) - 1) | "
            "d + (node.embedding[i] - $vector[i]) ^ 2) AS distance "
            "ORDER BY distance, path, line_start",
            index=index_name,
            limit=min(limit, index["chunks"]),
            vector=list(vector),
            repo=repository_root,
            build=build_id,
        )
        return [dict(row) for row in result]
