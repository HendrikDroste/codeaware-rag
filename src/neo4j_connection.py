"""Shared Neo4j connection settings for indexing and graph queries."""

import os

from neo4j import GraphDatabase


def graph_connection():
    """
    Use the same environment settings for manual indexing and MCP
    queries.
    """
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError(
            "Set NEO4J_PASSWORD before starting the indexer or MCP server"
        )
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), password),
        connection_timeout=5,
    )
    return driver, os.environ.get("NEO4J_DATABASE", "neo4j")
