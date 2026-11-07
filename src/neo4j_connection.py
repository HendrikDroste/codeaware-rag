"""Share one Neo4j driver per process and close it on exit."""

import atexit
import os
from threading import Lock

from neo4j import GraphDatabase

_driver = None
_settings = None
_lock = Lock()


def close_connection():
    global _driver, _settings
    with _lock:
        if _driver is not None:
            _driver.close()
            _driver = None
            _settings = None


def graph_connection():
    global _driver, _settings
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Set NEO4J_PASSWORD before starting Codeaware")
    settings = (
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        password,
    )
    with _lock:
        if _driver is None or settings != _settings:
            if _driver is not None:
                _driver.close()
            _driver = GraphDatabase.driver(
                settings[0], auth=settings[1:], connection_timeout=5
            )
            _settings = settings
    return _driver, os.environ.get("NEO4J_DATABASE", "neo4j")


atexit.register(close_connection)
