"""
Small fixtures exercise the real existing pipeline; no model downloads
in unit tests.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from src import index_repository as indexing
from src import index_state, neo4j_connection
from src.indexing import build_graph as graph
from src.indexing import model as embedding_model
from src.indexing.file_processor import (
    get_all_python_files,
    load_and_split_python_files,
)
from src.mcp_server import create_server
from src.retrievers import vector_search


class Result:
    def data(self):
        return []

    def consume(self):
        return None

    def single(self):
        return {"i": {}}


class RecordingDriver:
    """
    Capture graph writes; real Cypher is checked separately against
    Neo4j.
    """

    def __init__(self):
        self.writes = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def verify_connectivity(self):
        pass

    def session(self, **kwargs):
        return self

    def run(self, query, **parameters):
        self.writes.append((query, parameters))
        return Result()

    def close(self):
        self.closed = True


class Embeddings:
    def embed_query(self, text):
        return [
            float(text.lower().count(word) + 1)
            for word in ("first", "second", "return", "def")
        ]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    shutil.copytree(Path(__file__).parent / "fixtures/repository", root)
    monkeypatch.setattr(
        graph.jedi.settings, "cache_directory", str(tmp_path / "jedi-cache")
    )
    return root


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "models:\n  embeddings:\n    name: fixture\n    vendor: fixture\n    "
        "type: fixture\n    num_documents: 5\n"
    )
    return path


@pytest.fixture
def build(repository, config, tmp_path, monkeypatch):
    driver = RecordingDriver()
    from src.retrievers import query_graph

    monkeypatch.setattr(
        neo4j_connection, "graph_connection", lambda: (driver, "neo4j")
    )
    monkeypatch.setattr(graph, "graph_connection", lambda: (driver, "neo4j"))
    monkeypatch.setattr(
        query_graph, "graph_connection", lambda: (driver, "neo4j")
    )
    monkeypatch.setattr(
        embedding_model, "get_embedding_model", lambda config: Embeddings()
    )
    monkeypatch.setattr(
        vector_search, "graph_connection", lambda: (driver, "neo4j")
    )

    def search(vector, *, repository_root, build_id, limit):
        rows = [
            {
                key: row[key]
                for key in ("path", "line_start", "line_end", "content")
            }
            | {"distance": 0.0}
            for query, parameters in driver.writes
            if "CREATE (c:Codeaware:Chunk:" in query
            for row in parameters["rows"]
        ]
        return rows[:limit]

    monkeypatch.setattr(vector_search, "search_chunks", search)
    directory = tmp_path / "index"
    manifest = indexing.index_repository(repository, directory, config)
    return repository, directory, config, manifest, driver


def call(server, tool, **kwargs):
    async def run():
        async with Client(server) as client:
            result = await client.call_tool(tool, kwargs)
            return result

    return asyncio.run(run())


def data(result):
    assert not result.is_error, result.content
    return result.structured_content


def test_discovery_preserves_defaults_and_respects_repository_boundaries(
    repository, tmp_path
):
    (repository / ".venv").mkdir()
    (repository / ".venv/hidden.py").write_text("pass")
    outside = tmp_path / "outside.py"
    outside.write_text("pass")
    (repository / "link.py").symlink_to(outside)
    (repository / "linked").symlink_to(tmp_path, target_is_directory=True)
    (repository / "custom_environment").mkdir()
    (repository / "custom_environment/pyvenv.cfg").touch()
    (repository / "custom_environment/hidden.py").write_text("pass")
    files = index_state.source_files(repository)
    assert {Path(p).name for p in files} == {
        "__init__.py",
        "cli.py",
        "helpers.py",
        "worker.py",
    }
    assert str(repository / "link.py") not in get_all_python_files(
        str(repository)
    )


def test_splitter_reuses_baseline_and_correct_locations(repository):
    paths = index_state.source_files(repository)
    chunks = load_and_split_python_files(
        paths, 900, 0, repository_root=str(repository)
    )
    assert {d.metadata["source"] for d in chunks} == {
        "__init__.py",
        "cli.py",
        "helpers.py",
        "worker.py",
    }
    for chunk in chunks:
        lines = (
            (repository / chunk.metadata["source"]).read_text().splitlines()
        )
        assert chunk.page_content == "\n".join(
            lines[
                chunk.metadata["line_start"] - 1 : chunk.metadata["line_end"]
            ]
        )


def test_graph_keeps_two_passes_and_resolves_identity(repository, monkeypatch):
    driver = RecordingDriver()
    monkeypatch.setattr(graph, "graph_connection", lambda: (driver, "neo4j"))
    counts = graph.build_graph(
        str(repository),
        file_paths=index_state.source_files(repository),
        build_id="test",
    )
    definition_queries = [q for q, _ in driver.writes if "MERGE (func:" in q]
    assert any(
        "MERGE (func:Codeaware:Function " in q for q in definition_queries
    )
    assert any("MERGE (func:Codeaware:Class " in q for q in definition_queries)
    assert all(":Symbol" not in q for q, _ in driver.writes)
    assert all("symbol_name_constraint" not in q for q, _ in driver.writes)
    symbols = [p for q, p in driver.writes if "MERGE (func:" in q]
    calls = [p for q, p in driver.writes if "MERGE (caller)-" in q]
    ids = {p["qualified"]: p["id"] for p in symbols}
    assert ids["First.run"] != ids["Second.run"]
    assert "outer.inner" in ids
    edges = {(p["c1"], p["c2"]) for p in calls}
    for a, b in [
        ("entry", "first"),
        ("methods", "First.run"),
        ("methods", "Second.run"),
        ("outer", "outer.inner"),
        ("outer.inner", "first"),
    ]:
        assert (ids[a], ids[b]) in edges
    assert not any(a in {ids["ambiguous"], ids["dynamic"]} for a, _ in edges)
    unicode_call = next(p for p in calls if p["c1"] == ids["entry"])
    assert unicode_call["col"] == (
        repository / "worker.py"
    ).read_text().splitlines()[4].index("renamed")
    assert counts["ambiguous_calls"] >= 1
    assert counts["unresolved_calls"] >= 1
    assert counts["symbols"] == len(symbols)
    classes = [p for p in symbols if p["kind"] == "class"]
    assert counts["classes"] == len(classes)
    inheritance = [p for q, p in driver.writes if "INHERITS_FROM" in q]
    class_ids = {(p["path"], p["qualified"]): p["id"] for p in classes}
    child_id = class_ids["worker.py", "Child"]
    assert child_id != class_ids["helpers.py", "Child"]
    assert {(p["child"], p["parent"], p["position"]) for p in inheritance} == {
        (child_id, class_ids["helpers.py", "First"], 0),
        (child_id, class_ids["helpers.py", "Second"], 1),
        (class_ids["worker.py", "Grandchild"], child_id, 0),
        (
            class_ids["worker.py", "ÄChild"],
            class_ids["helpers.py", "First"],
            0,
        ),
    }
    assert counts["inheritance_edges"] == 4 and counts["unresolved_bases"] == 4
    assert driver.closed
    deletion = next((q, p) for q, p in driver.writes if "DETACH DELETE" in q)
    assert ":Codeaware" in deletion[0] and deletion[1]["repo"] == str(
        repository
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_graph_preserves_definition_content_and_unicode_positions(
    tmp_path, monkeypatch, newline
):
    root = tmp_path / "source"
    root.mkdir()
    code = (
        'def café():\n    return "ä"\n\n'
        'def entry():\n    label = "ä"; return café()\n'
    )
    path = root / "example.py"
    path.write_bytes(code.replace("\n", newline).encode("utf8"))
    driver = RecordingDriver()
    monkeypatch.setattr(graph, "graph_connection", lambda: (driver, "neo4j"))
    counts = graph.build_graph(root, file_paths=[str(path)])
    definitions = [p for q, p in driver.writes if "MERGE (func:" in q]
    assert [
        (p["id"], p["sln"], p["eln"], p["content"]) for p in definitions
    ] == [
        ("example.py:1:café", 1, 2, 'def café():\n    return "ä"\n'),
        ("example.py:4:entry", 4, 5, code[code.index("def entry") :]),
    ]
    call = next(p for q, p in driver.writes if "MERGE (caller)-" in q)
    assert (call["c1"], call["c2"], call["ln"], call["col"], call["ct"]) == (
        "example.py:4:entry",
        "example.py:1:café",
        5,
        code.splitlines()[4].index("café"),
        "function",
    )
    line = code.splitlines()[4]
    column = line.index("café") + len("café")
    assert graph._jedi_position(
        code.splitlines(), graph.Point(4, len(line[:column].encode("utf8")))
    ) == (5, column)
    assert counts["symbols"] == 2 and counts["calls"] == 1
    assert driver.closed


@pytest.mark.parametrize("failure", ["syntax", "neo4j"])
def test_graph_closes_session_and_propagates_failure(
    repository, monkeypatch, failure
):
    driver = RecordingDriver()
    monkeypatch.setattr(graph, "graph_connection", lambda: (driver, "neo4j"))
    if failure == "syntax":
        (repository / "broken.py").write_text("def broken(:\n")
        error_type, message = ValueError, "Cannot index Python syntax errors"
    else:
        error = RuntimeError("graph write failed")
        original_run = driver.run

        def fail_write(query, **parameters):
            if "MERGE (func:" in query:
                raise error
            return original_run(query, **parameters)

        monkeypatch.setattr(driver, "run", fail_write)
        error_type, message = RuntimeError, "graph write failed"
    with pytest.raises(error_type, match=message) as raised:
        graph.build_graph(repository)
    if failure == "neo4j":
        assert raised.value is error
    assert driver.closed


def add_import_fixture(repository):
    package = repository / "package"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from .items import action as exported\n"
    )
    (package / "items.py").write_text("def action():\n    pass\nVALUE = 1\n")
    (package / "consumer.py").write_text(
        "from .items import action as renamed, VALUE\n"
        "from . import items as local\n"
        "from package import exported\n"
        "import package.items as module, helpers\n"
        "def local_import():\n"
        "    label = 'ä'; from .items import action as nested\n"
        "if False:\n"
        "    from .items import action as conditional\n"
        "from package.items import *\n"
        "import sys\n"
        "from definitely_missing_codeaware_module import missing\n"
        "import package as pkg\n"
    )


@pytest.mark.parametrize("source_directory", ["", "src"])
def test_imports_follow_jedi_bindings_and_keep_sites(
    repository, monkeypatch, source_directory
):
    source_root = repository / source_directory
    source_root.mkdir(exist_ok=True)
    add_import_fixture(source_root)
    driver = RecordingDriver()
    monkeypatch.setattr(graph, "graph_connection", lambda: (driver, "neo4j"))
    counts = graph.build_graph(
        str(repository), file_paths=index_state.source_files(repository)
    )
    imports = [p for q, p in driver.writes if "[r:IMPORTS" in q]
    consumer = [
        p for p in imports if p["source"].endswith("package/consumer.py")
    ]
    assert len(consumer) == 9
    assert {p["name"] for p in consumer} == {
        "action",
        "VALUE",
        "items",
        "exported",
        "package.items",
        "helpers",
        "package",
    }
    assert {p["alias"] for p in consumer} == {
        "renamed",
        "local",
        "module",
        "nested",
        "conditional",
        "pkg",
        None,
    }
    for row in consumer:
        target = source_root / "package/items.py"
        if row["name"] == "helpers":
            target = repository / "helpers.py"
        elif row["name"] == "package":
            target = source_root / "package/__init__.py"
        assert row["target"] == str(target)
        source = Path(row["source"]).read_text().splitlines()[row["line"] - 1]
        assert source[row["column"] :].startswith(row["name"])
    nested = next(p for p in consumer if p["alias"] == "nested")
    assert nested["line"] == 6 and nested["module"] == ".items"
    assert counts["import_edges"] == len(imports)
    assert counts["external_imports"] == 1
    assert counts["unresolved_imports"] == 1
    assert counts["wildcard_imports"] == 1
    assert counts["imports"] == sum(
        counts[key]
        for key in (
            "import_edges",
            "external_imports",
            "unresolved_imports",
            "ambiguous_imports",
            "wildcard_imports",
        )
    )


def test_imports_omit_ambiguous_and_unindexed_targets(repository, monkeypatch):
    # Only the two listed files may become targets, even when Jedi can
    # resolve other Python files in the repository.
    (repository / "consumer.py").write_text(
        "from helpers import first\nimport worker\nimport excluded\n"
    )
    (repository / "excluded.py").write_text("pass\n")
    original = graph.jedi.Script.goto

    def ambiguous(script, *args, **kwargs):
        definitions = original(script, *args, **kwargs)
        return definitions * 2 if kwargs["line"] == 1 else definitions

    monkeypatch.setattr(graph.jedi.Script, "goto", ambiguous)
    driver = RecordingDriver()
    monkeypatch.setattr(graph, "graph_connection", lambda: (driver, "neo4j"))
    counts = graph.build_graph(
        str(repository),
        file_paths=[
            str(repository / p) for p in ("consumer.py", "helpers.py")
        ],
    )
    assert counts["ambiguous_imports"] == 1
    assert counts["external_imports"] == 2
    assert counts["import_edges"] == 0


def test_index_and_semantic_tool_keep_snippets_and_locations(
    build, monkeypatch
):
    from src.retrievers import query_graph

    root, directory, config, manifest, _ = build
    lookup = Mock(return_value=[["helpers.py:1:first"], []])
    monkeypatch.setattr(query_graph, "find_snippet_symbols", lookup)
    server = create_server(root, directory, config)
    result = data(call(server, "search_code", query="first", limit=1))
    assert result["build_id"] == manifest["build_id"]
    assert len(result["results"]) == 1 and result["truncated"]
    row = result["results"][0]
    assert row["content"] and (root / row["path"]).is_file()
    assert isinstance(row["distance"], float)
    assert row["symbol_ids"] == ["helpers.py:1:first"]
    assert lookup.call_args.kwargs == {
        "repository_root": str(root),
        "build_id": manifest["build_id"],
    }
    assert lookup.call_args.args[0][0]["path"] == row["path"]
    assert data(call(server, "index_status"))["fresh"]
    assert manifest["backend"] == "neo4j"
    assert manifest["vector_dimensions"] == 4
    assert manifest["counts"]["chunks"] > 0


def test_snippet_symbol_mapping_keeps_empty_and_multiple_matches(monkeypatch):
    from src.retrievers import query_graph

    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.side_effect = [
        Mock(single=Mock(return_value={"i": {}})),
        [
            {"snippet_index": 0, "symbol_id": "worker.py:8:outer"},
            {"snippet_index": 0, "symbol_id": "worker.py:9:outer.inner"},
            {"snippet_index": 2, "symbol_id": "helpers.py:1:first"},
        ],
    ]
    monkeypatch.setattr(
        query_graph, "graph_connection", lambda: (driver, "neo4j")
    )
    snippets = [
        {"path": "worker.py", "line_start": 10, "line_end": 10},
        {"path": "__init__.py", "line_start": 1, "line_end": 1},
        {"path": "helpers.py", "line_start": 1, "line_end": 2},
    ]
    assert query_graph.find_snippet_symbols(
        snippets, repository_root="/repo", build_id="test"
    ) == [
        ["worker.py:8:outer", "worker.py:9:outer.inner"],
        [],
        ["helpers.py:1:first"],
    ]
    parameters = session.run.call_args.kwargs
    assert parameters["repo"] == "/repo"
    assert parameters["snippets"][0] == {
        "index": 0,
        "path": "worker.py",
        "start": 10,
        "end": 10,
    }


@pytest.mark.parametrize("change", ["edit", "add", "delete"])
@pytest.mark.parametrize("include_code", [False, True])
def test_retrieval_refuses_stale_sources(build, change, include_code):
    root, directory, config, _, _ = build
    if change == "edit":
        (root / "worker.py").write_text("pass\n")
    elif change == "add":
        (root / "added.py").write_text("pass\n")
    else:
        (root / "worker.py").unlink()
    server = create_server(root, directory, config)
    assert not data(call(server, "index_status"))["fresh"]
    result = call(
        server, "find_definitions", name="first", include_code=include_code
    )
    assert result.is_error and "stale" in str(result.content)


@pytest.mark.parametrize(
    "include_code,max_lines,start_line",
    [
        (False, 2, None),
        (True, 2, None),
        (True, 100, None),
        (True, 1, 12),
        (True, 100, 12),
        (True, 1, 13),
        (True, 2, 9),
        (True, 2, 14),
        (True, 2, 0),
        (True, 2, True),
        (False, 2, 12),
    ],
)
def test_symbol_excerpts_preserve_ranges_and_bound_class_code(
    monkeypatch, include_code, max_lines, start_line
):
    from src.retrievers import query_graph

    content = (
        'class Café:\r\n    label = "Grüße"\r\n'
        '    def run(self):\r\n        return self.label'
    )
    symbol = {
        "symbol_id": "example.py:10:Café",
        "kind": "class",
        "path": "example.py",
        "line_start": 10,
        "line_end": 13,
    }
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.side_effect = [
        Mock(single=Mock(return_value={"i": {}})),
        [{**symbol, "source_content": content if include_code else None}],
    ]
    monkeypatch.setattr(
        query_graph, "graph_connection", lambda: (driver, "neo4j")
    )
    arguments = dict(
        repository_root="/repo",
        build_id="test",
        include_code=include_code,
        max_lines=max_lines,
        start_line=start_line,
    )
    if start_line is not None and (
        not include_code
        or isinstance(start_line, bool)
        or not 10 <= start_line <= 13
    ):
        with pytest.raises(ValueError, match="start_line"):
            query_graph.find_definitions("Café", **arguments)
        return
    rows = query_graph.find_definitions(
        "Café",
        **arguments,
    )
    assert session.run.call_args.kwargs["include_code"] is include_code
    row = rows[0]
    assert {key: row[key] for key in symbol} == symbol
    assert "source_content" not in row
    if include_code:
        first = 10 if start_line is None else start_line
        lines = content.splitlines(keepends=True)[
            first - 10 : first - 10 + max_lines
        ]
        assert row["code_excerpt"] == {
            "content": "".join(lines),
            "line_start": first,
            "line_end": first + len(lines) - 1,
            "truncated": first > 10 or max_lines < 4,
        }
    else:
        assert row == symbol


@pytest.mark.parametrize(
    "case", ["missing", "incomplete", "model", "graph_schema"]
)
def test_index_errors_are_actionable(build, case):
    root, directory, config, manifest, _ = build
    if case == "missing":
        (directory / "manifest.json").unlink()
    elif case == "incomplete":
        manifest["state"] = "building"
        index_state.write_manifest(directory, manifest)
    elif case == "graph_schema":
        manifest.pop("graph_schema")
        index_state.write_manifest(directory, manifest)
    else:
        config.write_text(
            config.read_text().replace("name: fixture", "name: changed")
        )
    result = call(
        create_server(root, directory, config), "search_code", query="first"
    )
    assert result.is_error and "Rebuild" in str(result.content)


@pytest.mark.parametrize("stage", ["graph", "vectors", "definitions"])
def test_failed_rebuild_remains_incomplete(build, monkeypatch, stage):
    root, directory, config, _, _ = build
    target, name = {
        "graph": (graph, "build_graph"),
        "vectors": (vector_search, "store_chunks"),
        "definitions": (vector_search, "store_definition_vectors"),
    }[stage]
    monkeypatch.setattr(
        target, name, Mock(side_effect=RuntimeError("fixture failure"))
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        indexing.index_repository(root, directory, config)
    assert (
        json.loads((directory / "manifest.json").read_text())["state"]
        == "building"
    )


@pytest.mark.parametrize("values", [[], [[float("nan")]], [[1], [1, 2]]])
def test_invalid_vectors_are_rejected(values):
    with pytest.raises(ValueError):
        vector_search.validate_vectors(values)


def test_definition_vectors_update_existing_nodes(monkeypatch):
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    function = {
        "id": "a.py:1:f",
        "path": "a.py",
        "name": "f",
        "content": "def f():\n    return 1\n",
    }
    class_definition = {
        "id": "a.py:4:C",
        "path": "a.py",
        "name": "C",
        "content": "class C:\n    pass\n",
    }

    def run(query, **kwargs):
        result = MagicMock()
        result.single.return_value = {"i": {}}
        if "RETURN d.symbol_id" in query:
            result.data.return_value = (
                [function] if ":Function " in query else [class_definition]
            )
        return result

    session.run.side_effect = run
    monkeypatch.setattr(
        vector_search, "graph_connection", lambda: (driver, "neo4j")
    )
    model = Mock(wraps=Embeddings())
    counts = vector_search.store_definition_vectors(
        model, repository_root="/repo", build_id="b", dimensions=4
    )
    assert counts == {"function_vectors": 1, "class_vectors": 1}
    assert model.embed_documents.call_args_list[0].args[0] == [
        "a.py:f\n" + function["content"]
    ]
    writes = [
        (args[0], kwargs)
        for args, kwargs in session.run.call_args_list
        if "UNWIND $rows" in args[0]
    ]
    assert [p["rows"][0]["id"] for _, p in writes] == [
        function["id"],
        class_definition["id"],
    ]
    assert all("CREATE" not in q and p["repo"] == "/repo" for q, p in writes)
    assert all(len(p["rows"][0]["embedding"]) == 4 for _, p in writes)


@pytest.mark.parametrize("kind", ["function", "class"])
def test_definition_search_returns_direct_ids(build, monkeypatch, kind):
    root, directory, config, _, _ = build
    search = Mock(
        return_value=[
            {
                "symbol_id": "a.py:1:Example",
                "path": "a.py",
                "content": "code",
                "line_start": 1,
                "line_end": 2,
                "distance": 0.0,
                "kind": kind,
            }
        ]
    )
    monkeypatch.setattr(vector_search, "search_vectors", search)
    result = data(
        call(
            create_server(root, directory, config),
            "search_code",
            query="example",
            kind=kind,
            limit=2,
        )
    )
    assert result["results"][0]["symbol_ids"] == ["a.py:1:Example"]
    assert search.call_args.kwargs["kind"] == kind
    assert search.call_args.kwargs["limit"] == 3


def test_definition_search_requires_rebuild_for_old_indexes(monkeypatch):
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = {
        "i": {"chunks": 2, "vector_dimensions": 4}
    }
    monkeypatch.setattr(
        vector_search, "graph_connection", lambda: (driver, "neo4j")
    )
    with pytest.raises(ValueError, match="rebuild"):
        vector_search.search_vectors(
            [1.0] * 4,
            repository_root="/repo",
            build_id="old",
            limit=2,
            kind="function",
        )


@pytest.mark.parametrize("problem", ["missing", "dimensions", "offline"])
def test_vector_backend_errors_are_explicit(monkeypatch, problem):
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    metadata = {"chunks": 2, "vector_dimensions": 4}
    if problem == "missing":
        metadata["chunks"] = 0
    elif problem == "dimensions":
        metadata["vector_dimensions"] = 3
    session.run.side_effect = [
        Mock(single=Mock(return_value={"i": metadata})),
        Mock(single=Mock(return_value={"state": "POPULATING"})),
    ]
    monkeypatch.setattr(
        vector_search, "graph_connection", lambda: (driver, "neo4j")
    )
    with pytest.raises(ValueError, match="rebuild"):
        vector_search.search_chunks(
            [1.0] * 4, repository_root="/repo", build_id="b", limit=2
        )


def test_neo4j_driver_is_shared_and_closed(monkeypatch):
    neo4j_connection.close_connection()
    driver = MagicMock()
    factory = Mock(return_value=driver)
    monkeypatch.setenv("NEO4J_PASSWORD", "fixture")
    monkeypatch.setattr(neo4j_connection.GraphDatabase, "driver", factory)
    try:
        first, _ = neo4j_connection.graph_connection()
        second, _ = neo4j_connection.graph_connection()
        assert first is second is driver
        assert factory.call_count == 1
        driver.close.assert_not_called()
    finally:
        neo4j_connection.close_connection()
    driver.close.assert_called_once()


def test_graph_tools_delegate_and_report_truncation(build, monkeypatch):
    from src.retrievers import query_graph

    root, directory, config, _, _ = build
    rows = [{"symbol_id": str(i)} for i in range(3)]
    lookup = Mock(return_value=rows)
    monkeypatch.setattr(query_graph, "find_definitions", lookup)
    monkeypatch.setattr(query_graph, "query_references", Mock(return_value=[]))
    inheritance = Mock(
        return_value=[
            {"relation": "parent", "symbol_id": str(i)} for i in range(3)
        ]
    )
    monkeypatch.setattr(query_graph, "query_inheritance", inheritance)
    imports = Mock(return_value=[{"path": str(i)} for i in range(3)])
    monkeypatch.setattr(query_graph, "query_imports", imports)
    server = create_server(root, directory, config)
    result = data(call(server, "find_definitions", name="run", limit=2))
    assert result["truncated"] and len(result["results"]) == 2
    assert lookup.call_args.kwargs["limit"] == 3
    assert lookup.call_args.kwargs["include_code"] is False
    assert lookup.call_args.kwargs["max_lines"] == 60
    call(
        server,
        "find_definitions",
        name="run",
        include_code=True,
        max_lines=1,
    )
    assert lookup.call_args.kwargs["include_code"] is True
    assert lookup.call_args.kwargs["max_lines"] == 1
    call(
        server,
        "find_definitions",
        name="run",
        include_code=True,
        start_line=12,
        max_lines=1,
    )
    assert lookup.call_args.kwargs["start_line"] == 12
    lookup.return_value = []
    empty = data(call(server, "find_definitions", name="missing"))
    assert empty["results"] == [] and "attributes" in empty["hint"]
    assert "hint" not in result
    for invalid in (0, 101):
        assert call(
            server,
            "find_definitions",
            name="run",
            include_code=True,
            max_lines=invalid,
        ).is_error
    assert (
        data(call(server, "get_callers", symbol_id="known"))["results"] == []
    )
    result = data(call(server, "get_inheritance", symbol_id="child", limit=2))
    assert result["truncated"] and len(result["results"]) == 2
    assert inheritance.call_args.kwargs["limit"] == 3
    assert call(
        server, "get_inheritance", symbol_id="child", limit=101
    ).is_error
    assert call(server, "search_code", query="first", limit=21).is_error
    assert call(server, "get_callers", symbol_id="known", limit=101).is_error
    assert call(
        server, "find_definitions", name="run", path="../outside.py"
    ).is_error
    result = data(
        call(
            server,
            "get_imports",
            path="worker.py",
            direction="importers",
            limit=2,
        )
    )
    assert result["truncated"] and len(result["results"]) == 2
    assert imports.call_args.kwargs["path"] == "worker.py"
    assert imports.call_args.kwargs["direction"] == "importers"
    assert imports.call_args.kwargs["limit"] == 3
    for arguments in (
        {"path": "../outside.py"},
        {"path": str(root / "worker.py")},
        {"path": "worker.py", "direction": "invalid"},
        {"path": "worker.py", "limit": 101},
    ):
        assert call(server, "get_imports", **arguments).is_error
    (root / "worker.py").write_text("pass\n")
    assert call(server, "get_imports", path="worker.py").is_error


def test_stdio_discovery_and_errors_do_not_load_models(build):
    root, directory, config, _, _ = build

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            cwd=str(Path(__file__).resolve().parents[1]),
            args=[
                "-m",
                "src.mcp_server",
                "--repo",
                str(root),
                "--index-dir",
                str(directory),
                "--config",
                str(config),
            ],
        )
        async with Client(params) as client:
            tools = await client.list_tools()
            assert {t.name for t in tools.tools} == {
                "search_code",
                "find_definitions",
                "get_callers",
                "get_callees",
                "get_inheritance",
                "get_imports",
                "index_status",
            }
            assert all(t.annotations.read_only_hint for t in tools.tools)
            schemas = {
                t.name: t.model_dump(by_alias=True)["inputSchema"]
                for t in tools.tools
            }
            for schema in schemas.values():
                assert all(
                    parameter.get("description")
                    for parameter in schema.get("properties", {}).values()
                )
            assert (
                schemas["search_code"]["properties"]["limit"]["default"] == 5
            )
            assert schemas["search_code"]["properties"]["kind"]["enum"] == [
                "chunk",
                "function",
                "class",
            ]
            assert (
                schemas["get_callers"]["properties"]["limit"]["default"] == 20
            )
            symbol_arguments = schemas["find_definitions"]["properties"]
            assert symbol_arguments["include_code"]["default"] is False
            assert symbol_arguments["max_lines"]["default"] == 60
            assert symbol_arguments["max_lines"]["minimum"] == 1
            assert symbol_arguments["max_lines"]["maximum"] == 100
            status = data(await client.call_tool("index_status", {}))
            assert status["fresh"]
            assert status["tool_description_version"] == (
                index_state.TOOL_DESCRIPTION_VERSION
            )
            assert (
                await client.call_tool(
                    "search_code", {"query": "first", "limit": 0}
                )
            ).is_error
            assert (
                await client.call_tool(
                    "get_inheritance", {"symbol_id": "child", "limit": 0}
                )
            ).is_error

    asyncio.run(run())
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.mcp_server, sys; assert 'torch' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def test_repeated_snippets_keep_their_actual_line_locations(repository):
    path = repository / "repeated.py"
    path.write_text(("# repeated content\n" + "x = 1\n" * 200) * 2)
    chunks = load_and_split_python_files(
        [str(path)], 900, 0, repository_root=str(repository)
    )
    starts = [d.metadata["line_start"] for d in chunks]
    assert starts == sorted(set(starts))
    for chunk in chunks:
        source = path.read_text().splitlines()
        assert chunk.page_content == "\n".join(
            source[
                chunk.metadata["line_start"] - 1 : chunk.metadata["line_end"]
            ]
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("CODEAWARE_TEST_NEO4J") != "1",
    reason="requires an isolated test Neo4j",
)
def test_real_neo4j_queries_rebuild_and_isolation(
    repository, config, tmp_path, monkeypatch
):
    from uuid import uuid4

    from src.neo4j_connection import graph_connection
    from src.retrievers.query_graph import find_snippet_symbols

    add_import_fixture(repository)
    monkeypatch.setattr(
        embedding_model, "get_embedding_model", lambda config: Embeddings()
    )
    directory = tmp_path / "index"
    other = tmp_path / "other-repository"
    other.mkdir()
    (other / "foreign.py").write_text('def first():\n    return "first"\n')
    other_directory = tmp_path / "other-index"
    marker = str(uuid4())
    driver, database = graph_connection()
    try:
        with driver.session(database=database) as session:
            session.run(
                "CREATE (:CodeawareTestMarker {id:$id})", id=marker
            ).consume()
            session.run(
                "CREATE (:Codeaware:Function {repository:$repo, "
                "symbol_id:'foreign'})",
                repo=marker,
            ).consume()
        indexing.index_repository(other, other_directory, config)
        manifest = indexing.index_repository(repository, directory, config)
        server = create_server(repository, directory, config)
        for kind in ("function", "class"):
            result = data(
                call(server, "search_code", query="first", kind=kind, limit=2)
            )
            assert len(result["results"]) == 2 and result["truncated"]
            for row in result["results"]:
                assert row["symbol_ids"] == [row["symbol_id"]]
                assert row["kind"] == kind
                assert row["content"]
            with driver.session(database=database) as session:
                stored = session.run(
                    f"MATCH (d:Codeaware:{kind.title()} {{repository:$repo}}) "
                    "RETURN count(d) AS total, "
                    "sum(CASE WHEN size(d.embedding) = 4 THEN 1 ELSE 0 END) "
                    "AS embedded",
                    repo=str(repository),
                ).single()
                assert stored["embedded"] == stored["total"]
                assert stored["total"] == manifest["counts"][f"{kind}_vectors"]
        imports = data(
            call(server, "get_imports", path="package/consumer.py")
        )["results"]
        assert len(imports) == 9
        assert {r["path"] for r in imports} == {
            "helpers.py",
            "package/items.py",
            "package/__init__.py",
        }
        assert {r["import_path"] for r in imports} == {"package/consumer.py"}
        assert (
            next(r for r in imports if r["alias"] == "nested")["import_column"]
            == 36
        )
        importers = data(
            call(
                server,
                "get_imports",
                path="package/items.py",
                direction="importers",
            )
        )["results"]
        assert len(importers) == 8
        assert {r["path"] for r in importers} == {
            "package/__init__.py",
            "package/consumer.py",
        }
        assert (
            data(call(server, "get_imports", path="package/items.py"))[
                "results"
            ]
            == []
        )
        bounded = data(
            call(server, "get_imports", path="package/consumer.py", limit=1)
        )
        assert bounded["truncated"] and len(bounded["results"]) == 1
        assert call(server, "get_imports", path="missing.py").is_error
        matches = data(call(server, "find_definitions", name="run"))["results"]
        assert {r["qualified_name"] for r in matches} == {
            "First.run",
            "Second.run",
        }
        first = data(
            call(server, "find_definitions", name="first", path="helpers.py")
        )["results"][0]
        assert "code_excerpt" not in first
        excerpt = data(
            call(
                server,
                "find_definitions",
                name="first",
                path="helpers.py",
                include_code=True,
                max_lines=1,
            )
        )["results"][0]
        assert excerpt["symbol_id"] == first["symbol_id"]
        assert excerpt["line_end"] == first["line_end"]
        assert excerpt["code_excerpt"] == {
            "content": "def first():\n",
            "line_start": 1,
            "line_end": 1,
            "truncated": True,
        }
        later = data(
            call(
                server,
                "find_definitions",
                name="first",
                path="helpers.py",
                include_code=True,
                start_line=2,
                max_lines=1,
            )
        )["results"][0]
        assert later["code_excerpt"] == {
            "content": '    return "first"\n',
            "line_start": 2,
            "line_end": 2,
            "truncated": True,
        }
        callers = data(
            call(server, "get_callers", symbol_id=first["symbol_id"])
        )["results"]
        assert {r["qualified_name"] for r in callers} == {
            "First.run",
            "entry",
            "outer.inner",
        }
        entry = next(r for r in callers if r["qualified_name"] == "entry")
        assert entry["call_path"] == "worker.py" and entry["call_line"] == 5
        for row in callers:
            lines = (
                (repository / row["path"])
                .read_text()
                .splitlines(keepends=True)
            )
            assert row["content"] == "".join(
                lines[row["line_start"] - 1 : row["line_end"]]
            )
        outgoing = data(
            call(server, "get_callees", symbol_id=entry["symbol_id"])
        )["results"]
        assert [r["symbol_id"] for r in outgoing] == [first["symbol_id"]]
        assert outgoing[0]["content"] == 'def first():\n    return "first"\n'
        nested = data(call(server, "find_definitions", name="outer.inner"))[
            "results"
        ][0]
        snippets = [
            {
                "path": nested["path"],
                "line_start": nested["line_end"],
                "line_end": nested["line_end"],
            },
            {"path": "worker.py", "line_start": 1, "line_end": 1},
        ]
        overlapping = find_snippet_symbols(
            snippets,
            repository_root=str(repository),
            build_id=manifest["build_id"],
        )
        outer = data(call(server, "find_definitions", name="outer"))[
            "results"
        ][0]
        assert overlapping == [[outer["symbol_id"], nested["symbol_id"]], []]
        search = data(call(server, "search_code", query="first", limit=20))
        helper = next(
            row for row in search["results"] if row["path"] == "helpers.py"
        )
        assert first["symbol_id"] in helper["symbol_ids"]
        assert {row["symbol_id"] for row in matches} <= set(
            helper["symbol_ids"]
        )
        assert "foreign.py" not in {row["path"] for row in search["results"]}
        for row in search["results"]:
            lines = (repository / row["path"]).read_text().splitlines()
            assert row["content"] == "\n".join(
                lines[row["line_start"] - 1 : row["line_end"]]
            )
            expected = sum(
                (left - right) ** 2
                for left, right in zip(
                    Embeddings().embed_query("first"),
                    Embeddings().embed_query(row["content"]),
                )
            )
            assert row["distance"] == pytest.approx(expected)
        classes = data(call(server, "find_definitions", name="Child"))[
            "results"
        ]
        assert len(classes) == 2 and all(
            row["kind"] == "class" for row in classes
        )
        child = next(row for row in classes if row["path"] == "worker.py")
        relatives = data(
            call(server, "get_inheritance", symbol_id=child["symbol_id"])
        )["results"]
        assert [
            (row["qualified_name"], row["relation"], row["base_index"])
            for row in relatives
        ] == [
            ("First", "parent", 0),
            ("Second", "parent", 1),
            ("Grandchild", "subclass", 0),
        ]
        child_snippet = next(
            row
            for row in search["results"]
            if child["symbol_id"] in row["symbol_ids"]
        )
        assert child_snippet["path"] == "worker.py"
        empty = next(row for row in classes if row["path"] == "helpers.py")
        assert (
            data(
                call(server, "get_inheritance", symbol_id=empty["symbol_id"])
            )["results"]
            == []
        )
        assert call(
            server, "get_inheritance", symbol_id=first["symbol_id"]
        ).is_error
        assert call(server, "get_inheritance", symbol_id="unknown").is_error
        assert call(
            server, "get_callers", symbol_id=child["symbol_id"]
        ).is_error

        async def inheritance_over_stdio():
            params = StdioServerParameters(
                command=sys.executable,
                cwd=str(Path(__file__).resolve().parents[1]),
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith("NEO4J_")
                },
                args=[
                    "-c",
                    "import runpy; from src.indexing import model; "
                    "from src.mcp_server import create_server; "
                    "E = runpy.run_path('tests/test_mcp.py')['Embeddings']; "
                    "model.get_embedding_model = lambda config: E(); "
                    f"create_server({str(repository)!r}, {str(directory)!r}, "
                    f"{str(config)!r}).run(transport='stdio')",
                ],
            )
            async with Client(params) as client:
                semantic = data(
                    await client.call_tool(
                        "search_code", {"query": "first", "limit": 2}
                    )
                )
                assert len(semantic["results"]) == 2
                assert all("distance" in row for row in semantic["results"])
                excerpt_result = data(
                    await client.call_tool(
                        "find_definitions",
                        {
                            "name": "first",
                            "path": "helpers.py",
                            "include_code": True,
                            "max_lines": 1,
                        },
                    )
                )
                assert excerpt_result["results"][0] == excerpt
                later_result = data(
                    await client.call_tool(
                        "find_definitions",
                        {
                            "name": "first",
                            "path": "helpers.py",
                            "include_code": True,
                            "start_line": 2,
                            "max_lines": 1,
                        },
                    )
                )
                assert later_result["results"][0] == later
                result = data(
                    await client.call_tool(
                        "get_inheritance", {"symbol_id": child["symbol_id"]}
                    )
                )
                assert result["results"] == relatives
                result = data(
                    await client.call_tool(
                        "get_imports", {"path": "package/consumer.py"}
                    )
                )
                assert result["results"] == imports

        asyncio.run(inheritance_over_stdio())
        unused = data(call(server, "find_definitions", name="unused"))[
            "results"
        ][0]
        assert (
            data(call(server, "get_callers", symbol_id=unused["symbol_id"]))[
                "results"
            ]
            == []
        )
        assert call(server, "get_callers", symbol_id="unknown").is_error
        assert (
            data(call(server, "find_definitions", name="first' RETURN f //"))[
                "results"
            ]
            == []
        )
        indexing.index_repository(repository, directory, config)
        assert call(
            server, "find_definitions", name="first"
        ).is_error  # Requires reconnect.
        fresh = create_server(repository, directory, config)
        assert (
            data(call(fresh, "get_imports", path="package/consumer.py"))[
                "results"
            ]
            == imports
        )
        assert (
            data(call(fresh, "index_status"))["build_id"]
            != manifest["build_id"]
        )
        foreign = create_server(other, other_directory, config)
        assert (
            data(call(foreign, "search_code", query="first"))["results"][0][
                "path"
            ]
            == "foreign.py"
        )
        index_name, _ = vector_search.vector_names(str(repository))
        with driver.session(database=database) as session:
            session.run(f"DROP INDEX {index_name}").consume()
        missing = call(fresh, "search_code", query="first")
        assert missing.is_error and "missing" in str(missing.content)
        with driver.session(database=database) as session:
            assert (
                session.run(
                    "MATCH (m:CodeawareTestMarker {id:$id}) RETURN "
                    "count(m) AS n",
                    id=marker,
                ).single()["n"]
                == 1
            )
            assert (
                session.run(
                    "MATCH (f:Codeaware {repository:$repo}) RETURN "
                    "count(f) AS n",
                    repo=marker,
                ).single()["n"]
                == 1
            )
    finally:
        with driver.session(database=database) as session:
            for root in (repository, other):
                for kind in ("chunk", "function", "class"):
                    index_name, _ = vector_search.vector_names(str(root), kind)
                    session.run(f"DROP INDEX {index_name} IF EXISTS").consume()
                session.run(
                    "MATCH (n:Codeaware {repository:$repo}) DETACH DELETE n",
                    repo=str(root),
                ).consume()
            session.run(
                "MATCH (n:Codeaware {repository:$repo}) DETACH DELETE n",
                repo=str(repository),
            ).consume()
            session.run(
                "MATCH (n:Codeaware {repository:$repo}) DETACH DELETE n",
                repo=marker,
            ).consume()
            session.run(
                "MATCH (n:CodeawareTestMarker {id:$id}) DELETE n", id=marker
            ).consume()
