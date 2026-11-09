from pathlib import Path

from src.neo4j_connection import graph_connection


def check_graph(session, repository_root, build_id):
    result = session.run(
        "MATCH (i:CodeawareIndex {repository: $repo, build_id: $build}) "
        "RETURN i",
        repo=repository_root,
        build=build_id,
    ).single()
    if result is None:
        raise ValueError(
            "Graph is missing or from another build; rebuild the index "
            "and reconnect"
        )
    return dict(result["i"])


def find_definitions(
    name,
    *,
    repository_root,
    build_id,
    path=None,
    limit=10,
    include_code=False,
    max_lines=60,
    start_line=None,
):
    """
    Exact function/method/class names; return all matches so the agent
    can disambiguate. Optionally include bounded indexed source code.
    """
    if (
        isinstance(max_lines, bool)
        or not isinstance(max_lines, int)
        or not 1 <= max_lines <= 100
    ):
        raise ValueError("max_lines must be an integer between 1 and 100")
    if start_line is not None and (
        isinstance(start_line, bool)
        or not isinstance(start_line, int)
        or start_line < 1
        or not include_code
    ):
        raise ValueError(
            "start_line must be a positive integer with include_code=true"
        )
    driver, database = graph_connection()
    with driver.session(database=database) as session:
        check_graph(session, repository_root, build_id)
        result = session.run(
            "MATCH (f:Codeaware {repository: $repo}) "
            "WHERE (f:Function OR f:Class) AND (f.name = $name OR "
            "f.qualified_name = $name) "
            "AND ($path IS NULL OR f.path = $path) "
            "RETURN f.symbol_id AS symbol_id, f.name AS name, "
            "f.qualified_name AS qualified_name, f.kind AS kind, "
            "f.path AS path, f.start_line AS line_start, f.end_line AS "
            "line_end, CASE WHEN $include_code THEN f.content ELSE null "
            "END AS source_content "
            "ORDER BY f.path, f.start_line LIMIT $limit",
            repo=repository_root,
            name=name,
            path=path,
            limit=limit,
            include_code=include_code,
        )
        rows = [dict(row) for row in result]
        for row in rows:
            content = row.pop("source_content", None)
            if include_code:
                if not content:
                    raise ValueError(
                        "Indexed symbol code is missing; rebuild the index"
                    )
                lines = content.splitlines(keepends=True)
                first = row["line_start"] if start_line is None else start_line
                if not row["line_start"] <= first <= row["line_end"]:
                    raise ValueError(
                        "start_line must be within each matched definition; "
                        "narrow by name/path or select another start line"
                    )
                offset = first - row["line_start"]
                excerpt = lines[offset : offset + max_lines]
                row["code_excerpt"] = {
                    "content": "".join(excerpt),
                    "line_start": first,
                    "line_end": first + len(excerpt) - 1,
                    "truncated": offset > 0 or len(lines) > max_lines,
                }
        return rows


def find_snippet_symbols(snippets, *, repository_root, build_id):
    """
    Map each snippet to functions/methods/classes overlapping its
    inclusive line range.
    """
    symbol_ids = [[] for _ in snippets]
    ranges = [
        {
            "index": i,
            "path": row["path"],
            "start": row["line_start"],
            "end": row["line_end"],
        }
        for i, row in enumerate(snippets)
    ]
    driver, database = graph_connection()
    with driver.session(database=database) as session:
        check_graph(session, repository_root, build_id)
        result = session.run(
            "UNWIND $snippets AS snippet "
            "MATCH (f:Codeaware {repository: $repo}) "
            "WHERE (f:Function OR f:Class) AND f.path = snippet.path "
            "AND f.start_line <= snippet.end AND f.end_line >= snippet.start "
            "RETURN snippet.index AS snippet_index, f.symbol_id AS symbol_id "
            "ORDER BY snippet_index, f.start_line, f.symbol_id",
            repo=repository_root,
            snippets=ranges,
        )
        for row in result:
            symbol_ids[row["snippet_index"]].append(row["symbol_id"])
    return symbol_ids


def query_imports(
    path, *, repository_root, build_id, direction="imports", limit=20
):
    """
    Return one hop of resolved import bindings or reverse importers.
    Paths and sites are relative to the configured repository.
    """
    patterns = {
        "imports": "(target)-[r:IMPORTS]->(other:Codeaware:File)",
        "importers": "(target)<-[r:IMPORTS]-(other:Codeaware:File)",
    }
    if direction not in patterns:
        raise ValueError("direction must be imports or importers")
    absolute_path = str(Path(repository_root) / path)
    driver, database = graph_connection()
    with driver.session(database=database) as session:
        check_graph(session, repository_root, build_id)
        exists = session.run(
            "MATCH (f:Codeaware:File {repository:$repo, path:$path}) RETURN f",
            repo=repository_root,
            path=absolute_path,
        ).single()
        if exists is None:
            raise ValueError(
                "Unknown file path; select an indexed Python file"
            )
        result = session.run(
            "MATCH (target:Codeaware:File {repository:$repo, path:$path}) "
            f"MATCH {patterns[direction]} WHERE other.repository = $repo "
            "RETURN substring(other.path, size($repo) + 1) AS path, "
            "substring(startNode(r).path, size($repo) + 1) AS import_path, "
            "r.line AS import_line, r.column AS import_column, "
            "r.module AS module, r.name AS name, r.alias AS alias "
            "ORDER BY path, import_line, import_column LIMIT $limit",
            repo=repository_root,
            path=absolute_path,
            limit=limit,
        )
        return [dict(row) for row in result]


def query_inheritance(symbol_id, *, repository_root, build_id, limit=20):
    """
    Return direct parents/subclasses; base_index preserves written base
    order, not MRO.
    """
    driver, database = graph_connection()
    with driver.session(database=database) as session:
        check_graph(session, repository_root, build_id)
        exists = session.run(
            "MATCH (c:Codeaware:Class {repository:$repo, symbol_id:$id}) "
            "RETURN c",
            repo=repository_root,
            id=symbol_id,
        ).single()
        if exists is None:
            raise ValueError(
                "Unknown class symbol_id; use find_definitions to select a "
                "class from this index"
            )
        result = session.run(
            "MATCH (target:Codeaware:Class {repository:$repo, symbol_id:$id}) "
            "MATCH (target)-[r:INHERITS_FROM]-(other:Codeaware:Class) "
            "WHERE other.repository = $repo "
            "RETURN CASE WHEN startNode(r) = target THEN 'parent' ELSE "
            "'subclass' END AS relation, "
            "other.symbol_id AS symbol_id, other.qualified_name AS "
            "qualified_name, "
            "other.path AS path, other.start_line AS line_start, "
            "other.end_line AS line_end, "
            "r.base_index AS base_index "
            "ORDER BY relation, base_index, other.path, other.symbol_id "
            "LIMIT $limit",
            repo=repository_root,
            id=symbol_id,
            limit=limit,
        )
        return [dict(row) for row in result]


def query_calls(
    symbol_id, *, repository_root, build_id, direction="callers", limit=20
):
    """
    One hop through statically resolved calls, including each call-site
    location.
    """
    patterns = {
        "callers": "(target)<-[r:CALLS]-(other:Codeaware:Function)",
        "callees": "(target)-[r:CALLS]->(other:Codeaware:Function)",
    }
    pattern = patterns[
        direction
    ]  # An internal choice, never agent-supplied Cypher.
    driver, database = graph_connection()
    with driver.session(database=database) as session:
        check_graph(session, repository_root, build_id)
        exists = session.run(
            "MATCH (f:Codeaware:Function {repository: $repo, symbol_id: "
            "$id}) RETURN f",
            repo=repository_root,
            id=symbol_id,
        ).single()
        if exists is None:
            raise ValueError(
                "Unknown function/method symbol_id; use find_definitions to "
                "select a function from this index"
            )
        result = session.run(
            "MATCH (target:Codeaware:Function {repository: $repo, "
            "symbol_id: $id}) "
            f"MATCH {pattern} WHERE other.repository = $repo "
            "RETURN other.symbol_id AS symbol_id, other.qualified_name AS "
            "qualified_name, "
            "other.path AS path, other.start_line AS line_start, "
            "other.end_line AS line_end, other.content AS content, "
            "r.path AS call_path, r.line AS call_line, r.column AS "
            "call_column, r.call_type AS call_type "
            "ORDER BY other.path, r.line, r.column, other.symbol_id LIMIT "
            "$limit",
            repo=repository_root,
            id=symbol_id,
            limit=limit,
        )
        return [dict(row) for row in result]


def query_references(*, symbol_id, repository_root, build_id, limit=20):
    return query_calls(
        symbol_id,
        repository_root=repository_root,
        build_id=build_id,
        limit=limit,
    )
