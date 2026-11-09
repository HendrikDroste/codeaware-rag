"""Build a partial Python graph with Tree-sitter and Jedi.

Start at build_graph(): set up the parser, store definitions, then
resolve relationships. Tree-sitter finds syntax; Jedi resolves targets.
Uncertain targets are omitted and counted rather than guessed.
"""

import io
import os
import sys
from pathlib import Path

import jedi
import tree_sitter_python as tspython
from neo4j import Session
from tree_sitter import Language, Node, Parser, Point, Query, Tree

from src.indexing.file_processor import get_all_python_files
from src.neo4j_connection import graph_connection

# Absolute file, one-based definition-name line, character column.
DefinitionLocation = tuple[str, int, int]

DEFINITIONS_QUERY = "[(function_definition) (class_definition)] @definition"
CALLS_QUERY = """
(call function: [
    (identifier) @function.call
    (attribute attribute: (identifier) @method.call)
])
"""
IMPORTS_QUERY = "[(import_statement) (import_from_statement)] @import"
CLASSES_QUERY = "(class_definition) @class"

DELETE_REPOSITORY = "MATCH (n:Codeaware {repository: $repo}) DETACH DELETE n"

CREATE_INDEX = (
    "CREATE (:Codeaware:CodeawareIndex {repository: $repo, "
    "build_id: $build})"
)

MERGE_FILE = "MERGE (f:Codeaware:File {path:$path, repository:$repo})"

# Definitions use Function or Class. Do not add the legacy Symbol label
# or symbol_name_constraint.
MERGE_DEFINITION = (
    "MERGE (func:Codeaware:{label} "
    "{symbol_id:$id, repository:$repo}) "
    "SET func.name = $fn, func.file = $fp, func.path "
    "= $path, "
    "func.qualified_name = $qualified, func.kind = $kind, "
    "func.start_line = $sln, func.end_line = $eln, "
    "func.content = $content "
    "WITH func "
    "MATCH (file:Codeaware:File {path:$fp, "
    "repository:$repo}) "
    "MERGE (func)-[:DEFINED_IN]->(file)"
)

MERGE_IMPORT = (
    "MATCH (source:Codeaware:File {path:$source, "
    "repository:$repo}), (target:Codeaware:File "
    "{path:$target, repository:$repo}) "
    "MERGE (source)-[r:IMPORTS {line:$line, column:$column}]"
    "->(target) "
    "SET r.module = $module, r.name = $name, r.alias = $alias"
)

MERGE_INHERITANCE = (
    "MATCH (child:Codeaware:Class {symbol_id:$child, "
    "repository:$repo}), "
    "(parent:Codeaware:Class {symbol_id:$parent, "
    "repository:$repo}) "
    "MERGE (child)-[:INHERITS_FROM "
    "{base_index:$position}]->(parent)"
)

MERGE_CALL = (
    "MATCH (caller:Codeaware:Function {symbol_id:$c1, "
    "repository:$repo}), "
    "(callee:Codeaware:Function {symbol_id:$c2, repository:$repo}) "
    "MERGE (caller)-[:CALLS {line:$ln, column:$col, path:$path, "
    "call_type:$ct}]->(callee)"
)


def build_graph(
    codebase_path: str | Path,
    *,
    file_paths: list[str] | None = None,
    build_id: str = "manual",
) -> dict[str, int]:
    """Store definitions, resolve relationships, then return counts.

    Both location maps connect Jedi's resolved targets to stored IDs.
    Keep the passes separate so references to later files can resolve.
    """
    # 1. Discover files and set up Tree-sitter and Jedi.
    codebase_path = os.path.realpath(codebase_path)
    file_paths = (
        get_all_python_files(codebase_path)
        if file_paths is None
        else file_paths
    )
    counts = _new_counts(len(file_paths))
    indexed_files = {os.path.realpath(path) for path in file_paths}
    function_locations: dict[DefinitionLocation, str] = {}
    class_locations: dict[DefinitionLocation, str] = {}
    project = jedi.Project(
        codebase_path,
        environment_path=sys.executable,
        added_sys_path=[os.path.join(codebase_path, "src")],
    )
    language = Language(tspython.language())
    parser = Parser(language)
    queries = {
        "definitions": language.query(DEFINITIONS_QUERY),
        "calls": language.query(CALLS_QUERY),
        "imports": language.query(IMPORTS_QUERY),
        "classes": language.query(CLASSES_QUERY),
    }
    driver, database = graph_connection()
    with driver.session(database=database) as session:
        # Replace only this repository's Codeaware data.
        session.run(DELETE_REPOSITORY, repo=codebase_path).consume()
        session.run(CREATE_INDEX, repo=codebase_path, build=build_id).consume()
        # 2. Store every function/class before resolving any references.
        for file_path in file_paths:
            if not file_path.endswith(".py"):
                continue
            filepath = os.path.abspath(file_path)
            tree, source_text = _read_source(filepath, parser)
            _store_definitions(
                tree,
                source_text,
                queries,
                session,
                filepath,
                codebase_path,
                function_locations,
                class_locations,
                counts,
            )
        # 3. Resolve imports, parent classes and calls using those IDs.
        for file_path in file_paths:
            if not file_path.endswith(".py"):
                continue
            filepath = os.path.abspath(file_path)
            tree, source_text = _read_source(filepath, parser)
            _store_relationships(
                tree,
                source_text,
                queries,
                project,
                session,
                filepath,
                codebase_path,
                indexed_files,
                function_locations,
                class_locations,
                counts,
            )
    return counts


def _new_counts(file_count: int) -> dict[str, int]:
    """Create the counters returned by the build."""
    return {
        "files": file_count,
        "symbols": 0,
        "calls": 0,
        "unresolved_calls": 0,
        "ambiguous_calls": 0,
        "external_calls": 0,
        "out_of_scope_calls": 0,
        "classes": 0,
        "inheritance_edges": 0,
        "unresolved_bases": 0,
        "imports": 0,
        "import_edges": 0,
        "unresolved_imports": 0,
        "ambiguous_imports": 0,
        "external_imports": 0,
        "wildcard_imports": 0,
    }


def _read_source(filepath: str, parser: Parser) -> tuple[Tree, str]:
    """Read UTF-8 source and reject syntax errors in either pass."""
    code = Path(filepath).read_bytes()
    tree = parser.parse(code)
    if tree.root_node.has_error:
        raise ValueError(f"Cannot index Python syntax errors in {filepath}")
    return tree, code.decode("utf8")


def _jedi_position(source_lines: list[str], point: Point) -> tuple[int, int]:
    """Translate Tree-sitter coordinates into Jedi coordinates.

    Tree-sitter counts bytes; Jedi counts characters. They differ when
    non-ASCII characters occur before the name being resolved.
    """
    row, byte_column = point
    column = len(source_lines[row].encode("utf8")[:byte_column].decode("utf8"))
    return row + 1, column


def _enclosing_function(node: Node) -> Node | None:
    """Find the executing function body, skipping default arguments."""
    parent = node.parent
    while parent:
        if parent.type == "function_definition":
            body = parent.child_by_field_name("body")
            if body.start_byte <= node.start_byte < body.end_byte:
                return parent
        parent = parent.parent
    return None


def _store_definitions(
    tree: Tree,
    source_text: str,
    queries: dict[str, Query],
    session: Session,
    filepath: str,
    codebase_path: str | Path,
    function_locations: dict[DefinitionLocation, str],
    class_locations: dict[DefinitionLocation, str],
    counts: dict[str, int],
) -> None:
    """Store definitions and remember where each name was declared.

    For example, First.run and Second.run have different IDs even
    though both methods share the simple name run.
    """
    # Match text-mode universal newlines used by the original loader.
    code_lines = io.StringIO(source_text, newline=None).readlines()
    # Byte splitting matches Tree-sitter rows, including Unicode text.
    source_lines = [
        line.decode("utf8") for line in source_text.encode("utf8").splitlines()
    ]
    session.run(
        MERGE_FILE,
        path=filepath,
        repo=codebase_path,
    ).consume()

    captures = queries["definitions"].captures(tree.root_node)

    for node in captures.get("definition", []):

        name_node = node.child_by_field_name('name')
        if name_node is None:
            continue

        definition_name = name_node.text.decode('utf8')
        start_line = node.start_point[0] + 1  # Convert to 1-based indexing
        end_line = node.end_point[0] + 1

        definition_content = ''.join(code_lines[start_line - 1 : end_line])

        qualified_name = get_qualified_name(node)
        relative_path = os.path.relpath(filepath, codebase_path)
        symbol_id = f"{relative_path}:{start_line}:{qualified_name}"
        name_line, name_column = _jedi_position(
            source_lines, name_node.start_point
        )
        is_class = node.type == "class_definition"
        definition_locations = (
            class_locations if is_class else function_locations
        )
        definition_locations[(filepath, name_line, name_column)] = symbol_id
        counts["symbols"] += 1
        counts["classes"] += int(is_class)
        label = (
            "Class" if is_class else "Function"
        )  # Fixed internal labels, never user input.

        session.run(
            MERGE_DEFINITION.replace("{label}", label),
            fn=definition_name,
            fp=filepath,
            sln=start_line,
            eln=end_line,
            content=definition_content,
            id=symbol_id,
            repo=codebase_path,
            path=relative_path,
            qualified=qualified_name,
            kind="class" if is_class else "function",
        ).consume()


def _store_relationships(
    tree: Tree,
    source_text: str,
    queries: dict[str, Query],
    project: jedi.Project,
    session: Session,
    filepath: str,
    codebase_path: str | Path,
    indexed_files: set[str],
    function_locations: dict[DefinitionLocation, str],
    class_locations: dict[DefinitionLocation, str],
    counts: dict[str, int],
) -> None:
    """Connect this file to the definitions from pass one."""
    source_lines = source_text.splitlines()
    captures = queries["calls"].captures(tree.root_node)
    script = jedi.Script(code=source_text, path=filepath, project=project)
    import_nodes = (
        queries["imports"].captures(tree.root_node).get("import", [])
    )
    process_imports(
        import_nodes,
        script,
        session,
        filepath,
        codebase_path,
        indexed_files,
        counts,
        source_lines,
    )
    class_nodes = queries["classes"].captures(tree.root_node).get("class", [])
    process_inheritance(
        class_nodes,
        script,
        session,
        filepath,
        codebase_path,
        class_locations,
        counts,
        source_lines,
    )

    process_function_calls(
        captures.get("function.call", []),
        script,
        session,
        filepath,
        codebase_path,
        function_locations,
        counts,
        source_lines,
    )
    process_method_calls(
        captures.get("method.call", []),
        script,
        session,
        filepath,
        codebase_path,
        function_locations,
        counts,
        source_lines,
    )


def process_imports(
    nodes: list[Node],
    script: jedi.Script,
    session: Session,
    filepath: str,
    repository_root: str,
    indexed_files: set[str],
    counts: dict[str, int],
    source_lines: list[str],
) -> None:
    """
    Resolve explicit import bindings to indexed files, following
    re-exports. These are static dependencies, not runtime load order.
    """
    for node in nodes:
        if any(c.type == "wildcard_import" for c in node.named_children):
            counts["imports"] += 1
            counts["wildcard_imports"] += 1
            continue
        module = node.child_by_field_name("module_name")
        for binding in node.children_by_field_name("name"):
            counts["imports"] += 1
            name = binding
            alias = None
            if binding.type == "aliased_import":
                name = binding.child_by_field_name("name")
                alias = binding.child_by_field_name("alias").text.decode(
                    "utf8"
                )
            line, column = _jedi_position(source_lines, name.end_point)
            definitions = script.goto(
                line=line, column=column, follow_imports=True
            )
            if not definitions:
                counts["unresolved_imports"] += 1
                continue
            # Do not guess even if ambiguous definitions share a file.
            if len(definitions) != 1:
                counts["ambiguous_imports"] += 1
                continue
            target = definitions[0].module_path
            if target is None or os.path.realpath(target) not in indexed_files:
                counts["external_imports"] += 1
                continue
            line, column = _jedi_position(source_lines, name.start_point)
            session.run(
                MERGE_IMPORT,
                source=filepath,
                target=os.path.realpath(target),
                repo=repository_root,
                module=module.text.decode("utf8") if module else None,
                name=name.text.decode("utf8"),
                alias=alias,
                line=line,
                column=column,
            ).consume()
            counts["import_edges"] += 1


def process_inheritance(
    nodes: list[Node],
    script: jedi.Script,
    session: Session,
    filepath: str,
    repository_root: str,
    locations: dict[DefinitionLocation, str],
    counts: dict[str, int],
    source_lines: list[str],
) -> None:
    """
    Resolve explicit named bases only; omit dynamic, external or
    ambiguous bases.
    """
    for node in nodes:
        bases = node.child_by_field_name("superclasses")
        if bases is None:
            continue
        name = node.child_by_field_name("name")
        line, column = _jedi_position(source_lines, name.start_point)
        child_id = locations[(filepath, line, column)]
        # Keyword arguments (such as metaclass=...) are not parent
        # classes.
        bases = [
            base
            for base in bases.named_children
            if base.type != "keyword_argument"
        ]
        for position, base in enumerate(bases):
            expression = base
            while expression.type == "attribute":
                expression = expression.child_by_field_name("object")
            if expression.type != "identifier":
                counts["unresolved_bases"] += 1
                continue
            line, column = _jedi_position(source_lines, base.end_point)
            definitions = script.infer(line=line, column=column)
            parent_id = None
            if (
                len(definitions) == 1
                and definitions[0].type == "class"
                and definitions[0].module_path is not None
            ):
                definition = definitions[0]
                parent_id = locations.get(
                    (
                        os.path.realpath(definition.module_path),
                        definition.line,
                        definition.column,
                    )
                )
            if parent_id is None:
                counts["unresolved_bases"] += 1
                continue
            session.run(
                MERGE_INHERITANCE,
                child=child_id,
                parent=parent_id,
                repo=repository_root,
                position=position,
            ).consume()
            counts["inheritance_edges"] += 1


def process_function_calls(
    nodes: list[Node],
    script: jedi.Script,
    session: Session,
    filepath: str,
    repository_root: str,
    locations: dict[DefinitionLocation, str],
    counts: dict[str, int],
    source_lines: list[str],
    call_type: str = "function",
) -> None:
    """Resolve call targets and store their call-site locations."""
    for node in nodes:
        parent = _enclosing_function(node)
        if parent is None:
            counts["out_of_scope_calls"] += 1
            continue

        caller_node = parent.child_by_field_name('name')
        line, column = _jedi_position(source_lines, caller_node.start_point)
        caller_id = locations[(filepath, line, column)]
        line, column = _jedi_position(source_lines, node.end_point)
        definitions = script.infer(line=line, column=column)
        targets = {
            (os.path.realpath(d.module_path), d.line, d.column)
            for d in definitions
            if d.module_path is not None
        }
        if not targets:
            counts["unresolved_calls"] += 1
            continue
        if len(targets) != 1:
            counts["ambiguous_calls"] += 1
            continue
        callee_id = locations.get(next(iter(targets)))
        if callee_id is None:
            counts["external_calls"] += 1
            continue
        call_node = node.parent
        while call_node.type != 'call':
            call_node = call_node.parent
        call_line, call_column = _jedi_position(
            source_lines, call_node.start_point
        )
        session.run(
            MERGE_CALL,
            c1=caller_id,
            c2=callee_id,
            repo=repository_root,
            ln=call_line,
            col=call_column,
            path=os.path.relpath(filepath, repository_root),
            ct=call_type,
        ).consume()
        counts["calls"] += 1


def process_method_calls(
    nodes: list[Node],
    script: jedi.Script,
    session: Session,
    filepath: str,
    repository_root: str,
    locations: dict[DefinitionLocation, str],
    counts: dict[str, int],
    source_lines: list[str],
) -> None:
    """Resolve method calls using the shared call-resolution checks."""
    return process_function_calls(
        nodes,
        script,
        session,
        filepath,
        repository_root,
        locations,
        counts,
        source_lines,
        call_type="method",
    )


def get_qualified_name(node: Node) -> str:
    """
    Include enclosing classes/functions so repeated method names remain
    distinct.
    """
    names = []
    while node:
        if node.type in {'class_definition', 'function_definition'}:
            names.append(node.child_by_field_name('name').text.decode('utf8'))
        node = node.parent
    return '.'.join(reversed(names))
