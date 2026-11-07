import logging
import os
import sys

import jedi
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from src.embeddings.file_processor import get_all_python_files
from src.neo4j_connection import graph_connection

# The code is currently in a prototype state and may not handle all edge
# cases.
# It is designed to build a graph of Python functions and their calls in
# a codebase.


def build_graph(codebase_path, *, file_paths=None, build_id="manual"):
    codebase_path = os.path.realpath(codebase_path)
    file_paths = (
        get_all_python_files(codebase_path)
        if file_paths is None
        else file_paths
    )
    counts = {
        "files": len(file_paths),
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
    indexed_files = {os.path.realpath(path) for path in file_paths}
    locations = {}
    class_locations = {}
    project = jedi.Project(
        codebase_path,
        environment_path=sys.executable,
        added_sys_path=[os.path.join(codebase_path, "src")],
    )
    # Initialize Tree-sitter Python parser
    py_language = Language(tspython.language())
    parser = Parser(py_language)
    driver, database = graph_connection()
    session = None

    try:
        session = driver.session(database=database)
        # Replace only this repository's Codeaware nodes, never the
        # whole database.
        session.run(
            "MATCH (n:Codeaware {repository: $repo}) DETACH DELETE n",
            repo=codebase_path,
        ).consume()
        session.run(
            "CREATE (:Codeaware:CodeawareIndex {repository: $repo, "
            "build_id: $build})",
            repo=codebase_path,
            build=build_id,
        ).consume()

        # Walk through codebase directory
        for files in file_paths:
            if not files.endswith(".py"):
                continue
            filepath = os.path.abspath(files)

            # Read file in both binary mode (for tree-sitter) and text
            # mode (for content extraction)
            code = open(filepath, "rb").read()
            tree = parser.parse(code)
            if tree.root_node.has_error:
                raise ValueError(
                    f"Cannot index Python syntax errors in {filepath}"
                )

            with open(filepath, "r", encoding="utf-8") as f:
                code_lines = f.readlines()

            # Create File node
            session.run(
                "MERGE (f:Codeaware:File {path:$path, repository:$repo})",
                path=filepath,
                repo=codebase_path,
            ).consume()

            # Classes reuse the existing definition identity, location
            # and content storage.
            query = py_language.query(
                """
                    [(function_definition) (class_definition)] @definition
                """
            )
            captures = query.captures(tree.root_node)

            for node in captures.get("definition", []):

                # Extract function name from the name field of
                # function_definition
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_name = name_node.text.decode('utf8')
                    start_line = (
                        node.start_point[0] + 1
                    )  # Convert to 1-based indexing
                    end_line = (
                        node.end_point[0] + 1
                    )  # Get end line of the function

                    # Extract the full function content
                    function_content = ''.join(
                        code_lines[start_line - 1 : end_line]
                    )

                    # Add identity without changing the existing
                    # Function/File graph structure.
                    qualified_name = get_qualified_name(node)
                    relative_path = os.path.relpath(filepath, codebase_path)
                    symbol_id = (
                        f"{relative_path}:{start_line}:{qualified_name}"
                    )
                    name_line, name_column = name_node.start_point
                    name_column = len(
                        code.splitlines()[name_line][:name_column].decode(
                            "utf8"
                        )
                    )
                    is_class = node.type == "class_definition"
                    definition_locations = (
                        class_locations if is_class else locations
                    )
                    definition_locations[
                        (filepath, name_line + 1, name_column)
                    ] = symbol_id
                    counts["symbols"] += 1
                    counts["classes"] += int(is_class)
                    label = (
                        "Class" if is_class else "Function"
                    )  # Fixed internal labels, never user input.

                    # Create Function node and link to File with start,
                    # end line and full content
                    session.run(
                        f"MERGE (func:Codeaware:{label} "
                        "{symbol_id:$id, repository:$repo}) "
                        "SET func.name = $fn, func.file = $fp, func.path "
                        "= $path, "
                        "func.qualified_name = $qualified, func.kind = $kind, "
                        "func.start_line = $sln, func.end_line = $eln, "
                        "func.content = $content "
                        "WITH func "
                        "MATCH (file:Codeaware:File {path:$fp, "
                        "repository:$repo}) "
                        "MERGE (func)-[:DEFINED_IN]->(file)",
                        fn=func_name,
                        fp=filepath,
                        sln=start_line,
                        eln=end_line,
                        content=function_content,
                        id=symbol_id,
                        repo=codebase_path,
                        path=relative_path,
                        qualified=qualified_name,
                        kind="class" if is_class else "function",
                    ).consume()

        # Second pass: resolve relationships after all nodes exist.
        for files in file_paths:
            if not files.endswith(".py"):
                continue
            filepath = os.path.abspath(files)
            code = open(filepath, "rb").read()
            tree = parser.parse(code)
            if tree.root_node.has_error:
                raise ValueError(
                    f"Cannot index Python syntax errors in {filepath}"
                )
            # Find all call expressions (both function calls and method
            # calls)
            query = py_language.query(
                """
                    (call function: [
                        (identifier) @function.call
                        (attribute attribute: (identifier) @method.call)
                    ])
                """
            )

            captures = query.captures(tree.root_node)
            script = jedi.Script(
                code=code.decode("utf8"), path=filepath, project=project
            )
            import_nodes = (
                py_language.query(
                    "[(import_statement) (import_from_statement)] @import"
                )
                .captures(tree.root_node)
                .get("import", [])
            )
            process_imports(
                import_nodes,
                script,
                session,
                filepath,
                codebase_path,
                indexed_files,
                counts,
                code.decode("utf8").splitlines(),
            )
            class_nodes = (
                py_language.query("(class_definition) @class")
                .captures(tree.root_node)
                .get("class", [])
            )
            process_inheritance(
                class_nodes,
                script,
                session,
                filepath,
                codebase_path,
                class_locations,
                counts,
                code.decode("utf8").splitlines(),
            )

            # Separate processing for function and method calls
            process_function_calls(
                captures.get("function.call", []),
                script,
                session,
                filepath,
                codebase_path,
                locations,
                counts,
                code.decode("utf8").splitlines(),
            )
            process_method_calls(
                captures.get("method.call", []),
                script,
                session,
                filepath,
                codebase_path,
                locations,
                counts,
                code.decode("utf8").splitlines(),
            )

        return counts
    finally:
        if session is not None:
            session.close()


def process_imports(
    nodes,
    script,
    session,
    filepath,
    repository_root,
    indexed_files,
    counts,
    source_lines,
):
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
            line, column = name.end_point
            column = len(
                source_lines[line].encode("utf8")[:column].decode("utf8")
            )
            definitions = script.goto(
                line=line + 1, column=column, follow_imports=True
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
            line, column = name.start_point
            column = len(
                source_lines[line].encode("utf8")[:column].decode("utf8")
            )
            session.run(
                "MATCH (source:Codeaware:File {path:$source, "
                "repository:$repo}), (target:Codeaware:File "
                "{path:$target, repository:$repo}) "
                "MERGE (source)-[r:IMPORTS {line:$line, column:$column}]"
                "->(target) "
                "SET r.module = $module, r.name = $name, r.alias = $alias",
                source=filepath,
                target=os.path.realpath(target),
                repo=repository_root,
                module=module.text.decode("utf8") if module else None,
                name=name.text.decode("utf8"),
                alias=alias,
                line=line + 1,
                column=column,
            ).consume()
            counts["import_edges"] += 1


def process_inheritance(
    nodes,
    script,
    session,
    filepath,
    repository_root,
    locations,
    counts,
    source_lines,
):
    """
    Resolve explicit named bases only; omit dynamic, external or
    ambiguous bases.
    """
    for node in nodes:
        bases = node.child_by_field_name("superclasses")
        if bases is None:
            continue
        name = node.child_by_field_name("name")
        line, column = name.start_point
        column = len(source_lines[line].encode("utf8")[:column].decode("utf8"))
        child_id = locations[(filepath, line + 1, column)]
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
            line, column = base.end_point
            column = len(
                source_lines[line].encode("utf8")[:column].decode("utf8")
            )
            definitions = script.infer(line=line + 1, column=column)
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
                "MATCH (child:Codeaware:Class {symbol_id:$child, "
                "repository:$repo}), "
                "(parent:Codeaware:Class {symbol_id:$parent, "
                "repository:$repo}) "
                "MERGE (child)-[:INHERITS_FROM "
                "{base_index:$position}]->(parent)",
                child=child_id,
                parent=parent_id,
                repo=repository_root,
                position=position,
            ).consume()
            counts["inheritance_edges"] += 1


def process_function_calls(
    nodes,
    script,
    session,
    filepath,
    repository_root,
    locations,
    counts,
    source_lines,
    call_type="function",
):
    """
    Process direct function calls and store them in the database.

    This function identifies direct function calls (in 'function()'
    format) in Python code.
    It determines the calling and called function context and creates
    corresponding
    CALLS relationships in the Neo4j graph database.

    Args:
        nodes (list): List of Tree-sitter AST nodes representing
            function calls
        script (jedi.Script): Jedi Script object for code analysis
        session (neo4j.Session): Active Neo4j database connection
        filepath (str): Absolute path to the analyzed Python file

    The CALLS relationships created in the database include:
    - line: Line number of the call
    - call_type: Type of call ("function")
    """
    for node in nodes:
        # Find the enclosing executing function body (not a default
        # argument).
        parent = node.parent
        while parent:
            if parent.type == 'function_definition':
                body = parent.child_by_field_name('body')
                if body.start_byte <= node.start_byte < body.end_byte:
                    break
            parent = parent.parent
        if parent is None:
            counts["out_of_scope_calls"] += 1
            continue

        caller_node = parent.child_by_field_name('name')
        line, column = caller_node.start_point
        column = len(source_lines[line].encode('utf8')[:column].decode('utf8'))
        caller_id = locations[(filepath, line + 1, column)]
        line, column = node.end_point
        column = len(source_lines[line].encode('utf8')[:column].decode('utf8'))
        definitions = script.infer(line=line + 1, column=column)
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
        call_line, call_column = call_node.start_point
        call_column = len(
            source_lines[call_line].encode('utf8')[:call_column].decode('utf8')
        )
        session.run(
            "MATCH (caller:Codeaware:Function {symbol_id:$c1, "
            "repository:$repo}), "
            "(callee:Codeaware:Function {symbol_id:$c2, repository:$repo}) "
            "MERGE (caller)-[:CALLS {line:$ln, column:$col, path:$path, "
            "call_type:$ct}]->(callee)",
            c1=caller_id,
            c2=callee_id,
            repo=repository_root,
            ln=call_line + 1,
            col=call_column,
            path=os.path.relpath(filepath, repository_root),
            ct=call_type,
        ).consume()
        counts["calls"] += 1


def process_method_calls(
    nodes,
    script,
    session,
    filepath,
    repository_root,
    locations,
    counts,
    source_lines,
):
    """
    Process method calls and store them in the database.

    This function identifies method calls (in 'object.method()' format)
    in Python code.
    It extracts the called method, determines the surrounding function
    context and
    creates corresponding CALLS relationships in the Neo4j graph
    database.

    Args:
        nodes (list): List of Tree-sitter AST nodes representing method
            calls
        script (jedi.Script): Jedi Script object for code analysis and
            resolution
        session (neo4j.Session): Active Neo4j database connection
        filepath (str): Absolute path to the analyzed Python file

    The CALLS relationships created in the database include:
    - line: Line number of the call
    - call_type: Type of call ("method")
    """
    # Both original helpers use the same resolution and identity checks.
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


def get_qualified_name(node):
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
