import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import jedi
from neo4j import GraphDatabase
from flask import get_all_python_files


def build_graph(codebase_path):
    # Initialize Tree-sitter Python parser
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    # Connect to Neo4j (default user/password)
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    session = driver.session()

    # Clear existing graph
    session.run("MATCH (n) DETACH DELETE n")

    # Walk through codebase directory
    for files in get_all_python_files(codebase_path):
        if not files.endswith(".py"):
            continue
        filepath = os.path.abspath(files)
        code = open(filepath, "rb").read()
        tree = parser.parse(code)

        # Create File node
        session.run("MERGE (f:File {path:$path})", path=filepath)

        # Find all function definitions in this file
        query = PY_LANGUAGE.query("""
                (function_definition name: (identifier) @function.def)
            """)
        captures = query.captures(tree.root_node)
        for _, nodes in captures.items():
            for node in nodes:
                # Extract function name and definition line
                func_name = node.text.decode('utf8')
                def_line = node.start_point[0] + 1
                # Create Function node and link to File
                session.run(
                    "MERGE (func:Function {name:$fn, file:$fp}) "
                    "SET func.line = $ln "
                    "WITH func "
                    "MATCH (file:File {path:$fp}) "
                    "MERGE (func)-[:DEFINED_IN]->(file)",
                    fn=func_name, fp=filepath, ln=def_line
                )

    # Second pass: resolve calls and create CALLS edges
    for files in get_all_python_files(codebase_path):
        if not files.endswith(".py"):
            continue
        filepath = os.path.abspath(files)
        code = open(filepath, "rb").read()
        tree = parser.parse(code)
        # Find all call expressions (both function calls and method calls)
        query = PY_LANGUAGE.query("""
                (call function: [(identifier) @function.call
                               (attribute attribute: (identifier) @method.call)])
            """)

        captures = query.captures(tree.root_node)
        script = jedi.Script(path=filepath)
        
        # Separate processing for function and method calls
        process_function_calls(captures.get("function.call", []), script, session, filepath)
        process_method_calls(captures.get("method.call", []), script, session, filepath)

    session.close()
    driver.close()

def process_function_calls(nodes, script, session, filepath):
    """Process direct function calls and store them in the database.

    This function identifies direct function calls (in 'function()' format) in Python code.
    It determines the calling and called function context and creates corresponding
    CALLS relationships in the Neo4j graph database.

    Args:
        nodes (list): List of Tree-sitter AST nodes representing function calls
        script (jedi.Script): Jedi Script object for code analysis
        session (neo4j.Session): Active Neo4j database connection
        filepath (str): Absolute path to the analyzed Python file

    The CALLS relationships created in the database include:
    - line: Line number of the call
    - call_type: Type of call ("function")
    """
    for node in nodes:
        call_name = node.text.decode('utf8')
        call_line, call_col = node.start_point
        call_line += 1  # convert to 1-based
        
        # Determine caller function by climbing ancestors
        parent = node
        while parent and parent.type != 'function_definition':
            parent = parent.parent
        if parent:
            caller_node = parent.child_by_field_name('name')
            caller_name = caller_node.text.decode('utf8')
            # Resolve callee using Jedi
            definitions = script.infer(line=call_line, column=call_col)
            for d in definitions:
                if d.name == call_name and d.type in ('function', 'method') and d.module_path:
                    callee_name = d.name
                    callee_file = os.path.abspath(d.module_path)
                    # Create CALLS edge with line property and call_type
                    session.run(
                        "MATCH (caller:Function {name:$c1, file:$f1}), "
                        "(callee:Function {name:$c2, file:$f2}) "
                        "MERGE (caller)-[:CALLS {line:$ln, call_type:$ct}]->(callee)",
                        c1=caller_name, f1=filepath, c2=callee_name, f2=callee_file, 
                        ln=call_line, ct="function"
                    )

def process_method_calls(nodes, script, session, filepath):
    """Process method calls and store them in the database.

    This function identifies method calls (in 'object.method()' format) in Python code.
    It extracts the called method, determines the surrounding function context and
    creates corresponding CALLS relationships in the Neo4j graph database.

    Args:
        nodes (list): List of Tree-sitter AST nodes representing method calls
        script (jedi.Script): Jedi Script object for code analysis and resolution
        session (neo4j.Session): Active Neo4j database connection
        filepath (str): Absolute path to the analyzed Python file

    The CALLS relationships created in the database include:
    - line: Line number of the call
    - call_type: Type of call ("method")
    """
    for node in nodes:
        call_name = node.text.decode('utf8')
        call_line, call_col = node.start_point
        call_line += 1  # convert to 1-based
        
        # Determine caller function by climbing ancestors
        parent = node
        while parent and parent.type != 'function_definition':
            parent = parent.parent
        if parent:
            caller_node = parent.child_by_field_name('name')
            caller_name = caller_node.text.decode('utf8')
            # Resolve callee using Jedi
            definitions = script.infer(line=call_line, column=call_col)
            for d in definitions:
                if d.name == call_name and d.type in ('function', 'method') and d.module_path:
                    callee_name = d.name
                    callee_file = os.path.abspath(d.module_path)
                    # Create CALLS edge with line property and call_type
                    session.run(
                        "MATCH (caller:Function {name:$c1, file:$f1}), "
                        "(callee:Function {name:$c2, file:$f2}) "
                        "MERGE (caller)-[:CALLS {line:$ln, call_type:$ct}]->(callee)",
                        c1=caller_name, f1=filepath, c2=callee_name, f2=callee_file, 
                        ln=call_line, ct="method"
                    )

if __name__ == "__main__":
    build_graph("../../")
