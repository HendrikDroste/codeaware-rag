import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

def parse_file(file_path):
    parser = Parser(PY_LANGUAGE)
    with open(file_path, 'r') as file:
        code = file.read()
    tree = parser.parse(bytes(code, "utf8"))
    return tree
