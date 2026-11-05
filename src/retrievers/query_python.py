"""Read-only Jedi queries over files belonging to the current index."""

import sys
from pathlib import Path
from threading import Lock

JEDI_LOCK = Lock()  # Jedi's parser cache and settings are process-global.


def query_python(
    operation,
    path,
    *,
    repository_root,
    index_dir,
    files,
    parent=None,
    line=None,
    column=None,
    include_declaration=False,
    limit=20,
):
    import jedi

    root = Path(repository_root)
    if path not in files:
        raise ValueError("path must identify a Python file in this index")
    if operation != "symbols":
        lines = (root / path).read_text(encoding="utf-8").splitlines()
        if (
            isinstance(line, bool)
            or not isinstance(line, int)
            or not 1 <= line <= len(lines)
            or isinstance(column, bool)
            or not isinstance(column, int)
            or not 0 <= column <= len(lines[line - 1])
        ):
            raise ValueError("Invalid one-based line or Unicode column")
    if parent is not None and not parent.strip():
        raise ValueError("parent must be a qualified scope name or null")

    with JEDI_LOCK:
        previous = jedi.settings.cache_directory
        try:
            jedi.settings.cache_directory = str(Path(index_dir) / "jedi-cache")
            project = jedi.Project(
                root,
                environment_path=sys.executable,
                added_sys_path=[str(root / "src")],
            )
            script = jedi.Script(path=root / path, project=project)
            if operation == "symbols":
                names = script.get_names(
                    all_scopes=True, definitions=True, references=False
                )
            elif operation == "definition":
                names = script.goto(line, column, follow_imports=True)
            elif operation == "references":
                names = script.get_references(
                    line, column, include_builtins=False, scope="project"
                )
                definitions = script.goto(line, column, follow_imports=True)
                if include_declaration:
                    names.extend(definitions)
                else:
                    declarations = {
                        (n.module_path, n.line, n.column) for n in definitions
                    }
                    names = [
                        n
                        for n in names
                        if (n.module_path, n.line, n.column)
                        not in declarations
                    ]
            else:
                raise ValueError("Unknown internal Python query")

            rows, excluded = {}, 0
            for name in names:
                if name.module_path is None or name.line is None:
                    excluded += 1
                    continue
                target = Path(name.module_path).resolve()
                if not target.is_relative_to(root):
                    excluded += 1
                    continue
                relative = target.relative_to(root).as_posix()
                if relative not in files:
                    excluded += 1
                    continue
                scopes, scope = [], name.parent()
                while scope is not None and scope.type != "module":
                    scopes.append(scope.name)
                    scope = scope.parent()
                owner = ".".join(reversed(scopes)) or None
                if parent is not None and owner != parent:
                    continue
                key = (relative, name.line, name.column)
                rows[key] = {
                    "name": name.name,
                    "kind": name.type,
                    "parent": owner,
                    "qualified_name": ".".join(
                        part for part in (owner, name.name) if part
                    ),
                    "path": relative,
                    "line": name.line,
                    "column": name.column,
                }
            return [rows[key] for key in sorted(rows)][:limit], excluded
        finally:
            jedi.settings.cache_directory = previous
