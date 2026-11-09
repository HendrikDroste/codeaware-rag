"""Discover repository Python files and retain exact chunk locations."""

import os
from pathlib import Path


def get_all_python_files(source_dir, *, exclude_dirs=()):
    """Return sorted absolute paths to Python files below source_dir.

    Skip named exclusion directories, virtual environments identified
    by pyvenv.cfg, and symlinked files/directories. Include tests and
    special filenames such as __init__.py; callers choose exclusions.
    """
    root = os.path.realpath(source_dir)
    python_files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        # Change dirs in place so os.walk never enters excluded folders.
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in exclude_dirs
            and not os.path.islink(os.path.join(directory, name))
            and not os.path.isfile(os.path.join(directory, name, "pyvenv.cfg"))
        )
        for name in sorted(names):
            path = os.path.join(directory, name)
            if name.endswith(".py") and not os.path.islink(path):
                python_files.append(path)
    return sorted(python_files)


def load_and_split_python_files(
    file_paths, chunk_size=900, chunk_overlap=0, *, repository_root
):
    """Return code chunks with relative paths and inclusive line ranges.

    Read the supplied UTF-8 files inside repository_root and split them
    using Python-aware separators. Size and overlap are in characters;
    chunks are not guaranteed to contain whole functions or classes.
    Raise ValueError if a chunk cannot be located in its source file.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import (
        Language,
        RecursiveCharacterTextSplitter,
    )

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = []
    for file_path in file_paths:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        document = Document(
            page_content=content,
            metadata={"source": path.relative_to(repository_root).as_posix()},
        )
        for chunk in splitter.split_documents([document]):
            # The splitter's offset distinguishes repeated snippets.
            start = chunk.metadata.pop("start_index")
            if start < 0 or content[
                start : start + len(chunk.page_content)
            ] != (chunk.page_content):
                raise ValueError(f"Cannot locate a chunk in {file_path}")
            # Convert the character offset to a one-based source line.
            first = content[:start].count("\n") + 1
            chunk.metadata.update(
                line_start=first,
                line_end=first + chunk.page_content.count("\n"),
            )
            chunks.append(chunk)
    return chunks
