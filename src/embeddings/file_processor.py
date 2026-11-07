"""Discover repository Python files and retain exact chunk locations."""

import os
from pathlib import Path


def get_all_python_files(source_dir, *, exclude_dirs=()):
    root = os.path.realpath(source_dir)
    python_files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
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
            start = chunk.metadata.pop("start_index")
            if start < 0 or content[
                start : start + len(chunk.page_content)
            ] != (chunk.page_content):
                raise ValueError(f"Cannot locate a chunk in {file_path}")
            first = content[:start].count("\n") + 1
            chunk.metadata.update(
                line_start=first,
                line_end=first + chunk.page_content.count("\n"),
            )
            chunks.append(chunk)
    return chunks
