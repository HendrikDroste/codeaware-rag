"""
Shared settings, tool guidance, fingerprints and manifest validation.
"""

import hashlib
import json
from pathlib import Path

from src.utils import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/app_config.yaml"
EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".codeaware-index",
}
COLLECTION = "codeaware_mcp"
GRAPH_SCHEMA = 3  # Resolved file-to-file import bindings.
TOOL_DESCRIPTION_VERSION = "focused-navigation-v9"
CONTEXT_GUIDANCE = (
    "Choose tools by capability, including native search when it directly "
    "exposes the needed fact. Use semantic search for unknown implementation "
    "locations. find_symbols supports functions, methods and classes, not "
    "assignments or attributes; use targeted text search for those. Treat "
    "remembered names as hypotheses. After one empty exact-name lookup, "
    "switch to behavior or text search instead of trying guessed name "
    "variants. Prefer relevant methods over large class excerpts, whose "
    "first lines may be mostly documentation. Start with 3..5 results and "
    "narrow by path when possible. Query graph relationships only when they "
    "answer an unresolved part of the question. For implementation lookup, "
    "request include_code=true initially; the parameter is not "
    "include_source. "
    "Codeaware returns indexed source with freshness checks: reuse it and "
    "cite returned paths and line ranges directly. Do not reread source "
    "merely to verify returned code. Truncation alone does not require "
    "another read. Make another call only to resolve a specific essential "
    "missing fact or unavailable tool capability. Stop when the question "
    "and citations are supported; continue when essential evidence is "
    "missing. Missing static edges are incomplete evidence, not proof "
    "that a runtime relationship is absent."
)
REBUILD = (
    "Rebuild with python -m src.index_repository, then reconnect the MCP "
    "server."
)


def source_files(repository_root):
    from src.embeddings.file_processor import get_all_python_files

    return get_all_python_files(
        str(repository_root), exclude_dirs=EXCLUDE_DIRS
    )


def file_hashes(repository_root, files=None):
    files = source_files(repository_root) if files is None else files
    return {
        Path(p)
        .relative_to(repository_root)
        .as_posix(): hashlib.sha256(Path(p).read_bytes())
        .hexdigest()
        for p in files
    }


def embedding_settings(config):
    settings = load_config(path=str(config))["models"]["embeddings"]
    return {key: settings[key] for key in ("name", "vendor", "type")}


def write_manifest(index_dir, manifest):
    temporary = index_dir / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(index_dir / "manifest.json")


def read_manifest(repository_root, index_dir, config, *, check_fresh=True):
    try:
        manifest = json.loads(
            (index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["repository_root"] != str(repository_root):
            raise ValueError(
                "Index belongs to another repository; choose the matching "
                "--repo and --index-dir"
            )
        if manifest["state"] != "complete":
            raise ValueError(f"Index build is incomplete. {REBUILD}")
        if manifest.get("graph_schema") != GRAPH_SCHEMA:
            raise ValueError(f"Index graph format is outdated. {REBUILD}")
        if manifest["embeddings"] != embedding_settings(config):
            raise ValueError(
                f"Embedding settings differ from this index. {REBUILD}"
            )
        for key in ("build_id", "file_hashes", "counts", "built_at"):
            manifest[key]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Missing or invalid index manifest. {REBUILD}"
        ) from exc
    if check_fresh and manifest["file_hashes"] != file_hashes(repository_root):
        raise ValueError(f"Repository changed; the index is stale. {REBUILD}")
    return manifest
