"""
Shared index settings, source fingerprints and manifest validation.
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
GRAPH_SCHEMA = 2  # Class symbols and direct inheritance relationships.
TOOL_DESCRIPTION_VERSION = "source-evidence-v4"
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
