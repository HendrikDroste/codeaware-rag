"""Test host usage accounting without making model or network calls."""

import argparse
import csv
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import benchmark_agent as benchmark


def trace(usage=None, mcp=False):
    events = [
        {
            "type": "item.started",
            "item": {"id": "1", "type": "command_execution"},
        },
        {
            "type": "item.completed",
            "item": {"id": "1", "type": "command_execution"},
        },
        {
            "type": "item.completed",
            "item": {"id": "2", "type": "agent_message", "text": "Answer"},
        },
    ]
    if mcp:
        events.append(
            {
                "type": "item.completed",
                "item": {
                    "id": "3",
                    "type": "mcp_tool_call",
                    "server": "codeaware",
                },
            }
        )
    if usage is not None:
        events.append({"type": "turn.completed", "usage": usage})
    return "\n".join(json.dumps(event) for event in events)


def test_cached_and_reasoning_tokens_are_not_double_counted():
    usage = {
        "input_tokens": 100,
        "cached_input_tokens": 80,
        "output_tokens": 25,
        "reasoning_output_tokens": 15,
    }
    summary = benchmark.summarize_events(trace(usage, mcp=True))
    assert summary["total_tokens"] == 125
    assert summary["uncached_input_tokens"] == 20
    assert summary["reasoning_output_tokens"] == 15
    assert summary["command_calls"] == 1
    assert summary["codeaware_calls"] == summary["mcp_calls"] == 1
    assert summary["answer"] == "Answer"


def test_missing_usage_is_not_zero_and_multiple_turns_are_ambiguous():
    summary = benchmark.summarize_events(trace())
    assert summary["total_tokens"] is None
    assert summary["usage_status"] == "missing"
    usage = {"input_tokens": 10, "output_tokens": 2}
    output = (
        trace(usage)
        + "\n"
        + json.dumps({"type": "turn.completed", "usage": usage})
    )
    summary = benchmark.summarize_events(output)
    assert summary["usage_status"] == "ambiguous"
    assert summary["total_tokens"] is None
    summary = benchmark.summarize_events(trace(usage))
    assert summary["total_tokens"] == 12
    assert summary["cached_input_tokens"] is None
    assert summary["reasoning_output_tokens"] is None


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": -1},
        {"input_tokens": True},
        {"input_tokens": 10, "cached_input_tokens": 11},
    ],
)
def test_invalid_usage_is_rejected(usage):
    with pytest.raises(ValueError):
        benchmark.summarize_events(trace(usage))


@pytest.fixture
def args(tmp_path):
    return argparse.Namespace(
        codex="codex",
        model="explicit-model",
        repo=tmp_path / "repo",
        index_dir=tmp_path / "index",
        config=tmp_path / "config.yaml",
        reasoning_effort="medium",
        timeout=1,
    )


def test_conditions_have_the_same_settings_except_codeaware(args):
    baseline = benchmark.command(args, "baseline")
    enhanced = benchmark.command(args, "codeaware")
    assert enhanced[: len(baseline) - 1] == baseline[:-1]
    assert "--ignore-user-config" in baseline
    assert baseline[baseline.index("--sandbox") + 1] == "read-only"
    assert "mcp_servers={}" in baseline
    assert not any("mcp_servers.codeaware" in value for value in baseline)
    config = next(
        value
        for value in enhanced
        if value.startswith("mcp_servers.codeaware=")
    )
    server = tomllib.loads(config)["mcp_servers"]["codeaware"]
    assert server["command"] == sys.executable
    assert server["required"] is True
    assert str(args.index_dir) in server["args"]
    assert "NEO4J_PASSWORD" in server["env_vars"]
    assert "env" not in server  # Credential values are not embedded in traces.


@pytest.mark.parametrize(
    "case", ["success", "failure", "timeout", "invalid", "baseline_mcp"]
)
def test_run_keeps_raw_traces_and_failed_usage_unknown(
    args, tmp_path, monkeypatch, case
):
    output = trace(
        {"input_tokens": 10, "output_tokens": 2}, mcp=case == "baseline_mcp"
    )

    def execute(*positional, **kwargs):
        assert kwargs["input"].endswith("Question")
        if case == "timeout":
            raise subprocess.TimeoutExpired(
                "codex", 1, output=b"", stderr=b"timeout"
            )
        return SimpleNamespace(
            stdout="not json" if case == "invalid" else output,
            stderr="diagnostic",
            returncode=1 if case == "failure" else 0,
        )

    monkeypatch.setattr(benchmark.subprocess, "run", execute)
    directory = tmp_path / "run"
    result = benchmark.run_question(args, "baseline", "Question", directory)
    assert (directory / "events.jsonl").exists()
    assert (directory / "stderr.log").exists()
    assert (directory / "prompt.txt").read_text().endswith("Question")
    if case == "success":
        assert result["status"] == "completed" and result["total_tokens"] == 12
        assert (directory / "answer.md").read_text() == "Answer"
    elif case == "failure":
        assert result["status"] == "failed" and result["total_tokens"] == 12
    elif case == "timeout":
        assert result["status"] == "timeout" and result["total_tokens"] is None
    elif case == "invalid":
        assert (
            result["usage_status"] == "invalid"
            and "total_tokens" not in result
        )
    else:
        assert result["status"] == "invalid_baseline"


@pytest.mark.parametrize("stale", [False, True])
def test_paired_runner_exports_usage_and_alternates_order(
    tmp_path, monkeypatch, stale
):
    repository = tmp_path / "repo"
    repository.mkdir()
    data = pd.DataFrame({"Question": ["first", "second"]})
    data.attrs["benchmark"] = {
        "dataset_id": "fixture",
        "revision": "pinned",
        "metric_version": "v2",
    }
    checks = []

    def load(*args):
        checks.append(None)
        if stale and len(checks) == 3:
            raise ValueError("Changed corpus")
        return data

    monkeypatch.setattr(benchmark, "load_benchmark", load)
    monkeypatch.setattr(
        benchmark, "read_manifest", lambda *args: {"build_id": "build"}
    )
    calls = []

    def execute(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(stdout="codex fixture")
        enhanced = any(
            value.startswith("mcp_servers.codeaware=") for value in command
        )
        calls.append("codeaware" if enhanced else "baseline")
        return SimpleNamespace(
            stdout=trace(
                {"input_tokens": 100, "output_tokens": 20}, mcp=enhanced
            ),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(benchmark.subprocess, "run", execute)
    output = tmp_path / "output"
    arguments = [
        "--repo",
        str(repository),
        "--index-dir",
        str(tmp_path / "index"),
        "--output-dir",
        str(output),
        "--model",
        "explicit-model",
        "--questions",
        "1",
        "2",
    ]
    if stale:
        with pytest.raises(ValueError, match="Changed corpus"):
            benchmark.main(arguments)
    else:
        benchmark.main(arguments)
    assert calls == (
        ["baseline"]
        if stale
        else ["baseline", "codeaware", "codeaware", "baseline"]
    )
    with (output / "usage.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == (1 if stale else 4)
    assert {row["total_tokens"] for row in rows} == {"120"}
    if stale:
        assert rows[0]["status"] == "invalid_corpus"
    assert {row["model"] for row in rows} == {"explicit-model"}
    assert {row["revision"] for row in rows} == {"pinned"}
    assert {row["cached_input_tokens"] for row in rows} == {""}
    assert json.loads((output / "run.json").read_text())["questions"] == [1, 2]
