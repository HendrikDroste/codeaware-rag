"""Compare native Codex repository tools with Codeaware MCP added."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from src.index_state import (
    CONTEXT_GUIDANCE,
    DEFAULT_CONFIG,
    TOOL_DESCRIPTION_VERSION,
    read_manifest,
)
from src.retrievers.validate_retriever import load_benchmark

PROJECT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT / "data/benchmarks/flask"
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
PROMPT = (
    "Answer this question about the local repository. Cite source paths "
    "and line numbers, and state any uncertainty. You may use the available "
    "repository tools. Do not edit files, use the web, inspect other "
    "repositories, or delegate to other agents.\n\n"
)
PROMPT_VERSION = "repository-qa-v4"
PROMPTS = {
    "auto": PROMPT,
    "guided": PROMPT + CONTEXT_GUIDANCE + "\n\n",
}


def summarize_events(output):
    """Read the single turn's reported token usage."""
    events = [json.loads(line) for line in output.splitlines() if line.strip()]
    completed = [
        event for event in events if event.get("type") == "turn.completed"
    ]
    summary = {field: None for field in TOKEN_FIELDS}
    summary.update(total_tokens=None, uncached_input_tokens=None)
    summary["usage_status"] = "ambiguous" if len(completed) > 1 else "missing"
    if len(completed) == 1 and isinstance(completed[0].get("usage"), dict):
        usage = completed[0]["usage"]
        for field in TOKEN_FIELDS:
            value = usage.get(field)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"Invalid reported {field}")
            summary[field] = value
        incoming, outgoing = summary["input_tokens"], summary["output_tokens"]
        if incoming is not None and outgoing is not None:
            summary["total_tokens"] = incoming + outgoing
            summary["usage_status"] = "reported"
        cached = summary["cached_input_tokens"]
        if incoming is not None and cached is not None:
            if cached > incoming:
                raise ValueError("Cached input exceeds total input tokens")
            summary["uncached_input_tokens"] = incoming - cached

    # Started/completed events share item IDs; count each item once.
    items = {
        event["item"]["id"]: event["item"]
        for event in events
        if event.get("type", "").startswith("item.") and "item" in event
    }
    mcp = [
        item for item in items.values() if item.get("type") == "mcp_tool_call"
    ]
    summary["command_calls"] = sum(
        item.get("type") == "command_execution" for item in items.values()
    )
    summary["mcp_calls"] = len(mcp)
    summary["codeaware_calls"] = sum(
        item.get("server") == "codeaware" for item in mcp
    )
    completed_ids = {
        event["item"]["id"]
        for event in events
        if event.get("type") == "item.completed"
    }
    summary.update(
        mcp_errors=0,
        code_excerpt_calls=0,
        code_excerpts=0,
        excerpt_truncations=0,
    )
    for item in mcp:
        result = item.get("result")
        result = result if isinstance(result, dict) else {}
        failed = (
            item.get("error") is not None
            or item.get("status") in {"failed", "cancelled"}
            or result.get("is_error") is True
            or result.get("isError") is True
        )
        summary["mcp_errors"] += int(failed)
        if (
            failed
            or item["id"] not in completed_ids
            or item.get("server") != "codeaware"
        ):
            continue
        payload = result.get("structured_content") or result.get(
            "structuredContent"
        )
        if payload is None:
            for block in result.get("content", []):
                if block.get("type") != "text":
                    continue
                try:
                    payload = json.loads(block["text"])
                except (ValueError, KeyError):
                    continue
                if isinstance(payload, dict):
                    break
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            continue
        excerpts = [
            row["code_excerpt"]
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("code_excerpt"), dict)
            and row["code_excerpt"].get("content")
        ]
        summary["code_excerpt_calls"] += int(bool(excerpts))
        summary["code_excerpts"] += len(excerpts)
        summary["excerpt_truncations"] += sum(
            excerpt.get("truncated") is True for excerpt in excerpts
        )
    messages = [
        event["item"].get("text", "")
        for event in events
        if event.get("type") == "item.completed"
        and event.get("item", {}).get("type") == "agent_message"
    ]
    summary["answer"] = messages[-1] if messages else ""
    return summary


def command(args, variant):
    """Build both conditions with the same model and native tools."""
    result = [
        args.codex,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--model",
        args.model,
        "--cd",
        str(args.repo),
        "-c",
        "mcp_servers={}",
        "-c",
        'web_search="disabled"',
        "-c",
        f"model_reasoning_effort={json.dumps(args.reasoning_effort)}",
    ]
    for feature in (
        "apps",
        "plugins",
        "memories",
        "multi_agent",
        "browser_use",
    ):
        result.extend(["--disable", feature])
    if variant == "codeaware":
        arguments = [
            "-m",
            "src.mcp_server",
            "--repo",
            str(args.repo),
            "--index-dir",
            str(args.index_dir),
            "--config",
            str(args.config),
        ]
        server = (
            f"command={json.dumps(sys.executable)}, "
            f"args={json.dumps(arguments)}, "
            f"cwd={json.dumps(str(PROJECT))}, required=true, "
            'env_vars=["NEO4J_PASSWORD", "NEO4J_URI", "NEO4J_USER", '
            '"NEO4J_DATABASE", "HF_HOME", "HF_HUB_OFFLINE"]'
        )
        result.extend(["-c", "mcp_servers.codeaware={" + server + "}"])
    # Pass the prompt through stdin, without shell interpolation.
    return result + ["-"]


def run_question(args, variant, question, directory):
    """Keep raw logs even if the host fails or omits token usage."""
    directory.mkdir()
    prompt = PROMPTS[args.tool_policy] + question
    started = time.monotonic()
    try:
        process = subprocess.run(
            command(args, variant),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        output, errors, code = (
            process.stdout,
            process.stderr,
            process.returncode,
        )
        status = "completed" if code == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        errors = exc.stderr or b""
        output = (
            output.decode(errors="replace")
            if isinstance(output, bytes)
            else output
        )
        errors = (
            errors.decode(errors="replace")
            if isinstance(errors, bytes)
            else errors
        )
        code, status = None, "timeout"
    (directory / "events.jsonl").write_text(output, encoding="utf-8")
    (directory / "stderr.log").write_text(errors, encoding="utf-8")
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    try:
        summary = summarize_events(output)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        summary = {"usage_status": "invalid", "trace_error": str(exc)}
    (directory / "answer.md").write_text(
        summary.pop("answer", ""), encoding="utf-8"
    )
    if variant == "baseline" and summary.get("mcp_calls", 0):
        status = "invalid_baseline"
    return {
        "variant": variant,
        "status": status,
        "exit_code": code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--questions", nargs="+", type=int, default=[16])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--variant", choices=("baseline", "codeaware", "both"), default="both"
    )
    parser.add_argument(
        "--tool-policy",
        choices=("auto", "guided"),
        default="auto",
        help="Neutral tool choice (auto) or conditional workflow guidance",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve(strict=True)
    args.config = args.config.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    data = load_benchmark(args.benchmark, args.repo)
    if (
        args.timeout < 1
        or args.repeats < 1
        or any(not 1 <= number <= len(data) for number in args.questions)
    ):
        parser.error(
            "Use positive timeout/repeats and valid one-based questions"
        )
    if len(set(args.questions)) != len(args.questions):
        parser.error("Question numbers must be unique")
    if args.output_dir.is_relative_to(args.repo):
        parser.error("Store run output outside the benchmarked repository")
    if (args.repo / ".codex/config.toml").exists():
        parser.error("Use a checkout without project Codex configuration")
    build_id = None
    if args.variant != "baseline":
        if args.index_dir is None:
            parser.error("--index-dir is required for Codeaware runs")
        args.index_dir = args.index_dir.resolve()
        if args.index_dir.is_relative_to(args.repo):
            parser.error("Use an index outside the benchmarked repository")
        build_id = read_manifest(args.repo, args.index_dir, args.config)[
            "build_id"
        ]
    version = subprocess.run(
        [args.codex, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        **data.attrs["benchmark"],
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "codex_version": version,
        "index_build_id": build_id,
        "usage_metric_version": "codex-turn-usage-v1",
        "tool_metric_version": "codeaware-tool-usage-v1",
        "tool_policy": args.tool_policy,
        "prompt_version": PROMPT_VERSION,
        "tool_description_version": TOOL_DESCRIPTION_VERSION,
        "timeout_seconds": args.timeout,
        "questions": args.questions,
        "repeats": args.repeats,
        "prompt_prefix": PROMPTS[args.tool_policy],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    rows = []
    schedule = [
        (repeat, pair, number)
        for repeat in range(args.repeats)
        for pair, number in enumerate(args.questions)
    ]
    for repeat, pair, number in schedule:
        variants = (
            ["baseline", "codeaware"]
            if args.variant == "both"
            else [args.variant]
        )
        if (pair + repeat) % 2:
            # Reverse pair order across questions and repetitions.
            variants.reverse()
        directory = args.output_dir
        if args.repeats > 1:
            directory = directory / f"repeat-{repeat + 1:02d}"
            directory.mkdir(exist_ok=True)
        for variant in variants:
            load_benchmark(args.benchmark, args.repo)
            row = run_question(
                args,
                variant,
                data.iloc[number - 1]["Question"],
                directory / f"q{number:02d}-{variant}",
            )
            try:
                load_benchmark(args.benchmark, args.repo)
            except ValueError as exc:
                row.update(status="invalid_corpus", corpus_error=str(exc))
            row.update(
                question_id=number,
                repetition=repeat + 1,
                question=data.iloc[number - 1]["Question"],
                **{
                    key: value
                    for key, value in metadata.items()
                    if key not in {"questions", "prompt_prefix"}
                },
            )
            rows.append(row)
            keys = list(dict.fromkeys(key for item in rows for key in item))
            with (args.output_dir / "usage.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            if "corpus_error" in row:
                raise ValueError(row["corpus_error"])
            print(
                f"R{repeat + 1} Q{number} {variant}: {row['status']}, "
                f"usage={row['usage_status']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
