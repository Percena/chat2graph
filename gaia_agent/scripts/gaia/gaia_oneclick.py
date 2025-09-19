#!/usr/bin/env python3
"""
Minimal GAIA agent runner compatible with Chat2Graph gaia_agent_bridge_tool.

Implements a 'run' subcommand that writes a JSONL with the first/response
fields and emits a final JSON summary line to stdout for robust parsing.

This is a lightweight port: it doesn't perform actual web research; it
returns a deterministic placeholder or echoes question metadata. Replace
`_produce_answer()` with real logic as needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional


def _produce_answer(task_id: Optional[str], question_hint: Optional[str]) -> str:
    # TODO: replace with real solving logic. For now return a deterministic token.
    # If a question hint exists, return its last non-empty line truncated.
    if question_hint:
        text = str(question_hint).strip()
        for line in reversed(text.splitlines()):
            s = line.strip()
            if s:
                return s[:200]
    return "GAIA_AGENT_PLACEHOLDER_ANSWER"


def cmd_run(args: argparse.Namespace) -> int:
    exp_id: str = args.exp_id
    limit: int = int(args.limit)
    task_id: Optional[str] = args.task_id
    question_hint: Optional[str] = os.getenv("GAIA_AGENT_QUESTION_HINT")

    # Resolve output JSONL path
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
    else:
        repo_root = Path(__file__).resolve().parents[3]
        out_path = repo_root / "data" / f"output_{exp_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional CSV report for compatibility
    repo_root = Path(__file__).resolve().parents[3]
    csv_out = repo_root / "data" / "tmp" / f"report_{exp_id}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    answer = _produce_answer(task_id, question_hint)

    # Write JSONL (single row)
    row = {
        "exp_id": exp_id,
        "task_id": task_id,
        "first_response": answer,
        "final_output": answer,
        "response": answer,
        "dataset_index": 1,
    }
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write CSV best-effort
    try:
        with csv_out.open("w", encoding="utf-8") as cf:
            cf.write("task_id,response\n")
            cf.write(f"{task_id or ''},{answer}\n")
    except Exception:
        pass

    # Emit a final JSON summary line for PTY/PIPE parsing
    summary = {"first_response": answer, "final_output": answer, "exp_id": exp_id}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal GAIA agent runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one evaluation task")
    p_run.add_argument("--exp_id", required=True)
    p_run.add_argument("--limit", default=1)
    p_run.add_argument("--concurrency", default=1)
    p_run.add_argument("--judge_concurrency", default=1)
    p_run.add_argument("--dataset", default="auto")
    p_run.add_argument("--task_id", default=None)
    p_run.add_argument("--output_jsonl", default=None)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

