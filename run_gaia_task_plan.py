#!/usr/bin/env python
"""Run selected GAIA tasks with per-task MemFuse toggles."""

# poetry run python run_gaia_task_plan.py --csv-path ./gaia_sample_10.csv --output-tag gaia-agent --limit-num 2

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

from app.core.common.system_env import SystemEnv
from test.benchmark.gaia.run_hf_gaia_test import (
    load_gaia_dataset,
    process_single_sample,
)

# Resolve project root to the repository root (this file lives at repo root)
project_root = os.path.abspath(os.path.dirname(__file__))
_gaia_base_dir = Path(project_root) / "test/benchmark/gaia"
logs_agent_dir = _gaia_base_dir / "logs" / "agent"
logs_external_dir = _gaia_base_dir / "logs" / "external"
artifacts_external_dir = _gaia_base_dir / "artifacts" / "external"
results_dir = _gaia_base_dir / "results"
for d in (logs_agent_dir, logs_external_dir, artifacts_external_dir, results_dir):
    d.mkdir(parents=True, exist_ok=True)


def parse_task_plan(csv_path: Path) -> list[Tuple[str, bool]]:
    """Parse `<task_id>,<memfuse_flag>` rows from a CSV file."""
    plan: list[Tuple[str, bool]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if not row:
                continue
            task_id = row[0].strip()
            flag = row[1].strip().lower() if len(row) > 1 else ""
            if not task_id:
                continue
            if idx == 0 and task_id.lower() in {"task_id", "gaia_task_id"}:
                continue
            if flag not in {"0", "1"}:
                raise ValueError(
                    f"Invalid memfuse flag '{row[1]}' for task '{task_id}'. Use 0 or 1."
                )
            plan.append((task_id, flag == "1"))
    if not plan:
        raise ValueError(f"No runnable tasks found in {csv_path}")
    return plan


def lookup_samples(
    dataset, desired_ids: Iterable[str]
) -> dict[str, dict]:  # type: ignore[no-any-unimported]
    """Collect GAIA samples by task_id."""
    desired = set(desired_ids)
    samples: dict[str, dict] = {}
    for sample in dataset:
        task_id = sample.get("task_id")
        if task_id in desired and task_id not in samples:
            samples[task_id] = dict(sample)
            if len(samples) == len(desired):
                break
    return samples


def extract_ground_truth(sample: dict | None) -> str:
    if not sample:
        return ""
    for key in ("answer", "Final answer", "final_answer", "ground_truth", "Ground Truth"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def run_single_sample(
    sample: dict,
    memfuse_enabled: bool,
    agent_config_path: str,
    project_root_path: str,
    split: str,
) -> dict:
    """Worker wrapper that toggles MemFuse before delegating to process_single_sample."""
    SystemEnv.ENABLE_MEMFUSE = memfuse_enabled
    os.environ["ENABLE_MEMFUSE"] = "1" if memfuse_enabled else "0"
    result = process_single_sample(sample, agent_config_path, project_root_path, split)
    result["memfuse_enabled"] = memfuse_enabled
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, type=Path, help="CSV with task plan.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="GAIA split to load.",
    )
    parser.add_argument(
        "--level",
        default="all",
        choices=["1", "2", "3", "all"],
        help="Optional GAIA difficulty filter before matching tasks.",
    )
    parser.add_argument(
        "--parallel-num",
        type=int,
        default=1,
        help="Maximum concurrent GAIA executions.",
    )
    parser.add_argument(
        "--limit-num",
        type=int,
        default=None,
        help="If set, only take the first N tasks from CSV.",
    )
    parser.add_argument(
        "--output-tag",
        default="csvplan",
        help="Label inserted into the result filename.",
    )
    parser.add_argument(
        "--agent-config",
        default=str(Path(project_root) / "test/benchmark/gaia/gaia_agents_min.yml"),
        help="Path to agent YAML. Default: minimal gaia_agent pipeline",
    )
    parser.add_argument(
        "--runner",
        choices=["agent", "oneclick"],
        default="agent",
        help="Execution backend: 'agent' (Chat2Graph pipeline) or 'oneclick' (call external GAIA agent per task)",
    )
    args = parser.parse_args()

    # Hard-disable subsystems not needed for GAIA text-only oneclick-aligned runs
    os.environ.setdefault("CHAT2GRAPH_DISABLE_MCP", "1")
    os.environ.setdefault("CHAT2GRAPH_DISABLE_KB", "1")

    plan = parse_task_plan(args.csv_path)
    if args.limit_num is not None:
        plan = plan[: max(0, int(args.limit_num))]
    task_ids = [task_id for task_id, _ in plan]

    print(f"✅ Loaded {len(plan)} tasks from {args.csv_path}")
    dataset = load_gaia_dataset(args.split, args.level)
    sample_map = lookup_samples(dataset, task_ids)

    missing = [task_id for task_id in task_ids if task_id not in sample_map]
    if missing:
        print(f"⚠️  Skipping {len(missing)} missing task_id(s): {missing}")

    work_items: list[Tuple[dict, bool]] = [
        (sample_map[task_id], memfuse)  # type: ignore[index]
        for task_id, memfuse in plan
        if task_id in sample_map
    ]
    if not work_items:
        print("❌ No matching GAIA tasks to run. Exiting.")
        return

    agent_config_path = args.agent_config
    results: list[dict] = []

    print(
        f"🚀 Starting evaluation ({args.runner}): split={args.split}, level={args.level}, "
        f"tasks={len(work_items)}, parallel={args.parallel_num}"
    )

    if args.runner == "agent":
        with ProcessPoolExecutor(max_workers=args.parallel_num) as executor:
            future_to_meta = {
                executor.submit(
                    run_single_sample,
                    sample,
                    memfuse_enabled,
                    agent_config_path,
                    project_root,
                    args.split,
                ): (sample["task_id"], memfuse_enabled)
                for sample, memfuse_enabled in work_items
            }

            for future in as_completed(future_to_meta):
                task_id, memfuse_enabled = future_to_meta[future]
                try:
                    result = future.result()
                    # Recompute correctness against ground truth to avoid upstream mismatch
                    gt = extract_ground_truth(sample_map.get(task_id))
                    model_answer = str(result.get("model_answer", "")).strip()
                    is_correct = (model_answer == gt) if gt else bool(result.get("is_correct"))
                    result["is_correct"] = is_correct
                    results.append(result)
                    state = "✅" if is_correct else "❌"
                    print(f"{state} Finished task {task_id} (memfuse={'on' if memfuse_enabled else 'off'})")
                except Exception as exc:
                    print(f"❌ Task {task_id} (memfuse={'on' if memfuse_enabled else 'off'}) failed: {exc}")
                    results.append(
                        {
                            "task_id": task_id,
                            "model_answer": "EXECUTION_ERROR",
                            "reasoning_trace": str(exc),
                            "is_correct": False,
                            "memfuse_enabled": memfuse_enabled,
                        }
                    )
    else:
        # oneclick: call external GAIA agent runner per task (sequential for stability)
        import asyncio
        from app.core.model.task import ToolCallContext
        from app.core.toolkit.system_tool.gaia_agent_bridge_tool import run_gaia_agent

        async def _run_one(task_id: str, memfuse_enabled: bool) -> dict:
            # Compose exp_id: y2c_<ts>_<task>
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_id = f"y2c_{ts}_{task_id}"
            # Build a minimal ToolCallContext
            ctx = ToolCallContext(job_id=task_id, operator_id="CSVOneClick")
            # Dataset auto-derive on external agent side with --task_id
            answer = await run_gaia_agent(
                tool_call_ctx=ctx,
                question_hint=sample_map.get(task_id, {}).get("question") or sample_map.get(task_id, {}).get("Question"),
                exp_id=exp_id,
                limit=1,
                concurrency=1,
                judge_concurrency=1,
                dataset="auto",
                match_task_id=task_id,
            )
            gt = extract_ground_truth(sample_map.get(task_id))
            is_correct = (str(answer).strip() == gt) if gt else False
            return {
                "task_id": task_id,
                "model_answer": str(answer),
                "reasoning_trace": "oneclick",
                "is_correct": is_correct,
                "memfuse_enabled": memfuse_enabled,
                "exp_id": exp_id,
            }

        for sample, memfuse_enabled in work_items:
            tid = sample["task_id"]
            try:
                result = asyncio.run(_run_one(tid, memfuse_enabled))
                results.append(result)
                state = "✅" if result.get("is_correct") else "❌"
                print(f"{state} Finished task {tid} (memfuse={'on' if memfuse_enabled else 'off'})")
                # Quick pointers to artifacts (standardized)
                print(f"  ↳ external log: {logs_external_dir}/log_{result['exp_id']}.log")
                print(f"  ↳ artifacts jsonl: {artifacts_external_dir}/output_{result['exp_id']}.jsonl")
                print(f"  ↳ artifacts csv: {artifacts_external_dir}/report_{result['exp_id']}.csv (if available)")
            except Exception as exc:
                print(f"❌ Task {tid} (memfuse={'on' if memfuse_enabled else 'off'}) failed: {exc}")
                results.append(
                    {
                        "task_id": tid,
                        "model_answer": "EXECUTION_ERROR",
                        "reasoning_trace": str(exc),
                        "is_correct": False,
                        "memfuse_enabled": memfuse_enabled,
                    }
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Simplify result filename: results_<tag>_<ts>.jsonl
    safe_tag = str(args.output_tag or "run").replace(" ", "-")
    output_path = results_dir / f"results_{safe_tag}_{timestamp}.jsonl"

    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in results:
            memfuse_enabled = entry.get("memfuse_enabled", False)
            stats[memfuse_enabled]["total"] += 1
            if entry.get("is_correct"):
                stats[memfuse_enabled]["correct"] += 1
            submission_entry = {
                "task_id": entry.get("task_id"),
                "model_answer": entry.get("model_answer"),
                "reasoning_trace": entry.get("reasoning_trace"),
                "memfuse_enabled": memfuse_enabled,
            }
            handle.write(f"{submission_entry}\n")

    print("\n" + "=" * 50)
    print("🎉 CSV-driven evaluation complete!")
    overall_total = len(results)
    overall_correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = (overall_correct / overall_total * 100) if overall_total else 0.0
    print(f"  - Total tasks: {overall_total}")
    print(f"  - Correct tasks: {overall_correct}")
    print(f"  - Accuracy: {accuracy:.2f}%")

    for mem_state, counters in stats.items():
        total = counters["total"]
        correct = counters["correct"]
        rate = (correct / total * 100) if total else 0.0
        label = "ON " if mem_state else "OFF"
        print(f"    • MemFuse {label}: {correct}/{total} correct ({rate:.2f}%)")

    # Per-task quick view (trimmed)
    print("\nPer-task summary:")
    for entry in results:
        tid = entry.get("task_id")
        correct = entry.get("is_correct")
        state = "✅" if correct else "❌"
        ma = str(entry.get("model_answer", ""))[:80]
        gt = extract_ground_truth(sample_map.get(tid))[:80]
        print(f"  {state} task_id={tid} | answer={ma} | gt={gt}")

    print(f"\n📄 Results written to: {output_path}")
    print(f"🪵 Chat2Graph per-task logs: {logs_agent_dir}")
    print(f"🪵 External GAIA agent logs: {logs_external_dir}")
    print(f"📦 External artifacts (jsonl/csv): {artifacts_external_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
