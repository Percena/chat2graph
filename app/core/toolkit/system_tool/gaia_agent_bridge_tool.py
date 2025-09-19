"""
Bridge tool that shells out to an external GAIA agent runner.

This module intentionally uses GAIA-centric names (gaia_agent) to avoid
coupling to specific vendor naming.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional
import sys

from app.core.common.logger import Chat2GraphLogger
from app.core.model.task import ToolCallContext
from app.core.toolkit.tool import Tool


logger = Chat2GraphLogger.get_logger(__name__)

# Resolve Chat2Graph repo root and standard output folders
def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(10):
        if (cur / ".git").exists() or (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start

_repo_root = _find_repo_root(Path(__file__).resolve().parent)
_gaia_base_dir = _repo_root / "test/benchmark/gaia"
_logs_external_dir = _gaia_base_dir / "logs" / "external"
_artifacts_external_dir = _gaia_base_dir / "artifacts" / "external"
_logs_external_dir.mkdir(parents=True, exist_ok=True)
_artifacts_external_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class RunnerParams:
    exp_id: str
    limit: int = 1
    concurrency: int = 1
    judge_concurrency: int = 1
    dataset: str = "GAIA_text_validation_Lmax1"
    output_jsonl: Optional[str] = None


async def run_gaia_agent(
    tool_call_ctx: ToolCallContext,
    question_hint: str | None = None,
    repo_path: Optional[str] = None,
    exp_id: Optional[str] = None,
    limit: int = 1,
    concurrency: int = 1,
    judge_concurrency: int = 1,
    dataset: str = "GAIA_text_validation_Lmax1",
    match_task_id: str | None = None,
    **kwargs,
) -> str:
    repo_path = (
        repo_path
        or os.getenv("GAIA_AGENT_PATH")
        or str(_repo_root / "gaia_agent")
    )
    exp_id = exp_id or f"gaia_{tool_call_ctx.job_id}_{tool_call_ctx.operator_id}"

    repo = Path(repo_path)
    oneclick = repo / "scripts/gaia/gaia_oneclick.py"

    dry_run_env = os.getenv("GAIA_AGENT_ONECLICK_DRY_RUN", "0")
    dry_run = str(dry_run_env).lower() in {"1", "true", "yes", "on"}
    if not dry_run and not oneclick.exists():
        raise FileNotFoundError(f"Cannot find gaia_oneclick.py at {oneclick}")

    # Write artifacts under Chat2Graph repo for consistent analysis
    output_jsonl = (_artifacts_external_dir / f"output_{exp_id}.jsonl").resolve()

    c2g_log_file = _logs_external_dir / f"log_{exp_id}.log"
    lock_file = _logs_external_dir / f"log_{exp_id}.lock"

    if output_jsonl.exists():
        try:
            with output_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        answer = obj.get("response") or obj.get("final_output") or ""
                        if answer:
                            logger.info(f"[gaia-agent] cache hit: {output_jsonl}")
                            return str(answer).strip()
        except Exception:
            pass

    if dry_run:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        fake = {"response": "DRY_RUN_ANSWER", "dataset_index": 1}
        with output_jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps(fake, ensure_ascii=False) + "\n")
        logger.info(f"[gaia-agent] dry-run wrote {output_jsonl}")
        return "DRY_RUN_ANSWER"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    if question_hint:
        env["GAIA_AGENT_QUESTION_HINT"] = str(question_hint)

    # Use current Python interpreter to avoid external project tooling requirements
    cmd = [
        sys.executable,
        str(oneclick),
        "run",
        "--exp_id",
        exp_id,
        "--limit",
        str(limit),
        "--concurrency",
        str(concurrency),
        "--judge_concurrency",
        str(judge_concurrency),
        "--dataset",
        dataset,
        "--output_jsonl",
        str(output_jsonl),
    ]
    if match_task_id:
        cmd += ["--task_id", match_task_id]

    logger.info(f"[gaia-agent] repo={repo_path} exp_id={exp_id} dataset={dataset} limit={limit}")
    if question_hint:
        logger.info(f"[gaia-agent] question_hint: {question_hint[:200]}")

    acquired = False
    try:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            logger.info(f"[gaia-agent] waiting for lock: {lock_file}")
            for _ in range(300):
                if output_jsonl.exists():
                    try:
                        with output_jsonl.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    obj = json.loads(line)
                                    answer = obj.get("response") or obj.get("final_output") or ""
                                    if answer:
                                        logger.info(f"[gaia-agent] cache hit while waiting: {output_jsonl}")
                                        return str(answer).strip()
                    except Exception:
                        pass
                if not lock_file.exists():
                    break
                await asyncio.sleep(2)

        if acquired:
            last_json_obj = None

            def _strip_ansi(s: str) -> str:
                import re
                return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", s)

            try:
                import os as _os
                master_fd, slave_fd = _os.openpty()
                env.setdefault("TERM", "xterm-256color")

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(repo),
                    env=env,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                )
                _os.close(slave_fd)

                async def _drain_pty(fd: int) -> dict | None:
                    nonlocal last_json_obj
                    loop = asyncio.get_running_loop()
                    with c2g_log_file.open("w", encoding="utf-8") as lf:
                        lf.write(f"CMD: {' '.join(cmd)}\n")
                        lf.flush()
                        while True:
                            try:
                                chunk = await loop.run_in_executor(None, _os.read, fd, 4096)
                            except Exception:
                                break
                            if not chunk:
                                break
                            text = chunk.decode("utf-8", errors="ignore")
                            for line in text.splitlines():
                                logger.info(f"[gaia-agent] {line}")
                                lf.write(line + "\n")
                                cleaned = _strip_ansi(line).strip()
                                if cleaned.startswith("{") and cleaned.endswith("}"):
                                    try:
                                        last_json_obj = json.loads(cleaned)
                                    except Exception:
                                        pass
                    return last_json_obj

                await _drain_pty(master_fd)
                rc = await proc.wait()
                try:
                    _os.close(master_fd)
                except Exception:
                    pass
            except Exception:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(repo),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                with c2g_log_file.open("w", encoding="utf-8") as lf:
                    lf.write(f"CMD: {' '.join(cmd)}\n")
                    lf.flush()
                    assert proc.stdout is not None
                    async for raw in proc.stdout:
                        line = raw.decode("utf-8", errors="ignore").rstrip()
                        logger.info(f"[gaia-agent] {line}")
                        lf.write(line + "\n")
                        cleaned = _strip_ansi(line).strip()
                        if cleaned.startswith("{") and cleaned.endswith("}"):
                            try:
                                last_json_obj = json.loads(cleaned)
                            except Exception:
                                pass
                rc = await proc.wait()

            if rc != 0:
                raise RuntimeError(
                    f"gaia agent runner failed with rc={rc}. See log: {c2g_log_file}"
                )

            if last_json_obj and isinstance(last_json_obj, dict):
                first_resp = last_json_obj.get("first_response") or last_json_obj.get("final_output")
                if first_resp:
                    return str(first_resp).strip()
    finally:
        if acquired and lock_file.exists():
            try:
                lock_file.unlink()
            except Exception:
                pass

    if not output_jsonl.exists():
        raise FileNotFoundError(f"Output JSONL not found: {output_jsonl}")
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answer = obj.get("response") or obj.get("final_output") or ""
            # Best-effort copy CSV report from external repo if it exists
            try:
                for candidate in [
                    Path(repo_path) / "data" / f"report_{exp_id}.csv",
                    Path(repo_path) / "data" / "tmp" / f"report_{exp_id}.csv",
                ]:
                    if candidate.exists():
                        shutil.copy2(candidate, _artifacts_external_dir / candidate.name)
                        break
            except Exception:
                pass
            return str(answer).strip()

    raise RuntimeError(f"No rows in exported JSONL: {output_jsonl}")


class GaiaAgentBridgeTool(Tool):
    """Expose GAIA agent runner as a Chat2Graph tool.

    Function name: run_gaia_agent
    Args: question_hint (str), repo_path (str, optional), exp_id (str, optional),
          limit (int), concurrency (int), judge_concurrency (int), dataset (str)
    Returns: final answer string from exported JSONL.
    """

    def __init__(self):
        super().__init__(
            name=run_gaia_agent.__name__,
            description=(
                "Run external GAIA agent pipeline and return the final answer. "
                "Streams native logs into Chat2Graph running_logs."
            ),
            function=run_gaia_agent,
        )
