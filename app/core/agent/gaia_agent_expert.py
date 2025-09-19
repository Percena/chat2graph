"""
GaiaAgentOneClickExpert: An Expert that delegates to an external GAIA agent's
one-click runner to retrieve the final answer, while preserving Chat2Graph
workflow lifecycle and logging.
"""

from __future__ import annotations

from datetime import datetime

from app.core.agent.expert import Expert
from app.core.common.async_func import run_async_function
from app.core.common.logger import Chat2GraphLogger
from app.core.common.type import JobStatus, WorkflowStatus
from app.core.model.job import SubJob
from app.core.model.message import AgentMessage, WorkflowMessage, MessageType
from app.core.model.task import ToolCallContext

from app.core.toolkit.system_tool.gaia_agent_bridge_tool import run_gaia_agent


logger = Chat2GraphLogger.get_logger(__name__)


class GaiaAgentExpert(Expert):
    """Expert that uses an external GAIA agent runner to fetch final answer."""

    def execute(self, agent_message: AgentMessage, retry_count: int = 0) -> AgentMessage:
        job_id = agent_message.get_job_id()
        job: SubJob = self._job_service.get_subjob(subjob_id=job_id)

        try:
            job_result = self._job_service.get_job_result(job_id=job.id)
            if job_result.has_result() and job_result.status == JobStatus.FINISHED:
                messages = self._message_service.get_message_by_job_id(job_id=job.id, message_type=MessageType.AGENT_MESSAGE)
                if messages:
                    return messages[-1]
                return AgentMessage(payload="", job_id=job.id)

            job_result.status = JobStatus.RUNNING
            self._job_service.save_job_result(job_result=job_result)

            tool_ctx = ToolCallContext(job_id=job.id, operator_id="GaiaAgentOneClickExpert")

            question = job.goal
            dataset = "auto"
            match_task_id = None
            try:
                import re as _re
                import json as _json
                goal = job.goal or ""
                m = _re.search(r"\[\[GAIA_TASK_META\]\](.*?)\[\[/GAIA_TASK_META\]\]", goal, _re.S)
                if m:
                    meta = _json.loads(m.group(1))
                    if isinstance(meta, dict):
                        dataset = meta.get("dataset", dataset)
                        match_task_id = meta.get("task_id", None)
                elif job.context:
                    ctx = _json.loads(job.context)
                    if isinstance(ctx, dict):
                        dataset = ctx.get("dataset", dataset)
                        match_task_id = ctx.get("task_id", None)
            except Exception:
                pass

            logger.info(f"[GaiaAgentExpert] Delegating to external GAIA agent at {datetime.now()} for job={job.id}")

            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            human_id = (match_task_id or job.id).replace(" ", "_")
            exp_id = f"{ts}_{human_id}"

            final_output: str = run_async_function(
                run_gaia_agent,
                tool_ctx,
                question_hint=question,
                exp_id=exp_id,
                limit=1,
                concurrency=1,
                judge_concurrency=1,
                dataset=dataset,
                match_task_id=match_task_id,
            )

            workflow_message = WorkflowMessage(
                payload={"output": final_output, "status": WorkflowStatus.SUCCESS, "scratchpad": final_output},
                job_id=job.id,
            )
            self._message_service.save_message(workflow_message)
            expert_message: AgentMessage = self.save_output_agent_message(job=job, workflow_message=workflow_message)
            job_result = self._job_service.get_job_result(job_id=job.id)
            job_result.status = JobStatus.FINISHED
            self._job_service.save_job_result(job_result=job_result)
            return expert_message

        except Exception as e:
            logger.error(f"GaiaAgentExpert execution failed: {e}")
            workflow_message = WorkflowMessage(
                payload={"output": f"GaiaAgentExpert error: {e}", "status": WorkflowStatus.EXECUTION_ERROR, "scratchpad": f"GaiaAgentExpert error: {e}"},
                job_id=job.id,
            )
            expert_message: AgentMessage = self.save_output_agent_message(job=job, workflow_message=workflow_message)
            job_result = self._job_service.get_job_result(job_id=job.id)
            job_result.status = JobStatus.FAILED
            self._job_service.save_job_result(job_result=job_result)
            return expert_message
