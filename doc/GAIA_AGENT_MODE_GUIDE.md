# GAIA Agent (Minimal Agent Mode)

本文档描述如何在 Chat2Graph 的工作流内，仅保留一个最小 Expert，通过外部 GAIA Agent 评测流程完成任务并产出最终答案。命名上统一使用 “gaia_agent”，避免依赖任何特定实现名。

## 组件概览
- Expert: `GaiaAgentOneClickExpert`
- Tool: `GaiaAgentBridgeTool`（子进程调用 `scripts/gaia/gaia_oneclick.py run`）
- YAML: `test/benchmark/gaia/gaia_agents_min.yml`
- 运行脚本：`run_gaia_task_plan.py`（默认禁用 KB/MCP，避免非必要系统调用）

## 环境变量
- `GAIA_AGENT_PATH`: 外部 GAIA Agent 仓库路径（默认回退到 `~/path/to/repo`）
- `GAIA_AGENT_ONECLICK_DRY_RUN`: 设为 `1` 可快速生成假结果做冒烟

## 运行命令（Agent 模式）
- 正常运行：
```
python run_gaia_task_plan.py \
  --csv-path ./gaia_sample_10.csv \
  --split validation --level all \
  --parallel-num 1 --limit-num 2 \
  --output-tag gaia-agent \
  --agent-config test/benchmark/gaia/gaia_agents_min.yml \
  --runner agent
```

- 快速冒烟（不真正执行执行流程）：
```
GAIA_AGENT_ONECLICK_DRY_RUN=1 python run_gaia_task_plan.py \
  --csv-path ./gaia_sample_10.csv \
  --split validation --level all \
  --parallel-num 1 --limit-num 2 \
  --output-tag gaia-agent \
  --agent-config test/benchmark/gaia/gaia_agents_min.yml \
  --runner agent
```

## 产物
- Chat2Graph 日志：`test/benchmark/gaia/running_logs/log_<exp_id>.log`
- 外部 GAIA Agent 导出：`<GAIA_AGENT_PATH>/data/output_<exp_id>.jsonl` 与 `report_<exp_id>.csv`

## 说明
- 该模式仍通过 Chat2Graph 的 Leader→Expert→Operator 框架执行，但不做 KB/MCP 相关工作，也不依赖多余工具链。
- Expert 直接调用外部 GAIA Agent one-click，最大限度对齐 one-click 行为与产物格式。
- 如需逐步替换现有命名到 `gaia_agent`，可优先使用本 YAML 与 Expert/Tool 实现；其它历史命名文件保持不变，便于对比审阅。
