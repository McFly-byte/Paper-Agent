# Run audit 示例与提交流程

## 默认输出位置

- 配置项 `workflow_v2.run_audit_dir` 默认为 `artifacts/run_audits`（相对项目根目录）。
- 该目录已在 `.gitignore` 中忽略（仅保留 `artifacts/run_audits/.gitkeep`），**请勿将真实运行产生的审计包直接提交到 Git**。

## 给他人审查时

1. 本地完成一次小样本 run，并开启 `workflow_v2.enable_run_audit_dump: true`。
2. 打开生成的 `artifacts/run_audits/<run_id>/`，人工检查已脱敏、无 API key、无用户隐私、无大段论文原文。
3. 将确认可公开的子集**手动复制**到 `examples/run_audits/<run_id>/` 后再提交。

## 占位示例

- `sample_min_run/` 仅含最小 JSON 片段，用于展示 manifest / state_summary 形态；**不是**一次真实 run 的完整审计包。

- 使用 `max_papers: 5`（或 8）的小样本，降低摘要与证据条目的体积。
- 若仅需结构校验，可将 `run_audit_include_full_report` 设为 `false`。
- 手动从保存的 state JSON 导出时，可使用：

  `python scripts/export_run_audit.py --state-json your_state.json --out examples/run_audits/manual_sample`

  注意：JSON 中未包含的外置 blob（Chroma tmp store）在包内会显示为 `not_available`，完整审计应在服务端 run 结束时自动导出。
