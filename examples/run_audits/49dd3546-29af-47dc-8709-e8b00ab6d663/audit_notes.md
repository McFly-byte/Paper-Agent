# Run audit notes

## 产物清单

- `manifest.json`
- `timeline.jsonl`
- `state_summary.json`
- `workflow_errors.json`
- `boundary_checks.json`
- `coordinator/brief.json`
- `background/background_context.json`
- `planner/plan.json`
- `planner/plan_review.json`
- `search/search_results.summary.json`
- `search/search_queries.json`
- `paper_filter/paper_candidates.summary.json`
- `paper_filter/filtered_papers.summary.json`
- `paper_filter/filter_report.json`
- `reading/extracted_data.summary.json`
- `reading/paper_readings.summary.json`
- `evidence/evidence_ledger.summary.json`
- `evidence/evidence_items.sample.json`
- `analysis/analysis_results.md`
- `writing/outline.md`
- `writing/written_sections.md`
- `writing/citation_map.json`
- `writing/evidence_bound_sections.json`
- `review/faithfulness_review.json`
- `review/report_citation_validation.json`
- `report/final_report.md`
- `audit_notes.md`

## 关键计数

```json
{
  "run_id": "49dd3546-29af-47dc-8709-e8b00ab6d663",
  "current_step": "ExecutionState.REPORTING",
  "max_papers": 50,
  "has_plan": true,
  "search_result_count": 16,
  "filtered_paper_count": 16,
  "extracted_paper_count": 0,
  "evidence_item_count": 0,
  "citation_ref_count": 0,
  "written_section_count": 6,
  "report_length": 18602,
  "workflow_error_count": 2,
  "boundary_check_keys": [
    "search",
    "reading",
    "analyse",
    "writing",
    "report"
  ],
  "faithfulness_verdict": "pass",
  "report_citation_verdict": null
}
```

## Warnings

- workflow_errors: 2

## Tmp store 读取

- (all ok)

## 建议人工检查点

- research brief 与 user_request 对齐
- background_context 关键词与检索意图
- plan / plan_review 可执行性
- search_queries 与 search_results.summary
- paper_filter_report 是否误杀
- extracted_data 字段完整性
- evidence_ledger 规模与 claim 覆盖
- analysis_results 信息增益
- citation_map 与 written_sections 中 [Cn] 使用
- faithfulness_review 与 report_citation_validation
