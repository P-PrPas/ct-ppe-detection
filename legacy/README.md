# Historical workflows

Files here preserve dated data policies, one-off diagnostics and report generation. New experiments use `ppe.py` and `core/`; these files are not the normal starting point for agents.

Run from the repository root using `python -m legacy.<filename_without_py>`, so package imports resolve. Dataset/model/report paths remain rooted at the repository; the files still expect their historical snapshots and artifacts.

| Task | Modules |
| --- | --- |
| Reproduce expanded releases with original review rules | `build_ppe_expanded`, `build_ppe_next_three`, `build_ppe_four_more`, shared `dataset_release` |
| Reproduce older full/reviewed/person releases | `build_ppe_full_current`, `build_ppe_reviewed_dataset`, `build_ppe_person_release`, `build_dataset` |
| Source preparation and review | `prepare_ppe_dataset`, `select_ppe_training_release`, `collect_verified_ppe_labels`, `apply_verified_ppe_labels`, `apply_ppe_pool_edits`, `annotation_script`, `remap_labels` |
| Historical multi-model benchmarks | `submit_ppe_noleak_all`, `submit_ppe_customer_nvr`, `benchmark_ppe_andromeda` |
| Diagnostics/reporting | `analyze_ppe_run5`, `compare_ppe_resolution`, `collect_ppe_report_metrics`, `build_ppe_training_report`, `check_ppe_usage`, `rf_search`, `rf_stats`, `yolo_stats` |

`dataset_release` retains frozen core val/test, auxiliary heldouts, shoe/Cap decisions, video exclusions and original seeded grouping order. The generic manifest builder intentionally does not infer those policies. Reproduce these workflows only with their recorded inputs; do not reduce their assertions to accommodate new inputs.

The old single-checkpoint `submit_ppe_benchmark98` is now `ppe.py submit configs/submit/full-yolo26s.json` (edit the config for other targets). The nine old training scripts and two launchers were replaced by the ten recipes in `configs/train/`; recipe names describe release/model combinations.

Some historical files perform work at import time or edit remote data. Inspect the selected module and its documented behavior before executing it. The reusable `core` modules do not import any of these files.
