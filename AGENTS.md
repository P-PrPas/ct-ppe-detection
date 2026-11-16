# Experiment workflow

Use `python ppe.py dataset|train|submit <config.json>` from the repository root. Read README.md before preparing a dataset, changing a recipe, resuming a run, or submitting a benchmark; it defines config paths, inputs, artifacts and recovery.

1. Translate the requested experiment into config changes under `configs/dataset`, `configs/train`, or `configs/submit`. Reuse `core/`; add a new experiment as data rather than a new script.
2. For datasets, record explicit family groups and split membership in a source manifest. Keep frozen evaluation membership and known benchmark families/videos out of training. The generic builder checks supplied groups and exact hashes; it cannot discover unrecorded scene relationships or replace historical review policies.
3. Run the selected command with `--dry-run`; inspect resolved paths and settings. This only validates config structure, not input files or GPU capacity. For a new experiment, choose new output directories; retain directories only for an intentional resume.
4. Execute the work the user requested, respecting the calibrated environment and jobs already running. Treat dataset creation, training and external submission as separate operations. Training completion does not authorize a submission on its own.
5. Report config paths, output artifacts, status and tests actually performed. Check `training-state.json` or `confirmed-server-run.json` before claiming completion.

Change `core/` when the requested behavior cannot be expressed by current configs. Add regression coverage in `tests/` and run `python -m unittest discover -s tests -v`. Keep imports free of GPU initialization, filesystem writes and server calls.

`legacy/` preserves dated workflows and source-specific review policies. Consult `legacy/README.md` when reproducing an old release; do not use those files as templates for new experiments. Do not modify a checkout used by an active training job.
