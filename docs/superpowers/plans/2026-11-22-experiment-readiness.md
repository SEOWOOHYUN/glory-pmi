# Experiment readiness implementation plan

> Execute in this session, using regression tests before behavior changes.

**Goal:** Make the reviewed experiment branch usable after data and single-GPU CUDA setup, and submit the changes as a PR to the shared experiment branch.

**Scope:** The user approved fixing the review findings and preparing an experiment workflow. Preserve the model and the Count/PPMI definitions. Do not merge existing PRs or claim a full CUDA run without hardware/data evidence.

**Architecture:** Validate each seed's behavior cache against its raw input and generation settings. Pin direct dependencies, add a read-only readiness command, and document one supported Linux/Python 3.11/CUDA installation path. Keep CPU checks available separately.

## Steps

- [x] Reproduce stale cross-seed behavior caches, missing shards and legacy caches in integration tests; make cache reuse require matching provenance and complete shards.
- [x] Test readiness checks against missing data, unavailable GPU and valid input; report actionable errors before training. Expose the existing training step limit through the common runner for a bounded trial.
- [x] Pin dependencies and use them in CI. Document fork/branch choice, data preparation, CUDA setup, preflight, trial and full runs.
- [x] Run the full CPU suite, syntax checks and dependency checks. Review the diff and create a PR targeting `feat/pmi-neighbor-experiments` with CUDA validation limits.

## Review focus

- Changing raw data then returning to an older seed must regenerate its samples.
- Legacy or incomplete caches must never be silently trusted.
- CPU-only hosts should give a clear GPU error, while CPU data checks remain usable.
- Missing MIND GloVe must be reported before preprocessing; Adressa must not require it.
- A short trial must have a separate output and must not be described as a complete experiment.

## Verification evidence

- macOS / Python 3.9.6: all 25 CPU tests passed, including both synthetic dataset models and interrupted cache publication.
- `pip check`, syntax compilation and diff whitespace checks passed.
- Readiness CLI: synthetic Adressa input passes with `--cpu-only`; default CUDA check correctly exits 1 on this CPU host.
- Runner dry-run carries `max_steps=2` and a separate trial directory.
- Read-only independent review found no blocking issues.
- Not run here: Linux/Python 3.11 installation, CUDA/NCCL training or full real datasets.
