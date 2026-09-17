# Pending code (not used by the application)

Code drafted before the phased plan was adopted, kept here so it is not lost or rewritten. **Nothing in this folder
is imported by `app.py` or covered by the test suite.** Each file is moved into `src/` only during its phase, after
inspection, adaptation to the current interfaces and tests.

| File | Target phase | Origin | Verification state |
|---|---|---|---|
| src/preprocessing/outliers.py | 3 or 6 (robustness) | research notebooks | tested there; also fitted on training rows. Not wired in for Phase 3: flagging-only outlier handling wasn't in the Phase 3 brief; revisit for Phase 6 robustness |
| src/preprocessing/split.py | superseded | research notebooks | tested there, but needs an explicit `train_end` / `test_start` date per dataset. `DataPreparer.prepare_split` (Phase 3, `src/preprocessing/levels.py`) uses a quantile-of-fractions temporal split instead, so it works on any dataset without per-dataset dates; kept here for reference |
| src/representation/transaction_formatter.py | 5/7 | research notebooks | tested there. `find_quasi_identifiers` was reimplemented (not imported) as a small local helper in `src/preprocessing/levels.py` for Phase 3's exclusion policy, so the rest of this file (LLM text formatting) stays deferred to phase 5/7 |
| src/representation/tabular_tokenizer.py, text_builder.py | 5/7 | drafted | **unverified draft** |
| src/models/sanity_transformer.py | 5 | drafted | **unverified draft** (binary only) |
| src/evaluation/metrics.py | 6 | research notebooks | tested there |
| src/models/hf_backend.py | 7 | research notebooks (Nemotron LoRA on T4) | tested there |
| src/models/hf_adapter.py, api_adapter.py, registry.py, base_draft.py | 7 | drafted | **unverified draft** |
| tests/synthetic.py | 3+ | drafted | copy now lives in src/ingestion/demo_data.py |

Moved into `src/` during Phase 3 (see `PROJECT_STATUS.md`): `src/preprocessing/missing_values.py`, `categorical.py`,
`numerical.py`, `datetime_features.py`, `pipeline.py`, `sampling.py` (unchanged); `src/features/behavioral_features.py`
(unchanged, plus `check_point_in_time` added); `src/preprocessing/levels.py` (rewritten from the draft: roles are
resolved from the already-confirmed `DatasetRoles` instead of re-detected, numeric features use the fitted
robust/log/decile-bin transforms instead of raw values, cyclical hour/day-of-week features added, E2's history
features gracefully skip when no entity/time/amount is set; full leakage scanning was deferred to Phase 4 at the time).

Moved into `src/` during Phase 4: `src/quality/leakage_detector.py` (unchanged; `scikit-learn` added to
`requirements.txt`). Wired into `DataPreparer._run_leakage` (`src/preprocessing/levels.py`), which reuses the
same train/validation cutoff as `prepare_split` (instead of the detector's own default holdout-fraction cutoff)
and skips gracefully, with no findings, when no datetime column is set. Only two checks (`post_event_time`,
`target_word_in_text` — structural, unambiguous leaks) are auto-removed from the feature set; every other
finding, including very high single-feature predictive power, is reported for human review only, never removed
automatically, per the comment on `AUTO_REMOVE_CHECKS`.
