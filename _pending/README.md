# Pending code (not used by the application)

Code drafted before the phased plan was adopted, kept here so it is not lost or rewritten. **Nothing in this folder
is imported by `app.py` or covered by the test suite.** Each file is moved into `src/` only during its phase, after
inspection, adaptation to the current interfaces and tests.

| File | Target phase | Origin | Verification state |
|---|---|---|---|
| src/preprocessing/outliers.py | 3 or 6 (robustness) | research notebooks | tested there; also fitted on training rows. Not wired in for Phase 3: flagging-only outlier handling wasn't in the Phase 3 brief; revisit for Phase 6 robustness |
| src/preprocessing/split.py | superseded | research notebooks | tested there, but needs an explicit `train_end` / `test_start` date per dataset. `DataPreparer.prepare_split` (Phase 3, `src/preprocessing/levels.py`) uses a quantile-of-fractions temporal split instead, so it works on any dataset without per-dataset dates; kept here for reference |
| src/representation/transaction_formatter.py | 5/7 | research notebooks | tested there. `find_quasi_identifiers` was reimplemented (not imported) as a small local helper in `src/preprocessing/levels.py` for Phase 3's exclusion policy, so the rest of this file (LLM text formatting) stays deferred to phase 5/7 |
| src/representation/text_builder.py | 7 | drafted | **unverified draft** |
| src/models/hf_backend.py | 7 | research notebooks (Nemotron LoRA on T4) | tested there |
| src/models/hf_adapter.py, api_adapter.py, registry.py | 7 | drafted | **unverified draft** |
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

Moved into `src/` during Phase 5: `src/evaluation/metrics.py`, `src/representation/tabular_tokenizer.py`
(unchanged); `src/models/sanity_transformer.py` (unchanged apart from an added `device` constructor parameter
that defaults to the old `torch.cuda.is_available()` check, so the UI can pass `hardware.py`'s already-detected
device instead of re-detecting it, and a `torch.set_num_threads(1)` added to `train()` — multi-threaded CPU
reductions are not bit-reproducible in PyTorch, and this was caught concretely: the same seed gave validation
PR-AUC 0.91 in one process and 0.10 in another before the fix). `src/models/base_draft.py`'s `ModelAdapter` (the
one `sanity_transformer.py` was actually written against — richer than the thin ABC Phase 1 had originally
committed as `src/models/base.py`, with default `evaluate`/`sanity_checks`, `ModelUnavailable`,
`check`/`basic_data_checks`) replaced Phase 1's `src/models/base.py` content; `base_draft.py` itself was deleted
as fully superseded. `PLANNED_BACKENDS` (used by the Model page) was kept, with `sanity_transformer` marked
implemented.

**Found and fixed in Phase 5, not a draft issue — a real bug in already-moved code:** `SanityTransformerAdapter.sanity_checks()`
calls `self._build()` and a short throwaway `self.train()` on a small sample as part of its diagnostics, and was
leaving `self.model` / `self.tokenizer` overwritten with that throwaway (2-epoch, ≤5,000-row) model afterwards.
Calling `train()` then `sanity_checks()` then `evaluate()` — exactly what the Model page does — silently evaluated
the throwaway model, not the real one. `sanity_checks()` now saves and restores `self.model` / `self.tokenizer`
around its internal checks (`try`/`finally`), so it has no side effects on an already-trained adapter.
