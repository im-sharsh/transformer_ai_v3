# Pending code (not used by the application)

Code drafted before the phased plan was adopted, kept here so it is not lost or rewritten. **Nothing in this folder
is imported by `app.py` or covered by the test suite.** Each file is moved into `src/` only during its phase, after
inspection, adaptation to the current interfaces and tests.

| File | Target phase | Origin | Verification state |
|---|---|---|---|
| src/preprocessing/outliers.py | 3 or 6 (robustness) | research notebooks | tested there; also fitted on training rows. Never wired in through Phase 8: not required by any phase as specified, and adding it now would be a new pipeline feature, not the polish Phase 8's brief asked for. Revisit if outlier *handling* (beyond the profiling counts already shown) turns out to matter on the real dataset |
| src/preprocessing/split.py | superseded | research notebooks | tested there, but needs an explicit `train_end` / `test_start` date per dataset. `DataPreparer.prepare_split` (Phase 3, `src/preprocessing/levels.py`) uses a quantile-of-fractions temporal split instead, so it works on any dataset without per-dataset dates; kept here for reference |
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

Moved into `src/` during Phase 7: `src/models/hf_adapter.py`, `hf_backend.py`, `api_adapter.py`, `registry.py`,
`src/representation/text_builder.py`, `transaction_formatter.py`, all unchanged **except** `hf_adapter.py`'s
`_load()`, which now passes an optional `target_modules` from `config.yaml` instead of always using
`hf_backend.py`'s LLaMA-family default (`q_proj`, `k_proj`, ...). Found by actually running it: LoRA attachment
on `gpt2` failed outright (`NoMatchingPeftModuleError`) because GPT-2's attention module is named `c_attn`, not
`q_proj`/`k_proj`/`v_proj` — an architecture-specific assumption baked into the original Nemotron-focused code.
`config.yaml`'s `models.huggingface.target_modules: [c_attn]` now matches the shipped `model_id: gpt2` default;
a LLaMA-family model (including Nemotron, which needs no override) still gets the correct default automatically.
`transformers`/`peft`/`accelerate` added to `requirements.txt` as **optional** (commented out): the app runs
fully without them, and `HuggingFaceAdapter.availability()` reports the backend unavailable with a clear reason
if they're missing, rather than the app failing to start.

**Verified for real, not just wired up:** with the fix above, a full train → predict → evaluate → save → load
cycle was run against `distilgpt2` (a real, small, public Hugging Face model — not a fabricated checkpoint) LoRA-
fine-tuned on CPU: 21 optimizer steps in 6.9 s, validation PR-AUC 0.71, save/load round-trip gave identical
predictions (`tests/test_backends.py`). The Custom/API adapter was verified against a real local HTTP server
(`tests/test_backends.py`'s `local_api_server` fixture) matching its documented `{"prompts": [...]} ->
{"probabilities": [...]}` contract, including that it correctly rejects an out-of-range response. Nemotron was
**not** run: per section 13/14 of the brief ("do NOT invent a model checkpoint"; requires a CUDA GPU), and this
environment has neither a verified checkpoint/config nor a GPU, `NemotronAdapter.availability()` honestly
reports it unavailable here rather than a fabricated success.
