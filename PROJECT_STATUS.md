# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 5 — built-in CPU transformer: complete.** Waiting for instruction before starting Phase 6.

## Cumulative summary (all phases so far)

**Phase 1 — Foundation:** repository structure, Streamlit app, hardware detection, CSV/JSON intake, generic
schema/task detection, profiling. 17 tests.

**Phase 2 — Quality and basic preprocessing:** quality engine, stateless cleaner, Quality Analysis UI page. 32 tests.

**Phase 3 — E0 / E1 / E2 processing:** subset selection, temporal/random split, sampling, three data levels,
Processing UI page. 46 tests.

**Phase 4 — Feature engineering + leakage checks:** the tested leakage detector wired into E1/E2, reusing the
real train/validation cutoff, auto-removing only structural leaks. 50 tests.

**Phase 5 — Built-in CPU transformer (complete, this session):** see below.

## Completed work (Phase 5, this session)

- Moved `src/evaluation/metrics.py`, `src/representation/tabular_tokenizer.py`, `src/models/sanity_transformer.py`
  from `_pending/` into `src/`.
- **Reconciled a real interface mismatch found while inspecting the repo** (per the mandatory phase-start
  protocol): `src/models/sanity_transformer.py` (drafted alongside `_pending/src/models/base_draft.py`) needs a
  richer `ModelAdapter` — default `evaluate()` (threshold on validation, applied to test) and `sanity_checks()`,
  `ModelUnavailable`, `check`/`basic_data_checks` helpers, `__init__(self, config)` — than the thin placeholder
  Phase 1 had already committed as `src/models/base.py` (`__init__(self, config, task)`, `evaluate` abstract, no
  shared logic). Confirmed nothing live used the old shape (`grep` — only `PLANNED_BACKENDS` was ever imported),
  so replaced `src/models/base.py`'s `ModelAdapter` with `base_draft.py`'s, kept `PLANNED_BACKENDS` (now marking
  `sanity_transformer` `implemented: true`), and deleted `base_draft.py` as fully superseded.
- `SanityTransformerAdapter`: added a `device` constructor parameter (defaults to the old
  `torch.cuda.is_available()` check) so `app.py` passes `hardware.py`'s already-detected device instead of a
  second, independent detection; added `torch.set_num_threads(1)` in `train()`.
- **Found and fixed a real bug, not a draft-quality issue, while debugging a flaky-looking test:** `sanity_checks()`
  calls `self._build()` and a short throwaway `self.train()` (2 epochs, ≤5,000 rows) as part of its diagnostics,
  and left `self.model` / `self.tokenizer` overwritten with that throwaway model afterwards. `train()` →
  `sanity_checks()` → `evaluate()` — exactly what the Model page does — silently evaluated the throwaway model.
  Root-caused by checksumming the synthetic data (ruled out data nondeterminism), reproducing the exact test
  setup in isolation (ruled out import-order/environment effects), then finding the mutation. Fixed with
  `try`/`finally` save-and-restore around `sanity_checks()`'s internals; it now has no side effects on an
  already-trained adapter. Confirmed the fix both in `pytest` and by re-running training in a live browser
  (validation PR-AUC 0.97, test PR-AUC 0.83, recall 0.78, precision 0.55 on the full synthetic dataset — before
  the fix, the same run's evaluation reported PR-AUC 0.01).
- New `representation:` and `models.sanity_transformer:` sections in `config.yaml`.
- New "Model" page content in `app.py` (`page_model`, `render_model_results`): trains on any already-built E0/E1/E2
  level with configurable epochs/batch size/learning rate/patience, then shows the loss and validation PR-AUC
  curves, the sanity-check table, and validation/test/test-unweighted/test-known-entities-only metrics (PR-AUC,
  ROC-AUC, precision, recall, F1, weighted confusion matrix) with a validation-chosen decision threshold.
  Accuracy is deliberately not shown alone (section 16: misleading under this imbalance).
- Sidebar pipeline tracker and Dashboard status table: "Train and evaluate" / "Built-in transformer" now reflect
  real `st.session_state.model` state; the "(later phase)" sidebar annotation threshold moved from step 6 to
  step 8, since steps 6 and 7 are both implemented as of this phase.

## Files changed this session

**New:** `tests/test_model.py`
**Modified:** `src/models/base.py` (replaced with the richer interface), `src/models/sanity_transformer.py`
(`device` param, single-threading), `app.py` (Model page, session state, status rows), `config.yaml`
(`representation`, `models.sanity_transformer`), `tests/test_app.py` (new `test_demo_flow_model_training`),
`_pending/README.md`
**Moved (git mv, content unchanged unless noted above):** `_pending/src/evaluation/metrics.py` → `src/evaluation/`;
`_pending/src/representation/tabular_tokenizer.py` → `src/representation/`; `_pending/src/models/sanity_transformer.py` → `src/models/`
**Deleted:** `_pending/src/models/base_draft.py` (content merged into `src/models/base.py`)

## Tests completed

`pytest` → **56 passed** (was 50 at the end of Phase 4; 6 added this session). Run cold.

| File | Covers |
|---|---|
| test_model.py (new) | `TabularTokenizer`: ids stay in vocabulary range, `to_dict`/`from_dict` round-trips to identical ids; the adapter trains on the full synthetic dataset (not a tiny subset, so fraud reliably appears in every split), loss decreases, all sanity checks pass, and it genuinely learns (test PR-AUC > 0.3 — this is what caught the `sanity_checks()` side-effect bug); `basic_data_checks` fails on an empty `PreparedLevel`; save/load round-trip gives identical predictions; `evaluate()` reports `None` for AUC (never a fabricated number) on a split with zero positives, which a small subset can genuinely produce given the synthetic fraud's clustered-burst pattern |
| test_app.py (extended) | headless UI: demo data → Processing → build E1 → Model → train (2 epochs, for speed) → Dashboard, all through the real `app.py` |

Also verified interactively in a real browser: full synthetic dataset (27,223 rows) → Processing (full subset,
E2 built) → Model → trained the default config (15 epochs) → 106,177 parameters, 63.76 s on CPU → all sanity
checks passed → validation PR-AUC 0.97, test PR-AUC 0.83, recall 0.78, precision 0.55, confusion matrix
TP 18 / FP 15 / FN 5 / TN 4046.

**Not yet executed:** the app on the real 1.3M-row transaction file (only synthetic data was available in the
build environment, same limitation as Phases 2–4).

## Commands used

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Current git commit

The commit titled "Phase 5: built-in CPU transformer …" (run `git log --oneline -1` for its exact hash; amending
this file changes the hash, so it is intentionally not pinned here).

## Known issues

- Everything carried over from Phases 2–4 still applies.
- PyTorch does not guarantee bit-exact reproducibility across processes/platforms even with every seed fixed;
  `torch.set_num_threads(1)` removes the specific multi-threaded-reduction source found here, but section 18's
  "reproducibility" should be understood as "the same seed reproduces the same result on the same machine," not
  an absolute cross-platform guarantee — this is a PyTorch characteristic, not something this project can fully
  close.
- A small subset (low row count, especially without "full") can land zero positives in validation and/or test,
  since the synthetic demo data's fraud is a rare, clustered burst per card; `evaluate()` correctly reports
  `None` for AUC-based metrics rather than fabricating one, but the UI's metrics table just shows "n/a" without
  explaining why — worth a note in the UI if this comes up often on the real dataset.
- "Sequence length" (section 12's list of configurable hyperparameters) isn't a separate knob: `TabularTokenizer`
  gives one token per feature plus a `[CLS]` token, so the sequence length is `1 + len(features)`, fixed by the
  chosen data level (E0/E1/E2). This fits the tabular representation used here; revisit if a future phase adds a
  genuinely variable-length sequence representation (e.g. the drafted `previous_transactions` /
  `format_previous_transactions` in `src/features/behavioral_features.py`, unused so far).
- The Hugging Face / Nemotron / API backends remain "not yet implemented (Phase 7)" in the Model page, honestly.
- Training and sanity-checking together on the full dataset with the default 15 epochs takes about 60–90 s on
  this development machine (single CPU thread, by design — see above); not yet measured on the real 1.3M-row file.

## Next task (Phase 6 — evaluation + experiment runner)

1. Read this file; run `pytest`.
2. The Model page already trains and evaluates one level at a time. Phase 6's job is the *comparison*: run E0,
   E1 and E2 with **identical model settings, seed and split** (section 17) and report PR-AUC/ROC-AUC/precision/
   recall/F1/training time/device for each, side by side — this is what the "Experiments" placeholder page
   promises. `src/evaluation/metrics.py`'s `evaluate(pred, threshold)` free function (with `_slice_unseen_entity`
   support) was moved in Phase 5 but is still unused — check whether it fits this job before writing something new.
3. Persist experiment results (section 19: `experiments/`, `reports/` directories already exist, empty) so a
   comparison isn't lost on rerun; decide a simple, honest format (e.g. one JSON per experiment run) — do not
   over-engineer an experiment-tracking system beyond what's asked.
4. Results page: comparison table/chart of E0 vs E1 vs E2's *measured* metrics only (section 27: no fabrication;
   if a level wasn't run, say so, don't fill in a guess).
5. Keep using `DataPreparer.build(level)`'s cache (already built E0/E1/E2 share the same split) rather than
   re-preparing data per experiment.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [x] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [x] Phase 4: feature engineering + leakage checks
- [x] Phase 5: built-in CPU transformer
- [ ] Phase 6: evaluation + experiment runner
- [ ] Phase 7: Hugging Face / Nemotron / API adapters
- [ ] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Later: generalization test on a non-fraud dataset
