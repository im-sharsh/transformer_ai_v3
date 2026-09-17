# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 6 — evaluation + experiment runner: complete.** Waiting for instruction before starting Phase 7.

## Cumulative summary (all phases so far)

**Phase 1 — Foundation:** repository structure, Streamlit app, hardware detection, CSV/JSON intake, generic
schema/task detection, profiling. 17 tests.

**Phase 2 — Quality and basic preprocessing:** quality engine, stateless cleaner, Quality Analysis UI page. 32 tests.

**Phase 3 — E0 / E1 / E2 processing:** subset selection, temporal/random split, sampling, three data levels,
Processing UI page. 46 tests.

**Phase 4 — Feature engineering + leakage checks:** the tested leakage detector wired into E1/E2. 50 tests.

**Phase 5 — Built-in CPU transformer:** `SanityTransformerAdapter` (train/predict/evaluate/save/load), Model UI page. 56 tests.

**Phase 6 — Evaluation + experiment runner (complete, this session):** see below.

## Completed work (Phase 6, this session)

- New `src/evaluation/experiment.py` (no `_pending` draft existed for this; built directly from section 17/19
  of the master prompt): `run_experiment`/`run_comparison` build (or reuse, via `DataPreparer`'s cache) each
  requested level, then train and evaluate the **same** `SanityTransformerAdapter` config, seed and split on
  each — so a difference in results comes from the data preparation, never from a different training setup.
  `ExperimentResult` records level, rows per split, feature count, model, device, seed, settings, train/inference
  time, epochs run, threshold and the full metrics dict. `comparison_table` renders one row per level.
  `save_experiment`/`list_experiments` persist to `experiments/` as never-overwritten
  `experiment_vNNN.json` manifests (dataset id, split info, all results) — section 19's "store metrics,
  configuration... in reports/experiments/" requirement; nothing is estimated when nothing has been saved.
- **Found and fixed a real bug in an already-shipped Phase 2 utility, hit immediately by the new code:**
  `src/utils/audit.py`'s `next_version()` (used since Phase 2 for `data/processed/dataset_vNNN/` directories)
  matched candidate version numbers against `p.name`. Called with a file that has an extension
  (`experiment_v001.json`, not a bare directory `dataset_v001`), the regex never matched the existing file, so
  every save silently reused `experiment_v001` and overwrote the previous one — caught immediately by
  `test_experiments_are_never_overwritten`. Fixed by matching `p.stem` instead of `p.name` (identical behavior
  for extension-less directories, correct for files with an extension); Phase 2's own usage and tests are
  unaffected (confirmed: full suite still green).
- New "Experiments" page (`app.py`): pick which levels to compare, shared epochs/batch size/patience settings,
  "Run comparison" (progress reported level by level), the comparison table and a PR-AUC/F1 bar chart, and a
  "Save this comparison" button. New "Results" page: lists every saved experiment from `experiments/` (oldest
  session, closed laptop, whatever — it reads the files, not session state) and renders the same table/chart for
  whichever one is selected. Both pages share one `render_experiment()` renderer.
- Sidebar pipeline tracker and Dashboard status table: "Compare experiments" / "Evaluation and experiments" now
  reflect real `st.session_state.experiment` state instead of always "later phase" / "planned".

## Files changed this session

**New:** `src/evaluation/experiment.py`, `tests/test_experiment.py`
**Modified:** `src/utils/audit.py` (`next_version` bug fix), `app.py` (Experiments/Results pages, removed the
now-fully-superseded `page_placeholder`/`PHASE_OF_PAGE`, session state, status rows), `tests/test_app.py`
(new `test_demo_flow_experiments`, updated the stale "Not run" assertion in `test_every_section_renders_without_data`)

## Tests completed

`pytest` → **62 passed** (was 56 at the end of Phase 5; 6 added this session). Run cold.

| File | Covers |
|---|---|
| test_experiment.py (new) | `run_comparison` uses the identical seed and the identical split/sample row counts for every level (the whole point of a controlled comparison); `comparison_table` returns exactly one row per level, in E0/E1/E2 order regardless of the order requested; save/list round-trip preserves the data; an empty `experiments/` directory reports no experiments (never fabricated); **repeated saves are never overwritten** (this is what caught the `next_version` bug) |
| test_app.py (extended) | headless UI: demo data → Processing → prepare split → Experiments → run comparison (all 3 levels, 2 epochs for speed) → session state has all three `ExperimentResult`s, all through the real `app.py` |

Also verified interactively in a real browser: prepared a 10,000-row split, ran a 5-epoch comparison of E0/E1/E2
(train times 2.3 s / 2.5 s / 7.8 s — E2's larger feature/token count costs more per epoch, as expected), saved
it (`experiments/experiment_v001.json`), then loaded the *same saved file* on the Results page after navigating
away — confirming persistence actually works across page loads, not just within one render. The demo artifact
was deleted afterwards; nothing was committed to `experiments/`.

**Not yet executed:** the app on the real 1.3M-row transaction file (only synthetic data was available in the
build environment, same limitation as Phases 2–5).

## Commands used

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Current git commit

The commit titled "Phase 6: evaluation + experiment runner …" (run `git log --oneline -1` for its exact hash;
amending this file changes the hash, so it is intentionally not pinned here).

## Known issues

- Everything carried over from Phases 2–5 still applies (including: a small/short-epoch comparison can make E2
  look artificially worse than E0/E1 simply because its larger model hasn't converged yet in few epochs — the
  Experiments page says this directly, but it's easy to miss).
- `run_comparison` is strictly sequential (one level fully trains before the next starts); fine for CPU/demo
  scale, would need reconsidering if experiments ever needed to run unattended for a long time or in parallel.
- No experiment/report currently gets written under `reports/` (only `experiments/`); the master prompt lists
  both directories in section 4's architecture without a stated distinction — `experiments/` was used for the
  raw manifest per section 19's "store... in reports/experiments/" wording; revisit if a distinct human-readable
  report format (e.g. Markdown, matching `cleaner.py`'s `save_cleaning_result` pattern for `data/processed/`) is
  wanted later.
- Saved experiments accumulate indefinitely in `experiments/` with no UI to delete one; acceptable for now
  (matches "never overwritten" — deleting is a deliberate, separate action) but worth a note if it becomes noisy.

## Next task (Phase 7 — Hugging Face / Nemotron / API adapters)

1. Read this file; run `pytest`.
2. Inspect `_pending/src/models/hf_adapter.py`, `api_adapter.py`, `registry.py` (**unverified drafts**) and
   `_pending/src/models/hf_backend.py` (research notebooks, Nemotron LoRA on a T4 — tested there, but on a GPU;
   verify what changes for CPU-only or no-GPU environments) against the current `ModelAdapter` interface
   (`src/models/base.py`) the same way Phase 5 reconciled `sanity_transformer.py` — check these drafts'
   constructor signature and method set match, don't assume.
3. Section 13's explicit guidance: Hugging Face support "can be implemented through an adapter" (weaker
   requirement than the built-in Transformer); Nemotron "should only be connected after verifying a suitable
   checkpoint/configuration" and "do NOT invent a model checkpoint" — do not fabricate a working Nemotron
   integration if no verified checkpoint/config is available in this environment; it is fine for `availability()`
   to honestly report it unavailable (section 14: CPU-only machines should be told fine-tuning large models
   isn't practical there, matching `recommendations()` in `src/utils/hardware.py`).
4. `PLANNED_BACKENDS` in `src/models/base.py` already has the right shape (`key`/`label`/`description`/`phase`/
   `implemented`) for the Model page's backend table; update entries as adapters actually become usable rather
   than marking `implemented: true` prematurely.
5. Whatever backend(s) end up implemented should plug into the *existing* `run_experiment`/`run_comparison`
   (Phase 6) and the Model page without those needing backend-specific changes — that's the point of the shared
   `ModelAdapter` interface; if a backend can't fit it, that's a design problem to flag, not route around.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [x] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [x] Phase 4: feature engineering + leakage checks
- [x] Phase 5: built-in CPU transformer
- [x] Phase 6: evaluation + experiment runner
- [ ] Phase 7: Hugging Face / Nemotron / API adapters
- [ ] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Later: generalization test on a non-fraud dataset
