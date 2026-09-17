# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 3 — E0 / E1 / E2 pipelines, subsets, temporal split: complete.** Waiting for instruction before starting Phase 4.

## Cumulative summary (all phases so far)

**Phase 1 — Foundation (complete):** repository structure, `requirements.txt`, `config.yaml`, README; Streamlit
app with all 8 sections and a pipeline tracker; CPU/CUDA hardware detection; CSV/JSON/JSONL intake (upload,
local path, synthetic demo generator) with SHA-256 dataset IDs; generic schema detection and framework roles
(TARGET, ENTITY, IDENTIFIER, DATETIME, NUMERICAL, CATEGORICAL, TEXT, UNKNOWN) with candidate detection and
overrides; task detection (binary / multiclass / regression); profiling (overview, numeric stats, categorical
cardinality, target distribution, time range, entity stats, early leakage indicators). 17 tests.

**Phase 2 — Quality and basic preprocessing (complete):** quality engine (five-dimension scorecard), stateless
cleaner (type normalization, duplicate/invalid-value policies, categorical normalization), `basic_preprocessing`
report, Quality Analysis UI page. 32 tests (was 17 at the end of Phase 1).

**Phase 3 — E0 / E1 / E2 processing (complete, this session):** see below.

## Completed work (Phase 3, this session)

- Inspected `_pending/`: moved the train-fitted preprocessing modules into `src/` unchanged
  (`missing_values.py`, `categorical.py`, `numerical.py`, `datetime_features.py`, `pipeline.py`, `sampling.py`)
  and `src/features/behavioral_features.py` unchanged, plus added a self-contained `check_point_in_time`
  function to it (copied from the Phase-4 `leakage_detector.py`, which needs `scikit-learn`; this one doesn't,
  so E2's history features can be verified now without pulling in a Phase-4 dependency early).
- New: `src/preprocessing/levels.py` — **rewritten**, not a straight move, from `_pending/src/preprocessing/levels.py`
  (an unverified draft). Differences from the draft, and why:
  - `infer_roles` takes the app's already-confirmed `DatasetRoles` (target / entity / datetime chosen on the
    Profiling page) instead of re-detecting them from `SchemaReport`, per the Phase-2 handoff note. It still
    resolves amount / category / merchant by name hint and applies the same identifier / personal-data /
    quasi-identifier exclusion policy to every level (E0, E1, E2). `find_quasi_identifiers` was **reimplemented**
    as a small local helper rather than importing `_pending/src/representation/transaction_formatter.py`
    (phase 5/7, LLM text formatting — importing it now would pull in unrelated scope for one 15-line function).
  - E1's numeric model inputs are the **fitted transforms** (`__robust`, `__log` where skewed, `__bin`) rather
    than the raw values the draft used: the draft computed these transforms but never put them in the feature
    list, which would have made E1's numeric treatment (section 9 of the brief) invisible to any model trained
    on it — defeating the E0-vs-E1 comparison this whole project measures.
  - E2 adds cyclical `hour_sin/cos`, `day_of_week_sin/cos` alongside the raw values (the brief asks for
    cyclical time representations; the draft only had the raw ones).
  - E2's history features (`src/features/behavioral_features.py`, card/merchant point-in-time aggregates)
    **gracefully skip** with a reported reason when no entity, time or amount column is set, instead of raising
    — the draft raised. Verified with `check_point_in_time` (recomputes history from truncated per-entity data
    and compares) rather than the full `LeakageDetector` (Phase 4, needs `scikit-learn`).
  - Full feature-level leakage scanning is honestly reported as **"not yet implemented (planned for Phase 4)"**
    for E1 and E2, matching the Phase-2 precedent for the same claim in `basic_preprocessing`.
  - Config keys were split to avoid colliding with Phase 2's `preprocessing:` section (stateless `CleaningConfig`
    fields): new `sampling:`, `processing:` (train-fitted numeric/categorical config) and `features:` (history
    window sizes) sections in `config.yaml`; `subsets:` and `split:` (already present as Phase-1 placeholders)
    are now live. `data:` gained optional `amount_column` / `category_column` / `merchant_column` hints.
  - `src/preprocessing/split.py` (needs an explicit per-dataset `train_end` date) was **not** wired in; the
    quantile-of-fractions temporal split already sketched in `config.yaml`'s `split:` section works on any
    dataset without per-dataset dates, so `DataPreparer.prepare_split` uses that instead. `outliers.py` (flag-only
    outlier handling) was left for Phase 6 per `_pending/README.md`'s own "3 or 6" note — not needed to get
    E0/E1/E2 working.
- New Processing page in `app.py` (`page_processing`, `render_level`): resolved-roles/exclusions panel, subset
  size selector (from `config.yaml`'s `subsets.options`), a "Prepare split and sample" step showing the
  boundaries and per-split rows/positives, and three tabs (E0/E1/E2) each with its own "Build" button showing
  feature count, rows, positives, the full step log, the point-in-time badge (E2) and a preview of the train
  split. Sidebar pipeline tracker and Dashboard status table now reflect real `st.session_state.levels` state
  instead of a hard-coded "planned" row.
- `_pending/README.md` updated to record what moved and why the draft was adapted rather than used as-is.

## Files changed this session

**New:** `src/preprocessing/levels.py`, `tests/test_levels.py`
**Modified:** `app.py` (Processing page + session state + status rows), `config.yaml` (new `sampling`, `processing`,
`features` sections; `data` amount/category/merchant hints; activated `subsets`/`split`), `tests/test_app.py`
(new `test_demo_flow_processing`), `_pending/README.md`
**Moved (git mv, content unchanged except the addition noted above):**
`_pending/src/preprocessing/{missing_values,categorical,numerical,datetime_features,pipeline,sampling}.py` →
`src/preprocessing/`; `_pending/src/features/behavioral_features.py` → `src/features/`

## Tests completed

`pytest` → **46 passed** (was 32 at the end of Phase 2; 14 added this session: 13 in `test_levels.py`, 1 new
headless flow in `test_app.py`). Run cold, after all changes.

| File | Covers |
|---|---|
| test_levels.py (new) | `infer_roles`: amount/category/merchant resolution, identifier/PII exclusion, and the quasi-identifier catch (a `job` column that's unique per card despite not being PII-typed) vs. legitimate low-cardinality columns kept; binary-target requirement; temporal split boundaries and chronology; random-split fallback without a datetime column; split reproducibility for a fixed seed; E0 keeps raw values incl. a column E1 will later drop; E1 uses fitted transforms (`__robust`/`__log`/`__bin`) not raw values, never includes an absolute timestamp, reports leakage checks as not yet implemented; E1 missing-value indicator; the E0-vs-E1 contrast (redundant epoch column kept in E0, dropped in E1) that the research question depends on; E2 cyclical time + history features present, diagnostic-only `card_txn_count_before` correctly excluded, point-in-time check passes on the real demo data; E2 gracefully skips history features without entity/amount and skips time features without a datetime column, on the generic (non-transaction) churn dataset; level caching |
| test_app.py (extended) | headless UI: demo data → Processing → prepare split and sample → build E0/E1/E2 → Dashboard, all through the real `app.py`, checking the point-in-time result too |

Also verified interactively: `streamlit run app.py` on the synthetic demo data (27,223 rows) through the actual
browser — loaded demo data, opened Processing, expanded "Resolved roles and exclusions" (15 columns correctly
excluded, `job` caught as a quasi-identifier), prepared a 10,000-row temporal split (7,000/1,500/1,500, 0
unassigned), and built all three levels (E0: 7 features; E1: 9 features; E2: 34 features, point-in-time check
passed on 100 rows). Preview table confirmed transformed columns (`amt__robust`, `amt__log`, `amt__bin`,
`age_years__robust`, ...), not raw values.

**Not yet executed:** the app on the real 1.3M-row transaction file (only synthetic data was available in the
build environment, same limitation as Phase 2).

## Commands used

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Current git commit

The commit titled "Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split …" (run `git log --oneline -1` for
its exact hash; amending this file changes the hash, so it is intentionally not pinned here).

## Known issues

- Everything carried over from Phase 2 (large files read fully into memory; 1GB browser-upload limit; quality/
  cleaning run on the full dataset, not a sample; a handful of quality checks are still name-based rather than
  value-based — see Phase 2's note in git history) still applies, now also to the E1/E2 cleaning + transform
  step, which runs on the *sampled* rows (bounded by the subset size), not the full dataset.
- `config.yaml`'s `subsets.cpu_default` (10,000) must be one of `subsets.options`; the UI falls back to the
  smallest option if a future edit breaks that invariant, rather than erroring.
- Leakage checks are still not implemented as a full feature-level scan (Phase 4); only E2's history features
  are verified, via point-in-time recomputation, not a general leakage detector.
- `DataPreparer` is binary-classification only (matches the project's fraud use case and the actual dataset);
  multiclass/regression targets are rejected with a clear error pointing back to the Profiling page.
- The subset draw (`SampleConfig`/`draw_sample`) oversamples the positive class below "full" size using fixed
  shares from `config.yaml` (`sampling:`); this is not user-configurable from the UI yet, only from the config file.
- Point-in-time verification samples up to 10 entities (≤100 rows); it is a real check on real data, not a proof
  for every row — a full leakage scan (Phase 4) is the stronger guarantee.

## Next task (Phase 4 — feature engineering + leakage checks)

1. Read this file; run `pytest`.
2. Add `scikit-learn` to `requirements.txt` and move `_pending/src/quality/leakage_detector.py` into `src/quality/`,
   adapting it to `DataPreparer`'s frames (it currently expects a single event-time column and a target on the
   full frame — check how that maps onto E1/E2's already-split, already-sampled `PreparedLevel.frames`).
3. Wire a full feature-level leakage scan into the E1/E2 "Leakage checks" step (currently honestly reported as
   "not yet implemented"), replacing that message with real findings; keep `check_point_in_time` for history
   features specifically (it isn't superseded — it verifies something the general scan doesn't: that the
   *computation* is point-in-time-safe, not just that the *values* don't look suspiciously predictive).
   Decide, and document, which findings (if any) are auto-removed vs. only reported for human review — Phase 2/3
   precedent is to auto-remove only structural, unambiguous leaks and report the rest.
4. Consider `_pending/src/preprocessing/outliers.py` (flagging-only; fitted on training rows) if outlier
   handling is wanted in E1/E2 now rather than Phase 6 — check with the user/spec first, it wasn't in the Phase 3
   brief.
5. Add a leakage-report UI section (Processing or a new tab) showing findings by risk level, matching the
   existing `render_quality`/`render_preprocessing` display conventions.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [x] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [ ] Phase 4: feature engineering + leakage checks
- [ ] Phase 5: built-in CPU transformer
- [ ] Phase 6: evaluation + experiment runner
- [ ] Phase 7: Hugging Face / Nemotron / API adapters
- [ ] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Later: generalization test on a non-fraud dataset
