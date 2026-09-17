# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 4 — feature engineering + leakage checks: complete.** Waiting for instruction before starting Phase 5.

## Cumulative summary (all phases so far)

**Phase 1 — Foundation (complete):** repository structure, `requirements.txt`, `config.yaml`, README; Streamlit
app with all 8 sections and a pipeline tracker; CPU/CUDA hardware detection; CSV/JSON/JSONL intake (upload,
local path, synthetic demo generator) with SHA-256 dataset IDs; generic schema detection and framework roles
(TARGET, ENTITY, IDENTIFIER, DATETIME, NUMERICAL, CATEGORICAL, TEXT, UNKNOWN) with candidate detection and
overrides; task detection (binary / multiclass / regression); profiling. 17 tests.

**Phase 2 — Quality and basic preprocessing (complete):** quality engine (five-dimension scorecard), stateless
cleaner, `basic_preprocessing` report, Quality Analysis UI page. 32 tests (was 17 after Phase 1).

**Phase 3 — E0 / E1 / E2 processing (complete):** subset selection, temporal (or random-fallback) split, sampling
with positive-class oversampling, E0 raw / E1 quality-processed (train-fitted missing-value/rare-category/numeric
transforms) / E2 feature-engineered (cyclical time + point-in-time-safe entity/merchant history features,
gracefully skipped without entity/time/amount) levels, Processing UI page. 46 tests (was 32 after Phase 2).

**Phase 4 — Feature engineering + leakage checks (complete, this session):** see below.

## Completed work (Phase 4, this session)

- Added `scikit-learn` to `requirements.txt` (needed for the leakage detector's univariate ROC-AUC checks; no
  other Phase 4 work needed it).
- Moved `_pending/src/quality/leakage_detector.py` into `src/quality/` **unchanged** (it was already "tested
  there" per `_pending/README.md`, unlike Phase 3's `levels.py` draft).
- Wired it into `DataPreparer` (`src/preprocessing/levels.py`) via a new `_run_leakage(level, frame, features)`
  method, called from `_build_e1` and `_build_e2` in place of Phase 3's honest "not yet implemented" message:
  - Reuses the **same train/validation cutoff** `prepare_split()` already computed (added a `cutoff` field to
    `split_info["boundaries"]`), so the leakage scan's "later period" is exactly the validation+test rows,
    consistent with the rest of the pipeline — rather than letting the detector pick its own independent
    holdout fraction. Falls back to `config.yaml`'s `leakage.holdout_fraction` only if no cutoff is available
    (e.g. split method configured to `random` even though a datetime column exists).
  - Runs on the level's **final** feature set (the transformed E1/E2 columns — `amt__robust`, `card_amount_mean_before`,
    etc. — not the raw source columns), since that is what a model would actually see.
  - **Skips gracefully** (empty findings, a clear step message) when no datetime column is selected: the
    detector inherently needs one to define a "later period," matching Phase 3's graceful-skip precedent for E2's
    history features.
  - Only two checks — `post_event_time` and `target_word_in_text` (structural, unambiguous leaks) — are
    auto-removed from the feature set at `high` risk. Every other finding (including near-perfect single-feature
    predictive power) is reported for human review only, per the detector's own `AUTO_REMOVE_CHECKS` comment:
    a legitimate signal can be very predictive, so predictive power alone is never grounds for automatic removal.
  - New `leakage:` section in `config.yaml` (`high_auc`, `medium_auc`, `instability_gap`, `holdout_fraction`).
- Processing page (`app.py`, `render_level`): new "Leakage findings" table (feature / risk / check / reason /
  recommendation, sorted by risk) under the point-in-time badge, for whichever level's `info["leakage"]` is set.
- **Incidental fix, found while verifying this in a real browser:** the Dashboard's "Go to data upload" button
  raised `StreamlitWidgetAlreadyInstantiatedError` (`st.session_state.nav = "Data Upload"` followed by
  `st.rerun()`, mutating a widget-bound key after that widget had already been instantiated this run — a
  pre-existing Phase 1 bug, not exercised by the headless tests since they navigate via the sidebar radio
  directly). Fixed with an `on_click` callback (`_goto`), which Streamlit runs before the rerun re-instantiates
  the widget. Out of Phase 4's brief but a one-line, zero-risk fix for a broken button a real user would hit
  immediately.

## Files changed this session

**New:** none
**Modified:** `src/preprocessing/levels.py` (`_run_leakage`, `AUTO_REMOVE_CHECKS`, cutoff in `prepare_split`),
`app.py` (leakage findings table; `_goto` callback fix), `config.yaml` (new `leakage:` section), `requirements.txt`
(`scikit-learn`), `tests/test_levels.py` (4 new tests, 1 updated assertion), `_pending/README.md`
**Moved (git mv, content unchanged):** `_pending/src/quality/leakage_detector.py` → `src/quality/`

## Tests completed

`pytest` → **50 passed** (was 46 at the end of Phase 3; 4 added this session, 1 assertion updated because the
message it checked for is now obsolete — leakage checks actually run instead of reporting "not yet implemented"). Run cold.

| File | Covers |
|---|---|
| test_levels.py (extended) | E2 leakage findings have the expected shape (`feature`/`check`/`leakage_risk`/`evidence`/`reason`/`recommendation`, valid risk levels) and the synthetic fraud-burst data triggers at least one; a deliberately injected structural text leak (`target_word_in_text`, values containing "confirmed fraud" only for the positive class) is found at `high` risk and **auto-removed** from E1's feature list; a genuinely predictive feature (the amount) is *not* auto-removed just for being predictive; leakage checks skip gracefully (empty findings, clear message) with no datetime column |
| test_app.py | unchanged from Phase 3 (still exercises Processing end to end; leakage findings now populate `info["leakage"]` as part of that flow, implicitly covered) |

Also verified interactively in a real browser (after the `_goto` fix): loaded demo data, Dashboard → Data upload
via the fixed button, Processing → prepared a split → built E2 (34 features) → point-in-time check passed →
"Leakage findings" table rendered 4 findings (`age_years__robust` and `card_amount_std_before` flagged
`temporal_instability` at medium risk, `cc_num` flagged `identity_proxy` and `entity_concentration` for review) —
correctly none auto-removed, all shown for human review, matching the "predictive power alone is never removed"
policy.

**Not yet executed:** the app on the real 1.3M-row transaction file (only synthetic data was available in the
build environment, same limitation as Phases 2–3).

## Commands used

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Current git commit

The commit titled "Phase 4: feature engineering + leakage checks …" (run `git log --oneline -1` for its exact
hash; amending this file changes the hash, so it is intentionally not pinned here).

## Known issues

- Everything carried over from Phase 3 (memory use scales with the sampled subset, not the full dataset; a
  handful of quality checks are still name-based; binary classification only) still applies.
- The leakage detector's univariate-power check runs a small (5-fold) AUC scan per feature; on the largest
  subset option (500,000 rows or "full") this could take noticeably longer than on the demo data — it has its
  own `max_rows` cap (600,000, hard-coded in `LeakageDetector`, not yet exposed in `config.yaml`) so it degrades
  by subsampling rather than by failing, but this hasn't been measured on the real 1.3M-row file.
  `temporal_instability`/`suspiciously_predictive` findings on features like `hour`/`day_of_week` are expected
  and were seen on the synthetic demo data: it plants fraud as a single time-clustered burst per card, which is
  a known property of the *synthetic generator*, not evidence of a pipeline bug — re-check this class of
  finding on the real dataset before treating it as informative.
- Auto-removal is deliberately narrow (two structural checks only); a feature that is a near-perfect target
  copy (`target_copy` / `suspiciously_predictive` at `high` risk) is flagged but *kept* in the feature set unless
  a human removes it — by design, but worth surfacing prominently in the UI if it comes up on the real dataset
  (currently just another row in the findings table, not specially highlighted).
- `check_point_in_time` now has two copies (one in `src/features/behavioral_features.py`, sklearn-free, used by
  `DataPreparer`; one still inside `src/quality/leakage_detector.py`, part of that module's original public API).
  Harmless duplication, not a shared dependency — noted in case a future refactor wants to deduplicate.

## Next task (Phase 5 — built-in CPU transformer)

1. Read this file; run `pytest`.
2. Inspect `_pending/src/models/sanity_transformer.py` (**unverified draft**, binary-only) and
   `_pending/src/models/base_draft.py`. Check what `PreparedLevel.frames` (train/validation/test DataFrames of
   `META + features`, from `src/preprocessing/levels.py`) needs to become the small Transformer's input: numeric
   features are already float; categorical features are still strings (rare-grouped, `__RARE__`/`__UNSEEN__`
   labels applied) and need an encoding step the draft may or may not already have.
3. Build the model interface from section 13 of the master prompt (`train`/`predict`/`evaluate`/`save`/`load`)
   with the built-in Transformer as the first, fully-implemented backend; CPU/CUDA device selection already
   exists in `src/utils/hardware.py` (Phase 1) — reuse it, don't reimplement.
4. Wire a "Model" page section that trains on a chosen level's (E0/E1/E2) train split and reports real, measured
   metrics only (section 27: no fabricated results) — this is also the first point where `PreparedLevel`'s
   `_weight` column (inverse-probability weights from the Phase 3 sample) should get used, for evaluation at the
   real class balance.
5. Keep it CPU-runnable on a small subset by default (section 2/15); don't default to the full dataset.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [x] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [x] Phase 4: feature engineering + leakage checks
- [ ] Phase 5: built-in CPU transformer
- [ ] Phase 6: evaluation + experiment runner
- [ ] Phase 7: Hugging Face / Nemotron / API adapters
- [ ] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Later: generalization test on a non-fraud dataset
