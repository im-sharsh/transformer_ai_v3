# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 2 — Quality analysis and basic preprocessing: complete.** Waiting for instruction before starting Phase 3.

## Cumulative summary (all phases so far)

**Phase 1 — Foundation (complete):** repository structure, `requirements.txt`, `config.yaml`, README; Streamlit
app with all 8 sections and a pipeline tracker; CPU/CUDA hardware detection; CSV/JSON/JSONL intake (upload,
local path, synthetic demo generator) with SHA-256 dataset IDs; generic schema detection and framework roles
(TARGET, ENTITY, IDENTIFIER, DATETIME, NUMERICAL, CATEGORICAL, TEXT, UNKNOWN) with candidate detection and
overrides; task detection (binary / multiclass / regression); profiling (overview, numeric stats, categorical
cardinality, target distribution, time range, entity stats, early leakage indicators). 17 tests.

**Phase 2 — Quality and basic preprocessing (complete, this session):** see below.

## Completed work (Phase 2, this session)

- Moved from `_pending/` into `src/` (unchanged, since their imports already matched this repo's layout):
  `src/quality/validators.py`, `src/quality/quality_engine.py`, `src/utils/audit.py`,
  `src/preprocessing/duplicates.py`, `src/preprocessing/cleaner.py`. Verified imports resolve.
- New: `src/preprocessing/basic_preprocessing.py` — the generic "Basic Preprocessing" layer from the
  architecture diagram (sits between Quality Analysis and E0/E1/E2). Runs `assess_quality` then
  `clean_dataset` on the whole dataset (stateless; no train/test split exists yet, so nothing here fits a
  statistic on data) and produces a `PreprocessingSummary` with only measured values (rows/columns
  before-after, duplicates detected/removed/retained, quarantined rows, invalid values repaired, categorical
  normalization count, datetime parsing/invalid count, identifier columns excluded, non-finite values
  remaining, processing time). Leakage checks are honestly reported as **"Not yet implemented (planned for
  Phase 4)"** rather than a fabricated "PASS".
- Quality Analysis page in the UI (previously a placeholder) now shows: the five-dimension quality scorecard
  (completeness, validity, consistency, uniqueness, integrity) with systematic-issue warnings and a full
  check table; a basic-preprocessing panel with configurable duplicate policy (remove/quarantine/retain),
  invalid-value policy (nullify/quarantine) and a drop-redundant-columns toggle; a transformation log,
  quarantined-rows and removed-duplicates previews, and a cleaned-data preview.
- Config: new `quality` and `preprocessing` sections in `config.yaml` (weights override, large-dataset
  warning threshold, all `CleaningConfig` toggles).
- Sidebar pipeline tracker and Dashboard status table updated to show Quality analysis / Basic preprocessing
  as done/ready/not-started based on real session state (not hard-coded).

## Files changed this session

**New:** `src/preprocessing/basic_preprocessing.py`, `tests/test_quality.py`, `tests/test_basic_preprocessing.py`
**Modified:** `app.py`, `config.yaml`, `tests/test_app.py`, `_pending/README.md`
**Moved (git mv, content unchanged):** `_pending/src/quality/{validators,quality_engine}.py` →
`src/quality/`; `_pending/src/utils/audit.py` → `src/utils/`; `_pending/src/preprocessing/{duplicates,cleaner}.py`
→ `src/preprocessing/`

## Tests completed

`pytest` → **32 passed** (was 17 at the end of Phase 1; 15 added this session). Run twice: once mid-session
after fixing three genuine failures, once cold at the very end before this update.

| File | Covers |
|---|---|
| test_quality.py (new) | quality score on clean data (with the deliberate synthetic systematic issue), injected problems (missing ids, duplicate ids, negative amounts, invalid/future timestamps) found with exact counts, genericity on the churn dataset, and **which checks are and aren't name-independent after renaming every column** |
| test_basic_preprocessing.py (new) | no-missing-values → "No imputation required"; never fabricates duplicate/quarantine counts on clean data; drops the systematically-redundant column and identifier columns; detects and removes real injected duplicates on the *churn* dataset (not just transactions); categorical normalization; duplicate policies (remove/retain) incl. the retain-vs-repeated-primary-key interaction; summary has only measured values; a full run with no transaction-specific column names present |
| test_app.py (extended) | headless UI: demo data → Quality Analysis → run quality → run basic preprocessing → Dashboard, all through the real `app.py` |

Also verified: `streamlit run app.py` starts cleanly (HTTP 200, `/_stcore/health` → `ok`, no
exception/traceback in the server log) after this session's changes, navigated pages confirmed via the
headless test above.

**Not yet executed:** the app on the real 1.3M-row transaction file (only synthetic data was available in
the build environment).

## Commands used

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Current git commit

The commit titled "Phase 2: quality analysis + basic preprocessing …" (run `git log --oneline -1` for its
exact hash; amending this file changes the hash, so it is intentionally not pinned here).

## Known issues

- Large files are read fully into memory (the 1.3M-row file needs ~1 GB RAM plus the file size during upload).
- Browser upload is limited to 1 GB (`.streamlit/config.toml`); use "Load from path" for bigger files.
- Quality analysis and basic preprocessing run on the **full** dataset, not a sample (duplicate and
  consistency counts would be misleading on a sample); above `quality.large_dataset_warn_rows` (300,000) the
  UI shows a patience warning but still runs on all rows. Not yet measured on the real 1.3M-row file.
- **New, found this session — genuine, documented limitation, not fixed:** a handful of quality-engine
  checks inherited from the original research code use column-**name** heuristics in addition to role-based
  detection — amount sign (`AMOUNT_HINT`), geographic coordinate range (`GEO_RANGES`, matches on a column
  name literally ending in `lat`/`long`), currency pattern (`CURRENCY_HINT`), and birth-date plausibility
  (schema detector's birth-date PII type is name-based: `(^dob$|birth)`, not value-based). On a dataset
  whose amount/coordinate/birth-date columns are named differently, these specific checks silently do not
  fire; the majority of checks (completeness, mandatory fields, duplicate ids, datetime parsing, id format,
  future-timestamp, entity-attribute-stability) are fully role-based and confirmed name-independent by
  `test_structural_quality_checks_do_not_depend_on_column_names`. Worth hardening before the Phase-8/later
  generalization test on a second dataset if that dataset has such columns under different names.
- The IBM Plex Sans web font needs internet access; without it the UI falls back to system fonts.

## Next task (Phase 3 — E0 / E1 / E2 pipelines, subsets, temporal split)

1. Read this file; run `pytest`.
2. Inspect `_pending/src/preprocessing/{missing_values,categorical,numerical,datetime_features,pipeline}.py`
   (train-fitted: imputation values, rare-category vocabulary, scaling parameters — need the split before
   they can be used) and `_pending/src/preprocessing/{split,sampling}.py`.
3. Consider `_pending/src/preprocessing/levels.py`, an **unverified draft** written for this exact E0/E1/E2
   brief before the phased plan — inspect carefully, do not trust it blindly, adapt to `DatasetRoles` /
   `schema_for_profiling` (it currently imports the old research `SchemaReport`-only path and a
   fraud-specific-looking `HistoryConfig` from `src/features/behavioral_features.py`, which is Phase 4).
   E0/E1 (no history features) should not need Phase 4 at all; keep E2's history-feature part optional/graceful
   per section 14's "if an entity or datetime column is unavailable, E2 should gracefully skip".
4. Build the Processing page: subset size selector (10k/50k/100k/200k/500k/full), temporal split (fallback to
   random with a clear reason when no datetime column), and E0/E1/E2 built from `basic_preprocessing` +
   the moved train-fitted modules, each reporting real rows/features/time — never fabricated.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [ ] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [ ] Phase 4: feature engineering + leakage checks
- [ ] Phase 5: built-in CPU transformer
- [ ] Phase 6: evaluation + experiment runner
- [ ] Phase 7: Hugging Face / Nemotron / API adapters
- [ ] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Later: generalization test on a non-fraud dataset
