# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 8 — polish, documentation, tests, Colab notebook: complete.** All 8 phases from the original brief are
now implemented. See "Next task" below for what's left (deliberately out of the original 8-phase scope).

## Cumulative summary (all phases)

**Phase 1 — Foundation:** repository structure, Streamlit app, hardware detection, CSV/JSON intake, generic
schema/task detection, profiling. 17 tests.

**Phase 2 — Quality and basic preprocessing:** quality engine, stateless cleaner, Quality Analysis UI page. 32 tests.

**Phase 3 — E0 / E1 / E2 processing:** subset selection, temporal/random split, sampling, three data levels,
Processing UI page. 46 tests.

**Phase 4 — Feature engineering + leakage checks:** the tested leakage detector wired into E1/E2. 50 tests.

**Phase 5 — Built-in CPU transformer:** `SanityTransformerAdapter`, Model UI page. 56 tests.

**Phase 6 — Evaluation + experiment runner:** `run_comparison` across E0/E1/E2, Experiments/Results UI pages. 62 tests.

**Phase 7 — Hugging Face / Nemotron / API adapters:** all three adapters behind one `ModelAdapter` interface,
Hugging Face and API verified for real (a real small model, a real local HTTP server); Nemotron honestly reports
unavailable without a GPU. 68 tests.

**Phase 8 — Polish, documentation, tests, Colab notebook (complete, this session):** see below.

## Completed work (Phase 8, this session)

- **README.md rewritten** against section 22's checklist (it was six phases stale, still describing Phase 1):
  research motivation, architecture (updated file tree, all 8 real UI sections), installation, local CPU usage
  (a concrete walk-through of what each page now actually does), Colab usage, dataset format, E0/E1/E2 (what
  each level actually contains, not just names), model (built-in Transformer + adapters, what was verified and
  what wasn't), evaluation, leakage prevention, experiments, limitations (rewritten to be phase-accurate, not
  "only Phase 1 is implemented"), future extensions.
- **`notebooks/colab_demo.ipynb` created** (was an empty placeholder directory before this session): load →
  profile → E0/E1/E2 → train the built-in Transformer → evaluate → compare E0 vs E1 vs E2 → optionally save the
  comparison, using the exact same `src/` modules as `app.py` (`DataPreparer`, `infer_roles`,
  `SanityTransformerAdapter`, `run_comparison`) — no logic duplicated into the notebook, matching section 21's
  explicit instruction. **Actually executed end to end** (not just written) via `nbclient` against this
  project's own venv, with the Colab-only git-clone setup cell skipped (not applicable to a local run) — this
  caught nothing wrong, which is itself the point: it confirms the notebook's cells are not stale copies that
  merely look plausible.
- Reviewed `_pending/`: `outliers.py` (flagging-only, fitted on training rows) remains the one genuinely
  deferred draft — its own catalog entry was already honestly labeled "3 or 6 (robustness)" and it was never
  required by any of the 8 phases as specified; wiring it in now would be a new pipeline feature, not "polish,"
  so it was left alone rather than added under Phase 8's name. `split.py` and `levels.py`'s original draft stay
  as historical reference (already explained in `_pending/README.md`).
- Confirmed test coverage (section 20) is already substantial across all phases (68 tests: ingestion, roles,
  quality, basic preprocessing, profiling, E0/E1/E2 levels, leakage, the built-in Transformer, HF/API backends,
  the experiment runner, config/hardware, and full headless Streamlit UI flows for every page) — no new tests
  were judged necessary for Phase 8 specifically; the notebook's real execution is this phase's own verification.

## Files changed this session

**New:** `notebooks/colab_demo.ipynb`
**Rewritten:** `README.md`

## Tests completed

`pytest` → **68 passed** (unchanged from Phase 7 — no test or source files under `src/`/`tests/` were touched
this session). Run cold, to confirm the documentation-only changes broke nothing.

`notebooks/colab_demo.ipynb` was executed end to end with `nbclient` (own verification method, not `pytest`) —
every cell after the Colab-only setup cell ran without error against this project's real synthetic demo data,
producing a genuine E0-vs-E1-vs-E2 comparison. The experiment file it wrote during that verification run
(`experiments/experiment_v001.json`) was deleted afterward — nothing was committed to `experiments/`.

## Commands used

```bash
pip install -r requirements.txt
pytest
streamlit run app.py
# Notebook verification (not part of the normal workflow):
pip install nbclient nbformat ipykernel
python -m ipykernel install --user --name tdi-test
python -c "import nbformat; from nbclient import NotebookClient; ..."  # executed notebooks/colab_demo.ipynb
```

## Current git commit

The commit titled "Phase 8: polish, documentation, tests, Colab notebook …" (run `git log --oneline -1` for its
exact hash; amending this file changes the hash, so it is intentionally not pinned here).

## Known issues

Everything listed in each phase's own section above still applies; nothing here changes prior known issues,
since Phase 8 was documentation and a notebook, not pipeline code. The most consequential ones, repeated once
more since this is the file meant to be read first:

- Never run on the real ~1.3M-row transaction file from the original brief — only synthetic demo data was
  available in this project's build/test environment.
- Binary classification only through Processing/Model/Experiments.
- Nemotron never run (no GPU, no verified checkpoint in this environment) — `availability()` reports this
  honestly rather than the app claiming a run that didn't happen.
- PyTorch reproducibility is machine-local, not an absolute cross-platform guarantee (see Phase 5's notes).

## Next task (beyond the original 8 phases)

The brief's own phase list (section 24) ends at Phase 8; section 23 ("Future extensions") and this project's own
accumulated known issues are the honest list of what's next, roughly in the order they'd matter for someone
actually trying to use this on their own data:

1. **Run it on the real dataset.** Every phase's tests and interactive verification used synthetic data; the
   real ~1.3M-row file (section 3 of the brief) has never been loaded. Do this before trusting any of this
   project's numbers on real fraud data — file size, memory behavior, and quality-check findings could all
   surface something the synthetic generator doesn't.
2. **Generalization test on a non-fraud dataset**, through the *full app* (upload → Processing → Model), not
   just the unit-level `churn` fixture tests that already exist (`tests/test_levels.py`,
   `test_basic_preprocessing.py`). This is the strongest test of "detected or configured, never assumed" (README
   §1) actually holding up end to end.
3. **Multiclass / regression through Processing.** `DatasetRoles`/Profiling already detect and report the task;
   `DataPreparer` is deliberately binary-only right now (a documented Phase 3 decision, not an oversight).
4. `_pending/src/preprocessing/outliers.py` — wire it in if outlier *handling* (not just the profiling counts
   already shown) turns out to matter on the real dataset.
5. Section 23's other items (database connectors, streaming, richer experiment tracking, drift detection, PII
   scanning beyond detection/masking, deployment) as and when they're actually needed — this project has
   consistently avoided building ahead of a stated requirement, and that's a deliberate choice, not a gap to
   rush closed.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [x] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [x] Phase 4: feature engineering + leakage checks
- [x] Phase 5: built-in CPU transformer
- [x] Phase 6: evaluation + experiment runner
- [x] Phase 7: Hugging Face / Nemotron / API adapters
- [x] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Beyond the 8 phases: run on the real dataset; generalization test on a non-fraud dataset through the full
      app; multiclass/regression through Processing; the rest of section 23
