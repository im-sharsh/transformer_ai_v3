# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**All 8 original phases complete, plus a code audit (this session) with 4 of its recommended fixes implemented.**
The audit was requested to answer one question precisely: what does the built-in Transformer actually receive
at each of E0/E1/E2, and is any of the "processing" in E1/E2 illusory once it passes through the tokenizer? The
audit read the pipeline code end to end and verified its claims by running scripts against the real repository
(not by inspection alone) before any code was changed. Full audit write-up is in the conversation history; the
short version and what was actually fixed is below.

## Audit findings and fixes (this session)

Five findings, verified empirically (not assumed) by running the real pipeline on synthetic demo data:

1. **Fixed.** E1/E2 passed three numeric columns per original feature (`__robust`, `__log`, `__bin`) into the
   tokenizer, which quantile-bins every numeric feature. Quantile binning is invariant under monotonic
   transforms, so `__robust` and `__log` produced *exactly* the same token as directly binning the raw value
   (measured: 100% exact match). Three redundant tokens, zero net new information. Now only `__robust` is a
   model input; `__log`/`__bin` are still computed (useful for a future continuous-representation experiment)
   but not selected. `src/preprocessing/levels.py`.
2. **Documented, not changed.** E0 is not literally raw: the same identifier/PII/quasi-identifier exclusion
   policy applies to every level (the code already says so in its own docstring). Worth being precise about
   in any write-up: E0 means "unprocessed feature values, standard exclusions applied," not "the CSV verbatim."
3. **Measured, not yet fixed.** E0's raw timestamp column collapses to the tokenizer's "unknown" token for
   ~100% of test rows (measured directly): a per-second timestamp is almost entirely unique, so every training
   value falls below `min_category_count` and no vocabulary is ever built for it. E0 effectively has *no* time
   signal at all, not "raw" time signal — a bigger and more mechanical gap than "raw vs. engineered." Left as
   a documented fact for now; fixing it (parsing E0's datetime to a numeric epoch) is future work (see below).
4. **Fixed.** The sampler's inverse-probability `_weight` (correcting for positive-class oversampling, e.g.
   10% sampled vs ~0.6% real) was used only in evaluation metrics, never in the training loss — the model was
   trained at the oversampled rate and only *scored* as if at the real rate. `models.sanity_transformer.
   class_weighting` (new config key, default `sample_weight`) reuses the existing `_weight` column as a
   per-example loss weight; `none` reproduces the exact previous behaviour for comparison.
   `src/models/sanity_transformer.py`.
5. **Fixed.** `previous_transactions()` (the entity's last K transactions: hours ago / amount / hour /
   category) was fully implemented and leakage-tested in `behavioral_features.py` but had zero callers
   anywhere in the repository — confirmed by a repo-wide search, not filtered out, just never wired in. This
   is the single highest-value fix in the audit: a separate research track (fine-tuning a language model on
   the equivalent text representation, outside this app) found it to be the largest effect measured in that
   whole project. Now wired into E2 by default (`features.include_sequence_features: true`,
   `features.sequence_length: 3`, both configurable). `src/preprocessing/levels.py`.

Also added: `run_comparison_multiseed()` / `aggregate_seeds()` in `src/evaluation/experiment.py`, wired into
the Experiments UI page — the project's own brief requires reporting results as mean ± std across seeds
(42/123/456), and `run_comparison` had only ever run a single seed before this.

**Not yet done from the audit's recommended order** (see "Next task" below): R2/R3 (continuous numeric
projection instead of quantile bins — the "most important experiment" per the audit, deliberately sequenced
after the redundancy fix above so it isn't confounded by it), C1 (parse E0's datetime instead of leaving it as
the near-total-information-loss categorical string described in finding 3), T1 (feed cyclical time features
continuously so their periodicity survives), S1 (extend finding 5's fix into the text/LLM representation path,
`text_builder.py`, which currently has the same dead-code gap).

## Tests completed (this session)

`pytest` → **77 passed, 1 skipped** (was 67 passed, 1 skipped before this session; the skip is unchanged —
no network access to the Hugging Face Hub in this environment). Run cold after every one of the four commits
below, not just once at the end.

12 new tests: `test_e1_numeric_representation_is_not_redundant` (regression guard for finding 1),
`test_e2_includes_point_in_time_verified_sequence_features_by_default` /
`test_e2_sequence_features_can_be_disabled` / `test_sequence_features_are_leakage_safe_on_a_first_transaction`
(finding 5), `test_class_weighting_default_uses_sample_weight_and_can_be_disabled` /
`test_class_weighting_rejects_unknown_value` (finding 4), `test_run_comparison_multiseed_...` /
`test_aggregate_seeds_...` (×2) (multi-seed harness, library-level), `test_demo_flow_experiments_multiseed`
(the same harness exercised through the real Streamlit UI, not just the function). One existing test
(`test_e1_transforms_numeric_features_instead_of_using_raw_values`) was updated because it had asserted the
old, buggy triple-numeric behaviour as correct.

Four commits, one per fix, each with the full suite run cold before committing:
`87d3547` (findings 1+5), `54d6cfe` (finding 4), `548b34e` (multi-seed harness). Run `git log --oneline` for
the exact current head.

## Measured, not just implemented: what the audit fixes actually did (this session, after the fixes above)

Per the brief's own instruction not to claim a change is beneficial without running the experiment, I used the
new multi-seed harness (3 seeds, synthetic demo data, `rows=6000` so oversampling actually triggers — `rows=
"full"` barely oversamples on this synthetic generator and would not have exercised finding 4 at all) to check
findings 4 and 5 empirically rather than leaving them as assumed improvements. **One of them failed this check
and was reverted; this is exactly the kind of result the audit exists to catch, not a problem to hide:**

- **Finding 4 fix (`class_weighting: sample_weight`) measurably regressed performance and was reverted.**
  Isolated on E0 alone (no sequence features involved, so nothing else could explain it):
  PR-AUC 0.470±0.287 (`none`) vs 0.144±0.179 (`sample_weight`) — a large, consistent drop, confirmed the same
  direction on E1 and E2 too. Diagnosis: at ~0.4% true prevalence, reweighting the loss back toward the true
  rate makes the rare class's gradient contribution negligible relative to the abundant majority class,
  undermining exactly what oversampling exists to provide. The original diagnosis (`_weight` computed but
  never used in training — a real train/eval mismatch) still stands; this specific fix for it doesn't survive
  contact with severe imbalance. Default reverted to `none`; `sample_weight` kept available, not recommended.
  Full numbers and diagnosis are in the "Revert class_weighting default…" commit message.
- **Finding 5 (sequence features in E2) is not yet a clear win on synthetic data, but looks stabilizing.**
  With `class_weighting=none` isolating just this change: PR-AUC 0.858±0.124 (off) vs 0.806±0.012 (on) — the
  means are within each other's noise band (3 seeds is not enough to call this either way), but sequence
  features cut seed-to-seed variance roughly 10×. Left enabled by default on that basis (more reliable, not
  measurably worse), but **this is a much weaker claim than "the single largest effect measured in [the
  separate Nemotron] research" quoted earlier in this file** — that result was on a fine-tuned 4B language
  model reading real transaction sequences as text, not this small from-scratch tabular transformer on
  synthetic data; the two are not directly comparable and this session's measurement should not be read as
  confirming that other result transfers here.

**Takeaway for anyone continuing this project: measure before defaulting, the same discipline that caught
finding 4's regression should be applied to every future config-default change, including the ones already
made (finding 5's default is a judgement call on weak evidence, not a settled result).**

## Original phase history

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

## Files changed in Phase 8's own session

**New:** `notebooks/colab_demo.ipynb`
**Rewritten:** `README.md`

## Tests completed at the end of Phase 8 (before the audit)

`pytest` → **68 passed** (unchanged from Phase 7 — no test or source files under `src/`/`tests/` were touched
that session). Run cold, to confirm the documentation-only changes broke nothing. (For the audit session's own
test count — 77 passed, 1 skipped — see "Tests completed (this session)" near the top of this file.)

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

The commit titled "Update PROJECT_STATUS.md: document the measured class_weighting revert …" (run
`git log --oneline -1` for its exact hash; amending this file changes the hash, so it is intentionally not
pinned here). Five commits this session in order: "Audit fixes 1/2: …" (findings 1+5), "Audit fix 3/4: …"
(finding 4), "Audit fix 4/4: …" (multi-seed harness), "Update PROJECT_STATUS.md: document audit findings…",
"Revert class_weighting default to 'none': measured regression, not assumed benefit" — each with the full
suite run cold before committing.

## Known issues

Everything listed in each phase's own section above still applies. Additions from the audit session:

- Never run on the real ~1.3M-row transaction file from the original brief — only synthetic demo data was
  available in this project's build/test environment.
- Binary classification only through Processing/Model/Experiments.
- Nemotron never run (no GPU, no verified checkpoint in this environment) — `availability()` reports this
  honestly rather than the app claiming a run that didn't happen.
- PyTorch reproducibility is machine-local, not an absolute cross-platform guarantee (see Phase 5's notes).
- **`models.sanity_transformer.class_weighting` now defaults to `sample_weight`, not the old unweighted
  behaviour.** Any numbers from before this session's fixes are not directly comparable to numbers after —
  re-run rather than diff old saved `experiments/*.json` files against new ones.
- **E1/E2's numeric feature count per column dropped from up to 3 to 1** (audit finding 1). A saved model
  from before this session will not load against the current tokenizer/feature shape.
- **E2 now includes ~12 extra sequence-feature positions by default** (`features.sequence_length: 3` ×
  4 sub-features, category only when a category column is set) — training is correspondingly slower per
  epoch; `features.include_sequence_features: false` restores the old, smaller E2.
- Audit findings 2 and 3 (E0's near-total timestamp information loss; cyclical time features losing their
  periodicity once quantile-binned) are documented but **not yet fixed** — see "Next task."

## Next task (continuing the audit's recommended order)

Full audit (architecture trace, 6 verified findings, prioritized experiment matrix) is in the conversation
history that produced this session's commits, not duplicated here. What's left from it, in the order the
audit recommended:

1. **R2/R3 — continuous numeric representation.** The audit's own "most important experiment": replace or
   augment the single quantile-bin token per numeric column (the corrected finding-1 state) with either a
   continuous linear projection, or a continuous projection concatenated with a coarse bin embedding. This
   needs a `TabularTokenizer`/`SanityTransformerModel` change (a numeric column can no longer be *only* a
   token id), not just a `levels.py` change like the four fixes in this session — hasn't been started.
2. **C1 — parse E0's datetime.** Currently a full-precision string that collapses to "unknown" for ~100% of
   rows (finding 3, measured, not yet fixed). Feed it as a numeric epoch (or similar) even in the "minimal
   processing" baseline, so E0-vs-E1/E2 comparisons reflect processing quality, not "has a time feature at
   all vs. doesn't."
3. **T1 — feed cyclical features continuously.** Depends on R2/R3's continuous path existing first; otherwise
   `hour_sin`/`hour_cos` keep losing their periodicity to quantile-bin discretization (finding 2/3).
4. **S1 — extend finding 5's fix to the text/LLM path.** `text_builder.py` (feeds Hugging Face/Nemotron/API
   adapters) has the same gap `levels.py` had: it renders `prepared.numeric`/`prepared.categorical`, which
   still doesn't include `previous_transactions()` for the *text* representation even after this session's
   fix (that fix only reached the tabular/`TabularTokenizer` path via E2's feature list, which
   `text_builder.py` also reads — so it likely already inherited the fix for free; verify this rather than
   assume, then add `format_previous_transactions()` as an explicit text field if the plain field-by-field
   rendering of `prevK_*` numeric columns isn't as good as the purpose-built text formatter).
5. Add `class_weighting: pos_weight_natural` (§12's Experiment 4: natural sampling + `pos_weight`, as opposed
   to this session's oversampling + sample-weight fix) as a second, directly comparable option, if L1 vs L2
   turns out to matter empirically once run.
6. **Partially done this session** for findings 4 and 5 specifically (see "Measured, not just implemented"
   above) — finding 4's fix was measured to regress performance and reverted; finding 5 is measured but
   inconclusive (3 seeds, small synthetic data). Finding 1 was not A/B tested via the harness (its fix isn't
   config-reversible — the redundant columns were removed from the code, not toggled), but was proven
   lossless analytically (100% exact token match, see the original audit). Once items 1–5 above land, extend
   this same measure-before-trusting discipline to them, and re-run the full E0/E1/E2 comparison, not just
   the isolated finding-4/5 checks done so far.
7. **Run it on the real dataset** — still never done, unchanged from before this session.
8. **Generalization test on a non-fraud dataset**, through the *full app*, not just the `churn` fixture unit
   tests that already exist.
9. Multiclass / regression through Processing; `_pending/src/preprocessing/outliers.py`; section 23's other
   items — unchanged from before this session, still deliberately deferred.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [x] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [x] Phase 4: feature engineering + leakage checks
- [x] Phase 5: built-in CPU transformer
- [x] Phase 6: evaluation + experiment runner
- [x] Phase 7: Hugging Face / Nemotron / API adapters
- [x] Phase 8: polish, documentation, tests, Colab notebook
- [x] Audit: findings 1, 4, 5 fixed (redundant numeric triple, unweighted loss under oversampling, dead
      sequence-feature code); multi-seed comparison harness added
- [ ] Audit: findings 2, 3 fixed (E0 timestamp information loss; cyclical features losing periodicity);
      R2/R3 (continuous numeric representation); S1 (extend finding 5 to the text/LLM path)
- [ ] Beyond the 8 phases: run on the real dataset; generalization test on a non-fraud dataset through the full
      app; multiclass/regression through Processing; the rest of section 23
