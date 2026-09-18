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

`pytest` → **87 passed, 1 skipped** (was 67 passed, 1 skipped before this session; the skip is unchanged —
no network access to the Hugging Face Hub in this environment). Run cold after every commit. Note: partway
through this session the suite grew large enough that a single `pytest` invocation exceeds this sandbox's
per-command time limit (~300s); from the R2/R3 work onward it was run **per test file** instead (still cold,
still every file, just not one combined process) — see each file's pass count below.

23 new tests across the session: `test_e1_numeric_representation_is_not_redundant` (finding 1 regression
guard), `test_e2_includes_point_in_time_verified_sequence_features_by_default` /
`test_e2_sequence_features_can_be_disabled` / `test_sequence_features_are_leakage_safe_on_a_first_transaction`
(finding 5), `test_class_weighting_option_actually_changes_training_and_can_be_selected` /
`test_class_weighting_rejects_unknown_value` (finding 4), `test_run_comparison_multiseed_...` /
`test_aggregate_seeds_...` (×2) + `test_demo_flow_experiments_multiseed` (multi-seed harness),
`test_continuous_numeric_mode_preserves_magnitude_that_quantile_bin_discards` /
`test_continuous_mode_with_coarse_bins_keeps_both_signals` /
`test_sanity_transformer_learns_with_continuous_numeric_mode` /
`test_continuous_mode_forward_requires_numeric_tensors` (R2/R3), `test_e0_time_epoch_is_opt_in_and_does_not_
replace_the_raw_string` (C1). Two existing tests were updated because they asserted behaviour later found to
be wrong or renamed: `test_e1_transforms_numeric_features_instead_of_using_raw_values` (had asserted the old,
buggy triple-numeric output as correct) and the `class_weighting` default test (had asserted `sample_weight`
as the default, before it was measured and reverted).

Per-file pass counts (this exact, freshly re-run breakdown): `test_ingestion.py` + `test_roles.py` +
`test_profiling.py` + `test_config_hardware.py` + `test_quality.py` + `test_basic_preprocessing.py` +
`test_backends.py` + `test_levels.py` + `test_experiment.py` + `test_app.py` → 72 passed, 1 skipped
(re-verified: `test_app.py` alone → 8 passed, up from 7, after adding and fixing
`test_generalization_full_app_flow_on_a_structurally_different_dataset`);
`test_model.py` → 15 passed (was 13, +2 for T1's `continuous_features`). **Total: 72 + 15 = 87 passed, 1 skipped.**

Commits this session, each with tests run (fully or per-file as above) before committing: `87d3547`
(findings 1+5), `54d6cfe` (finding 4), `548b34e` (multi-seed harness), `30f7639` (class_weighting revert),
`de86c62` (R2/R3), plus the C1 and documentation commits — run `git log --oneline` for the exact current head.

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

## Measured, not just implemented (continued): R2/R3 and the finding-3 fix

Two more items implemented this session, each measured with the same 3-seed harness before drawing any
conclusion (synthetic demo data, `rows=6000`, 10 epochs, patience 4) — no defaults were changed on the
strength of a single measurement, per the `class_weighting` lesson above.

**R2/R3 — continuous numeric representation** (`representation.numeric_mode`, `src/representation/
tabular_tokenizer.py`, `src/models/sanity_transformer.py`). Opt-in; `quantile_bin` (the original behaviour)
stays the default. `continuous` standardizes numeric values (train-fit median/IQR) and gives each numeric
feature a learned per-feature (weight, bias) affine embedding of the real value, instead of a lookup table
keyed by quantile bin; `numeric_coarse_bins > 0` additionally keeps a coarse bin token alongside the
continuous value at the same position (R3). All plumbed end to end (train/predict/save/load); a model built
without `numeric_positions` is byte-for-byte the original architecture (verified: the full suite's
`quantile_bin`-mode tests are unaffected).

Measured (E0/E1/E2, 3 seeds):

| Level | quantile_bin (current default) | continuous (R2) | continuous + coarse bin (R3) |
|---|---|---|---|
| E0 | 0.513 ± 0.042 | **0.620 ± 0.038** | 0.550 ± 0.083 |
| E1 | 0.470 ± 0.031 | **0.620 ± 0.035** | 0.568 ± 0.036 |
| E2 | **0.861 ± 0.033** | 0.674 ± 0.158 | **0.875 ± 0.029** |

A genuinely mixed, level-dependent result, reported as such rather than cherry-picked: R2 clearly helps E0/E1
but clearly hurts E2 (both mean and 5x the variance) — E2 has ~20+ numeric-ish positions (time, history
aggregates, sequence features) and a shared-architecture continuous embedding across that many heterogeneous
signals may be harder for this tiny model to fit than discrete bins are. **Diagnosed** (this session,
continued): E2-specific engineered ratio/z-score features (`card_amount_ratio_to_mean`, `card_amount_zscore`,
`card_amount_sum_24h`, `prev1/2/3_amount`, …) have standardized values up to |z|≈30 under train-fit median/IQR
scaling — a handful of unclipped outliers feeding straight into the linear numeric embedding. Added
`representation.numeric_clip` (default `5.0`, only active when `numeric_mode: continuous`) to cap this.

Re-measured (3 seeds, real code — not the diagnostic monkeypatch used to find this):

| Level | quantile_bin (default) | continuous, no clip | continuous, clip=5 | continuous + coarse bin, clip=5 |
|---|---|---|---|---|
| E0 | 0.513 ± 0.042 | **0.620 ± 0.038** | 0.591 ± 0.032 | 0.547 ± 0.066 |
| E1 | 0.470 ± 0.031 | **0.620 ± 0.035** | 0.603 ± 0.027 | 0.358 ± 0.304 (!) |
| E2 | 0.861 ± 0.033 | 0.674 ± 0.158 | 0.838 ± 0.055 | **0.860 ± 0.012** |

Clipping confirms the diagnosis (E2 recovers most of the way, variance back near baseline) but is **not a
clean, uniform win**: it slightly *reduces* R2's gain on E0/E1 (clipping discards some of the real signal
that helped there, since E0/E1 don't have E2's extreme-outlier columns), and combining it with coarse bins
(R3) produced a severe, unstable collapse on E1 in this run (0.358 ± 0.304 — one seed likely failed badly;
not further diagnosed). **This is exactly the same lesson as `class_weighting`, repeating**: a config
combination that looks good on the metric you're focused on (E2) can hide instability elsewhere in the
matrix. No default changed. `numeric_mode` stays `quantile_bin`.**C1 — E0's timestamp fix** (`data.e0_add_time_epoch`, `src/preprocessing/levels.py`). Opt-in, default `false`.
Adds a numeric epoch-seconds column to E0 alongside the unchanged raw timestamp string, closing the ~100%
"unknown"-token gap finding 3 measured. Measured effect on E0 alone: PR-AUC 0.513 ± 0.042 (off) vs
0.523 ± 0.032 (on) — barely inside the noise band, not the clear win R2 gave. Plausible explanation, not yet
confirmed: absolute epoch time under a *temporal* split means every test-period value lies outside the
training range entirely (pure extrapolation), a problem the audit's underlying research (outside this app)
found for raw timestamps in a different model too — a relative/derived time signal (hour, day of week, as E1/
E2 already compute) may matter far more than the absolute epoch value does. Left available, not defaulted on.

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
that session). Run cold, to confirm the documentation-only changes broke nothing. (For the audit session's
current test count — 87 passed, 1 skipped, and it will keep growing — see "Tests completed (this session)"
near the top of this file.)

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

The commit titled "Document R2/R3 and finding-3 (C1) measurements in PROJECT_STATUS.md …" (run
`git log --oneline -1` for its exact hash; amending this file changes the hash, so it is intentionally not
pinned here). Commits this session, in order: "Audit fixes 1/2: …" (findings 1+5), "Audit fix 3/4: …"
(finding 4), "Audit fix 4/4: …" (multi-seed harness), "Update PROJECT_STATUS.md: document audit findings…",
"Revert class_weighting default to 'none': measured regression, not assumed benefit", "Document the measured
class_weighting revert…", "Audit R2/R3: continuous numeric representation…", "Audit C1: opt-in numeric epoch
for E0…" — each with the full suite passing before committing. The suite is now large enough that a single
`pytest` invocation exceeds this environment's per-command time limit; run per test file (see "Tests
completed" above for the exact breakdown) rather than assuming a single `pytest` call will finish.

## Known issues

Everything listed in each phase's own section above still applies. Additions from the audit session:

- Never run on the real ~1.3M-row transaction file from the original brief — only synthetic demo data was
  available in this project's build/test environment.
- Binary classification only through Processing/Model/Experiments.
- Nemotron never run (no GPU, no verified checkpoint in this environment) — `availability()` reports this
  honestly rather than the app claiming a run that didn't happen.
- PyTorch reproducibility is machine-local, not an absolute cross-platform guarantee (see Phase 5's notes).
- `models.sanity_transformer.class_weighting` default is `none` (the original behaviour); `sample_weight` is
  implemented and available but empirically regresses performance at severe imbalance — see the measured
  section above. Do not flip this default without re-measuring on whatever data is current at the time.
- **E1/E2's numeric feature count per column dropped from up to 3 to 1** (audit finding 1). A saved model
  from before this session will not load against the current tokenizer/feature shape.
- **E2 now includes ~12 extra sequence-feature positions by default** (`features.sequence_length: 3` ×
  4 sub-features, category only when a category column is set) — training is correspondingly slower per
  epoch; `features.include_sequence_features: false` restores the old, smaller E2.
- `representation.numeric_mode` (`continuous`, `numeric_coarse_bins`) and `data.e0_add_time_epoch` are
  implemented, tested, and measured (see above) but both **default off** — R2 helps E0/E1 and clearly hurts
  E2 (not yet diagnosed why); the E0 timestamp fix barely moves the needle on its own, plausibly because
  absolute epoch time still extrapolates poorly under a temporal split. Neither is a settled recommendation.
- Cyclical time features (`hour_sin`/`hour_cos` etc.) still lose their periodicity when fed through
  `quantile_bin` mode (finding 2/3's remaining, unfixed half — T1 below); `numeric_mode: continuous` would
  let them stay continuous, but hasn't been tried for cyclical features specifically, only for `amt`-style
  columns in E0/E1/E2 as a whole.

## Next task (continuing the audit's recommended order)

Full audit (architecture trace, 6 verified findings, prioritized experiment matrix) is in the conversation
history that produced this session's commits, not duplicated here. What's left from it:

1. **~~Diagnose why R2 hurts E2~~ — done this session.** Cause: unclipped outlier standardized values
   (|z|≈30) in engineered ratio/z-score features. `representation.numeric_clip` (default `5.0`, only active
   under `numeric_mode: continuous`) fixes E2 but is not a uniform win — see the measured table above,
   including R3's severe, unexplained instability on E1 in one configuration. **Still open:** why R3+clip
   collapsed on E1 (0.358 ± 0.304); whether a per-feature clip threshold (rather than one global value) does
   better than the single global `5.0` used here.
2. **~~T1 — feed cyclical features continuously~~ — implemented, tested, and measured this session.**
   `representation.continuous_features` (a per-column override independent of `numeric_mode`) isolates
   cyclical features from the R2/E2 outlier confound in item 1. Measured (E2, 3 seeds,
   `continuous_features: [hour_sin, hour_cos, day_of_week_sin, day_of_week_cos]` vs the `quantile_bin`
   baseline): **PR-AUC 0.861 → 0.833, F1 0.860 → 0.725 — a small regression, not the hoped-for improvement.**
   The theoretical argument (quantile-binning breaks a cyclical value's smooth wraparound) may still be sound
   in principle, but it does not survive contact with this data/model: plausibly because the already-present
   discrete `hour`/`day_of_week` bins give this small model, trained for only 10 epochs, everything it needs
   from time of day already, making the continuous `sin`/`cos` versions redundant rather than additive. Left
   off by default (`continuous_features: []`); this is a negative result, not a recommendation to enable it.
3. **~~S1 — extend finding 5's fix to the text/LLM path~~ — verified this session, no gap found.**
   `text_builder.py` reads `prepared.numeric`/`prepared.categorical` generically, so it already inherited the
   `prevK_*` sequence features for free (confirmed empirically: rendered a real prompt with `prev1_amount`,
   `prev1_category`, etc. present). The remaining question is representation *quality*, not data availability
   — the 12 sequence fields are scattered as individual `key=value` tokens among ~30 unrelated fields, rather
   than grouped as a coherent block the way `format_previous_transactions()` (still unused) would render them.
   Not measured whether that rendering difference matters to a language model; deprioritized below item 4.
4. **~~Add `class_weighting: pos_weight_natural`~~ — done this session; extended to E1/E2 as a follow-up
   (see the measured section below).** Mixed across levels, not a clean win — not made a default.
5. Once 1–4 land (or are explicitly deferred with a reason): re-run the full E0/E1/E2 comparison with
   whatever combination of `numeric_mode`/`numeric_clip`/`e0_add_time_epoch`/`class_weighting` settings this
   session's measurements actually support, not just the isolated per-finding checks done so far.

## `class_weighting: pos_weight_natural` — implemented and measured (§12 Experiment 4)

A second, mechanistically different option from `sample_weight`: instead of discounting the negative class
toward the true population rate (which was measured to regress performance), `pos_weight_natural` boosts the
positive class's loss *within whatever training sample was already drawn* (`BCEWithLogitsLoss`'s own
`pos_weight = n_negative / n_positive` in the sampled set) — standard imbalanced-loss practice, and it leaves
the negative class untouched, unlike `sample_weight`. `src/models/sanity_transformer.py`,
`tests/test_model.py` (2 new tests: must actually change training; must not require `_weight` to be present).

Measured (E0, 3 seeds, all three read together from one script run so the *relative* comparison is trustworthy
even though the `none` baseline itself showed some run-to-run variability — see caveat below):

| `class_weighting` | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| `none` (default) | 0.513 ± 0.042 | 0.990 | 0.366 |
| `sample_weight` (reverted default, kept for comparison) | 0.240 ± 0.202 | 0.837 | 0.280 |
| **`pos_weight_natural`** | **0.587 ± 0.138** | **0.994** | **0.444** |

A genuinely promising result on E0 — better mean, better ROC-AUC, better F1 than the current default — but
this did **not** generalize to E1/E2 when checked as a follow-up (see below), for two reasons flagged at the
time: (1) only E0 had been checked so far; (2) the `none` baseline itself measured differently across two
separate script executions with nominally identical settings (0.470 in the earlier finding-4 decomposition vs
0.513 here), a real reproducibility gap in the measurement methodology, not just noise in the treatment —
worth investigating (possibly unseeded randomness somewhere in schema/role detection) before trusting any
small-to-moderate effect size measured this way.

### Extended to E1/E2 (follow-up measurement)

| Level | `none` | `pos_weight_natural` |
|---|---|---|
| E0 | 0.513 ± 0.042 | **0.587 ± 0.138** (better mean, 3× the variance) |
| E1 | 0.470 ± 0.031 | 0.385 ± 0.195 (**worse mean**, 6× the variance) |
| E2 | 0.861 ± 0.033 | 0.867 ± 0.089 (roughly tied on PR-AUC, F1 dropped 0.860 → 0.809) |

**Confirms the mixed pattern rather than the E0-only promise**: `pos_weight_natural` helps E0, clearly hurts
E1, and is a wash on E2 with worse F1 and much higher variance everywhere. This is the same shape of result
as `sample_weight`'s original (reverted) measurement and R2's E0/E1-vs-E2 split — **a config change that
looks good on whichever single level you happened to check first, and doesn't hold up once you check the
rest.** Not made a default. `class_weighting` stays `none`.

6. **Run it on the real dataset** — still never done, unchanged from before this session.
7. **~~Generalization test on a non-fraud dataset, through the full app~~ — done this session.**
   `test_generalization_full_app_flow_on_a_structurally_different_dataset` pushes the `churn` fixture (no
   entity column, string dates, yes/no target) through the real Streamlit app end to end — upload, schema
   detection, quality, basic preprocessing, Processing's split/sample, all three E0/E1/E2 levels via
   Experiments, back to the Dashboard — not just unit-level `DataPreparer` calls. Confirms: role detection is
   correct without any override (target=`churn`, task=binary classification, entity=`None` since
   `customer_id` is unique per row, datetime=`signup_date`); E2 gracefully skips history/sequence features
   with no entity column rather than crashing or silently assuming one exists; every metric stays in a valid
   range across all three levels. Passed on the first real run after fixing one bug in the test itself
   (see "Tests completed" below).
8. Multiclass / regression through Processing; `_pending/src/preprocessing/outliers.py`; section 23's other
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
- [x] Audit: R2/R3 (continuous numeric representation) and C1 (E0 timestamp fix) implemented, tested and
      measured — both opt-in, neither defaulted on; R2 helps E0/E1 and hurt E2 until diagnosed (unclipped
      outlier values; `numeric_clip` fixes E2 but isn't a uniform win across levels), C1 barely moves E0's
      own number.
- [x] Audit: S1 verified — `text_builder.py` already inherits finding 5's sequence features for free (no
      code gap); remaining question is text-rendering quality, not data availability, and is deprioritized.
- [x] Audit: `class_weighting: pos_weight_natural` implemented and measured — promising on E0 (better than
      `none`), not yet checked on E1/E2 or made a default; flagged a baseline-reproducibility gap worth
      investigating (the `none` measurement itself varied across two separate script runs).
- [x] Generalization test on a non-fraud dataset through the full app — done, passing (see next-task item 7).
- [x] Audit: T1's `continuous_features` per-column toggle implemented and tested (not yet measured — see
      next-task item 2)
- [ ] Audit: T1's actual measurement (does it help E2's cyclical features once isolated); why R3+clip
      collapsed on E1 in one run (0.358 ± 0.304, unexplained); investigate the `none`-baseline
      reproducibility gap; extend `pos_weight_natural` to E1/E2
- [ ] Beyond the 8 phases: run on the real dataset; multiclass/regression through Processing; the rest of
      section 23
