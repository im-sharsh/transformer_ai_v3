# Project status

_Read this first in every new session. Then inspect the files and re-run the tests before continuing._

## Current phase

**Phase 7 — Hugging Face / Nemotron / API adapters: complete.** Waiting for instruction before starting Phase 8.

## Cumulative summary (all phases so far)

**Phase 1 — Foundation:** repository structure, Streamlit app, hardware detection, CSV/JSON intake, generic
schema/task detection, profiling. 17 tests.

**Phase 2 — Quality and basic preprocessing:** quality engine, stateless cleaner, Quality Analysis UI page. 32 tests.

**Phase 3 — E0 / E1 / E2 processing:** subset selection, temporal/random split, sampling, three data levels,
Processing UI page. 46 tests.

**Phase 4 — Feature engineering + leakage checks:** the tested leakage detector wired into E1/E2. 50 tests.

**Phase 5 — Built-in CPU transformer:** `SanityTransformerAdapter`, Model UI page. 56 tests.

**Phase 6 — Evaluation + experiment runner:** `run_comparison` across E0/E1/E2, Experiments/Results UI pages. 62 tests.

**Phase 7 — Hugging Face / Nemotron / API adapters (complete, this session):** see below.

## Completed work (Phase 7, this session)

- Moved `src/models/hf_adapter.py`, `hf_backend.py`, `api_adapter.py`, `registry.py`, and
  `src/representation/text_builder.py`, `transaction_formatter.py` from `_pending/` into `src/`.
- **Found a real interface/architecture problem by actually running the code, not just importing it:**
  `hf_backend.py`'s LoRA attachment used a hard-coded, LLaMA-family target-module list
  (`q_proj`/`k_proj`/`v_proj`/...) inherited from the code's Nemotron-focused origin. Tried against `gpt2` (a
  different, GPT-2-family architecture, whose attention module is named `c_attn`), it failed outright
  (`NoMatchingPeftModuleError`) — none of the target names existed in the model. Fixed by making `target_modules`
  read from `config.yaml` (`hf_adapter.py`'s `_load()`), with `models.huggingface.target_modules: [c_attn]` set
  to match the shipped `gpt2` default; a LLaMA-family model (Nemotron included) still gets the correct default
  with no override needed.
- Added `transformers`, `peft`, `accelerate` (and `bitsandbytes`, GPU-only) to `requirements.txt` as **optional**,
  commented-out dependencies: the app runs fully without them, and `HuggingFaceAdapter.availability()` reports
  the backend honestly unavailable with a clear reason ("missing library: ...") if they're absent, rather than
  the app failing to start over an optional feature.
- New `models.huggingface`, `models.nemotron`, `models.api` sections and `representation.max_tokens` in
  `config.yaml`.
- **Verified for real, not just wired up** (installed the optional dependencies and actually ran the code):
  - `HuggingFaceAdapter` — full train → predict → evaluate → save → load cycle against `distilgpt2` (real, small,
    public Hub model — section 13/27: not a fabricated checkpoint), LoRA-fine-tuned on CPU: 21 optimizer steps in
    6.9 s, validation PR-AUC 0.71, save/load round-trip gave identical predictions. Also confirmed the real
    `gpt2` tokenizer (the shipped default, distinct from the fine-tuning run above) gives the required
    single-token " no"/" yes" labels.
  - `APIAdapter` — a genuine round trip against a real local HTTP server matching the documented
    `{"prompts": [...]} -> {"probabilities": [...]}` contract, including that it correctly rejects a response
    with out-of-[0, 1] probabilities.
  - `NemotronAdapter` — **not run.** Per section 13/14 ("do NOT invent a model checkpoint"; Nemotron "should only
    be connected after verifying a suitable checkpoint/configuration"; requires a CUDA GPU), and this environment
    has neither a verified checkpoint/config nor a GPU, `availability()` honestly reports it unavailable here.
    This is the correct behavior, not a gap: fabricating a "successful" Nemotron run would violate section 27.
- `src/models/base.py`: removed the now-redundant static `PLANNED_BACKENDS` list (its job — describing what
  backends exist and their status — is now done properly by `registry.available_backends()`, which checks real
  availability against the actual machine and config, not a hard-coded phase number).
- Model page (`app.py`): "Backends" table now uses `available_backends()` (real checks). New "Other backends"
  section: pick any non-built-in backend and run its own `sanity_checks()` against the selected level — real
  checks against the real adapter (including a real Hugging Face forward pass when the optional dependencies are
  installed), not a canned status. The rich train/chart/evaluate UI stays specific to the built-in Transformer,
  matching section 13's weaker requirement for the other backends ("can be implemented through an adapter").

## Files changed this session

**New:** `tests/test_backends.py`
**Modified:** `src/models/hf_adapter.py` (`target_modules` config), `src/models/base.py` (removed
`PLANNED_BACKENDS`), `app.py` (Model page backend table + "Other backends" section), `config.yaml`
(`models.huggingface/nemotron/api`, `representation.max_tokens`), `requirements.txt` (optional HF deps),
`_pending/README.md`
**Moved (git mv, content unchanged unless noted above):** `_pending/src/models/{hf_adapter,hf_backend,
api_adapter,registry}.py` → `src/models/`; `_pending/src/representation/{text_builder,transaction_formatter}.py`
→ `src/representation/`

## Tests completed

`pytest` → **68 passed** (was 62 at the end of Phase 6; 6 added this session). Run cold, with the optional
Hugging Face dependencies installed (`pip install transformers peft accelerate`) so those tests actually ran
rather than skipping.

| File | Covers |
|---|---|
| test_backends.py (new) | the registry lists all four backends as real `ModelAdapter` subclasses; `available_backends` reports real, differentiated availability (sanity_transformer always available, api unavailable with no endpoint configured, nemotron unavailable without a GPU) rather than a static list; `get_adapter` rejects an unknown key; a **real local HTTP server** round trip for the API adapter matching its documented contract, including rejecting an out-of-range response; a **real Hugging Face Hub model** (`distilgpt2`) trains, predicts, evaluates and save/load-round-trips correctly on CPU — skipped (not failed) if `transformers`/`peft` or network access to the Hub aren't available, since these are optional |

Also verified interactively in a real browser: Model page's "Backends" table showed real availability
(Hugging Face available since the optional libraries were installed; API and Nemotron correctly unavailable by
default); ran "Other backends" → Hugging Face Model → sanity checks against a built E1 level, which genuinely
tokenized real transaction rows with the real `gpt2` tokenizer and reported pass for schema/labels/text
representation/libraries.

**Not yet executed:** the app on the real 1.3M-row transaction file (only synthetic data was available in the
build environment, same limitation as Phases 2–6); any Nemotron run (no GPU in this environment, see above).

## Commands used

```bash
pip install -r requirements.txt
pip install transformers peft accelerate   # optional: only for the Hugging Face / Nemotron backends
pytest
streamlit run app.py
```

## Current git commit

The commit titled "Phase 7: Hugging Face / Nemotron / API adapters …" (run `git log --oneline -1` for its exact
hash; amending this file changes the hash, so it is intentionally not pinned here).

## Known issues

- Everything carried over from Phases 2–6 still applies.
- `HuggingFaceAdapter.sanity_checks()` is simpler than `SanityTransformerAdapter`'s (schema/labels/text
  representation/libraries only — no actual forward-pass or mini-training check), matching the original draft.
  Could be extended to match Phase 5's rigor if the Hugging Face backend gets promoted to a first-class,
  fully-supported backend later; not done now to keep Phase 7 at its stated ("adapter") scope.
- `target_modules` is only auto-correct for the two architectures actually checked (GPT-2 family via
  `c_attn`, LLaMA family via the `hf_backend.py` default); a different architecture (e.g. Mistral, Qwen, Gemma)
  may need its own `target_modules` override in `config.yaml` — this is inherent to "any compatible model"
  (section 13's own wording), not something a generic adapter can fully auto-detect without more work.
- `NemotronAdapter` inherits everything from `HuggingFaceAdapter` and has never been exercised even with a stub
  checkpoint in this project's CI/build environment (no GPU here, and section 14 says CPU fine-tuning of a
  language model this size isn't practical anyway) — its correctness rests entirely on `HuggingFaceAdapter`'s
  now-verified plumbing plus the original author's own GPU testing (per `_pending/README.md`'s prior
  "tested there" note for `hf_backend.py`), not on anything verified in this session.
- The Model page's rich per-epoch training UI (loss/PR-AUC charts, sanity-check table, evaluation table) is
  built only for the built-in Transformer; Hugging Face/API get the lighter "run sanity checks" action. Building
  the full UI for every backend was judged out of scope for Phase 7 (section 13's wording distinguishes "fully
  implemented" from "can be implemented through an adapter").

## Next task (Phase 8 — polish, documentation, tests, Colab notebook)

1. Read this file; run `pytest`.
2. README (section 22 lists exactly what it needs: research motivation, architecture, installation, local CPU
   usage, Colab usage, dataset format, E0/E1/E2, model architecture, evaluation, leakage prevention, experiments,
   limitations, future extensions) — check the current `README.md` against that list; it was written after
   Phase 1 and is now six phases stale.
3. `notebooks/colab_demo.ipynb` (section 21): load → profile → E0 → E1 → E2 → train the built-in Transformer →
   evaluate → compare, reusing the same `src/` modules the Streamlit app uses — do not duplicate logic into the
   notebook.
4. Section 20 (testing) is already substantially covered (68 tests across ingestion, profiling, quality,
   preprocessing, levels, leakage, models, backends, experiments, and full headless UI flows) — read what
   exists before adding more; look for gaps rather than assuming there are none.
5. Consider the "Later: generalization test on a non-fraud dataset" TODO item — the `churn` fixture already
   exercises this at the unit level (`tests/test_levels.py`, `test_basic_preprocessing.py`) but the full app has
   never been driven end to end (upload → process → train) on a genuinely different dataset; decide whether
   that belongs in Phase 8 or stays a separate "Later" item.
6. `_pending/src/preprocessing/outliers.py` (flagging-only, fitted on training rows) — the one remaining
   phase-labeled draft ("3 or 6") never wired in; decide whether Phase 8 polish includes it or it stays deferred.
7. Do not restructure or rewrite what's already working (section 24: "Only work on the current phase unless
   explicitly asked to continue") — Phase 8 is explicitly about polish and documentation, not new pipeline features.

## TODO by phase

- [x] Phase 1: foundation
- [x] Phase 2: quality analysis + basic preprocessing
- [x] Phase 3: E0 / E1 / E2 pipelines, subsets, temporal split
- [x] Phase 4: feature engineering + leakage checks
- [x] Phase 5: built-in CPU transformer
- [x] Phase 6: evaluation + experiment runner
- [x] Phase 7: Hugging Face / Nemotron / API adapters
- [ ] Phase 8: polish, documentation, tests, Colab notebook
- [ ] Later: generalization test on a non-fraud dataset
