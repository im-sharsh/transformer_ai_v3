# Transaction Data Intelligence

A research prototype for **automated intake, profiling, quality analysis, preprocessing and model readiness of
tabular data**, with a controlled comparison of how data preparation affects transformer training.

> **Research question.** How does systematic preprocessing and quality control affect transformer training
> compared with training on raw data? The framework is built to *measure* this, using the same model, split,
> seed and settings on each preparation level. It does not assume that preprocessing helps.

**Status: all 8 phases complete.** See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for exactly what was done in each
phase, what was verified (and how), and known issues — including what's honestly still missing (the real
dataset was never loaded in this project's build environment; see PROJECT_STATUS.md's "Next task").

## 1. Research motivation

Model results depend heavily on how data is prepared, yet preparation is often ad hoc and undocumented. This
project makes each preparation level explicit, reproducible, and — critically — *measurable*: the same model
trained on differently-prepared versions of the same rows tells you whether the preparation work actually mattered.

| Level | Meaning |
|---|---|
| E0 Raw | Minimal formatting only, values kept as loaded (the uncurated baseline) |
| E1 Quality processed | Type/categorical/numeric/datetime processing, fitted on training rows only |
| E2 Feature engineered | E1 + cyclical time features + leakage-safe, point-in-time entity/merchant history |

Every column-role decision (target, entity, datetime, amount, category, merchant, identifiers, personal data) is
**detected from the data or configured — never hard-coded to one dataset.** Tests rename every column in a
structurally different dataset (customer churn, no transaction-specific names) to prove this.

## 2. Architecture

```text
Dataset → Ingestion → Schema detection → Profiling → Quality → Basic preprocessing
        → E0 / E1 / E2 (+ leakage checks) → Tabular / text representation → Model → Evaluation → Experiments
```

```text
app.py                    Streamlit interface (8 sections; see below)
config.yaml                all settings — column hints, thresholds, model hyperparameters — optional where detectable
src/ingestion/              loading, metadata, schema detection, roles/task detection, synthetic demo data
src/profiling/               dataset profiling
src/quality/                 five-dimension quality scorecard, leakage detector
src/preprocessing/            stateless cleaner, train-fitted transforms, E0/E1/E2 level building (levels.py)
src/features/                 point-in-time entity/merchant history features
src/representation/           tabular tokenizer (built-in Transformer) and text representation (LM backends)
src/models/                   model interface (base.py), built-in Transformer, HF/Nemotron/API adapters, registry
src/evaluation/                classification metrics, controlled-experiment runner
src/utils/                     configuration loading, hardware detection, versioned-file/audit helpers
data/                          raw (as uploaded), processed, samples — not committed
experiments/, reports/         saved experiment manifests (experiment_vNNN.json) — not committed
notebooks/colab_demo.ipynb      same src/ modules as the app, for optional GPU acceleration
tests/                          pytest suite: unit tests per module + headless Streamlit UI flow tests
_pending/                       code drafted before the phased plan; moved into src/ only after inspection,
                                adaptation to the current interfaces, and real tests — see _pending/README.md
                                for exactly what changed on the way in, including bugs found along the way
```

The Streamlit app has 8 sections, each a real implemented page: **Dashboard, Data Upload, Profiling, Quality
Analysis, Processing (E0/E1/E2), Model, Experiments, Results.**

## 3. Installation

```bash
git clone <your-repository-url>
cd transaction-data-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the address Streamlit prints (usually http://localhost:8501). A GPU is **not** required for the core
workflow. To use the optional Hugging Face backend, additionally run:

```bash
pip install transformers peft accelerate
```

(Nemotron additionally needs `bitsandbytes` and a CUDA GPU with enough memory — see §9.)

## 4. Local CPU usage

1. **Data upload**: upload a CSV/JSON/JSON Lines file (stored unchanged in `data/raw/`), load a large file from a
   local path (avoids the 1 GB browser-upload limit), or load the built-in **synthetic** demo file (clearly
   labeled as not real data — useful for trying the whole pipeline without a real dataset).
2. **Profiling**: review and, if needed, correct the detected target/entity/datetime columns and task type.
3. **Quality Analysis**: run the quality scorecard, then basic (stateless) preprocessing.
4. **Processing**: pick a subset size (10,000–500,000 rows, or the full dataset) and build E0, E1 and E2. Each
   reports real rows/features/timing and, for E1/E2, real leakage-check findings — nothing is estimated.
5. **Model**: train the built-in Transformer (CPU-friendly by design) on any built level; see loss/PR-AUC curves,
   sanity checks, and validation/test metrics.
6. **Experiments** → **Results**: run the same model/settings/seed on E0 vs E1 vs E2, save the comparison, and
   review it later (Results reads saved files from disk, independent of the current session).

For a CPU-only machine, keep subsets in the 10,000–50,000 row range and the built-in Transformer's default
hyperparameters; the app tells you this in the Model page based on real hardware detection (`torch.cuda.is_available()`).

## 5. Colab usage

`notebooks/colab_demo.ipynb` runs the same pipeline (load → profile → E0 → E1 → E2 → train → evaluate → compare)
using the project's own `src/` modules — no logic is duplicated into the notebook. Upload the repository (or
clone it into the Colab runtime), `pip install -r requirements.txt`, and run top to bottom. A GPU runtime speeds
up the built-in Transformer and is closer to necessary for the optional Hugging Face/Nemotron backends.

## 6. Dataset format

Structured tabular data as CSV, JSON (array of records) or JSON Lines. Detected column roles:
`TARGET, ENTITY, IDENTIFIER, DATETIME, NUMERICAL, CATEGORICAL, TEXT, UNKNOWN` — inferred from **values first,
names second** (parse rates, uniqueness, repetition per entity, card-number checksums, category cardinality,
target-like characteristics), always overridable in the UI or `config.yaml`.

The full E0/E1/E2/model/experiment pipeline currently supports **binary classification** only (matching the
primary fraud-detection case study); multiclass and regression are detected and shown in Profiling but not yet
carried through Processing — see `PROJECT_STATUS.md`'s Phase 3 notes.

**Current case study**: credit-card transactions with a binary `is_fraud` target (≈0.58% positive in the real
dataset this was built against), a card identifier as entity, and a transaction timestamp. These roles are
detected from the data, not hard-coded — confirmed by tests that run the same pipeline on a structurally
different dataset (customer churn) with every column renamed.

## 7. E0 / E1 / E2

- **E0 — Raw.** Every column kept exactly as loaded, minus identifiers, personal data, and quasi-identifiers
  (columns effectively unique per entity, like a job title that happens to be 1:1 with a card) — the same
  exclusion policy every level shares. Deliberately includes redundant/uncurated columns (e.g. a raw epoch
  timestamp alongside a human-readable one) — it is the uncurated baseline, not a cleaned-up "level 0".
- **E1 — Quality processed.** Stateless cleaning (type/whitespace/case normalization, duplicate and invalid-value
  handling) followed by train-fitted transforms: median/mode imputation with a missingness indicator, rare-category
  grouping, and numeric transforms (robust scaling always; log1p for skewed columns; decile bins) — the model
  sees the *transformed* columns, never the raw values. Absolute timestamps are excluded (later periods lie
  outside the training range).
- **E2 — Feature engineered.** E1 plus cyclical time features (`hour`/`day_of_week` and their sin/cos encodings,
  `month`, `is_weekend`) and point-in-time entity/merchant history features (counts and amount statistics over
  1h/24h/7d/30d windows, verified by recomputing them from truncated per-entity history and checking they match).
  Gracefully skips the history-feature part — with a stated reason, not silently — when no entity, datetime or
  amount column is available.

All three levels share one temporal (or, without a usable datetime column, random-fallback) train/validation/test
split and one reproducible sample, drawn once per "Prepare split and sample" action and reused by every level, so
a difference in results comes from data preparation, not from different rows.

## 8. Model

**Built-in Sanity Transformer** (Phase 5, fully implemented): a small PyTorch Transformer encoder (2–4 layers,
configurable), purpose-built to run on CPU and to validate that a prepared level produces valid, learnable
model-ready data end to end — not to compete with large pretrained models. Each E0/E1/E2 feature becomes one
token (categorical values → per-feature vocabulary; numeric values → per-feature quantile bins, fitted on
training rows), plus a `[CLS]` position. `torch.set_num_threads(1)` is set deliberately: multi-threaded CPU
matrix reductions are not bit-reproducible in PyTorch, which was caught concretely during development (the same
seed gave validation PR-AUC 0.91 in one process and 0.10 in another before the fix).

**Hugging Face / Nemotron / Custom-API** (Phase 7, adapters — a weaker requirement per the brief than the
built-in Transformer): any causal LM from the Hub can be LoRA-fine-tuned as a two-class (" no"/" yes" label-token)
classifier over a text rendering of the same E0/E1/E2 rows. Verified for real against `distilgpt2` (a real, small,
public model). Nemotron reuses the same code but needs a CUDA GPU with enough memory; this was **not** run in
this project's build/test environment (no GPU, no verified checkpoint) — `availability()` reports that honestly
rather than the app claiming a successful run that didn't happen. The Custom/API backend scores rows through
your own HTTP endpoint (`{"prompts": [...]} -> {"probabilities": [...]}`); training happens on your side, this
backend is inference-only.

All backends share one interface (`src/models/base.py`'s `ModelAdapter`): `train`, `predict`, `save`, `load` are
backend-specific; `evaluate` (threshold chosen on validation by best weighted F1, applied unchanged to test) and
`sanity_checks` have a correct shared default so every backend reports metrics identically.

## 9. Evaluation

Metrics are computed at the **real class balance** (the Processing subset's sample carries inverse-probability
weights from any oversampling) as well as unweighted, and — when the entity column and split make it meaningful —
separately for test rows whose entity was never seen in training. Primary metrics: PR-AUC, ROC-AUC, precision,
recall, F1, and a weighted confusion matrix; accuracy is intentionally not shown by itself, since it is
misleading at this class imbalance (real dataset ≈0.58% positive). A validation-chosen decision threshold
(best weighted F1) is applied unchanged to test — never re-tuned on test.

## 10. Leakage prevention

- **By construction**: identifiers, personal-data columns, and quasi-identifiers are excluded from every level;
  history features use strictly-earlier rows only (verified per build, not just assumed); absolute timestamps
  never reach a model.
- **By detection** (Phase 4): a feature-level scan on E1/E2's *final* feature set, reusing the exact
  train/validation cutoff the split already computed. Checks for name hints, post-event timestamps, text that
  reveals the label, suspiciously high single-feature predictive power on a held-out later period, temporal
  instability, identity proxies, and cold-start artifacts. Only two checks — a genuinely post-event timestamp,
  or text that reveals the label — are auto-removed; everything else (including very high predictive power alone)
  is reported for human review, never removed automatically: a legitimate signal can be very predictive.

## 11. Experiments

The **Experiments** page trains the same model, with the same settings, seed and split, on however many of
E0/E1/E2 you select, and reports one comparison row per level (features, rows, train time, PR-AUC, ROC-AUC,
precision, recall, F1). Saved comparisons are written as never-overwritten `experiments/experiment_vNNN.json`
manifests; the **Results** page reads them back from disk (not from session state), so a comparison survives a
closed browser tab or a different machine reading the same repository.

## 12. Limitations (current)

* Binary classification only through Processing/Model/Experiments (multiclass/regression are detected in
  Profiling but not carried further yet).
* Not yet run on the real ~1.3M-row transaction file described in the original brief — only synthetic demo data
  was available in this project's build/test environment; the pipeline is designed for that scale (temporal
  split, subset sizes up to 500,000 rows or "full", memory-conscious profiling) but this specific claim is
  unverified at that size.
* Very large files are loaded fully into memory; profiling offers a reproducible sample above 500,000 rows, but
  quality analysis and E1/E2 processing run on the full sampled subset (not a further sample), since duplicate
  and consistency counts would be misleading on a sample of a sample.
* PyTorch does not guarantee bit-exact reproducibility across different machines/processes even with every seed
  fixed; `torch.set_num_threads(1)` closes the specific source found here (multi-threaded reductions), but "the
  same seed reproduces the same result" should be understood as true on the same machine, not as an absolute
  cross-platform guarantee.
* Nemotron has never been run in this environment (no CUDA GPU, no verified checkpoint) — its correctness rests
  on the now-verified Hugging Face adapter plumbing it shares, not on a Nemotron-specific test.
* A handful of quality checks (amount-sign, geographic-coordinate-range, currency-pattern, birth-date-plausibility)
  still use column-**name** heuristics in addition to role-based detection, so they may silently not fire on a
  dataset whose equivalent columns are named unusually; the majority of checks are fully role-based and confirmed
  name-independent by a dedicated test.

See `PROJECT_STATUS.md` for the complete, per-phase list of known issues, including exactly what was and wasn't
verified and how.

## 13. Future extensions

Additional datasets beyond the fraud case study (churn, credit risk) end to end through the full app; multiclass
and regression support in Processing/Model/Experiments; database connectors; streaming transactions; richer
experiment tracking (currently one JSON manifest per comparison); drift detection; privacy/PII scanning beyond
the current detection-and-masking; production deployment.

### Pretrained GPU fine-tuning benchmark

`notebooks/pretrained_gpu_finetuning.ipynb` provides a Colab/Kaggle GPU path for comparing E0/E1/E2 with a
pretrained sequence-classification Transformer. It starts with `distilbert/distilbert-base-uncased` and supports
frozen-head, LoRA/PEFT, and optional full fine-tuning. The notebook deliberately reuses one deterministic subset
and split across all processing levels; pretrained models tokenize the rendered row text with their own tokenizer
rather than consuming the built-in Transformer's Step-11 token IDs.

### AutoData GPU research benchmark

`notebooks/AutoData_GPU_Research_Benchmark.ipynb` is the recommended Colab notebook for the current
raw-vs-AI-ready investigation. It runs the repository's real backend modules on CUDA, not a reimplemented
notebook-only pipeline. The notebook:

- robustly locates an extracted project or accepts the latest project ZIP directly;
- builds E0/E1/E2 from one deterministic split/sample;
- runs representation/signal diagnostics before expensive training;
- compares E0/E1/E2 across multiple seeds on GPU;
- ablates E2 temporal, history, and previous-transaction sequence feature groups independently;
- tests continuous numeric representation against the current quantile-token baseline;
- exports a reproducible CSV result matrix and JSON run context under `experiments/colab_gpu/`.

For fraud work, use PR-AUC as the primary comparison metric and inspect seed-to-seed variance. Do not promote a
pipeline change to the default from one seed or one dataset.

### Post-audit GPU validation

After the 30k real-data research matrix, use `notebooks/AutoData_GPU_100K_Validation.ipynb` for the next T4 run.
It keeps fraud-enriched training but evaluates validation/test at natural prevalence, requires the stricter
point-in-time audit to pass, and compares only E0 raw, E1 continuous, E2 full, E2 history, E2 sequence, and
E2 history+sequence. See `reports/leakage_audit_2026-09-19.md` for the rationale.

### Strict external GPU evaluation
After internal validation, use `notebooks/AutoData_GPU_External_Holdout.ipynb` to train/select thresholds on `fraudTrain.csv` and score `fraudTest.csv` as a fixed external holdout. See `reports/external_holdout_plan.md` for the leakage controls and interpretation rules.

## Cross-schema benchmark: PaySim

The validated financial-risk baseline can now be tested on PaySim without changing downstream E0/E1/E2 code. `src/ingestion/adapters.py` maps PaySim's hourly `step` clock and transaction roles explicitly. Run `notebooks/AutoData_GPU_PaySim_Benchmark.ipynb` in Colab. Start with the `strict` profile; use `full_research` only as an ablation for simulator-specific balance fields. See `reports/PAYSIM_ADAPTER_PLAN.md` and the benchmark validation report in `reports/`.

## Pilot API / canonical schema layer

The validated data pipeline can also be run behind a small FastAPI boundary:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

or:

```bash
docker build -t autodata .
docker run --rm -p 8000:8000 autodata
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `POST /profile`
- `POST /validate`
- `POST /prepare`

`/profile` returns schema inference, a canonical semantic mapping and the adaptive E1/E2 recommendation. Canonical roles are `target`, `entity`, `time`, `amount`, `category`, and `counterparty`; uncertain mappings are reported rather than silently invented. A client can supply explicit JSON overrides.

`/prepare` is intended for pilot-sized synchronous jobs and returns a ZIP containing train/validation/test AI-ready files plus `manifest.json`. The manifest records the mapping, recommendation, selected processing level, representation/feature groups, split information and point-in-time audit. Large production jobs should move to an asynchronous job/queue execution model.

The HTTP API is CPU-compatible. CUDA accelerates benchmark/model execution when available but is not a product dependency.
