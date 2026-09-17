"""Transaction Data Intelligence: Streamlit interface.

Run locally:  streamlit run app.py
Phase 1: data upload, generic schema and task detection, profiling, hardware detection.
Later phases add quality analysis, E0/E1/E2 processing, models and experiments.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ingestion.loader import DatasetIntegrityError, UnsupportedFormatError, load_dataset
from src.ingestion.roles import (ROLE_LABELS, TASK_LABELS, TASKS, detect_roles, entity_summary, leakage_indicators,
                                 schema_for_profiling, target_summary, time_summary)
from src.ingestion.schema_detector import detect_schema
from src.models.base import PLANNED_BACKENDS
from src.preprocessing.basic_preprocessing import run_basic_preprocessing
from src.preprocessing.cleaner import CleaningConfig
from src.preprocessing.levels import DataPreparer, infer_roles
from src.profiling.profiler import profile_dataset
from src.quality.quality_engine import assess_quality
from src.utils.config import ROOT, load_config
from src.utils.hardware import detect_hardware, recommendations

logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="Transaction Data Intelligence", layout="wide")

PAGES = ["Dashboard", "Data Upload", "Profiling", "Quality Analysis", "Processing", "Model", "Experiments", "Results"]
PHASE_OF_PAGE = {"Model": 5, "Experiments": 6, "Results": 6}

CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: "IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif;
}
h1 { font-weight: 600; letter-spacing: -0.01em; font-size: 1.9rem; }
h2, h3 { font-weight: 600; letter-spacing: -0.005em; }
[data-testid="stMetricLabel"] { color: #5F6B7A; }
[data-testid="stMetricValue"] { font-weight: 500; font-size: 1.55rem; }
.tdi-lead { color: #5F6B7A; max-width: 70ch; margin-top: -0.4rem; }
.tdi-step { display: flex; gap: 0.6rem; align-items: baseline; padding: 0.18rem 0; font-size: 0.93rem; }
.tdi-step .n { width: 1.35rem; height: 1.35rem; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: 0.75rem; font-weight: 600; flex: none; }
.tdi-done .n { background: #2F7D4F; color: #fff; }
.tdi-ready .n { background: #2B5F8A; color: #fff; }
.tdi-wait .n { background: #DDE3EA; color: #5F6B7A; }
.tdi-wait { color: #7B8694; }
.tdi-pill { display: inline-block; padding: 0.05rem 0.5rem; border-radius: 0.6rem; font-size: 0.8rem; font-weight: 500; }
.tdi-pass { background: #E3F1E8; color: #2F7D4F; }
.tdi-warn { background: #F7EBDD; color: #8A5314; }
.tdi-fail { background: #F6E1E1; color: #9B2F2F; }
.tdi-none { background: #EDF1F5; color: #5F6B7A; }
</style>
"""


# ------------------------------------------------------------------ state and cached work
def init_state():
    defaults = {"nav": "Dashboard", "df": None, "meta": None, "schema": None, "roles": None, "profile": None,
                "profile_info": None, "source": None, "error": None, "quality": None, "preprocessing": None,
                "preparer": None, "levels": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def hardware() -> dict:
    return detect_hardware()


@st.cache_resource(show_spinner=False)
def config() -> dict:
    return load_config()


def pill(text: str, kind: str) -> str:
    return f'<span class="tdi-pill tdi-{kind}">{text}</span>'


def fmt_int(x) -> str:
    return "—" if x is None else f"{int(x):,}"


def _goto(page: str):
    st.session_state.nav = page


def set_dataset(path: Path, source: str):
    cfg = config()
    t0 = time.time()
    ds = load_dataset(path, metadata_dir=ROOT / cfg["project"]["data_dir"] / "raw" / "_metadata")
    schema = detect_schema(ds.df, dataset_id=ds.metadata.dataset_id, sample_size=cfg["schema"]["sample_size"])
    hints = cfg["data"]
    roles = detect_roles(ds.df, schema, target=hints.get("target") if hints.get("target") in ds.df else None)
    overrides = {k: hints.get(k) for k in ("entity_column", "datetime_column") if hints.get(k) in ds.df}
    if overrides or hints.get("task"):
        roles = roles.with_overrides(ds.df, entity=overrides.get("entity_column"), datetime=overrides.get("datetime_column"),
                                     task=hints.get("task"))
    st.session_state.update(df=ds.df, meta=ds.metadata, schema=schema, roles=roles, profile=None, profile_info=None,
                            quality=None, preprocessing=None, preparer=None, levels=None, source=source, error=None,
                            load_seconds=round(time.time() - t0, 1))


def save_upload(uploaded) -> Path:
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:12]
    folder = ROOT / config()["upload"]["save_uploads_to"]
    folder.mkdir(parents=True, exist_ok=True)
    name = Path(uploaded.name)
    target = folder / f"{name.stem}_{digest}{name.suffix.lower()}"
    if not target.exists():
        target.write_bytes(data)
    return target


# ------------------------------------------------------------------ sidebar
def sidebar():
    with st.sidebar:
        st.markdown("### Transaction Data Intelligence")
        st.caption("Data intake, quality and model readiness for tabular data")
        st.radio("Section", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.markdown("**Pipeline**")
        df, roles = st.session_state.df, st.session_state.roles
        prof, qual, prep = st.session_state.profile, st.session_state.quality, st.session_state.preprocessing
        levels = st.session_state.levels
        steps = [("Load data", "done" if df is not None else "ready"),
                 ("Confirm schema and task", "done" if roles is not None and roles.target else ("ready" if df is not None else "wait")),
                 ("Profile", "done" if prof is not None else ("ready" if df is not None else "wait")),
                 ("Quality analysis", "done" if qual is not None else ("ready" if df is not None else "wait")),
                 ("Basic preprocessing", "done" if prep is not None else ("ready" if df is not None else "wait")),
                 ("Process E0 / E1 / E2", "done" if levels else ("ready" if df is not None else "wait")),
                 ("Train and evaluate", "wait"), ("Compare experiments", "wait")]
        html = "".join(f'<div class="tdi-step tdi-{state}"><span class="n">{i}</span><span>{name}'
                       f'{" <small>(later phase)</small>" if state == "wait" and i > 5 else ""}</span></div>'
                       for i, (name, state) in enumerate(steps, start=1))
        st.markdown(html, unsafe_allow_html=True)
        st.divider()
        hw = hardware()
        st.caption(f"Device: {'CUDA GPU · ' + hw['gpu_name'] if hw['cuda'] else 'CPU'}")


# ------------------------------------------------------------------ pages
def page_dashboard():
    st.title("Dashboard")
    df, roles, meta = st.session_state.df, st.session_state.roles, st.session_state.meta
    if df is None:
        st.markdown('<p class="tdi-lead">Load a CSV or JSON file to see its structure, roles and profile. '
                    "Nothing is modified: files are read as they are.</p>", unsafe_allow_html=True)
        st.button("Go to data upload", type="primary", on_click=_goto, args=("Data Upload",))
    else:
        st.markdown(f'<p class="tdi-lead">{meta.file_name} · loaded from {st.session_state.source}</p>', unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("Rows", fmt_int(meta.rows))
        c[1].metric("Columns", fmt_int(meta.columns))
        c[2].metric("Target", roles.target or "not set")
        c[3].metric("Task", TASK_LABELS.get(roles.task, "not set"))
        c = st.columns(4)
        if roles.target and roles.task in ("binary_classification", "multiclass_classification"):
            summ = target_summary(df, roles.target, roles.task)
            label, value = (("Positive rate", f"{summ['positive_rate']:.3%}") if "positive_rate" in summ
                            else ("Classes", fmt_int(summ["classes"])))
            c[0].metric(label, value)
        elif roles.target:
            c[0].metric("Target mean", f"{pd.to_numeric(df[roles.target], errors='coerce').mean():,.3f}")
        c[1].metric("Missing cells", fmt_int(meta.missing_cells))
        c[2].metric("Entity column", roles.entity or "none")
        c[3].metric("Datetime column", roles.datetime or "none")

    st.subheader("Pipeline status")
    status = [("Data upload", 1, "done" if df is not None else "ready"),
              ("Schema and task detection", 1, "done" if roles is not None else "ready"),
              ("Profiling", 1, "done" if st.session_state.profile is not None else "ready"),
              ("Quality analysis", 2, "done" if st.session_state.quality is not None else ("ready" if df is not None else "wait")),
              ("Basic preprocessing", 2, "done" if st.session_state.preprocessing is not None else ("ready" if df is not None else "wait")),
              ("E0 / E1 / E2 processing", 3, "done" if st.session_state.levels else ("ready" if df is not None else "wait")),
              ("Feature engineering and leakage checks", 4, "planned"), ("Built-in transformer", 5, "planned"),
              ("Evaluation and experiments", 6, "planned"), ("Hugging Face / Nemotron / API adapters", 7, "planned")]
    st.dataframe(pd.DataFrame([{"Stage": s, "Phase": p, "Status": {"done": "Done", "ready": "Ready to run",
                                                                   "wait": "Load data first",
                                                                   "planned": "Not yet implemented"}[k]} for s, p, k in status]),
                 hide_index=True, width="stretch")

    st.subheader("Hardware")
    hw = hardware()
    c = st.columns(4)
    c[0].metric("Device", "CUDA GPU" if hw["cuda"] else "CPU")
    c[1].metric("GPU", hw["gpu_name"] or "none")
    c[2].metric("CPU cores", fmt_int(hw["cpu_count"]))
    c[3].metric("RAM", f"{hw['ram_gb']} GB" if hw["ram_gb"] else "unknown")
    for level, msg in recommendations(hw):
        (st.success if level == "ok" else st.warning)(msg)
    if hw["note"]:
        st.info(hw["note"])


def page_upload():
    st.title("Data upload")
    st.markdown('<p class="tdi-lead">CSV, JSON (records) or JSON Lines. Uploaded files are stored unchanged in '
                "<code>data/raw</code>. For very large files, load from a local path instead of uploading.</p>",
                unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload a file", type=["csv", "json", "jsonl"])
        if uploaded is not None and st.button("Load uploaded file", type="primary", key="load_upload"):
            try:
                with st.spinner("Reading file and detecting schema…"):
                    set_dataset(save_upload(uploaded), "upload")
                st.success("File loaded")
            except (UnsupportedFormatError, DatasetIntegrityError, ValueError) as e:
                st.error(f"Could not load the file: {e}")
    with right:
        path = st.text_input("Or load from a local path", placeholder="data/raw/fraudTrain.csv")
        if st.button("Load from path", key="load_path", disabled=not path):
            p = Path(path).expanduser()
            p = p if p.is_absolute() else ROOT / p
            if not p.exists():
                st.error(f"No file at {p}. Check the path; relative paths start from the project folder.")
            else:
                try:
                    with st.spinner("Reading file and detecting schema…"):
                        set_dataset(p, "local path")
                    st.success("File loaded")
                except (UnsupportedFormatError, DatasetIntegrityError, ValueError) as e:
                    st.error(f"Could not load the file: {e}")
        st.caption("No data at hand? Generate a small synthetic transaction file (clearly not real data).")
        if st.button("Load synthetic demo data", key="load_demo"):
            from src.ingestion.demo_data import make_transactions
            demo = ROOT / config()["project"]["data_dir"] / "samples" / "demo_transactions_SYNTHETIC.csv"
            demo.parent.mkdir(parents=True, exist_ok=True)
            if not demo.exists():
                make_transactions().to_csv(demo, index=False)
            with st.spinner("Reading file and detecting schema…"):
                set_dataset(demo, "synthetic demo generator")
            st.success("Synthetic demo data loaded")

    meta = st.session_state.meta
    if meta is None:
        st.info("No dataset loaded yet.")
        return
    st.subheader("Loaded dataset")
    c = st.columns(5)
    c[0].metric("Rows", fmt_int(meta.rows))
    c[1].metric("Columns", fmt_int(meta.columns))
    c[2].metric("Memory", f"{meta.memory_mb:,.0f} MB")
    c[3].metric("Missing cells", fmt_int(meta.missing_cells))
    c[4].metric("Load time", f"{st.session_state.get('load_seconds', 0)} s")
    st.caption(f"Dataset ID {meta.dataset_id} · SHA-256 {meta.sha256[:16]}… · format {meta.file_format}")
    st.dataframe(st.session_state.df.head(20), width="stretch")


def schema_editor():
    df, schema, roles = st.session_state.df, st.session_state.schema, st.session_state.roles
    st.subheader("Schema and task")
    st.caption("Detected automatically from the values; change anything that is wrong and apply.")
    cols = list(df.columns)
    none = "(none)"
    c = st.columns(4)
    target = c[0].selectbox("Target column", cols, index=cols.index(roles.target) if roles.target in cols else 0)
    task = c[1].selectbox("Task", TASKS, index=TASKS.index(roles.task) if roles.task in TASKS else 0,
                          format_func=lambda t: TASK_LABELS[t], help=f"Detected: {roles.task_reason}")
    entity = c[2].selectbox("Entity column", [none] + cols, index=([none] + cols).index(roles.entity) if roles.entity else 0,
                            help="Customer / account / card identifier, used later for history features")
    dtcol = c[3].selectbox("Datetime column", [none] + cols, index=([none] + cols).index(roles.datetime) if roles.datetime else 0,
                           help="Event time, used later for temporal splits")
    table = roles.to_frame(schema)
    edited = st.data_editor(table, hide_index=True, width="stretch", key="role_editor",
                            column_config={"role": st.column_config.SelectboxColumn("role", options=ROLE_LABELS, required=True)},
                            disabled=[c for c in table.columns if c != "role"])
    if st.button("Apply schema changes", key="apply_schema"):
        changes = {r["column"]: r["role"] for _, r in edited.iterrows() if r["role"] != roles.columns[r["column"]]}
        try:
            task_override = task if task != roles.task or target != roles.target else None
            new = roles.with_overrides(df, role_overrides=changes, target=target, entity=entity if entity != none else "",
                                       datetime=dtcol if dtcol != none else "")
            if task_override and task_override != new.task:
                new = new.with_overrides(df, task=task_override)
            st.session_state.roles, st.session_state.profile = new, None
            st.session_state.quality, st.session_state.preprocessing = None, None
            st.session_state.preparer, st.session_state.levels = None, None
            st.success("Schema updated. Profile again to use the changes.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    with st.expander("Detection candidates"):
        st.write("Target candidates", pd.DataFrame(roles.candidates.get("target", [])))
        st.write("Entity candidates", pd.DataFrame(roles.candidates.get("entity", [])))
        st.write("Datetime candidates", pd.DataFrame(roles.candidates.get("datetime", [])))


def page_profiling():
    st.title("Profiling")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first in Data upload.")
        return
    schema_editor()
    cfg = config()["profiling"]
    st.subheader("Profile")
    big = len(df) > cfg["sample_above_rows"]
    use_sample = st.checkbox(f"Profile a reproducible sample of {cfg['sample_rows']:,} rows (faster)", value=big,
                             disabled=not big, help="Row, missing-value and duplicate counts always use all rows.")
    if st.button("Profile data", type="primary", key="run_profile"):
        roles, schema = st.session_state.roles, st.session_state.schema
        with st.spinner("Profiling…"):
            t0 = time.time()
            frame = df.sample(cfg["sample_rows"], random_state=config()["project"]["seed"]) if use_sample and big else df
            prof = profile_dataset(frame, schema_for_profiling(schema, roles, df), top_k=cfg["top_k"],
                                   rare_threshold=cfg["rare_threshold"], skew_warning=cfg["skew_warning"])
            info = {"sampled": bool(use_sample and big), "rows_profiled": len(frame), "seconds": round(time.time() - t0, 1),
                    "full_missing_cells": int(df.isna().sum().sum()), "full_duplicate_rows": int(df.duplicated().sum()),
                    "target": target_summary(df, roles.target, roles.task) if roles.target else None,
                    "time": time_summary(df, roles.datetime, roles.target, roles.task) if roles.datetime else None,
                    "entity": entity_summary(df, roles.entity) if roles.entity else None,
                    "leakage": leakage_indicators(schema, roles)}
        st.session_state.profile, st.session_state.profile_info = prof, info
    prof, info = st.session_state.profile, st.session_state.profile_info
    if prof is None:
        st.info("Not run yet. Profiling reads the data and changes nothing.")
        return
    render_profile(prof, info)


def render_profile(prof, info):
    roles = st.session_state.roles
    note = f"sample of {info['rows_profiled']:,} rows" if info["sampled"] else f"all {info['rows_profiled']:,} rows"
    st.caption(f"Profiled {note} in {info['seconds']} s")
    c = st.columns(5)
    c[0].metric("Rows", fmt_int(len(st.session_state.df)))
    c[1].metric("Columns", fmt_int(prof.overview["columns"]))
    c[2].metric("Missing cells", fmt_int(info["full_missing_cells"]))
    c[3].metric("Duplicate rows", fmt_int(info["full_duplicate_rows"]))
    c[4].metric("Warnings", fmt_int(len(prof.warnings)))
    tabs = st.tabs(["Target", "Numerical", "Categorical", "Datetime and entity", "Roles", "Warnings and indicators"])
    with tabs[0]:
        t = info["target"]
        if not t:
            st.info("No target selected.")
        elif t["task"] == "regression":
            st.dataframe(pd.DataFrame([t]).drop(columns=["task"]), hide_index=True)
        else:
            dist = pd.DataFrame(t["distribution"])
            st.markdown(f"**{roles.target}** · {TASK_LABELS[t['task']]} · {t['classes']} classes · "
                        f"imbalance ratio {t['imbalance_ratio']:,.1f} : 1")
            if "positive_rate" in t:
                st.markdown(f"Positive class `{t['positive_class']}`: {t['positive_rate']:.3%} of rows. "
                            "With classes this unbalanced, accuracy is misleading; later phases report PR-AUC, recall, precision and F1.")
            st.bar_chart(dist.set_index("class")["rows"])
            st.dataframe(dist.assign(share=dist["share"].map(lambda v: f"{v:.3%}")), hide_index=True)
    with tabs[1]:
        rows = [{"column": c.name, "missing %": c.missing_pct, **{k: c.stats.get(k) for k in
                 ["mean", "median", "std", "min", "max", "skewness", "zero_count", "negative_count", "outliers_iqr_1_5"]}}
                for c in prof.columns.values() if c.role == "numeric"]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.info("No numerical columns.")
        st.caption("Outlier counts are reported, never removed: large values can be legitimate.")
    with tabs[2]:
        rows = [{"column": c.name, "unique": c.unique_count, "missing %": c.missing_pct,
                 "rare categories": c.stats.get("rare_category_count"), "rows in rare %": c.stats.get("rare_rows_pct"),
                 "case/space variants": c.stats.get("case_or_whitespace_variants"),
                 "top values": ", ".join(f"{k} ({v:.1%})" for k, v in (c.stats.get("top_values") or {}).items())}
                for c in prof.columns.values() if c.role in ("categorical", "categorical_code", "boolean")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.info("No categorical columns.")
        st.caption("Values of columns detected as personal data are masked.")
    with tabs[3]:
        t = info["time"]
        if t:
            c = st.columns(4)
            c[0].metric("From", t["min"][:10] if t["min"] else "—")
            c[1].metric("To", t["max"][:10] if t["max"] else "—")
            c[2].metric("Invalid timestamps", fmt_int(t["invalid"]))
            c[3].metric("File in time order", "yes" if t["chronological_in_file"] else "no")
            st.markdown("Rows per month")
            st.bar_chart(pd.Series(t["monthly_rows"], name="rows"))
            if t.get("monthly_positive_rate"):
                st.markdown("Positive rate per month")
                st.line_chart(pd.Series(t["monthly_positive_rate"], name="positive rate"))
        else:
            st.info("No datetime column selected.")
        e = info["entity"]
        if e:
            c = st.columns(4)
            c[0].metric("Entity column", e["column"])
            c[1].metric("Entities", fmt_int(e["entities"]))
            c[2].metric("Median rows per entity", f"{e['rows_per_entity_median']:,.0f}")
            c[3].metric("Max rows per entity", fmt_int(e["rows_per_entity_max"]))
    with tabs[4]:
        table = roles.to_frame(st.session_state.schema)
        st.dataframe(table, hide_index=True, width="stretch")
        counts = table["role"].value_counts()
        st.caption(" · ".join(f"{k}: {v}" for k, v in counts.items()))
    with tabs[5]:
        if prof.warnings:
            st.dataframe(pd.DataFrame({"warning": prof.warnings}), hide_index=True, width="stretch")
        else:
            st.success("No profiling warnings")
        st.markdown("**Possible leakage indicators** (early hints from names and flags; full checks come in Phase 4)")
        ind = pd.DataFrame(info["leakage"])
        st.dataframe(ind, hide_index=True, width="stretch") if len(ind) else st.success("No indicators found")


def page_quality():
    st.title("Quality Analysis")
    df, schema, roles = st.session_state.df, st.session_state.schema, st.session_state.roles
    if df is None:
        st.info("Load a dataset first in Data upload.")
        return
    cfg = config()
    ps = schema_for_profiling(schema, roles, df)
    if len(df) > cfg["quality"]["large_dataset_warn_rows"]:
        st.info(f"{len(df):,} rows: this may take a little while. Nothing is modified until you click Apply.")

    st.subheader("Data quality")
    st.caption("Five dimensions, each scored from checks generated from the detected roles: completeness, validity, "
              "consistency, uniqueness, integrity. A dimension with no applicable checks is left out of the score "
              "rather than counted as perfect. Nothing is deleted at this stage.")
    if st.button("Run quality analysis", type="primary", key="run_quality"):
        with st.spinner("Assessing data quality…"):
            weights = cfg["quality"]["weights"]
            st.session_state.quality = assess_quality(df, ps, **({"weights": weights} if weights else {}))
        st.session_state.preprocessing = None
    render_quality(st.session_state.quality)

    st.divider()
    st.subheader("Basic preprocessing")
    st.caption("Stateless steps only (safe before any train/validation/test split exists): type normalization, "
              "duplicate handling, invalid-value handling, categorical whitespace/case normalization, datetime "
              "parsing. Value imputation, rare-category grouping and scaling are fitted on training rows only and "
              "run during E1 / E2 processing (Phase 3).")
    pc = cfg["preprocessing"]
    c = st.columns(3)
    dup_policy = c[0].selectbox("Duplicate rows", ["remove", "quarantine", "retain"],
                                index=["remove", "quarantine", "retain"].index(pc["duplicate_policy"]),
                                help="remove: delete extra copies. quarantine: set aside for review. retain: keep, only report.")
    invalid_policy = c[1].selectbox("Invalid values", ["nullify", "quarantine"],
                                    index=["nullify", "quarantine"].index(pc["invalid_value_policy"]),
                                    help="nullify: set the impossible value to missing. quarantine: set the whole row aside.")
    drop_redundant = c[2].checkbox("Drop columns the quality check flags as systematically redundant",
                                   value=pc["drop_systematic_redundant_columns"])
    if st.button("Run basic preprocessing", type="primary", key="run_preprocessing"):
        pcfg = CleaningConfig(drop_systematic_redundant_columns=drop_redundant, drop_row_index=pc["drop_row_index"],
                              drop_constant_columns=pc["drop_constant_columns"], convert_types=pc["convert_types"],
                              zero_pad_codes=pc["zero_pad_codes"], normalize_whitespace=pc["normalize_whitespace"],
                              merge_case_variants=pc["merge_case_variants"], strip_shared_prefixes=pc["strip_shared_prefixes"],
                              invalid_value_policy=invalid_policy, quarantine_missing_mandatory=pc["quarantine_missing_mandatory"],
                              duplicate_policy=dup_policy)
        with st.spinner("Running basic preprocessing…"):
            st.session_state.preprocessing = run_basic_preprocessing(df, ps, roles, config=pcfg)
    render_preprocessing(st.session_state.preprocessing)


def render_quality(report):
    if report is None:
        st.info("Not run yet.")
        return
    st.caption(f"Assessed {report.rows:,} rows in {report.seconds} s")
    dims = ["completeness", "validity", "consistency", "uniqueness", "integrity"]
    cols = st.columns(len(dims) + 1)
    for col, dim in zip(cols, dims):
        s = report.scores[dim]["score"]
        col.metric(dim.capitalize(), f"{s:.1f}" if s is not None else "n/a",
                  help=f"{report.scores[dim]['checks']} checks" if s is not None else "no applicable checks")
    cols[-1].metric("Overall", f"{report.scores['overall']:.1f}")
    if report.systematic_issues:
        st.warning("Systematic issues (affect most rows; a column-level problem, not individual bad records): "
                   + ", ".join(f"`{c}`" for c in report.systematic_issues))
    st.metric("Rows with at least one individual issue", f"{report.records_with_issues:,}")
    rows = [{"Dimension": c.dimension, "Check": c.name, "Failed": f"{c.failed:,} {c.unit}",
            "Pass %": round(c.pass_rate * 100, 3), "Severity": c.severity, "Source": c.source,
            "Details": str(c.details)[:200]} for c in report.checks]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if report.notes:
        with st.expander("Notes"):
            for n in report.notes:
                st.markdown(f"- {n}")


def render_preprocessing(result):
    if result is None:
        st.info("Not run yet.")
        return
    s = result.summary
    c = st.columns(4)
    c[0].metric("Rows", f"{s.rows_after:,}", delta=f"{s.rows_after - s.rows_before:,}" if s.rows_after != s.rows_before else None)
    c[1].metric("Columns", f"{s.columns_after}", delta=f"{s.columns_after - s.columns_before}" if s.columns_after != s.columns_before else None)
    c[2].metric("Duplicates removed", f"{s.duplicates_removed:,}")
    c[3].metric("Quarantined rows", f"{s.quarantined_rows:,}")
    st.markdown("**Preprocessing summary**")
    st.code("\n".join(s.to_lines()), language=None)
    if s.dropped_columns:
        st.markdown(f"**Dropped columns:** {', '.join(f'`{c}`' for c in s.dropped_columns)}")
    if s.identifier_columns_excluded:
        st.caption(f"Identifier / entity columns kept in the data but excluded from modelling: "
                  f"{', '.join(f'`{c}`' for c in s.identifier_columns_excluded)}")
    tabs = st.tabs(["Transformation log", "Quarantined rows", "Removed duplicates", "Preview"])
    with tabs[0]:
        audit = result.cleaning.audit.to_frame()
        st.dataframe(audit[["operation", "reason", "records_modified", "records_removed"]] if len(audit) else audit,
                    hide_index=True, width="stretch")
    with tabs[1]:
        q = result.cleaning.quarantine
        st.dataframe(q, width="stretch") if len(q) else st.success("No rows quarantined")
    with tabs[2]:
        d = result.cleaning.removed_duplicates
        st.dataframe(d, width="stretch") if len(d) else st.success("No duplicate rows removed")
    with tabs[3]:
        preview_cols = [c for c in result.cleaning.df.columns if not c.startswith("_")]
        st.dataframe(result.cleaning.df[preview_cols].head(20), width="stretch")


def page_placeholder(name: str):
    st.title(name)
    phase = PHASE_OF_PAGE[name]
    descriptions = {
        "Experiments": "Run E0, E1 and E2 with identical model settings, seed and split.",
        "Results": "Comparison tables and charts of measured metrics.",
    }
    st.info(f"Not yet implemented: planned for Phase {phase}.")
    if name in descriptions:
        st.markdown(descriptions[name])
    if name in ("Experiments", "Results"):
        st.dataframe(pd.DataFrame([{"Experiment": e, "Status": "Not run"} for e in ["E0 Raw", "E1 Quality processed",
                                                                                     "E2 Feature engineered"]]),
                     hide_index=True)


def page_processing():
    st.title("Processing")
    df, schema, roles = st.session_state.df, st.session_state.schema, st.session_state.roles
    if df is None:
        st.info("Load a dataset first in Data upload.")
        return
    if not roles.target:
        st.info("Choose a target column in Profiling first.")
        return
    st.markdown('<p class="tdi-lead">E0 raw, E1 quality processed and E2 feature engineered versions of the same '
               "rows, with a reproducible subset size and a train / validation / test split shared by every "
               "level.</p>", unsafe_allow_html=True)
    cfg = config()
    ps = schema_for_profiling(schema, roles, df)
    try:
        level_roles = infer_roles(df, ps, roles)
    except ValueError as e:
        st.warning(str(e))
        return

    with st.expander("Resolved roles and exclusions"):
        c = st.columns(4)
        c[0].metric("Time column", level_roles.time or "none")
        c[1].metric("Entity column", level_roles.entity or "none")
        c[2].metric("Amount column", level_roles.amount or "none")
        c[3].metric("Category / merchant", f"{level_roles.category or '—'} / {level_roles.merchant or '—'}")
        if not level_roles.entity or not level_roles.amount:
            st.caption("E2's history features need an entity and an amount column; without them E2 gracefully skips "
                      "that part and still runs.")
        if level_roles.excluded:
            st.caption(f"{len(level_roles.excluded)} columns excluded from every level (identifiers, personal data, "
                      "quasi-identifiers — same policy for E0, E1 and E2):")
            st.dataframe(pd.DataFrame([{"column": c, "reason": r} for c, r in level_roles.excluded.items()]),
                        hide_index=True, width="stretch")

    sc = cfg["subsets"]
    options = [str(o) for o in sc["options"]]
    default = str(sc["cpu_default"]) if str(sc["cpu_default"]) in options else options[0]
    left, right = st.columns([2, 1])
    subset = left.selectbox("Subset size (rows drawn for this run; reused by every level)", options,
                            index=options.index(default), help="'full' uses every available row after the split.")
    method = "temporal" if cfg["split"]["method"] == "temporal" and level_roles.time else "random"
    f = cfg["split"]["fractions"]
    right.metric("Split", method.capitalize(), help=f"{f[0]:.0%} train / {f[1]:.0%} validation / {f[2]:.0%} test")

    if st.button("Prepare split and sample", type="primary", key="prepare_split"):
        with st.spinner("Assigning train / validation / test and drawing the sample…"):
            preparer = DataPreparer(df, ps, level_roles, cfg, rows="full" if subset == "full" else int(subset),
                                    seed=cfg["project"]["seed"])
            preparer.prepare_split()
        st.session_state.preparer, st.session_state.levels = preparer, {}

    preparer = st.session_state.preparer
    if preparer is None:
        st.info("Not prepared yet.")
        return

    info = preparer.split_info
    st.subheader("Split and sample")
    st.caption(f"Method: {info['boundaries']['method']} · prepared in {info['seconds']} s · "
              f"{info['rows_unassigned']:,} rows unassigned (missing target or time)")
    rep = info["sample"]
    cols = st.columns(3)
    for col, split in zip(cols, ["train", "validation", "test"]):
        r = rep.get(split, {})
        col.metric(split.capitalize(), f"{r.get('rows', 0):,} rows",
                  help=f"{r.get('positives', 0):,} positives ({r.get('sample_positive_share', 0):.2%}); "
                       f"represents {r.get('represents_rows', 0):,} rows at the real class balance")

    st.subheader("Levels")
    tabs = st.tabs(["E0 · Raw", "E1 · Quality processed", "E2 · Feature engineered"])
    for tab, level in zip(tabs, ["E0", "E1", "E2"]):
        with tab:
            render_level(preparer, level)


def render_level(preparer, level: str):
    levels = st.session_state.levels
    if level not in levels:
        if st.button(f"Build {level}", key=f"build_{level}"):
            with st.spinner(f"Building {level}…"):
                levels[level] = preparer.build(level)
        else:
            st.info("Not built yet.")
            return
    prepared = levels[level]
    info = prepared.info
    c = st.columns(4)
    c[0].metric("Features", info["features"])
    c[1].metric("Train rows", f"{info['rows']['train']:,}")
    c[2].metric("Validation rows", f"{info['rows']['validation']:,}")
    c[3].metric("Test rows", f"{info['rows']['test']:,}")
    st.caption(f"Positives — train {info['positives']['train']:,}, validation {info['positives']['validation']:,}, "
              f"test {info['positives']['test']:,} · built in {info['seconds']} s")
    st.markdown("**Steps**")
    st.code("\n".join(info["steps"]), language=None)
    pit = info.get("point_in_time")
    if pit:
        (st.success if pit["passed"] else st.error)(
            f"Point-in-time check: {'passed' if pit['passed'] else 'FAILED'} on {pit['rows_checked']} sampled rows "
            "(history features recomputed from truncated data must match the real ones)")
    findings = info.get("leakage")
    if findings:
        st.markdown("**Leakage findings** (nothing is removed except structural leaks flagged below; the rest need human review)")
        risk_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        rows = sorted(({"Feature": f["feature"], "Risk": f["leakage_risk"], "Check": f["check"],
                       "Reason": f["reason"], "Recommendation": f["recommendation"]} for f in findings),
                     key=lambda r: risk_order.get(r["Risk"], 9))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif findings is not None:
        st.success("No leakage findings.")
    st.markdown("**Preview (train split)**")
    preview_cols = [c for c in prepared.frames["train"].columns if not c.startswith("_")]
    st.dataframe(prepared.frames["train"][preview_cols].head(20), width="stretch")


def page_model():
    st.title("Model")
    st.info("Model training is not yet implemented: the built-in transformer is planned for Phase 5, other backends for Phase 7.")
    hw = hardware()
    st.markdown(f"**Detected device:** {'CUDA GPU (' + hw['gpu_name'] + ')' if hw['cuda'] else 'CPU'}")
    for level, msg in recommendations(hw):
        (st.success if level == "ok" else st.warning)(msg)
    st.dataframe(pd.DataFrame([{"Backend": b["label"], "Description": b["description"],
                                "Status": f"Not yet implemented (Phase {b['phase']})"} for b in PLANNED_BACKENDS]),
                 hide_index=True, width="stretch")


# ------------------------------------------------------------------ main
def main():
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()
    page = st.session_state.nav
    try:
        if page == "Dashboard":
            page_dashboard()
        elif page == "Data Upload":
            page_upload()
        elif page == "Profiling":
            page_profiling()
        elif page == "Quality Analysis":
            page_quality()
        elif page == "Processing":
            page_processing()
        elif page == "Model":
            page_model()
        else:
            page_placeholder(page)
    except Exception as e:                                         # show errors clearly instead of a blank page
        st.error(f"Something went wrong on this page: {type(e).__name__}: {e}")
        st.exception(e)


main()
