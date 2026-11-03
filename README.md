# Autonomous Data Analyst Agent

Businesses spend significant time analysing datasets, finding trends and writing reports.
This agent analyses business data (CSV/Excel), answers questions in natural language,
detects anomalies, generates insights and automates recurring analyses.

Datasets:

* **Primary:** [Sample Superstore](https://huggingface.co/datasets/An-j96/SuperstoreData)
  (`data/raw/superstore.csv`, GPL-2.0): 9,994 US retail order lines, 2014-2017.
* **ML benchmark:** [NYC taxi passengers](https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv)
  from the Numenta Anomaly Benchmark, with 5 labelled anomaly windows, used to evaluate
  anomaly detectors against ground truth.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m analyst.data.profile data\raw\superstore.csv --json profile.json
.\.venv\Scripts\streamlit run src/analyst/dashboard/app.py
```

## Roadmap

| # | Milestone | Package | Status |
|---|---|---|---|
| 1 | Scope, dataset, architecture, shared contracts | `contracts.py`, `datasets.py` | done |
| 2 | Ingestion (CSV/Excel/Parquet) and data profiling | `data/` | done |
| 3 | Cleaning + aggregation of transactions to time series; ML anomaly detection with evaluation | `data/`, `ml/` | next |
| 4 | Agent: LangGraph + Gemini tool calling over profiling/query/chart/anomaly tools; written report | `agent/` | |
| 5 | Dashboard (upload, profile, Q&A, anomalies, report); SQLite analysis history; APScheduler jobs | `dashboard/`, `storage/`, `automation/` | |
| 6 | Testing, documentation, demo | all | |

Later: sales forecasting, scheduled analyses, automated alerts, SQL database sources.

## Architecture

```
load_dataset ─► check_quality ─► clean ─► profile ─┬─► detect_anomalies ─┐
  (raw frame)                      (clean frame)    └─► forecast ─────────┴─► generate_insights ─► AnalysisReport ─► dashboard
```

* `src/analyst/contracts.py` defines every object passed between stages
  (`DatasetSpec`, `CleanResult`, `QualityReport`, `DatasetProfile`, `AnomalyResult`,
  `ForecastResult`, `Insight`, `AnalysisReport`), the clean-frame convention, and the
  stage signatures. Validators reject malformed results at construction time.
* `src/analyst/agent/pipeline.py` (`run_analysis`) wires the stages. Data-stage failures
  raise; ML-stage failures become `report.warnings` so the dashboard still renders.
* The dashboard only consumes `AnalysisReport`; it never imports `analyst.data` or `analyst.ml`.

## File ownership

| Owner | Files | May import |
|---|---|---|
| Coordinator | `contracts.py`, `config.py`, `datasets.py`, `synthetic.py`, `agent/`, `tests/*.py` (top level), `pyproject.toml`, `requirements.txt`, `README.md` | everything |
| Data | `src/analyst/data/`, `tests/data/` | `contracts`, `config`, `datasets`, `synthetic` |
| ML | `src/analyst/ml/`, `tests/ml/` | `contracts`, `config`, `datasets`, `synthetic` |
| Dashboard | `src/analyst/dashboard/`, `tests/dashboard/`, `.streamlit/` | `contracts`, `config`, `datasets`, `agent` |

Changes to coordinator-owned files (including new dependencies) are requested, not made
directly. Workstreams must not import each other's packages.

## Integration gate

A workstream branch is merged only when, on top of current `main`:

1. `python -m pytest` passes (workstream tests + `tests/test_contracts.py`,
   `tests/test_pipeline_smoke.py`, `tests/test_dashboard_smoke.py`);
2. public signatures in `contracts.py` are unchanged;
3. no files outside the workstream's ownership are modified.

Branches are merged one at a time (data, then ml, then dashboard) and the gate is re-run after each.

## Data profiling

`analyst.data.profile.profile_dataset(raw)` returns a JSON-serialisable `DatasetProfile`:
per column the semantic role (identifier, datetime, numeric, categorical, boolean, text,
constant, empty), missing/unique counts, numeric distribution (quartiles, zeros, negatives,
IQR outliers), date range or top values, plus a `QualityReport`. Quality checks: missing
values, empty and constant columns, duplicate and near-duplicate rows (identical except for
a unique key), codes that lost leading zeros, numbers stored as text, categories differing
only by case/whitespace, outliers.

On Superstore it finds: `Country` is constant, `Postal Code` was stored as a number so 449
order lines with zip codes starting with 0 (New England, New Jersey) lost that digit, 1 order line is duplicated under a different
`Row ID`, and 1,871 order lines (18.7%) are loss-making.

## Current ML baseline (NYC taxi)

| Stage | Method | Result |
|---|---|---|
| Anomaly | seasonal robust z-score (lag 48, threshold 3.5) | precision 0.20, recall 0.21, event recall 5/5, 10.4% of points flagged |
| Forecast | seasonal naive (48-step backtest) | MAE 5126, MAPE 41% vs last-value naive MAE 7635, MAPE 103% |
