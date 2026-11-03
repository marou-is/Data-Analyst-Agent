# Autonomous Data Analyst Agent

Given a time-series dataset (CSV/Parquet), the agent checks data quality, cleans and
profiles it, detects anomalies, forecasts ahead, and writes plain-language insights,
all rendered in a Streamlit dashboard.

Demo dataset: [NYC taxi passengers](https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv)
from the Numenta Anomaly Benchmark (30-minute buckets, Jul 2014 - Jan 2015, 5 labelled
anomaly windows: NYC marathon, Thanksgiving, Christmas, New Year, blizzard).

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\streamlit run src/analyst/dashboard/app.py
```

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

## Current baseline (Phase 0, NYC taxi)

| Stage | Method | Result |
|---|---|---|
| Anomaly | seasonal robust z-score (lag 48, threshold 3.5) | precision 0.20, recall 0.21, event recall 5/5, 10.4% of points flagged |
| Forecast | seasonal naive (48-step backtest) | MAE 5126, MAPE 41% vs last-value naive MAE 7635, MAPE 103% |
