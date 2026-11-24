# Project 09 — Logistics Delivery ETA Prediction & Operational Intelligence

**Status:** Completed retrospective ML portfolio analysis (synthetic data).  
**Objective:** Predict dispatch-to-delivery duration in minutes and identify shipments that may miss their promised delivery windows.  
**Dataset:** One reproducible, synthetic `data/deliveries.csv` containing **10,000 shipments and 16 columns**.  
**Model:** `LightGBMRegressor`, selected by chronological **validation** performance, then refitted on all eligible pre-test records.

> **Disclosure:** This is a simulated logistics operation. All performance figures and operational findings refer to synthetic historical shipments. The model has not been deployed or proven to improve real-world delivery times, customer outcomes, or profitability.

## 1. Executive results

| Metric | Verified result | Interpretation |
|---|---:|---|
| Final test cohort | **2,000 shipments** | Chronologically latest 20% of dispatches |
| Eligible final training records | **7,998 shipments** | Earlier completed training and validation records |
| Final test MAE | **11.59 min** | Mean absolute ETA error |
| Final test RMSE | **15.56 min** | Sensitive to larger ETA misses |
| Final test MSE | **242.01 min²** | Mean squared ETA error |
| Final test R² | **0.8889** | Relative squared-error improvement versus test-cohort mean reference |
| Final test MAPE | **13.39%** | Average absolute percentage error |
| Mean signed error | **+0.29 min** | Slight overall underprediction (actual − predicted) |
| Mean actual / predicted duration | **98.87 / 98.58 min** | Final test group |
| Mean promised duration | **116.01 min** | Evaluated against each shipment individually |
| Actual late shipments | **358 / 2,000 (17.90%)** | Actual duration exceeded promise |
| Strict potentially-late alerts | **179 / 2,000 (8.95%)** | Predicted duration ≥ promise |
| Strict alert precision / recall | **65.36% / 32.68%** | 117 TP, 62 FP, 241 FN, 1,580 TN |
| Expanded review precision / recall | **36.35% / 81.84%** | Strict alerts + 15-minute monitoring band; 806 reviews |

### Model validation comparison (1,998 shipments)

| Model | MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| **LightGBM** | **11.3821** | **15.1235** | **0.8979** |
| Linear Regression | 12.2408 | 16.3168 | 0.8812 |
| Random Forest | 12.5322 | 16.7345 | 0.8750 |
| Decision Tree | 13.9340 | 18.4971 | 0.8473 |
| Median Baseline | 37.3015 | 49.0555 | −0.0742 |

The selected LightGBM model reduced validation MAE by approximately **69.49%** relative to the median baseline (**37.30 → 11.38 minutes**). That is a reduction in *prediction error*, not actual travel time. The final 2,000-shipments test set was evaluated after selecting LightGBM; test results were not used to retune the model or risk rule.

## 2. Operational findings

- **Strict risk alerts:** 179 flagged, with **117** actually late. Of **358** observed late shipments, **241** were not flagged by the strict rule.
- **Monitor band:** 627 shipments, **176** actually late. Reviewing strict plus monitor bands retrospectively covers **293 of 358** late shipments, at a workload of 806 reviews (**40.30% of the test cohort**).
- **Highest late-rate distance band:** 60+ km, **58/187 = 31.02%** late; corresponding test MAE **17.22 minutes**.
- **Largest missed-delay volume:** 15–30 km (**72 misses**) and 30–45 km (**67 misses**); together **139/241 = 57.68%** of strict alert misses.
- **Higher late-rate segments:** Severe forecast traffic **87/244 = 35.66%**; Express **151/525 = 28.76%**; Truck **101/438 = 23.06%**.
- **Permutation importance on validation:** Distance was strongest, with a **+36.136 minute** rise in MAE when shuffled; traffic forecast **+5.915**, planned stops **+3.518**, vehicle type **+2.014**. Importance is not a causal effect or coefficient.
- **Reliability on validation:** 72.67% of ETA estimates fell within ±15 minutes; the 90th percentile of absolute error was **24.35 minutes**. A fixed ±15-minute operational band is a heuristic, **not** a statistically calibrated prediction interval.

### Business review scenarios — assumptions, not real costs

Assume a **₹20 review cost per shipment** and a **₹150 conditional service credit per qualifying late delivery**.

| Scenario | Reviewed | Includes actual late | Hypothetical review cost |
|---|---:|---:|---:|
| Strict alerts | 179 | 117 | ₹3,580 |
| Strict + monitor | 806 | 293 | ₹16,120 |

If **every** one of the 358 late shipments qualified for ₹150, the conditional gross credit exposure would be **₹53,700**. This is not a measured liability or estimated savings. Reviewing additional shipments has no demonstrated preventive effect in this retrospective dataset.

**Business recommendation:** Pilot a two-tier, capacity-aware dispatch review: prioritize 179 strict alerts, then monitor-band shipments as capacity permits. Measure review workload, late shipment coverage, operational interventions and eventual outcomes on **new future** shipments before changing thresholds or customer commitments.

For the complete ranked findings, assumptions, owners, actions and targets, see [`docs/executive_decision_report.md`](docs/executive_decision_report.md).

## 3. Dataset, inputs and leakage controls

**Grain:** One row per completed historical shipment. **Target:** `actual_delivery_minutes`, positive dispatch-to-completion time.

The **12 raw inputs** known at dispatch comprise five numeric features (`dispatch_hour`, `day_of_week`, `distance_km`, `package_weight_kg`, `planned_stops`) and seven categorical features (`origin_region`, `destination_region`, `vehicle_type`, `courier_service`, `delivery_priority`, `forecast_traffic_level`, `forecast_weather_condition`).

`shipment_id`, `dispatch_datetime` (used for chronological splitting), `promised_delivery_minutes` (used as the operational threshold) and the post-delivery `actual_delivery_minutes` **are excluded from model input features**. Preprocessing is fit only on fitting records inside a scikit-learn Pipeline.

The input features expand to **25 encoded columns** for the reference-category Linear Regression configuration and **32 encoded columns** for tree-based regressors.

### Evaluation chronology

1. Sort 10,000 shipments by dispatch timestamp and ID.
2. Reserve the earliest 6,000 dispatches for training; reserve the next 2,000 dispatches for validation, excluding the two whose outcomes were not yet available by the test cutoff. **Validation count = 1,998**.
3. Use validation metrics (plus diagnostics and complexity) to select LightGBM.
4. Refit the selected preprocessing/model pipeline on **7,998 eligible completed shipments**.
5. Evaluate **once** on the final 2,000 dispatches from **2026-08-17 12:46**, then export and reload the saved pipeline.

Feature importance and residual breakdowns were examined on the validation set; operational alert performance was retrospectively assessed on the final test set without using it to choose a new model or revise thresholds.

## 4. Reproduction in Windows / VS Code

**Environment used in this project:** Python **3.13.16** in project-local `.venv`. Package versions are recorded during final QA in `output/final_qa_report.json`.

```powershell
# Run from the repository root in an existing activated .venv
python -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, open Command Prompt inside the VS Code terminal:

```powershell
cmd.exe /k .\.venv\Scripts\activate.bat
```

Open `notebooks/delivery_eta_regression.ipynb`, choose the `.venv` kernel and **Run All** to reproduce the analysis and outputs. Run QA *after* saving notebook results:

```powershell
python python/final_qa.py
```

Expected end message: **`QA STATUS: PASS — READY FOR GITHUB`**. The script does not retrain or modify the saved model; it verifies source data, chronological cutoffs, notebook content, evaluation outputs, risk flags, key metrics, charts and report completeness, and writes `output/final_qa_report.json`.

To reproduce the same synthetic data from seed, see `python/generate_data.py`. **Do not regenerate the data to optimize results.**

## 5. Repository organization

```text
logistics-delivery-eta-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/deliveries.csv
├── notebooks/delivery_eta_regression.ipynb
├── python/
│   ├── generate_data.py
│   └── final_qa.py
├── models/eta_regression_pipeline.joblib
├── output/
│   ├── figures/                        # Six charts
│   ├── predictions/shipment_eta_predictions.csv
│   ├── model_comparison.csv
│   ├── model_metrics.json
│   ├── feature_importance_validation.csv
│   ├── operational_segment_analysis.csv
│   ├── risk_review_policy_comparison.csv
│   ├── operational_scenario.json
│   └── final_qa_report.json
└── docs/
    ├── business_requirements.md
    ├── executive_decision_report.md
    └── interview_preparation.md
```

## 6. Limitations

This is synthetic, observational, retrospective data; relationships learned from the generator may not transfer to real routes, drivers, traffic or demand. Distance dominates model predictions; extreme events and some long journeys remain hard to predict. Model feature importance is **not causal identification**. Prediction-error averages do not guarantee individual accuracy, and the ±15-minute monitor threshold is **not a calibrated probability or interval**. These ETA predictions do not change contractual promises or automate dispatch. Additional review work has an assumed cost and no demonstrated savings without a prospective study.

## 7. Interview / project defense

See [`docs/interview_preparation.md`](docs/interview_preparation.md) for a 60–90-second pitch, model design defense, metric interpretation, data leakage answers, limitations and difficult follow-ups.
