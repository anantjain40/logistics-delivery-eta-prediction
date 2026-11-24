# Project 09 — Executive Decision Report

**Logistics Delivery ETA Prediction & Operational Intelligence**  
**Data status:** Synthetic demonstration dataset; all operational findings are retrospective and illustrative.  
**Evaluation cohort:** 2,000 chronologically held-out shipments dispatched from 17 August 2026, 12:46 onward.  
**Selected model:** LightGBM regressor, selected by validation performance before the final holdout was evaluated.  
**Decision status:** Recommend a limited, monitored dispatch-support pilot—not autonomous changes to customer commitments.

## 1. Executive summary

The LightGBM model, retrained on **7,998 earlier completed shipments**, achieved a **test-set MAE of 11.59 minutes**, RMSE of **15.56 minutes**, R² of **0.8889**, and MAPE of **13.39%** on 2,000 untouched test shipments. Mean signed error (actual minus prediction) was **+0.29 minutes**. These figures describe ETA prediction accuracy, not a reduction in travel time.

Historical service performance remains a concern: **358 of 2,000 shipments arrived after their promised durations (17.90%)**. The model-generated **strict Potentially late** flag identified **117** of those late shipments, with **65.36% precision**, **32.68% recall**, and **179 alerts (8.95% of the cohort)**. It missed **241** late shipments. The intermediate **Monitor closely** band contained **176** of those misses; reviewing both bands would retrospectively cover **293 of 358** late shipments (**81.84% recall**) at the cost of reviewing **806 shipments (40.30%)**, including 513 ultimately on-time shipments.

**Recommendation:** Pilot two-tier dispatch triage. Prioritize 179 strict alerts, then allocate review capacity to the 627 monitor-closely shipments, while monitoring missed delays, false alerts, segment reliability, and actual intervention effectiveness. The 15-minute monitoring buffer is a heuristic, **not a calibrated probability or prediction interval**. Do not adjust the trained model or operational rule after viewing this final holdout without evaluating on genuinely new future data.

## 2. Business problem and analytical scope

The fictional logistics operation needs more accurate dispatch-time ETAs and an actionable way to identify consignments at risk of exceeding customer delivery commitments. The prediction target is **actual dispatch-to-delivery duration in minutes**. Only information known by dispatch is used by the model; shipment ID, actual duration, post-delivery information, and promised minutes are excluded from model predictors. Promised minutes are retained solely for operational risk comparisons. Training and validation use earlier shipments with outcomes known by the relevant chronological cutoff.

This is a decision-support proof of concept on simulated data, not an operationally deployed system. Historical arrival status is used solely for retrospective assessment, never for flagging a shipment at dispatch.

## 3. Historical logistics KPIs — final test cohort

| KPI | Result | Interpretation |
|---|---:|---|
| Historical shipments evaluated | 2,000 | Final held-out cohort |
| Average actual duration | 98.87 min | Mean realized dispatch-to-delivery time |
| Average model-predicted duration | 98.58 min | Mean predicted duration |
| Average promised duration | 116.01 min | Mean shipment-level commitment |
| Actually late shipments | 358 | Actual time exceeded that shipment's promise |
| Actual late-delivery rate | 17.90% | 358 / 2,000 |
| On-time shipments | 1,642 | 2,000 − 358 |
| Final model mean signed error | +0.29 min | Slight average underprediction (actual − predicted) |

A lower **average** actual duration than average promise does not imply universal on-time performance: lateness is assessed against **each shipment's own** promised duration.

## 4. Regression model selection and final evaluation

### 4.1 Validation comparison (same 1,998 validation shipments)

| Model | MAE (min) | MSE (min²) | RMSE (min) | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | **11.3821** | **228.7206** | **15.1235** | **0.8979** | **13.0303%** |
| Linear Regression | 12.2408 | 266.2390 | 16.3168 | 0.8812 | 14.1858% |
| Random Forest | 12.5322 | 280.0425 | 16.7345 | 0.8750 | 14.0689% |
| Decision Tree | 13.9340 | 342.1444 | 18.4971 | 0.8473 | 15.4595% |
| Median Baseline | 37.3015 | 2406.4440 | 49.0555 | −0.0742 | 42.8958% |

LightGBM reduced validation MAE versus the median baseline by **25.92 minutes / 69.49%**, and beat Linear Regression by approximately **0.86 minutes**. These figures are predictive improvements, not reductions in actual delivery duration. Linear Regression's training R² was **0.8728** and adjusted training R² **0.8723** with **25** encoded predictors; adjusted R² was not used to select tree ensembles.

### 4.2 One-time final test evaluation (2,000 shipments)

| Metric | Final test result |
|---|---:|
| MAE | **11.59 min** |
| MSE | 242.01 min² |
| RMSE | **15.56 min** |
| R² | **0.8889** |
| MAPE | 13.39% |
| Mean signed error | +0.29 min |

The final LightGBM pipeline was saved as `models/eta_regression_pipeline.joblib`, reloaded, and verified to reproduce the sampled predictions. Comparisons and final metrics are exported to `output/model_comparison.csv` and `output/model_metrics.json`. The final 2,000 shipments were not used for model selection or refitting.

## 5. Prediction reliability and model limitations

**Earlier validation residual analysis (1,998 validation shipments):** mean signed error **+0.83 min**, median signed error **−0.12 min**, **72.67%** of predictions within ±15 minutes, 90th-percentile absolute error **24.35 min**, and **75** underpredictions above 30 minutes (3.75%). The validation 60+ km band had **MAE 17.40 min**, compared with **8.53 min** for under-15-km trips. These figures suggest unequal error magnitude across route distances. The 90th percentile of observed absolute errors is **not** a calibrated prediction interval.

**Test cohort segment MAE:** 60+ km **17.22 min** versus under 15 km **9.24 min**; Truck **14.18 min** versus Bike **9.75 min**; Severe traffic **13.54 min** versus Low traffic **10.77 min**. Higher variability in long journeys warrants monitoring. These are segment associations, not causal estimates.

### 5.1 Original-input permutation importance (validation MAE increase, minutes)

| Feature | Importance | Interpretation |
|---|---:|---|
| Distance (km) | **+36.136** | Largest loss of predictive accuracy when shuffled |
| Forecast traffic | +5.915 | Traffic information contributes to ETA prediction |
| Planned stops | +3.518 | Route stop count provides useful information |
| Vehicle type | +2.014 | Transport mode contributes |
| Forecast weather | +1.313 | Weather forecast contributes |
| Destination region | +0.179 | Small unique impact on this validation sample |
| Delivery priority | +0.159 | Small unique impact |
| Courier service | +0.123 | Small unique impact |
| Day of week | −0.001 | Approximately zero importance |
| Dispatch hour | −0.015 | Approximately zero importance |
| Origin region | −0.016 | Approximately zero importance |
| Package weight | −0.024 | Approximately zero importance |

An importance of **+36.136** does **not** mean each additional kilometre costs 36 minutes. It means shuffling that feature increased validation MAE by about 36.136 minutes on average. Small negative values are shuffle noise or shared information, not evidence of negative causal effects.

## 6. Dispatch-time ETA risk classifications

**Predicted buffer = promised duration − predicted duration.** Strict **Potentially late** if buffer ≤ 0; **Monitor closely** if 0 < buffer ≤ 15 minutes; **Lower concern** if buffer > 15 minutes. These are operational heuristics, not late-delivery probabilities.

| Risk band | Shipments | Share of test | Actually late | Actual late rate |
|---|---:|---:|---:|---:|
| Potentially late | 179 | 8.95% | 117 | 65.36% |
| Monitor closely | 627 | 31.35% | 176 | 28.07% |
| Lower concern | 1,194 | 59.70% | 65 | 5.44% |
| **Total** | **2,000** | **100.00%** | **358** | **17.90%** |

### 6.1 Alert confusion matrices

| Strict Potentially late alert | Actually late | Actually on time |
|---|---:|---:|
| Alert triggered | **TP 117** | **FP 62** |
| No strict alert | **FN 241** | **TN 1,580** |

| Expanded alert = Potentially late or Monitor closely | Actually late | Actually on time |
|---|---:|---:|
| Review | **TP 293** | **FP 513** |
| No review | **FN 65** | **TN 1,129** |

| Operational policy | Reviewed | Review load | Precision | Recall | Missed actual delays |
|---|---:|---:|---:|---:|---:|
| **Strict alerts only** | 179 | 8.95% | 65.36% | 32.68% | 241 |
| **Strict + monitor** | 806 | 40.30% | 36.35% | 81.84% | 65 |

Expanding the review group adds **627** shipments and retrospectively includes **176** additional late shipments but also **451** additional on-time shipments. This is a **workload versus coverage** comparison; it does not show that alerts prevent lateness.

## 7. Ranked operational problems

### Priority 1 — Missed actual delays

The strict rule missed **241 / 358** actual late deliveries (**67.32%**). Of these misses, **176** were in the Monitor closely category and **65** were in Lower concern. The strict rule is suitable for focused prioritization but not comprehensive delay detection.

### Priority 2 — Long-distance delivery reliability

**60+ km:** 58 late of 187 (**31.02% late**), MAE **17.22 min**, and 39 missed late alerts. This group has the highest late rate by distance and the highest distance-band MAE. Longer routes may need greater ETA buffers and exception visibility, subject to future validation.

### Priority 3 — Concentration of missed alerts in high-volume medium distances

**15–30 km** recorded **72 missed late shipments** (**29.88% of all 241 misses**); **30–45 km** recorded **67** (**27.80%**). Combined, **139 missed delays (57.68%)** originated in the 15–45 km bands. **High absolute miss volume and high within-band late rate are different phenomena**, requiring separate priorities.

### Priority 4 — Traffic and service-priority pressures

Severe-traffic dispatches recorded **87 late of 244 (35.66%)**, with MAE **13.54 min**. Express commitments recorded **151 late of 525 (28.76%)**, versus **207 late of 1,475 standard shipments (14.03%)**. These associations warrant investigation into promise realism and operating procedures, not causal conclusions.

### Priority 5 — Regional, vehicle and review-capacity differences

The Central region recorded a **22.22% late rate** (76/342) and East **21.78%** (88/404). Trucks recorded **23.06% late** (101/438) and **14.18-min MAE**, while vans contributed **128 missed late cases (53.11% of all missed alerts)** because of their volume. Capacity to review alerts is a separate operational constraint.

## 8. Detailed segment evidence — final test

### 8.1 Distance bands

| Band | Shipments | Actually late | Late rate | Missed late | MAE (min) |
|---|---:|---:|---:|---:|---:|
| Under 15 km | 344 | 43 | 12.50% | 37 | 9.24 |
| 15–30 km | 691 | 95 | 13.75% | 72 | 9.97 |
| 30–45 km | 530 | 105 | 19.81% | 67 | 12.07 |
| 45–60 km | 248 | 57 | 22.98% | 26 | 14.12 |
| 60+ km | 187 | 58 | 31.02% | 39 | 17.22 |

### 8.2 Destination regions

| Region | Shipments | Actually late | Late rate | Missed late | MAE (min) |
|---|---:|---:|---:|---:|---:|
| East | 404 | 88 | 21.78% | 54 | 12.18 |
| West | 438 | 72 | 16.44% | 52 | 11.54 |
| North | 443 | 73 | 16.48% | 51 | 11.35 |
| Central | 342 | 76 | 22.22% | 46 | 11.25 |
| South | 373 | 49 | 13.14% | 38 | 11.63 |

### 8.3 Transport mode

| Vehicle | Shipments | Actually late | Late rate | Missed late | MAE (min) |
|---|---:|---:|---:|---:|---:|
| Van | 1,079 | 194 | 17.98% | 128 | 11.37 |
| Truck | 438 | 101 | 23.06% | 70 | 14.18 |
| Bike | 483 | 63 | 13.04% | 43 | 9.75 |

### 8.4 Forecast traffic

| Forecast traffic | Shipments | Actually late | Late rate | Missed late | MAE (min) |
|---|---:|---:|---:|---:|---:|
| Low | 429 | 49 | 11.42% | 36 | 10.77 |
| Moderate | 745 | 118 | 15.84% | 98 | 11.49 |
| High | 582 | 104 | 17.87% | 70 | 11.51 |
| Severe | 244 | 87 | 35.66% | 37 | 13.54 |

### 8.5 Delivery priority

| Priority | Shipments | Actually late | Late rate | Missed late | MAE (min) |
|---|---:|---:|---:|---:|---:|
| Standard | 1,475 | 207 | 14.03% | 155 | 11.53 |
| Express | 525 | 151 | 28.76% | 86 | 11.77 |

## 9. Illustrative financial and staffing scenarios

**Explicit scenario assumptions:** ₹150 service credit **only if** a late delivery qualifies; ₹20 review cost per reviewed shipment. These rates are hypothetical. The dataset contains no observed liability, intervention or savings ledger.

| Scenario | Shipments reviewed | Hypothetical review cost | Actually late among reviewed | Conditional credit exposure for those late shipments |
|---|---:|---:|---:|---:|
| Strict alert only | 179 | ₹3,580 | 117 | ₹17,550 |
| Strict + monitor | 806 | ₹16,120 | 293 | ₹43,950 |

If **all 358** historical late shipments separately qualified for the assumed credit, total conditional credit exposure would be **358 × ₹150 = ₹53,700**, irrespective of alert policy **absent verified prevention**.

An expanded review adds **627** reviews and **₹12,540** in assumed review expense, bringing **176** additional late shipments into review. The added review cost per additional **historically late case included for review** is **₹12,540 / 176 = ₹71.25**. This is **not** a prevented-loss estimate, return on investment, or causal effect. A real cost-benefit case would require measured intervention efficacy, confirmed credit eligibility, staffing constraints and customer impact.

## 10. Recommendations and management action plan

| Priority | Proposed action | Responsible owner | Proposed timing | Measurable pilot outcome |
|---|---|---|---|---|
| P1 | Pilot a two-tier triage queue; first review strict potentially-late cases, then monitor-band shipments within staffing capacity | Dispatch Operations Manager | Weeks 1–2 | Cases reviewed, alert precision/recall, review time, missed-delay count |
| P1 | Investigate 60+ km routes and long-tail prediction errors; record driver/route exceptions and promise adherence | Regional Logistics Manager | Weeks 1–3 | 60+ km late rate and MAE, underpredictions >30 min |
| P1 | Audit moderate-traffic and 15–45 km missed alerts for repeatable operating patterns without retroactively tuning the final test rule | Operations Excellence Lead | Weeks 2–3 | Missed alerts by band and traffic category |
| P2 | Examine severe-traffic and express commitments for feasibility; separate route delay from aggressive customer promises | Service Delivery and Customer Experience | Weeks 2–4 | Late rate by traffic, priority and promise window |
| P2 | Monitor prediction accuracy, signed bias, drift, segment performance and alert-workload capacity on new future shipments | ML / Analytics Lead | Ongoing from pilot start | MAE, RMSE, missed alerts, error by segment, calibration checks if probabilities introduced |
| P2 | Run a prospective comparison of dispatch outcomes with and without model-supported review before claiming actual improvements | Analytics Lead + Operations | After pilot readiness | Actual intervention rate, late rate, service credits incurred, net review cost |

**Illustrative pilot targets to approve or revise—not achieved results:** Keep future-period MAE ≤12 minutes; examine segments above 15-minute MAE; target review-policy recall ≥75% while keeping operational review load near an agreed capacity bound; aim for a late-delivery rate below the 17.90% historical test benchmark **only if intervention data supports that goal**. Targets are management proposals, not predictions of success. The expanded review policy produced 81.84% recall at 40.30% workload retrospectively; this must be confirmed prospectively.

## 11. Decision and limitations

**Decision:** Approve a limited **human-in-the-loop** operational pilot with logged risk categories, actual outcomes and actions. Do not automatically revise contractual promises or treat an alert as a probability. Use dispatch-time features only; track outcome availability and avoid future leakage.

Key limitations: all 10,000 deliveries are synthetic; operational patterns reflect data-generation assumptions; model performance on real routes may differ; historical test-set comparisons of alert rules are descriptive rather than a fresh validation; historical risk categories cannot prove intervention benefit; alert thresholds are heuristics; feature importance is predictive rather than causal; long-distance and unexpected-delay reliability is weaker; ₹150 credit and ₹20 review amounts are unverified assumptions. The final test was used once for model evaluation and subsequent retrospective business analysis; it should not be used for further model/rule optimization without a new prospective evaluation dataset.

## 12. Reproduction and evidence references

- Source: `data/deliveries.csv` (synthetic, one row per completed dispatch).
- Notebook: `notebooks/delivery_eta_regression.ipynb`.
- Selected model: `models/eta_regression_pipeline.joblib` (reloaded and verified).
- Model comparison and final metrics: `output/model_comparison.csv`, `output/model_metrics.json`.
- Validation feature ranking: `output/feature_importance_validation.csv`.
- Final cohort predictions and risk flags: `output/predictions/shipment_eta_predictions.csv`.
- Operational segment summaries: `output/operational_segment_analysis.csv`.
- Risk policy evaluation: `output/risk_review_policy_comparison.csv`.
- Cost and credit assumptions: `output/operational_scenario.json`.
- Charts: `output/figures/` (EDA, actual vs predicted, residual diagnostics, feature importance).

*All numbers in this report are based on the supplied Project 09 notebook results. The project simulates a fictional company and does not establish results for a real logistics operator.*
