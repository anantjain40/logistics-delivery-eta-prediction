# Business Requirements Document (BRD)
**Project:** Logistics Delivery ETA Prediction & Operational Intelligence  
**Business:** Fictional multi-region e-commerce logistics company  
**Status:** Approved project specification / synthetic demonstration  
**Scope:** One dataset, one regression target, five development phases

## Background & business problem
Delivery duration varies with distance, forecast traffic/weather, dispatch period, vehicle, region, and planned stops. Inaccurate ETAs can lead to missed commitments, reactive dispatch operations and poor customer communication. Dispatch teams need consistent estimates **at the moment of dispatch** and an auditable way to prioritize potentially late shipments.

## Objectives & main business questions
1. Predict dispatch-to-completion delivery duration in **minutes** using dispatch-available information only.
2. Compare a median baseline with Linear Regression, Decision Tree, Random Forest and LightGBM.
3. Quantify typical errors and unusually large/optimistic errors; check differences by distance, region, dispatch period and vehicle.
4. Flag predicted ETA overruns relative to contractual promises for operational review, not as calibrated late probabilities.
5. Recommend responsible pilot actions and transparent scenario-based business opportunities.

## Users & stakeholders
Logistics Operations Manager (decision owner); Dispatch Supervisors (shipment triage); Regional Delivery Managers (segment investigations); Customer Experience Manager (communication); Data Analytics Team (model validation and monitoring).

## Data contract
**Grain:** one row per historical completed shipment.  
**Only source:** `data/deliveries.csv` (10,000 reproducible synthetic observations).  
**Target:** `actual_delivery_minutes` — positive elapsed minutes from dispatch to delivery completion.  
**Candidate dispatch-time inputs:** `dispatch_hour`, `day_of_week`, `distance_km`, `origin_region`, `destination_region`, `vehicle_type`, `courier_service`, `package_weight_kg`, `delivery_priority`, `planned_stops`, `forecast_traffic_level`, `forecast_weather_condition`.  
**Administrative/business columns:** `shipment_id`, `dispatch_datetime`, `promised_delivery_minutes`. **Do not feed these directly into the model**; use datetime for chronological split and promised duration for risk comparison. Never use actual duration, delivery-completion information, realized delay causes or future observed traffic as inputs.

## Method & acceptance criteria
Perform consolidated audit → clean → validate → freeze; EDA; chronological train/validation/test split; training-fitted preprocessing; comparable model training; validation-driven selection; one untouched final test evaluation; model export/reload check. Report MAE and RMSE in minutes, MSE in minutes², R², residual patterns, dangerous underprediction and per-segment error. Initial **proposed** benchmark: target ≥15% validation MAE improvement over median baseline, with no unacceptable segment-level deterioration; performance is not guaranteed. Validate operational flag precision/recall against historical lateness before pilot use.

## Deliverables
Reproducible notebook; data generator and CSV; validation and model comparison artifacts; figures; saved prediction pipeline; historical shipment predictions and lateness flags; executive decision report; README; GitHub repository; interview explanations.

## Exclusions, assumptions & safeguards
Single-duration target only. No cost model, Power BI, Streamlit, route optimization, GPS integration or deployment. All records and any hypothetical financial scenarios are **synthetic**; findings cannot be presented as real-world impact. Predicted risk flags are not calibrated probabilities, association is not causation, and improved ETA accuracy does not itself shorten trips. Operational adoption requires prospective validation.
