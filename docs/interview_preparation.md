# Project 09 — Interview Preparation and Technical Defense

**Portfolio context:** First regression project following a customer-churn classification project. All data and business outcomes below are from a disclosed **synthetic** simulation.

## 60–90-second interview presentation

> I built an end-to-end Logistics Delivery ETA Prediction and Operational Intelligence project using 10,000 synthetic historical shipments. The business question was whether dispatch-time information—such as distance, vehicle, traffic forecast and planned stops—could predict delivery duration and help flag commitments at risk. I built chronological training, validation and final test cohorts and excluded actual outcomes, shipment IDs and promised delivery minutes from the predictive features. I compared a median baseline, Linear Regression, Decision Tree, Random Forest and LightGBM. LightGBM achieved the best validation MAE at 11.38 minutes compared with a 37.30-minute baseline. After selection, I refitted on 7,998 earlier completed deliveries, obtaining 11.59 minutes MAE, 15.56 minutes RMSE and R² 0.8889 on 2,000 untouched test shipments. Residual analysis exposed larger errors on long-distance routes, and permutation importance identified distance as the main predictor. For business use, I compared predicted duration against promised minutes and evaluated strict versus expanded operational alert policies. Strict alerts had 65.36% precision but only 32.68% recall; a larger review group covered 81.84% of historically late deliveries at higher workload. I documented risks, hypothetical review costs and a monitored-pilot recommendation. These are retrospective results on synthetic data, not claims of deployed business savings.

## Core questions and strong answers

| Question | Interview-ready answer |
|---|---|
| 1. What was the business objective? | Predict dispatch-to-delivery duration in minutes and turn that ETA into transparent, limited-risk operational alerts. |
| 2. Why regression, not classification? | The target is continuous minutes, not a binary event. Lateness flags are a separate rule based on the regression output and the shipment's promise. |
| 3. What was the data grain? | One row per completed shipment; 10,000 rows and 16 source columns in one synthetic CSV. |
| 4. What inputs did you use? | Twelve original dispatch-known features: five numeric and seven categorical. Encoded input widths were 25 for linear and 32 for trees. |
| 5. Why not use promised minutes as a feature? | It is a policy threshold and could shortcut the ETA task. We compare promise against the prediction after the regression model runs. |
| 6. How did you prevent leakage? | Fit the encoder/imputer on training records within pipelines, sort dispatches chronologically, exclude post-delivery outcomes, and require training outcomes to have completed by the appropriate cutoff. |
| 7. Why 6,000 / 1,998 / 2,000 rather than 6,000 / 2,000 / 2,000? | Two validation-window shipments were still underway when the test window began, so their labels could not have been known at that point. |
| 8. Why establish a median baseline? | It shows how much accuracy the features and algorithms add beyond predicting one fixed duration. Its validation MAE was 37.30 minutes. |
| 9. What did Linear Regression teach you? | It estimates an intercept and coefficients by minimizing training squared errors. Distance was approximately +2.15 predicted minutes per km, and planned stops +5.11 minutes per additional stop, holding other encoded predictors fixed. Those are associations, not proof of causation. |
| 10. What changed from classifier trees? | Regression leaves predict continuous minutes; Random Forest Regression averages tree predictions instead of averaging class probabilities/voting. |
| 11. Why LightGBM? | It delivered the lowest validation MAE (11.38 minutes), lowest validation RMSE (15.12), and highest validation R² (0.8979) among the five compared algorithms, with manageable complexity. |
| 12. Why not select on the test set? | Using the final test set to choose settings would make its reported performance optimistic. Test was held out until selection and final refitting were complete. |
| 13. Did you tune on the test results? | No. The final model and initial operational risk band were established before assessing the final test cohort. |
| 14. What does MAE 11.59 mean? | The mean absolute ETA error across 2,000 test shipments was 11.59 minutes; it does not guarantee each shipment is within that error. |
| 15. Why examine RMSE as well? | It weights large errors more heavily; final test RMSE was 15.56 minutes, larger than MAE, encouraging investigation of extreme misses. |
| 16. What does R² 0.8889 mean? | Relative to predicting the test cohort mean, model squared error was about 88.89% lower under the standard R² calculation. It is not 88.89% classification accuracy. |
| 17. Why did the median baseline have negative validation R²? | It predicted the training median, which can have greater squared error than a constant prediction at the validation cohort's actual mean. |
| 18. What is adjusted R²? | A training OLS fit diagnostic that penalizes extra predictors. Our training Linear Regression R² was 0.8728 and adjusted R² 0.8723 for 25 encoded predictors; we did not use it to select tree ensembles. |
| 19. What is a residual? | Actual minus predicted duration. Positive residuals are optimistic predictions that underestimate trip time; negative residuals are overpredictions. |
| 20. Did the model exhibit bias? | Final test average signed error was +0.29 minutes, near zero overall. That does not eliminate segment-level bias or large errors. |
| 21. What did your residual investigation reveal? | On validation, 72.67% were within ±15 minutes and the 90th-percentile absolute error was 24.35 minutes. MAE was higher for 60+ km than for short-distance trips. |
| 22. What was the most important input? | Distance: shuffling it increased validation MAE by 36.136 minutes. This is a performance perturbation, not extra travel minutes per kilometre. |
| 23. What happens when feature importance is negative? | Shuffling produced a negligible average MAE improvement over five repetitions; that suggests limited unique signal under this fitted model, not a negative causal relationship. |
| 24. How do ETA risk rules work? | Compute promised minus predicted minutes. Potentially late if buffer ≤0; monitor closely if 0<buffer≤15; otherwise lower concern. The band is a heuristic, not calibrated probability. |
| 25. What were the observed late outcomes? | 358/2,000 test shipments (17.90%) arrived later than their individual promises. |
| 26. What are the strict-alert TP/FP/FN/TN counts? | TP=117, FP=62, FN=241, TN=1,580. Precision 117/179=65.36%; recall 117/358=32.68%. |
| 27. Why compare broader reviews? | Reviewing strict plus monitor groups would historically include 293 of 358 late shipments (81.84% recall), but entail 806 reviews including 513 on-time shipments. This shows workload/coverage trade-offs. |
| 28. Which segments should management investigate? | 60+ km deliveries had 31.02% late rate and 17.22-minute MAE; severe traffic had a 35.66% late rate; 15–45 km accounted for 139 of 241 strict-alert misses. |
| 29. What financial impact did you estimate? | An explicitly hypothetical ₹20 review cost implies ₹3,580 for strict or ₹16,120 for expanded reviews. ₹150 per qualifying late shipment would create ₹53,700 conditional gross exposure if all 358 were eligible; no verified savings or liability was claimed. |
| 30. How would you validate operational benefits? | Run a prospective, capacity-aware pilot comparing interventions, customer outcomes and review workload on genuinely new shipments; monitor MAE, miss rate and segment-level reliability. |
| 31. Why use synthetic data? | To demonstrate full ML workflow with explicit dispatch-time features and reproducible outcomes, while acknowledging that simulation-specific patterns do not establish real production performance. |
| 32. How is Project 09 different from Project 08? | This predicts a continuous duration, uses MAE/RMSE/R² and regression residuals, and derives risk-alert labels from ETA vs promise, rather than directly predicting churn as a classification target. |

## Difficult follow-ups

**Q: Your model has R²≈0.89. Why is alert recall only 32.68%?**  
A: Strong average regression performance does not ensure precise classification near a deadline. The strict rule only flags predicted ETAs at or beyond the promise, so late shipments with slightly optimistic estimates are missed. A wider monitoring band improves retrospective coverage but incurs more reviews.

**Q: Can we claim your model reduced late deliveries or saved ₹53,700?**  
A: No. It reduced ETA prediction error versus a baseline. The ₹53,700 is a conditional illustration of service-credit exposure, not measured cost or savings, and no intervention experiment was conducted.

**Q: Why not remove package weight, dispatch hour or origin region immediately?**  
A: Their permutation importances were near zero or marginally negative on this validation cohort, but correlated signals and randomness complicate interpretation. Redesign requires a new validation protocol; the final test set should not be reused to optimize features.

**Q: Would you deploy the saved pipeline unchanged?**  
A: Not directly. First validate on representative real data, verify that forecast values are truly available at dispatch, assess fairness and regional performance, design reliable input handling and monitoring, and run a limited pilot.

**Q: Why save preprocessing together with the model?**  
A: The model expects a particular imputation/encoding representation. A joint pipeline prevents deployment from applying a different transformation and was verified by reloading and comparing predictions.

## Quick code/function defense

| Code | Why it is in the project |
|---|---|
| `pd.to_datetime`, `.sort_values` | Establish chronological order and validate timestamps |
| `pd.to_timedelta` | Ensure a historical shipment's actual result was available at the cutoff |
| `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder` | Convert 5 numeric and 7 categorical inputs consistently without fitting on validation/test |
| `Pipeline`, `clone` | Fit preprocessing and model together; create unfitted copies for fair comparisons |
| `DummyRegressor(strategy='median')` | Fixed-duration baseline |
| `LinearRegression()` | Linear least-squares reference and interpretable coefficients |
| `DecisionTreeRegressor`, `RandomForestRegressor` | Nonlinear rule-based and bagged-tree regressors |
| `LGBMRegressor` | Gradient-boosted tree regression |
| `mean_absolute_error`, `mean_squared_error`, `r2_score` | Predictive performance measured on unseen data |
| `permutation_importance` | Measure how disrupting each original dispatch feature changes validation MAE |
| `np.select` | Apply the predefined ETA-versus-promise rules to create alert groups |
| `joblib.dump` / `joblib.load` | Save and verify the exact fitted transformation/model pipeline |
| `assert`, `np.testing.assert_allclose` | Catch inconsistencies in data, splitting, predictions and export/reload |

**Interview discipline:** Clearly separate *model performance*, *risk-alert coverage*, and *actual intervention benefit*. They measure different things.
