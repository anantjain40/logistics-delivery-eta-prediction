"""Project 09 — consolidated, read-only final QA.

Run from the project root with: python python/final_qa.py
Reads source data, notebook, saved model, predictions and reports; does not refit
or modify the model or any historic observations. Writes one QA JSON summary.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "deliveries.csv"
NOTEBOOK = ROOT / "notebooks" / "delivery_eta_regression.ipynb"
MODEL = ROOT / "models" / "eta_regression_pipeline.joblib"
PRED = ROOT / "output" / "predictions" / "shipment_eta_predictions.csv"
METRICS = ROOT / "output" / "model_metrics.json"
COMPARISON = ROOT / "output" / "model_comparison.csv"
REPORT = ROOT / "docs" / "executive_decision_report.md"
BRD = ROOT / "docs" / "business_requirements.md"
INTERVIEW = ROOT / "docs" / "interview_preparation.md"
README = ROOT / "README.md"
FIGURES = ROOT / "output" / "figures"
FEATURES = ["dispatch_hour", "day_of_week", "distance_km", "package_weight_kg", "planned_stops",
            "origin_region", "destination_region", "vehicle_type", "courier_service",
            "delivery_priority", "forecast_traffic_level", "forecast_weather_condition"]
FORBIDDEN = {"shipment_id", "dispatch_datetime", "promised_delivery_minutes", "actual_delivery_minutes"}
EXPECTED_FIGURES = [
    "01_duration_and_traffic.png", "02_duration_by_segment.png",
    "03_actual_vs_predicted.png", "04_residual_distribution.png",
    "05_residuals_vs_predictions.png", "06_feature_importance.png",
]
checks: list[dict] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    condition = bool(condition)
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    print(f"{'PASS' if condition else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))


def near(actual: float, expected: float, tolerance: float = 0.03) -> bool:
    return np.isfinite(actual) and abs(float(actual) - expected) <= tolerance


def bools(s: pd.Series) -> pd.Series:
    b = s.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if b.isna().any():
        raise ValueError(f"Unknown boolean entries in {s.name}: {s[b.isna()].head().tolist()}")
    return b.astype(bool)


def main() -> None:
    print("\nPROJECT 09 — FINAL CONSOLIDATED QA")
    print("Project root:", ROOT)
    for label, path in [
        ("Dataset", DATA), ("Notebook", NOTEBOOK), ("Saved pipeline", MODEL),
        ("Shipment predictions", PRED), ("Model metrics", METRICS),
        ("Model comparison", COMPARISON), ("BRD", BRD),
        ("Executive decision report", REPORT), ("Interview preparation", INTERVIEW),
        ("Final README", README),
    ]:
        check(f"Artifact present: {label}", path.is_file(), str(path.relative_to(ROOT)))

    if not all(p.is_file() for p in [DATA, NOTEBOOK, MODEL, PRED, METRICS, COMPARISON]):
        print("\nRequired core files missing. Finish earlier notebook steps and rerun QA.")
        save_report({})
        sys.exit(1)

    data = pd.read_csv(DATA, parse_dates=["dispatch_datetime"])
    check("One dataset: 10,000 rows and 16 columns", data.shape == (10_000, 16), str(data.shape))
    check("10,000 unique shipment IDs", data.shipment_id.nunique() == 10_000)
    check("Source contains no missing values", int(data.isna().sum().sum()) == 0)
    check("Valid positive target and promise", bool(data.actual_delivery_minutes.gt(0).all()
          and data.promised_delivery_minutes.gt(0).all()))
    check("Source features available", set(FEATURES).issubset(data.columns))

    data = data.sort_values(["dispatch_datetime", "shipment_id"]).reset_index(drop=True)
    train_cutoff = data.loc[6000, "dispatch_datetime"]
    test_cutoff = data.loc[8000, "dispatch_datetime"]
    train = data.loc[data.dispatch_datetime < train_cutoff]
    validation = data.loc[(data.dispatch_datetime >= train_cutoff) & (data.dispatch_datetime < test_cutoff)]
    test = data.loc[data.dispatch_datetime >= test_cutoff]
    def complete_by(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        complete = df.dispatch_datetime + pd.to_timedelta(df.actual_delivery_minutes, unit="min")
        return df.loc[complete <= cutoff]
    train_ready = complete_by(train, train_cutoff)
    val_ready = complete_by(validation, test_cutoff)
    check("Chronological split 6,000 / 1,998 / 2,000",
          (len(train_ready), len(val_ready), len(test)) == (6000, 1998, 2000),
          f"{len(train_ready):,} / {len(val_ready):,} / {len(test):,}")
    check("Training/validation outcomes completed before test period",
          bool((pd.concat([train_ready, val_ready]).dispatch_datetime
            + pd.to_timedelta(pd.concat([train_ready, val_ready]).actual_delivery_minutes, unit="min")
            <= test_cutoff).all()))
    check("Test cutoff", test_cutoff == pd.Timestamp("2026-08-17 12:46:00"), str(test_cutoff))

    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    code = ["".join(c.get("source", [])) for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
    check("Notebook contains substantial analysis code", len(code) >= 12, f"{len(code)} code cells")
    notebook_source = "\n".join(code)
    required_patterns = ["permutation_importance", "LGBMRegressor", "RandomForestRegressor",
                         "LinearRegression", "DecisionTreeRegressor", "DummyRegressor",
                         "mean_absolute_error", "final_pipeline", "risk_category"]
    check("Notebook includes all core analysis stages",
          all(word in notebook_source for word in required_patterns))
    executed_cells = sum(c.get("execution_count") is not None for c in notebook.get("cells", []) if c.get("cell_type") == "code")
    check("Notebook execution output saved", executed_cells >= 10,
          f"{executed_cells} executed cells recorded; Save All in VS Code if low")

    pipeline = joblib.load(MODEL)  # Load only this locally trained, trusted file.
    prep = pipeline.named_steps["preprocessor"]
    used_features = set()
    for name, transformer, columns in prep.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
            used_features.update(map(str, columns))
    check("Predictive columns exclude IDs, promises and target",
          used_features == set(FEATURES) and used_features.isdisjoint(FORBIDDEN),
          f"{len(used_features)} raw model features")
    check("Selected model is LightGBM", type(pipeline.named_steps["model"]).__name__ == "LGBMRegressor")

    predictions = pd.read_csv(PRED)
    check("Exactly 2,000 saved predictions", len(predictions) == 2000)
    check("Prediction IDs unique", predictions.shipment_id.is_unique)
    check("Prediction IDs and order match chronological holdout",
          predictions.shipment_id.tolist() == test.shipment_id.tolist())

    model_pred = pipeline.predict(test[FEATURES])
    stored_pred = predictions.predicted_delivery_minutes.to_numpy(dtype=float)
    check("Reloaded model reproduces saved shipment predictions",
          bool(np.allclose(model_pred, stored_pred, rtol=0, atol=1e-5)),
          f"maximum difference {np.max(np.abs(model_pred-stored_pred)):.7f} min")
    actual = test.actual_delivery_minutes.to_numpy(dtype=float)
    promised = test.promised_delivery_minutes.to_numpy(dtype=float)
    check("Stored historical actual durations match dataset",
          bool(np.allclose(predictions.actual_delivery_minutes.to_numpy(dtype=float), actual, atol=1e-8)))
    check("Promises match original holdout records",
          bool(np.allclose(predictions.promised_delivery_minutes.to_numpy(dtype=float), promised, atol=1e-8)))
    check("Stored residuals follow Actual - Predicted",
          bool(np.allclose(predictions.prediction_error_minutes, actual-stored_pred, atol=1e-7)))
    check("Predicted buffer follows Promise - Predicted",
          bool(np.allclose(predictions.predicted_buffer_minutes, promised-stored_pred, atol=1e-7)))

    mae = mean_absolute_error(actual, model_pred)
    mse = mean_squared_error(actual, model_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(actual, model_pred)
    mape = 100 * mean_absolute_percentage_error(actual, model_pred)
    signed = float(np.mean(actual - model_pred))
    with open(METRICS, "r", encoding="utf-8") as f:
        mm = json.load(f)
    test_metrics = mm.get("final_test", {})
    recorded = [test_metrics.get(k) for k in ["mae_min", "mse_min2", "rmse_min", "r2", "mape_pct", "mean_signed_error_min"]]
    computed = [mae, mse, rmse, r2, mape, signed]
    check("Saved test metrics agree with independent recalculation",
          all(isinstance(x,(float,int)) and abs(x-y)<1e-5 for x,y in zip(recorded, computed)))
    check("Final test metrics match verified project results",
          all([near(mae, 11.59), near(mse, 242.01, 0.08), near(rmse, 15.56),
               near(r2, 0.8889, .0002), near(mape, 13.39, .05), near(signed, 0.29, .03)]),
          f"MAE {mae:.2f} | RMSE {rmse:.2f} | R2 {r2:.4f} | MAPE {mape:.2f}%")

    flags = bools(predictions.estimated_delay_risk_flag)
    actual_late = bools(predictions.actual_late_flag)
    calc_late = actual > promised
    calc_flag = model_pred >= promised
    cat = np.where(calc_flag, "Potentially late",
                   np.where((promised-model_pred) <= 15, "Monitor closely", "Lower concern"))
    check("Observed lateness and strict risk flags independently verified",
          bool(np.array_equal(actual_late.to_numpy(), calc_late)
          and np.array_equal(flags.to_numpy(), calc_flag)
          and np.array_equal(predictions.risk_category.astype(str).to_numpy(), cat)))
    tp = int(np.sum(calc_flag & calc_late))
    fp = int(np.sum(calc_flag & ~calc_late))
    fn = int(np.sum(~calc_flag & calc_late))
    tn = int(np.sum(~calc_flag & ~calc_late))
    check("TP/FP/FN/TN counts verified", (tp, fp, fn, tn) == (117, 62, 241, 1580),
          f"TP {tp} | FP {fp} | FN {fn} | TN {tn}")
    cat_counts = pd.Series(cat).value_counts().to_dict()
    check("Risk categories verified", cat_counts == {"Potentially late":179,
          "Monitor closely":627,"Lower concern":1194}, str(cat_counts))
    check("Expanded review correctly covers 293 late shipments",
          int(np.sum(np.isin(cat, ["Potentially late", "Monitor closely"]) & calc_late)) == 293)

    comparison = pd.read_csv(COMPARISON)
    check("Five validation models recorded", len(comparison) == 5
          and set(comparison.Model) == {"Median Baseline", "Linear Regression",
          "Decision Tree", "Random Forest", "LightGBM"})
    if "MAE_min" in comparison:
        comparison_mae = dict(zip(comparison.Model, comparison.MAE_min))
    else:
        comparison_mae = dict(zip(comparison.Model, comparison.Validation_MAE))
    check("LightGBM was chosen using validation MAE",
          comparison_mae.get("LightGBM", float('inf')) == min(comparison_mae.values()))
    missing_figures = [f for f in EXPECTED_FIGURES if not (FIGURES / f).is_file()]
    check("Six required evidence figures saved", not missing_figures,
          "All six found" if not missing_figures else "Missing: " + ", ".join(missing_figures))
    report_text = REPORT.read_text(encoding="utf-8") if REPORT.is_file() else ""
    check("Report discloses synthetic data, risks and exposure",
          all(w in report_text.lower() for w in ["synthetic", "11.59", "17.90", "53,700", "recall"]))
    readme_text = README.read_text(encoding="utf-8") if README.is_file() else ""
    check("README updated with actual model results",
          all(w in readme_text for w in ["11.59", "15.56", "0.8889", "17.90%"])
          and "Pending Phase" not in readme_text)

    summary = {"model": "LightGBM", "test_shipments":len(test),
        "metrics":{"MAE_min":round(mae,4),"MSE_min2":round(mse,4),
                   "RMSE_min":round(rmse,4),"R2":round(r2,6),
                   "MAPE_pct":round(mape,4),"mean_signed_error_min":round(signed,4)},
        "flags":{"TP":tp,"FP":fp,"FN":fn,"TN":tn},
        "python":platform.python_version(),
        "packages":{p:__import__(m).__version__ for p,m in
                    [("pandas","pandas"),("numpy","numpy"),("scikit-learn","sklearn"),
                     ("lightgbm","lightgbm"),("joblib","joblib")]}}
    save_report(summary)
    failures = [c for c in checks if c["status"] == "FAIL"]
    print("\n=== FINAL RESULT ===")
    print(f"PASSED: {len(checks)-len(failures)}/{len(checks)} checks")
    print("QA report: output/final_qa_report.json")
    print("QA STATUS:", "PASS — READY FOR GITHUB" if not failures else "FAIL — FIX BEFORE GITHUB")
    if failures:
        print("Review failed checks above and rerun. No model was modified.")
        sys.exit(1)


def save_report(summary: dict) -> None:
    path = ROOT / "output" / "final_qa_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"project": "Logistics Delivery ETA Prediction — Project 09",
        "checks": checks, "passed": sum(c["status"] == "PASS" for c in checks),
        "failed": sum(c["status"] == "FAIL" for c in checks),
        "summary": summary}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
