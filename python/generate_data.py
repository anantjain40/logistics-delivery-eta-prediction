"""Reproducible *synthetic* logistics ETA dataset for Project 09.

One row is a completed shipment. Predict at the moment of dispatch. All X
fields describe known dispatch-time information, including *forecasts* of
traffic/weather rather than realized future conditions. The only outcome is
actual_delivery_minutes, defined as dispatch-to-completion duration.

Simulation assumptions (illustrative, NOT measured business facts):
- 10,000 shipments dated 2025-11-01 to 2026-10-31, across five fictional regions;
- distances roughly 2–85 km, with dispatch-selected transport and planned stops;
- busy hours and weekdays tend to receive higher forecast traffic levels;
- longer drives, heavier traffic, certain vehicles/regions, stops and adverse
  weather contribute different minutes to the duration;
- nonlinear interactions, individual variability and rare unexpected incidents
  make the target imperfectly predictable;
- promises are calculated from dispatch features by a separate, imperfect
  policy rule; *the actual observed duration is never used to set a promise*;
- fixed seed, no tuned rerolls to inflate downstream ML scores.

This dataset is for demonstration and cannot support real-world causal or
operational performance claims. No public/private customer information used.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261113
N_SHIPMENTS = 10_000
REGIONS = np.array(["North", "South", "East", "West", "Central"])


def generate_dataset(n: int = N_SHIPMENTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Dispatch timestamps (one-year history); all later chronological evaluation
    # will be based on these timestamps rather than random shuffling.
    start = pd.Timestamp("2025-11-01 00:00:00")
    end = pd.Timestamp("2026-11-01 00:00:00")
    total_minutes = int((end - start).total_seconds() // 60)
    minute_offsets = rng.integers(0, total_minutes, size=n)
    dispatch_datetime = start + pd.to_timedelta(minute_offsets, unit="m")
    dispatch_hour = np.asarray(dispatch_datetime.hour, dtype=int)
    day_of_week = np.asarray(dispatch_datetime.dayofweek, dtype=int)  # Monday=0
    month = np.asarray(dispatch_datetime.month, dtype=int)

    origin_region = rng.choice(REGIONS, size=n, p=[0.22, 0.19, 0.18, 0.22, 0.19])
    destination_region = rng.choice(REGIONS, size=n, p=[0.21, 0.20, 0.20, 0.21, 0.18])
    same_region = origin_region == destination_region

    # Distances in this simulation represent local/metropolitan delivery legs.
    distance_km = np.where(
        same_region,
        rng.gamma(shape=2.5, scale=6.2, size=n),
        rng.gamma(shape=3.3, scale=10.5, size=n),
    )
    distance_km = np.round(np.clip(distance_km + 1.8, 2.0, 85.0), 2)

    # A dispatch planner chooses a feasible mode using known distance; these
    # probabilities are assumptions, not empirical transport optimization.
    p_bike = np.where(distance_km < 13, 0.72, np.where(distance_km < 28, 0.35, 0.06))
    p_truck = np.where(distance_km < 13, 0.03, np.where(distance_km < 28, 0.12, 0.36))
    mode_draw = rng.random(n)
    vehicle_type = np.where(mode_draw < p_bike, "Bike", np.where(mode_draw < p_bike + p_truck, "Truck", "Van"))
    courier_service = rng.choice(["In-House", "Partner-A", "Partner-B"], size=n, p=[0.49, 0.31, 0.20])

    package_weight_kg = rng.lognormal(mean=0.15, sigma=0.77, size=n)
    package_weight_kg *= np.where(vehicle_type == "Truck", 2.5, np.where(vehicle_type == "Van", 1.5, 0.75))
    package_weight_kg = np.round(np.clip(package_weight_kg, 0.08, 28.0), 2)
    delivery_priority = rng.choice(["Standard", "Express"], size=n, p=[0.73, 0.27])
    planned_stops = np.clip(
        rng.poisson(lam=np.where(vehicle_type == "Truck", 3.2, np.where(vehicle_type == "Van", 2.2, 1.4)), size=n) + 1,
        1,
        9,
    ).astype(int)

    # 'Forecast' labels are generated before constructing actual duration.
    rush_hour = np.isin(dispatch_hour, [8, 9, 10, 17, 18, 19, 20])
    business_day = day_of_week < 5
    traffic_pressure = rng.normal(0, 0.85, n) + 0.85 * rush_hour + 0.35 * business_day
    traffic_pressure += np.where(destination_region == "Central", 0.26, 0)
    forecast_traffic_level = np.select(
        [traffic_pressure < -0.2, traffic_pressure < 0.75, traffic_pressure < 1.65],
        ["Low", "Moderate", "High"],
        default="Severe",
    )

    monsoon_like = np.isin(month, [6, 7, 8, 9])
    winter_like = np.isin(month, [11, 12, 1, 2])
    weather_draw = rng.random(n)
    # Simple fictional forecast distribution; labels are known at dispatch.
    rain_p = np.where(monsoon_like, 0.30, 0.10)
    fog_p = np.where(winter_like, 0.13, 0.025)
    storm_p = np.where(monsoon_like, 0.04, 0.012)
    cloudy_p = 0.24
    forecast_weather_condition = np.where(
        weather_draw < storm_p, "Storm",
        np.where(weather_draw < storm_p + fog_p, "Fog",
        np.where(weather_draw < storm_p + fog_p + rain_p, "Rain",
        np.where(weather_draw < storm_p + fog_p + rain_p + cloudy_p, "Cloudy", "Clear"))),
    )

    # Ground-truth simulation: cannot be used by predictors. Long-distance trips
    # have higher conditional uncertainty; unobserved incidents create tail risk.
    nominal_speed_kmh = np.select(
        [vehicle_type == "Bike", vehicle_type == "Van"],
        [31.0, 38.0], default=33.0,
    )
    traffic_multiplier = pd.Series(forecast_traffic_level).map(
        {"Low": 1.0, "Moderate": 1.13, "High": 1.32, "Severe": 1.54}
    ).to_numpy(dtype=float)
    weather_multiplier = pd.Series(forecast_weather_condition).map(
        {"Clear": 1.00, "Cloudy": 1.035, "Rain": 1.14, "Fog": 1.12, "Storm": 1.30}
    ).to_numpy(dtype=float)
    region_offset = pd.Series(destination_region).map(
        {"North": 2.0, "South": -1.5, "East": 3.6, "West": 1.2, "Central": 5.0}
    ).to_numpy(dtype=float)
    courier_offset = pd.Series(courier_service).map(
        {"In-House": -1.0, "Partner-A": 1.8, "Partner-B": 4.0}
    ).to_numpy(dtype=float)

    travel_minutes = distance_km * 60 / nominal_speed_kmh * traffic_multiplier * weather_multiplier
    # Additional compounding under severe traffic, more than four planned stops
    # and a heavy vehicle means a single straight line is not a perfect fit.
    extra_interaction = np.where(
        (forecast_traffic_level == "Severe") & (distance_km > 28), 0.07 * travel_minutes, 0
    ) + np.maximum(planned_stops - 4, 0) ** 1.4 * 1.5
    base_duration = (
        8.5 + travel_minutes + planned_stops * 4.5 + 0.22 * package_weight_kg
        + region_offset + courier_offset + extra_interaction
        + np.where(rush_hour, 2.5, 0.0)
        - np.where(delivery_priority == "Express", 3.8, 0.0)
    )
    random_noise = rng.normal(loc=0.0, scale=5.0 + 0.075 * base_duration, size=n)
    unexpected_incident = (rng.random(n) < 0.048) * rng.gamma(shape=2.0, scale=13.0, size=n)
    actual_delivery_minutes = np.round(np.maximum(base_duration + random_noise + unexpected_incident, 7.0), 1)

    # Policy-based promise known at dispatch. This does not use the actual target.
    # It is *only* used later to compare risk flags, not as an input feature.
    policy_speed = np.select([vehicle_type == "Bike", vehicle_type == "Van"], [30.0, 36.0], default=31.0)
    promise_base = 15 + distance_km * 60 / policy_speed * (
        1 + np.where(forecast_traffic_level == "Severe", 0.39, np.where(
            forecast_traffic_level == "High", 0.22, np.where(forecast_traffic_level == "Moderate", 0.10, 0.0)
        ))
    )
    promise_base += planned_stops * 5.0 + np.where(delivery_priority == "Standard", 19.0, 5.0)
    promise_base += rng.normal(loc=0.0, scale=8.5, size=n)
    promised_delivery_minutes = np.ceil(np.maximum(promise_base, 15) / 5) * 5

    df = pd.DataFrame({
        "shipment_id": [f"SH-{i:06d}" for i in range(1, n + 1)],
        "dispatch_datetime": dispatch_datetime,
        "dispatch_hour": dispatch_hour,
        "day_of_week": day_of_week,
        "distance_km": distance_km,
        "origin_region": origin_region,
        "destination_region": destination_region,
        "vehicle_type": vehicle_type,
        "courier_service": courier_service,
        "package_weight_kg": package_weight_kg,
        "delivery_priority": delivery_priority,
        "planned_stops": planned_stops,
        "forecast_traffic_level": forecast_traffic_level,
        "forecast_weather_condition": forecast_weather_condition,
        "promised_delivery_minutes": promised_delivery_minutes.astype(int),
        "actual_delivery_minutes": actual_delivery_minutes,
    })
    return df.sort_values(["dispatch_datetime", "shipment_id"]).reset_index(drop=True)


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "data" / "deliveries.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    assert len(dataset) == N_SHIPMENTS
    assert dataset["shipment_id"].is_unique
    assert (dataset["actual_delivery_minutes"] > 0).all()
    assert (dataset["distance_km"] >= 0).all()
    assert (dataset["promised_delivery_minutes"] > 0).all()
    assert not dataset.isna().any().any()
    dataset.to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Generated {len(dataset):,} synthetic shipments → {destination}")
    print(f"Columns: {dataset.shape[1]}; observed late rate: "
          f"{(dataset['actual_delivery_minutes'] > dataset['promised_delivery_minutes']).mean():.2%}")
    print("SYNTHETIC DATA ONLY — these are not real business outcomes.")


if __name__ == "__main__":
    main()
