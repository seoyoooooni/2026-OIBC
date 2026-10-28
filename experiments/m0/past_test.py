"""Test the first prediction model on January and February 2025.

Train only through 2025-01-01 00:00 KST, then predict the 253 sites for
2025-01-01 01:00 through 2025-03-01 00:00 KST (the F5 development fold).
The held-out actual generation is used only after predictions are saved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "01_participant_dataset"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "m0_f5"
TZ = "Asia/Seoul"
TRAIN_FROM = pd.Timestamp("2023-06-01 01:00", tz=TZ)
TRAIN_CUTOFF = pd.Timestamp("2025-01-01 00:00", tz=TZ)
VALID_START = pd.Timestamp("2025-01-01 01:00", tz=TZ)
VALID_END = pd.Timestamp("2025-03-01 00:00", tz=TZ)
ROUNDS = (10, 17)
WA_COLS = [
    "wa_cloud", "wa_temp", "wa_humidity", "wa_rain",
    "wa_wind_speed", "wa_ground_press",
]
WB_COLS = [
    "wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx",
    "wb_precip_prob", "wb_total_liq", "wb_vis",
    "wb_wind_speed", "wb_dew_point",
]
FEATURE_COLS = WA_COLS + WB_COLS + [
    "elev", "azim", "cosz", "doy_sin", "doy_cos",
    "hour", "cap", "lat", "lon",
]


def kst_times(values: pd.Series, label: str) -> pd.Series:
    """Reject timezone-free source timestamps instead of guessing their zone."""
    result = pd.to_datetime(values, errors="raise")
    if result.dt.tz is None:
        raise ValueError(f"{label}: ts has no timezone")
    return result.dt.tz_convert(TZ)


def require_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    count = int(frame.duplicated(columns).sum())
    if count:
        raise ValueError(f"{label}: {count:,} duplicate keys for {columns}")


def solar(ts: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray):
    """The same NOAA approximation used in make_submission.py."""
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    )
    tst = (hour - 0.5) * 60 + eqtime + 4 * (lon - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(
        np.arctan2(
            -np.sin(ha),
            np.tan(decl) * np.cos(la) - np.sin(la) * np.cos(ha),
        )
    ) % 360
    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")


def build_features(
    base: pd.DataFrame,
    sites: pd.DataFrame,
    forecast_a: pd.DataFrame,
    forecast_b: pd.DataFrame,
    round_number: int,
) -> pd.DataFrame:
    rows = base.merge(sites, on="site_id", how="left", validate="many_to_one")
    if rows["capacity_kw"].isna().any():
        raise ValueError("Some generation rows have no matching site metadata")
    a = forecast_a.loc[forecast_a["round"] == round_number].drop(columns="round")
    b = forecast_b.loc[forecast_b["round"] == round_number].drop(columns="round")
    rows = rows.merge(a, on=["wa_station", "ts"], how="left", validate="many_to_one")
    rows = rows.merge(b, on=["wb_station", "ts"], how="left", validate="many_to_one")
    ts = pd.DatetimeIndex(rows["ts"])
    rows["elev"], rows["azim"], rows["cosz"] = solar(
        ts, rows["lat"].to_numpy(), rows["lon"].to_numpy()
    )
    doy = ts.dayofyear.to_numpy()
    rows["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    rows["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    rows["hour"] = ts.hour.to_numpy().astype("int8")
    rows["cap"] = rows["capacity_kw"].astype("float32")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the first model on past data (F5)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(data_dir / "sites.csv")[
        ["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]
    ]
    require_unique(sites, ["site_id"], "sites")
    if len(sites) != 253:
        raise ValueError(f"Expected 253 sites; got {len(sites)}")
    if (sites["capacity_kw"] <= 0).any():
        raise ValueError("Every site needs a positive capacity")

    smp = pd.read_csv(data_dir / "train" / "smp.csv", usecols=["ts"])
    smp["ts"] = kst_times(smp["ts"], "train/smp.csv")
    require_unique(smp, ["ts"], "train/smp.csv")
    hours = pd.DatetimeIndex(
        smp.loc[smp["ts"].between(VALID_START, VALID_END), "ts"].sort_values()
    )
    expected_hours = pd.date_range(VALID_START, VALID_END, freq="h")
    if not hours.equals(expected_hours):
        raise ValueError("F5 hours are missing, duplicated, or out of order")
    print(f"F5: {len(hours):,} hours; {len(sites)} sites", flush=True)

    generation = pd.read_parquet(data_dir / "train" / "generation.parquet")
    generation["ts"] = kst_times(generation["ts"], "generation")
    require_unique(generation, ["site_id", "ts"], "generation")
    train = generation.loc[
        generation["ts"].between(TRAIN_FROM, TRAIN_CUTOFF),
        ["site_id", "ts", "energy"],
    ].copy()
    actual = generation.loc[
        generation["ts"].between(VALID_START, VALID_END),
        ["site_id", "ts", "energy"],
    ].copy()
    del generation
    if train.empty:
        raise ValueError("No training labels before the cutoff")
    if train["ts"].max() > TRAIN_CUTOFF:
        raise AssertionError("Validation labels entered training data")
    print(f"Training labels: {len(train):,} rows, ending {train.ts.max()}", flush=True)

    a = pd.read_parquet(
        data_dir / "train" / "forecast_wa.parquet",
        columns=["wa_station", "round", "ts"] + WA_COLS,
    )
    b = pd.read_parquet(
        data_dir / "train" / "forecast_wb.parquet",
        columns=["wb_station", "round", "ts"] + WB_COLS,
    )
    for frame, name, key in ((a, "A forecast", "wa_station"), (b, "B forecast", "wb_station")):
        frame["ts"] = kst_times(frame["ts"], name)
        require_unique(frame, [key, "round", "ts"], name)
        if not frame["round"].isin(ROUNDS).all():
            raise ValueError(f"{name}: unexpected issue round")
    a = a.loc[a["ts"].between(TRAIN_FROM, VALID_END)]
    b = b.loc[b["ts"].between(TRAIN_FROM, VALID_END)]

    valid_base = pd.MultiIndex.from_product(
        [sites["site_id"], hours], names=["site_id", "ts"]
    ).to_frame(index=False)
    require_unique(valid_base, ["site_id", "ts"], "F5 prediction grid")
    prediction_sum = np.zeros(len(valid_base), dtype=np.float64)
    model_paths: dict[str, str] = {}

    for round_number in ROUNDS:
        print(f"Training round {round_number}...", flush=True)
        train_rows = build_features(train, sites, a, b, round_number)
        daylight = train_rows["elev"] > 0
        train_rows = train_rows.loc[daylight]
        y = (train_rows["energy"] / train_rows["cap"]).clip(0, 1.2).astype("float32")
        x = train_rows[FEATURE_COLS].astype("float32")
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.06,
            num_leaves=127,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=args.n_jobs,
        )
        model.fit(x, y)
        del train_rows, daylight, x, y
        gc.collect()

        valid_rows = build_features(valid_base, sites, a, b, round_number)
        x_valid = valid_rows[FEATURE_COLS].astype("float32")
        utilization = np.where(
            valid_rows["elev"].to_numpy() > 0,
            model.predict(x_valid),
            0.0,
        )
        prediction_sum += np.clip(utilization, 0, None) * valid_rows["cap"].to_numpy()
        model_path = output_dir / f"round_{round_number}.txt"
        model.booster_.save_model(str(model_path))
        model_paths[str(round_number)] = str(model_path)
        del valid_rows, x_valid, utilization, model
        gc.collect()

    predictions = valid_base.copy()
    predictions["pred_kwh"] = prediction_sum / len(ROUNDS)
    require_unique(predictions, ["site_id", "ts"], "F5 predictions")
    if len(predictions) != len(sites) * len(hours):
        raise AssertionError("F5 prediction grid is incomplete")
    if not np.isfinite(predictions["pred_kwh"]).all():
        raise ValueError("Predictions contain NaN or infinity")
    if (predictions["pred_kwh"] < 0).any():
        raise ValueError("Predictions contain negative kWh")

    output_path = output_dir / "site_predictions.parquet"
    predictions.to_parquet(output_path, index=False, compression="zstd")
    # Held-out generation is consulted only now, after both models predicted.
    comparison = predictions.merge(actual, on=["site_id", "ts"], how="left", validate="one_to_one")
    missing_actual = int(comparison["energy"].isna().sum())
    matched = comparison.loc[comparison["energy"].notna()]
    report = {
        "fold": "F5",
        "train_from": str(TRAIN_FROM),
        "train_cutoff": str(TRAIN_CUTOFF),
        "validation_start": str(VALID_START),
        "validation_end": str(VALID_END),
        "site_count": len(sites),
        "hour_count": len(hours),
        "prediction_rows": len(predictions),
        "training_label_rows": len(train),
        "missing_validation_actual_rows": missing_actual,
        "site_level_mae_kwh": float(np.abs(matched["energy"] - matched["pred_kwh"]).mean()),
        "site_predictions_sha256": sha256(output_path),
        "model_paths": model_paths,
        "note": "Site-level MAE is diagnostic, not the official G1/G2 score.",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(predictions):,} predictions to {output_path}", flush=True)
    print(f"Site-level MAE: {report['site_level_mae_kwh']:.3f} kWh", flush=True)
    print(f"Missing actual rows: {missing_actual:,}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
