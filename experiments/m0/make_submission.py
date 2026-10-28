"""첫 예측 모델로 평가 기간 제출 파일을 만드는 예시.

참조 구성(sites.csv의 253개소 전부 + current_group)을 그대로 쓰고, 하루전 기상예보로
발전소별 설비이용률을 LightGBM으로 학습해 예측한 뒤 집합 단위로 합산한다.
라운드 10·17 예보로 각각 예측하고 그 **평균**을 시각당 하나의 입찰로 제출한다.
학습 구간 데이터만 사용하므로 참가자가 그대로 재현할 수 있다.

    python experiments/m0/make_submission.py

--sites-out 를 주면 발전소별 예측도 저장한다. 구성을 직접 최적화하려면
집합 합계가 아니라 발전소별 예측이 필요하다 (OIBC2026-문제정의서.md 7.2절).
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", "data", "01_participant_dataset"))
TRAIN_FROM = pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul")   # 최근 2년만 사용
ROUNDS = (10, 17)
WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain",
           "wa_wind_speed", "wa_ground_press"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob",
           "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]

t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def solar(ts: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray):
    """태양 고도·방위 (NOAA 근사, 도 단위). 외부 의존성 없이 계산."""
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                       - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))     # 분
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)
            - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)
            - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))                 # rad
    # KST(UTC+9) 기준 진태양시. ts는 우측닫힘이므로 구간 중앙(-30분)을 쓴다.
    tst = (hour - 0.5) * 60 + eqtime + 4 * (lon - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cosz = np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha),
                               np.tan(decl)*np.cos(la) - np.sin(la)*np.cos(ha))) % 360
    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")


def features(gen: pd.DataFrame | None, sites: pd.DataFrame, fa: pd.DataFrame,
             fb: pd.DataFrame, index: pd.DatetimeIndex | None, r: int) -> pd.DataFrame:
    """(site, ts) 격자에 예보·태양기하를 붙인 특징 행렬."""
    if gen is not None:
        base = gen[["site_id", "ts", "energy"]]
    else:
        base = pd.MultiIndex.from_product([sites.site_id, index],
                                          names=["site_id", "ts"]).to_frame(index=False)
    d = base.merge(sites, on="site_id", how="left")
    d = d.merge(fa[fa["round"] == r].drop(columns="round"), on=["wa_station", "ts"], how="left")
    d = d.merge(fb[fb["round"] == r].drop(columns="round"), on=["wb_station", "ts"], how="left")
    ts = pd.DatetimeIndex(d.ts)
    d["elev"], d["azim"], d["cosz"] = solar(ts, d.lat.to_numpy(), d.lon.to_numpy())
    doy = ts.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2*np.pi*doy/365.25).astype("float32")
    d["doy_cos"] = np.cos(2*np.pi*doy/365.25).astype("float32")
    d["hour"] = ts.hour.to_numpy().astype("int8")
    d["cap"] = d.capacity_kw.astype("float32")
    return d


FEATS = WA_COLS + WB_COLS + ["elev", "azim", "cosz", "doy_sin", "doy_cos",
                              "hour", "cap", "lat", "lon"]


def main() -> None:
    ap = argparse.ArgumentParser(description="첫 예측 모델로 평가 기간 제출 CSV 생성")
    ap.add_argument("--sites-out", default=None,
                    help="발전소별 예측을 저장할 parquet 경로 (구성 최적화용)")
    args = ap.parse_args()

    sites = pd.read_csv(f"{HERE}/sites.csv")[
        ["site_id", "current_group", "capacity_kw", "lat", "lon",
         "wa_station", "wb_station"]]
    scope = sites   # sites.csv 에 실린 253개소 전부가 구성·예측 대상
    log(f"발전소 {len(sites)}개소 (전부 구성·예측 대상)")

    gen = pd.read_parquet(f"{HERE}/train/generation.parquet")
    gen["ts"] = pd.to_datetime(gen.ts)
    gen = gen[gen.ts >= TRAIN_FROM]
    log(f"학습 발전량 {len(gen):,}행 ({gen.ts.min():%Y-%m-%d} ~ {gen.ts.max():%Y-%m-%d})")

    fa_tr = pd.read_parquet(f"{HERE}/train/forecast_wa.parquet",
                            columns=["wa_station", "round", "ts"] + WA_COLS)
    fb_tr = pd.read_parquet(f"{HERE}/train/forecast_wb.parquet",
                            columns=["wb_station", "round", "ts"] + WB_COLS)
    for d in (fa_tr, fb_tr):
        d["ts"] = pd.to_datetime(d.ts)
    fa_tr = fa_tr[fa_tr.ts >= TRAIN_FROM]; fb_tr = fb_tr[fb_tr.ts >= TRAIN_FROM]
    log(f"학습 예보 A {len(fa_tr):,} / B {len(fb_tr):,}")

    models = {}
    for r in ROUNDS:
        d = features(gen, sites, fa_tr, fb_tr, None, r)
        d = d[d.elev > 0]                                  # 야간은 0으로 고정
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
        X = d[FEATS].astype("float32")
        log(f"round={r} 학습 행렬 {X.shape}")
        models[r] = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.06, num_leaves=127,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbose=-1, n_jobs=-1).fit(X, y)
        log(f"round={r} 학습 완료  train MAE {np.abs(models[r].predict(X)-y).mean():.4f}")
        del d, X, y

    del gen, fa_tr, fb_tr
    index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(f"{HERE}/smp.csv").ts)
                             .dt.tz_convert("Asia/Seoul"))          # 채점 시간축
    log(f"채점 시각 {len(index):,}h ({index[0]} ~ {index[-1]})")

    fa_te = pd.read_parquet(f"{HERE}/test/forecast_wa.parquet",
                            columns=["wa_station", "round", "ts"] + WA_COLS)
    fb_te = pd.read_parquet(f"{HERE}/test/forecast_wb.parquet",
                            columns=["wb_station", "round", "ts"] + WB_COLS)
    for d in (fa_te, fb_te):
        d["ts"] = pd.to_datetime(d.ts)

    grp = scope.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    out, per_site = [], []
    for r in ROUNDS:
        d = features(None, scope, fa_te, fb_te, index, r)
        X = d[FEATS].astype("float32")
        p = np.where(d.elev.to_numpy() > 0, models[r].predict(X), 0.0)
        d["kwh"] = np.clip(p, 0, None) * d.cap.to_numpy()
        d["group"] = d.site_id.map(grp)
        out.append(d.groupby(["group", "ts"], as_index=False).kwh.sum())
        if args.sites_out:
            per_site.append(d[["site_id", "ts", "kwh"]])
        log(f"round={r} 예측 완료  집합 합계 {d.kwh.sum()/1e6:,.2f} GWh")

    # 라운드 10·17 예측의 평균 = 시각당 단일 입찰
    f = (pd.concat(out).groupby(["group", "ts"], as_index=False).kwh.mean()
           .rename(columns={"kwh": "forecast_kwh"})
           .sort_values(["group", "ts"]))
    f.to_csv(f"{HERE}/forecast.csv", index=False)
    grp.rename("group").reset_index().to_csv(f"{HERE}/composition.csv", index=False)
    log(f"저장: composition.csv ({len(grp)}행) / forecast.csv ({len(f):,}행)")
    if args.sites_out:
        ps = pd.concat(per_site).groupby(["site_id", "ts"], as_index=False).kwh.mean()
        ps.to_parquet(args.sites_out, index=False, compression="zstd")
        log(f"저장: {args.sites_out} ({len(ps):,}행)")


if __name__ == "__main__":
    main()
