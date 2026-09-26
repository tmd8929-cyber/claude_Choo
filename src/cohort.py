"""고령 유입 레코드 → 다수준 모형용 코호트 테이블 + 기술통계.

두 가지 설계를 지원한다.
  --dest-level gu   : 셀 = 출발 시군구 × 도착 자치구 × 코호트  (교차분류, R/mlm_glmmtmb.R)
  --dest-level dong : 셀 = 도착 행정동(⊂자치구) × 코호트, 출발지 합산 (위계, R/mlm_dong.R)
                      --with-origin 이면 출발 시군구를 유지하되 관측된 출발–도착동 쌍만 격자화
코호트 = 목적(H/W/E) × 5세 연령 × 성별 × 주중/주말 × 시간(낮밤 또는 세분 시간대)

- 관측되지 않은 셀은 0으로 채운다 (이전 연구는 양수 셀만 사용 → 선택편의).
- 노출량(offset) = 주중/주말 일수 × 시간대 길이(시간).
- 지역변수는 도착 동(d_), 도착 구(g_), 출발 시군구(o_)에 **따로** 붙이고 각 수준 단위 기준으로 표준화.
  (도착-출발 차이 변수는 출발지·도착지 효과와 완전 공선성을 만든다.)
- 출발–도착 평균 이동시간(od_travel)을 셀 공변량으로 붙인다 (거리 대용).

사용법:
    python -m src.cohort --dest-level dong --time-var timeband
"""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

COHORT = ["purpose", "age5", "sex", "daytype"]


def day_counts(yms) -> dict[str, int]:
    wd = wk = 0
    for ym in yms:
        y, m = int(ym[:4]), int(ym[4:6])
        for d in range(1, calendar.monthrange(y, m)[1] + 1):
            if calendar.weekday(y, m, d) >= 5:
                wk += 1
            else:
                wd += 1
    return {"wd": wd, "wk": wk}


def time_hours(time_var: str) -> dict[str, int]:
    if time_var == "timeband":
        return C.TIMEBAND_HOURS
    return {"day": len(C.DAY_HOURS), "night": 24 - len(C.DAY_HOURS)}


def build_cohort(df: pd.DataFrame, dest_level: str, time_var: str, with_origin: bool) -> pd.DataFrame:
    dest_key = "dest_gu" if dest_level == "gu" else "dest"
    place = ["orig_sgg", dest_key] if (dest_level == "gu" or with_origin) else [dest_key]
    keys = place + COHORT + [time_var]

    agg = df.groupby(keys, as_index=False).agg(flow=("flow", "sum"), n_masked=("masked", "sum"))
    cohort_levels = pd.MultiIndex.from_product([sorted(df[k].unique()) for k in COHORT + [time_var]],
                                               names=COHORT + [time_var]).to_frame(index=False)
    if dest_level == "dong" and with_origin:
        # 출발 시군구 × 도착 동 전체 격자는 너무 크고 대부분 구조적 0 → 관측된 쌍만 사용
        places = df[place].drop_duplicates()
    else:
        places = pd.MultiIndex.from_product([sorted(df[k].unique()) for k in place],
                                            names=place).to_frame(index=False)
    grid = places.merge(cohort_levels, how="cross")
    grid = grid.merge(agg, how="left", on=keys).fillna({"flow": 0, "n_masked": 0})
    grid["y"] = np.round(grid["flow"]).astype(int)

    days = day_counts(df["ym"].unique())
    grid["exposure"] = grid["daytype"].map(days) * grid[time_var].map(time_hours(time_var))
    if dest_level == "dong":
        grid["dest_gu"] = grid["dest"].str[:5]
    if "orig_sgg" in grid:
        grid["od"] = grid["orig_sgg"] + "_" + grid[dest_key]

    if "travel_min" in df and df["travel_min"].notna().any():
        # 이동량 가중 평균 이동시간 (출발지를 합산한 동 단위에서는 도착지로 오는 평균 이동시간)
        tkey = place
        w = df.dropna(subset=["travel_min"])
        tt = (w.assign(tw=w["travel_min"] * w["flow"]).groupby(tkey)[["tw", "flow"]].sum())
        tt = (tt["tw"] / tt["flow"]).rename("od_travel").reset_index()
        grid = grid.merge(tt, on=tkey, how="left")
        grid["od_travel"] = grid["od_travel"].fillna(grid["od_travel"].median())
        grid["z_od_travel"] = (grid["od_travel"] - grid["od_travel"].mean()) / grid["od_travel"].std()
    return grid.rename(columns={time_var: "time"})


def vif(X: pd.DataFrame) -> pd.Series:
    Z = (X - X.mean()) / X.std(ddof=0)
    Z = Z.loc[:, Z.std() > 0]
    out = {}
    for c in Z:
        A = np.column_stack([np.ones(len(Z)), Z.drop(columns=c)])
        beta, *_ = np.linalg.lstsq(A, Z[c], rcond=None)
        r2 = 1 - (Z[c] - A @ beta).var() / Z[c].var()
        out[c] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return pd.Series(out, name="VIF")


def attach_level(grid: pd.DataFrame, table: pd.DataFrame, key: str, prefix: str,
                 vars_: list[str] | None, out: Path) -> pd.DataFrame:
    """table: code + 변수들. 수준 단위(key의 고유값) 분포 기준으로 표준화해 prefix를 붙인다."""
    num = [c for c in table.select_dtypes("number").columns if not vars_ or c in vars_]
    tab = table.drop_duplicates("code").set_index("code")[num]
    units = pd.Index(grid[key].unique())
    sub = tab.reindex(units)
    miss = sub.index[sub.isna().all(axis=1)]
    if len(miss):
        raise ValueError(f"지역변수에 없는 {key} 코드 {len(miss)}개: {list(miss[:8])} "
                         f"(regional 표의 code 컬럼과 코드체계 확인)")
    sub = sub.loc[:, sub.std(ddof=0) > 0]
    z = (sub - sub.mean()) / sub.std(ddof=0)
    v = vif(sub)
    v.to_csv(out / f"vif_{prefix}.csv", encoding="utf-8-sig")
    sub.corr().round(3).to_csv(out / f"corr_{prefix}.csv", encoding="utf-8-sig")
    high = v[v > 5]
    print(f"[cohort] {prefix}_ 변수 {sub.shape[1]}개 (단위 {len(units)}개), VIF>5: "
          f"{high.round(1).to_dict() if len(high) else '없음'}")
    return grid.merge(z.add_prefix(f"{prefix}_"), left_on=key, right_index=True, how="left")


def describe(grid: pd.DataFrame, out: Path, dest_key: str) -> None:
    tot = grid["y"].sum()
    rows = []
    for key in ["purpose", "age5", "sex", "daytype", "time"]:
        for lvl, g in grid.groupby(key):
            # 노출량 보정 비율: 셀당 시간당 유입 (주중/주말 일수·시간대 길이 차이 보정)
            rows.append({"변수": key, "범주": lvl, "이동량": g["y"].sum(), "비중": g["y"].sum() / tot,
                         "시간당_유입률": g["y"].sum() / g["exposure"].sum()})
    pd.DataFrame(rows).to_csv(out / "desc_margins.csv", index=False, encoding="utf-8-sig")

    for key in ["age5", "sex", "daytype", "time"]:
        t = grid.pivot_table(index="purpose", columns=key, values="y", aggfunc="sum")
        t.div(t.sum(axis=1), axis=0).to_csv(out / f"desc_purpose_{key}.csv", encoding="utf-8-sig")
    for key in {dest_key, "dest_gu", "orig_sgg"} & set(grid.columns):
        t = grid.pivot_table(index=key, columns="purpose", values="y", aggfunc="sum", fill_value=0)
        t["합계"] = t.sum(axis=1)
        t.sort_values("합계", ascending=False).to_csv(out / f"desc_{key}_by_purpose.csv",
                                                     encoding="utf-8-sig")
    by_p = grid.groupby("purpose")["y"].sum()
    print(f"[cohort] 총 유입 {tot:,} / 목적별 비중: " +
          ", ".join(f"{k} {v / tot:.1%}" for k, v in by_p.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=C.PROC_DIR / "elderly_inflow.parquet")
    ap.add_argument("--dest-level", choices=["gu", "dong"], default="dong")
    ap.add_argument("--time-var", choices=["daynight", "timeband"], default="daynight")
    ap.add_argument("--with-origin", action="store_true")
    ap.add_argument("--regional-dir", type=Path, default=C.REGIONAL_DIR)
    ap.add_argument("--vars", default="", help="사용할 지역변수(콤마). 비우면 전부")
    ap.add_argument("--cohort", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cohort_path = a.cohort or C.PROC_DIR / f"cohort_{a.dest_level}.csv"

    df = pd.read_parquet(a.inp)
    if a.dest_level == "dong" and df["dest"].str.len().max() <= 5:
        raise ValueError("원자료 도착 코드가 시군구 단위입니다. 행정동 단위 자료를 쓰거나 --dest-level gu")
    grid = build_cohort(df, a.dest_level, a.time_var, a.with_origin)
    dest_key = "dest_gu" if a.dest_level == "gu" else "dest"
    print(f"[cohort] {a.dest_level} 설계, 셀 {len(grid):,}개 (0 셀 {(grid['y'] == 0).mean():.1%}), "
          f"도착 {grid[dest_key].nunique()}개" +
          (f", 출발 {grid['orig_sgg'].nunique()}개" if "orig_sgg" in grid else ""))
    describe(grid, a.out, dest_key)

    vars_ = [v for v in a.vars.split(",") if v] or None
    f_dong, f_sgg = a.regional_dir / "regional_dong.csv", a.regional_dir / "regional_sgg.csv"
    if f_sgg.exists():
        sgg = pd.read_csv(f_sgg, dtype={"code": str, "sgg": str})
        grid = attach_level(grid, sgg, "dest_gu", "g", vars_, a.out)
        if "orig_sgg" in grid:
            grid = attach_level(grid, sgg, "orig_sgg", "o", vars_, a.out)
    if a.dest_level == "dong" and f_dong.exists():
        dong = pd.read_csv(f_dong, dtype={"code": str, "adm_cd2": str, "sgg": str})
        grid = attach_level(grid, dong, "dest", "d", vars_, a.out)
    if not f_sgg.exists():
        print(f"[cohort] 지역변수 없음({a.regional_dir}) → 개인·시간 변수만으로 진행")

    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(cohort_path, index=False)
    print(f"[cohort] → {cohort_path}")


if __name__ == "__main__":
    main()
