"""고령 유입 레코드 → 다수준 모형용 코호트 테이블 + 기술통계.

코호트 셀 = 출발 시군구 × 도착 자치구 × 목적(H/W/E) × 5세 연령 × 성별 × 주중/주말 × 낮/밤
- 관측되지 않은 셀은 0으로 채운다 (이전 연구는 양수 셀만 사용 → 선택편의).
- 노출량(offset) = 해당 주중/주말 일수 × 낮/밤 시간 수.
  이전 연구에서 주중 5일/주말 2일을 사후에 손으로 보정하던 것을 모형 안에서 처리한다.
- 지역변수는 출발지(o_)와 도착지(d_) 값을 **따로** 붙인다.
  (도착-출발 차이 변수는 출발지·도착지 효과와 완전 공선성을 만든다.)

사용법:
    python -m src.cohort [--in data/processed/elderly_inflow.parquet]
"""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

CELL_KEYS = ["orig_sgg", "dest_gu", "purpose", "age5", "sex", "daytype", "daynight"]


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


def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(CELL_KEYS, as_index=False).agg(flow=("flow", "sum"), n_masked=("masked", "sum"))
    levels = [sorted(df[k].unique()) for k in CELL_KEYS]
    grid = pd.MultiIndex.from_product(levels, names=CELL_KEYS).to_frame(index=False)
    grid = grid.merge(agg, how="left", on=CELL_KEYS).fillna({"flow": 0, "n_masked": 0})
    grid["y"] = np.round(grid["flow"]).astype(int)

    days = day_counts(df["ym"].unique())
    hours = {"day": len(C.DAY_HOURS), "night": 24 - len(C.DAY_HOURS)}
    grid["exposure"] = grid["daytype"].map(days) * grid["daynight"].map(hours)
    grid["od"] = grid["orig_sgg"] + "_" + grid["dest_gu"]
    return grid


def vif(X: pd.DataFrame) -> pd.Series:
    Z = (X - X.mean()) / X.std(ddof=0)
    out = {}
    for c in Z:
        others = Z.drop(columns=c)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), others]), Z[c], rcond=None)
        resid = Z[c] - np.column_stack([np.ones(len(Z)), others]) @ beta
        r2 = 1 - resid.var() / Z[c].var()
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out, name="VIF")


def attach_regional(grid: pd.DataFrame, path: Path, out: Path) -> pd.DataFrame:
    """regional_vars.csv: sgg_code, [area_km2], 변수들… (출발·도착 시군구 모두 포함)."""
    reg = pd.read_csv(path, dtype={"sgg_code": str})
    reg["sgg_code"] = reg["sgg_code"].str[:5]
    vars_ = [c for c in reg.select_dtypes("number").columns if c != "area_km2"]
    if "area_km2" in reg:
        reg[vars_] = reg[vars_].div(reg["area_km2"], axis=0)  # 1km² 당 밀도
    reg = reg.set_index("sgg_code")[vars_]

    for side, key in [("d", "dest_gu"), ("o", "orig_sgg")]:
        units = grid[key].unique()
        sub = reg.reindex(units)
        if sub.isna().any().any():
            miss = sub.index[sub.isna().any(axis=1)].tolist()
            raise ValueError(f"지역변수에 없는 {key} 코드: {miss[:10]}")
        # 수준 단위(도착 25개 구 / 출발 시군구) 분포 기준 표준화
        z = (sub - sub.mean()) / sub.std(ddof=0)
        v = vif(sub)
        v.to_csv(out / f"vif_{side}.csv", encoding="utf-8-sig")
        sub.corr().round(3).to_csv(out / f"corr_{side}.csv", encoding="utf-8-sig")
        print(f"[cohort] {key} 수준 VIF (단위 {len(units)}개):\n{v.round(2).to_string()}")
        grid = grid.merge(z.add_prefix(f"{side}_"), left_on=key, right_index=True)
    return grid


def describe(df: pd.DataFrame, grid: pd.DataFrame, out: Path) -> None:
    tot = grid["y"].sum()
    rows = []
    for key in ["purpose", "age5", "sex", "daytype", "daynight"]:
        for lvl, g in grid.groupby(key):
            # 노출량 보정 비율: 셀당 시간당 유입 (주중/주말 일수 차이 보정)
            rows.append({"변수": key, "범주": lvl, "이동량": g["y"].sum(), "비중": g["y"].sum() / tot,
                         "시간당_유입률": g["y"].sum() / g["exposure"].sum()})
    pd.DataFrame(rows).to_csv(out / "desc_margins.csv", index=False, encoding="utf-8-sig")

    for key in ["age5", "sex", "daytype", "daynight"]:
        t = grid.pivot_table(index="purpose", columns=key, values="y", aggfunc="sum")
        t.div(t.sum(axis=1), axis=0).to_csv(out / f"desc_purpose_{key}.csv", encoding="utf-8-sig")
    for key in ["dest_gu", "orig_sgg"]:
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
    ap.add_argument("--regional", type=Path, default=C.REGIONAL_DIR / "regional_vars.csv")
    ap.add_argument("--cohort", type=Path, default=C.PROC_DIR / "cohort.csv")
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.inp)
    grid = build_cohort(df)
    print(f"[cohort] 코호트 셀 {len(grid):,}개 (0 셀 {(grid['y'] == 0).mean():.1%}), "
          f"출발 {grid['orig_sgg'].nunique()} × 도착 {grid['dest_gu'].nunique()}")
    describe(df, grid, a.out)
    if a.regional.exists():
        grid = attach_regional(grid, a.regional, a.out)
    else:
        print(f"[cohort] 지역변수 파일 없음({a.regional}) → 개인·시간 변수만으로 진행")
    a.cohort.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(a.cohort, index=False)
    print(f"[cohort] → {a.cohort}")


if __name__ == "__main__":
    main()
