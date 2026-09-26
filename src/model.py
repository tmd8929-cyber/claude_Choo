"""서울 유입 고령인구 생활이동의 목적별 다수준 분석.

구조 (교차분류 다수준):
    이동량 y[o,d,p]  (출발 시군구 o → 도착 행정동 d, 목적 p, 분석기간 합계)
      ├─ 도착 행정동 d  ⊂ 도착 자치구 g   (위계)
      └─ 출발 시군구 o                    (d, g와 교차)

모형: 목적별 포아송 GLMM (변분 베이즈, statsmodels PoissonBayesMixedGLM)
    log E[y] = β0 + u_o + v_g + w_d
분산성분으로 "어느 수준이 목적별 유입 편차를 설명하는가"를 비교한다.
통합모형은 목적을 고정효과로 넣어 목적 간 평균 차이를 추정한다.

사용법:
    python -m src.model [--in data/processed/elderly_inflow.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM

from . import config as C

LEVELS = {"orig_sgg": "출발 시군구", "dest_gu": "도착 자치구", "dest": "도착 행정동"}
VC_FORMULAS = {k: f"0 + C({k})" for k in LEVELS}
VB_OPTS = {"maxiter": 5000}


def describe(df: pd.DataFrame, out: Path) -> None:
    n_days = df["date"].nunique() if "date" in df and df["date"].str.len().max() >= 8 else 1
    tot = df["flow"].sum()

    by_purpose = (df.groupby("purpose_lbl")["flow"].sum().sort_values(ascending=False)
                  .to_frame("flow_total"))
    by_purpose["daily_mean"] = by_purpose["flow_total"] / n_days
    by_purpose["share"] = by_purpose["flow_total"] / tot
    by_purpose.to_csv(out / "desc_by_purpose.csv", encoding="utf-8-sig")

    for key, name in [(["purpose_lbl", "age_grp"], "desc_purpose_age.csv"),
                      (["purpose_lbl", "sex"], "desc_purpose_sex.csv"),
                      (["purpose_lbl", "daytype"], "desc_purpose_daytype.csv")]:
        if all(k in df for k in key):
            t = df.pivot_table(index=key[0], columns=key[1], values="flow", aggfunc="sum")
            (t.div(t.sum(axis=1), axis=0)).to_csv(out / name, encoding="utf-8-sig")

    for lvl in ["dest_gu", "orig_sgg"]:
        t = df.pivot_table(index=lvl, columns="purpose_lbl", values="flow", aggfunc="sum", fill_value=0)
        t["합계"] = t.sum(axis=1)
        t.sort_values("합계", ascending=False).to_csv(out / f"desc_{lvl}_by_purpose.csv",
                                                     encoding="utf-8-sig")
    print(f"[describe] 분석일수 {n_days}, 총 유입 {tot:,.0f}명·회")
    print(by_purpose.round(3).to_string())


def build_od_grid(df: pd.DataFrame) -> pd.DataFrame:
    """출발 시군구 × 도착 행정동 × 목적 격자를 만들고 관측 안 된 칸은 0으로 채운다."""
    agg = df.groupby(["orig_sgg", "dest", "purpose_lbl"], as_index=False)["flow"].sum()
    grid = pd.MultiIndex.from_product(
        [agg["orig_sgg"].unique(), agg["dest"].unique(), agg["purpose_lbl"].unique()],
        names=["orig_sgg", "dest", "purpose_lbl"]).to_frame(index=False)
    grid = grid.merge(agg, how="left").fillna({"flow": 0})
    grid["dest_gu"] = grid["dest"].str[:5]
    grid["y"] = np.round(grid["flow"]).astype(int)
    return grid


def variance_table(res, grid_mean: float) -> pd.DataFrame:
    sd = np.exp(res.vcp_mean)
    names = [LEVELS[n] for n in res.model.vcp_names]
    var = sd ** 2
    # Nakagawa & Schielzeth(2013)의 포아송 잠재척도 관측수준 분산 ln(1 + 1/λ)
    resid = np.log1p(1.0 / grid_mean)
    total = var.sum() + resid
    return pd.DataFrame({"수준": names, "SD": sd, "분산": var,
                         "분산비중_랜덤효과내": var / var.sum(),
                         "VPC_잠재척도": var / total})


def fit_by_purpose(grid: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows, blups = [], []
    for p, g in grid.groupby("purpose_lbl"):
        g = g.reset_index(drop=True)
        if g["y"].sum() == 0:
            continue
        m = PoissonBayesMixedGLM.from_formula("y ~ 1", VC_FORMULAS, g)
        r = m.fit_vb(minim_opts=VB_OPTS)
        vt = variance_table(r, max(g["y"].mean(), 1e-6))
        vt.insert(0, "목적", p)
        vt["절편"] = r.fe_mean[0]
        rows.append(vt)
        re = r.random_effects()
        re["목적"] = p
        blups.append(re)
        print(f"[model] {p}: n={len(g):,}, 총량={g['y'].sum():,}")
        print(vt.round(3).to_string(index=False))
    vc = pd.concat(rows, ignore_index=True)
    vc.to_csv(out / "vc_by_purpose.csv", index=False, encoding="utf-8-sig")
    pd.concat(blups).to_csv(out / "random_effects_by_purpose.csv", encoding="utf-8-sig")
    return vc


def fit_pooled(grid: pd.DataFrame, out: Path) -> pd.DataFrame:
    ref = grid.groupby("purpose_lbl")["y"].sum().idxmax()
    formula = f"y ~ C(purpose_lbl, Treatment('{ref}'))"
    r = PoissonBayesMixedGLM.from_formula(formula, VC_FORMULAS, grid).fit_vb(minim_opts=VB_OPTS)
    fe = pd.DataFrame({"term": r.model.exog_names, "coef": r.fe_mean, "sd": r.fe_sd})
    fe["IRR"] = np.exp(fe["coef"])
    fe.to_csv(out / "pooled_fixed_effects.csv", index=False, encoding="utf-8-sig")
    vt = variance_table(r, max(grid["y"].mean(), 1e-6))
    vt.to_csv(out / "pooled_vc.csv", index=False, encoding="utf-8-sig")
    print(f"[model] 통합모형 (기준 목적: {ref})")
    print(fe.round(3).to_string(index=False))
    print(vt.round(3).to_string(index=False))
    return fe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=C.PROC_DIR / "elderly_inflow.parquet")
    ap.add_argument("--out", type=Path, default=C.OUT_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.inp)
    describe(df, a.out)
    grid = build_od_grid(df)
    print(f"[model] OD 격자 {len(grid):,}칸 (0 비율 {(grid['y'] == 0).mean():.1%})")
    fit_by_purpose(grid, a.out)
    fit_pooled(grid, a.out)


if __name__ == "__main__":
    main()
