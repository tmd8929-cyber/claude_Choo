"""파이프라인 검증용 합성 생활이동 CSV 생성 (서울생활이동 시군구 단위 형식 모사, CP949).

알고 있는 구조를 심어서 모형이 복원하는지 확인한다.
- 목적 E(기타)는 도착 자치구 편차가 크고 병원 밀도(d_hosp)에 반응
- 목적 W(일)는 출발 시군구 편차가 크다
- 주말·밤, 연령 효과 포함
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(__file__).resolve().parents[1]
out = root / "data" / "synthetic" / "synthetic_sample.csv"
reg_out = root / "data" / "regional" / "synthetic_regional_vars.csv"
reg_out.parent.mkdir(parents=True, exist_ok=True)
out.parent.mkdir(parents=True, exist_ok=True)

seoul = [f"11{i:03d}" for i in range(110, 750, 26)][:25]              # 행안부 자치구 25
origins = [f"41{i:03d}" for i in range(110, 830, 23)][:31] + [f"28{i:03d}" for i in range(110, 720, 61)][:10]
others = [f"11{i:03d}" for i in (110, 140)] + ["26110", "30110"]        # 필터링될 출발지
types = ["HH", "HW", "HE", "WH", "WW", "WE", "EH", "EW", "EE"]
ages = [0, 20, 40, 60, 65, 70, 75, 80, 85]
dates = pd.date_range("2026-08-01", "2026-08-31")

reg = pd.DataFrame({"sgg_code": seoul + origins, "area_km2": rng.uniform(10, 60, len(seoul) + len(origins))})
for v in ["hosp", "market", "welfare", "subway"]:
    reg[v] = rng.gamma(2, 5, len(reg)) * reg["area_km2"] / 20
reg.to_csv(reg_out, index=False)
dens = reg.set_index("sgg_code")
z_hosp = ((dens["hosp"] / dens["area_km2"]) - (dens["hosp"] / dens["area_km2"]).loc[seoul].mean()) \
    / (dens["hosp"] / dens["area_km2"]).loc[seoul].std(ddof=0)

sd = {"H": (0.6, 0.4), "W": (1.0, 0.3), "E": (0.4, 0.9)}               # (출발 SD, 도착 SD)
u_o = {p: dict(zip(origins, rng.normal(0, s[0], len(origins)))) for p, s in sd.items()}
u_d = {p: dict(zip(seoul, rng.normal(0, s[1], len(seoul)))) for p, s in sd.items()}
b_p = {"H": 0.0, "W": -0.3, "E": 0.4}

n = 800_000
o = rng.choice(origins + others, n)
dd = rng.choice(seoul + origins[:3], n)
t = rng.choice(types, n)
a = rng.choice(ages, n)
day = rng.choice(dates, n)
hr = rng.integers(0, 24, n)
sx = rng.choice(["M", "F"], n)
p = np.array([x[-1] for x in t])
wk = pd.DatetimeIndex(day).dayofweek >= 5
night = ~((hr >= 7) & (hr <= 18))
eta = np.array([b_p[pp] + u_o[pp].get(oo, 0) + u_d[pp].get(d_, 0) + (0.5 * z_hosp.get(d_, 0) if pp == "E" else 0)
                for pp, oo, d_ in zip(p, o, dd)])
eta += np.where(p == "W", -0.9, 0.3) * wk + np.where(p == "W", -0.5, 0.2) * night
eta += np.select([a >= 80, a >= 75, a >= 70], [-0.8, -0.5, -0.2], 0) + 0.1 * (sx == "M")
mu = np.exp(0.8 + eta)
flow = rng.negative_binomial(2, 2 / (2 + mu))
keep = flow > 0
flow_s = np.where(flow < 3, "*", flow.astype(str))
df = pd.DataFrame({
    "대상연월": day.strftime("%Y%m%d") if hasattr(day, "strftime") else pd.DatetimeIndex(day).strftime("%Y%m%d"),
    "요일": pd.DatetimeIndex(day).dayofweek.map(dict(enumerate("월화수목금토일"))),
    "도착시간": hr, "출발 시군구 코드": o, "도착 시군구 코드": dd, "성별": sx, "나이": a,
    "이동유형": t, "평균 이동 시간(분)": rng.integers(10, 150, n), "이동인구(합)": flow_s,
})[keep]
df.to_csv(out, index=False, encoding="cp949")
print(f"wrote {out} ({len(df):,} rows), {reg_out}")
