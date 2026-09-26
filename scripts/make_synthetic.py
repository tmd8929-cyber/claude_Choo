"""파이프라인 검증용 합성 데이터 (실제 행정동 경계 위에 생성).

1단계 poi   : 가짜 시설 좌표·격자인구 → data/synthetic_regional/ (regional.py 입력)
2단계 flows : regional.py 결과를 읽어 알려진 효과를 심은 생활이동 CSV → data/synthetic/
심은 구조
- 목적 E(기타)는 동의 병원 접근성(acc_hospital)에 +0.5, 목적 H는 무반응
- 목적 W(일)는 출발 시군구 편차가 크고, E는 도착 구·동 편차가 크다
- 주말·밤·연령·성별 효과

사용법:
    python scripts/make_synthetic.py poi
    python -m src.regional --poi-dir data/synthetic_regional/poi --grid-pop data/synthetic_regional/grid_pop.csv --out data/synthetic_regional
    python scripts/make_synthetic.py flows
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(__file__).resolve().parents[1]
reg_dir = root / "data" / "synthetic_regional"
flow_path = root / "data" / "synthetic" / "synthetic_sample.csv"
boundary = sorted((root / "data" / "boundary").glob("HangJeongDong_ver*.geojson"))[-1]


def sample_points(polys: gpd.GeoDataFrame, n: int) -> gpd.GeoSeries:
    """폴리곤 면적 비례 + 동별 밀집도 편차를 두고 점을 뿌린다."""
    w = polys.area * rng.gamma(1.0, 1.0, len(polys))
    counts = rng.multinomial(n, (w / w.sum()).to_numpy())
    sub = polys.geometry[counts > 0]
    return sub.sample_points(size=counts[counts > 0], rng=rng).explode(index_parts=False)


def stage_poi():
    b = gpd.read_file(boundary).to_crs("EPSG:5179")
    area = b[b["sido"].isin(["11", "41", "28"])]
    seoul = b[b["sido"] == "11"]
    (reg_dir / "poi").mkdir(parents=True, exist_ok=True)

    # 상가정보 모사: 업종명 텍스트 + 경도/위도
    kinds = {"종합병원": 300, "요양병원": 400, "내과의원": 6000, "치과의원": 4000, "약국": 5000,
             "전통시장": 300, "슈퍼마켓": 5000, "목욕탕": 800, "기원": 300, "동물병원": 1000, "커피전문점": 8000}
    rows = []
    for k, n in kinds.items():
        p = sample_points(area, n).to_crs("EPSG:4326")
        rows.append(pd.DataFrame({"상권업종소분류명": k, "경도": p.x, "위도": p.y}))
    pd.concat(rows).to_csv(reg_dir / "poi" / "store.csv", index=False, encoding="utf-8-sig")
    # 좌표만 있는 파일 → 파일명이 범주
    for name, n in {"subway": 400, "welfare": 900}.items():
        p = sample_points(area, n).to_crs("EPSG:4326")
        pd.DataFrame({"lon": p.x, "lat": p.y}).to_csv(reg_dir / "poi" / f"{name}.csv", index=False)

    # 100m 격자인구 (서울): 동마다 밀도가 다르게
    p = sample_points(seoul, 200_000)
    g = pd.DataFrame({"x": (p.x // 100) * 100 + 50, "y": (p.y // 100) * 100 + 50})
    g = g.value_counts().rename("n").reset_index()
    g["pop_total"] = g["n"] * rng.integers(20, 60, len(g))
    g["pop_65p"] = rng.binomial(g["pop_total"], rng.beta(4, 16, len(g)))
    g[["x", "y", "pop_total", "pop_65p"]].to_csv(reg_dir / "grid_pop.csv", index=False)
    print(f"poi/grid → {reg_dir}")


def stage_flows():
    b = gpd.read_file(boundary, ignore_geometry=True)
    dongs = b[b["sido"] == "11"]["adm_cd2"].str[:8].tolist()          # 서울 행정동 (행안부 8자리)
    origins = b[b["sido"].isin(["41", "28"])]["adm_cd2"].str[:8].tolist()
    others = b[b["sido"].isin(["26", "30"])]["adm_cd2"].str[:8].tolist()[:20]  # 필터링 대상
    reg = pd.read_csv(reg_dir / "regional_dong.csv", dtype={"code": str, "adm_cd2": str})
    reg = reg.drop_duplicates("adm_cd2").assign(code=lambda d: d["adm_cd2"].str[:8]).set_index("code")
    zh = (reg["acc_hospital"] - reg["acc_hospital"].mean()) / reg["acc_hospital"].std(ddof=0)

    gus = sorted({d[:5] for d in dongs})
    sggs = sorted({o[:5] for o in origins})
    sd = {"H": (0.5, 0.3, 0.3), "W": (1.0, 0.3, 0.3), "E": (0.4, 0.6, 0.5)}  # (출발시군구, 구, 동)
    u_o = {p: dict(zip(sggs, rng.normal(0, s[0], len(sggs)))) for p, s in sd.items()}
    u_g = {p: dict(zip(gus, rng.normal(0, s[1], len(gus)))) for p, s in sd.items()}
    u_d = {p: dict(zip(dongs, rng.normal(0, s[2], len(dongs)))) for p, s in sd.items()}
    b_p = {"H": 0.0, "W": -0.3, "E": 0.4}
    beta_hosp = {"H": 0.0, "W": 0.0, "E": 0.5}

    types = ["HH", "HW", "HE", "WH", "WW", "WE", "EH", "EW", "EE"]
    ages = [0, 20, 40, 60, 65, 70, 75, 80, 85]
    dates = pd.date_range("2026-08-01", "2026-08-31")
    n = 1_500_000
    o = rng.choice(origins + others + dongs[:30], n)
    d = rng.choice(dongs + origins[:30], n)
    t = rng.choice(types, n)
    a = rng.choice(ages, n)
    day = pd.DatetimeIndex(rng.choice(dates, n))
    hr = rng.integers(0, 24, n)
    sx = rng.choice(["M", "F"], n)
    p = np.array([x[-1] for x in t])
    eta = np.array([b_p[pp] + u_o[pp].get(oo[:5], 0) + u_g[pp].get(dd[:5], 0) + u_d[pp].get(dd, 0)
                    + beta_hosp[pp] * zh.get(dd, 0) for pp, oo, dd in zip(p, o, d)])
    wk = day.dayofweek >= 5
    night = ~((hr >= 7) & (hr <= 18))
    eta += np.where(p == "W", -0.9, 0.3) * wk + np.where(p == "W", -0.5, 0.2) * night
    eta += np.select([a >= 80, a >= 75, a >= 70], [-0.8, -0.5, -0.2], 0) + 0.1 * (sx == "M")
    mu = np.exp(-0.5 + eta)
    flow = rng.negative_binomial(2, 2 / (2 + mu))
    keep = flow > 0
    df = pd.DataFrame({
        "대상연월": day.strftime("%Y%m%d"), "요일": day.dayofweek.map(dict(enumerate("월화수목금토일"))),
        "도착시간": hr, "출발 행정동 코드": o, "도착 행정동 코드": d, "성별": sx, "나이": a, "이동유형": t,
        "평균 이동 시간(분)": rng.integers(20, 150, n), "이동인구(합)": np.where(flow < 3, "*", flow.astype(str)),
    })[keep]
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flow_path, index=False, encoding="cp949")
    print(f"flows → {flow_path} ({len(df):,} rows)")


if __name__ == "__main__":
    {"poi": stage_poi, "flows": stage_flows}[sys.argv[1]]()
