"""100m 격자 기반 지역변수 생성 → 행정동·자치구·출발 시군구 단위 표.

입력
- data/boundary/HangJeongDong_ver*.geojson : 행정동 경계 (vuski/admdongkor, SGIS 원자료, CC BY 4.0)
- data/regional/poi/*.csv : 시설 좌표 (src/collect.py가 만든다). 파일마다 다음 중 하나
    · 경도/위도(WGS84) 또는 x/y(EPSG:5179) + 업종명 컬럼 → POI_RULES 키워드로 범주 분류
    · 좌표만 있으면 파일명(확장자 제외)을 범주로 사용 (예: subway.csv → subway)
    · 한 범주가 여러 파일에 있으면 C.POI_CATEGORY_SOURCE의 파일만 쓴다
- data/regional/grid_pop.csv (선택) : 100m 격자 인구. x,y(EPSG:5179 중심점) + pop_total, pop_65p

처리
1. 서울 전역을 100m 격자로 나누고 격자 중심점을 행정동에 배정한다.
2. 격자마다 반경 ACCESS_RADIUS_M 안의 범주별 시설 수(보행 접근성)와 최근접 지하철역 거리를 계산한다.
3. 동·구 단위로 (고령인구 가중) 평균 접근성과 1km²당 밀도를 집계한다.
4. 출발 시군구(경기·인천)는 1km²당 밀도로 집계한다.

출력 (data/regional/)
- regional_dong.csv : code(동 코드 여러 체계), d_* 변수
- regional_sgg.csv  : code(시군구 코드 여러 체계), 변수 (서울 구 → g_, 출발지 → o_ 로 cohort 단계에서 접두)
- poi_uncategorized.csv : 규칙으로 분류 못 한 업종명 (수작업/LLM 분류용)

사용법:
    python -m src.regional [--boundary ...] [--poi-dir data/regional/poi] [--grid-pop ...]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config as C

CRS = "EPSG:5179"
LON_COLS = ["경도", "lon", "longitude", "xpos", "x좌표wgs84", "lng"]
LAT_COLS = ["위도", "lat", "latitude", "ypos", "y좌표wgs84"]
X_COLS = ["x", "x5179", "tm_x"]
Y_COLS = ["y", "y5179", "tm_y"]
TEXT_COLS = ["상권업종소분류명", "상권업종중분류명", "종별코드명", "업종명", "category_text", "name"]


def _find(cols, cands):
    norm = {re.sub(r"[\s_()]", "", c).lower(): c for c in cols}
    for c in cands:
        if c.lower() in norm:
            return norm[c.lower()]
    return None


def load_boundary(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path).to_crs(CRS)
    g["sido"] = g["sido"].astype(str)
    g["area_km2"] = g.area / 1e6
    return g


def dong_codes(row) -> list[str]:
    """생활이동 자료가 쓸 수 있는 동 코드 변형: 행안부 10/8자리, 통계청 8자리."""
    return list({row["adm_cd2"], row["adm_cd2"][:8], row["adm_cd"]})


def sgg_codes(mois_sgg: str, kostat_sgg: str) -> list[str]:
    return list({mois_sgg, kostat_sgg})


def classify(text: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=text.index, dtype="object")
    t = text.fillna("").astype(str)
    excluded = t.str.contains("|".join(map(re.escape, C.POI_EXCLUDE)))
    for cat, kws in C.POI_RULES.items():
        hit = out.isna() & ~excluded & t.str.contains("|".join(map(re.escape, kws)))
        out[hit] = cat
    return out


def load_pois(poi_dir: Path, out_dir: Path) -> gpd.GeoDataFrame:
    frames, uncategorized = [], []
    for f in sorted(poi_dir.glob("*.csv")):
        df = None
        for enc in ("utf-8-sig", "cp949"):
            try:
                df = pd.read_csv(f, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        lon, lat = _find(df.columns, LON_COLS), _find(df.columns, LAT_COLS)
        x, y = _find(df.columns, X_COLS), _find(df.columns, Y_COLS)
        if lon and lat:
            geom = gpd.points_from_xy(pd.to_numeric(df[lon], errors="coerce"),
                                      pd.to_numeric(df[lat], errors="coerce"), crs="EPSG:4326")
            g = gpd.GeoDataFrame(df, geometry=geom).to_crs(CRS)
        elif x and y:
            geom = gpd.points_from_xy(pd.to_numeric(df[x], errors="coerce"),
                                      pd.to_numeric(df[y], errors="coerce"), crs=CRS)
            g = gpd.GeoDataFrame(df, geometry=geom)
        else:
            raise KeyError(f"{f.name}: 좌표 컬럼을 찾지 못함 ({list(df.columns)[:15]})")

        text_col = _find(df.columns, TEXT_COLS)
        if text_col:
            g["category"] = classify(g[text_col])
            unc = g.loc[g["category"].isna(), text_col].value_counts()
            uncategorized.append(unc.rename_axis("업종명").reset_index(name="n").assign(file=f.name))
        else:
            g["category"] = f.stem
        g = g.dropna(subset=["category"])
        g = g[~g.geometry.is_empty & g.geometry.x.notna()]
        frames.append(g[["category", "geometry"]].assign(source=f.stem))
        print(f"[regional] {f.name}: {len(g):,}개 → {g['category'].value_counts().to_dict()}")
    if uncategorized:
        pd.concat(uncategorized).to_csv(out_dir / "poi_uncategorized.csv", index=False, encoding="utf-8-sig")
    pois = pd.concat(frames, ignore_index=True)
    # 같은 범주를 여러 파일이 주면 지정 출처만 남긴다 (중복 집계 방지)
    stems = set(pois["source"])
    owner = pois["category"].map(C.POI_CATEGORY_SOURCE)
    drop = owner.isin(stems) & (owner != pois["source"])
    if drop.any():
        dropped = pois[drop].groupby(["source", "category"]).size().to_dict()
        print(f"[regional] 우선 출처가 있어 제외: {dropped}")
    return gpd.GeoDataFrame(pois[~drop].drop(columns="source"), crs=CRS)


def build_grid(dongs: gpd.GeoDataFrame, size: int) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = dongs.total_bounds
    xs = np.arange(np.floor(minx / size) * size, maxx, size) + size / 2
    ys = np.arange(np.floor(miny / size) * size, maxy, size) + size / 2
    X, Y = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame({"gx": X.ravel(), "gy": Y.ravel()},
                           geometry=gpd.points_from_xy(X.ravel(), Y.ravel()), crs=CRS)
    grid = gpd.sjoin(pts, dongs[["adm_cd2", "sgg", "geometry"]], predicate="within", how="inner")
    return grid.drop(columns="index_right").reset_index(drop=True)


def attach_grid_pop(grid: gpd.GeoDataFrame, path: Path | None, size: int) -> gpd.GeoDataFrame:
    if path and path.exists():
        pop = pd.read_csv(path)
        key = lambda x, y: (np.floor(x / size).astype(np.int64), np.floor(y / size).astype(np.int64))  # noqa: E731
        pop["kx"], pop["ky"] = key(pop["x"], pop["y"])
        grid["kx"], grid["ky"] = key(grid["gx"], grid["gy"])
        grid = grid.merge(pop[["kx", "ky", "pop_total", "pop_65p"]], on=["kx", "ky"], how="left")
        grid[["pop_total", "pop_65p"]] = grid[["pop_total", "pop_65p"]].fillna(0)
        grid["w"] = grid["pop_65p"]
        print(f"[regional] 격자인구 결합: 고령인구 {grid['pop_65p'].sum():,.0f}명")
    else:
        grid["w"] = 1.0  # 격자인구가 없으면 면적 가중(균등)
    return grid


def grid_access(grid: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, radius: float) -> pd.DataFrame:
    xy = np.column_stack([grid["gx"], grid["gy"]])
    acc = {}
    for cat, sub in pois.groupby("category"):
        tree = cKDTree(np.column_stack([sub.geometry.x, sub.geometry.y]))
        acc[f"acc_{cat}"] = tree.query_ball_point(xy, r=radius, return_length=True)
        if cat in C.NEAREST_DIST_CATEGORIES:
            acc[f"dist_{cat}"] = tree.query(xy, k=1)[0]
    return pd.DataFrame(acc, index=grid.index)


def weighted_mean(df: pd.DataFrame, cols, by: str, w: str) -> pd.DataFrame:
    tmp = df[cols].mul(df[w], axis=0)
    tmp[by] = df[by]
    s = tmp.groupby(by).sum()
    wsum = df.groupby(by)[w].sum()
    fallback = df.groupby(by)[cols].mean()  # 가중치 합이 0인 동(고령인구 0)은 단순평균
    return s.div(wsum.replace(0, np.nan), axis=0).fillna(fallback)


def density(units: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, key: str) -> pd.DataFrame:
    j = gpd.sjoin(pois, units[[key, "geometry"]], predicate="within", how="inner")
    cnt = j.groupby([key, "category"]).size().unstack(fill_value=0)
    area = units.dissolve(by=key, aggfunc={"area_km2": "sum"})["area_km2"]
    cnt = cnt.reindex(area.index, fill_value=0)
    return cnt.div(area, axis=0).add_prefix("dens_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=Path,
                    default=sorted((C.ROOT / "data" / "boundary").glob("HangJeongDong_ver*.geojson"))[-1])
    ap.add_argument("--poi-dir", type=Path, default=C.REGIONAL_DIR / "poi")
    ap.add_argument("--grid-pop", type=Path, default=C.REGIONAL_DIR / "grid_pop.csv")
    ap.add_argument("--out", type=Path, default=C.REGIONAL_DIR)
    ap.add_argument("--grid-size", type=int, default=C.GRID_SIZE_M)
    ap.add_argument("--radius", type=float, default=C.ACCESS_RADIUS_M)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    b = load_boundary(a.boundary)
    b["kostat_sgg"] = b["adm_cd"].str[:5]
    seoul = b[b["sido"] == C.SEOUL_PREFIX].copy()
    origin = b[b["sido"].isin(C.ORIGIN_PREFIXES["mois"])].copy()
    pois = load_pois(a.poi_dir, a.out)

    # --- 서울: 100m 격자 접근성 → 동·구 ---
    grid = build_grid(seoul, a.grid_size)
    grid = attach_grid_pop(grid, a.grid_pop, a.grid_size)
    acc = grid_access(grid, pois, a.radius)
    grid = pd.concat([grid, acc], axis=1)
    acc_cols = list(acc.columns)
    print(f"[regional] 서울 {a.grid_size}m 격자 {len(grid):,}개, 반경 {a.radius:.0f}m 접근성 {len(acc_cols)}개 변수")

    dong = weighted_mean(grid, acc_cols, "adm_cd2", "w")
    dong = dong.join(density(seoul, pois, "adm_cd2"))
    gu = weighted_mean(grid, acc_cols, "sgg", "w").join(density(seoul, pois, "sgg"))
    if "pop_65p" in grid:
        for tbl, key in [(dong, "adm_cd2"), (gu, "sgg")]:
            p = grid.groupby(key)[["pop_total", "pop_65p"]].sum()
            tbl["elderly_share"] = p["pop_65p"] / p["pop_total"].replace(0, np.nan)

    # --- 출발 시군구: 밀도 ---
    osgg = density(origin, pois, "sgg")

    # --- 코드 변형 전개 후 저장 ---
    meta = b.set_index("adm_cd2")
    dong_rows = [(c, k) for k in dong.index for c in dong_codes(meta.loc[k].to_dict() | {"adm_cd2": k})]
    dong_out = pd.DataFrame(dong_rows, columns=["code", "adm_cd2"]).merge(
        dong.reset_index().rename(columns={"index": "adm_cd2"}), on="adm_cd2")
    dong_out.insert(1, "sgg", dong_out["adm_cd2"].str[:5])
    dong_out.to_csv(a.out / "regional_dong.csv", index=False, encoding="utf-8-sig")

    kmap = b.drop_duplicates("sgg").set_index("sgg")["kostat_sgg"]
    sgg_all = pd.concat([gu, osgg])  # 출발지엔 격자 접근성이 없으므로 결측으로 둔다
    sgg_rows = [(c, k) for k in sgg_all.index for c in sgg_codes(k, kmap[k])]
    sgg_out = pd.DataFrame(sgg_rows, columns=["code", "sgg"]).merge(
        sgg_all.reset_index().rename(columns={"index": "sgg"}), on="sgg")
    sgg_out.to_csv(a.out / "regional_sgg.csv", index=False, encoding="utf-8-sig")
    print(f"[regional] 동 {len(dong)}개 × {dong.shape[1]}변수, 시군구 {len(sgg_all)}개 × "
          f"{sgg_all.shape[1]}변수 → {a.out}")


if __name__ == "__main__":
    main()
