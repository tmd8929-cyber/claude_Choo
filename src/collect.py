"""지역변수 원자료 수집 → data/regional/poi/*.csv, data/regional/grid_pop.csv (regional.py 입력).

출처별 명령 (키는 환경변수 또는 .env, .env.example 참고)
    subway    서울 열린데이터 역사마스터(subwayStationMaster)          SEOUL_API_KEY
    welfare   서울 열린데이터 노인여가복지시설 (서비스명 --service로 지정)  SEOUL_API_KEY
    store     소상공인시장진흥공단 상가(상권)정보 sdsc2 (시도 단위)        DATA_GO_KR_KEY
    hospital  건강보험심사평가원 병원정보서비스 v2                      DATA_GO_KR_KEY
    pharmacy  건강보험심사평가원 약국정보서비스                         DATA_GO_KR_KEY
    osm       OpenStreetMap Overpass: 종교시설(religion), 공원(park)    (키 없음)
    grid      SGIS 100m 격자통계 파일(로컬) → grid_pop.csv            (자료신청 후 내려받은 파일)

- 수집 범위는 서울 + 출발지(경기·인천). 출발 시군구 밀도(o_dens_*)에도 쓰인다.
- API 응답은 페이지별로 data/regional/raw_api/<출처>/ 에 저장하고, 다시 실행하면 저장된 페이지는 건너뛴다
  (일일 호출 한도에 걸려도 이어받기 가능).
- 같은 범주를 두 출처가 주면(예: 상가정보의 '병원'과 심평원 병원) regional.py가 C.POI_CATEGORY_SOURCE의
  출처만 남긴다.

사용법:
    python -m src.collect subway store hospital pharmacy osm
    python -m src.collect welfare --service welfare=<서울 열린데이터 서비스명>
    python -m src.collect grid --grid-src <SGIS 격자 파일 또는 폴더> --elderly-items <항목코드,...>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests

from . import config as C

RAW_API_DIR = C.REGIONAL_DIR / "raw_api"
POI_DIR = C.REGIONAL_DIR / "poi"
KEYS = ["SEOUL_API_KEY", "DATA_GO_KR_KEY", "SGIS_KEY", "SGIS_SECRET", "KOSIS_KEY"]


# ---------------------------------------------------------------------------- 공통
def load_env(path: Path = C.ROOT / ".env") -> None:
    """.env가 있으면 환경변수에 없는 키만 채운다 (python-dotenv 없이)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and m.group(2) and not os.environ.get(m.group(1)):
            os.environ[m.group(1)] = m.group(2).strip("'\"")


def need_key(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"[collect] {name}가 없습니다. 환경 Secrets 또는 .env에 넣어 주세요 (.env.example 참고).")
    return v


def mask(text: str) -> str:
    for k in KEYS:
        v = os.environ.get(k)
        if v:
            text = text.replace(v, "***").replace(requests.utils.quote(v, safe=""), "***")
    return text


def get_json(url: str, params: dict | None = None, retries: int = 4) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                # data.go.kr는 키 오류 등을 XML로 돌려준다
                raise RuntimeError(f"JSON 아님: {r.text[:300]}") from None
        except (requests.RequestException, RuntimeError) as e:
            if i == retries - 1:
                raise RuntimeError(mask(f"{url} 실패: {e}")) from None
            time.sleep(2 ** (i + 1))


def cached_pages(source: str, fetch_page, n_pages_of) -> list[dict]:
    """페이지 1을 받아 전체 페이지 수를 알아낸 뒤 나머지를 받는다. 받은 페이지는 저장해 재사용한다."""
    d = RAW_API_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    pages, p, total = [], 1, None
    while total is None or p <= total:
        f = d / f"page_{p:05d}.json"
        if f.exists():
            js = json.loads(f.read_text(encoding="utf-8"))
        else:
            js = fetch_page(p)
            f.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.2)
        if total is None:
            total = n_pages_of(js)
            print(f"[collect] {source}: {total}쪽")
        pages.append(js)
        p += 1
    return pages


def save_poi(df: pd.DataFrame, name: str, lon: str = "경도", lat: str = "위도") -> Path:
    df = df.copy()
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    bad = df[lon].isna() | df[lat].isna() | ~df[lon].between(124, 132) | ~df[lat].between(33, 39)
    if bad.any():
        print(f"[collect] {name}: 좌표 결측·범위 밖 {bad.sum():,}건 제외")
    POI_DIR.mkdir(parents=True, exist_ok=True)
    out = POI_DIR / f"{name}.csv"
    df[~bad].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[collect] → {out} ({(~bad).sum():,}건)")
    return out


# ---------------------------------------------------------------------------- 서울 열린데이터
SEOUL_URL = "http://openapi.seoul.go.kr:8088/{key}/json/{service}/{start}/{end}/"
SEOUL_PAGE = 1000


def seoul_rows(source: str, service: str) -> pd.DataFrame:
    key = need_key("SEOUL_API_KEY")

    def fetch(p):
        js = get_json(SEOUL_URL.format(key=key, service=service,
                                       start=(p - 1) * SEOUL_PAGE + 1, end=p * SEOUL_PAGE))
        if service not in js:
            res = js.get("RESULT", js)
            raise RuntimeError(f"서울 열린데이터 {service}: {res} (서비스명·인증키 확인)")
        return js

    def n_pages(js):
        return -(-int(js[service].get("list_total_count", 0)) // SEOUL_PAGE)

    rows = [r for js in cached_pages(source, fetch, n_pages) for r in js[service].get("row", [])]
    return pd.DataFrame(rows)


def pick(df: pd.DataFrame, cands: list[str]) -> str:
    up = {c.upper(): c for c in df.columns}
    for c in cands:
        if c.upper() in up:
            return up[c.upper()]
    raise KeyError(f"{cands} 중 해당 컬럼 없음: {list(df.columns)}")


LON_FIELDS = ["LOT", "LNG", "LON", "LONGITUDE", "XCNTS", "X_COORD", "경도"]
LAT_FIELDS = ["LAT", "LATITUDE", "YDNTS", "Y_COORD", "위도"]
NAME_FIELDS = ["BLDN_NM", "FCLT_NM", "FACI_NM", "NM", "NAME", "시설명"]


def collect_subway(service: str) -> None:
    df = seoul_rows("subway", service)
    lon, lat, nm = pick(df, LON_FIELDS), pick(df, LAT_FIELDS), pick(df, NAME_FIELDS)
    line = next((c for c in df.columns if c.upper() in ("ROUTE", "LINE_NUM", "호선")), None)
    out = pd.DataFrame({"역명": df[nm], "호선": df[line] if line else pd.NA,
                        "경도": pd.to_numeric(df[lon], errors="coerce"),
                        "위도": pd.to_numeric(df[lat], errors="coerce")})
    # 역사마스터는 노선마다 한 행 → 환승역을 역 하나로 (역명 정규화 + 평균 좌표)
    out["역"] = out["역명"].astype(str).str.replace(r"\(.*\)|역$", "", regex=True).str.strip()
    n0 = len(out)
    out = (out.groupby("역", as_index=False)
              .agg(역명=("역명", "first"), 호선=("호선", lambda s: ",".join(map(str, s.dropna().unique()))),
                   경도=("경도", "mean"), 위도=("위도", "mean"))
              .drop(columns="역"))
    print(f"[collect] subway: 노선별 {n0:,}행 → 역 {len(out):,}개")
    save_poi(out, "subway")  # 업종명 컬럼이 없으므로 regional.py가 파일명(subway)을 범주로 쓴다


def collect_welfare(service: str) -> None:
    df = seoul_rows("welfare", service)
    lon, lat = pick(df, LON_FIELDS), pick(df, LAT_FIELDS)
    nm = next((c for c in df.columns if c.upper() in NAME_FIELDS), None)
    out = pd.DataFrame({"시설명": df[nm] if nm else pd.NA, "경도": df[lon], "위도": df[lat]})
    save_poi(out, "welfare")


# ---------------------------------------------------------------------------- 공공데이터포털
def data_go_key() -> str:
    k = need_key("DATA_GO_KR_KEY")
    return unquote(k) if "%" in k else k  # 인코딩 키를 넣어도 requests가 다시 인코딩하지 않게


STORE_URL = "https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInDong"
STORE_SIDO = [C.SEOUL_PREFIX, *C.ORIGIN_PREFIXES["mois"]]  # 행안부 시도코드 11, 41, 28
DG_PAGE = 1000


def collect_store() -> None:
    key, frames = data_go_key(), []
    for sido in STORE_SIDO:
        def fetch(p, sido=sido):
            js = get_json(STORE_URL, {"serviceKey": key, "divId": "ctprvnCd", "key": sido,
                                      "pageNo": p, "numOfRows": DG_PAGE, "type": "json"})
            code = js.get("header", {}).get("resultCode")
            if code not in ("00", "03"):  # 03: 데이터 없음
                raise RuntimeError(f"상가정보 {sido}: {js.get('header')}")
            return js

        pages = cached_pages(f"store_{sido}", fetch,
                             lambda js: -(-int(js.get("body", {}).get("totalCount", 0)) // DG_PAGE))
        frames += [pd.DataFrame(js["body"]["items"]) for js in pages if js.get("body", {}).get("items")]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("bizesId") if "bizesId" in df else df
    out = df.rename(columns={"bizesNm": "상호명", "indsLclsNm": "상권업종대분류명",
                             "indsMclsNm": "상권업종중분류명", "indsSclsNm": "상권업종소분류명",
                             "lon": "경도", "lat": "위도", "signguCd": "시군구코드", "adongCd": "행정동코드"})
    cols = ["상호명", "상권업종대분류명", "상권업종중분류명", "상권업종소분류명", "시군구코드", "행정동코드", "경도", "위도"]
    save_poi(out[[c for c in cols if c in out]], "store")


HIRA_URL = {
    "hospital": "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList",
    "pharmacy": "https://apis.data.go.kr/B551182/pharmacyInfoService/getParmacyBasisList",
}
HIRA_SIDO = {"서울": "110000", "경기": "310000", "인천": "220000"}


def hira_items(js: dict) -> list[dict]:
    items = (js.get("response", {}).get("body", {}).get("items") or {})
    items = items.get("item", []) if isinstance(items, dict) else items
    return [items] if isinstance(items, dict) else list(items)


def collect_hira(kind: str) -> None:
    key, frames = data_go_key(), []
    for sido_nm, sido in HIRA_SIDO.items():
        def fetch(p, sido=sido):
            js = get_json(HIRA_URL[kind], {"serviceKey": key, "sidoCd": sido, "pageNo": p,
                                           "numOfRows": DG_PAGE, "_type": "json"})
            head = js.get("response", {}).get("header", {})
            if head.get("resultCode") not in ("00", None):
                raise RuntimeError(f"심평원 {kind} {sido_nm}: {head}")
            return js

        pages = cached_pages(f"{kind}_{sido}", fetch, lambda js: -(
            -int(js.get("response", {}).get("body", {}).get("totalCount", 0)) // DG_PAGE))
        frames += [pd.DataFrame(hira_items(js)) for js in pages]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("ykiho") if "ykiho" in df else df
    out = df.rename(columns={"yadmNm": "요양기관명", "clCdNm": "종별코드명", "XPos": "경도", "YPos": "위도",
                             "sgguCdNm": "시군구명"})
    if kind == "hospital":
        # 종별코드명(상급종합·종합병원·병원·요양병원·의원·치과의원·한의원·보건소…)으로 hospital/clinic 분류
        save_poi(out[["요양기관명", "종별코드명", "시군구명", "경도", "위도"]], "hospital")
    else:
        # 업종명 컬럼을 두지 않아 파일명(pharmacy)이 범주가 된다
        save_poi(out[["요양기관명", "시군구명", "경도", "위도"]].rename(columns={"요양기관명": "약국명"}),
                 "pharmacy")


# ---------------------------------------------------------------------------- OpenStreetMap
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# 수도권(서울·경기·인천) 행정경계 relation을 area로 쓴다. ISO3166-2 코드로 찾는다.
OSM_AREAS = ["KR-11", "KR-41", "KR-28"]
OSM_QUERIES = {
    "religion": 'nwr["amenity"="place_of_worship"](area.a);',
    "park": 'nwr["leisure"="park"](area.a);',
}


def collect_osm() -> None:
    areas = "".join(f'area["ISO3166-2"="{a}"];' for a in OSM_AREAS)
    for cat, q in OSM_QUERIES.items():
        f = RAW_API_DIR / "osm" / f"{cat}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            query = f"[out:json][timeout:600];({areas})->.a;({q});out center tags;"
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=900)
            r.raise_for_status()
            f.write_text(r.text, encoding="utf-8")
        els = json.loads(f.read_text(encoding="utf-8"))["elements"]
        rows = [{"osm_id": f"{e['type'][0]}{e['id']}",
                 "시설명": e.get("tags", {}).get("name"),
                 "종류": e.get("tags", {}).get("religion") if cat == "religion" else None,
                 "경도": e.get("lon", e.get("center", {}).get("lon")),
                 "위도": e.get("lat", e.get("center", {}).get("lat"))} for e in els]
        print(f"[collect] osm {cat}: {len(rows):,}개 (way·relation은 중심점)")
        save_poi(pd.DataFrame(rows), cat)  # 업종명 컬럼 없음 → 파일명이 범주


# ---------------------------------------------------------------------------- SGIS 100m 격자
# 국가지점번호 격자: 문자 2개(100km 단위) + x·y 숫자 반씩. EPSG:5179 기준
#   x: 가=700,000 나=800,000 … / y: 가=1,300,000 나=1,400,000 … (서울은 '다사')
GRID_LETTERS = "가나다라마바사아"
GRID_X0, GRID_Y0 = 700_000, 1_300_000


def grid_code_to_xy(code: pd.Series) -> pd.DataFrame:
    """'다사604796'(100m) · '다사6079'(1km) 같은 격자코드 → 격자 중심점 x, y (EPSG:5179)."""
    m = code.astype(str).str.replace(r"\s", "", regex=True).str.extract(r"^([가-힣])([가-힣])(\d+)$")
    digits = m[2].fillna("")
    n = digits.str.len() // 2
    if (digits.str.len() % 2 != 0).any() or m[0].isna().any():
        raise ValueError(f"격자코드 형식 오류 예: {code[m[0].isna() | (digits.str.len() % 2 != 0)].head(3).tolist()}")
    size = 100_000 / np.power(10, n.astype(float))
    xd = pd.to_numeric([d[:k] for d, k in zip(digits, n)])
    yd = pd.to_numeric([d[k:] for d, k in zip(digits, n)])
    x0 = m[0].map(lambda c: GRID_X0 + 100_000 * GRID_LETTERS.index(c))
    y0 = m[1].map(lambda c: GRID_Y0 + 100_000 * GRID_LETTERS.index(c))
    return pd.DataFrame({"x": x0 + xd * size + size / 2, "y": y0 + yd * size + size / 2, "size": size},
                        index=code.index)


def read_sgis_grid(paths: list[Path]) -> pd.DataFrame:
    """SGIS 격자통계 텍스트: '기준연도^격자코드^항목코드^값' (헤더 없음, 비식별 셀은 값이 N/A 등)."""
    frames = []
    for f in paths:
        for enc in ("utf-8-sig", "cp949"):
            try:
                df = pd.read_csv(f, sep="^", header=None, dtype=str, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        df = df.iloc[:, :4]
        df.columns = ["year", "grid", "item", "value"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def collect_grid(src: Path, total_item: str, elderly_items: list[str], masked: float) -> None:
    files = sorted(src.rglob("*.txt")) + sorted(src.rglob("*.csv")) if src.is_dir() else [src]
    if not files:
        sys.exit(f"[collect] {src}에 격자 파일이 없습니다.")
    df = read_sgis_grid(files)
    items = df["item"].value_counts()
    if not elderly_items or total_item not in items:
        print("[collect] 파일의 항목코드 (행 수):")
        print(items.to_string())
        sys.exit("[collect] --total-item / --elderly-items를 위 항목코드 중에서 지정하세요 "
                 "(65세 이상에 해당하는 연령 항목들을 쉼표로).")
    n_mask = df["value"].isna().sum()
    df["value"] = df["value"].fillna(masked)  # 비식별(5명 미만 등) 셀 대체
    wide = df[df["item"].isin([total_item, *elderly_items])].pivot_table(
        index="grid", columns="item", values="value", aggfunc="sum", fill_value=0)
    out = pd.DataFrame({"grid": wide.index, "pop_total": wide[total_item].to_numpy(),
                        "pop_65p": wide.reindex(columns=elderly_items, fill_value=0).sum(axis=1).to_numpy()})
    out = pd.concat([out, grid_code_to_xy(out["grid"]).reset_index(drop=True)], axis=1)
    if (out["size"] != C.GRID_SIZE_M).any():
        print(f"[collect] 경고: 격자 크기 {sorted(out['size'].unique())}m (설정 {C.GRID_SIZE_M}m)")
    C.REGIONAL_DIR.mkdir(parents=True, exist_ok=True)
    dst = C.REGIONAL_DIR / "grid_pop.csv"
    out[["grid", "x", "y", "pop_total", "pop_65p"]].to_csv(dst, index=False)
    print(f"[collect] 격자 {len(out):,}개, 총인구 {out['pop_total'].sum():,.0f}, 65+ {out['pop_65p'].sum():,.0f} "
          f"(비식별 {n_mask:,}칸 → {masked}) → {dst}")


# ----------------------------------------------------------------------------
SOURCES = ["subway", "welfare", "store", "hospital", "pharmacy", "osm", "grid"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", choices=SOURCES)
    ap.add_argument("--service", action="append", default=[], metavar="출처=서비스명",
                    help="서울 열린데이터 서비스명 지정/변경 (예: welfare=xxxx)")
    ap.add_argument("--grid-src", type=Path, help="SGIS 격자통계 파일 또는 폴더")
    ap.add_argument("--total-item", default="to_in_001", help="총인구 항목코드")
    ap.add_argument("--elderly-items", default="", help="65세 이상 항목코드(쉼표 구분)")
    ap.add_argument("--grid-masked", type=float, default=C.MASKED_VALUE, help="격자 비식별 셀 대체값")
    a = ap.parse_args()
    load_env()
    services = dict(C.SEOUL_SERVICES) | dict(s.split("=", 1) for s in a.service)

    for s in a.sources:
        print(f"[collect] === {s}")
        if s in ("subway", "welfare"):
            if not services.get(s):
                sys.exit(f"[collect] {s}: 서울 열린데이터 서비스명이 없습니다. --service {s}=<서비스명> "
                         "(데이터셋 페이지 'Open API' 탭의 서비스명)")
            (collect_subway if s == "subway" else collect_welfare)(services[s])
        elif s == "store":
            collect_store()
        elif s in ("hospital", "pharmacy"):
            collect_hira(s)
        elif s == "osm":
            collect_osm()
        elif s == "grid":
            if not a.grid_src:
                sys.exit("[collect] grid: --grid-src가 필요합니다 (SGIS 자료신청으로 받은 100m 격자 인구 파일).")
            collect_grid(a.grid_src, a.total_item, [x for x in a.elderly_items.split(",") if x], a.grid_masked)


if __name__ == "__main__":
    main()
