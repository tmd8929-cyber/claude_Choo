"""생활이동 원자료 CSV → 서울로 유입되는 수도권 고령인구 이동 레코드.

사용법:
    python -m src.preprocess [--raw data/raw] [--out data/processed/elderly_inflow.parquet]

처리 단계 (이전 청년 연구의 전처리 1)~5)와 동일한 흐름)
1. data/raw 의 모든 CSV(대개 CP949, 시간대별 파일)를 청크 단위로 읽어 합친다.
2. 컬럼명을 표준화하고, 경기·인천 출발 → 서울 도착, 65세 이상만 남긴다.
3. 이동유형 9개를 도착지 기준 H/W/E 3개 목적으로 축소한다.
4. 주중/주말, 낮/밤, 5세 연령구간을 만든다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def normalize_colname(name: str) -> str:
    return re.sub(r"[\s_()\[\]]", "", str(name)).lower()


def build_rename_map(columns) -> dict[str, str]:
    lookup = {normalize_colname(c): std for std, cands in C.COLUMN_ALIASES.items() for c in cands}
    rename = {}
    for col in columns:
        std = lookup.get(normalize_colname(col))
        if std and std not in rename.values():
            rename[col] = std
    return rename


def sniff_encoding(path: Path) -> str:
    head = path.open("rb").read(200_000)
    for enc in ("utf-8-sig", "cp949"):
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "cp949"


def clean_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def detect_code_scheme(dest: pd.Series) -> str:
    """서울 자치구 코드로 코드체계 판별: 행안부 11110~11740, 통계청 11010~11250."""
    gu = pd.to_numeric(dest[dest.str.startswith(C.SEOUL_PREFIX)].str[:5], errors="coerce")
    return "mois" if gu.max() >= 11300 else "kostat"


def parse_age(s: pd.Series) -> pd.Series:
    """'65', '65~69', '65세 이상' 등에서 구간 하한값을 뽑는다."""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def parse_flow(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = s.astype(str).str.strip()
    masked = raw.eq("*")
    val = pd.to_numeric(raw.where(~masked), errors="coerce").where(~masked, C.MASKED_VALUE)
    return val, masked


def assign_purpose(df: pd.DataFrame) -> pd.Series:
    dest_type = df["move_type"].astype(str).str.strip().str[-1].str.upper()
    return dest_type.map(C.MOVE_TYPE_DEST_PURPOSE)


def assign_daytype(df: pd.DataFrame) -> pd.Series:
    if "dow" in df:
        dow = df["dow"].astype(str).str.strip().str.lower()
    else:
        dow = pd.to_datetime(df["date"].str[:8], format="%Y%m%d").dt.dayofweek.add(1).astype(str)
    return np.where(dow.isin(C.WEEKEND_DOW), "wk", "wd")


def filter_chunk(chunk: pd.DataFrame, scheme: str | None) -> tuple[pd.DataFrame, str]:
    chunk = chunk.rename(columns=build_rename_map(chunk.columns))
    missing = {"orig", "dest", "age", "flow", "move_type", "arr_time", "date"} - set(chunk.columns)
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing} / 실제 컬럼: {list(chunk.columns)}")

    chunk["orig"] = clean_code(chunk["orig"])
    chunk["dest"] = clean_code(chunk["dest"])
    scheme = scheme or detect_code_scheme(chunk["dest"])
    chunk["age_lb"] = parse_age(chunk["age"])

    keep = (
        chunk["dest"].str.startswith(C.SEOUL_PREFIX)
        & chunk["orig"].str.startswith(C.ORIGIN_PREFIXES[scheme])
        & (chunk["age_lb"] >= C.ELDERLY_MIN_AGE)
    )
    df = chunk.loc[keep].copy()
    if df.empty:
        return df, scheme

    df["date"] = clean_code(df["date"])
    df["ym"] = df["date"].str[:6]
    df["flow"], df["masked"] = parse_flow(df["flow"])
    df["purpose"] = assign_purpose(df)
    df["age5"] = np.minimum(df["age_lb"] // 5 * 5, C.AGE_TOP).astype(int)
    hour = pd.to_numeric(df["arr_time"], errors="coerce")
    df["daynight"] = np.where(hour.isin(C.DAY_HOURS), "day", "night")
    df["timeband"] = pd.cut(hour, C.TIMEBAND_EDGES, right=False, labels=C.TIMEBAND_LABELS).astype(str)
    df["timeband"] = df["timeband"].replace(C.TIMEBAND_MERGE)
    df["daytype"] = assign_daytype(df)
    df["orig_sgg"] = df["orig"].str[:5]
    df["dest_gu"] = df["dest"].str[:5]
    if "travel_min" in df:
        df["travel_min"] = pd.to_numeric(df["travel_min"], errors="coerce")
    cols = ["ym", "date", "daytype", "daynight", "timeband", "orig", "orig_sgg", "dest", "dest_gu",
            "sex", "age5", "purpose", "move_type", "travel_min", "flow", "masked"]
    return df[[c for c in cols if c in df]], scheme


def process(raw_dir: Path, out_path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in (".csv", ".txt"))
    if not files:
        raise FileNotFoundError(f"{raw_dir} 에 CSV가 없습니다.")
    parts, n_in, scheme = [], 0, None
    for f in files:
        enc = sniff_encoding(f)
        for chunk in pd.read_csv(f, encoding=enc, dtype=str, chunksize=chunksize, low_memory=False):
            n_in += len(chunk)
            part, scheme = filter_chunk(chunk, scheme)
            if not part.empty:
                parts.append(part)
        print(f"[preprocess] {f.name} ({enc}) 누적 입력 {n_in:,}행")
    df = pd.concat(parts, ignore_index=True)
    unknown = df["purpose"].isna().mean()
    if unknown:
        print(f"[preprocess] 경고: 이동유형 미분류 {unknown:.1%} → 제외")
        df = df.dropna(subset=["purpose"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[preprocess] 코드체계 {scheme}, 고령 서울유입 {len(df):,}행 "
          f"(마스킹 {df['masked'].mean():.1%}, 이동량 중 마스킹 비중 "
          f"{df.loc[df['masked'], 'flow'].sum() / df['flow'].sum():.1%}) → {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--out", type=Path, default=C.PROC_DIR / "elderly_inflow.parquet")
    a = ap.parse_args()
    process(a.raw, a.out)


if __name__ == "__main__":
    main()
