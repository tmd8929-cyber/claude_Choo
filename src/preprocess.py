"""생활이동 원자료 CSV → 서울 유입 고령인구 이동 테이블.

사용법:
    python -m src.preprocess [--raw data/raw] [--out data/processed/elderly_inflow.parquet]

처리 단계
1. data/raw 의 모든 CSV(원자료는 대부분 CP949)를 청크 단위로 읽는다.
2. 컬럼명을 표준화하고 고령(65세+) · 서울 외 출발 → 서울 도착 이동만 남긴다.
3. 이동목적(없으면 이동유형으로 추정)을 붙여 행 단위로 저장한다.
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
    lookup = {}
    for std, cands in C.COLUMN_ALIASES.items():
        for c in cands:
            lookup[normalize_colname(c)] = std
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


def parse_age(s: pd.Series) -> pd.Series:
    """'65', '65~69', '65세 이상' 등에서 구간 하한값을 뽑는다."""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def parse_flow(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = s.astype(str).str.strip()
    masked = raw.eq("*")
    val = pd.to_numeric(raw.where(~masked), errors="coerce")
    val = val.where(~masked, C.MASKED_VALUE)
    return val, masked


def assign_purpose(df: pd.DataFrame) -> pd.Series:
    if "purpose" in df:
        code = df["purpose"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        return code.map(C.PURPOSE_CODES).fillna(code)
    if "move_type" in df:
        dest_type = df["move_type"].astype(str).str.strip().str[-1].str.upper()
        return dest_type.map(C.MOVE_TYPE_DEST_PURPOSE).fillna("미상")
    raise KeyError("이동목적/이동유형 컬럼이 모두 없습니다.")


def clean_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns=build_rename_map(chunk.columns))
    missing = {"orig", "dest", "age", "flow"} - set(chunk.columns)
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing} / 실제 컬럼: {list(chunk.columns)}")

    chunk["orig"] = clean_code(chunk["orig"])
    chunk["dest"] = clean_code(chunk["dest"])
    chunk["age_lb"] = parse_age(chunk["age"])

    keep = (
        chunk["dest"].str.startswith(C.SEOUL_PREFIX)
        & ~chunk["orig"].str.startswith(C.SEOUL_PREFIX)
        & (chunk["age_lb"] >= C.ELDERLY_MIN_AGE)
    )
    df = chunk.loc[keep].copy()
    if df.empty:
        return df

    df["flow"], df["masked"] = parse_flow(df["flow"])
    df["purpose_lbl"] = assign_purpose(df)
    df["age_grp"] = np.where(df["age_lb"] >= C.OLD_OLD_MIN_AGE, "75+", "65-74")
    # 상위 수준 식별자: 코드 앞 5자리 = 시군구
    df["orig_sgg"] = df["orig"].str[:5]
    df["dest_gu"] = df["dest"].str[:5]
    if "dow" in df:
        df["daytype"] = np.where(df["dow"].astype(str).str.strip().isin(["토", "일", "6", "7", "0"]),
                                 "주말", "주중")
    else:
        df["daytype"] = "전체"
    cols = [c for c in ["date", "dow", "daytype", "arr_time", "orig", "orig_sgg", "dest",
                        "dest_gu", "sex", "age_lb", "age_grp", "purpose_lbl", "move_type",
                        "travel_min", "flow", "masked"] if c in df]
    return df[cols]


def process(raw_dir: Path, out_path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in (".csv", ".txt"))
    if not files:
        raise FileNotFoundError(f"{raw_dir} 에 CSV가 없습니다.")
    parts, n_in = [], 0
    for f in files:
        enc = sniff_encoding(f)
        for chunk in pd.read_csv(f, encoding=enc, dtype=str, chunksize=chunksize, low_memory=False):
            n_in += len(chunk)
            part = filter_chunk(chunk)
            if not part.empty:
                parts.append(part)
        print(f"[preprocess] {f.name} ({enc}) 처리 완료, 누적 입력 {n_in:,}행")
    df = pd.concat(parts, ignore_index=True)
    if "travel_min" in df:
        df["travel_min"] = pd.to_numeric(df["travel_min"], errors="coerce")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[preprocess] 고령 서울유입 {len(df):,}행 (마스킹 {df['masked'].mean():.1%}) → {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR)
    ap.add_argument("--out", type=Path, default=C.PROC_DIR / "elderly_inflow.parquet")
    a = ap.parse_args()
    process(a.raw, a.out)


if __name__ == "__main__":
    main()
