"""분석 전역 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

# 고령 기준 (나이 구간의 하한값 기준)
ELDERLY_MIN_AGE = 65
# 75세 이상을 후기고령으로 구분
OLD_OLD_MIN_AGE = 75

# 서울 행정구역 코드 접두어 (행안부 '11', 통계청 '11' 동일)
SEOUL_PREFIX = "11"

# 비식별 처리('*')된 소량 셀(1~2명)에 넣을 대체값
MASKED_VALUE = 1.5

# 원자료 컬럼명 후보 → 표준 컬럼명.
# 공백/밑줄/괄호를 제거한 뒤 비교한다 (normalize_colname 참고).
COLUMN_ALIASES = {
    "date": ["대상연월", "기준일id", "기준일자", "일자", "기준일"],
    "dow": ["요일"],
    "arr_time": ["도착시간"],
    "orig": ["출발행정동코드", "출발시군구코드", "출발지코드", "o행정동코드"],
    "dest": ["도착행정동코드", "도착시군구코드", "도착지코드", "d행정동코드"],
    "sex": ["성별"],
    "age": ["나이", "연령", "연령대"],
    "move_type": ["이동유형"],
    "purpose": ["이동목적"],
    "travel_min": ["평균이동시간분", "평균이동시간"],
    "flow": ["이동인구합", "이동인구", "이동인구수"],
}

# 이동목적 코드표 (수도권 생활이동 기준. 실제 데이터 명세서로 확인 필요)
PURPOSE_CODES = {
    "1": "출근", "2": "등교", "3": "귀가", "4": "쇼핑",
    "5": "관광", "6": "병원", "7": "기타",
}

# 이동목적 컬럼이 없을 때 이동유형(출발-도착 장소유형)의 도착 측으로 목적 추정
# H: 거주지, W: 직장(상주지), E: 기타
MOVE_TYPE_DEST_PURPOSE = {"H": "귀가", "W": "출근", "E": "기타활동"}
