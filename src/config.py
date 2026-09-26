"""분석 전역 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REGIONAL_DIR = ROOT / "data" / "regional"
OUT_DIR = ROOT / "outputs"

# 고령 기준 (나이 구간 하한값)
ELDERLY_MIN_AGE = 65
# 5세 구간 상한. 이 값 이상은 하나로 묶는다 (예: 85 → '85+')
AGE_TOP = 85

# 낮 시간대(도착시간 기준). 나머지는 밤. 오프셋 계산에 시간 수를 쓴다.
DAY_HOURS = set(range(7, 19))  # 07~18시, 12시간

# 서울 및 출발지(수도권) 코드 접두어. 행안부/통계청 코드체계를 자동 판별한다.
SEOUL_PREFIX = "11"
ORIGIN_PREFIXES = {
    "mois": ("41", "28"),  # 행안부: 경기 41, 인천 28
    "kostat": ("31", "23"),  # 통계청: 경기 31, 인천 23
}

# 비식별 처리('*')된 소량 셀(1~2명)에 넣을 대체값. 민감도 분석 시 0 / 2 로 바꿔 본다.
MASKED_VALUE = 1.5

# 원자료 컬럼명 후보 → 표준 컬럼명.
# 공백/밑줄/괄호를 제거한 뒤 비교한다 (preprocess.normalize_colname 참고).
COLUMN_ALIASES = {
    "date": ["대상연월", "기준일id", "기준일자", "일자", "기준일"],
    "dow": ["요일"],
    "arr_time": ["도착시간"],
    "orig": ["출발시군구코드", "출발행정동코드", "출발지코드", "o행정동코드"],
    "dest": ["도착시군구코드", "도착행정동코드", "도착지코드", "d행정동코드"],
    "sex": ["성별"],
    "age": ["나이", "연령", "연령대"],
    "move_type": ["이동유형"],
    "purpose": ["이동목적"],
    "travel_min": ["평균이동시간분", "평균이동시간"],
    "flow": ["이동인구합", "이동인구", "이동인구수"],
}

# 이동유형(9개: 출발-도착 장소유형 조합)을 도착지 기준 3개 목적으로 축소
# H: 야간상주지(집) 관련, W: 주간상주지(일) 관련, E: 그 외
MOVE_TYPE_DEST_PURPOSE = {"H": "H", "W": "W", "E": "E"}

WEEKEND_DOW = {"토", "일", "sat", "sun", "6", "7"}
