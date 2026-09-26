# 수도권 고령인구는 왜 서울에 오는가 — 서울생활이동 목적별 다수준 분석

서울열린데이터광장 **서울생활이동 데이터**로 경기·인천에서 서울로 들어오는 65세 이상 인구의 이동을
목적(H/W/E)별로 다수준 분석한다. 이전 청년(20–34세) 연구(「수도권 청년들은 어떤 이유로 서울에 오는가」,
2022.08 시군구 자료)의 설계를 잇되, 막혔던 다수준 모형을 식별 가능한 형태로 다시 구성하고
도착지를 **행정동(427) ⊂ 자치구(25)** 로 내려 100m 격자 기반 지역변수를 붙였다.

## 파이프라인
```bash
pip install -r requirements.txt
apt-get install r-base-core r-cran-glmmtmb r-cran-data.table
bash scripts/get_boundary.sh                 # 행정동 경계 (GitHub, SGIS 원자료)

# 1) 원자료: 생활이동 CSV(시간대별 전부) → data/raw/
python -m src.preprocess                     # → data/processed/elderly_inflow.parquet
# 2) 지역변수: 시설 좌표 CSV → data/regional/poi/, (선택) 100m 격자인구 → data/regional/grid_pop.csv
python -m src.regional                       # → data/regional/regional_{dong,sgg}.csv
# 3) 코호트
python -m src.cohort --dest-level dong --time-var timeband   # → data/processed/cohort_dong.csv
python -m src.cohort --dest-level gu                         # → data/processed/cohort_gu.csv
# 4) 다수준 모형
Rscript R/mlm_dong.R data/processed/cohort_dong.csv outputs "d_acc_hospital,d_acc_clinic,d_acc_welfare,d_acc_market,d_dist_subway"
Rscript R/mlm_gu.R   data/processed/cohort_gu.csv   outputs
```
합성 데이터 검증 순서는 `scripts/make_synthetic.py` 머리말 참고.

## 정의
| 항목 | 정의 (`src/config.py`) |
|---|---|
| 대상 | 나이 구간 하한 ≥ 65, 5세 구간(65, 70, 75, 80, 85+) |
| 공간 | 출발: 경기·인천 / 도착: 서울 행정동(→자치구). 행안부·통계청 코드 자동 판별 |
| 목적 | 이동유형 9개 → 도착지 기준 H(야간상주지), W(주간상주지), E(그 외) |
| 시간 | 주중/주말 × 낮밤(07–18시) 또는 세분 시간대(06–09, 09–12, 12–15, 15–18, 18–21, 21–06) |
| 비식별 `*` | 1.5로 대체(`MASKED_VALUE`), 0/2로 바꿔 민감도 분석 |

## 변수 체계
| 수준 | 변수 | 만드는 곳 |
|---|---|---|
| 코호트(개인) | 목적, 5세 연령, 성별, 주중/주말, 시간대 | 생활이동 원자료 |
| 출발–도착 | 평균 이동시간(`z_od_travel`, 거리 대용) | 생활이동 원자료 |
| 도착 행정동 `d_` | 반경 500m 내 시설 수(`acc_*`, 100m 격자에서 고령인구 가중 평균), 최근접 지하철·병원 거리(`dist_*`), 1km²당 밀도(`dens_*`), 고령인구 비율 | `src/regional.py` |
| 도착 자치구 `g_` | 위와 같은 변수의 구 집계 | `src/regional.py` |
| 출발 시군구 `o_` | 시설 밀도(`dens_*`) | `src/regional.py` |

시설 범주(`POI_RULES`): hospital, clinic, pharmacy, market, welfare, leisure, religion, park, subway.
업종명 키워드로 분류하며, 못 한 업종은 `poi_uncategorized.csv`로 빠진다(수작업·LLM 분류 대상).

### 지역변수 원자료 (네트워크·키 필요)
| 범주 | 자료 | 형식 |
|---|---|---|
| 상가 업종 전반 | 소상공인시장진흥공단 상가(상권)정보 | CSV (소분류명, 경도, 위도) → `poi/store.csv` |
| 병원·의원 | 건강보험심사평가원 병원정보서비스 | CSV (종별코드명, XPos, YPos) |
| 노인복지시설 | 서울 열린데이터 노인여가복지시설 | 좌표 CSV → `poi/welfare.csv` |
| 지하철역 | 서울 열린데이터 역 좌표 | 좌표 CSV → `poi/subway.csv` |
| 공원·종교 | OpenStreetMap (Geofabrik) | 좌표 CSV |
| 100m 격자인구 | SGIS 격자통계(총인구·65세 이상) | x, y(EPSG:5179 중심점), pop_total, pop_65p |

## 이전 연구에서 모형이 막힌 원인과 수정
| 문제 | 원인 | 수정 |
|---|---|---|
| 다중공선성(식별 불가) | 지역변수를 **(도착 − 출발) 차이**로 만들면 출발지·도착지 효과의 선형결합. 출발지·도착지를 함께 넣으면 지역변수 수만큼 rank 결손(5개 → 결손 5) | 출발(`o_`)·도착(`d_`,`g_`) 변수를 **따로**, 지역 식별자는 **무선효과** |
| 상위 단위 부족 | 도착 25개 구에 지역변수 5개 이상 | 도착을 행정동 427개로. 같은 17개 변수의 VIF가 구 단위 최대 28.9 → 동 단위 모두 5 미만(합성자료) |
| log(이동량)에 음이항 | 음이항은 가산자료 원값 모형 | 원값 `y`에 NB2(log link) |
| 주중 5일/주말 2일 사후 보정 | 셀마다 관측 일수가 다름 | `offset(log(일수 × 시간))` |
| 양수 셀만 분석 | 0 셀 누락 → 선택편의 | 코호트 격자 전체를 0으로 채움 |
| W/E 분리모형 비교 | 모형 간 계수 차이 검정 불가 | 통합모형 + 목적 × 변수 상호작용, 목적별 조건부 IRR |
| RF_NEW 설명력 83.8% | 차이 변수가 출발–도착 쌍을 사실상 식별 | 지역효과 해석은 다수준 모형 계수로 |

## 모형
**동 단위 (`R/mlm_dong.R`)** — L1 코호트 셀 ⊂ L2 도착 행정동 ⊂ L3 자치구, 출발지 합산

| 모형 | 구성 | 보는 것 |
|---|---|---|
| D0 | `(1|구) + (1|동)` | 동·구 수준 분산 분해(VPC) |
| D1 | + 목적·연령·성별·주중주말·시간대·이동시간 | 개인·시간 효과 |
| D2 | + 목적 × (연령·성별·주중주말·시간대) | 목적별 시간 패턴 |
| D3 | `(0+목적|구) + (0+목적|동)` | 목적마다 구·동 편차의 크기, 목적 간 지역 상관 |
| D4 | + `d_` 지역변수 + 목적 × `d_` | 목적별 동 흡인요인 → `dong_purpose_specific_regional_irr.csv` |

**구 단위 (`R/mlm_gu.R`)** — 출발 시군구 × 도착 자치구 교차분류 + 출발–도착 쌍. 출발지 효과(push)를 볼 때 사용.
