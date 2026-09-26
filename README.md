# 수도권 고령인구는 왜 서울에 오는가 — 서울생활이동 목적별 다수준 분석

서울열린데이터광장 **서울생활이동 데이터**(시군구 단위)로 경기·인천에서 서울 25개 구로 들어오는
65세 이상 인구의 이동을 목적(H/W/E)별로 다수준 분석한다.
이전 청년(20–34세) 연구(「수도권 청년들은 어떤 이유로 서울에 오는가」, 2022.08 자료)의 설계를
이어받되, 그때 막혔던 다수준 모형을 식별 가능한 형태로 다시 구성했다.

## 파이프라인
```bash
pip install -r requirements.txt            # Python: 전처리·코호트
apt-get install r-base-core r-cran-glmmtmb r-cran-data.table   # R: 다수준 모형

# 원자료 CSV(시간대별 파일 전부)를 data/raw/ 에, 지역변수를 data/regional/regional_vars.csv 에 둔다
python -m src.preprocess                   # → data/processed/elderly_inflow.parquet
python -m src.cohort                       # → data/processed/cohort.csv, outputs/desc_*.csv, vif_*.csv
Rscript R/mlm_glmmtmb.R data/processed/cohort.csv outputs d_hosp,d_welfare,d_market
```
합성 데이터 검증: `python scripts/make_synthetic.py` 후 `--raw data/synthetic` 등으로 같은 순서 실행.

## 정의
| 항목 | 정의 (`src/config.py`) |
|---|---|
| 대상 | 나이 구간 하한 ≥ 65, 5세 구간(65, 70, 75, 80, 85+) |
| 공간 | 출발: 경기·인천 시군구 / 도착: 서울 25개 자치구 (행안부·통계청 코드 자동 판별) |
| 목적 | 이동유형 9개 → 도착지 기준 3개: H(야간상주지), W(주간상주지), E(그 외) |
| 시간 | 주중(wd)/주말(wk), 낮(07–18시)/밤 |
| 비식별 `*` | 1.5로 대체(`MASKED_VALUE`), 0/2로 바꿔 민감도 분석 |

## 이전 연구에서 모형이 막힌 원인과 수정
| 문제 | 원인 | 수정 |
|---|---|---|
| 다중공선성(식별 불가) | 지역변수를 **(도착 − 출발) 차이**로 만들면 출발지·도착지 효과의 선형결합이 된다. 출발지·도착지를 함께 넣으면 지역변수 수만큼 rank 결손(5개 → 결손 5) | 출발지(`o_`)·도착지(`d_`) 변수를 **따로** 넣고, 지역 식별자는 고정 더미가 아닌 **무선효과**로 |
| 수준 과다 | 도착 단위가 25개뿐인데 도착지 변수 5개 + 상호작용 | 도착지 변수는 3–4개로 제한, 수준별 VIF 점검(`outputs/vif_*.csv`) |
| log(이동량)에 음이항 | 음이항은 **가산자료 원값**에 쓰는 모형. log를 취하면 가정 위반 | 원값 `y`에 NB2(log link) |
| 주중 5일/주말 2일 사후 보정 | 셀마다 관측 일수가 다름 | `offset(log(일수 × 시간))` |
| 양수 셀만 분석 | 0 이동 셀이 빠져 선택편의 | 코호트 격자 전체를 0으로 채움 |
| W/E 분리모형 비교 | 모형 간 계수 차이를 검정할 수 없음 | 통합모형 + 목적 × 변수 상호작용 |

## 모형 (`R/mlm_glmmtmb.R`, glmmTMB NB2)
수준: L1 코호트 셀 ⊂ L2 출발–도착 쌍(`od`) ⊂ 출발 시군구 × 도착 자치구 (교차분류)

| 모형 | 구성 | 보는 것 |
|---|---|---|
| M0 | 빈 모형 + `(1|출발)+(1|도착)+(1|od)` | 수준별 분산 분해(VPC) |
| M1 | + 목적·연령·성별·주중주말·낮밤 | 개인·시간 효과(IRR) |
| M2 | + 목적 × (연령·성별·주중주말·낮밤) | 목적별 시간 패턴 차이 검정 |
| M3 | 무선기울기 `(0+목적|도착)`, `(0+목적|출발)` | 목적마다 어느 지역 수준이 편차를 만드는가, 목적 간 지역 상관 |
| M4 | + `o_`·`d_` 지역변수 + 목적 × `d_` 교차수준 상호작용 | 목적별 도착지 흡인요인 |

산출물: `mlm_model_comparison.csv`, `mlm_variance_components.csv`, `mlm_fixed_effects_irr.csv`,
`mlm_random_effects_{dest_gu,orig_sgg}.csv`, `mlm_dest_purpose_correlation.csv`.

## 지역변수 파일 형식
`data/regional/regional_vars.csv` — 출발·도착 시군구를 모두 포함, 원자료와 같은 코드체계
```
sgg_code,area_km2,hosp,welfare,market,subway
11110,23.9,...
```
`area_km2`가 있으면 1km²당 밀도로 바꾼 뒤 수준별로 표준화한다.
고령 후보 변수: 종합병원·의원, 노인복지관·경로당, 전통시장, 지하철역(65세 무임승차), 공원·종교시설.
