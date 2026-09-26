# 서울 유입 고령인구 생활이동 — 목적별 다수준 분석

서울열린데이터광장 **생활이동 데이터**(KT 통신 기반)를 이용해 서울 외 지역에서 서울로 들어오는
65세 이상 인구의 이동을 목적별로 분석한다.

## 파이프라인
```bash
pip install -r requirements.txt
# 1) 원자료 CSV를 data/raw/ 에 둔다 (최근 1개월 샘플)
python -m src.preprocess          # → data/processed/elderly_inflow.parquet
python -m src.model               # → outputs/*.csv
```
합성 데이터로 검증:
```bash
python scripts/make_synthetic.py
python -m src.preprocess --raw data/synthetic --out data/processed/synthetic.parquet
python -m src.model --in data/processed/synthetic.parquet --out outputs/synthetic
```

## 정의
| 항목 | 정의 (`src/config.py`) |
|---|---|
| 고령 | 나이 구간 하한 ≥ 65 (65–74 / 75+ 구분) |
| 서울 유입 | 도착 코드 `11*` & 출발 코드 ≠ `11*` |
| 목적 | `이동목적` 컬럼이 있으면 그대로, 없으면 `이동유형` 도착 측(H→귀가, W→출근, E→기타활동) |
| 비식별 셀 `*` | 1.5로 대체, `masked` 플래그로 표시 |

## 모형
교차분류 다수준 포아송 GLMM (statsmodels `PoissonBayesMixedGLM`, 변분 베이즈):

    y[o,d,p] ~ Poisson(λ),  log λ = β0 (+ β_p) + u_출발시군구 + v_도착자치구 + w_도착행정동

- 목적별 모형: 수준별 분산성분·VPC 비교 → `outputs/vc_by_purpose.csv`
- 통합모형: 목적 고정효과(IRR) → `outputs/pooled_fixed_effects.csv`
- 수준별 랜덤효과(BLUP) → `outputs/random_effects_by_purpose.csv`
- 기술통계 → `outputs/desc_*.csv`

한계: 과산포는 반영하지 않는다(음이항 GLMM은 R `glmmTMB`로 교차검증 권장). OD 격자는 관측된 출발지·도착지 조합으로만 만든다.
