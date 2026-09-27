# 인수인계 — 서울 유입 고령인구 생활이동 목적별 다수준 분석

로컬 Claude Code에서 이어가기 위한 요약. 저장소: `tmd8929-cyber/claude_Choo`,
브랜치 `claude/awesome-mccarthy-hn9nn5`. 세부 사항은 `README.md`에 더 있음.

## 배경
- 이전 원고(HWP): 「수도권 청년들은 어떤 이유로 서울에 오는가」(2022.08 서울생활이동 시군구 자료,
  청년 20–34세). 랜덤포레스트 + 음이항회귀로 W(주간상주지)/E(그 외) 이동을 분석했으나,
  **다수준 모형은 다중공선성과 코드 오류로 완성하지 못함.**
- 이번 작업: 같은 자료로 **65세 이상 고령인구**의 서울 유입을 목적별(H/W/E)로 다수준 분석.
  막혔던 원인을 고쳐서 다수준 모형까지 완성하는 것이 목표.

## 막힌 원인과 수정 (README "이전 연구에서 모형이 막힌 원인과 수정" 표 참고)
| 문제 | 원인 | 수정 |
|---|---|---|
| 다중공선성 | 지역변수를 (도착−출발) **차이**로 만들어 출발지·도착지 효과와 완전 공선 | 출발(`o_`)·도착(`d_`,`g_`) 변수를 **따로**, 지역 식별자는 **무선효과**로 |
| 상위 단위 부족 | 도착 25개 구에 지역변수 5개 이상 넣음 | 도착을 **행정동(427개)** 으로 내림 → 동 단위 VIF 모두 5 미만(합성자료) |
| log(이동량)+음이항 | 가정 위반 | 원값에 NB2(log link) |
| 주중5/주말2 사후보정 | 셀마다 관측일수 다름 | `offset(log(일수×시간))` |
| 0 이동 셀 제외 | 선택편의 | 코호트 격자 전체를 0으로 채움 |
| W/E 분리모형 비교 불가 | 검정 불가 | 통합모형 + 목적×변수 상호작용, 목적별 조건부 IRR |

## 지금까지 만든 것
```
src/config.py      정의 전부 (고령기준, 코드 접두어, 목적 매핑, 시간대, POI 분류 규칙)
src/preprocess.py  원자료 → 고령·서울유입 필터, 목적 H/W/E, 5세연령, 주중주말×시간대
src/regional.py    행정동 경계 + 100m 격자 + 시설좌표 → 반경 접근성·밀도·최근접거리
src/cohort.py      코호트 격자(0 채움) + offset + 지역변수 부착(o_/d_/g_) + VIF
R/common.R         glmmTMB 공통 함수 (분산분해/VPC, IRR, 목적별 조건부 효과)
R/mlm_dong.R       도착 행정동⊂자치구 3수준 NB2 GLMM (D0~D4)
R/mlm_gu.R         출발 시군구×도착 자치구 교차분류 NB2 GLMM (M0~M4)
scripts/get_boundary.sh   행정동 경계 다운로드 (vuski/admdongkor, SGIS 원자료)
scripts/make_synthetic.py 합성 시설·인구·이동 생성 (검증용)
.env.example       필요한 키 이름 목록 (값 없음, 커밋 안전)
```

## 합성 데이터 검증 완료 (README "합성자료 검증" 표)
실제 행정동 경계 위에 효과를 심은 가짜 자료로 전체 파이프라인을 끝까지 돌림.
- 동 단위: E 목적×병원접근성 IRR 심은값 1.65 → 추정 1.55 (1.48–1.62). 효과 없는 변수 8개는 0.97–1.03으로 정상.
- 구 단위(25개)로 합치면 같은 효과가 사라짐 → **도착지는 반드시 동 단위로 분석**.
- 모든 모형(D0–D4, M0–M4) glmmTMB Hessian 정칙(pdHess=TRUE) 확인.

## 아직 안 된 것 / 필요한 것
1. **실제 원자료**: 서울생활이동(경기·인천→서울, 65세+, 최근 1개월), 시설 좌표, 100m 격자인구.
   이번 세션은 네트워크 정책상 `data.seoul.go.kr`, `sgis.kostat.go.kr`, `kosis.kr`, `data.go.kr` 등에 접속 불가.
   **로컬에서는 이 사이트들에 직접 접속 가능한지부터 확인.**
2. **API 키** (`.env.example` 참고, 로컬 `.env`에 채워서 사용):
   - `SGIS_KEY`/`SGIS_SECRET` — 필수(100m 격자 인구·고령인구)
   - `SEOUL_API_KEY` — 노인복지시설·지하철역 좌표
   - `DATA_GO_KR_KEY` — 심평원 병원정보(병원·의원 좌표)
   - `KOSIS_KEY` — 선택
   - SGIS 100m 격자는 API 키와 별도로 **자료제공 신청**(승인 며칠 소요)이 필요할 수 있음. 격자인구 없이도
     `src/regional.py`는 면적 가중으로 동작하니, 신청 넣어두고 먼저 진행 가능.
3. **행정동 경계**: `bash scripts/get_boundary.sh`로 GitHub에서 받음(로컬은 네트워크 문제 없을 것).
4. **POI 업종 분류 다듬기**: `POI_RULES`(config.py)가 규칙 기반 키워드 매칭이라 실제 상가정보 받으면
   `poi_uncategorized.csv`에 빠지는 업종이 많을 것. AI로 재분류해 규칙을 보강할 것.
5. **미실행**: 실제 데이터로 `src/regional.py` → `src/cohort.py` → `R/mlm_dong.R` 전체를 한 번도 못 돌려봄
   (전부 합성자료로만 검증). 실행 시간·수렴 여부를 실데이터로 재확인 필요.

## 로컬 실행 순서
```bash
git clone https://github.com/tmd8929-cyber/claude_Choo.git && cd claude_Choo
git checkout claude/awesome-mccarthy-hn9nn5
pip install -r requirements.txt
apt-get install r-base-core r-cran-glmmtmb r-cran-data.table   # 또는 R에서 install.packages

cp .env.example .env   # 키 채우기

bash scripts/get_boundary.sh
# 원자료(CSV) → data/raw/
# 시설좌표 → data/regional/poi/*.csv, 격자인구 → data/regional/grid_pop.csv (README 표 참고)

python -m src.preprocess
python -m src.regional
python -m src.cohort --dest-level dong --time-var timeband
Rscript R/mlm_dong.R data/processed/cohort_dong.csv outputs \
  "d_acc_hospital,d_acc_clinic,d_acc_welfare,d_acc_market,d_dist_subway"
```

## 로컬 Claude Code에 그대로 붙여넣을 프롬프트 (예시)
> claude_Choo 저장소의 claude/awesome-mccarthy-hn9nn5 브랜치야. HANDOFF.md와 README.md 읽고
> 이어서 진행해줘. [여기에 방금 받은 원자료/시설좌표 파일 경로, 또는 "원자료는 data/raw/에 넣어놨어" 등]
