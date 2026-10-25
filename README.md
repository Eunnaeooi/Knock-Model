# 노크 — 사회적 고립 예방 및 대응을 위한 AI 에이전트

제14회 2026 빅콘테스트 · AI데이터활용분야 · 팀 노크

강남구 22개 동과 춘천시 25개 동, 총 47개 행정동을 대상으로 분석했습니다. 주민등록, SKT 유동인구, 신한카드 소비 데이터로 중장년(5060) 사회적 고립 위험을 군집화하고, 그 결과를 AI 에이전트의 입력으로 사용합니다.

## 관련 저장소

| 저장소 | 내용 |
|---|---|
| **Knock-Model** (이 저장소) | 데이터 전처리 · EDA · 군집 모델링 → 결과 파일(`artifacts/`) |
| [**knock**](https://github.com/OvO-Hi/knock) | AI 에이전트 — `artifacts/`의 군집 결과·판별 근거를 입력으로 사용 |

## 주요 결과

- **최종 모델**: 인구축·지역축 2차원 시군구 내 z-score + K-means, **K=5** (64개 조합 튜닝 후 선택, 실루엣 0.435)
- **군집 5개**: HH1 6 · HH2 10 · HL 11 · LH 11 · LL 9개 동
  - **HH1 (극고위험)**: 춘천 남면·서면·동산면·남산면·북산면, 강남 삼성1동
- **위험도 5단계 (최우선 개입, 10개 동)**: 춘천 서면·남면·동산면·북산면·남산면·신동면·사북면, 강남 삼성1동·역삼1동·논현2동
- **군집을 가르는 핵심 변수 (SHAP)**: 2-2 인구밀도(0.345)·1-3 생활 노출(0.326) > 2-1 생활 리듬(0.256) > 1-1 1인 노출(0.230) > 1-2 남성 비율(0.152)
  - 대리 모델(랜덤포레스트) 정확도: 반복 5-fold × 20 기준 0.71

결과 파일은 `artifacts/`에 있습니다 (`dong_cluster.csv`: 47개 동 × 군집·위험도·판별 근거).

## 방법 요약

```
원본 데이터 ─▶ 전처리 (preprocessing/) ─▶ DB 테이블 (database/)
                                              │
               EDA (eda/) ◀──────────────────┤  파생변수 5개 설계 근거
                                              ▼
               군집 모델링 (modeling/) ─▶ 군집 · 위험도 5단계 · 검증 · SHAP ─▶ artifacts/
```

| 축 | 파생변수 | 정의 | 위험 방향 |
|---|---|---|---|
| 인구축 | 1-1 `HH1_EXPO_5060` | 1인 세대 비율 × 5060 인구 비중 | 높을수록 |
| 인구축 | 1-2 `MALE_SHARE_5060` | 5060 인구 중 남성 비율 | 높을수록 |
| 인구축 | 1-3 `VISIBLE_IDX_50P` | (유동인구 중 5060 비중) ÷ (주민 중 5060 비중) | 낮을수록 |
| 지역축 | 2-1 `TIME_ENTROPY` | 하루 유동인구의 시간대(0~23시) 분포 엔트로피 | 낮을수록 |
| 지역축 | 2-2 `POP_DENSITY` | 총인구 ÷ 면적(km²) | 낮을수록 |

1. **변수 표준화**: 강남·춘천은 인구밀도·상권 구조가 크게 달라, 모든 변수를 **시군구 안에서** 5분위 단계 점수(1~5점)와 z-score로 바꿉니다.
2. **두 축 점수**: 인구축(그 동에 고립되기 쉬운 중장년이 얼마나 있는가)과 지역축(사람을 만나기 어려운 환경인가)을 각 축 변수의 평균으로 만듭니다. 참고 논문(김소연 외, 2024)의 2차원 군집 방식을 따랐습니다.
3. **튜닝**: 입력 공간 4 × 알고리즘 4(K-means, Ward, Average, GMM) × K 4(3~6) = 64개 조합을 실루엣·Davies-Bouldin·부트스트랩 안정성으로 비교합니다.
4. **군집 이름**: 두 축 점수의 중앙(3점) 기준으로 HH / HL / LH / LL, 같은 사분면이 둘이면 더 위험한 쪽을 1번으로 둡니다.
5. **위험도 5단계**: 두 축 단계 점수의 합(2~10점)을 47개 동 기준 5분위로 나눕니다. 군집은 대응 **유형**, 위험도는 개입 **우선순위**로 씁니다.
6. **검증·설명**: 점수 방식·가중치·변수 구성을 바꾼 민감도 분석, 의사결정나무 판단 규칙, 대리 모델 + SHAP으로 동별 판별 근거를 만듭니다.

카드 소비 데이터는 가맹점 시군구 단위라 동별 군집에는 넣지 않고, EDA에서 지역·성·연령별 소비 특성(관계형 소비 비중 등)을 확인하는 데만 사용했습니다.

## 폴더 구성

| 폴더 | 내용 | 보고서 |
|---|---|---|
| `preprocessing/` | 원본 데이터를 DB 테이블로 가공하는 스크립트 4개 + 공통 설정 `common.py` | 5.2 |
| `database/` | ERD, 전처리 결과 테이블 (공공데이터로 만든 4개만 저장소에 포함) | 5.3 |
| `eda/` | `EDA.ipynb` — 데이터 품질, 인구·세대 구조, 카드 소비, 유동인구, 파생변수 EDA | 6~7 |
| `modeling/` | `Clustering.ipynb` — 단계 점수화, K 선택, 튜닝, 위험도 5단계, 검증, SHAP | 8~9 |
| `artifacts/` | 에이전트 입력 데이터 — 군집 결과(`dong_cluster.csv`, `cluster_profile.csv`), 검증·SHAP·민감도 표, 변수 설명서(`variable_dictionary.md`) | 10 |
| `public_data/` | 공공데이터 원본과 출처 목록 (`출처목록.md`), 행정구역 경계 `2. 경계/` | 5.1 |

## 빠른 시작

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python preprocessing/prep_admin_population.py
python preprocessing/prep_card.py
python preprocessing/prep_flow.py
python preprocessing/prep_flow_monthly.py
```

- 위 순서대로 실행합니다. `prep_flow_monthly.py`는 `prep_flow.py`의 결과를 읽으므로 반드시 그 다음에 실행합니다.
- 저장소 루트(`Knock-Model/`)에서 실행합니다. 경로는 `preprocessing/common.py`가 저장소 루트를 기준으로 잡기 때문에 다른 폴더에서 실행해도 같은 위치를 읽고 씁니다.
- `--dry-run`을 붙이면 저장하지 않고 검증과 기존 파일 비교만 합니다.
- 각 스크립트는 저장 전에 `database/`의 같은 이름 파일과 내용을 비교해 `기존 파일과 동일 / 다름 / 새 파일`을 출력합니다.

## 데이터 준비

| 위치 | 내용 | 저장소 포함 |
|---|---|---|
| `public_data/` | 주민등록 인구·세대, 행정동 면적, 행정동 코드 | 포함 |
| `public_data/2. 경계/` | 2025년 2분기 시도·시군구·행정동 경계 shp ([공공데이터포털](https://www.data.go.kr/data/15129688/fileData.do)) | 미포함 (용량) — 내려받아 이 위치에 두세요 |
| `private_data/2026빅콘테스트_신한카드_데이터_레이아웃/` | 신한카드 결제 데이터 (`신한카드_빅콘테스트2026_데이터1.txt`, `데이터2.txt`) | 미포함 (빅콘테스트 제공) |
| `private_data/2026빅콘테스트_sk통신인구_데이터_레이아웃/` | SKT 유동인구 월별 파일 18개 (`flow_{age,wkdy,time}_pop_2025MM.csv`) | 미포함 (빅콘테스트 제공) |

입력 파일이 없으면 스크립트가 없는 파일 목록을 출력하고 멈춥니다.

## 전처리 흐름과 기대 결과

| 스크립트 | 입력 | 만드는 테이블 (`database/`) | 기대 행 수 |
|---|---|---|---|
| `prep_admin_population.py` | `public_data/` | `SGG_MST`, `EMD_MST`, `POP_SEX_AGE`, `HH_SIZE` | 2 / 47 / 1,034 / 470 |
| `prep_card.py` | `private_data/…신한카드…/` | `CARD_SPEND_TIME`, `CARD_SPEND_RESID` | 1,044,710 / 1,806,800 |
| `prep_flow.py` | `private_data/…sk통신인구…/`, `2. 경계/` | `FLOW_POP_SEX_AGE`, `FLOW_POP_WKDY`, `FLOW_POP_TIME` | 595,903 / 720,153 / 570,463 |
| `prep_flow_monthly.py` | `FLOW_POP_*` | `flow_age_monthly`, `flow_day_monthly`, `flow_time_monthly` | 282 / 282 / 282 |

`prep_flow.py` 참고: SKT `202512` 요일별·시간대별 파일은 모든 행이 두 번씩 들어 있어 통합 후 완전 중복 행을 제거합니다(요일별 121,267행, 시간대별 92,545행).

이후 `eda/EDA.ipynb` → `modeling/Clustering.ipynb` 순서로 실행합니다. 두 노트북은 `database/`의 테이블을 읽습니다.
- VS Code에서는 노트북 오른쪽 위 **커널 선택**에서 `.venv` 환경을 고른 뒤 **모두 실행**합니다. 경로는 노트북이 저장소 루트를 찾아 `database/`로 맞춥니다 (Colab에서는 Drive의 `빅콘/DATA/DB/`).
- 그래프 한글 표시에 나눔고딕(NanumGothic) 글꼴이 필요합니다.
- `Clustering.ipynb`는 튜닝(64개 조합 × 부트스트랩 100회) 때문에 실행에 10분 이상 걸립니다.

## 저장소에서 제외한 데이터

- **주최 측 제공 데이터와 그 가공본**: `private_data/`, `CARD_SPEND_*`, `FLOW_POP_*`, `flow_*_monthly`. 대회 데이터 이용 규정에 따라 재배포하지 않으며, 대회가 끝나면 폐기합니다.
- **용량이 큰 공공데이터**: `public_data/2. 경계/`. 출처는 `public_data/출처목록.md`에 적었습니다.

## 실행 환경

- Python 3.9 (Windows 11에서 확인)
- 라이브러리 버전은 `requirements.txt`에 있습니다.
