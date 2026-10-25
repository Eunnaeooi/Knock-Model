# 노크 — 사회적 고립 예방 및 대응을 위한 AI 에이전트

제14회 2026 빅콘테스트 · AI데이터활용분야 · 팀 노크

강남구 22개 동과 춘천시 25개 동, 총 47개 행정동을 대상으로 분석했습니다. 주민등록, SKT 유동인구, 신한카드 소비 데이터로 중장년(5060) 사회적 고립 위험을 군집화하고, 그 결과를 AI 에이전트의 입력으로 사용합니다.

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
