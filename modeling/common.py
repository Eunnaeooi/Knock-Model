"""모델링 공통 설정 — 경로, 변수 정의, 실행 옵션·검증·저장, 입력 데이터(단계 점수·z-score), 군집 함수.

모든 스크립트는 database/features.csv (preprocessing/prep_features.py 산출물)에서 시작한다.

    Knock-Model/
    ├─ database/features.csv     ← 입력 (47개 동 × 파생변수 5개)
    ├─ modeling/                 ← 이 파일 + tune → cluster → validate → xai_explain
    └─ artifacts/                ← 결과 CSV, figures/, tuning/
"""
import argparse
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'database'
ART = ROOT / 'artifacts'
FIG = ART / 'figures'
TUNING_FILE = ART / 'tuning' / 'tuning_results.csv'
FEATURES_FILE = DB / 'features.csv'

SEED = 42
REGIONS = ['강남구', '춘천시']

# ---- 변수 정의 ----
POP_AX = ['HH1_EXPO_5060', 'MALE_SHARE_5060', 'VISIBLE_IDX_50P']
REG_AX = ['TIME_ENTROPY', 'POP_DENSITY']
VARS = POP_AX + REG_AX
QCOLS = [f'{k}_Q' for k in VARS]
RISK_UP = {'HH1_EXPO_5060': True, 'MALE_SHARE_5060': True, 'VISIBLE_IDX_50P': False,
           'TIME_ENTROPY': False, 'POP_DENSITY': False}             # True = 높을수록 위험
SHORT = {'HH1_EXPO_5060': '1-1 1인 노출', 'MALE_SHARE_5060': '1-2 남성 비율', 'VISIBLE_IDX_50P': '1-3 생활 노출',
         'TIME_ENTROPY': '2-1 생활 리듬', 'POP_DENSITY': '2-2 인구밀도'}

# ---- 튜닝 설정 ----
ALGOS = ['K-means', 'Ward', 'Average', 'GMM']
KS = [3, 4, 5, 6]
N_BOOT = 100                                     # 부트스트랩 안정성 반복 횟수
N_INIT_BASE = 50                                 # 기본 모델·K 선택의 K-means 초기값 수
N_INIT_TUNE = 30                                 # 튜닝·최종 모델·민감도의 K-means 초기값 수 (64개 조합 공통)
MIN_CLUSTER = 4                                  # 사용 가능 모델의 최소 군집 크기 (동 수)
BASE = ('2D 단계점수', 'K-means', 4)              # 기본 모델 (논문 방식): 입력공간, 알고리즘, K
MID = 3.0                                        # 단계 점수(1~5) 축의 중앙 → 군집 이름(H/L) 기준


# ---- 실행 옵션 · 검증 · 저장 ----
def parse_args(description):
    """--dry-run 이면 저장하지 않고 검증·기존 파일 비교만 수행."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dry-run', action='store_true',
                        help='결과를 저장하지 않고 검증과 기존 파일 비교만 수행')
    return parser.parse_args()


def require(*paths):
    """입력 파일이 없으면 무엇을 먼저 실행해야 하는지 알려주고 종료."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print('[중단] 입력 파일이 없습니다:')
        for p in missing:
            print('   -', Path(p).relative_to(ROOT))
        print('README의 "실행 순서"를 참고해 앞 단계를 먼저 실행하세요.')
        sys.exit(1)


def section(title):
    print(f'\n=== {title}')


def check(name, checks):
    """검증 항목을 출력하고, 하나라도 실패하면 중단."""
    print(f'[{name}]')
    for label, ok in checks.items():
        print(f'   {"OK  " if ok else "실패"} {label}')
    if not all(checks.values()):
        sys.exit(f'[중단] {name} 검증 실패')


def save(df, path, dtype=None, float_format=None, dry_run=False):
    """CSV 저장 (utf-8-sig, index 없음). 같은 이름의 파일이 있으면 저장될 내용과 비교해 알려준다.

    비교는 '저장될 CSV를 다시 읽은 표'와 '기존 파일을 읽은 표'로 한다 → 반올림·float_format까지 반영.
    dtype은 코드값(EMD_CD 등)을 문자열로 읽기 위한 용도.
    """
    path = Path(path)
    text = df.to_csv(index=False, float_format=float_format)
    if path.exists():
        old = pd.read_csv(path, dtype=dtype, encoding='utf-8-sig')
        new = pd.read_csv(io.StringIO(text), dtype=dtype)
        try:
            pd.testing.assert_frame_equal(old, new, check_dtype=False)
            status = '기존 파일과 동일'
        except AssertionError as e:
            status = f'기존 파일과 다름 → {" ".join(str(e).split())[:160]}'
    else:
        status = '새 파일'
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    print(f'   {str(path.relative_to(ROOT)):40s} {str(df.shape):9s} | {status} | '
          f'{"저장 안 함(--dry-run)" if dry_run else "저장"}')


# ---- 입력 데이터 ----
def quintile_score(s, risk_up=True):
    """시군구 내 5분위 점수 (1~5). 위험 방향이 아닌 변수는 뒤집는다."""
    r = s.rank(method='first')
    q = pd.qcut(r, 5, labels=False) + 1
    return q if risk_up else 6 - q


def load_data():
    """features.csv → 단계 점수·축 점수(feat), 5분위 점수(Qr), 시군구 내 z-score(Zr).

    - feat: 동 정보 + 원값 5개 + {변수}_Q (1~5) + POP_SCORE·REG_SCORE (축별 단계 점수 평균, 1~5)
    - Qr  : 변수별 5분위 점수 (float)
    - Zr  : 시군구 내 z-score, 위험 방향 통일 (클수록 위험). 인구밀도는 log 후 표준화
    - Zr_raw_dens: 인구밀도를 log 없이 표준화한 값 (민감도 분석용)
    """
    require(FEATURES_FILE)
    feat = pd.read_csv(FEATURES_FILE, dtype={'EMD_CD': str, 'SGG_CD': str}, encoding='utf-8-sig').set_index('EMD_CD')

    S = pd.DataFrame(index=feat.index)
    for k in VARS:
        S[k] = feat.groupby('SGG_CD')[k].transform(lambda s: quintile_score(s, RISK_UP[k]))
    feat['POP_SCORE'] = S[POP_AX].mean(axis=1)     # 인구축 점수 (1~5)
    feat['REG_SCORE'] = S[REG_AX].mean(axis=1)     # 지역축 점수 (1~5)
    S.columns = QCOLS
    feat = feat.join(S)

    Zr = pd.DataFrame({k: feat.groupby('SGG_CD')[k].transform(lambda s: (s - s.mean()) / s.std()) * (1 if RISK_UP[k] else -1)
                       for k in VARS})
    Zr_raw_dens = Zr.POP_DENSITY.copy()
    Zr['POP_DENSITY'] = feat.groupby('SGG_CD').POP_DENSITY.transform(lambda s: -(np.log(s) - np.log(s).mean()) / np.log(s).std())
    Qr = pd.DataFrame({k: feat[f'{k}_Q'].astype(float) for k in VARS})

    check(f'입력 데이터 {feat.shape}', {
        '행정동 47개': len(feat) == 47,
        '결측·무한대 없음': bool(np.isfinite(feat[VARS]).all().all()) and bool(np.isfinite(Zr).all().all()),
        '시군구 내 동점 없음 (5분위 순서 의존 없음)':
            all(int(feat.groupby('SGG_CD')[k].apply(lambda s: s.duplicated().sum()).sum()) == 0 for k in VARS),
    })
    return SimpleNamespace(feat=feat, Qr=Qr, Zr=Zr, Zr_raw_dens=Zr_raw_dens)


def axis_scores(d, kind, pop_vars=POP_AX, reg_vars=REG_AX, src=None):
    """두 축 점수 = 축에 속한 변수의 평균 (kind: 'q' = 5분위 단계점수, 'z' = 시군구 내 z-score)"""
    src = (d.Qr if kind == 'q' else d.Zr) if src is None else src
    return src[pop_vars].mean(axis=1), src[reg_vars].mean(axis=1)


def input_spaces(d):
    """튜닝 입력 공간 4가지 (위험 방향 통일: 값이 클수록 위험)"""
    return {
        '2D 단계점수': np.c_[axis_scores(d, 'q')],
        '2D z-score':  np.c_[axis_scores(d, 'z')],
        '5D 단계점수': d.Qr[VARS].values,
        '5D z-score':  d.Zr[VARS].values,
    }


def space_kind(space):
    return 'z' if 'z-score' in space else 'q'


# ---- 군집 ----
def fit_base(X):
    """기본 모델: 2D 단계 점수 + K-means (K=4, 초기값 50회)"""
    return KMeans(n_clusters=BASE[2], n_init=N_INIT_BASE, random_state=SEED).fit(X)


def fit_predict(algo, Xs, k, seed=SEED):
    """튜닝 공통 학습 함수 (K-means는 초기값 30회)"""
    if algo == 'K-means':
        return KMeans(n_clusters=k, n_init=N_INIT_TUNE, random_state=seed).fit_predict(Xs)
    if algo == 'Ward':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xs)
    if algo == 'Average':
        return AgglomerativeClustering(n_clusters=k, linkage='average').fit_predict(Xs)
    if algo == 'GMM':
        return GaussianMixture(n_components=k, covariance_type='diag', n_init=5, random_state=seed).fit_predict(Xs)
    raise ValueError(algo)


def name_clusters(cent, mid=MID):
    """중심점이 축 중앙(mid) 이상이면 H, 미만이면 L (첫 글자 = 인구축, 둘째 = 지역축).
    같은 사분면에 군집이 2개 이상이면 두 축 합이 큰 순서로 번호를 붙인다 (예: HH1 = 더 위험)"""
    name = pd.Series([('H' if p >= mid else 'L') + ('H' if r >= mid else 'L') for p, r in cent.values], index=cent.index)
    for q, idx in name.groupby(name).groups.items():
        if len(idx) > 1:
            rk = cent.loc[idx].sum(axis=1).rank(ascending=False, method='first').astype(int)
            name[idx] = [f'{q}{r}' for r in rk]
    return name


def group_order(names):
    return sorted(set(names), key=lambda s: (s[0] != 'H', s[1] != 'H', s))


def surrogate():
    """대리 모델: 표본이 47개라 과적합을 막기 위해 얕은 나무(max_depth=3)·잎 최소 3개"""
    return RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=3, random_state=SEED)


def shap_inputs(d):
    """SHAP·대리 모델 입력: 시군구 내 z-score (위험 방향 통일), 열 이름은 보고서용 짧은 이름"""
    X = d.Zr[VARS].copy()
    X.columns = [SHORT[k] for k in VARS]
    return X
