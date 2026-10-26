"""전처리: DB 테이블 → features.csv (군집 모델 입력 파생변수 5개)

대상: 47개 행정동 (강남구 22 + 춘천시 25). 유동인구는 2025년 7~12월(6개월) 합산.

| 축     | 파생변수              | 정의                                              | 입력 테이블                  |
|--------|-----------------------|---------------------------------------------------|------------------------------|
| 인구축 | 1-1 HH1_EXPO_5060     | 1인 세대 비율 × 5060 인구 비중                     | HH_SIZE, POP_SEX_AGE         |
| 인구축 | 1-2 MALE_SHARE_5060   | 5060 인구 중 남성 비율                             | POP_SEX_AGE                  |
| 인구축 | 1-3 VISIBLE_IDX_50P   | (유동인구 중 5060 비중) ÷ (10세 이상 주민 중 5060 비중) | FLOW_POP_SEX_AGE, POP_SEX_AGE |
| 지역축 | 2-1 TIME_ENTROPY      | 하루 유동인구의 시간대(0~23시) 분포 엔트로피       | FLOW_POP_TIME                |
| 지역축 | 2-2 POP_DENSITY       | 총인구 ÷ 면적(km²)                                 | EMD_MST                      |

결과 열: 동 정보(EMD_CD, SGG_CD, SGG_NM, EMD_NM, AREA_KM2, TOT_POP_CNT) + POP_5060 + 파생변수 원값 5개 (반올림 없음)
이 파일부터 modeling/ 스크립트가 시작한다.

입력: database/ 의 SGG_MST, EMD_MST, POP_SEX_AGE, HH_SIZE (저장소 포함)
      + FLOW_POP_SEX_AGE, FLOW_POP_TIME (prep_flow.py 산출물, 저장소에 포함하지 않음)

실행: python preprocessing/prep_features.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 경로는 저장소 루트(Knock-Model/) 기준 → 어느 폴더에서 실행해도 같은 위치를 가리킨다
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'database'

INPUTS = ['SGG_MST', 'EMD_MST', 'POP_SEX_AGE', 'HH_SIZE', 'FLOW_POP_SEX_AGE', 'FLOW_POP_TIME']
CODE_COLS = {'SGG_CD': str, 'EMD_CD': str, 'STD_YM': str, 'BLOCK_CD': str}
VARS = ['HH1_EXPO_5060', 'MALE_SHARE_5060', 'VISIBLE_IDX_50P', 'TIME_ENTROPY', 'POP_DENSITY']
OLD = ['60대', '70대', '80대', '90대', '100세이상']


# ---- 실행 옵션 · 검증 · 저장 ----
def parse_args(description):
    """--dry-run 이면 저장하지 않고 검증·기존 파일 비교만 수행."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dry-run', action='store_true',
                        help='결과를 저장하지 않고 검증과 기존 파일 비교만 수행')
    return parser.parse_args()


def require(*paths):
    """입력 파일이 없으면 어디에 무엇을 넣어야 하는지 알려주고 종료."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print('[중단] 입력 파일이 없습니다:')
        for p in missing:
            print('   -', Path(p).relative_to(ROOT))
        print('README의 "데이터 준비"를 참고해 파일을 넣은 뒤 다시 실행하세요.')
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


def save(df, name, dtype=None, dry_run=False):
    """DB/<name>.csv 로 저장. 같은 이름의 파일이 있으면 저장 전에 내용을 비교해 알려준다.

    형식: CSV, utf-8-sig, index 없음. dtype은 기존 파일을 읽을 때 코드값을 문자열로 맞추는 용도.
    """
    path = DB / f'{name}.csv'
    if path.exists():
        old = pd.read_csv(path, dtype=dtype, encoding='utf-8-sig', low_memory=False)
        try:
            pd.testing.assert_frame_equal(old.reset_index(drop=True), df.reset_index(drop=True),
                                          check_dtype=False)
            status = '기존 파일과 동일'
        except AssertionError as e:
            status = f'기존 파일과 다름 → {str(e).splitlines()[0]}'
    else:
        status = '새 파일'
    if not dry_run:
        df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'   {name:18s} {str(df.shape):14s} | {status} | {"저장 안 함(--dry-run)" if dry_run else "저장"}')


# ---- 읽기 · 파생변수 ----
def load_tables():
    """DB 테이블 읽기. 유동인구에는 47개 동이 아닌 이웃 동 행이 약 0.1% 섞여 있어 47개 동만 남긴다."""
    t = {name: pd.read_csv(DB / f'{name}.csv', dtype=CODE_COLS, encoding='utf-8-sig', low_memory=False)
         for name in INPUTS}
    emd = set(t['EMD_MST'].EMD_CD)
    for name in ['FLOW_POP_SEX_AGE', 'FLOW_POP_TIME']:
        n = len(t[name])
        t[name] = t[name][t[name].EMD_CD.isin(emd)]
        print(f'   {name:18s} {n:>9,}행 → 47개 동 밖 제외 {n - len(t[name]):>5,}행 → {len(t[name]):>9,}행')
    return t


def build_features(t):
    emd_mst, pop, hh = t['EMD_MST'], t['POP_SEX_AGE'], t['HH_SIZE']
    flow_sa, flow_tm = t['FLOW_POP_SEX_AGE'], t['FLOW_POP_TIME']

    feat = emd_mst.set_index('EMD_CD')[['SGG_CD', 'EMD_NM', 'AREA_KM2', 'TOT_POP_CNT']].copy()
    feat.insert(1, 'SGG_NM', feat.SGG_CD.map(dict(zip(t['SGG_MST'].SGG_CD, t['SGG_MST'].SGG_NM))))

    # ---------- 인구축 ----------
    P = pop.pivot_table(index='EMD_CD', columns=['SEX_CCD', 'AGE_CCD'], values='POP_CNT', aggfunc='sum')
    PA = pop.pivot_table(index='EMD_CD', columns='AGE_CCD', values='POP_CNT', aggfunc='sum')
    H = hh.pivot_table(index='EMD_CD', columns='HH_SIZE_CCD', values='HH_CNT', aggfunc='sum')
    pop5060 = PA['50대'] + PA['60대']
    feat['POP_5060'] = pop5060

    # 1-1 중장년 1인 세대 노출도
    feat['HH1_EXPO_5060'] = (H['1인'] / H.sum(axis=1)) * (pop5060 / PA.sum(axis=1))

    # 1-2 5060 남성 비율
    feat['MALE_SHARE_5060'] = (P['남성']['50대'] + P['남성']['60대']) / pop5060

    # 1-3 중장년 생활 노출 지수
    FA = flow_sa.groupby('EMD_CD')[[c for c in flow_sa.columns if 'FLOW_POP_CNT' in c]].sum()
    f50 = FA.MAN_FLOW_POP_CNT_50G + FA.WMAN_FLOW_POP_CNT_50G
    f60gu = FA.MAN_FLOW_POP_CNT_60GU + FA.WMAN_FLOW_POP_CNT_60GU
    r_d = PA['60대'] / PA[OLD].sum(axis=1)                       # 주민 60세+ 중 60대 비율
    flow5060 = f50 + f60gu * r_d                                   # 60GU 중 60대만 추정
    resident_10p = PA.drop(columns='10대미만').sum(axis=1)          # 유동인구에 10세 미만이 없으므로 분모 맞춤
    feat['VISIBLE_IDX_50P'] = (flow5060 / FA.sum(axis=1)) / (pop5060 / resident_10p)

    # ---------- 지역축 ----------
    # 2-1 생활 리듬 다양성: 시간대별 유동 비중 p_h → 엔트로피 -Σ p_h log p_h (최댓값 log 24 ≈ 3.178)
    T = flow_tm.groupby('EMD_CD')[[f'TMST_{h:02d}' for h in range(24)]].sum()
    p_h = T.div(T.sum(axis=1), axis=0)
    feat['TIME_ENTROPY'] = -(p_h * np.log(p_h)).sum(axis=1)

    # 2-2 물리적 밀집도: 총인구 ÷ 면적
    feat['POP_DENSITY'] = feat.TOT_POP_CNT / feat.AREA_KM2

    return feat.reset_index()


def validate(feat):
    v = feat[VARS]
    check(f'features {feat.shape}', {
        '행정동 47개 (강남 22 + 춘천 25)': len(feat) == 47 and feat.EMD_CD.is_unique
            and feat.SGG_NM.value_counts().to_dict() == {'춘천시': 25, '강남구': 22},
        '결측 없음': int(feat.isna().sum().sum()) == 0,
        '무한대 없음': int(np.isinf(v).sum().sum()) == 0,
        '파생변수 양수': bool((v > 0).all().all()),
    })
    print(v.describe().loc[['min', 'mean', 'max']].T.to_string(float_format=lambda x: f'{x:,.4f}'))


def main():
    args = parse_args(__doc__.splitlines()[0])
    require(*[DB / f'{name}.csv' for name in INPUTS])

    section('1. 읽기 (database/)')
    t = load_tables()

    section('2. 파생변수 5개 · 검증')
    feat = build_features(t)
    validate(feat)

    section(f'3. 저장 ({DB.name}/)')
    save(feat, 'features', dtype={'SGG_CD': str, 'EMD_CD': str}, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
