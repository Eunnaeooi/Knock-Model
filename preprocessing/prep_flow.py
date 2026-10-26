"""전처리: SKT 유동인구 → FLOW_POP_SEX_AGE, FLOW_POP_WKDY, FLOW_POP_TIME + flow_age/day/time_monthly

대상: 서울 강남구(22개 동) + 강원 춘천시(25개 읍·면·동) = 47개 행정동, 2025년 7~12월(6개월)

| 만드는 테이블    | 원본 (월별 6개 파일)    | 단위                                         |
|------------------|-------------------------|----------------------------------------------|
| FLOW_POP_SEX_AGE | flow_age_pop_2025MM.csv | 월 × 50m 격자 × 성·연령(남녀 × 10대~60대이상) |
| FLOW_POP_WKDY    | flow_wkdy_pop_2025MM.csv| 월 × 50m 격자 × 요일(월~일)                   |
| FLOW_POP_TIME    | flow_time_pop_2025MM.csv| 월 × 50m 격자 × 시간대(00~23시)               |

| 만드는 월별 집계  | 입력 (위 테이블)  | 단위 (47개 동 × 6개월 = 282행)      |
|-------------------|-------------------|-------------------------------------|
| flow_age_monthly  | FLOW_POP_SEX_AGE  | 동 × 월 × 성·연령(12구간) 합계       |
| flow_day_monthly  | FLOW_POP_WKDY     | 동 × 월 × 요일(7구간) 합계           |
| flow_time_monthly | FLOW_POP_TIME     | 동 × 월 × 시간대(24구간) 합계        |

원본
  - private_data/2026빅콘테스트_sk통신인구_데이터_레이아웃/  (빅콘테스트 제공, 저장소에 포함하지 않음)
  - public_data/2. 경계/  행정동·시군구 경계 shp (2025년 2분기, EPSG:5179, 공공데이터포털)

처리 흐름
  1. 월별 파일 6개 통합 + 완전 중복 행 제거
     (202512 요일별·시간대별 파일은 모든 행이 두 번씩 들어 있음)
  2. 격자 → 행정동 배정: 격자 좌표를 전국 행정동 경계와 공간결합(point within polygon)
     - 성·연령별 자료의 고유 격자로 1차 매핑 → 요일별·시간대별에만 있는 격자를 추가 결합
     - 격자 코드 앞자리로 행정동을 추정하지 않음 (같은 격자 코드가 여러 동에 걸치는 경우가 있음)
  3. 대상 지역 필터: 강남구(11230)·춘천시(32010) 시군구 경계 안의 격자만 남김
     (경계 부근 격자 일부가 한남동·서초4동·가평읍 등 인접 시군구에 속함)
  4. 검증
  5. 행정동 × 월 합산 (EDA 입력) → 검증
     - 키: SIGUNGU_NM, EMD_CD, EMD_NM, STD_YM / 값: 각 유동인구 컬럼의 합계(sum)
     - flow_age_monthly의 '기준월'은 기존 파일 형식을 맞추기 위해 남긴 컬럼이다
       (행마다 STD_YM이 합산된 값이라 의미 없음). EDA에서 읽은 직후 삭제한다.
  6. 저장 (테이블 6개)

실행: python preprocessing/prep_flow.py [--dry-run]   (공간결합·비교 포함 수 분)
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# 경로는 저장소 루트(Knock-Model/) 기준 → 어느 폴더에서 실행해도 같은 위치를 가리킨다
ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / 'public_data' / '2. 경계'
SKT_RAW = ROOT / 'private_data' / '2026빅콘테스트_sk통신인구_데이터_레이아웃'
DB = ROOT / 'database'

TARGET_SGG = {'11230': '강남구', '32010': '춘천시'}   # 분석 대상 시군구 (통계청 5자리 코드)

DONG_SHP = BOUNDARY / '3. 2025년 2분기 기준 행정동 경계' / 'bnd_dong_00_2025_2Q.shp'
SGG_SHP = BOUNDARY / '2. 2025년 2분기 기준 시군구 경계' / 'bnd_sigungu_00_2025_2Q.shp'
MONTHS = [f'2025{m:02d}' for m in range(7, 13)]
CRS = 'EPSG:5179'
KEYS = ['BLOCK_CD', 'X_COORD', 'Y_COORD']          # 격자 식별 키 (격자 코드 + 좌표)

# 만드는 테이블: (원본 파일 접두어, 값 컬럼 접두어, 원본 월을 '기준월' 컬럼으로 추가할지)
JOBS = {
    'FLOW_POP_SEX_AGE': ('flow_age_pop', ('MAN_FLOW_POP_CNT_', 'WMAN_FLOW_POP_CNT_'), True),
    'FLOW_POP_WKDY':    ('flow_wkdy_pop', ('FLOW_POP_CNT_',), False),
    'FLOW_POP_TIME':    ('flow_time_pop', ('TMST_',), False),
}
CODE_COLS = {'BLOCK_CD': str, 'EMD_CD': str}

# 월별 집계: 만드는 테이블 → 입력 테이블 (값 컬럼 접두어는 JOBS와 같음)
MONTHLY = {'flow_age_monthly': 'FLOW_POP_SEX_AGE',
           'flow_day_monthly': 'FLOW_POP_WKDY',
           'flow_time_monthly': 'FLOW_POP_TIME'}
MONTHLY_KEYS = ['SIGUNGU_NM', 'EMD_CD', 'EMD_NM', 'STD_YM']


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


# ---- 통합 · 공간결합 ----


def raw_files(prefix):
    return [SKT_RAW / f'{prefix}_{ym}.csv' for ym in MONTHS]


def combine_monthly(prefix, add_month_col):
    """월별 파일 6개를 하나로 합치고 완전 중복 행을 제거한다. 원본은 '|' 구분, BLOCK_CD는 20자리 문자열."""
    parts = []
    for path in raw_files(prefix):
        df = pd.read_csv(path, sep='|', dtype={'BLOCK_CD': str}, encoding='utf-8-sig', low_memory=False)
        if add_month_col:
            df['기준월'] = int(path.stem.split('_')[-1])
        parts.append(df)
    combined = pd.concat(parts, ignore_index=True)
    n = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    for c in ['X_COORD', 'Y_COORD']:
        combined[c] = pd.to_numeric(combined[c], errors='coerce')
    print(f'   {prefix:14s} {n:>9,}행 → 중복 제거 {n - len(combined):>7,}행 → {len(combined):>9,}행 '
          f'| 월별 {combined.STD_YM.value_counts().sort_index().to_dict()}')
    return combined


def to_points(grid):
    return gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid.X_COORD, grid.Y_COORD), crs=CRS)


def build_grid_mapping(flows, dong):
    """격자 → 행정동(ADM_CD, ADM_NM) 매핑. 성·연령별 격자로 1차 결합 후, 나머지 자료에만 있는 격자를 추가 결합."""
    first = flows['FLOW_POP_SEX_AGE'][KEYS].drop_duplicates().dropna(subset=['X_COORD', 'Y_COORD'])
    joined = gpd.sjoin(to_points(first), dong[['ADM_CD', 'ADM_NM', 'geometry']], how='left', predicate='within')
    mapping = joined[KEYS + ['ADM_CD', 'ADM_NM']].drop_duplicates()
    print(f'   1차: 성·연령별 고유 격자 {len(first):,}개 → 미매칭 {mapping.ADM_CD.isna().sum()}개')

    others = pd.concat([flows['FLOW_POP_WKDY'][KEYS], flows['FLOW_POP_TIME'][KEYS]]).drop_duplicates()
    extra = others.merge(mapping[KEYS], on=KEYS, how='left', indicator=True)
    extra = extra.loc[extra._merge == 'left_only', KEYS].dropna(subset=['X_COORD', 'Y_COORD'])
    extra_joined = gpd.sjoin(to_points(extra), dong[['ADM_CD', 'ADM_NM', 'geometry']], how='left', predicate='within')

    mapping = (pd.concat([mapping, extra_joined[KEYS + ['ADM_CD', 'ADM_NM']]], ignore_index=True)
                 .drop_duplicates(subset=KEYS))
    print(f'   2차: 요일별·시간대별에만 있는 격자 {len(extra):,}개 추가 → 전체 {len(mapping):,}개, '
          f'미매칭 {mapping.ADM_CD.isna().sum()}개')
    return mapping


def build_target_mapping(mapping, sigungu):
    """대상 시군구(강남구·춘천시) 경계 안에 있는 격자만 남기고 SIGUNGU_NM을 붙인다."""
    target = sigungu[sigungu.SIGUNGU_CD.astype(str).isin(TARGET_SGG)]
    joined = gpd.sjoin(to_points(mapping.copy()), target[['SIGUNGU_CD', 'SIGUNGU_NM', 'geometry']],
                       how='inner', predicate='within')
    target_map = (joined.drop_duplicates(KEYS)[KEYS + ['ADM_CD', 'ADM_NM', 'SIGUNGU_NM']]
                        .rename(columns={'ADM_CD': 'EMD_CD', 'ADM_NM': 'EMD_NM'}))
    print(f'   대상 지역 격자 {len(target_map):,}개 | '
          f'{target_map.groupby("SIGUNGU_NM").size().to_dict()}')
    return target_map


def attach_dong(flow, target_map):
    """행정동·시군구 정보를 붙이고 대상 지역 밖 격자를 제외.
    left merge 후 필터링해 원본 행 순서(월 → 원본 파일 순서)를 그대로 유지한다
    (inner merge는 pandas 버전에 따라 같은 격자끼리 묶어 순서가 바뀜)."""
    out = flow.merge(target_map, on=KEYS, how='left', validate='many_to_one')
    out = out[out.EMD_CD.notna()].reset_index(drop=True)
    print(f'   {len(flow):>9,}행 → 대상 지역 밖 제외 {len(flow) - len(out):>5,}행 → {len(out):>9,}행')
    return out


def validate(name, df, value_prefixes):
    vals = df[[c for c in df.columns if c.startswith(value_prefixes)]]
    month_counts = df.STD_YM.value_counts()
    check(f'{name} {df.shape}', {
        '결측 없음': df.isna().sum().sum() == 0,
        '음수 없음': bool((vals >= 0).all().all()),
        '모든 값이 0인 행 없음': not vals.eq(0).all(axis=1).any(),
        '기준월 + 격자 키 중복 없음': not df.duplicated(['STD_YM'] + KEYS).any(),
        '6개월 모두 존재': sorted(month_counts.index.astype(str)) == MONTHS,
        '행정동 47개': df.EMD_CD.nunique() == 47,
        '시군구 2개 (강남구·춘천시)': set(df.SIGUNGU_NM) == set(TARGET_SGG.values()),
    })
    print(f'   0인 셀 비율 {vals.eq(0).to_numpy().mean() * 100:.1f}% | '
          f'월별 행 수 {month_counts.min():,} ~ {month_counts.max():,}')


def aggregate_monthly(df, prefixes):
    """50m 격자 단위 → 행정동 × 월 합계 (EMD_CD는 정수로 저장)."""
    vals = [c for c in df.columns if c.startswith(prefixes)]
    if '기준월' in df.columns:
        vals = vals + ['기준월']
    df = df.astype({'EMD_CD': 'int64'})
    return (df.groupby(MONTHLY_KEYS, as_index=False)[vals].sum()
              .sort_values(MONTHLY_KEYS).reset_index(drop=True))


def validate_monthly(name, monthly, src, prefixes):
    vals = [c for c in src.columns if c.startswith(prefixes)]
    check(f'{name} {monthly.shape}', {
        '282행 (47개 동 × 6개월)': len(monthly) == 47 * 6,
        '행정동 47개': monthly.EMD_CD.nunique() == 47,
        '키 중복 없음': not monthly.duplicated(MONTHLY_KEYS).any(),
        '결측 없음': not monthly[vals].isna().any().any(),
        '음수 없음': bool((monthly[vals] >= 0).all().all()),
        '합산 전후 총합 보존': bool(np.isclose(monthly[vals].to_numpy().sum(), src[vals].to_numpy().sum())),
    })


def main():
    args = parse_args(__doc__.splitlines()[0])
    require(DONG_SHP, SGG_SHP, *[p for prefix, _, _ in JOBS.values() for p in raw_files(prefix)])

    section('1. 월별 파일 통합 · 중복 제거')
    flows = {name: combine_monthly(prefix, add_month) for name, (prefix, _, add_month) in JOBS.items()}

    section('2. 격자 → 행정동 배정 (공간결합)')
    dong = gpd.read_file(DONG_SHP).to_crs(CRS)
    print(f'   행정동 경계 {len(dong):,}개 ({dong.crs})')
    mapping = build_grid_mapping(flows, dong)

    section('3. 대상 지역(강남구·춘천시) 필터')
    sigungu = gpd.read_file(SGG_SHP).to_crs(CRS)
    target_map = build_target_mapping(mapping, sigungu)
    result = {}
    for name, flow in flows.items():
        print(f'   [{name}]')
        result[name] = attach_dong(flow, target_map)

    section('4. 검증')
    for name, (_, prefixes, _) in JOBS.items():
        validate(name, result[name], prefixes)

    section('5. 행정동 × 월 합산 · 검증')
    monthly = {}
    for out_name, in_name in MONTHLY.items():
        prefixes = JOBS[in_name][1]
        monthly[out_name] = aggregate_monthly(result[in_name], prefixes)
        validate_monthly(out_name, monthly[out_name], result[in_name], prefixes)

    section(f'6. 저장 ({DB.name}/)')
    for name, df in result.items():
        save(df, name, dtype=CODE_COLS, dry_run=args.dry_run)
    for name, df in monthly.items():
        save(df, name, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
