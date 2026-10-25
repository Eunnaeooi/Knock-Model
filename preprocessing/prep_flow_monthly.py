"""전처리: 유동인구 월별 집계 → flow_age_monthly, flow_day_monthly, flow_time_monthly

대상: 47개 행정동 × 2025년 7~12월(6개월) = 282행

| 만드는 테이블     | 내용                                         | 입력 (prep_flow 산출물) |
|-------------------|----------------------------------------------|------------------------|
| flow_age_monthly  | 동 × 월 × 성·연령(12구간) 유동인구 합계       | FLOW_POP_SEX_AGE       |
| flow_day_monthly  | 동 × 월 × 요일(7구간) 유동인구 합계           | FLOW_POP_WKDY          |
| flow_time_monthly | 동 × 월 × 시간대(24구간) 유동인구 합계        | FLOW_POP_TIME          |

원천 SKT 데이터에는 행정동 코드가 없고 격자 좌표만 있으므로, 행정동 배정은 prep_flow에서 끝내고
이 단계는 그 결과(50m 격자 단위)를 행정동 × 월로 합산만 한다. 결과는 EDA 6장(유동인구)의 입력이다.

집계 규칙
  - 키: SIGUNGU_NM, EMD_CD, EMD_NM, STD_YM / 값: 각 유동인구 컬럼의 합계(sum)
  - flow_age_monthly의 '기준월'은 기존 파일 형식을 맞추기 위해 남긴 컬럼이다
    (행마다 STD_YM이 합산된 값이라 의미 없음). EDA에서 읽은 직후 삭제한다.

실행: python preprocessing/prep_flow_monthly.py [--dry-run]
"""
import numpy as np
import pandas as pd

from common import DB, check, parse_args, require, save, section

KEYS = ['SIGUNGU_NM', 'EMD_CD', 'EMD_NM', 'STD_YM']
JOBS = {   # 만드는 테이블: (입력 테이블, 값 컬럼 접두어)
    'flow_age_monthly':  ('FLOW_POP_SEX_AGE', ('MAN_FLOW_POP_CNT_', 'WMAN_FLOW_POP_CNT_')),
    'flow_day_monthly':  ('FLOW_POP_WKDY', ('FLOW_POP_CNT_',)),
    'flow_time_monthly': ('FLOW_POP_TIME', ('TMST_',)),
}


def aggregate(df, prefixes):
    vals = [c for c in df.columns if c.startswith(prefixes)]
    if '기준월' in df.columns:
        vals = vals + ['기준월']
    return (df.groupby(KEYS, as_index=False)[vals].sum()
              .sort_values(KEYS).reset_index(drop=True))


def validate(name, monthly, src, prefixes):
    vals = [c for c in src.columns if c.startswith(prefixes)]
    check(f'{name} {monthly.shape}', {
        '282행 (47개 동 × 6개월)': len(monthly) == 47 * 6,
        '행정동 47개': monthly.EMD_CD.nunique() == 47,
        '키 중복 없음': not monthly.duplicated(KEYS).any(),
        '결측 없음': not monthly[vals].isna().any().any(),
        '음수 없음': bool((monthly[vals] >= 0).all().all()),
        '합산 전후 총합 보존': bool(np.isclose(monthly[vals].to_numpy().sum(), src[vals].to_numpy().sum())),
    })


def main():
    args = parse_args(__doc__.splitlines()[0])
    require(*[DB / f'{in_name}.csv' for in_name, _ in JOBS.values()])

    section('1. 읽기 (prep_flow 산출물)')
    src = {}
    for in_name, _ in JOBS.values():
        df = pd.read_csv(DB / f'{in_name}.csv', encoding='utf-8-sig', dtype={'BLOCK_CD': str}, low_memory=False)
        src[in_name] = df
        print(f'   {in_name:18s} {len(df):>9,}행 | 행정동 {df.EMD_CD.nunique()}개')

    section('2. 행정동 × 월 합산 · 검증')
    monthly = {}
    for out_name, (in_name, prefixes) in JOBS.items():
        monthly[out_name] = aggregate(src[in_name], prefixes)
        validate(out_name, monthly[out_name], src[in_name], prefixes)

    section(f'3. 저장 ({DB.name}/)')
    for out_name, df in monthly.items():
        save(df, out_name, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
