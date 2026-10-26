"""전처리: 신한카드 → CARD_SPEND_TIME, CARD_SPEND_RESID

대상: 서울 강남구·강원 춘천시 가맹점 카드 결제 (2025-07-01 ~ 2025-12-31, 184일)

| 만드는 테이블    | 원본                                | 단위                                                  |
|------------------|-------------------------------------|-------------------------------------------------------|
| CARD_SPEND_TIME  | 신한카드_빅콘테스트2026_데이터1.txt | 일자 × 시간대(4구간) × 가맹점 시군구 × 업종 × 성별 × 연령 |
| CARD_SPEND_RESID | 신한카드_빅콘테스트2026_데이터2.txt | 일자 × 가맹점 시군구 × 업종 × 고객 거주 시도 × 성별 × 연령 |

원본: private_data/2026빅콘테스트_신한카드_데이터_레이아웃/ (빅콘테스트 제공, 저장소에 포함하지 않음)
      레이아웃 정의서: 2026빅콘테스트_신한카드_데이터_레이아웃_안_20260903.xlsx

처리 내용
  1. 탭 구분 · CP949 인코딩 텍스트 읽기 (UTF-8로 열면 한글이 깨짐)
  2. 가맹점 시군구 이름('서울 강남구') → 시군구 코드 SGG_CD('11230') 추가 → 다른 테이블과 join 키
  3. 연령 값의 공백 제거 ('20 대' → '20대')
  4. 법인(SEX_CCD·AGE_CCD = '법인'), 업종 등 나머지 값은 원본 그대로 유지 (분석 단계에서 필터링)

실행: python preprocessing/prep_card.py [--dry-run]   (파일이 커서 비교 포함 1~2분)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# 경로는 저장소 루트(Knock-Model/) 기준 → 어느 폴더에서 실행해도 같은 위치를 가리킨다
ROOT = Path(__file__).resolve().parents[1]
CARD_RAW = ROOT / 'private_data' / '2026빅콘테스트_신한카드_데이터_레이아웃'
DB = ROOT / 'database'

SGG_CODE = {'서울 강남구': '11230', '강원 춘천시': '32010'}
FILES = {'CARD_SPEND_TIME': CARD_RAW / '신한카드_빅콘테스트2026_데이터1.txt',
         'CARD_SPEND_RESID': CARD_RAW / '신한카드_빅콘테스트2026_데이터2.txt'}
CODE_COLS = {'TA_YMD': str, 'TIME_GB': str, 'SGG_CD': str}


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


# ---- 읽기 · 검증 ----


def load_card(path):
    df = pd.read_csv(path, sep='\t', encoding='cp949', dtype={'TA_YMD': str, 'TIME_GB': str})
    df.columns = df.columns.str.strip()
    # 가맹점 시군구 코드 추가 (MCT_SGG_CD 바로 뒤)
    df.insert(df.columns.get_loc('MCT_SGG_CD') + 1, 'SGG_CD', df.MCT_SGG_CD.map(SGG_CODE))
    # 연령 공백 제거: '20 대' → '20대' (법인은 그대로)
    df['AGE_CCD'] = df.AGE_CCD.str.replace(' ', '', regex=False)
    return df


def validate(name, df):
    dates = pd.to_datetime(df.TA_YMD, format='%Y%m%d')
    full = pd.date_range('2025-07-01', '2025-12-31')
    checks = {
        '결측 없음': df.isna().sum().sum() == 0,
        '중복 행 없음': not df.duplicated().any(),
        'SGG_CD 매핑 누락 없음 (2개 시군구)': df.SGG_CD.notna().all() and set(df.SGG_CD) == {'11230', '32010'},
        '기간 2025-07-01 ~ 12-31, 184일 빠짐없음': set(dates.unique()) == set(full),
        '시군구별로도 184일 모두 존재': (df.groupby('SGG_CD').TA_YMD.nunique() == 184).all(),
        '연령 값에 공백 없음': not df.AGE_CCD.str.contains(' ').any(),
        '금액·건수 음수 없음': (df.TS_AT >= 0).all() and (df.USE_CNT >= 0).all(),
    }
    if name == 'CARD_SPEND_TIME':
        checks['시간대 4구간 (00_05, 06_11, 12_17, 18_23)'] = set(df.TIME_GB) == {'00_05', '06_11', '12_17', '18_23'}
    check(f'{name} {df.shape} | 업종 {df.MCT_RY_CD.nunique()}개', checks)


def main():
    args = parse_args(__doc__.splitlines()[0])
    require(*FILES.values())

    section('1. 읽기·정제')
    card = {name: load_card(path) for name, path in FILES.items()}
    for name, df in card.items():
        print(f'   {name:17s} {df.shape} | SEX_CCD {sorted(df.SEX_CCD.unique())}')
        print(f'   {"":17s} AGE_CCD {sorted(df.AGE_CCD.unique())}')

    section('2. 검증')
    for name, df in card.items():
        validate(name, df)

    section(f'3. 저장 ({DB.name}/)')
    for name, df in card.items():
        save(df, name, dtype=CODE_COLS, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
