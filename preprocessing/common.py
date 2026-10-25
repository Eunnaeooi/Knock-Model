"""전처리 공통 설정 — 경로, 실행 옵션, 검증·저장 함수.

모든 경로는 저장소 루트(Knock-Model/) 기준이며, 어느 폴더에서 실행해도 같은 위치를 가리킨다.

    Knock-Model/
    ├─ preprocessing/   ← 이 파일과 전처리 스크립트 4개
    ├─ database/        ← 전처리 결과 테이블 (DB)
    ├─ public_data/     ← 공공데이터 원본, 2. 경계/ (행정구역 경계 shp)
    └─ private_data/    ← 빅콘테스트 제공 원천 데이터 (저장소에 포함하지 않음)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PUBLIC = ROOT / 'public_data'
BOUNDARY = PUBLIC / '2. 경계'
PRIVATE = ROOT / 'private_data'
CARD_RAW = PRIVATE / '2026빅콘테스트_신한카드_데이터_레이아웃'
SKT_RAW = PRIVATE / '2026빅콘테스트_sk통신인구_데이터_레이아웃'
DB = ROOT / 'database'

TARGET_SGG = {'11230': '강남구', '32010': '춘천시'}   # 분석 대상 시군구 (통계청 5자리 코드)


def parse_args(description):
    """공통 실행 옵션: --dry-run 이면 저장하지 않고 검증·기존 파일 비교만 수행."""
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
