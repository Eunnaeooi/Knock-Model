"""전체 실행: 전처리 → 튜닝 → 군집 → 검증 → 설명

| 명령                       | 실행 단계                                                        |
|----------------------------|------------------------------------------------------------------|
| python run_all.py          | cluster → validate → xai_explain  (저장소만으로 실행 가능)         |
| python run_all.py --tune   | tune부터 (10분 이상)                                              |
| python run_all.py --prep   | 전처리 4개(prep_features 포함)부터 전부 (private_data/ 필요)        |
| --dry-run (위 명령에 붙임)  | 저장하지 않고 검증과 기존 파일 비교만                              |

각 스크립트를 따로 실행하고, 한 단계가 실패하면 멈춘다.
--dry-run이면 앞 단계 결과가 저장되지 않으므로, 뒤 단계는 기존 파일을 읽어 비교한다.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PREP = ['preprocessing/prep_admin_population.py', 'preprocessing/prep_card.py',
        'preprocessing/prep_flow.py', 'preprocessing/prep_features.py']
TUNE = ['modeling/tune.py']
MODEL = ['modeling/cluster.py', 'modeling/validate.py', 'modeling/xai_explain.py']


def main():
    parser = argparse.ArgumentParser(description='전체 실행: 전처리 → 튜닝 → 군집 → 검증 → 설명')
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--prep', action='store_true', help='전처리부터 전부 실행 (private_data/ 필요)')
    start.add_argument('--tune', action='store_true', help='튜닝부터 실행')
    parser.add_argument('--dry-run', action='store_true', help='저장하지 않고 검증과 기존 파일 비교만')
    args = parser.parse_args()

    steps = (PREP if args.prep else []) + (TUNE if args.prep or args.tune else []) + MODEL
    extra = ['--dry-run'] if args.dry_run else []
    print(f'실행 단계 ({len(steps)}개): {" → ".join(Path(s).stem for s in steps)}'
          + ('  [--dry-run]' if args.dry_run else ''))

    t_all = time.time()
    for i, step in enumerate(steps, 1):
        print(f'\n{"#" * 70}\n# [{i}/{len(steps)}] {step}\n{"#" * 70}', flush=True)
        t = time.time()
        result = subprocess.run([sys.executable, str(ROOT / step), *extra], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f'\n[중단] {step} 실패 (종료 코드 {result.returncode}) → 이후 단계를 실행하지 않음')
        print(f'# 완료: {step} ({time.time() - t:.0f}초)', flush=True)
    print(f'\n전체 완료 ({time.time() - t_all:.0f}초)')


if __name__ == '__main__':
    main()
