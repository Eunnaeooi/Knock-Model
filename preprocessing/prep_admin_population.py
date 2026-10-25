"""전처리: 행정구역·인구 → SGG_MST, EMD_MST, POP_SEX_AGE, HH_SIZE

대상: 서울 강남구(22개 동) + 강원 춘천시(25개 읍·면·동) = 47개 행정동

| 만드는 테이블 | 내용                                              | 원본          |
|---------------|---------------------------------------------------|---------------|
| SGG_MST       | 시군구 코드·이름 (2행)                            | ①             |
| EMD_MST       | 행정동 코드·이름·면적·총인구 (47행)               | ①, ②, ④, ⑤    |
| POP_SEX_AGE   | 동 × 성별 × 연령(10세 단위 11구간) 인구 (1,034행) | ②             |
| HH_SIZE       | 동 × 세대원수(1인~10인이상) 세대수 (470행)        | ③             |

원본 (public_data/, 출처는 public_data/출처목록.md)
  ① 국가데이터처_행정동 정보_20250704.csv                       행정동 코드 (최신 개정 2025-04-01)
  ② 강남구·춘천시_202512_202512_연령별인구현황_연간.csv          주민등록 연령별 인구 (2025.12)
  ③ 강남구·춘천시_..._주민등록인구기타현황(세대원수별 세대수).csv 세대원수별 세대수 (2025.12)
  ④ 강남구_행정구역(동별)_20260428.xlsx                          강남구 동별 면적
  ⑤ 춘천시_토지면적_20241231.csv                                 춘천시 읍면동별 면적

실행: python preprocessing/prep_admin_population.py [--dry-run]
"""
import re
import warnings

import pandas as pd

from common import DB, PUBLIC, TARGET_SGG, check, parse_args, require, save, section

AGE_ORDER = ['10대미만', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대', '90대', '100세이상']
AGE_RAW = ['0~9세', '10~19세', '20~29세', '30~39세', '40~49세', '50~59세', '60~69세',
           '70~79세', '80~89세', '90~99세', '100세 이상']
AGE_MAP = dict(zip(AGE_RAW, AGE_ORDER))
HH_ORDER = [f'{i}인' for i in range(1, 10)] + ['10인이상']

CODE_FILE = PUBLIC / '국가데이터처_행정동 정보_20250704.csv'
AREA_GN_FILE = PUBLIC / '강남구_행정구역(동별)_20260428.xlsx'
AREA_CC_FILE = PUBLIC / '춘천시_토지면적_20241231.csv'
POP_FILES = {'11230': PUBLIC / '강남구_202512_202512_연령별인구현황_연간.csv',
             '32010': PUBLIC / '춘천시_202512_202512_연령별인구현황_연간.csv'}
HH_FILES = {'11230': PUBLIC / '강남구_202512_202512_주민등록인구기타현황(세대원수별 세대수).csv',
            '32010': PUBLIC / '춘천시_202512_202512_주민등록인구기타현황(세대원수별 세대수)_year.csv'}

CODE_COLS = {'SGG_CD': str, 'EMD_CD': str}


def to_int(s):
    """'1,246' → 1246"""
    return s.astype(str).str.replace(',', '').str.strip().astype(int)


def parse_dong(s):
    """'서울특별시 강남구 신사동(1168051000)' → '신사동' / 시군구 합계 행은 None"""
    m = re.match(r'.* (\S+)\((\d+)\)$', s)
    return m.group(1) if m else None


def load_codes():
    """① 행정동 코드: 가장 최근 개정일자만 사용, 8자리 = 행정동 / 5자리 = 시군구."""
    code = pd.read_csv(CODE_FILE, encoding='cp949', dtype=str).dropna(subset=['행정동코드'])
    latest = code[code.개정일자 == code.개정일자.max()]
    target = (latest[latest.부모행정동코드.isin(TARGET_SGG)]
              [['행정동코드', '행정동명', '부모행정동코드']].drop_duplicates())

    sgg_mst = (target[target.행정동코드.str.len() == 5]
               .rename(columns={'행정동코드': 'SGG_CD', '행정동명': 'SGG_NM'})[['SGG_CD', 'SGG_NM']]
               .sort_values('SGG_CD').reset_index(drop=True))
    emd_code = (target[target.행정동코드.str.len() == 8]
                .rename(columns={'행정동코드': 'EMD_CD', '행정동명': 'EMD_NM', '부모행정동코드': 'SGG_CD'})
                [['SGG_CD', 'EMD_CD', 'EMD_NM']].reset_index(drop=True))
    print(f'   최신 개정일자 {code.개정일자.max()} | 행정동 수 {emd_code.groupby("SGG_CD").size().to_dict()}')
    return sgg_mst, emd_code


def load_area():
    """④ 강남 xlsx(3줄 머리글 + 병합 셀), ⑤ 춘천 csv → 동 이름별 면적(km²)."""
    with warnings.catch_warnings():                         # 서울시 xlsx에 기본 스타일이 없다는 openpyxl 경고 무시
        warnings.filterwarnings('ignore', message='Workbook contains no default style')
        xl = pd.read_excel(AREA_GN_FILE, header=None, skiprows=3)
    xl[1] = xl[1].ffill()                                   # 병합된 시군구 셀 채우기
    area_gn = xl[(xl[1] == '강남구') & (xl[2] != '소계')][[2, 3]].rename(columns={2: 'EMD_NM', 3: 'AREA_KM2'})
    area_gn['SGG_CD'] = '11230'

    cc = pd.read_csv(AREA_CC_FILE, encoding='utf-8-sig')
    area_cc = cc.rename(columns={'읍면동': 'EMD_NM', '면적_km2': 'AREA_KM2'})[['EMD_NM', 'AREA_KM2']]
    area_cc['SGG_CD'] = '32010'

    area = pd.concat([area_gn, area_cc], ignore_index=True)
    area['AREA_KM2'] = area.AREA_KM2.astype(float)
    print(f'   면적 행 수 {area.groupby("SGG_CD").size().to_dict()} | '
          f'합계 {area.groupby("SGG_CD").AREA_KM2.sum().round(2).to_dict()}')
    return area


def load_population():
    """② 연령별 인구 → 성별 × 연령 11구간 long 형식 + 동별 총인구."""
    pop_rows, tot_rows = [], []
    for sgg_cd, path in POP_FILES.items():
        raw = pd.read_csv(path, encoding='cp949')
        raw['EMD_NM'] = raw['행정구역'].map(parse_dong)
        raw = raw.dropna(subset=['EMD_NM'])                 # 시군구 합계 행 제외
        tot_rows.append(pd.DataFrame({'SGG_CD': sgg_cd, 'EMD_NM': raw.EMD_NM,
                                      'TOT_POP_CNT': to_int(raw['2025년_계_총인구수'])}))
        for sx, sex in [('남', '남성'), ('여', '여성')]:
            for a in AGE_RAW:
                pop_rows.append(pd.DataFrame({'SGG_CD': sgg_cd, 'EMD_NM': raw.EMD_NM, 'SEX_CCD': sex,
                                              'AGE_CCD': AGE_MAP[a], 'POP_CNT': to_int(raw[f'2025년_{sx}_{a}'])}))
    pop_long = pd.concat(pop_rows, ignore_index=True)
    tot_pop = pd.concat(tot_rows, ignore_index=True)
    print(f'   동 수 {tot_pop.groupby("SGG_CD").size().to_dict()} | 인구 행 수 {len(pop_long)}')
    return pop_long, tot_pop


def load_households():
    """③ 세대원수별 세대수 → long 형식. 세대원수 합 = 전체세대인지 확인."""
    hh_rows = []
    for sgg_cd, path in HH_FILES.items():
        raw = pd.read_csv(path, encoding='cp949')
        raw['EMD_NM'] = raw['행정구역'].map(parse_dong)
        raw = raw.dropna(subset=['EMD_NM'])
        for h in HH_ORDER:
            hh_rows.append(pd.DataFrame({'SGG_CD': sgg_cd, 'EMD_NM': raw.EMD_NM, 'HH_SIZE_CCD': h,
                                         'HH_CNT': to_int(raw[f'2025년12월_{h}세대'])}))
        assert (sum(to_int(raw[f'2025년12월_{h}세대']) for h in HH_ORDER)
                == to_int(raw['2025년12월_전체세대'])).all(), f'{path.name}: 세대원수 합 ≠ 전체세대'
    hh_long = pd.concat(hh_rows, ignore_index=True)
    print(f'   세대 행 수 {len(hh_long)} | 세대원수 합 = 전체세대')
    return hh_long


def build_tables(emd_code, area, pop_long, tot_pop, hh_long):
    """동 이름 → EMD_CD는 '시군구 코드 + 동 이름'으로 매칭 (행정안전부 10자리 코드 ≠ 통계청 8자리 코드)."""
    key = emd_code[['SGG_CD', 'EMD_NM', 'EMD_CD']]

    emd_mst = (tot_pop.merge(key, on=['SGG_CD', 'EMD_NM'], how='left', validate='one_to_one')
                      .merge(area, on=['SGG_CD', 'EMD_NM'], how='left', validate='one_to_one'))
    emd_mst = emd_mst[['SGG_CD', 'EMD_CD', 'EMD_NM', 'AREA_KM2', 'TOT_POP_CNT']]

    pop = pop_long.merge(key, on=['SGG_CD', 'EMD_NM'], how='left')
    pop_sex_age = (pop.assign(_a=pop.AGE_CCD.map(AGE_ORDER.index), _s=pop.SEX_CCD.eq('여성'))
                      .sort_values(['EMD_CD', '_s', '_a'])[['EMD_CD', 'SEX_CCD', 'AGE_CCD', 'POP_CNT']]
                      .reset_index(drop=True))

    hh = hh_long.merge(key, on=['SGG_CD', 'EMD_NM'], how='left')
    order = {c: i for i, c in enumerate(emd_mst.EMD_CD)}     # 동 순서 = 인구 원본(행정안전부) 순서
    hh_size = (hh.assign(_o=hh.EMD_CD.map(order), _h=hh.HH_SIZE_CCD.map(HH_ORDER.index))
                 .sort_values(['_o', '_h'])[['EMD_CD', 'HH_SIZE_CCD', 'HH_CNT']].reset_index(drop=True))
    return emd_mst, pop_sex_age, hh_size


def validate(sgg_mst, emd_mst, pop_sex_age, hh_size):
    area_sum = emd_mst.groupby('SGG_CD').AREA_KM2.sum()
    # 동별 면적이 소수 둘째 자리로 반올림돼 있어 강남 합계는 39.51 (원본 소계 39.50) → 0.05 이내면 통과
    area_ref = pd.Series({'11230': 39.50, '32010': 1116.40})
    pop_sum = pop_sex_age.groupby('EMD_CD').POP_CNT.sum().reindex(emd_mst.EMD_CD)
    check('행정구역·인구', {
        '행정동 47개 (강남 22 + 춘천 25)': len(emd_mst) == 47
            and emd_mst.SGG_CD.value_counts().to_dict() == {'32010': 25, '11230': 22},
        'EMD_CD 매칭 누락 없음': emd_mst.EMD_CD.notna().all() and pop_sex_age.EMD_CD.notna().all()
            and hh_size.EMD_CD.notna().all(),
        'EMD_CD 8자리·중복 없음': emd_mst.EMD_CD.str.len().eq(8).all() and not emd_mst.EMD_CD.duplicated().any(),
        '면적 매칭 누락 없음': emd_mst.AREA_KM2.notna().all(),
        '면적 합 ≈ 원본 소계 (강남 39.50, 춘천 1,116.40)': bool((area_sum - area_ref).abs().lt(0.05).all()),
        '성·연령 인구 합 = 총인구': pop_sum.values.tolist() == emd_mst.TOT_POP_CNT.tolist(),
        'POP_SEX_AGE 1,034행 (47 × 2 × 11)': len(pop_sex_age) == 1034,
        'HH_SIZE 470행 (47 × 10)': len(hh_size) == 470,
        '결측 없음': all(d.isna().sum().sum() == 0 for d in [sgg_mst, emd_mst, pop_sex_age, hh_size]),
        '음수 없음': (pop_sex_age.POP_CNT >= 0).all() and (hh_size.HH_CNT >= 0).all(),
    })


def main():
    args = parse_args(__doc__.splitlines()[0])
    require(CODE_FILE, AREA_GN_FILE, AREA_CC_FILE, *POP_FILES.values(), *HH_FILES.values())

    section('1. 행정동 코드 (원본 ①)')
    sgg_mst, emd_code = load_codes()
    section('2. 행정동 면적 (원본 ④ 강남, ⑤ 춘천)')
    area = load_area()
    section('3. 연령별 인구 (원본 ②)')
    pop_long, tot_pop = load_population()
    section('4. 세대원수별 세대수 (원본 ③)')
    hh_long = load_households()

    section('5. 테이블 조립 · 검증')
    emd_mst, pop_sex_age, hh_size = build_tables(emd_code, area, pop_long, tot_pop, hh_long)
    validate(sgg_mst, emd_mst, pop_sex_age, hh_size)

    section(f'6. 저장 ({DB.name}/)')
    for name, df in [('SGG_MST', sgg_mst), ('EMD_MST', emd_mst),
                     ('POP_SEX_AGE', pop_sex_age), ('HH_SIZE', hh_size)]:
        save(df, name, dtype=CODE_COLS, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
