"""검증: 기본 vs 최종 · 민감도 · 판단 규칙 · 대리 모델 교차검증
→ artifacts/validation_summary.csv, artifacts/sensitivity_final.csv

정답(동별 실제 고독사) 데이터가 없으므로 아래로 검증한다.
  1. 분리도·안정성: 기본 모델과 같은 함수·설정으로 잰 실루엣·DB·부트스트랩 ARI (tuning_results.csv)
  2. 민감도: 점수 방식·가중치·변수 구성을 바꿔 같은 알고리즘·K로 다시 군집 → 순위상관·ARI·상위 10개 동 일치
  3. 판단 규칙: 의사결정나무(깊이 3)가 5개 변수만으로 최종 군집을 재현하는가 (출력만)
  4. 대리 모델(랜덤포레스트) 정확도: 반복 5-fold × 20 → SHAP 설명의 신뢰도

실행: python modeling/validate.py [--dry-run]
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from cluster import build
from common import (ART, BASE, N_BOOT, SEED, SHORT, VARS, axis_scores, check, fit_predict, parse_args, save,
                    section, shap_inputs, surrogate)

SUMMARY_FILE = ART / 'validation_summary.csv'
SENSITIVITY_FILE = ART / 'sensitivity_final.csv'
METRICS = ['실루엣', 'DB지수', '안정성ARI', '최소군집', '기본모델과ARI']


def sensitivity(m):
    """최종 모델과 같은 알고리즘·K로 입력만 바꿔 다시 군집 → 최종 모델과 비교"""
    feat, d = m.feat, m.d
    ref_pop, ref_reg = feat.AX_POP, feat.AX_REG
    ref_total = ref_pop + ref_reg                  # 최종 모델 입력의 종합 점수
    other = 'z' if m.KIND == 'q' else 'q'
    alts = {
        f'최종 ({m.SPACE}, 동일 가중)': (ref_pop, ref_reg),
        ('5분위 단계점수 (기본 모델 입력)' if other == 'q' else '2D z-score (척도 변경)'): axis_scores(d, other),
        '인구축 2배 가중 (논문식 우선순위)': (ref_pop * 2, ref_reg),
        '1-2 남성 비율 제외': axis_scores(d, m.KIND, pop_vars=['HH1_EXPO_5060', 'VISIBLE_IDX_50P']),
        '2-2 인구밀도 제외': axis_scores(d, m.KIND, reg_vars=['TIME_ENTROPY']),
    }
    if m.KIND == 'z':
        alts['인구밀도 log 미적용'] = axis_scores(d, 'z', src=d.Zr.assign(POP_DENSITY=d.Zr_raw_dens))

    top10_base = set(ref_total.nlargest(10).index)
    rows = []
    for name, (a, b) in alts.items():
        total = a + b
        lab = fit_predict(m.ALGO, np.c_[a, b], m.K)
        rows.append({'방식': name, '종합점수 순위상관': spearmanr(total, ref_total)[0],
                     '군집 일치(ARI)': adjusted_rand_score(feat.CLUSTER_T, lab),
                     '상위 10개 동 일치': len(set(total.nlargest(10).index) & top10_base)})
    return pd.DataFrame(rows).round(3)


def tree_rules(tree, names, digits=2):
    """판단 규칙 출력 - 아래 잎이 모두 같은 군집인 가지는 하나로 합친다 (불필요한 분기 제거)"""
    t, cls = tree.tree_, tree.classes_

    def leaves(n):
        if t.children_left[n] == -1:
            return {cls[t.value[n][0].argmax()]}
        return leaves(t.children_left[n]) | leaves(t.children_right[n])

    lines = []

    def rec(n, depth):
        pad = '│   ' * depth
        lc = leaves(n)
        if len(lc) == 1:
            lines.append(f'{pad}→ {lc.pop()}  ({t.n_node_samples[n]}개 동)'); return
        f, th = names[t.feature[n]], t.threshold[n]
        lines.append(f'{pad}{f} ≤ {th:.{digits}f}'); rec(t.children_left[n], depth + 1)
        lines.append(f'{pad}{f} > {th:.{digits}f}');  rec(t.children_right[n], depth + 1)

    rec(0, 0)
    return '\n'.join(lines)


def decision_rules(m):
    """의사결정나무(깊이 3)로 최종 군집 재현 → 5-fold 교차검증 정확도 + 판단 규칙"""
    y = m.feat.GROUP_T
    X_rule = (m.d.Zr if m.KIND == 'z' else m.d.Qr)[VARS]
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    cv = StratifiedKFold(n_splits=min(5, y.value_counts().min()), shuffle=True, random_state=SEED)
    acc = cross_val_score(tree, X_rule.values, y, cv=cv)
    print(f'   {cv.n_splits}-fold 정확도: {acc.mean():.3f} (± {acc.std():.3f})  | fold별 {np.round(acc, 2)}')
    tree.fit(X_rule.values, y)
    print(f'   학습 데이터 정확도: {tree.score(X_rule.values, y):.3f}')
    unit = '시군구 내 z, 클수록 위험' if m.KIND == 'z' else '1~5점, 클수록 위험'
    print(f'\n[판단 규칙 - AI 에이전트용] ({unit})')
    print(tree_rules(tree, [SHORT[k] for k in VARS]))


def surrogate_accuracy(m):
    """대리 모델(랜덤포레스트) 반복 5-fold × 20 정확도 + 학습 데이터 정확도"""
    X, y = shap_inputs(m.d).values, m.feat.GROUP_T.values
    rf = surrogate()
    acc = cross_val_score(rf, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED))
    train = rf.fit(X, y).score(X, y)
    print(f'   대리 모델 정확도 (반복 5-fold × 20): {acc.mean():.3f} (± {acc.std():.3f}) | 학습 데이터 {train:.3f}')
    return acc.mean(), acc.std(), train


def summary(m, rf_mean, rf_std, rf_train):
    """최종 모델 검증 요약 한 줄 (artifacts/validation_summary.csv 형식)"""
    b = m.best
    counts = m.feat.GROUP_T.value_counts()
    return pd.DataFrame([{
        '모델': '최종 튜닝 모델',
        '입력공간': f'{m.SPACE} (인구축·지역축)',
        '알고리즘': m.ALGO,
        'K': m.K,
        '동 수': len(m.feat),
        '군집 구성': ' / '.join(f'{g} {counts[g]}' for g in m.TG),
        '실루엣 계수': round(b.실루엣, 3),
        'Davies-Bouldin 지수': round(b.DB지수, 3),
        f'부트스트랩 안정성 ARI ({N_BOOT}회)': round(b.안정성ARI, 3),
        '최소 군집 크기': int(b.최소군집),
        f'기본 모델({BASE[0]} {BASE[1]} K={BASE[2]})과 ARI': round(b.기본모델과ARI, 3),
        '대리모델 RF 정확도 (반복 5-fold×20)': round(rf_mean, 3),
        '대리모델 RF 정확도 표준편차': round(rf_std, 3),
        '대리모델 RF 학습 정확도': round(rf_train, 3),
    }])


def main():
    args = parse_args(__doc__.splitlines()[0])

    section('1. 기본 모델 vs 최종 모델 (같은 함수·같은 설정)')
    m = build()
    vs = pd.DataFrame({f'기본 모델 ({BASE[0]} · {BASE[1]}, K={BASE[2]})': m.base_row[METRICS],
                       f'최종 모델 ({m.SPACE} · {m.ALGO}, K={m.K})': m.best[METRICS]}).astype(float)
    print(vs.round(3).to_string())

    section('2. 민감도 (입력을 바꿔 같은 알고리즘·K로 다시 군집)')
    sens = sensitivity(m)
    print(sens.to_string(index=False))

    section('3. 판단 규칙 (의사결정나무, 깊이 3)')
    decision_rules(m)

    section('4. 대리 모델 정확도 (랜덤포레스트)')
    rf_mean, rf_std, rf_train = surrogate_accuracy(m)
    summ = summary(m, rf_mean, rf_std, rf_train)
    check('검증 결과', {
        '민감도 첫 행 = 최종 모델 (순위상관·ARI 1)': np.isclose(sens.iloc[0, 1], 1) and np.isclose(sens.iloc[0, 2], 1),
        '결측 없음': not sens.isna().any().any() and not summ.isna().any().any(),
    })

    section('5. 저장')
    save(summ, SUMMARY_FILE, dry_run=args.dry_run)
    save(sens, SENSITIVITY_FILE, float_format='%.3f', dry_run=args.dry_run)


if __name__ == '__main__':
    main()
