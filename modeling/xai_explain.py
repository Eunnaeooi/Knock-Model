"""설명: 대리 모델 + SHAP → artifacts/shap_importance.csv, artifacts/dong_cluster.csv

K-means는 예측 모델이 아니어서 SHAP을 직접 쓸 수 없으므로 대리 모델(surrogate)을 거친다.
  1. 5개 파생변수(시군구 내 z-score, 클수록 위험)로 최종 군집을 맞히는 랜덤포레스트 학습
     (정확도는 validate.py에서 반복 5-fold × 20으로 확인)
  2. SHAP(TreeExplainer)으로 전체 변수 중요도(군집별 평균 |SHAP|)와 동별 판별 근거를 분해
  3. 동별 결과표(dong_cluster.csv): 동 정보 + 원값 + 단계 점수 + 군집 + 위험도 + 판별 근거 2개
     - 판별 근거 = 그 동이 속한 군집의 확률을 가장 크게 올린 변수 2개, 변수 값이 위험 쪽인지 안전 쪽인지 표시
     - P_OWN = 대리 모델이 그 동을 자기 군집으로 볼 확률

실행: python modeling/xai_explain.py [--dry-run]
"""
import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter('ignore')              # shap 불러올 때 진행막대(tqdm) 안내 경고만 숨김
    import shap

from cluster import build
from common import ART, QCOLS, SHORT, VARS, check, parse_args, save, section, shap_inputs, surrogate
from plot_style import DIV, INK2, clean, group_color, plt, save_fig, subtitle

IMPORTANCE_FILE = ART / 'shap_importance.csv'
DONG_FILE = ART / 'dong_cluster.csv'
CODE_COLS = {'EMD_CD': str, 'SGG_CD': str}


def fit_shap(m):
    """대리 모델 학습 + SHAP 값 [동 × 변수 × 군집]"""
    X = shap_inputs(m.d)
    rf = surrogate().fit(X.values, m.feat.GROUP_T.values)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X.values)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)    # shap 버전별 형태 통일
    print(f'   SHAP 배열 크기: {sv.shape} | 군집 순서: {list(rf.classes_)} | 학습 데이터 정확도 {rf.score(X.values, m.feat.GROUP_T):.3f}')
    return X, rf, explainer, sv


def importance(m, X, rf, sv):
    """전체 변수 중요도: 군집별 평균 |SHAP|"""
    classes = list(rf.classes_)
    imp = pd.DataFrame({c: np.abs(sv[:, :, i]).mean(axis=0) for i, c in enumerate(classes)}, index=X.columns)
    return imp[[g for g in m.TG if g in imp.columns]]


def importance_table(imp):
    """artifacts/shap_importance.csv 형식: 변수코드·변수명·군집별·합계, 합계 큰 순서, 셋째 자리
    (합계 = 반올림한 군집별 값의 합 → 표에서 더한 값과 맞음)"""
    code = {SHORT[k]: k for k in VARS}
    t = imp.round(3)
    t['합계'] = t.sum(axis=1).round(3)
    t = t.sort_values('합계', ascending=False)
    t.insert(0, '변수코드', t.index.map(code))
    return t.rename_axis('변수명').reset_index()[['변수코드', '변수명'] + list(imp.columns) + ['합계']]


def explain_dong(m, X, rf, explainer, sv, emd_cd, top=5):
    """해당 동이 속한 군집 확률을 각 변수가 얼마나 올리고/내렸는지"""
    classes = list(rf.classes_)
    i = m.feat.index.get_loc(emd_cd); gname = m.feat.GROUP_T.iloc[i]; ci = classes.index(gname)
    s = pd.Series(sv[i, :, ci], index=X.columns).sort_values(key=np.abs, ascending=False)
    base = explainer.expected_value[ci] if np.ndim(explainer.expected_value) else explainer.expected_value
    return gname, s.head(top), base


def dong_table(m, X, rf, explainer, sv):
    """artifacts/dong_cluster.csv 형식: 47개 동 × 군집·위험도·판별 근거"""
    feat = m.feat.copy()
    proba = pd.DataFrame(rf.predict_proba(X.values), index=feat.index, columns=rf.classes_)
    feat['P_OWN'] = [proba.at[i, g] for i, g in zip(feat.index, feat.GROUP_T)]

    def reason(emd_cd, var, v):
        side = '위험' if X.at[emd_cd, var] > 0 else '안전'
        return f'{var} ({side} 쪽, {v:+.2f})'

    r1, r2 = [], []
    for emd_cd in feat.index:
        _, s, _ = explain_dong(m, X, rf, explainer, sv, emd_cd, top=len(VARS))
        pos = s[s > 0].sort_values(ascending=False)
        r1.append(reason(emd_cd, pos.index[0], pos.iloc[0]) if len(pos) > 0 else '-')
        r2.append(reason(emd_cd, pos.index[1], pos.iloc[1]) if len(pos) > 1 else '-')
    feat['REASON_1'], feat['REASON_2'] = r1, r2

    out = feat.reset_index()[['EMD_CD', 'SGG_CD', 'SGG_NM', 'EMD_NM', 'AREA_KM2', 'TOT_POP_CNT', 'POP_5060']
                             + VARS + QCOLS + ['POP_SCORE', 'REG_SCORE', 'TOTAL_SCORE', 'GROUP_T', 'RISK_LEVEL',
                                               'P_OWN', 'REASON_1', 'REASON_2']]
    out = out.rename(columns={'GROUP_T': 'GROUP'})
    out[VARS] = out[VARS].round(4)
    out['POP_DENSITY'] = out.POP_DENSITY.round(1)
    out[['POP_SCORE', 'REG_SCORE', 'TOTAL_SCORE', 'P_OWN']] = out[['POP_SCORE', 'REG_SCORE', 'TOTAL_SCORE', 'P_OWN']].round(4)
    # 반올림 후 정렬: 같은 점수(예: 2 + 3.3333)가 부동소수점 오차로 갈리지 않도록
    return out.sort_values(['RISK_LEVEL', 'TOTAL_SCORE', 'POP_5060'], ascending=False).reset_index(drop=True)


# ---- 그래프 ----
def plot_importance(imp, ct, dry_run):
    order = imp.sum(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    left = np.zeros(len(order))
    for gname in imp.columns:
        v = imp.loc[order, gname].values
        ax.barh(range(len(order)), v, left=left, color=ct[gname], label=gname, height=0.62, edgecolor='#fcfcfb', linewidth=2)
        left += v
    ax.set_yticks(range(len(order)), order)
    for i, tot in enumerate(left):
        ax.text(tot + 0.004, i, f'{tot:.2f}', va='center', fontsize=9, color=INK2)
    ax.set_xlabel('평균 |SHAP| (군집 분류 확률에 미치는 영향)')
    ax.set_title('변수 중요도 (SHAP)'); subtitle(ax, '막대 색 = 각 군집을 판별하는 데 쓰인 몫')
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.0), ncol=5); clean(ax, grid='x')
    save_fig(fig, '08_shap_importance', dry_run)


def plot_beeswarm(imp, X, rf, sv, ct, dry_run):
    """군집별 SHAP 분포 - 점 = 동, 가로 = 그 군집일 확률에 미치는 영향, 색 = 변수 값(빨강 = 위험 쪽)"""
    classes = list(rf.classes_)
    order = imp.sum(axis=1).sort_values().index
    fig, axes = plt.subplots(1, len(imp.columns), figsize=(3.9 * len(imp.columns), 4.6), sharey=True)
    rng_j = np.random.default_rng(0)
    for ax, gname in zip(axes, imp.columns):
        ci = classes.index(gname)
        for j, var in enumerate(order):
            vj = list(X.columns).index(var)
            ax.scatter(sv[:, vj, ci], j + rng_j.uniform(-0.22, 0.22, len(X)), c=X[var].values, cmap=DIV,
                       vmin=-2, vmax=2, s=22, edgecolor='none', zorder=3)
        ax.axvline(0, color='#c9c8c3', lw=1)
        ax.set_yticks(range(len(order)), order); ax.set_title(f'{gname} 판별', color=ct[gname])
        ax.set_xlabel('SHAP'); clean(ax, grid='x')
    subtitle(axes[0], '점 색: 파랑 = 안전 쪽 값, 빨강 = 위험 쪽 값')
    plt.tight_layout()
    save_fig(fig, '09_shap_beeswarm', dry_run)


def plot_representative(m, X, rf, explainer, sv, ct, dry_run):
    """대표 동: 군집별로 대리 모델이 그 군집일 확률을 가장 높게 준 동 (가장 전형적인 동)"""
    feat = m.feat
    classes = list(rf.classes_)
    proba = pd.DataFrame(rf.predict_proba(X.values), index=feat.index, columns=classes)
    p_own = pd.Series([proba.at[i, g] for i, g in zip(feat.index, feat.GROUP_T)], index=feat.index)
    rep = p_own.groupby(feat.GROUP_T).idxmax().reindex(m.TG).dropna()

    fig, axes = plt.subplots(1, len(rep), figsize=(4.0 * len(rep), 3.8))
    for ax, (gname, emd_cd) in zip(np.atleast_1d(axes), rep.items()):
        _, s, base = explain_dong(m, X, rf, explainer, sv, emd_cd)
        s = s[::-1]
        ax.barh(range(len(s)), s.values, color=[ct[gname] if v > 0 else '#c9c8c3' for v in s.values], height=0.62)
        ax.set_yticks(range(len(s)), s.index); ax.axvline(0, color=INK2, lw=0.8)
        lo, hi = min(s.min(), 0), max(s.max(), 0); pad = (hi - lo) * 0.25
        ax.set_xlim(lo - (pad if lo < 0 else 0.01), hi + pad)                 # 값 라벨이 축 이름과 겹치지 않게 여백
        for y_, v in enumerate(s.values):
            ax.text(v + (pad * 0.08 if v >= 0 else -pad * 0.08), y_, f'{v:+.2f}', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=8, color=INK2)
        ax.set_title(f'{feat.at[emd_cd, "SGG_NM"]} {feat.at[emd_cd, "EMD_NM"]} → {gname}', color=ct[gname])
        subtitle(ax, f'기준 확률 {base:.2f} → 예측 확률 {p_own[emd_cd]:.2f}')
        clean(ax, grid='x')
    plt.tight_layout()
    save_fig(fig, '10_shap_representative', dry_run)


def main():
    args = parse_args(__doc__.splitlines()[0])

    section('1. 최종 군집 → 대리 모델 + SHAP')
    m = build()
    X, rf, explainer, sv = fit_shap(m)

    section('2. 전체 변수 중요도 (군집별 평균 |SHAP|)')
    imp = importance(m, X, rf, sv)
    imp_t = importance_table(imp)
    print(imp_t.to_string(index=False))

    section('3. 동별 판별 근거')
    dong = dong_table(m, X, rf, explainer, sv)
    print(dong[['SGG_NM', 'EMD_NM', 'GROUP', 'RISK_LEVEL', 'P_OWN', 'REASON_1', 'REASON_2']].head(12).to_string(index=False))
    check('설명 결과', {
        '동별 결과 47개 동, 결측 없음': len(dong) == 47 and not dong.isna().any().any(),
        '모든 동에 판별 근거 1 있음': (dong.REASON_1 != '-').all(),
        '중요도 5개 변수 × 군집': imp.shape == (len(VARS), m.K),
    })

    section('4. 저장')
    save(imp_t, IMPORTANCE_FILE, dry_run=args.dry_run)
    save(dong, DONG_FILE, dtype=CODE_COLS, dry_run=args.dry_run)
    ct = {g: group_color(g) for g in m.TG}
    plot_importance(imp, ct, args.dry_run)
    plot_beeswarm(imp, X, rf, sv, ct, args.dry_run)
    plot_representative(m, X, rf, explainer, sv, ct, args.dry_run)


if __name__ == '__main__':
    main()
