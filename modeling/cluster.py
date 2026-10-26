"""군집: 기본 모델 · 최종 모델 선택·학습 · 군집 프로파일 · 위험도 5단계 → artifacts/cluster_profile.csv

1. 기본 모델: (인구축, 지역축) 2D 단계 점수 + K-means. K 선택(엘보우·실루엣) 후 K=4
2. 최종 모델 선택: tuning_results.csv에서 2D 공간별 종합순위 1위 → 그중 부트스트랩 안정성이 높은 쪽
3. 군집 이름: 군집별 단계 점수 평균이 중앙(3) 이상이면 H, 미만이면 L → HH / HL / LH / LL
   (같은 사분면이 둘이면 두 축 합이 큰 쪽이 1번, 예: HH1 = 더 위험)
4. 프로파일: 군집별 원값·단계 점수 평균
5. 위험도 5단계: 종합 점수(인구축 + 지역축 단계 점수, 2~10점)를 47개 동 기준 5분위 (동점은 동 코드 순서)

계산 함수(build 등)는 validate.py · xai_explain.py가 불러 쓴다.

실행: python modeling/cluster.py [--dry-run]   (먼저 tune.py로 tuning_results.csv를 만들어야 함)
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common import (ART, BASE, MID, N_INIT_BASE, QCOLS, REGIONS, SEED, SHORT, TUNING_FILE, VARS, check, fit_base,
                    fit_predict, group_order, input_spaces, load_data, name_clusters, parse_args, require, save,
                    section, space_kind)
from plot_style import C, DIV, INK, INK2, MUTED, SURF, clean, group_color, label_points, plt, save_fig, subtitle

PROFILE_FILE = ART / 'cluster_profile.csv'


# ---- 계산 (validate.py · xai_explain.py 공통) ----
def load_tuning():
    require(TUNING_FILE)
    return pd.read_csv(TUNING_FILE, encoding='utf-8-sig')


def base_model(feat, X):
    """기본 모델: 2D 단계 점수 + K-means (K=4, 초기값 50회) → CLUSTER, GROUP"""
    km = fit_base(X)
    feat['CLUSTER'] = km.labels_
    cent = pd.DataFrame(km.cluster_centers_, columns=['POP_SCORE', 'REG_SCORE'])
    cent['NAME'] = name_clusters(cent[['POP_SCORE', 'REG_SCORE']])
    feat['GROUP'] = feat.CLUSTER.map(cent.NAME)
    return cent


def select_final(tune):
    """해석 가능성(사분면 이름)을 위해 2D 공간만 후보 → 공간별 종합순위 1위 → 안정성 ARI가 높은 쪽"""
    winners = (tune[tune.사용가능 & tune.입력공간.str.startswith('2D')]
               .sort_values(['종합순위점수', '안정성ARI'], ascending=[True, False]).groupby('입력공간').head(1))
    best = winners.sort_values('안정성ARI', ascending=False).iloc[0]
    base_row = tune[(tune.입력공간 == BASE[0]) & (tune.알고리즘 == BASE[1]) & (tune.K == BASE[2])].iloc[0]
    return winners, best, base_row


def final_model(feat, Xb, algo, k):
    """최종 모델 학습 → CLUSTER_T, AX_POP·AX_REG(입력 축), GROUP_T.
    이름은 기본 모델·논문과 같은 기준(단계 점수 1~5의 중앙 3)으로 사분면 판정"""
    feat['CLUSTER_T'] = fit_predict(algo, Xb, k)
    feat['AX_POP'], feat['AX_REG'] = Xb[:, 0], Xb[:, 1]
    ct = feat.groupby('CLUSTER_T')[['POP_SCORE', 'REG_SCORE']].mean()
    ct['NAME'] = name_clusters(ct)
    feat['GROUP_T'] = feat.CLUSTER_T.map(ct.NAME)
    return ct


def add_risk_level(feat):
    """종합 점수 = 단계 점수 합 (2~10점) → 47개 동 기준 5분위 (5 = 상위 20%, 최우선 개입)"""
    feat['TOTAL_SCORE'] = feat.POP_SCORE + feat.REG_SCORE
    feat['RISK_LEVEL'] = pd.qcut(feat.TOTAL_SCORE.rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)


def build():
    """features.csv + tuning_results.csv → 기본·최종 군집과 위험도가 붙은 결과 (m)

    m.feat(동별 결과), m.d(입력 데이터), m.spaces, m.tune, m.winners, m.best, m.base_row,
    m.SPACE·ALGO·K·KIND(최종 모델), m.TG(최종 군집 순서), m.cent(기본 모델 중심), m.ct(최종 모델 군집 평균)
    """
    d = load_data()
    feat = d.feat
    spaces = input_spaces(d)
    tune = load_tuning()

    cent = base_model(feat, spaces[BASE[0]])
    winners, best, base_row = select_final(tune)
    SPACE, ALGO, K = best.입력공간, best.알고리즘, int(best.K)
    ct = final_model(feat, spaces[SPACE], ALGO, K)
    add_risk_level(feat)
    return SimpleNamespace(feat=feat, d=d, spaces=spaces, tune=tune, winners=winners, best=best,
                           base_row=base_row, SPACE=SPACE, ALGO=ALGO, K=K, KIND=space_kind(SPACE),
                           TG=group_order(ct.NAME), cent=cent, ct=ct)


def cluster_profile(m):
    """군집별 원값·단계 점수 평균 (artifacts/cluster_profile.csv 형식)"""
    feat = m.feat
    prof = feat.groupby('GROUP_T')[VARS + QCOLS + ['POP_SCORE', 'REG_SCORE', 'TOT_POP_CNT', 'POP_5060']].mean()
    prof.insert(0, 'N_DONG', feat.GROUP_T.value_counts())
    prof = prof.reindex(m.TG)
    prof[VARS] = prof[VARS].round(4)
    prof['POP_DENSITY'] = prof.POP_DENSITY.round(1)
    prof[QCOLS + ['POP_SCORE', 'REG_SCORE']] = prof[QCOLS + ['POP_SCORE', 'REG_SCORE']].round(2)
    prof[['TOT_POP_CNT', 'POP_5060']] = prof[['TOT_POP_CNT', 'POP_5060']].round(1)
    return prof.rename_axis('GROUP').reset_index()


def members(feat, col, groups):
    for g in groups:
        d = feat[feat[col] == g].sort_values(['SGG_NM', 'POP_SCORE'], ascending=[True, False])
        print(f'   ■ {g} ({len(d)}개 동)')
        for nm in REGIONS:
            names = d[d.SGG_NM == nm].EMD_NM.tolist()
            if names:
                print(f'      {nm}: {", ".join(names)}')


# ---- 그래프 ----
def plot_k_selection(X, dry_run):
    """기본 모델 K 선택: 엘보우 + 실루엣 (K=2~8)"""
    Ks = range(2, 9)
    inertia, sil = [], []
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=N_INIT_BASE, random_state=SEED).fit(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    print(pd.DataFrame({'K': list(Ks), 'inertia': inertia, 'silhouette': sil}).set_index('K').T.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(Ks), inertia, marker='o', color='#2a78d6', markeredgecolor=SURF, markeredgewidth=1.5)
    axes[0].set_title('엘보우 (군집 내 제곱합)'); axes[0].set_xlabel('K')
    axes[1].plot(list(Ks), sil, marker='o', color='#eb6834', markeredgecolor=SURF, markeredgewidth=1.5)
    axes[1].set_title('실루엣 계수'); axes[1].set_xlabel('K')
    for ax in axes:
        ax.axvline(BASE[2], color=MUTED, lw=1, ls='--'); clean(ax)
    subtitle(axes[0], f'점선 = K={BASE[2]} (논문과 동일)'); subtitle(axes[1], '높을수록 군집이 잘 분리됨')
    plt.tight_layout()
    save_fig(fig, '01_k_selection', dry_run)


def plot_base_model(m, dry_run):
    feat, cent = m.feat, m.cent
    groups = group_order(cent.NAME)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.axhline(MID, color='#c9c8c3', lw=1); ax.axvline(MID, color='#c9c8c3', lw=1)
    jit = np.random.default_rng(1).uniform(-0.06, 0.06, (len(feat), 2))     # 같은 점수 겹침 방지용 미세 흔들기
    for g in groups:
        for nm, mk in [('강남구', 'o'), ('춘천시', 's')]:
            d = feat[(feat.GROUP == g) & (feat.SGG_NM == nm)]
            j = jit[feat.index.get_indexer(d.index)]
            ax.scatter(d.POP_SCORE + j[:, 0], d.REG_SCORE + j[:, 1], s=70, marker=mk, color=group_color(g),
                       edgecolor=SURF, linewidth=1.3, zorder=3)
    for _, r in cent.iterrows():
        ax.scatter(r.POP_SCORE, r.REG_SCORE, s=320, marker='X', color=group_color(r.NAME), edgecolor=INK, linewidth=1.2, zorder=4)
        ax.annotate(r.NAME, (r.POP_SCORE, r.REG_SCORE), xytext=(12, 10), textcoords='offset points',
                    fontsize=14, fontweight='bold', color=group_color(r.NAME))
    ax.set_xlim(0.8, 5.2); ax.set_ylim(0.8, 5.2)
    ext = pd.concat([feat.nlargest(4, 'POP_SCORE'), feat.nlargest(4, 'REG_SCORE'), feat.nsmallest(2, 'POP_SCORE')])
    ext = ext[~ext.index.duplicated()]
    label_points(ax, ext.POP_SCORE.tolist(), ext.REG_SCORE.tolist(), ext.EMD_NM.tolist(), gap=0.04)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=MUTED, label='강남구'),
                       Line2D([], [], marker='s', ls='', color=MUTED, label='춘천시'),
                       Line2D([], [], marker='X', ls='', color=INK, label='군집 중심')],
              loc='lower right', bbox_to_anchor=(1, 1.0), ncol=3)
    ax.set_xlabel('인구축 점수 (1~5, 높을수록 위험)'); ax.set_ylabel('지역축 점수 (1~5, 높을수록 위험)')
    ax.set_title('기본 모델: 2차원 K-means 결과'); subtitle(ax, 'H/L 첫 글자 = 인구축, 둘째 글자 = 지역축')
    ax.grid(False); ax.tick_params(length=0)
    save_fig(fig, '02_base_model', dry_run)


def plot_profile_heatmap(feat, col, groups, title, name, dry_run):
    """군집별 변수 평균 단계 점수 (1~5, 5 = 가장 위험)"""
    prof = feat.groupby(col)[QCOLS + ['POP_SCORE', 'REG_SCORE']].mean().reindex(groups)
    prof.columns = [SHORT[k] for k in VARS] + ['인구축', '지역축']
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(groups) + 1.2))
    im = ax.imshow(prof, cmap=DIV, vmin=1, vmax=5, aspect='auto')
    ax.set_xticks(range(prof.shape[1]), prof.columns)
    ax.set_yticks(range(len(groups)), [f'{g} ({(feat[col] == g).sum()}개 동)' for g in groups])
    for i in range(prof.shape[0]):
        for j in range(prof.shape[1]):
            ax.text(j, i, f'{prof.iloc[i, j]:.1f}', ha='center', va='center', fontsize=10,
                    color='white' if abs(prof.iloc[i, j] - 3) > 1.3 else INK)
    ax.axvline(4.5, color=SURF, lw=4); ax.xaxis.tick_top(); ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, pad=34); fig.colorbar(im, ax=ax, shrink=0.9, label='1 = 안전 · 5 = 위험')
    save_fig(fig, name, dry_run)


def plot_base_vs_final(m, dry_run):
    feat = m.feat
    mid_b = 0.0 if m.KIND == 'z' else MID              # 그래프의 축 중앙선 (z-score 중앙 = 0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    panels = [(axes[0], 'POP_SCORE', 'REG_SCORE', 'GROUP', MID, f'기본 모델 ({BASE[0]} · {BASE[1]}, K={BASE[2]})', '1~5점'),
              (axes[1], 'AX_POP', 'AX_REG', 'GROUP_T', mid_b, f'최종 모델 ({m.SPACE} · {m.ALGO}, K={m.K})',
               '시군구 내 z' if m.KIND == 'z' else '1~5점')]
    for ax, xc, yc, gc, mid, title, unit in panels:
        ax.axhline(mid, color='#c9c8c3', lw=1); ax.axvline(mid, color='#c9c8c3', lw=1)
        j = np.random.default_rng(1).uniform(-0.06, 0.06, (len(feat), 2)) * (1 if unit == '1~5점' else 0)   # 동점 겹침 방지
        for g in group_order(feat[gc]):
            mask = (feat[gc] == g).values
            ax.scatter(feat.loc[mask, xc] + j[mask, 0], feat.loc[mask, yc] + j[mask, 1], s=60, color=group_color(g),
                       edgecolor=SURF, linewidth=1.2, label=f'{g} ({mask.sum()})', zorder=3)
        ax.set_xlabel(f'인구축 ({unit}, 높을수록 위험)'); ax.set_ylabel(f'지역축 ({unit}, 높을수록 위험)')
        ax.set_title(title); ax.grid(False); ax.tick_params(length=0)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1.0), ncol=len(set(feat[gc])))
    top = (feat.AX_POP + feat.AX_REG).sort_values(ascending=False).index[:8]
    label_points(axes[1], feat.loc[top, 'AX_POP'], feat.loc[top, 'AX_REG'], feat.loc[top, 'EMD_NM'], gap=0.03)
    plt.tight_layout()
    save_fig(fig, '05_base_vs_final', dry_run)


def plot_risk_level(feat, dry_run):
    lv = pd.crosstab(feat.RISK_LEVEL, feat.SGG_NM)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(5); w = 0.38
    for i, nm in enumerate(REGIONS):
        bars = ax.bar(x + (i - 0.5) * w, lv[nm], width=w - 0.03, color=C[nm], label=nm)
        ax.bar_label(bars, padding=2, fontsize=8, color=INK2)
    ax.set_xticks(x, [f'{lvl}단계' for lvl in range(1, 6)]); ax.set_ylabel('동 수')
    ax.set_title('위험도 단계별 동 수'); subtitle(ax, '5단계 = 최우선 개입'); ax.legend(); clean(ax)
    save_fig(fig, '07_risk_level', dry_run)


def main():
    args = parse_args(__doc__.splitlines()[0])

    section('1. 입력 데이터 · 튜닝 결과 → 기본·최종 모델')
    m = build()
    feat = m.feat
    print(f'   최종 모델: {m.SPACE} / {m.ALGO} / K = {m.K}')
    cmp = pd.concat([m.tune[(m.tune.입력공간 == BASE[0]) & (m.tune.알고리즘 == BASE[1]) & (m.tune.K == BASE[2])]
                     .assign(구분='기본 모델'), m.winners.assign(구분='공간별 1위')])
    print(cmp.set_index(['구분', '입력공간', '알고리즘', 'K'])
             [['실루엣', 'DB지수', '안정성ARI', '최소군집', '기본모델과ARI', '종합순위점수']].round(3).to_string())

    section(f'2. 기본 모델 ({BASE[0]} · {BASE[1]}, K={BASE[2]})')
    print(m.cent.round(2).to_string())
    members(feat, 'GROUP', group_order(m.cent.NAME))

    section(f'3. 최종 모델 ({m.SPACE} · {m.ALGO}, K={m.K})')
    print(m.ct.round(2).to_string())
    print(pd.crosstab(feat.GROUP_T, feat.GROUP, margins=True, margins_name='합계').reindex(m.TG + ['합계']).to_string())
    members(feat, 'GROUP_T', m.TG)

    section('4. 위험도 5단계')
    print(pd.crosstab(feat.GROUP_T, feat.RISK_LEVEL).reindex(m.TG).to_string())
    top = (feat[feat.RISK_LEVEL >= 4].sort_values(['RISK_LEVEL', 'POP_5060'], ascending=[False, False])
           [['SGG_NM', 'EMD_NM', 'GROUP_T', 'RISK_LEVEL', 'POP_SCORE', 'REG_SCORE', 'TOTAL_SCORE', 'POP_5060']])
    print(top.round(2).to_string(index=False))

    prof = cluster_profile(m)
    check('군집 결과', {
        f'최종 군집 {m.K}개, 모두 이름 있음': feat.GROUP_T.nunique() == m.K and feat.GROUP_T.notna().all(),
        '위험도 5단계 각 9~10개 동': feat.RISK_LEVEL.value_counts().between(9, 10).all(),
        '프로파일 동 수 합 = 47': prof.N_DONG.sum() == 47,
    })

    section('5. 저장')
    save(prof, PROFILE_FILE, dry_run=args.dry_run)
    plot_k_selection(m.spaces[BASE[0]], args.dry_run)
    plot_base_model(m, args.dry_run)
    plot_profile_heatmap(feat, 'GROUP', group_order(m.cent.NAME), '기본 모델 군집별 평균 단계 점수', '03_base_profile', args.dry_run)
    plot_base_vs_final(m, args.dry_run)
    plot_profile_heatmap(feat, 'GROUP_T', m.TG, '최종 모델 군집별 평균 단계 점수', '06_final_profile', args.dry_run)
    plot_risk_level(feat, args.dry_run)


if __name__ == '__main__':
    main()
