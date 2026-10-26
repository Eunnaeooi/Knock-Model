"""튜닝: 입력 공간 4 × 알고리즘 4 × K 4 = 64개 조합 학습·비교 → artifacts/tuning/tuning_results.csv

| 튜닝 항목 | 후보                                                     |
|-----------|----------------------------------------------------------|
| 입력 공간 | 2D 단계점수 (기본) · 2D z-score · 5D 단계점수 · 5D z-score |
| 알고리즘  | K-means, Ward, Average, GMM                              |
| 군집 수 K | 3, 4, 5, 6                                               |

평가 지표: 실루엣(↑), Davies-Bouldin(↓), 부트스트랩 안정성 ARI(↑, 100회), 최소 군집 크기(4개 동 이상만 사용)
  - 실루엣·DB는 입력 공간마다 척도가 달라 '입력 공간 안 순위'로 종합한다 (종합순위점수, 작을수록 좋음)
  - 기본모델과ARI: 기본 모델(2D 단계점수 · K-means, K=4, 초기값 50회) 군집과의 일치도
  - 64개 조합의 K-means는 같은 조건(초기값 30회)으로 학습한다

결과는 cluster.py가 읽어 최종 모델을 고른다.

실행: python modeling/tune.py [--dry-run]   (부트스트랩 때문에 10분 이상)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from common import (ALGOS, BASE, KS, MIN_CLUSTER, N_BOOT, SEED, TUNING_FILE, check, fit_base, fit_predict,
                    input_spaces, load_data, parse_args, section, save)
from plot_style import MUTED, SURF, clean, plt, save_fig, subtitle


def nearest_centroid(Xfit, lab, Xall):
    """부트스트랩 표본에서 만든 군집 중심으로 전체 동을 배정 (알고리즘 공통)"""
    cs = np.array([Xfit[lab == c].mean(axis=0) for c in np.unique(lab)])
    return np.argmin(((Xall[:, None, :] - cs[None]) ** 2).sum(-1), axis=1)


def stability(algo, Xs, k, base, n_boot=N_BOOT):
    """동을 복원추출해 다시 학습 → 전체 동 재배정 → 원래 군집과의 ARI 평균"""
    rng = np.random.default_rng(SEED); out = []
    for b in range(n_boot):
        idx = rng.choice(len(Xs), len(Xs), replace=True)
        lab = fit_predict(algo, Xs[idx], k, seed=b)
        out.append(adjusted_rand_score(base, nearest_centroid(Xs[idx], lab, Xs)))
    return np.mean(out)


def run_tuning(spaces, base_labels):
    rows = []
    for sp, Xs in spaces.items():
        for algo in ALGOS:
            for k in KS:
                lab = fit_predict(algo, Xs, k)
                if len(np.unique(lab)) < k:            # 군집이 덜 만들어진 경우 제외
                    continue
                rows.append({'입력공간': sp, '알고리즘': algo, 'K': k,
                             '실루엣': silhouette_score(Xs, lab),
                             'DB지수': davies_bouldin_score(Xs, lab),
                             '안정성ARI': stability(algo, Xs, k, lab),
                             '최소군집': np.bincount(lab).min(),
                             '기본모델과ARI': adjusted_rand_score(base_labels, lab)})
            print(f'   {sp:10s} {algo:8s} 완료')
    tune = pd.DataFrame(rows)

    # 종합 순위: 입력공간 안에서 실루엣·DB·안정성 순위 평균 → 최소 군집 4개 동 이상만 사용 가능
    tune['순위_실루엣'] = tune.groupby('입력공간').실루엣.rank(ascending=False)
    tune['순위_DB'] = tune.groupby('입력공간').DB지수.rank(ascending=True)
    tune['순위_안정성'] = tune.groupby('입력공간').안정성ARI.rank(ascending=False)
    tune['종합순위점수'] = tune[['순위_실루엣', '순위_DB', '순위_안정성']].mean(axis=1)
    tune['사용가능'] = tune.최소군집 >= MIN_CLUSTER
    return tune


def plot_stability(tune, spaces, dry_run):
    """입력 공간별 · 알고리즘별 안정성 비교 (K=3~6)"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    AC = {'K-means': '#2a78d6', 'Ward': '#eb6834', 'Average': '#1baf7a', 'GMM': '#4a3aa7'}
    for ax, sp in zip(axes, spaces):
        d = tune[tune.입력공간 == sp]
        for algo in ALGOS:
            e = d[d.알고리즘 == algo].sort_values('K')
            ax.plot(e.K, e.안정성ARI, marker='o', color=AC[algo], label=algo, markeredgecolor=SURF, markeredgewidth=1.2)
            small = e[~e.사용가능]
            ax.scatter(small.K, small.안정성ARI, s=90, facecolor='none', edgecolor=MUTED, linewidth=1.2, zorder=4)
        ax.set_xticks(KS); ax.set_xlabel('K'); ax.set_ylim(0, 1.02)
        ax.set_title(sp, fontsize=11); clean(ax)
    axes[0].set_ylabel('부트스트랩 안정성 (ARI)')
    subtitle(axes[0], f'회색 원 = 최소 군집 {MIN_CLUSTER - 1}개 동 이하 (제외)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', ncol=4, bbox_to_anchor=(1, 1.08))
    plt.tight_layout()
    save_fig(fig, '04_tuning_stability', dry_run)


def main():
    args = parse_args(__doc__.splitlines()[0])

    section('1. 입력 데이터 · 입력 공간 4가지')
    d = load_data()
    spaces = input_spaces(d)
    base_labels = fit_base(spaces[BASE[0]]).labels_                     # 기본 모델 (초기값 50회)
    print('   ' + ' | '.join(f'{sp} {Xs.shape}' for sp, Xs in spaces.items()))

    section(f'2. 전체 조합 학습 ({len(spaces)} 공간 × {len(ALGOS)} 알고리즘 × {len(KS)} K, 부트스트랩 {N_BOOT}회)')
    tune = run_tuning(spaces, base_labels)
    base = tune[(tune.입력공간 == BASE[0]) & (tune.알고리즘 == BASE[1]) & (tune.K == BASE[2])]
    check(f'튜닝 결과 {tune.shape}', {
        '학습한 조합 64개 이하, 결측 없음': len(tune) <= 64 and not tune.isna().any().any(),
        '기본 모델 조합 포함': len(base) == 1,
    })

    cols = ['입력공간', '알고리즘', 'K', '실루엣', 'DB지수', '안정성ARI', '최소군집', '기본모델과ARI', '종합순위점수']
    top = (tune[tune.사용가능].sort_values(['종합순위점수', '안정성ARI'], ascending=[True, False])
               .groupby('입력공간').head(3)[cols].sort_values(['입력공간', '종합순위점수']))
    print('\n[입력 공간별 종합 상위 3개 (최소 군집 4개 동 이상)]')
    print(top.round(3).to_string(index=False))

    section('3. 저장')
    save(tune, TUNING_FILE, dry_run=args.dry_run)
    plot_stability(tune, spaces, args.dry_run)


if __name__ == '__main__':
    main()
