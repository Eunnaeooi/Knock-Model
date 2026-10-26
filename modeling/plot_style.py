"""그래프 스타일 — 색, rcParams, 보조 함수, PNG 저장.

그래프는 화면에 띄우지 않고 artifacts/figures/<이름>.png 로 저장한다. 한글 표시에 나눔고딕 글꼴이 필요하다.
"""
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import FIG, ROOT  # noqa: E402

INK, INK2, MUTED, GRID, SURF, BAND = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1', '#fcfcfb', '#f0efec'
C   = {'강남구': '#2a78d6', '춘천시': '#eb6834'}                       # 지역
CL  = {'HH': '#e34948', 'HL': '#eda100', 'LH': '#4a3aa7', 'MM': '#8a8984', 'LL': '#1baf7a'}   # 군집 (고정)
LV  = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']           # 위험도 1~5단계 (연→진)
DIV = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
SPLIT = {'HH': '#e87ba4', 'HL': '#f5c96b', 'LH': '#9a8fd6', 'LL': '#7fd3b0'}     # 같은 사분면의 두 번째 군집

plt.rcParams.update({
    'font.family': 'NanumGothic', 'axes.unicode_minus': False, 'figure.dpi': 110,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': '#c9c8c3', 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 26,
    'axes.titlecolor': INK, 'axes.labelcolor': INK2, 'axes.labelsize': 10,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 9, 'lines.linewidth': 2, 'lines.markersize': 7,
})


def clean(ax, grid='y'):
    ax.grid(axis='x' if grid == 'y' else 'y', visible=False); ax.tick_params(length=0)


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=INK2, va='bottom')


def label_points(ax, xs, ys, names, gap=0.05, **kw):
    """점 이름 표시 - 이미 표시한 이름과 너무 가까우면 건너뜀 (gap = 축 범위 대비 거리)"""
    xr = np.ptp(ax.get_xlim()); yr = np.ptp(ax.get_ylim())
    placed = []
    for x, y, n in zip(xs, ys, names):
        if any(abs(x - px) / xr < gap and abs(y - py) / yr < gap for px, py in placed):
            continue
        placed.append((x, y))
        ax.annotate(n, (x, y), xytext=(6, 4), textcoords='offset points', fontsize=8, color=INK2, **kw)


def group_color(n):
    return CL[n] if n in CL else (CL[n[:2]] if n.endswith('1') else SPLIT[n[:2]])


def save_fig(fig, name, dry_run=False):
    """artifacts/figures/<name>.png 로 저장하고 닫는다."""
    path = FIG / f'{name}.png'
    if not dry_run:
        FIG.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # 글꼴에 없는 기호 경고 숨김
            fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'   {str(path.relative_to(ROOT)):40s} | {"저장 안 함(--dry-run)" if dry_run else "저장"}')
