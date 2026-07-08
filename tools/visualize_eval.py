# 평가 결과 시각화 — GT 궤적 위에 리로컬라이징 추정 위치를 겹쳐 그린다 (top-down XY).
#   회색 선   : GT 전체 궤적
#   초록 점   : 성공 쿼리 (추정 위치)
#   빨간 X    : 실패 쿼리 (GT 위치에 표시, 추정이 있으면 빨간 선으로 연결)
# 사용: python tools/visualize_eval.py [config.yaml] [출력.png]
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'AppleGothic'   # macOS 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, '.')


def main(cfg_path='config.yaml', out_png='db/eval_plot.png'):
    cfg = yaml.safe_load(open(cfg_path))
    gt = np.loadtxt(cfg['gt_path'])                     # t x y z qx qy qz qw
    rec = np.load(Path(cfg['db_dir']) / 'eval_results.npz')['records']
    t, g, e, ok = rec[:, 0], rec[:, 1:4], rec[:, 4:7], rec[:, 7].astype(bool)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                  gridspec_kw={'width_ratios': [1.4, 1]})
    # ── 왼쪽: 궤적 top-down ──
    ax.plot(gt[:, 1], gt[:, 2], '-', c='0.75', lw=1.5, label='GT 궤적')
    ax.scatter(e[ok, 0], e[ok, 1], s=14, c='#2ca02c', zorder=3,
               label=f'성공 {ok.sum()}')
    bad = ~ok
    for gi, ei in zip(g[bad], e[bad]):
        if np.isfinite(ei[0]):
            ax.plot([gi[0], ei[0]], [gi[1], ei[1]], 'r-', lw=0.8, alpha=0.6)
            ax.scatter(ei[0], ei[1], s=30, c='red', marker='x', zorder=4)
    ax.scatter(g[bad, 0], g[bad, 1], s=40, facecolors='none',
               edgecolors='red', zorder=4, label=f'실패 {bad.sum()}')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('리로컬라이징 결과 (top-down)')
    ax.legend()

    # ── 오른쪽: 시간축 위치 오차 ──
    err = np.linalg.norm(e - g, axis=1)
    fin = np.isfinite(err)
    ax2.semilogy(t[fin] - t[0], err[fin], '.-', ms=4, lw=0.5, c='#1f77b4')
    ax2.axhline(0.25, c='red', ls='--', lw=1, label='성공 기준 0.25m')
    ax2.grid(alpha=0.3, which='both')
    ax2.set_xlabel('시간 [s]'); ax2.set_ylabel('위치 오차 [m] (log)')
    med = np.median(err[fin])
    ax2.set_title(f'위치 오차 — 성공률 {100*ok.mean():.1f}%, 중앙값 {med*100:.1f}cm')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f'저장: {out_png}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'config.yaml', a[1] if len(a) > 1 else 'db/eval_plot.png')
