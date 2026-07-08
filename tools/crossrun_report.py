# 크로스-run 공정 채점 — eval_results.npz를 재채점한다 (평가 재실행 불필요).
#   1. 커버리지 필터: 쿼리 GT가 run1(DB) 궤적에서 cover_r 이내인 것만 채점
#   2. 좌표계 정렬: run1/run2 GT 등록 어긋남을 Umeyama(SE3)로 보정한 오차 병기
# 주의: 저장된 records에는 회전오차가 없어 위치 기준(<0.25m)만 채점한다.
# 사용: python tools/crossrun_report.py [config.yaml] [--cover-r 3.0]
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, '.')


def umeyama(src, dst):
    """src → dst 강체변환 (R, t). scale은 1로 고정 (둘 다 metric)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def main(cfg_path='config.yaml', cover_r=3.0):
    cfg = yaml.safe_load(open(cfg_path))
    rec = np.load(Path(cfg['db_dir']) / 'eval_results.npz')['records']
    g, e = rec[:, 1:4], rec[:, 4:7]
    db_traj = np.loadtxt(cfg['gt_path'])[:, 1:4]     # run1 = DB 커버 영역

    # 1. 커버리지: 쿼리 GT가 DB 궤적에서 cover_r 이내
    d_cov, _ = cKDTree(db_traj).query(g)
    cov = d_cov < cover_r
    fin = np.isfinite(e[:, 0])
    print(f'전체 쿼리 {len(g)} | 커버 영역 내 {cov.sum()} '
          f'(반경 {cover_r}m) | 영역 밖 {(~cov).sum()} → 채점 제외')
    print(f'커버 영역 내 추정 반환: {(cov & fin).sum()}/{cov.sum()} '
          f'({100 * (cov & fin).sum() / max(cov.sum(), 1):.1f}%) — 나머지는 None')
    # 영역 밖에서 None으로 거절한 비율 (높을수록 좋음 — 모르는 곳을 아는 척 안 함)
    if (~cov).sum():
        rej = (~cov & ~fin).sum() / (~cov).sum()
        print(f'영역 밖 거절률: {100 * rej:.1f}% (None이 정답인 구간)')

    m = cov & fin
    err_raw = np.linalg.norm(e[m] - g[m], axis=1)

    # 2. GT 좌표계 정렬 (gross error 제외하고 피팅, 전체에 적용)
    inl = err_raw < 1.5
    R, t = umeyama(e[m][inl], g[m][inl])
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    err_al = np.linalg.norm((R @ e[m].T).T + t - g[m], axis=1)
    print(f'\nGT 좌표계 어긋남 추정: 회전 {ang:.2f}°, '
          f'평행이동 {np.linalg.norm(t):.3f}m')

    for name, err in [('원본  ', err_raw), ('정렬 후', err_al)]:
        ok = (err < 0.25).sum()
        print(f'{name}: 위치<0.25m {ok}/{m.sum()} '
              f'({100 * ok / max(m.sum(), 1):.1f}%) | '
              f'중앙값 {np.median(err):.3f}m | 90pct {np.percentile(err, 90):.3f}m')


if __name__ == '__main__':
    a = sys.argv[1:]
    r = float(a[a.index('--cover-r') + 1]) if '--cover-r' in a else 3.0
    cfg = a[0] if a and not a[0].startswith('--') else 'config.yaml'
    main(cfg, r)
