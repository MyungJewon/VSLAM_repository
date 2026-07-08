# DB의 삼각측량 3D점들을 PLY로 내보내기 (희소 특징점 맵).
# 색: 높이(z) 컬러맵 — 바닥/천장/구조가 한눈에 구분된다.
# 사용: python tools/export_ply.py [config.yaml] [출력.ply]
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

sys.path.insert(0, '.')


def main(cfg_path='config.yaml', out_ply='db/sparse_map.ply'):
    cfg = yaml.safe_load(open(cfg_path))
    db = Path(cfg['db_dir'])

    pts = [np.load(f)['pts3d_w'] for f in sorted(db.glob('frame_*.npz'))]
    P = np.vstack([p for p in pts if len(p)])

    # 뷰 방해하는 극단 outlier 제거 (중앙값 기준 MAD 게이트)
    med = np.median(P, 0)
    mad = np.median(np.abs(P - med), 0) + 1e-6
    keep = (np.abs(P - med) < 15 * mad).all(1)
    P = P[keep]

    # 높이 → 색 (2~98 percentile로 정규화해 극단값에 안 눌리게)
    z = P[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / (hi - lo + 1e-9), 0, 1)
    C = (matplotlib.colormaps['turbo'](t)[:, :3] * 255).astype(np.uint8)

    with open(out_ply, 'w') as fp:
        fp.write('ply\nformat ascii 1.0\n'
                 f'element vertex {len(P)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 'end_header\n')
        for p, c in zip(P, C):
            fp.write(f'{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n')
    print(f'저장: {out_ply} ({len(P)}점, outlier {int((~keep).sum())}개 제거)')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'config.yaml', a[1] if len(a) > 1 else 'db/sparse_map.ply')
