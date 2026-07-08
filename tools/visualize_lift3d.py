# 이미지 위에 LiDAR 투영점을 깊이 색으로 오버레이 — lift3d/캘리브 검증용 (M2 게이트).
# 벽·기둥 윤곽과 점이 겹치면 캘리브·좌표계가 맞는 것. 어긋나면 T_cam_lidar 재확인.
# 사용: python tools/visualize_lift3d.py <img.png> <lidar_xyz.npy> [config.yaml]
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, '.')
from src.lift3d import project_lidar  # noqa: E402


def main(img_path, lidar_npy, cfg_path='config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    fx, fy, cx, cy = cfg['calib']['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.array(cfg['calib']['T_cam_lidar'], float).reshape(4, 4)
    img = cv2.imread(img_path)
    pts_l = np.load(lidar_npy)                     # (N,3) LiDAR 좌표
    pts_c = (T[:3, :3] @ pts_l.T).T + T[:3, 3]
    uv, z, _ = project_lidar(pts_c, K, img.shape[:2])
    zn = np.clip((z - 1) / 20, 0, 1)               # 1~21m → 색
    for (u, v), d in zip(uv.astype(int), zn):
        cv2.circle(img, (u, v), 1, (int(255 * (1 - d)), 64, int(255 * d)), -1)
    cv2.imwrite('lift3d_overlay.png', img)
    print(f'saved lift3d_overlay.png (투영점 {len(uv)}개) — 구조 윤곽과 겹치는지 육안 확인')


if __name__ == '__main__':
    main(*sys.argv[1:])
