# 카메라좌표계 LiDAR 점을 이미지에 투영하고, 특징점에 최근접 3D를 부여한다.
# 여기가 "이미지 특징점이 라이다의 정확한 3D를 얻는" 연결 고리 — 파이프라인의 고비.
import numpy as np
from scipy.spatial import cKDTree


def project_lidar(pts_cam, K, img_hw):
    """pts_cam(N,3, 카메라좌표) → 화면 안(z>0.1) 점의 (uv(M,2), z(M,), 원본 idx(M,))."""
    pts_cam = np.asarray(pts_cam, float)
    z = pts_cam[:, 2]
    front = z > 0.1
    p = pts_cam[front]
    if len(p) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, int)
    uv = (K @ (p / p[:, 2:3]).T).T[:, :2]
    h, w = img_hw
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    idx = np.flatnonzero(front)[inside]
    return uv[inside], p[inside, 2], idx


def assign_depth(kpts, pts_cam, K, img_hw, radius: float = 4.0):
    """각 키포인트에 반경 radius(픽셀) 내 최근접 LiDAR 투영점의 3D(카메라좌표)를 부여.

    반환: (p3d_cam (M,3), valid (M,) bool). 근처에 라이다 점이 없으면 valid=False —
    억지로 먼 점을 갖다 붙이면 PnP가 오염되므로 없는 건 없다고 한다.
    """
    kpts = np.asarray(kpts, float)
    p3d = np.zeros((len(kpts), 3))
    valid = np.zeros(len(kpts), bool)
    uv, _, idx = project_lidar(pts_cam, K, img_hw)
    if len(uv) == 0:
        return p3d, valid
    tree = cKDTree(uv)
    dist, j = tree.query(kpts, k=1)
    ok = dist <= radius
    p3d[ok] = np.asarray(pts_cam, float)[idx[j[ok]]]
    valid[ok] = True
    return p3d, valid
