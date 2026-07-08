import cv2
import numpy as np

from src.pnp import solve_pnp


def test_recovers_known_pose():
    rng = np.random.default_rng(0)
    pts3d = rng.uniform([-2, -2, 4], [2, 2, 8], (50, 3))  # 카메라 앞 3D 점
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    rvec_true = np.array([0.1, -0.2, 0.05])
    tvec_true = np.array([0.3, -0.1, 0.5])
    pts2d, _ = cv2.projectPoints(pts3d, rvec_true, tvec_true, K, None)
    res = solve_pnp(pts2d.reshape(-1, 2), pts3d, K, dist=None)
    assert res is not None
    T = res['T_cam_world']
    R_true, _ = cv2.Rodrigues(rvec_true)
    assert np.allclose(T[:3, :3], R_true, atol=1e-4)
    assert np.allclose(T[:3, 3], tvec_true, atol=1e-3)
    assert res['inliers'] >= 45


def test_too_few_points_returns_none():
    K = np.eye(3)
    assert solve_pnp(np.zeros((3, 2)), np.zeros((3, 3)), K, None) is None


def test_outliers_rejected():
    # 20% 아웃라이어를 섞어도 pose 복원 + inlier로 걸러지는지
    rng = np.random.default_rng(1)
    pts3d = rng.uniform([-2, -2, 4], [2, 2, 8], (100, 3))
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    rvec = np.array([0.05, 0.1, -0.02])
    tvec = np.array([0.1, 0.2, 0.3])
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    pts2d = pts2d.reshape(-1, 2)
    pts2d[:20] += rng.uniform(50, 100, (20, 2))  # 아웃라이어
    res = solve_pnp(pts2d, pts3d, K, None)
    assert res is not None
    R_true, _ = cv2.Rodrigues(rvec)
    assert np.allclose(res['T_cam_world'][:3, :3], R_true, atol=1e-3)
    assert 70 <= res['inliers'] <= 85  # 아웃라이어 20개는 제외됐어야
