import cv2
import numpy as np

from src.triangulate import triangulate_pairs

K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)


def _project(pts_w, T_wc):
    T_cw = np.linalg.inv(T_wc)
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    uv, _ = cv2.projectPoints(pts_w, rvec, T_cw[:3, 3], K, None)
    return uv.reshape(-1, 2)


def test_recovers_known_3d():
    # 알려진 월드 3D 점들을 두 시점에 투영 → 삼각측량으로 원복
    rng = np.random.default_rng(0)
    pts_w = rng.uniform([-2, -2, 4], [2, 2, 10], (40, 3))
    T_a = np.eye(4)                       # 원점
    T_b = np.eye(4); T_b[0, 3] = 0.5      # x로 0.5m 이동 (baseline)
    uv_a, uv_b = _project(pts_w, T_a), _project(pts_w, T_b)
    p3d, valid = triangulate_pairs(uv_a, uv_b, K, T_a, T_b)
    assert valid.all()
    assert np.allclose(p3d, pts_w, atol=1e-3)


def test_small_baseline_rejected():
    # 거의 안 움직인 두 시점 → 시차 부족 → 전부 무효
    pts_w = np.array([[0.0, 0.0, 5.0]])
    T_a = np.eye(4)
    T_b = np.eye(4); T_b[0, 3] = 0.001    # 1mm
    uv_a, uv_b = _project(pts_w, T_a), _project(pts_w, T_b)
    _, valid = triangulate_pairs(uv_a, uv_b, K, T_a, T_b)
    assert not valid.any()


def test_bad_match_rejected():
    # 한쪽 픽셀을 엉뚱한 곳으로 (잘못된 매칭) → 재투영 오차로 기각
    pts_w = np.array([[0.0, 0.0, 5.0], [1.0, 0.5, 6.0]])
    T_a = np.eye(4)
    T_b = np.eye(4); T_b[0, 3] = 0.5
    uv_a, uv_b = _project(pts_w, T_a), _project(pts_w, T_b)
    uv_b[0] += [80, 40]                   # 첫 매칭 오염
    _, valid = triangulate_pairs(uv_a, uv_b, K, T_a, T_b)
    assert not valid[0] and valid[1]


def test_far_point_rejected():
    # 깊이 > ratio×baseline 원거리 점은 불확실 → 기각 (0.5m×50 = 25m 한계)
    pts_w = np.array([[0.0, 0.0, 200.0]])
    T_a = np.eye(4)
    T_b = np.eye(4); T_b[0, 3] = 0.5
    uv_a, uv_b = _project(pts_w, T_a), _project(pts_w, T_b)
    _, valid = triangulate_pairs(uv_a, uv_b, K, T_a, T_b, max_depth_ratio=50.0)
    assert not valid.any()


def test_merge_multiview_consistency():
    from src.triangulate import merge_multiview
    cam = np.zeros(3)
    ests = {
        0: [np.array([1.0, 0, 5.0]), np.array([1.01, 0, 5.02])],  # 일치 → 평균
        1: [np.array([1.0, 0, 5.0]), np.array([3.0, 0, 9.0])],    # 불일치 → 기각
        2: [np.array([2.0, 1, 4.0])],                             # 단일 → 유지
    }
    idxs, pts = merge_multiview(ests, cam)
    assert set(idxs) == {0, 2}
    assert np.allclose(pts[list(idxs).index(0)], [1.005, 0, 5.01])
