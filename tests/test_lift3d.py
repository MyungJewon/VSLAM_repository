import numpy as np

from src.lift3d import project_lidar, assign_depth

K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)


def test_project_center():
    # 카메라 정면 5m 점 → 주점(50,50)
    pts_cam = np.array([[0, 0, 5.0]])
    uv, z, idx = project_lidar(pts_cam, K, (100, 100))
    assert np.allclose(uv[0], [50, 50], atol=1e-6)
    assert z[0] == 5.0 and idx[0] == 0


def test_behind_camera_excluded():
    pts_cam = np.array([[0, 0, -5.0], [0, 0, 5.0]])
    uv, _, idx = project_lidar(pts_cam, K, (100, 100))
    assert len(uv) == 1 and idx[0] == 1  # 뒤쪽 점은 제외


def test_assign_depth_nearest():
    pts_cam = np.array([[0, 0, 5.0]])          # (50,50)에 투영됨
    kpts = np.array([[51.0, 50.0], [90.0, 90.0]])
    p3d, valid = assign_depth(kpts, pts_cam, K, (100, 100), radius=3.0)
    assert valid[0] and not valid[1]           # 두 번째는 근처에 라이다 점 없음
    assert np.allclose(p3d[0], [0, 0, 5.0], atol=1e-6)


def test_empty_lidar():
    p3d, valid = assign_depth(np.array([[50.0, 50.0]]), np.zeros((0, 3)), K, (100, 100))
    assert not valid.any()
