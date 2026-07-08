import numpy as np

from src.pose_provider import GTPoseProvider


def test_interpolation(tmp_path):
    # t=0에 원점, t=2에 x=2 (무회전) → t=1이면 x=1
    gt = tmp_path / 'gt.txt'
    gt.write_text('0 0 0 0 0 0 0 1\n2 2 0 0 0 0 0 1\n')
    p = GTPoseProvider(str(gt), max_gap=5.0)
    T = p.pose_at(1.0)
    assert np.allclose(T[:3, 3], [1, 0, 0], atol=1e-6)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)


def test_rotation_slerp(tmp_path):
    # z축 0° → 90°(qz=sin45, qw=cos45) 회전: t=1(중간)이면 45°
    s = np.sin(np.pi / 4)
    gt = tmp_path / 'gt.txt'
    gt.write_text(f'0 0 0 0 0 0 0 1\n2 0 0 0 0 0 {s} {s}\n')
    p = GTPoseProvider(str(gt), max_gap=5.0)
    T = p.pose_at(1.0)
    c45 = np.cos(np.pi / 4)
    expected = np.array([[c45, -c45, 0], [c45, c45, 0], [0, 0, 1]])
    assert np.allclose(T[:3, :3], expected, atol=1e-6)


def test_out_of_range_returns_none(tmp_path):
    gt = tmp_path / 'gt.txt'
    gt.write_text('0 0 0 0 0 0 0 1\n2 2 0 0 0 0 0 1\n')
    p = GTPoseProvider(str(gt))
    assert p.pose_at(5.0) is None
    assert p.pose_at(-1.0) is None


def test_gap_returns_none(tmp_path):
    # 0초와 10초 사이 공백 > max_gap → 보간 거부
    gt = tmp_path / 'gt.txt'
    gt.write_text('0 0 0 0 0 0 0 1\n10 2 0 0 0 0 0 1\n')
    p = GTPoseProvider(str(gt), max_gap=0.5)
    assert p.pose_at(5.0) is None
