import numpy as np

from src.rectify import load_kalibr, Rectifier

CALIB = 'data/rosbag1/kalibr_imucam_chain.yaml'


def test_load_kalibr_cam0():
    c = load_kalibr(CALIB, 'cam0')
    assert c['size'] == (1472, 1440)
    assert abs(c['K'][0, 0] - 465.3015) < 0.01     # fx
    assert len(c['D']) == 4                         # equidistant k1..k4
    assert c['T_cam_imu'].shape == (4, 4)
    assert abs(c['timeshift'] + 0.00657) < 1e-4


def test_rectifier_output_shape_and_K():
    c = load_kalibr(CALIB, 'cam0')
    r = Rectifier(c['K'], c['D'], c['size'], out_size=(400, 400))
    img = np.zeros((1440, 1472, 3), np.uint8)
    out = r.rectify(img)
    assert out.shape == (400, 400, 3)
    assert np.allclose(r.K_new[:2, 2], [200, 200])  # 주점 = 출력 중심
