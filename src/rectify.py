# 200° 피시아이(equidistant) → 중앙 영역 핀홀로 펴는 전처리.
# 특징 추출/매칭/PnP는 전부 "펴진 이미지 + 새 K" 기준으로 동작한다 (DB·쿼리 동일 적용).
import cv2
import numpy as np
import yaml


def load_kalibr(path: str, cam: str = 'cam0') -> dict:
    """Kalibr imucam chain yaml에서 카메라 캘리브 로드.

    반환: {'K':3x3, 'D':(4,), 'size':(w,h), 'T_cam_imu':4x4, 'timeshift': float}
    """
    txt = open(path).read().replace('%YAML:1.0', '')  # opencv 헤더 제거
    y = yaml.safe_load(txt)[cam]
    fx, fy, cx, cy = y['intrinsics']
    assert y['distortion_model'] == 'equidistant', y['distortion_model']
    return {
        'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        'D': np.array(y['distortion_coeffs'], float),
        'size': tuple(y['resolution']),          # (w, h)
        'T_cam_imu': np.array(y['T_cam_imu'], float),
        'timeshift': float(y.get('timeshift_cam_imu', 0.0)),
    }


class Rectifier:
    """equidistant 피시아이 이미지를 핀홀로 펴는 리매퍼 (맵은 1회 계산 후 재사용).

    out_size: 펴진 출력 크기. fov_scale: 작을수록 중앙을 좁게(왜곡 적게) 편다.
    200° 렌즈는 전체를 펼 수 없으므로 한 뷰당 ~90°만 사용하고,
    yaw_deg로 좌/우를 향한 가상 뷰를 추가로 펼 수 있다 (다중 뷰 rectify).
    """

    def __init__(self, K, D, in_size, out_size=(800, 800), fov_scale: float = 0.5,
                 yaw_deg: float = 0.0):
        w, h = out_size
        # 새 핀홀 K: 출력 중심 주점 + fov_scale로 초점거리 조절
        f_new = K[0, 0] / fov_scale * (w / in_size[0])
        self.K_new = np.array([[f_new, 0, w / 2], [0, f_new, h / 2], [0, 0, 1]])
        # 뷰 회전: 카메라 yaw θ 방향이 뷰 정면([0,0,1])이 되도록 R = R_y(-θ)
        th = np.radians(yaw_deg)
        R = np.array([[np.cos(th), 0, -np.sin(th)],
                      [0, 1, 0],
                      [np.sin(th), 0, np.cos(th)]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, self.K_new, (w, h), cv2.CV_16SC2)
        # 뷰 → 카메라 회전 (가상 카메라 포즈: T_w_view = T_w_cam @ T_cam_view)
        self.T_cam_view = np.eye(4)
        self.T_cam_view[:3, :3] = R.T

    def rectify(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
