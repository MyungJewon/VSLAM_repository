# 사용자용 리로컬라이징 API — 모델·DB를 1회 로드하고 재사용한다.
#
#   from src.reloc import Relocalizer
#   r = Relocalizer('config.yaml')          # 로드 ~10초 (1회)
#   res = r.localize('photo.jpg')           # 경로 또는 BGR numpy 이미지
#   # 성공: {'ok': True, 'xyz': [...], 'quat': [qx,qy,qz,qw],
#   #        'T_wc': 4x4 list, 'inliers': N, 'view_yaw': 0}
#   # 실패: {'ok': False, 'reason': ...}   ← 틀린 포즈를 반환하지 않는다
#
# CLI: python -m src.reloc <이미지> [config.yaml]
import sys

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, '.')
from src.features import XFeat
from src.localize import Localizer
from src.rectify import Rectifier, load_kalibr
from src.retrieval import Retrieval


class Relocalizer:
    def __init__(self, cfg_path: str = 'config.yaml'):
        cfg = yaml.safe_load(open(cfg_path))
        calib = load_kalibr(cfg['calib']['kalibr_yaml'], cfg['calib']['cam'])
        yaws = cfg['calib'].get('query_view_yaws',
                                cfg['calib'].get('view_yaws', [0]))
        self.rects = [(y, Rectifier(calib['K'], calib['D'], calib['size'],
                                    out_size=tuple(cfg['calib']['rect_size']),
                                    fov_scale=cfg['calib']['fov_scale'],
                                    yaw_deg=y))
                      for y in yaws]
        self.loc = Localizer(cfg['db_dir'], XFeat(),
                             Retrieval(cfg['retrieval_model']),
                             k=cfg['top_k'],
                             min_inliers=cfg.get('min_inliers', 15))

    def localize(self, img, K=None, dist=None) -> dict:
        """이미지 → 포즈 dict.

        두 입력 모드:
        · K=None (기본): 데이터셋의 원본 어안 이미지 — config 캘리브로
          다중 뷰 rectify 후 측위 (평가/데모용)
        · K 제공: 임의의 핀홀 카메라 이미지 — 클라이언트가 자기 캘리브
          (3x3 K, 왜곡계수 dist 선택)를 보내는 실전 API 모드.
          왜곡이 있으면 먼저 펴서(undistort) 순수 핀홀로 만든 뒤 측위한다.
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        if img is None:
            return {'ok': False, 'reason': 'invalid_image'}
        if K is not None:
            K = np.asarray(K, float)
            if dist is not None and np.any(np.asarray(dist)):
                img = cv2.undistort(img, K, np.asarray(dist, float))
            r = self.loc.localize(img, K=K)
            if r is None:
                return {'ok': False, 'reason': 'not_enough_inliers'}
            return self._pack(r['T_wc'], r['inliers'], view_yaw=None)
        best, best_yaw = None, 0
        for yaw, rect in self.rects:     # 다중 뷰 시도, 인라이어 최다 채택
            r = self.loc.localize(rect.rectify(img))
            if r and (best is None or r['inliers'] > best['inliers']):
                # 뷰 포즈 → 실제 카메라 포즈로 환원
                r['T_wc'] = r['T_wc'] @ np.linalg.inv(rect.T_cam_view)
                best, best_yaw = r, yaw
        if best is None:
            return {'ok': False, 'reason': 'not_enough_inliers'}
        return self._pack(best['T_wc'], best['inliers'], best_yaw)

    @staticmethod
    def _pack(T, inliers, view_yaw):
        return {'ok': True,
                'xyz': T[:3, 3].tolist(),
                'quat': Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
                'T_wc': T.tolist(),
                'inliers': int(inliers),
                'view_yaw': view_yaw}


if __name__ == '__main__':
    import json
    import time
    a = sys.argv[1:]
    if not a:
        sys.exit('사용: python -m src.reloc <이미지> [config.yaml]')
    r = Relocalizer(a[1] if len(a) > 1 else 'config.yaml')
    t0 = time.time()
    res = r.localize(a[0])
    res['time_sec'] = round(time.time() - t0, 2)
    print(json.dumps(res, indent=2))
