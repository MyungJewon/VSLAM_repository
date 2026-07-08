# 2D-3D 대응 → 6DOF (PnP+RANSAC). 실패는 명시적 None — 틀린 pose를 자신 있게 반환 금지.
import cv2
import numpy as np

MIN_POINTS = 6
MIN_INLIERS = 10


def solve_pnp(pts2d, pts3d, K, dist, min_inliers: int = MIN_INLIERS):
    """반환: {'T_cam_world': 4x4, 'inliers': int, 'inlier_idx': (M,)} 또는 None.

    T_cam_world = "월드 점 → 카메라 좌표" 변환. 카메라의 월드 pose는 역행렬.
    """
    if len(pts2d) < max(MIN_POINTS, 4):
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        np.asarray(pts3d, np.float64), np.asarray(pts2d, np.float64),
        np.asarray(K, np.float64), dist,
        iterationsCount=1000, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    # 인라이어만으로 반복 정밀화 (RANSAC 결과는 최소셋 기반이라 미세 오차 잔존)
    idx = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineVVS(
        np.asarray(pts3d, np.float64)[idx], np.asarray(pts2d, np.float64)[idx],
        np.asarray(K, np.float64), dist, rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return {'T_cam_world': T, 'inliers': int(len(inl)),
            'inlier_idx': inl.ravel()}
