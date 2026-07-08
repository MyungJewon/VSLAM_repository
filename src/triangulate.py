# GT 포즈 기반 두-뷰 삼각측량 — LiDAR 없는 데이터에서 특징점에 3D를 부여한다 (플랜 C).
# 포즈를 아는(GT) 두 시점에서 같은 점을 보면 광선 교차로 3D가 직접 나온다.
# SFM처럼 포즈 추정이 필요 없고, GT가 LiDAR 기반이라 스케일이 metric이다.
import cv2
import numpy as np


def triangulate_pairs(kpts_a, kpts_b, K, T_wc_a, T_wc_b,
                      min_baseline: float = 0.05,
                      max_reproj_px: float = 2.0,
                      max_depth_ratio: float = 50.0):
    """매칭된 픽셀쌍을 월드 3D로 삼각측량.

    kpts_a/b: (N,2) 매칭된 픽셀 (왜곡 보정된 핀홀 좌표 기준)
    K: 3x3 intrinsic, T_wc_*: 각 시점의 T_world_cam(4x4)
    반환: (pts3d_w (N,3), valid (N,) bool)

    기각 게이트 — 억지 3D가 PnP를 오염시키는 것 방지:
      · 두 카메라 간 거리 < min_baseline (시차 부족 → 깊이 불확실)이면 전체 무효
      · 어느 한쪽 카메라 뒤(z<=0)에 놓이는 점
      · 재투영 오차 > max_reproj_px (잘못된 매칭)
      · 깊이 > max_depth_ratio × baseline — 깊이 오차는 depth²/baseline에
        비례하므로 절대 깊이가 아닌 베이스라인 대비 상대 게이트를 쓴다.
        도보(0.3m)면 ~15m, 차량(3m)이면 ~150m 허용 → 실내외 겸용.
    """
    kpts_a = np.asarray(kpts_a, float)
    kpts_b = np.asarray(kpts_b, float)
    n = len(kpts_a)
    pts3d = np.zeros((n, 3))
    valid = np.zeros(n, bool)
    if n == 0:
        return pts3d, valid

    baseline = np.linalg.norm(T_wc_a[:3, 3] - T_wc_b[:3, 3])
    if baseline < min_baseline:
        return pts3d, valid

    # 투영행렬 P = K [R|t] (world→cam 방향이므로 T_cam_world 사용)
    T_cw_a = np.linalg.inv(T_wc_a)
    T_cw_b = np.linalg.inv(T_wc_b)
    P_a = K @ T_cw_a[:3, :]
    P_b = K @ T_cw_b[:3, :]

    X = cv2.triangulatePoints(P_a, P_b, kpts_a.T, kpts_b.T)   # (4,N) 동차좌표
    w = X[3]
    ok = np.abs(w) > 1e-9
    Xw = np.zeros((n, 3))
    Xw[ok] = (X[:3, ok] / w[ok]).T

    def cam_z(T_cw, pts):
        return (T_cw[:3, :3] @ pts.T).T[:, 2] + T_cw[2, 3]

    max_depth = max_depth_ratio * baseline
    za, zb = cam_z(T_cw_a, Xw), cam_z(T_cw_b, Xw)
    ok &= (za > 0.1) & (zb > 0.1) & (za < max_depth) & (zb < max_depth)

    # 재투영 오차 게이트
    def reproj(P, pts):
        uvw = (P @ np.hstack([pts, np.ones((n, 1))]).T).T
        return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9)

    err_a = np.linalg.norm(reproj(P_a, Xw) - kpts_a, axis=1)
    err_b = np.linalg.norm(reproj(P_b, Xw) - kpts_b, axis=1)
    ok &= (err_a < max_reproj_px) & (err_b < max_reproj_px)

    pts3d[ok] = Xw[ok]
    valid = ok
    return pts3d, valid


def merge_multiview(estimates, cam_center,
                    abs_tol: float = 0.05, rel_tol: float = 0.03):
    """같은 특징점을 여러 쌍에서 삼각측량한 결과를 일관성 검사 후 병합.

    estimates: {kpt_idx: [3D점, ...]} — 쌍마다 얻은 추정 목록
    cam_center: 기준 프레임 카메라 위치 (허용오차를 깊이에 비례시키는 데 사용)
    반환: (kpt_indices (M,), pts3d (M,3))

    추정이 2개 이상이면 서로 abs_tol + rel_tol×깊이 이내일 때만 채택(평균).
    불일치 = 잘못된 매칭이나 깊이 오차 → 버림. 추정 1개는 그대로 사용.
    """
    idxs, pts = [], []
    for k, ests in estimates.items():
        if not ests:
            continue
        if len(ests) == 1:
            idxs.append(k)
            pts.append(ests[0])
            continue
        E = np.asarray(ests)
        depth = np.linalg.norm(E.mean(0) - cam_center)
        tol = abs_tol + rel_tol * depth
        spread = np.linalg.norm(E - E.mean(0), axis=1).max()
        if spread < tol:
            idxs.append(k)
            pts.append(E.mean(0))
    return np.array(idxs, int), (np.array(pts) if pts else np.zeros((0, 3)))
