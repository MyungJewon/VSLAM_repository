# 리로컬라이징 — 쿼리 이미지 1장 → 6DOF 포즈 (CPU only).
#   1. MegaLoc 임베딩 → DB top-k 후보
#   2. 후보별 XFeat 매칭 + 개별 PnP → 인라이어 최다 후보 채택
#      (alias 후보가 대응을 오염시키지 못하게 후보를 섞지 않는다)
#   3. 가이드 재매칭: 1차 포즈로 top-k 전체의 3D점을 쿼리에 투영,
#      "보여야 할 위치" 근처의 특징점과 재대응 → 2차 PnP로 정밀화.
#      경로에서 벗어난 쿼리일수록 1차 매칭이 빈약해서 이 단계의 이득이 크다.
from pathlib import Path

import numpy as np

from src.pnp import solve_pnp
from src.retrieval import top_k as topk_fn


class Localizer:
    def __init__(self, db_dir: str, xfeat, retrieval, k: int = 5,
                 min_inliers: int = 15, guided_radius_px: float = 30.0,
                 guided_min_cossim: float = 0.7):
        self.db_dir = Path(db_dir)
        idx = np.load(self.db_dir / 'index.npz')
        self.vecs, self.K = idx['vecs'], idx['K']
        self.xf, self.ret, self.k = xfeat, retrieval, k
        self.min_inliers = min_inliers
        self.g_radius = guided_radius_px
        self.g_cossim = guided_min_cossim

    def _guided_refine(self, q_feats, frames, T_cw, img_shape, K):
        """1차 포즈로 DB 3D점을 투영 → 근처 쿼리 특징점과 재대응 → 2차 PnP."""
        p3d = np.vstack([f['pts3d_w'] for f in frames])
        desc = np.vstack([f['descriptors'] for f in frames])
        # 투영 (카메라 앞 + 화면 안만)
        pc = (T_cw[:3, :3] @ p3d.T).T + T_cw[:3, 3]
        front = pc[:, 2] > 0.1
        uv = (K @ pc[front].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        h, w = img_shape[:2]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        p3d, desc, uv = p3d[front][inside], desc[front][inside], uv[inside]
        if len(p3d) < self.min_inliers:
            return None
        # 각 쿼리 특징점에서 반경 내 투영점 중 디스크립터 최유사 선택
        qk, qd = q_feats['keypoints'], q_feats['descriptors']
        pts2d, pts3d = [], []
        for i in range(len(qk)):
            d2 = np.sum((uv - qk[i]) ** 2, axis=1)
            near = np.where(d2 < self.g_radius ** 2)[0]
            if len(near) == 0:
                continue
            sims = desc[near] @ qd[i]
            j = near[np.argmax(sims)]
            if sims.max() > self.g_cossim:
                pts2d.append(qk[i])
                pts3d.append(p3d[j])
        if len(pts2d) < self.min_inliers:
            return None
        return solve_pnp(np.array(pts2d), np.array(pts3d), K, None,
                         min_inliers=self.min_inliers)

    def localize(self, img_rect: np.ndarray, K=None):
        """왜곡 없는(핀홀) 쿼리 이미지 → {'T_wc', 'inliers', ...} 또는 None.

        K: 쿼리 카메라 intrinsic. 생략 시 DB 기본(데이터셋 rectify K) 사용.
        """
        K = self.K if K is None else np.asarray(K, float)
        q_feats = self.xf.extract(img_rect)
        cands = topk_fn(self.ret.embed(img_rect), self.vecs, self.k)
        frames, best = [], None
        for c in cands:
            f = dict(np.load(self.db_dir / f'frame_{c:05d}.npz'))
            if len(f['keypoints']) == 0:
                continue
            frames.append(f)
            iq, idb = self.xf.match(q_feats, {'descriptors': f['descriptors']})
            r = solve_pnp(q_feats['keypoints'][iq], f['pts3d_w'][idb],
                          K, None, min_inliers=self.min_inliers)
            if r and (best is None or r['inliers'] > best['inliers']):
                best = {**r, 'cand': int(c)}
        if best is None:
            return None
        # 가이드 재매칭 — 인라이어가 늘어난 경우에만 교체
        refined = self._guided_refine(q_feats, frames,
                                      best['T_cam_world'], img_rect.shape, K)
        if refined and refined['inliers'] > best['inliers']:
            best = {**refined, 'cand': best['cand']}
        return {'T_wc': np.linalg.inv(best['T_cam_world']),
                'inliers': best['inliers'], 'candidates': cands,
                'best_cand': best['cand']}
