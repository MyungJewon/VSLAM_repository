# 리로컬라이징 — 쿼리 이미지 1장 → 6DOF 포즈 (CPU only).
#   1. MegaLoc 임베딩 → DB top-k 후보
#   2. 후보별 XFeat 매칭 (쿼리 2D ↔ DB의 3D 태깅된 특징점)
#   3. 누적된 2D-3D 대응으로 PnP+RANSAC → T_world_cam
from pathlib import Path

import numpy as np

from src.pnp import solve_pnp
from src.retrieval import top_k as topk_fn


class Localizer:
    def __init__(self, db_dir: str, xfeat, retrieval, k: int = 5):
        self.db_dir = Path(db_dir)
        idx = np.load(self.db_dir / 'index.npz')
        self.vecs, self.K = idx['vecs'], idx['K']
        self.xf, self.ret, self.k = xfeat, retrieval, k

    def localize(self, img_rect: np.ndarray):
        """rectify된 쿼리 이미지 → {'T_wc', 'inliers', 'candidates'} 또는 None"""
        q_feats = self.xf.extract(img_rect)
        cands = topk_fn(self.ret.embed(img_rect), self.vecs, self.k)
        # 후보별로 PnP를 따로 풀고 인라이어 최다 결과 채택 —
        # 비슷하게 생긴 다른 장소(alias) 후보가 섞여도 대응을 오염시키지 못한다.
        best = None
        for c in cands:
            f = np.load(self.db_dir / f'frame_{c:05d}.npz')
            if len(f['keypoints']) == 0:
                continue
            iq, idb = self.xf.match(q_feats, {'descriptors': f['descriptors']})
            r = solve_pnp(q_feats['keypoints'][iq], f['pts3d_w'][idb],
                          self.K, None)
            if r and (best is None or r['inliers'] > best['inliers']):
                best = {**r, 'cand': int(c)}
        if best is None:
            return None
        return {'T_wc': np.linalg.inv(best['T_cam_world']),
                'inliers': best['inliers'], 'candidates': cands,
                'best_cand': best['cand']}
