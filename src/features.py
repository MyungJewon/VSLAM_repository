# XFeat 래퍼 — CPU 친화적 로컬 특징 추출 + 자체 LighterGlue 매칭.
# (공식 cvg/LightGlue는 XFeat 디스크립터를 지원하지 않아 XFeat 내장 매처를 쓴다)
import numpy as np
import torch


class XFeat:
    def __init__(self, top_k: int = 2048):
        self.model = torch.hub.load('verlab/accelerated_features', 'XFeat',
                                    pretrained=True, top_k=top_k, trust_repo=True)
        self.model.eval()

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        return torch.from_numpy(img.transpose(2, 0, 1)[None]).float()

    @torch.no_grad()
    def extract(self, img: np.ndarray) -> dict:
        """이미지(BGR/gray uint8) → {'keypoints':(N,2), 'descriptors':(N,64)} (numpy)"""
        out = self.model.detectAndCompute(self._to_tensor(img), top_k=None)[0]
        return {'keypoints': out['keypoints'].cpu().numpy(),
                'descriptors': out['descriptors'].cpu().numpy(),
                'scores': out['scores'].cpu().numpy()}

    @torch.no_grad()
    def match(self, feats_a: dict, feats_b: dict, min_cossim: float = 0.82):
        """두 extract 결과 매칭 → (idx_a, idx_b) 인덱스 배열."""
        da = torch.from_numpy(feats_a['descriptors'])
        db = torch.from_numpy(feats_b['descriptors'])
        idx0, idx1 = self.model.match(da, db, min_cossim=min_cossim)
        return idx0.cpu().numpy(), idx1.cpu().numpy()
